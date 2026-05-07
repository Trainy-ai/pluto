"""
Integration test for SIGTERM-shutdown drain in pluto.sync._sync_main.

Pre-fix _sync_main's SIGTERM handler set shutdown_requested=True, the
loop exited, and the `finally` closed the uploader and store WITHOUT
flushing any pending records. Anything still in SQLite at the moment
SIGTERM arrived was left behind and required a manual `pluto sync`
to recover. In DDP runs that crash, this is the common case: torchrun
sends SIGTERM to the surviving rank's process group, the sync
subprocess gets ~30s before SIGKILL — but pluto exited in ~100ms
without using any of it.

This test spawns the real `python -m pluto.sync` subprocess against a
local recording HTTP server, enqueues records into its DB, sends
SIGTERM, and verifies the records actually POST before the subprocess
exits. With the fix in place this passes; without it, the records
never reach the local server and the assertion fires.
"""

from __future__ import annotations

import http.server
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from pluto.sync.store import RecordType, SyncStore


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _RecordingHandler(http.server.BaseHTTPRequestHandler):
    """Captures every POST body in handler.server.posts (path -> [body, ...])."""

    def do_POST(self):  # noqa: N802 — http.server convention
        length = int(self.headers.get('Content-Length', '0') or 0)
        body = self.rfile.read(length) if length else b''
        self.server.posts.setdefault(self.path, []).append(body)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *_args, **_kwargs):
        # Silence the default per-request stderr spam.
        pass


@pytest.fixture
def recording_server():
    """Background HTTP server that records POST bodies per path."""
    port = _free_port()
    server = http.server.HTTPServer(('127.0.0.1', port), _RecordingHandler)
    server.posts = {}  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, port
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _seed_db_with_pending(db_path: str, n_console: int = 5) -> int:
    """Create a SyncStore at db_path with n pending CONSOLE records."""
    store = SyncStore(db_path)
    store.register_run('run-x', 'project-x', op_id=42)
    ts_ms = int(time.time() * 1000)
    for i in range(n_console):
        store.enqueue(
            'run-x',
            RecordType.CONSOLE,
            {'message': f'line-{i}', 'logType': 'INFO', 'lineNumber': i},
            ts_ms,
        )
    store.close()
    return n_console


def _spawn_sync_subprocess(db_path: str, settings_dict: dict) -> subprocess.Popen:
    """Spawn `python -m pluto.sync` with the given DB path and settings."""
    return subprocess.Popen(
        [
            sys.executable,
            '-m',
            'pluto.sync',
            '--db-path',
            db_path,
            '--settings',
            json.dumps(settings_dict),
            '--parent-pid',
            str(os.getpid()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll predicate up to timeout; return True if it ever became truthy."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_sigterm_drains_pending_records_before_exit(tmp_path, recording_server):
    """SIGTERM during a run with pending records must trigger a final drain.

    Spawns the real sync subprocess, enqueues 5 console records, sends
    SIGTERM, and waits for the process to exit. Asserts the recording
    server received POSTs to /ingest/logs covering all 5 records before
    the subprocess shut down. Pre-fix this fails because _sync_main's
    SIGTERM path skipped _flush_remaining entirely and the records
    stayed in SQLite.
    """
    server, port = recording_server
    base_url = f'http://127.0.0.1:{port}'

    db_path = str(tmp_path / 'sync.db')
    n = _seed_db_with_pending(db_path, n_console=5)

    settings = {
        '_auth': 'tok',
        '_op_id': 42,
        '_op_name': 'run-x',
        'project': 'project-x',
        'tag': 'pluto',
        'url_num': f'{base_url}/ingest/metrics',
        'url_data': f'{base_url}/ingest/data',
        'url_file': f'{base_url}/files',
        'url_message': f'{base_url}/ingest/logs',
        'url_update_config': f'{base_url}/api/runs/config/update',
        'url_update_tags': f'{base_url}/api/runs/tags/update',
        # Short shutdown timeout so the test finishes quickly even if the
        # drain has nothing to do; still long enough for a localhost POST.
        'sync_process_shutdown_timeout': 5.0,
        'sync_process_flush_interval': 60.0,  # Long → drain only happens on shutdown
        'sync_process_orphan_timeout': 60.0,
        'sync_process_retry_max': 1,
        'sync_process_batch_size': 50,
        'sync_process_file_batch_size': 10,
    }

    proc = _spawn_sync_subprocess(db_path, settings)
    try:
        # Wait for the subprocess to start its main loop. It logs
        # 'Sync process started' immediately after registering signal
        # handlers and opening the store, so a short delay is enough.
        # We don't want to read stderr in a blocking way (would race
        # with subprocess termination), so just give it a beat.
        time.sleep(1.0)
        assert proc.poll() is None, (
            'sync subprocess died before SIGTERM. stderr: '
            f'{proc.stderr.read().decode(errors="replace") if proc.stderr else "?"}'
        )

        proc.send_signal(signal.SIGTERM)

        # The fix gives the drain up to shutdown_timeout (5s here) plus
        # close overhead. Cap our wait at 15s so a regression doesn't
        # hang the test suite.
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            pytest.fail(
                f'Sync subprocess did not exit within 15s of SIGTERM. '
                f'stderr: {stderr.decode(errors="replace")[:2000]}'
            )

        # Sanity: the subprocess actually got the signal and ran the
        # shutdown path, not crashed mid-startup.
        stderr_text = stderr.decode(errors='replace')
        assert 'Sync process exiting' in stderr_text, (
            f'Sync subprocess exited but did not run the shutdown finally '
            f'block. stderr:\n{stderr_text[:3000]}'
        )

        log_posts = server.posts.get('/ingest/logs', [])
        assert log_posts, (
            'sync subprocess exited on SIGTERM without POSTing any '
            'pending console records. This is the pre-fix bug: the '
            'finally block tore down the uploader without calling '
            f'_flush_remaining. server.posts={dict(server.posts)} '
            f'stderr:\n{stderr_text[:3000]}'
        )

        # All 5 line-N messages should appear somewhere across the POST
        # bodies (NDJSON, batched in any order/grouping).
        all_log_bytes = b''.join(log_posts)
        for i in range(n):
            assert f'line-{i}'.encode() in all_log_bytes, (
                f'line-{i} missing from posted bodies. '
                f'Got {len(log_posts)} log POSTs totaling '
                f'{len(all_log_bytes)} bytes.'
            )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_clean_exit_with_no_pending_does_not_block(tmp_path, recording_server):
    """SIGTERM with empty queue should still exit promptly (no hang).

    Guards against the drain path waiting for batches that will never
    arrive. The drain helper has its own timeout but if a future
    refactor accidentally blocks on something else, this catches it.
    """
    server, port = recording_server
    base_url = f'http://127.0.0.1:{port}'

    db_path = str(tmp_path / 'sync.db')
    # Register a run but enqueue nothing.
    s = SyncStore(db_path)
    s.register_run('run-empty', 'project-x', op_id=43)
    s.close()

    settings = {
        '_auth': 'tok',
        '_op_id': 43,
        '_op_name': 'run-empty',
        'project': 'project-x',
        'tag': 'pluto',
        'url_num': f'{base_url}/ingest/metrics',
        'url_data': f'{base_url}/ingest/data',
        'url_file': f'{base_url}/files',
        'url_message': f'{base_url}/ingest/logs',
        'url_update_config': f'{base_url}/api/runs/config/update',
        'url_update_tags': f'{base_url}/api/runs/tags/update',
        'sync_process_shutdown_timeout': 5.0,
        'sync_process_flush_interval': 60.0,
        'sync_process_orphan_timeout': 60.0,
    }

    proc = _spawn_sync_subprocess(db_path, settings)
    try:
        time.sleep(1.0)
        assert proc.poll() is None
        start = time.time()
        proc.send_signal(signal.SIGTERM)
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            pytest.fail('Sync subprocess hung on SIGTERM with empty queue')
        elapsed = time.time() - start
        # Drain helper checks emptiness up front and returns fast.
        # 8s is a generous bound for a local subprocess teardown.
        assert elapsed < 8.0, f'Sync took {elapsed:.1f}s to exit on SIGTERM'
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
