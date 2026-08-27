"""
End-to-end fault-injection tests for file-upload durability.

Reproduces a silent file-artifact loss seen in production: a transient
network disturbance during the startup window — connection resets on the
first few presign POSTs — permanently lost several file payloads while
everything logged afterwards uploaded fine.

These tests spawn the real `python -m pluto.sync` subprocess against a
local fake ingest+S3 server whose fault mode abruptly resets the first N
connections (RST via SO_LINGER), then verify:

1. transient faults: payloads eventually land once the fault clears —
   there is no retry budget to exhaust.
2. persistent faults: the process never claims "All records synced
   successfully"; the rows stay in SQLite and the loss is reported.
"""

from __future__ import annotations

import http.server
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

from pluto.sync.store import SyncStatus, SyncStore


class _FaultyIngestHandler(http.server.BaseHTTPRequestHandler):
    """Fake ingest + S3 endpoint with an injectable connection-reset fault.

    Server attributes (set by the fixture):
    - faults_remaining: list[int] of one element — while > 0 (or < 0 for
      "always"), every request is answered with a TCP RST.
    - uploads: dict of s3 filename -> bytes received via PUT.
    - presign_requests: count of POST /files requests that got through.
    """

    protocol_version = 'HTTP/1.1'

    def _reset_connection(self) -> None:
        # Drain the request body, then tear the connection down without a
        # response (SO_LINGER 0 → RST on close). Surfaces client-side as an
        # immediate httpx.RemoteProtocolError — the same "server disconnected
        # without sending a response" wire event suspected in the incident.
        length = int(self.headers.get('Content-Length', '0') or 0)
        if length:
            self.rfile.read(length)
        self.connection.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0)
        )
        self.wfile.flush()
        self.connection.shutdown(socket.SHUT_RDWR)
        self.connection.close()

    def _maybe_fault(self) -> bool:
        with self.server.lock:
            n = self.server.faults_remaining[0]
            if n != 0:
                if n > 0:
                    self.server.faults_remaining[0] = n - 1
                return True
        return False

    def do_POST(self):  # noqa: N802 — http.server convention
        if self._maybe_fault():
            self._reset_connection()
            return
        length = int(self.headers.get('Content-Length', '0') or 0)
        body = self.rfile.read(length) if length else b''
        if self.path == '/files':
            with self.server.lock:
                self.server.presign_requests += 1
            payload = json.loads(body)
            urls = {}
            for entry in payload.get('files', []):
                name = entry['fileName']
                urls[name] = f'http://127.0.0.1:{self.server.server_port}/s3/{name}'
            response = json.dumps({'File': [urls]}).encode()
        else:
            response = b'{"ok": true}'
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_PUT(self):  # noqa: N802
        if self._maybe_fault():
            self._reset_connection()
            return
        length = int(self.headers.get('Content-Length', '0') or 0)
        body = self.rfile.read(length) if length else b''
        name = self.path.rsplit('/', 1)[-1]
        with self.server.lock:
            self.server.uploads[name] = body
        self.send_response(200)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def log_message(self, *_args, **_kwargs):
        pass


def _start_faulty_server(initial_faults: int):
    """Start the fault-injection server. initial_faults < 0 → fault forever."""
    # Bind port 0 directly — probing for a free port and rebinding races
    # against parallel xdist workers claiming the same port in between.
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _FaultyIngestHandler)
    port = server.server_port
    server.lock = threading.Lock()
    server.faults_remaining = [initial_faults]
    server.uploads = {}
    server.presign_requests = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, thread


def _seed_db_with_files(db_path: str, tmp_path, n: int = 3) -> list:
    """Create a SyncStore with n pending file uploads backed by real files."""
    store = SyncStore(db_path)
    store.register_run('run-x', 'project-x', op_id=42)
    names = []
    for i in range(n):
        staged = tmp_path / f'config{i}.yaml'
        staged.write_text(f'param: {i}\n')
        store.enqueue_file(
            run_id='run-x',
            local_path=str(staged),
            file_name=f'config{i}',
            file_ext='.yaml',
            file_type='text/yaml',
            file_size=staged.stat().st_size,
            log_name=f'hydra/config{i}.yaml',
            timestamp_ms=int(time.time() * 1000),
            step=0,
        )
        names.append(f'config{i}.yaml')
    store.close()
    return names


def _settings(base_url: str) -> dict:
    return {
        '_auth': 'tok',
        '_op_id': 42,
        '_op_name': 'run-x',
        'project': 'project-x',
        'tag': 'pluto',
        'url_num': f'{base_url}/ingest/metrics',
        'url_data': f'{base_url}/ingest/data',
        'url_file': f'{base_url}/files',
        'url_message': f'{base_url}/ingest/logs',
        'sync_process_shutdown_timeout': 5.0,
        'sync_process_flush_interval': 0.1,
        'sync_process_orphan_timeout': 60.0,
        'sync_process_retry_max': 5,
        # Flat, fast retry spacing so faults clear within test time.
        'sync_process_retry_backoff': 1.0,
        'sync_process_batch_size': 50,
        'sync_process_file_batch_size': 10,
    }


def _spawn_sync_subprocess(db_path: str, settings_dict: dict) -> subprocess.Popen:
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


def _file_statuses(db_path: str) -> list:
    store = SyncStore(db_path)
    cursor = store.conn.execute('SELECT status, retry_count FROM file_uploads')
    rows = cursor.fetchall()
    store.close()
    return [(SyncStatus(r[0]), r[1]) for r in rows]


def test_files_recover_after_transient_resets(tmp_path):
    """Payloads must eventually upload once a transient fault clears.

    The first 6 requests are answered with TCP RST — more than the old
    retry_count < 5 budget, exactly the startup-window disturbance from
    the incident. Pre-fix, all files are silently abandoned and this
    times out; post-fix, retries continue until the fault clears and
    every payload lands.
    """
    server, port, thread = _start_faulty_server(initial_faults=6)
    db_path = str(tmp_path / 'sync.db')
    names = _seed_db_with_files(db_path, tmp_path, n=3)

    proc = _spawn_sync_subprocess(db_path, _settings(f'http://127.0.0.1:{port}'))
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            with server.lock:
                done = set(names) <= set(server.uploads)
            if done:
                break
            assert proc.poll() is None, (
                'sync subprocess died: '
                f'{proc.stderr.read().decode(errors="replace") if proc.stderr else "?"}'
            )
            time.sleep(0.2)

        with server.lock:
            uploaded = dict(server.uploads)
        assert set(names) <= set(uploaded), (
            f'files never uploaded after transient fault cleared; '
            f'got {sorted(uploaded)}. This is the silent-loss bug: '
            f'the retry budget was exhausted during the fault window.'
        )
        for i, name in enumerate(names):
            assert uploaded[name] == f'param: {i}\n'.encode()

        # DB should agree: every row COMPLETED, none stuck FAILED/ABANDONED.
        statuses = _file_statuses(db_path)
        assert all(s == SyncStatus.COMPLETED for s, _ in statuses), statuses
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_persistent_fault_never_reports_synced(tmp_path):
    """With the fault never clearing, the process must not claim success.

    Pre-fix, once every file burned its 5 retries the shutdown drain
    selected nothing (max_retries=1 filter), counted nothing pending
    (FAILED excluded), and logged 'All records synced successfully' at
    INFO — the exact silent-loss signature from the incident.
    """
    server, port, thread = _start_faulty_server(initial_faults=-1)
    db_path = str(tmp_path / 'sync.db')
    _seed_db_with_files(db_path, tmp_path, n=2)

    proc = _spawn_sync_subprocess(db_path, _settings(f'http://127.0.0.1:{port}'))
    try:
        # Let it burn through several attempts, then shut down.
        time.sleep(4.0)
        assert proc.poll() is None
        proc.send_signal(signal.SIGTERM)
        try:
            _stdout, stderr = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            _stdout, stderr = proc.communicate()

        stderr_text = stderr.decode(errors='replace')
        assert 'All records synced successfully' not in stderr_text, (
            'sync process claimed success while file payloads were never '
            f'uploaded. stderr:\n{stderr_text[:3000]}'
        )
        # The unsent files must be reported at WARNING or above.
        assert (
            'WARNING' in stderr_text or 'ERROR' in stderr_text
        ), f'no warning/error about unsent files. stderr:\n{stderr_text[:3000]}'

        # Rows must remain in SQLite, not COMPLETED, for later recovery.
        statuses = _file_statuses(db_path)
        assert len(statuses) == 2
        assert all(s != SyncStatus.COMPLETED for s, _ in statuses), statuses
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
