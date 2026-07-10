"""
Unit tests for pluto._fd_capture.FdCapture — fd-level console capture.

Why fd-level capture exists: frameworks like torchtitan configure a
logging.StreamHandler at process start, BEFORE pluto.init() runs. CPython's
StreamHandler binds the sys.stderr *object* at construction, so pluto's
sys.stdout/sys.stderr swap (pluto.log.setup_logger_file) never sees those
writes — they go straight to fd 2 and the run's console section on the
server stays empty. Capturing at the file-descriptor level (dup2 tee, the
same approach as wandb's console="redirect") catches output regardless of
which Python object — or C extension — wrote it.

These tests intentionally exercise the REAL fds 1/2. pytest's default
capture is fd-level too and RE-dup2s the fds between the setup and call
phases (that's how it attributes output per phase) — which silently unwires
any FdCapture started inside a fixture. So captures here are always started
and stopped inside the test body, via context-manager helpers.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import time
from types import SimpleNamespace
from unittest import mock

from pluto._fd_capture import FdCapture


class FakeSyncManager:
    """Collects enqueue_console_batch calls like tests/test_log_console_handler.py."""

    def __init__(self):
        self.batches: list[list[tuple]] = []

    def enqueue_console_batch(self, lines):
        self.batches.append(list(lines))

    @property
    def lines(self) -> list[str]:
        return [line for batch in self.batches for (line, *_rest) in batch]


@contextlib.contextmanager
def capture_fd(fd: int, level: int, sanitizer=None):
    sm = FakeSyncManager()
    cap = FdCapture(fd=fd, level=level, sync_manager=sm, sanitizer=sanitizer)
    cap.start()
    try:
        yield cap, sm
    finally:
        cap.stop()  # idempotent — tests may have stopped already


def _fd2_bound_text_stream() -> io.TextIOWrapper:
    """A text stream permanently bound to fd 2 — what a logging.StreamHandler
    holds after binding sys.stderr at import time."""
    return io.TextIOWrapper(io.FileIO(2, 'w', closefd=False), write_through=True)


class TestTheTorchtitanBug:
    def test_logging_handler_bound_to_fd_before_capture_is_captured(self):
        """THE bug: a StreamHandler created before capture starts must still
        have its output reach the sync manager. This is torchtitan's
        init_logger() → wandb.init() ordering."""
        handler = logging.StreamHandler(_fd2_bound_text_stream())
        log = logging.getLogger('test_fdcap_titan_repro')
        log.propagate = False
        log.setLevel(logging.INFO)
        log.addHandler(handler)
        try:
            with capture_fd(2, logging.ERROR) as (cap, sm):
                log.info('step: 10  loss: 2.31')
                cap.stop()
        finally:
            log.removeHandler(handler)
        assert any('step: 10  loss: 2.31' in line for line in sm.lines)

    def test_raw_os_write_is_captured(self):
        with capture_fd(2, logging.ERROR) as (cap, sm):
            os.write(2, b'raw fd write\n')
            cap.stop()
        assert any('raw fd write' in line for line in sm.lines)


class TestTeeToOriginalDestination:
    def test_output_still_reaches_original_fd(self, capfd):
        """Capture must tee, not swallow — the terminal (here: pytest's
        capture file) keeps receiving everything."""
        with capture_fd(1, logging.INFO) as (cap, sm):
            os.write(1, b'tee me\n')
            cap.stop()
        assert 'tee me' in capfd.readouterr().out
        assert any('tee me' in line for line in sm.lines)


class TestStopSemantics:
    def test_stop_restores_fd_and_stops_enqueueing(self):
        with capture_fd(2, logging.ERROR) as (cap, sm):
            os.write(2, b'during\n')
            cap.stop()
            os.write(2, b'after\n')
            time.sleep(0.05)
        assert any('during' in line for line in sm.lines)
        assert not any('after' in line for line in sm.lines)

    def test_stop_flushes_pending_partial_line(self):
        with capture_fd(2, logging.ERROR) as (cap, sm):
            os.write(2, b'no-trailing-newline')
            cap.stop()
        assert any('no-trailing-newline' in line for line in sm.lines)

    def test_stop_is_idempotent(self):
        with capture_fd(2, logging.ERROR) as (cap, sm):
            os.write(2, b'once\n')
            cap.stop()
            cap.stop()
        assert sum('once' in line for line in sm.lines) == 1

    def test_start_after_stop_is_a_noop(self):
        """One capture instance = one lifecycle; Op re-init builds new ones."""
        with capture_fd(2, logging.ERROR) as (cap, sm):
            cap.stop()
            cap.start()
            os.write(2, b'zombie\n')
            cap.stop()
        assert not any('zombie' in line for line in sm.lines)


class TestLineHandling:
    def test_partial_writes_coalesce_into_one_line(self):
        with capture_fd(2, logging.ERROR) as (cap, sm):
            os.write(2, b'abc')
            os.write(2, b'def')
            os.write(2, b'ghi\n')
            cap.stop()
        assert any(line.endswith('abcdefghi') for line in sm.lines)
        assert not any(line.endswith('abc') for line in sm.lines)

    def test_empty_lines_are_skipped(self):
        with capture_fd(2, logging.ERROR) as (cap, sm):
            os.write(2, b'one\n\ntwo\n')
            cap.stop()
        assert any('one' in line for line in sm.lines)
        assert any('two' in line for line in sm.lines)
        assert '' not in sm.lines

    def test_non_utf8_bytes_do_not_crash(self):
        with capture_fd(2, logging.ERROR) as (cap, sm):
            os.write(2, b'bad \xff\xfe bytes\n')
            cap.stop()
        assert any('bad' in line and 'bytes' in line for line in sm.lines)

    def test_tuple_shape_matches_console_handler_contract(self):
        """(message, log_type, timestamp_ms, line_number) — same contract as
        pluto.log.ConsoleHandler / SyncProcessManager.enqueue_console_batch."""
        with capture_fd(2, logging.ERROR) as (cap, sm):
            os.write(2, b'first\nsecond\n')
            cap.stop()
        rows = [
            row
            for batch in sm.batches
            for row in batch
            if row[0].endswith(('first', 'second'))
        ]
        assert len(rows) == 2
        for message, log_type, timestamp_ms, line_number in rows:
            assert log_type == 'ERROR'  # fd 2 capture logs at ERROR
            assert isinstance(timestamp_ms, int) and timestamp_ms > 0
            assert isinstance(line_number, int)
        assert rows[1][3] == rows[0][3] + 1  # line numbers increase


class TestRankPrefix:
    def test_rank_env_var_prefixes_captured_lines(self, monkeypatch, capfd):
        monkeypatch.setenv('RANK', '3')
        with capture_fd(2, logging.ERROR) as (cap, sm):
            os.write(2, b'starting fit\n')
            cap.stop()
        assert '[rank3] starting fit' in sm.lines
        # tee stays unprefixed — torchrun adds its own [defaultN]: prefix
        assert '[rank3]' not in capfd.readouterr().err

    def test_no_rank_env_var_means_no_prefix(self, monkeypatch):
        monkeypatch.delenv('RANK', raising=False)
        with capture_fd(2, logging.ERROR) as (cap, sm):
            os.write(2, b'hello\n')
            cap.stop()
        assert 'hello' in sm.lines


class TestSanitizer:
    def test_sanitizer_runs_on_captured_lines(self):
        sanitizer = mock.MagicMock()
        sanitizer.sanitize.side_effect = lambda s: s.replace('mlpi_secret', '***')
        with capture_fd(2, logging.ERROR, sanitizer=sanitizer) as (cap, sm):
            os.write(2, b'api_key=mlpi_secret here\n')
            cap.stop()
        assert any('api_key=*** here' in line for line in sm.lines)
        assert not any('mlpi_secret' in line for line in sm.lines)


def _fake_settings(tmp_path, fd_capture=True):
    return SimpleNamespace(
        tag='pluto-fdcap-test',
        mode='perf',
        sanitize_logs=False,
        x_console_fd_capture=fd_capture,
        get_dir=lambda: str(tmp_path),
    )


@contextlib.contextmanager
def _logger_env(tmp_path, name, fd_capture=True):
    """setup_logger_file + guaranteed teardown, all inside the call phase."""
    from pluto.log import setup_logger_file, teardown_logger

    sm = FakeSyncManager()
    log = logging.getLogger(f'test_fdcap_{name}')
    console = logging.getLogger(f'test_fdcap_{name}_console')
    log.handlers.clear()
    console.handlers.clear()
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    try:
        setup_logger_file(
            _fake_settings(tmp_path, fd_capture), log, console, sync_manager=sm
        )
        yield sm, log, console
    finally:
        teardown_logger(log, console=console)
        # teardown_logger restores to import-time streams; put back what
        # pytest had so later tests see the exact same objects.
        sys.stdout, sys.stderr = saved_stdout, saved_stderr


class TestSetupLoggerFileIntegration:
    """setup_logger_file wires fd capture in and prevents double-enqueue."""

    def test_pre_bound_handler_output_is_enqueued(self, tmp_path):
        from pluto.log import _stop_fd_captures

        handler = logging.StreamHandler(_fd2_bound_text_stream())
        titan = logging.getLogger('test_fdcap_setup_titan')
        titan.propagate = False
        titan.setLevel(logging.INFO)
        titan.addHandler(handler)
        try:
            with _logger_env(tmp_path, 'pre_bound') as (sm, _log, _console):
                titan.info('loss: 1.23')
                _stop_fd_captures()
        finally:
            titan.removeHandler(handler)
        assert any('loss: 1.23' in line for line in sm.lines)

    def test_print_is_enqueued_exactly_once(self, tmp_path):
        """Exactly one layer owns each stream's enqueue duty. When sys.stdout
        is fd-backed the fd layer owns it (wrapper enqueue disabled); when it
        isn't — Jupyter's ZMQ stream, pytest's capture object (this test) —
        the wrapper keeps enqueueing since the fd pipe never sees the write.
        Either way a printed line must reach the sync manager exactly once."""
        from pluto.log import _stop_fd_captures

        with _logger_env(tmp_path, 'print_once') as (sm, _log, _console):
            print('exactly-once-sentinel')
            _stop_fd_captures()
        assert sum('exactly-once-sentinel' in line for line in sm.lines) == 1

    def test_console_logger_does_not_propagate(self, tmp_path):
        """Captured lines re-emitted through root handlers would loop back
        into the fd capture (root handler → old stderr → fd 2 → pipe)."""
        with _logger_env(tmp_path, 'propagate') as (_sm, _log, console):
            assert console.propagate is False

    def test_teardown_stops_captures(self, tmp_path):
        from pluto.log import _fd_captures

        with _logger_env(tmp_path, 'teardown') as (sm, _log, _console):
            assert len(_fd_captures) == 2
        # _logger_env's finally ran teardown_logger
        from pluto.log import _fd_captures as after

        assert after == []
        os.write(2, b'post-teardown\n')
        time.sleep(0.05)
        assert not any('post-teardown' in line for line in sm.lines)

    def test_flag_off_falls_back_to_wrapper_enqueue(self, tmp_path):
        """x_console_fd_capture=False keeps the legacy sys-swap capture."""
        from pluto.log import _fd_captures

        with _logger_env(tmp_path, 'flag_off', fd_capture=False) as (
            sm,
            _log,
            _console,
        ):
            assert _fd_captures == []
            print('legacy-path-sentinel')
            sys.stdout.flush()
        assert any('legacy-path-sentinel' in line for line in sm.lines)
