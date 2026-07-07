"""
Unit tests for settings.disable_system_metrics and Op._log_console.

Migration tooling (pluto.migrate) replays runs recorded elsewhere: the
migration host's own GPU/CPU stats must not leak into the imported run,
and historical console lines must be enqueueable with their original
timestamps. These tests pin both behaviors.
"""

from __future__ import annotations

import os
from unittest import mock

from pluto.op import Op
from pluto.sets import Settings

TS = 1600000000.5
TS_MS = 1600000000500


def _make_op(tmp_path) -> Op:
    settings = Settings()
    settings.mode = 'noop'
    settings.dir = str(tmp_path)
    os.makedirs(os.path.join(settings.get_dir(), 'files'), exist_ok=True)
    op = Op(config={}, settings=settings)
    op._sync_manager = mock.MagicMock()
    return op


def _run_monitor_once(op: Op) -> None:
    """Drive exactly one _worker_monitor iteration without sleeping."""
    op._monitor._stop_event.set()  # make the end-of-loop wait() return instantly
    seq = iter([False, True])
    op._monitor._worker_monitor(lambda: next(seq))


class TestDisableSystemMetrics:
    def test_monitor_sends_system_metrics_by_default(self, tmp_path):
        op = _make_op(tmp_path)
        op.settings._sys = mock.MagicMock()
        op.settings._sys.monitor.return_value = {'cpu': 1.0}
        _run_monitor_once(op)
        op._sync_manager.enqueue_system_metrics.assert_called_once()

    def test_monitor_skips_system_metrics_when_disabled(self, tmp_path):
        op = _make_op(tmp_path)
        op.settings.disable_system_metrics = True
        op.settings._sys = mock.MagicMock()
        _run_monitor_once(op)
        op._sync_manager.enqueue_system_metrics.assert_not_called()
        op.settings._sys.monitor.assert_not_called()

    def test_start_skips_sys_name_registration_when_disabled(self, tmp_path):
        op = _make_op(tmp_path)
        op.settings.disable_system_metrics = True
        op.settings._sys = mock.MagicMock()
        op._sync_manager = None
        op._monitor = mock.MagicMock()
        op._iface = mock.MagicMock()
        op.start()
        op._iface._update_meta.assert_not_called()
        op.settings._sys.monitor.assert_not_called()


class TestLogConsole:
    def test_log_console_converts_seconds_to_ms(self, tmp_path):
        op = _make_op(tmp_path)
        op._log_console([('hello', 'INFO', TS, 1), ('oops', 'ERROR', TS + 1, 2)])
        op._sync_manager.enqueue_console_batch.assert_called_once_with(
            [('hello', 'INFO', TS_MS, 1), ('oops', 'ERROR', TS_MS + 1000, 2)]
        )

    def test_log_console_is_noop_without_sync_manager(self, tmp_path):
        op = _make_op(tmp_path)
        op._sync_manager = None
        op._log_console([('hello', 'INFO', TS, 1)])  # must not raise
