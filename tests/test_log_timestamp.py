"""
Unit tests for historical-timestamp support on Op.log (backfill path).

Migration/backfill tooling (pluto.migrate) must be able to preserve the
original wall-clock time of each data point. These tests pin that an
explicit ``timestamp`` (epoch seconds) passed to ``Op.log`` reaches the
sync layer as ``timestamp_ms`` for metrics, structured data, and files —
and that a missing or invalid timestamp falls back to "now" without
crashing the caller.
"""

from __future__ import annotations

import math
import os
import time
from unittest import mock

import numpy as np
import pytest

import pluto
from pluto.op import Op
from pluto.sets import Settings

TS = 1600000000.5
TS_MS = 1600000000500


def _make_op(tmp_path) -> Op:
    settings = Settings()
    settings.mode = 'noop'
    settings.dir = str(tmp_path)
    settings.meta = []  # shadow the class-level shared list (test isolation)
    # Op.__init__ only prepares the staging dir outside noop mode.
    os.makedirs(os.path.join(settings.get_dir(), 'files'), exist_ok=True)
    op = Op(config={}, settings=settings)
    op._sync_manager = mock.MagicMock()
    return op


class TestLogTimestampMetrics:
    def test_explicit_timestamp_reaches_enqueue_metrics(self, tmp_path):
        op = _make_op(tmp_path)
        op.log({'loss': 0.5}, step=5, timestamp=TS)
        op._sync_manager.enqueue_metrics.assert_called_once_with(
            {'loss': 0.5}, TS_MS, 5
        )

    def test_default_timestamp_is_now(self, tmp_path):
        op = _make_op(tmp_path)
        before_ms = int(time.time() * 1000)
        op.log({'loss': 0.5}, step=1)
        after_ms = int(time.time() * 1000)
        (_, timestamp_ms, _), _ = (
            op._sync_manager.enqueue_metrics.call_args.args,
            op._sync_manager.enqueue_metrics.call_args.kwargs,
        )
        assert before_ms <= timestamp_ms <= after_ms

    @pytest.mark.parametrize('bad', [0, -5, float('nan'), float('inf')])
    def test_invalid_timestamp_falls_back_to_now(self, tmp_path, bad):
        op = _make_op(tmp_path)
        before_ms = int(time.time() * 1000)
        op.log({'loss': 0.5}, step=1, timestamp=bad)
        (_, timestamp_ms, _), _ = (
            op._sync_manager.enqueue_metrics.call_args.args,
            op._sync_manager.enqueue_metrics.call_args.kwargs,
        )
        assert timestamp_ms >= before_ms
        assert math.isfinite(timestamp_ms)


class TestLogTimestampDataAndFiles:
    def test_timestamp_reaches_enqueue_data_for_histogram(self, tmp_path):
        op = _make_op(tmp_path)
        op.log({'dist': pluto.Histogram(np.arange(10))}, step=3, timestamp=TS)
        kwargs = op._sync_manager.enqueue_data.call_args.kwargs
        assert kwargs['timestamp_ms'] == TS_MS
        assert kwargs['step'] == 3

    def test_timestamp_reaches_enqueue_file_for_image(self, tmp_path):
        op = _make_op(tmp_path)
        img = pluto.Image(np.zeros((4, 4, 3), dtype=np.uint8))
        op.log({'sample': img}, step=7, timestamp=TS)
        kwargs = op._sync_manager.enqueue_file.call_args.kwargs
        assert kwargs['timestamp_ms'] == TS_MS
        assert kwargs['step'] == 7


class TestLogMetricsBatch:
    def test_batch_enqueues_groups_in_one_call(self, tmp_path):
        op = _make_op(tmp_path)
        op._log_metrics_batch(
            [
                ({'loss': 1.0, 'acc': 0.1}, 0, TS),
                ({'loss': 0.5}, 1, TS + 1),
            ]
        )
        op._sync_manager.enqueue_metrics_batch.assert_called_once_with(
            [
                ({'loss': 1.0, 'acc': 0.1}, TS_MS, 0),
                ({'loss': 0.5}, TS_MS + 1000, 1),
            ]
        )

    def test_batch_registers_new_metric_names(self, tmp_path):
        op = _make_op(tmp_path)
        op._iface = mock.MagicMock()
        op._log_metrics_batch([({'loss': 1.0}, 0, TS)])
        op._iface._update_meta.assert_called_once_with(num=['loss'])

    def test_batch_noop_without_sync_manager(self, tmp_path):
        op = _make_op(tmp_path)
        op._sync_manager = None
        op._log_metrics_batch([({'loss': 1.0}, 0, TS)])  # must not raise


class TestLogTimestampLegacyPath:
    def test_timestamp_forwarded_to_legacy_log(self, tmp_path):
        op = _make_op(tmp_path)
        op._sync_manager = None  # force legacy offline path
        op.settings.mode = 'debug'  # not the perf queue
        with mock.patch.object(op, '_log') as legacy_log:
            op.log({'loss': 0.5}, step=2, timestamp=TS)
        legacy_log.assert_called_once_with(data={'loss': 0.5}, step=2, t=TS)

    def test_timestamp_forwarded_in_perf_mode_queue(self, tmp_path):
        op = _make_op(tmp_path)
        op._sync_manager = None
        op.settings.mode = 'perf'
        op.log({'loss': 0.5}, step=2, timestamp=TS)
        assert op._queue.get_nowait() == ({'loss': 0.5}, 2, TS)
