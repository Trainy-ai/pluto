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


class TestLogTimestampLegacyPath:
    def test_timestamp_forwarded_to_legacy_log(self, tmp_path):
        op = _make_op(tmp_path)
        op._sync_manager = None  # force legacy offline path
        with mock.patch.object(op, '_log') as legacy_log:
            op.log({'loss': 0.5}, step=2, timestamp=TS)
        legacy_log.assert_called_once_with(data={'loss': 0.5}, step=2, t=TS)
