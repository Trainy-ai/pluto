"""Unit tests for the startup NCCL environment log line.

NCCL's configuration is the first thing anyone asks for when a distributed
job hangs, runs slow, or quietly falls back to TCP. Pluto ships it two ways
on startup: inside ``systemMetadata`` on run create (see
``tests/test_system_info.py``) and as a console line emitted by
``Op.start()``, which is captured and uploaded like any other log output.
These tests pin the log-line half.
"""

from __future__ import annotations

import atexit
import logging
import os
from unittest import mock

import pluto
from pluto.op import NCCL_ENV_LOG_MAX_CHARS, Op
from pluto.sets import Settings
from pluto.sys import MASKED_VALUE


def _make_op(tmp_path) -> Op:
    """An Op with no server/sync side effects, ready to log."""
    settings = Settings()
    settings.mode = 'noop'  # skips login/run-create in Op.__init__
    settings.dir = str(tmp_path)
    settings.meta = []  # shadow the class-level shared list (test isolation)
    os.makedirs(os.path.join(settings.get_dir(), 'files'), exist_ok=True)
    op = Op(config={}, settings=settings)
    # Past construction, act like a live run: 'noop' means nothing is sent
    # anywhere, so the log line is intentionally suppressed for it.
    op.settings.mode = 'online'
    return op


def _messages(caplog) -> str:
    return '\n'.join(r.getMessage() for r in caplog.records)


class TestLogNcclEnv:
    def test_logs_nccl_env_at_info(self, tmp_path, caplog):
        op = _make_op(tmp_path)
        env = {'NCCL_DEBUG': 'INFO', 'NCCL_SOCKET_IFNAME': 'eth0'}
        with mock.patch.dict(os.environ, env, clear=False):
            with caplog.at_level(logging.INFO, logger='pluto'):
                op._log_nccl_env()
        out = _messages(caplog)
        assert 'NCCL_DEBUG=INFO' in out
        assert 'NCCL_SOCKET_IFNAME=eth0' in out

    def test_nothing_logged_when_no_nccl_vars_set(self, tmp_path, caplog):
        op = _make_op(tmp_path)
        with mock.patch('pluto.op.collect_nccl_env', return_value={}):
            with caplog.at_level(logging.INFO, logger='pluto'):
                op._log_nccl_env()
        assert 'NCCL environment' not in _messages(caplog)

    def test_credentials_are_masked_in_the_log_line(self, tmp_path, caplog):
        op = _make_op(tmp_path)
        with mock.patch.dict(os.environ, {'NCCL_AUTH_TOKEN': 'hunter2'}):
            with caplog.at_level(logging.INFO, logger='pluto'):
                op._log_nccl_env()
        out = _messages(caplog)
        assert 'hunter2' not in out
        assert MASKED_VALUE in out

    def test_long_env_is_truncated(self, tmp_path, caplog):
        op = _make_op(tmp_path)
        env = {f'NCCL_PAD_{i:03d}': 'x' * 64 for i in range(100)}
        with mock.patch('pluto.op.collect_nccl_env', return_value=env):
            with caplog.at_level(logging.INFO, logger='pluto'):
                op._log_nccl_env()
        out = _messages(caplog)
        assert '(truncated)' in out
        # The cap bounds the rendered vars, not the whole formatted message.
        assert len(out) < NCCL_ENV_LOG_MAX_CHARS + 200

    def test_suppressed_under_disable_system_metrics(self, tmp_path, caplog):
        """Backfill/migration: this host's env says nothing about the run."""
        op = _make_op(tmp_path)
        op.settings.disable_system_metrics = True
        with mock.patch.dict(os.environ, {'NCCL_DEBUG': 'INFO'}):
            with caplog.at_level(logging.INFO, logger='pluto'):
                op._log_nccl_env()
        assert 'NCCL environment' not in _messages(caplog)

    def test_suppressed_in_noop_mode(self, tmp_path, caplog):
        op = _make_op(tmp_path)
        op.settings.mode = 'noop'
        with mock.patch.dict(os.environ, {'NCCL_DEBUG': 'INFO'}):
            with caplog.at_level(logging.INFO, logger='pluto'):
                op._log_nccl_env()
        assert 'NCCL environment' not in _messages(caplog)

    def test_collection_failure_never_raises(self, tmp_path, caplog):
        op = _make_op(tmp_path)
        with mock.patch('pluto.op.collect_nccl_env', side_effect=RuntimeError('boom')):
            with caplog.at_level(logging.INFO, logger='pluto'):
                op._log_nccl_env()  # must not propagate
        assert 'NCCL environment (' not in _messages(caplog)

    def test_start_emits_the_line(self, tmp_path, caplog):
        """The hook is wired into Op.start(), not just callable on its own."""
        op = _make_op(tmp_path)
        op._monitor = mock.MagicMock()
        op._sync_manager = None
        op._iface = None
        # start() publishes module-level globals and appends to pluto.ops; put
        # them back so a later test in this worker sees an untouched module.
        saved = (pluto.ops, pluto.log, pluto.alert, pluto.watch)
        try:
            with mock.patch.dict(os.environ, {'NCCL_DEBUG': 'INFO'}, clear=False):
                with caplog.at_level(logging.INFO, logger='pluto'):
                    op.start()
        finally:
            atexit.unregister(op.finish)
            pluto.ops, pluto.log, pluto.alert, pluto.watch = saved
        assert 'NCCL_DEBUG=INFO' in _messages(caplog)
