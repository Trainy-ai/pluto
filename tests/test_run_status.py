"""
Tests for run status logic (COMPLETED vs FAILED).

These tests verify:
1. Unhandled exceptions mark runs as FAILED (status=1) via sys.excepthook
2. Normal exit marks runs as COMPLETED (status=0)
3. Signal-based shutdown marks runs as TERMINATED
4. The excepthook preserves FAILED status through the finish() path
5. The STATUS map covers all expected status codes
"""

import signal
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock

import pytest

from pluto.api import STATUS
from pluto.op import (
    OpMonitor,
    _excepthook,
    _register_excepthook,
    _unregister_excepthook,
)


class TestStatusMap:
    """Test the STATUS map covers all expected codes."""

    def test_status_map_has_running(self):
        assert STATUS[-1] == 'RUNNING'

    def test_status_map_has_completed(self):
        assert STATUS[0] == 'COMPLETED'

    def test_status_map_has_failed(self):
        assert STATUS[1] == 'FAILED'

    def test_status_map_has_terminated_sigint(self):
        assert STATUS[signal.SIGINT.value] == 'TERMINATED'

    def test_status_map_has_terminated_sigterm(self):
        assert STATUS[signal.SIGTERM.value] == 'TERMINATED'


class TestExcepthook:
    """Test the sys.excepthook integration for FAILED status detection."""

    def setup_method(self):
        """Reset excepthook state before each test."""
        import pluto.op as op_module

        op_module._excepthook_registered = False
        op_module._original_excepthook = None

    def teardown_method(self):
        """Clean up excepthook state after each test."""
        import pluto.op as op_module

        # Restore original excepthook
        if op_module._original_excepthook is not None:
            sys.excepthook = op_module._original_excepthook
        elif sys.excepthook == _excepthook:
            sys.excepthook = sys.__excepthook__
        op_module._excepthook_registered = False
        op_module._original_excepthook = None

    def test_register_excepthook(self):
        """Test that _register_excepthook installs our hook."""
        import pluto.op as op_module

        original = sys.excepthook
        _register_excepthook()

        assert op_module._excepthook_registered
        assert sys.excepthook == _excepthook
        assert op_module._original_excepthook == original

    def test_register_excepthook_idempotent(self):
        """Test that registering twice doesn't double-wrap."""
        import pluto.op as op_module

        original = sys.excepthook
        _register_excepthook()
        _register_excepthook()

        assert op_module._original_excepthook == original
        assert sys.excepthook == _excepthook

    def test_unregister_excepthook(self):
        """Test that _unregister_excepthook restores the original hook."""
        import pluto.op as op_module

        original = sys.excepthook
        _register_excepthook()
        _unregister_excepthook()

        assert not op_module._excepthook_registered
        assert sys.excepthook == original

    def test_excepthook_marks_active_ops_as_failed(self):
        """Test that _excepthook sets _op_status=1 on active ops."""
        import pluto
        import pluto.op as op_module

        # Create mock ops with _op_status = -1 (RUNNING)
        mock_settings1 = MagicMock()
        mock_settings1._op_status = -1
        mock_settings1._op_id = 'test-1'
        mock_op1 = MagicMock()
        mock_op1.settings = mock_settings1

        mock_settings2 = MagicMock()
        mock_settings2._op_status = -1
        mock_settings2._op_id = 'test-2'
        mock_op2 = MagicMock()
        mock_op2.settings = mock_settings2

        original_ops = pluto.ops
        pluto.ops = [mock_op1, mock_op2]

        # Save original hook to call
        op_module._original_excepthook = MagicMock()

        try:
            # Simulate an unhandled ValueError
            try:
                raise ValueError('training failed')
            except ValueError:
                exc_type, exc_value, exc_tb = sys.exc_info()
                _excepthook(exc_type, exc_value, exc_tb)

            # Both ops should be marked as FAILED
            assert mock_settings1._op_status == 1
            assert mock_settings2._op_status == 1
        finally:
            pluto.ops = original_ops

    def test_excepthook_does_not_overwrite_non_running_status(self):
        """Test that _excepthook only marks RUNNING ops, not already-finished ones."""
        import pluto
        import pluto.op as op_module

        # Op already marked as TERMINATED (e.g., from SIGINT)
        mock_settings = MagicMock()
        mock_settings._op_status = signal.SIGINT.value
        mock_settings._op_id = 'test-1'
        mock_op = MagicMock()
        mock_op.settings = mock_settings

        original_ops = pluto.ops
        pluto.ops = [mock_op]

        op_module._original_excepthook = MagicMock()

        try:
            try:
                raise RuntimeError('crash')
            except RuntimeError:
                exc_type, exc_value, exc_tb = sys.exc_info()
                _excepthook(exc_type, exc_value, exc_tb)

            # Should NOT have been changed to FAILED
            assert mock_settings._op_status == signal.SIGINT.value
        finally:
            pluto.ops = original_ops

    def test_excepthook_calls_original_hook(self):
        """Test that _excepthook chains to the original hook for traceback printing."""
        import pluto
        import pluto.op as op_module

        original_ops = pluto.ops
        pluto.ops = []

        mock_original = MagicMock()
        op_module._original_excepthook = mock_original

        try:
            try:
                raise ValueError('test')
            except ValueError:
                exc_type, exc_value, exc_tb = sys.exc_info()
                _excepthook(exc_type, exc_value, exc_tb)

            mock_original.assert_called_once_with(exc_type, exc_value, exc_tb)
        finally:
            pluto.ops = original_ops

    def test_excepthook_handles_no_ops(self):
        """Test that _excepthook works when pluto.ops is None or empty."""
        import pluto
        import pluto.op as op_module

        op_module._original_excepthook = MagicMock()

        original_ops = pluto.ops
        pluto.ops = None

        try:
            try:
                raise ValueError('test')
            except ValueError:
                exc_type, exc_value, exc_tb = sys.exc_info()
                # Should not raise
                _excepthook(exc_type, exc_value, exc_tb)
        finally:
            pluto.ops = original_ops


class TestOpMonitorStatusPreservation:
    """Test that OpMonitor.stop() preserves FAILED status."""

    def test_stop_with_no_code_sets_completed(self):
        """Normal finish: _op_status goes from RUNNING (-1) to COMPLETED (0)."""
        mock_op = MagicMock()
        mock_op.settings._op_status = -1
        mock_op.settings.x_thread_join_timeout_seconds = 1

        monitor = OpMonitor(mock_op)
        monitor.stop(code=None)

        assert mock_op.settings._op_status == 0

    def test_stop_with_signal_code_sets_signal(self):
        """Signal finish: code from signal handler overrides status."""
        mock_op = MagicMock()
        mock_op.settings._op_status = -1
        mock_op.settings.x_thread_join_timeout_seconds = 1

        monitor = OpMonitor(mock_op)
        monitor.stop(code=signal.SIGTERM)

        assert mock_op.settings._op_status == signal.SIGTERM

    def test_stop_preserves_failed_status(self):
        """FAILED status set by excepthook should NOT be overwritten to COMPLETED."""
        mock_op = MagicMock()
        mock_op.settings._op_status = 1  # Already marked FAILED by excepthook
        mock_op.settings.x_thread_join_timeout_seconds = 1

        monitor = OpMonitor(mock_op)
        monitor.stop(code=None)  # Normal finish path (no signal code)

        # Should remain FAILED, not get overwritten to COMPLETED
        assert mock_op.settings._op_status == 1

    def test_stop_with_explicit_code_overrides_failed(self):
        """An explicit code (e.g., from signal) takes precedence over FAILED."""
        mock_op = MagicMock()
        mock_op.settings._op_status = 1  # FAILED
        mock_op.settings.x_thread_join_timeout_seconds = 1

        monitor = OpMonitor(mock_op)
        monitor.stop(code=signal.SIGTERM)

        # Explicit signal code should override
        assert mock_op.settings._op_status == signal.SIGTERM


class TestNoopRunStatus:
    """Test run status in noop mode (no server communication)."""

    def test_normal_finish_sets_completed(self):
        """A run that finishes normally should have _op_status=0."""
        from pluto.op import Op
        from pluto.sets import Settings

        settings = Settings()
        settings.mode = 'noop'

        op = Op(config={}, settings=settings)
        op.start()
        op.finish()

        assert settings._op_status == 0

    def test_finish_with_sigterm_sets_terminated(self):
        """A run finished with SIGTERM should have _op_status=SIGTERM."""
        from pluto.op import Op
        from pluto.sets import Settings

        settings = Settings()
        settings.mode = 'noop'

        op = Op(config={}, settings=settings)
        op.start()
        op.finish(code=signal.SIGTERM)

        assert settings._op_status == signal.SIGTERM

    def test_excepthook_then_finish_sets_failed(self):
        """Simulating: unhandled exception → excepthook → atexit finish()."""

        from pluto.op import Op
        from pluto.sets import Settings

        settings = Settings()
        settings.mode = 'noop'

        op = Op(config={}, settings=settings)
        op.start()

        # Simulate what happens when user code raises:
        # 1. sys.excepthook fires → marks as FAILED
        op.settings._op_status = 1

        # 2. atexit fires → calls op.finish()
        op.finish()

        # Status should still be FAILED
        assert settings._op_status == 1
        assert STATUS[settings._op_status] == 'FAILED'


class TestExcepthookSubprocess:
    """
    End-to-end test: run a script that raises an unhandled exception
    and verify the run gets marked as FAILED.
    """

    def test_unhandled_exception_marks_run_failed(self):
        """Run a subprocess that raises and verify _op_status is set to 1."""
        script = textwrap.dedent("""\
            import sys
            import json

            # Minimal mock to avoid needing server connectivity
            from unittest.mock import MagicMock, patch

            # Patch out server communication
            with patch('pluto.op.login'), \\
                 patch('pluto.op.ServerInterface'), \\
                 patch('pluto.op.setup_logger'), \\
                 patch('pluto.op.teardown_logger'), \\
                 patch('pluto.op.System') as MockSystem, \\
                 patch('pluto.op.SyncProcessManager'), \\
                 patch('pluto.op.to_json'), \\
                 patch('pluto.op.print_url', return_value='http://test'), \\
                 patch('os.makedirs'):

                MockSystem.return_value.get_info.return_value = {}
                MockSystem.return_value.monitor.return_value = {}

                from pluto.sets import Settings
                from pluto.op import Op

                settings = Settings()
                settings.mode = 'noop'

                op = Op(config={}, settings=settings)
                op.start()

                # Simulate unhandled exception by calling excepthook directly
                # (In real scenario, Python runtime calls this before atexit)
                try:
                    raise ValueError("training crashed!")
                except ValueError:
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    sys.excepthook(exc_type, exc_value, exc_tb)

                # Now simulate atexit calling finish
                op.finish()

                # Report the status
                print(json.dumps({"status": settings._op_status}))
        """)

        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # The script itself should exit cleanly (it handles the exception)
        # Parse the JSON output to check status
        stdout_lines = result.stdout.strip().split('\n')
        # Find the JSON line (last non-empty line)
        import json

        for line in reversed(stdout_lines):
            line = line.strip()
            if line.startswith('{'):
                data = json.loads(line)
                assert data['status'] == 1, (
                    f'Expected FAILED (1) but got {data["status"]} '
                    f'({STATUS.get(data["status"], "UNKNOWN")})'
                )
                return

        pytest.fail(
            f'No JSON output found.\nstdout: {result.stdout}\nstderr: {result.stderr}'
        )

    def test_normal_exit_marks_run_completed(self):
        """Run a subprocess that exits normally and verify _op_status is 0."""
        script = textwrap.dedent("""\
            import sys
            import json
            from unittest.mock import MagicMock, patch

            with patch('pluto.op.login'), \\
                 patch('pluto.op.ServerInterface'), \\
                 patch('pluto.op.setup_logger'), \\
                 patch('pluto.op.teardown_logger'), \\
                 patch('pluto.op.System') as MockSystem, \\
                 patch('pluto.op.SyncProcessManager'), \\
                 patch('pluto.op.to_json'), \\
                 patch('pluto.op.print_url', return_value='http://test'), \\
                 patch('os.makedirs'):

                MockSystem.return_value.get_info.return_value = {}
                MockSystem.return_value.monitor.return_value = {}

                from pluto.sets import Settings
                from pluto.op import Op

                settings = Settings()
                settings.mode = 'noop'

                op = Op(config={}, settings=settings)
                op.start()

                # Normal training - no exceptions
                for i in range(10):
                    pass

                op.finish()

                print(json.dumps({"status": settings._op_status}))
        """)

        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=10,
        )

        import json

        stdout_lines = result.stdout.strip().split('\n')
        for line in reversed(stdout_lines):
            line = line.strip()
            if line.startswith('{'):
                data = json.loads(line)
                assert data['status'] == 0, (
                    f'Expected COMPLETED (0) but got {data["status"]} '
                    f'({STATUS.get(data["status"], "UNKNOWN")})'
                )
                return

        pytest.fail(
            f'No JSON output found.\nstdout: {result.stdout}\nstderr: {result.stderr}'
        )
