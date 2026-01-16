"""
Tests for shutdown behavior and file handling fixes.

These tests verify:
1. _stat is refreshed after _mkcopy() to prevent S3 signature mismatch
2. Thread joins have timeouts to prevent hang during shutdown
3. Connection errors are treated as shutdown signals, not retriable errors
4. Signal handling for graceful Ctrl+C shutdown
"""

import os
import signal
import threading
from unittest.mock import MagicMock, patch

from pluto.file import Artifact, File


class TestFileStatRefresh:
    """Test that _stat is refreshed after _mkcopy() to match actual file."""

    def test_stat_matches_copied_file(self, tmp_path):
        """Verify _stat reflects the temp copy, not the original file."""
        # Create a test file
        original = tmp_path / 'test.yaml'
        original.write_text('key: value\n')
        original_size = original.stat().st_size

        # Create File instance
        f = File(str(original))
        assert f._stat.st_size == original_size

        # Create temp directory for copy
        copy_dir = tmp_path / 'pluto_run'
        copy_dir.mkdir()
        (copy_dir / 'files').mkdir()

        # Simulate file modification before _mkcopy (race condition scenario)
        original.write_text('key: value\nextra: data\n')
        new_size = original.stat().st_size
        assert new_size > original_size

        # _mkcopy should refresh _stat to match the copied file
        f._mkcopy(str(copy_dir))

        # _stat should now reflect the actual copied file size
        assert f._stat.st_size == new_size
        assert f._stat.st_size == os.stat(f._path).st_size

    def test_artifact_stat_after_load_and_copy(self, tmp_path):
        """Test Artifact specifically since it was mentioned in the bug report."""
        # Create a YAML file (the type mentioned in the bug)
        yaml_file = tmp_path / 'config.yaml'
        yaml_file.write_text('learning_rate: 0.001\nbatch_size: 32\n')

        # Create Artifact
        artifact = Artifact(str(yaml_file))

        # Create temp directory
        copy_dir = tmp_path / 'pluto_run'
        copy_dir.mkdir()
        (copy_dir / 'files').mkdir()

        # Load and copy
        artifact.load(str(copy_dir))
        artifact._mkcopy(str(copy_dir))

        # Verify _stat matches the actual file at _path
        assert artifact._stat.st_size == os.stat(artifact._path).st_size

    def test_mkcopy_only_runs_once(self, tmp_path):
        """Test that _mkcopy only copies on first call (idempotent)."""
        original = tmp_path / 'test.txt'
        original.write_text('content')

        f = File(str(original))

        copy_dir = tmp_path / 'pluto_run'
        copy_dir.mkdir()
        (copy_dir / 'files').mkdir()

        # First call should copy
        f._mkcopy(str(copy_dir))
        first_path = f._path
        first_stat = f._stat

        # Second call should be no-op
        f._mkcopy(str(copy_dir))
        assert f._path == first_path
        assert f._stat == first_stat


class TestConnectionErrorHandling:
    """Test that connection errors don't cause infinite retries."""

    def test_broken_pipe_no_retry(self):
        """Test BrokenPipeError causes immediate return, not retry."""
        from pluto.iface import ServerInterface
        from pluto.sets import Settings

        settings = Settings()
        settings._op_id = 'test-op-id'
        settings._run_id = 12345
        settings.x_file_stream_retry_max = 4

        with patch('pluto.iface.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            iface = ServerInterface({}, settings)

            # Mock method that raises BrokenPipeError
            def raise_broken_pipe(*args, **kwargs):
                raise BrokenPipeError('Connection closed')

            mock_method = MagicMock(side_effect=raise_broken_pipe)
            mock_method.__name__ = 'post'

            # _try should return None immediately, not retry
            result = iface._try(
                mock_method,
                'http://example.com',
                {},
                b'content',
                name='test',
            )

            assert result is None
            # Should only be called once (no retries)
            assert mock_method.call_count == 1

    def test_connection_reset_no_retry(self):
        """Test ConnectionResetError causes immediate return, not retry."""
        from pluto.iface import ServerInterface
        from pluto.sets import Settings

        settings = Settings()
        settings._op_id = 'test-op-id'
        settings._run_id = 12345
        settings.x_file_stream_retry_max = 4

        with patch('pluto.iface.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            iface = ServerInterface({}, settings)

            def raise_connection_reset(*args, **kwargs):
                raise ConnectionResetError('Connection reset by peer')

            mock_method = MagicMock(side_effect=raise_connection_reset)
            mock_method.__name__ = 'post'

            result = iface._try(
                mock_method,
                'http://example.com',
                {},
                b'content',
                name='test',
            )

            assert result is None
            assert mock_method.call_count == 1

    def test_regular_exception_does_retry(self):
        """Test that regular exceptions still trigger retries."""
        from pluto.iface import ServerInterface
        from pluto.sets import Settings

        settings = Settings()
        settings._op_id = 'test-op-id'
        settings._run_id = 12345
        settings.x_file_stream_retry_max = 2
        settings.x_file_stream_retry_wait_min_seconds = 0.001
        settings.x_file_stream_retry_wait_max_seconds = 0.001

        with patch('pluto.iface.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            iface = ServerInterface({}, settings)

            def raise_timeout(*args, **kwargs):
                raise TimeoutError('Request timed out')

            mock_method = MagicMock(side_effect=raise_timeout)
            mock_method.__name__ = 'post'

            result = iface._try(
                mock_method,
                'http://example.com',
                {},
                b'content',
                name='test',
            )

            assert result is None
            # retry=0 (first try), then retry=1, then retry=2 hits max and returns
            # So total calls = x_file_stream_retry_max
            assert mock_method.call_count == settings.x_file_stream_retry_max


class TestThreadJoinTimeout:
    """Test that thread joins use timeouts."""

    def test_thread_join_called_with_timeout(self):
        """Verify thread.join() is called with timeout parameter."""
        from pluto.iface import ServerInterface
        from pluto.sets import Settings

        settings = Settings()
        settings._op_id = 'test-op-id'
        settings._run_id = 12345

        with patch('pluto.iface.httpx.Client'):
            iface = ServerInterface({}, settings)

            # Create a mock thread
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False

            # Assign mock threads
            iface._thread_num = mock_thread
            iface._thread_data = None
            iface._thread_file = None
            iface._thread_storage = None
            iface._thread_message = None
            iface._thread_meta = None
            iface._thread_progress = None

            # Mock the methods that would make network calls
            iface._update_status = MagicMock()
            iface.save = MagicMock()
            iface._stop_event = MagicMock()
            iface._progress = MagicMock()
            iface._progress_task = None

            iface.stop()

            # Verify join was called with configured timeout
            mock_thread.join.assert_called_once_with(
                timeout=settings.x_thread_join_timeout_seconds
            )

    def test_warning_logged_when_thread_alive_after_timeout(self):
        """Test that a warning is logged if thread doesn't terminate."""
        from pluto.iface import ServerInterface
        from pluto.sets import Settings

        settings = Settings()
        settings._op_id = 'test-op-id'
        settings._run_id = 12345

        with patch('pluto.iface.httpx.Client'):
            iface = ServerInterface({}, settings)

            # Create a mock thread that stays alive
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = True  # Thread didn't terminate
            mock_thread.name = 'test-thread'

            iface._thread_num = mock_thread
            iface._thread_data = None
            iface._thread_file = None
            iface._thread_storage = None
            iface._thread_message = None
            iface._thread_meta = None
            iface._thread_progress = None

            iface._update_status = MagicMock()
            iface.save = MagicMock()
            iface._stop_event = MagicMock()
            iface._progress = MagicMock()
            iface._progress_task = None

            # Should complete without hanging even with stuck thread
            with patch('pluto.iface.logger') as mock_logger:
                iface.stop()
                # Verify warning was logged
                mock_logger.warning.assert_called()


class TestSignalHandling:
    """Test SIGINT/SIGTERM signal handling for graceful shutdown."""

    def setup_method(self):
        """Reset signal handling state before each test."""
        import pluto.op as op_module

        op_module._signal_count = 0
        op_module._signal_handler_registered = False
        op_module._original_sigint_handler = None
        op_module._original_sigterm_handler = None

    def teardown_method(self):
        """Clean up signal handling state after each test."""
        import pluto.op as op_module

        # Reset state
        op_module._signal_count = 0
        op_module._signal_handler_registered = False

        # Restore default handlers if we changed them
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, signal.default_int_handler)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
            except (ValueError, OSError):
                pass

    def test_signal_handler_registration_in_main_thread(self):
        """Test that signal handlers are registered when in main thread."""
        import pluto.op as op_module

        # Only run in main thread
        if threading.current_thread() is not threading.main_thread():
            return

        assert not op_module._signal_handler_registered

        op_module._register_signal_handler()

        assert op_module._signal_handler_registered
        # Verify both handlers were actually set
        assert signal.getsignal(signal.SIGINT) == op_module._shutdown_handler
        assert signal.getsignal(signal.SIGTERM) == op_module._shutdown_handler

    def test_signal_handler_only_registered_once(self):
        """Test that signal handler registration is idempotent."""
        import pluto.op as op_module

        if threading.current_thread() is not threading.main_thread():
            return

        op_module._register_signal_handler()
        first_handler = signal.getsignal(signal.SIGINT)

        # Register again - should be no-op
        op_module._register_signal_handler()
        second_handler = signal.getsignal(signal.SIGINT)

        assert first_handler == second_handler
        assert op_module._signal_handler_registered

    def test_first_sigint_increments_count(self):
        """Test that first SIGINT increments count and starts graceful shutdown."""
        import pluto
        import pluto.op as op_module

        # Mock pluto.ops to prevent actual shutdown attempts
        original_ops = pluto.ops
        pluto.ops = []

        try:
            with (
                patch.object(op_module, 'logger') as mock_logger,
                patch('sys.exit') as mock_exit,
            ):
                # Simulate SIGINT
                op_module._shutdown_handler(signal.SIGINT, None)

                assert op_module._signal_count == 1
                mock_logger.warning.assert_called()
                mock_exit.assert_called_once_with(128 + signal.SIGINT)
        finally:
            pluto.ops = original_ops

    def test_first_sigterm_increments_count(self):
        """Test that first SIGTERM increments count and starts graceful shutdown."""
        import pluto
        import pluto.op as op_module

        original_ops = pluto.ops
        pluto.ops = []

        try:
            with (
                patch.object(op_module, 'logger') as mock_logger,
                patch('sys.exit') as mock_exit,
            ):
                # Simulate SIGTERM (K8s termination)
                op_module._shutdown_handler(signal.SIGTERM, None)

                assert op_module._signal_count == 1
                mock_logger.warning.assert_called()
                # SIGTERM exit code is 128 + 15 = 143
                mock_exit.assert_called_once_with(128 + signal.SIGTERM)
        finally:
            pluto.ops = original_ops

    def test_second_sigint_force_exits(self):
        """Test that second SIGINT forces immediate exit."""
        import pluto.op as op_module

        # Set count to 1 to simulate first signal already received
        op_module._signal_count = 1

        with patch('os._exit') as mock_exit:
            op_module._shutdown_handler(signal.SIGINT, None)

            assert op_module._signal_count == 2
            mock_exit.assert_called_once_with(128 + signal.SIGINT)

    def test_second_sigterm_force_exits(self):
        """Test that second SIGTERM forces immediate exit."""
        import pluto.op as op_module

        # Set count to 1 to simulate first signal already received
        op_module._signal_count = 1

        with patch('os._exit') as mock_exit:
            op_module._shutdown_handler(signal.SIGTERM, None)

            assert op_module._signal_count == 2
            mock_exit.assert_called_once_with(128 + signal.SIGTERM)

    def test_sigint_handler_calls_finish_on_active_ops(self):
        """Test that SIGINT handler calls finish() on all active ops."""
        import pluto
        import pluto.op as op_module

        # Create mock ops
        mock_op1 = MagicMock()
        mock_op2 = MagicMock()
        original_ops = pluto.ops
        pluto.ops = [mock_op1, mock_op2]

        try:
            with patch('sys.exit'):
                op_module._shutdown_handler(signal.SIGINT, None)

                # Both ops should have finish called
                mock_op1.finish.assert_called_once_with(code=signal.SIGINT)
                mock_op2.finish.assert_called_once_with(code=signal.SIGINT)
        finally:
            pluto.ops = original_ops

    def test_sigterm_handler_calls_finish_on_active_ops(self):
        """Test that SIGTERM handler calls finish() on all active ops."""
        import pluto
        import pluto.op as op_module

        mock_op1 = MagicMock()
        mock_op2 = MagicMock()
        original_ops = pluto.ops
        pluto.ops = [mock_op1, mock_op2]

        try:
            with patch('sys.exit'):
                op_module._shutdown_handler(signal.SIGTERM, None)

                # Both ops should have finish called with SIGTERM code
                mock_op1.finish.assert_called_once_with(code=signal.SIGTERM)
                mock_op2.finish.assert_called_once_with(code=signal.SIGTERM)
        finally:
            pluto.ops = original_ops

    def test_handler_handles_finish_exceptions(self):
        """Test that handler continues even if finish() raises."""
        import pluto
        import pluto.op as op_module

        # Create mock ops - first one raises, second should still be called
        mock_op1 = MagicMock()
        mock_op1.finish.side_effect = Exception('Simulated error')
        mock_op2 = MagicMock()

        original_ops = pluto.ops
        pluto.ops = [mock_op1, mock_op2]

        try:
            with patch('sys.exit'):
                # Should not raise despite first op failing
                op_module._shutdown_handler(signal.SIGINT, None)

                # Second op should still have finish called
                mock_op2.finish.assert_called_once_with(code=signal.SIGINT)
        finally:
            pluto.ops = original_ops

    def test_signal_count_thread_safety(self):
        """Test that _signal_count is thread-safe."""
        import pluto
        import pluto.op as op_module

        original_ops = pluto.ops
        pluto.ops = []

        try:
            results = []

            def call_handler():
                op_module._shutdown_handler(signal.SIGINT, None)
                with op_module._signal_lock:
                    results.append(op_module._signal_count)

            # Patch sys.exit and os._exit at module level for all threads
            with patch('sys.exit'), patch('os._exit'):
                threads = [threading.Thread(target=call_handler) for _ in range(5)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            # Count should be exactly 5 after 5 calls
            assert op_module._signal_count == 5
        finally:
            pluto.ops = original_ops

    def test_register_skipped_in_non_main_thread(self):
        """Test that signal handler registration is skipped in non-main thread."""
        import pluto.op as op_module

        result = {'registered': None}

        def register_in_thread():
            op_module._register_signal_handler()
            result['registered'] = op_module._signal_handler_registered

        thread = threading.Thread(target=register_in_thread)
        thread.start()
        thread.join()

        # Should not have registered since we're not in main thread
        assert result['registered'] is False

    def test_unregister_restores_original_handlers(self):
        """Test that _unregister_signal_handler restores original handlers."""
        import pluto.op as op_module

        if threading.current_thread() is not threading.main_thread():
            return

        # Store original handlers
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        # Register our handlers
        op_module._register_signal_handler()
        assert op_module._signal_handler_registered
        assert signal.getsignal(signal.SIGINT) == op_module._shutdown_handler

        # Unregister - should restore originals
        op_module._unregister_signal_handler()
        assert not op_module._signal_handler_registered
        assert signal.getsignal(signal.SIGINT) == original_sigint
        assert signal.getsignal(signal.SIGTERM) == original_sigterm


class TestFinishIdempotency:
    """Test that Op.finish() is idempotent."""

    def test_finish_only_executes_once(self):
        """Test that finish() only executes cleanup once even if called multiple times."""
        from pluto.op import Op
        from pluto.sets import Settings

        settings = Settings()
        settings.mode = 'noop'  # Skip server communication

        op = Op(config={}, settings=settings)
        op.start()

        # First finish should work
        op.finish()
        assert op._finished is True

        # Second finish should be a no-op (no errors)
        op.finish()
        assert op._finished is True

    def test_finish_thread_safe(self):
        """Test that concurrent finish() calls don't cause issues."""
        from pluto.op import Op
        from pluto.sets import Settings

        settings = Settings()
        settings.mode = 'noop'

        op = Op(config={}, settings=settings)
        op.start()

        finish_count = {'count': 0}
        original_monitor_stop = op._monitor.stop

        def counting_stop(code=None):
            finish_count['count'] += 1
            return original_monitor_stop(code)

        op._monitor.stop = counting_stop

        # Call finish from multiple threads
        threads = [threading.Thread(target=op.finish) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Monitor.stop should only be called once
        assert finish_count['count'] == 1
