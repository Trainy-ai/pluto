"""
Tests for shutdown behavior and file handling fixes.

These tests verify:
1. _stat is refreshed after _mkcopy() to prevent S3 signature mismatch
2. Thread joins have timeouts to prevent hang during shutdown
3. Connection errors are treated as shutdown signals, not retriable errors
"""

import os
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
