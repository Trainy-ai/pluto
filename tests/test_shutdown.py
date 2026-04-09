"""
Tests for shutdown behavior and file handling fixes.

These tests verify:
1. _stat is refreshed after _mkcopy() to prevent S3 signature mismatch
2. Thread joins have timeouts to prevent hang during shutdown
3. Connection errors are treated as shutdown signals, not retriable errors
4. Signal handling for graceful Ctrl+C shutdown
5. Process terminates promptly on SIGTERM (integration test)
6. Sentry breadcrumb isolation for Pluto's internal HTTP traffic
7. Scoped httpx log suppression during Pluto's HTTP calls
"""

import logging
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

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

    def _make_iface(self, **overrides):
        """Helper to create a ServerInterface with test settings."""
        from pluto.iface import ServerInterface
        from pluto.sets import Settings

        settings = Settings()
        settings._op_id = 'test-op-id'
        settings._run_id = 12345
        settings.x_file_stream_retry_max = 4
        settings.x_file_stream_retry_wait_min_seconds = 0.001
        settings.x_file_stream_retry_wait_max_seconds = 0.001
        for k, v in overrides.items():
            setattr(settings, k, v)

        with patch('pluto.iface.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            iface = ServerInterface({}, settings)
        return iface

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
            # max_retries = x_file_stream_retry_max (default 4)
            # 1 initial attempt + 4 retries = 5 total calls
            assert mock_method.call_count == settings.x_file_stream_retry_max + 1

    def test_max_retries_override_limits_attempts(self):
        """Test that max_retries parameter overrides the settings default."""
        iface = self._make_iface()

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
            max_retries=1,
        )

        assert result is None
        # max_retries=1 means: 1 initial attempt + 1 retry = 2 total calls
        assert mock_method.call_count == 2

    def test_max_retries_zero_tries_once(self):
        """Test that max_retries=0 makes a single attempt with no retries."""
        iface = self._make_iface()

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
            max_retries=0,
        )

        assert result is None
        # max_retries=0 means: 1 initial attempt, 0 retries = 1 total call
        assert mock_method.call_count == 1

    def test_http_502_retries_up_to_max(self):
        """Test that HTTP 502 responses trigger retries up to max_retries."""
        iface = self._make_iface()

        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = '<html>502 Bad Gateway</html>'

        mock_method = MagicMock(return_value=mock_response)
        mock_method.__name__ = 'post'

        result = iface._try(
            mock_method,
            'http://example.com',
            {},
            b'content',
            name='test',
            max_retries=2,
        )

        assert result is None
        # max_retries=2: 1 initial attempt + 2 retries = 3 total calls
        assert mock_method.call_count == 3

    def test_http_502_with_max_retries_0_tries_once(self):
        """Test heartbeat-like call: max_retries=0 tries once on 502."""
        iface = self._make_iface()

        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = '<html>502 Bad Gateway</html>'

        mock_method = MagicMock(return_value=mock_response)
        mock_method.__name__ = 'post'

        result = iface._try(
            mock_method,
            'http://example.com',
            {},
            b'content',
            name='trigger',
            max_retries=0,
        )

        assert result is None
        # max_retries=0: 1 initial attempt, 0 retries = 1 total call
        assert mock_method.call_count == 1

    def test_timeout_kwarg_passed_to_http_method(self):
        """Test that timeout parameter is forwarded to the HTTP method."""
        iface = self._make_iface()

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_method = MagicMock(return_value=mock_response)
        mock_method.__name__ = 'post'

        result = iface._try(
            mock_method,
            'http://example.com',
            {'Content-Type': 'application/json'},
            b'content',
            name='test',
            timeout=5.0,
        )

        assert result is not None
        mock_method.assert_called_once_with(
            'http://example.com',
            content=b'content',
            headers={'Content-Type': 'application/json'},
            timeout=5.0,
        )

    def test_no_timeout_kwarg_when_none(self):
        """Test that timeout is not passed when set to None (default)."""
        iface = self._make_iface()

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_method = MagicMock(return_value=mock_response)
        mock_method.__name__ = 'post'

        iface._try(
            mock_method,
            'http://example.com',
            {},
            b'content',
            name='test',
            timeout=None,
        )

        # When timeout is None, kwargs should be empty (no timeout key)
        mock_method.assert_called_once_with(
            'http://example.com',
            content=b'content',
            headers={},
        )

    def test_post_v1_passes_max_retries_and_timeout(self):
        """Test that _post_v1 forwards max_retries and timeout to _try."""
        iface = self._make_iface()

        with patch.object(iface, '_try', return_value=None) as mock_try:
            mock_client = MagicMock()
            iface._post_v1(
                'http://example.com',
                {},
                b'payload',
                mock_client,
                name='trigger',
                max_retries=0,
                timeout=5.0,
            )

            mock_try.assert_called_once()
            call_kwargs = mock_try.call_args
            assert call_kwargs[1]['max_retries'] == 0
            assert call_kwargs[1]['timeout'] == 5.0


class TestSentryBreadcrumbSuppression:
    """Test that Pluto's HTTP calls don't leak breadcrumbs to the host's Sentry."""

    def _make_iface(self):
        """Helper to create a ServerInterface with test settings."""
        from pluto.iface import ServerInterface
        from pluto.sets import Settings

        settings = Settings()
        settings._op_id = 'test-op-id'
        settings._run_id = 12345
        settings.x_file_stream_retry_max = 0
        settings.x_file_stream_retry_wait_min_seconds = 0.001
        settings.x_file_stream_retry_wait_max_seconds = 0.001

        with patch('pluto.iface.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            iface = ServerInterface({}, settings)
        return iface

    def test_suppress_context_manager_without_sentry(self):
        """Context manager works when sentry_sdk is not installed."""
        from pluto.iface import _suppress_sentry_breadcrumbs

        with patch.dict('sys.modules', {'sentry_sdk': None}):
            with _suppress_sentry_breadcrumbs():
                pass  # Should not raise

    def test_suppress_context_manager_with_sentry_v2(self):
        """Context manager uses isolation_scope on sentry_sdk 2.x."""
        from pluto.iface import _suppress_sentry_breadcrumbs

        mock_sentry = MagicMock()
        mock_sentry.isolation_scope = MagicMock()
        # Make isolation_scope a real context manager
        mock_scope = MagicMock()
        mock_sentry.isolation_scope.return_value.__enter__ = MagicMock(
            return_value=mock_scope
        )
        mock_sentry.isolation_scope.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch.dict('sys.modules', {'sentry_sdk': mock_sentry}):
            with _suppress_sentry_breadcrumbs():
                pass
            mock_sentry.isolation_scope.assert_called_once()

    def test_suppress_context_manager_with_sentry_v1(self):
        """Context manager uses Hub.push_scope on sentry_sdk 1.x."""
        from pluto.iface import _suppress_sentry_breadcrumbs

        mock_sentry = MagicMock(spec=[])  # No isolation_scope attr
        mock_hub = MagicMock()
        mock_scope = MagicMock()
        mock_hub.push_scope.return_value.__enter__ = MagicMock(return_value=mock_scope)
        mock_hub.push_scope.return_value.__exit__ = MagicMock(return_value=False)
        mock_sentry.Hub = MagicMock()
        mock_sentry.Hub.current = mock_hub

        with patch.dict('sys.modules', {'sentry_sdk': mock_sentry}):
            with _suppress_sentry_breadcrumbs():
                pass
            mock_hub.push_scope.assert_called_once()

    def test_try_wraps_http_call_in_sentry_suppression(self):
        """_try() wraps the HTTP method call with _suppress_sentry_breadcrumbs."""
        iface = self._make_iface()

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_method = MagicMock(return_value=mock_response)
        mock_method.__name__ = 'post'

        with patch('pluto.iface._suppress_sentry_breadcrumbs') as mock_suppress:
            mock_suppress.return_value.__enter__ = MagicMock()
            mock_suppress.return_value.__exit__ = MagicMock(return_value=False)

            iface._try(
                mock_method,
                'http://example.com',
                {},
                b'content',
                name='trigger',
            )

            mock_suppress.assert_called_once()

    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('sentry_sdk'),
        reason='sentry_sdk not installed',
    )
    def test_breadcrumbs_not_leaked_to_host_scope(self):
        """End-to-end: HTTP calls inside suppression don't add breadcrumbs
        to the host app's isolation scope."""
        import sentry_sdk

        # Simulate host app initialising Sentry
        sentry_sdk.init(
            dsn='https://examplePublicKey@o0.ingest.sentry.io/0',
            traces_sample_rate=0,
        )

        try:
            scope = sentry_sdk.get_isolation_scope()
            scope.clear_breadcrumbs()
            before = list(scope._breadcrumbs)

            from pluto.iface import _suppress_sentry_breadcrumbs

            with _suppress_sentry_breadcrumbs():
                # Manually add a breadcrumb (simulates what httpx integration does)
                sentry_sdk.add_breadcrumb(
                    category='http',
                    message='POST https://pluto-py.trainy.ai/api/runs/trigger',
                    level='info',
                )

            after = list(scope._breadcrumbs)
            assert after == before, (
                'Breadcrumbs leaked to host scope: '
                f'before={len(before)}, after={len(after)}'
            )
        finally:
            sentry_sdk.init()  # Reset global state


class TestHttpxLoggingSuppression:
    """Test that httpx logging is suppressed only during Pluto's HTTP calls."""

    def test_httpx_suppressed_during_try(self):
        """httpx logger is WARNING inside _try(), restored after."""
        from pluto.iface import ServerInterface
        from pluto.sets import Settings

        settings = Settings()
        settings._op_id = 'test-op-id'
        settings._run_id = 12345
        settings.x_file_stream_retry_max = 0
        settings.x_file_stream_retry_wait_min_seconds = 0.001
        settings.x_file_stream_retry_wait_max_seconds = 0.001

        with patch('pluto.iface.httpx.Client'):
            iface = ServerInterface({}, settings)

        httpx_logger = logging.getLogger('httpx')
        httpx_logger.setLevel(logging.DEBUG)

        levels_during_call = []

        def capture_level(*args, **kwargs):
            levels_during_call.append(httpx_logger.level)
            resp = MagicMock()
            resp.status_code = 200
            return resp

        mock_method = MagicMock(side_effect=capture_level)
        mock_method.__name__ = 'post'

        iface._try(mock_method, 'http://example.com', {}, b'', name='test')

        # During the call, httpx logger should have been WARNING
        assert levels_during_call[0] == logging.WARNING
        # After the call, restored to DEBUG
        assert httpx_logger.level == logging.DEBUG

    def test_httpx_level_restored_on_exception(self):
        """httpx logger level is restored even if the HTTP call raises."""
        from pluto.iface import _suppress_httpx_logging

        httpx_logger = logging.getLogger('httpx')
        httpx_logger.setLevel(logging.INFO)

        try:
            with _suppress_httpx_logging():
                assert httpx_logger.level == logging.WARNING
                raise RuntimeError('boom')
        except RuntimeError:
            pass

        assert httpx_logger.level == logging.INFO

    def test_e2e_try_suppresses_breadcrumbs_and_restores_logger(self):
        """_try() with a real HTTP server produces zero Sentry breadcrumbs
        and restores the httpx logger level afterwards."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        import httpx
        import sentry_sdk

        from pluto.iface import ServerInterface
        from pluto.sets import Settings

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self, *a):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status": "OK"}')

            def log_message(self, *a):
                pass

        srv = HTTPServer(('127.0.0.1', 0), _Handler)
        port = srv.server_address[1]
        url = f'http://127.0.0.1:{port}/api/runs/trigger'
        threading.Thread(target=srv.serve_forever, daemon=True).start()

        sentry_sdk.init(
            dsn='https://examplePublicKey@o0.ingest.sentry.io/0',
            traces_sample_rate=0.0,
        )

        settings = Settings()
        settings._op_id = 'test-op-id'
        settings._run_id = 12345
        settings.x_file_stream_retry_max = 0
        settings.x_file_stream_retry_wait_min_seconds = 0.001
        settings.x_file_stream_retry_wait_max_seconds = 0.001

        with patch('pluto.iface.httpx.Client'):
            iface = ServerInterface({}, settings)

        client = httpx.Client()

        try:
            logging.getLogger('httpx').setLevel(logging.NOTSET)
            sentry_sdk.get_current_scope().clear_breadcrumbs()
            sentry_sdk.get_isolation_scope().clear_breadcrumbs()

            # Make real HTTP calls through _try()
            for _ in range(3):
                iface._try(
                    client.post,
                    url,
                    {'Content-Type': 'application/json'},
                    b'{}',
                    name='trigger',
                    max_retries=0,
                    timeout=5.0,
                )

            # Capture what Sentry would attach to a crash report
            events = []
            sentry_sdk.get_client().options['before_send'] = lambda e, h: (
                events.append(e),
                None,
            )[1]
            try:
                raise RuntimeError('test')
            except Exception:
                sentry_sdk.capture_exception()

            crumbs = events[0].get('breadcrumbs', {}).get('values', [])
            trigger_crumbs = [b for b in crumbs if 'trigger' in str(b)]

            assert len(trigger_crumbs) == 0, (
                f'{len(trigger_crumbs)} trigger breadcrumbs leaked to Sentry'
            )
            assert logging.getLogger('httpx').level < logging.WARNING, (
                'httpx logger level was not restored after _try()'
            )
        finally:
            srv.shutdown()
            client.close()
            sentry_sdk.init()


class TestFinishIdempotency:
    """Test that Op.finish() is idempotent."""

    def test_finish_only_executes_once(self):
        """Test that finish() only executes cleanup once even if called twice."""
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


# Helper scripts used by integration tests below.  Spawned as subprocesses
# so we can send real signals and assert the process exits in time.
_TRAINING_SCRIPT = textwrap.dedent("""\
    import sys
    import time

    import pluto

    run = pluto.init(project="signal-test", settings={"mode": "noop"})

    sys.stdout.write("READY\\n")
    sys.stdout.flush()

    while True:
        time.sleep(0.1)
""")


# Reproduces the exact production failure: a PyTorch DataLoader worker
# hits "RuntimeError: unable to allocate shared memory" due to /dev/shm
# exhaustion, and we verify the process terminates promptly when SIGTERM
# arrives (instead of hanging with the heartbeat loop still running).
_TORCH_SHM_SCRIPT = textwrap.dedent("""\
    import os
    import shutil
    import subprocess
    import sys
    import time

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import pluto

    run = pluto.init(project="signal-test", settings={"mode": "noop"})

    # Fill /dev/shm to trigger the real shared memory error.
    # Leave 2MB for semaphores/queues; not enough for tensor data.
    fill_path = "/dev/shm/_pluto_test_fill"
    total, used, free = shutil.disk_usage("/dev/shm")
    fill_size = free - (2 * 1024 * 1024)
    subprocess.run(["fallocate", "-l", str(fill_size), fill_path], check=True)

    try:
        data = torch.randn(10000, 1000)  # ~40MB float32
        labels = torch.randint(0, 2, (10000,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=5000, num_workers=2)

        sys.stdout.write("READY\\n")
        sys.stdout.flush()

        # This will raise RuntimeError from the DataLoader worker
        batch = next(iter(loader))
    except RuntimeError as e:
        # The error happened - now sit in a "stuck" training loop.
        sys.stdout.write("ERROR_HIT\\n")
        sys.stdout.flush()
        while True:
            time.sleep(0.1)
    finally:
        if os.path.exists(fill_path):
            os.remove(fill_path)
""")


# Torchrun variant: rank 1 crashes from /dev/shm OOM, torchrun sends SIGTERM
# to rank 0.  Rank 0 wraps its training loop in ``except BaseException``
# (common in ML frameworks like PyTorch Lightning, HuggingFace Trainer, etc.).
# With no custom signal handler (the fix), OS-level SIGTERM kills rank 0
# immediately.  With the OLD handler (sys.exit → SystemExit), the except block
# would swallow the signal and rank 0 would hang, causing the entire torchrun
# job to stall until the elastic agent's grace-period SIGKILL (~30 s).
_TORCHRUN_SHM_SCRIPT = textwrap.dedent("""\
    import os
    import shutil
    import subprocess
    import sys
    import time

    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, TensorDataset

    import pluto

    rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group("gloo")
    run = pluto.init(project="signal-test", settings={"mode": "noop"})

    fill_path = "/dev/shm/_pluto_test_fill_torchrun"

    if rank == 0:
        # Simulate typical ML-framework training loop: broad exception handler
        # that would catch SystemExit if a custom signal handler raised it.
        sys.stderr.write("RANK0_TRAINING\\n")
        sys.stderr.flush()
        try:
            while True:
                time.sleep(0.1)
        except BaseException:
            # If we get here, a signal was converted to a Python exception
            # (the old bug).  Fall through to a stuck state.
            sys.stderr.write("RANK0_SIGNAL_SWALLOWED\\n")
            sys.stderr.flush()
        # If signal was swallowed, sit here forever — reproducing the hang.
        while True:
            time.sleep(0.1)
    else:
        # Rank 1: fill /dev/shm and trigger the real DataLoader OOM.
        try:
            total, used, free = shutil.disk_usage("/dev/shm")
            fill_size = free - (2 * 1024 * 1024)  # leave 2 MB
            subprocess.run(
                ["fallocate", "-l", str(fill_size), fill_path], check=True,
            )

            data = torch.randn(10000, 1000)   # ~40 MB float32
            labels = torch.randint(0, 2, (10000,))
            dataset = TensorDataset(data, labels)
            loader = DataLoader(dataset, batch_size=5000, num_workers=2)

            sys.stderr.write("RANK1_LOADING\\n")
            sys.stderr.flush()
            batch = next(iter(loader))   # should raise RuntimeError
        except RuntimeError:
            sys.stderr.write("RANK1_SHM_OOM\\n")
            sys.stderr.flush()
        finally:
            if os.path.exists(fill_path):
                os.remove(fill_path)

        # Crash this rank — torchrun will SIGTERM rank 0.
        sys.exit(1)
""")


try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestSignalTerminationIntegration:
    """Integration tests: send real signals to a subprocess, assert it exits.

    Pluto does NOT register signal handlers — it relies on default signal
    behavior (immediate process termination) plus atexit-registered finish().
    These tests verify the process actually dies when signalled, and that
    daemon threads (heartbeat, monitor) don't prevent termination.
    """

    _DEADLINE = 10

    def _spawn(self, script: str) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, '-c', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _wait_ready(self, proc: subprocess.Popen, timeout: float = 15):
        """Block until the child prints READY."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if line and b'READY' in line:
                return
            if proc.poll() is not None:
                raise RuntimeError(
                    f'Child exited early (rc={proc.returncode}): '
                    + (proc.stderr.read() or b'').decode()
                )
            time.sleep(0.05)
        raise TimeoutError('Child did not print READY in time')

    def _wait_for_line(
        self,
        proc: subprocess.Popen,
        marker: str,
        timeout: float = 15,
    ) -> bool:
        """Wait for a specific line from the child's stdout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if line and marker.encode() in line:
                return True
            if proc.poll() is not None:
                return False
            time.sleep(0.05)
        return False

    def test_sigterm_kills_process(self):
        """SIGTERM must kill the process promptly via default handler."""
        proc = self._spawn(_TRAINING_SCRIPT)
        try:
            self._wait_ready(proc)
            proc.send_signal(signal.SIGTERM)
            start = time.monotonic()
            proc.wait(timeout=self._DEADLINE)
            elapsed = time.monotonic() - start
            assert elapsed < self._DEADLINE, (
                f'Process took {elapsed:.1f}s to exit after SIGTERM'
            )
            # Default SIGTERM kills with negative signal code
            assert proc.returncode == -signal.SIGTERM
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_sigint_kills_process(self):
        """SIGINT must kill the process promptly."""
        proc = self._spawn(_TRAINING_SCRIPT)
        try:
            self._wait_ready(proc)
            proc.send_signal(signal.SIGINT)
            start = time.monotonic()
            proc.wait(timeout=self._DEADLINE)
            elapsed = time.monotonic() - start
            assert elapsed < self._DEADLINE, (
                f'Process took {elapsed:.1f}s to exit after SIGINT'
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    @pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
    def test_sigterm_after_shm_oom_exits_promptly(self):
        """Reproduce the production failure: DataLoader hits shared memory
        OOM, process stays alive, then SIGTERM must still kill it.

        Fills /dev/shm to trigger the real
        ``RuntimeError: unable to allocate shared memory`` from PyTorch,
        then sends SIGTERM and asserts the process exits promptly.
        """
        fill_path = '/dev/shm/_pluto_test_fill'
        proc = self._spawn(_TORCH_SHM_SCRIPT)
        try:
            self._wait_ready(proc)

            got_error = self._wait_for_line(proc, 'ERROR_HIT')
            assert got_error, (
                'DataLoader did not hit shared memory error — '
                'test environment may have too much /dev/shm'
            )

            # Process is "stuck" in recovery loop. SIGTERM must kill it.
            proc.send_signal(signal.SIGTERM)
            start = time.monotonic()
            proc.wait(timeout=self._DEADLINE)
            elapsed = time.monotonic() - start
            assert elapsed < self._DEADLINE, (
                f'Process took {elapsed:.1f}s to exit after SIGTERM '
                f'following shm OOM (would waste GPU time in production)'
            )
            assert proc.returncode == -signal.SIGTERM
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            # Safety cleanup in case child didn't remove the fill file
            if os.path.exists(fill_path):
                os.remove(fill_path)

    @pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
    def test_torchrun_exits_after_rank_shm_oom(self, tmp_path):
        """Under torchrun: rank 1 crashes from /dev/shm OOM, torchrun sends
        SIGTERM to rank 0.  Rank 0 wraps its loop in ``except BaseException``
        (as ML frameworks do).  With no custom signal handler, rank 0 dies
        from OS-level SIGTERM and torchrun exits promptly.

        With the old handler (sys.exit → SystemExit), the except block would
        swallow the signal and rank 0 would hang until torchrun's grace-period
        SIGKILL (~30 s), pushing total runtime well past the deadline.
        """
        script_path = tmp_path / 'torchrun_shm_test.py'
        script_path.write_text(_TORCHRUN_SHM_SCRIPT)

        fill_path = '/dev/shm/_pluto_test_fill_torchrun'
        torchrun_deadline = 30  # generous; normal case finishes in ~10-20 s

        proc = subprocess.Popen(
            [
                sys.executable,
                '-m',
                'torch.distributed.run',
                '--standalone',
                '--nproc-per-node=2',
                '--max-restarts=0',
                str(script_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            start = time.monotonic()
            proc.wait(timeout=60)
            elapsed = time.monotonic() - start
            assert elapsed < torchrun_deadline, (
                f'torchrun took {elapsed:.1f}s to exit — rank 0 likely '
                f'hung after SIGTERM (old signal handler bug)'
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail(
                'torchrun did not exit within 60s — rank 0 is stuck '
                '(old signal handler bug)'
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if os.path.exists(fill_path):
                os.remove(fill_path)
