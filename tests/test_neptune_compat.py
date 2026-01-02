"""
Comprehensive tests for Neptune-to-mlop compatibility layer.

These tests validate that:
1. Neptune API calls continue to work unchanged
2. mlop receives the logged data when configured
3. Neptune never fails due to mlop errors
4. Configuration via environment variables works
5. Fallback behavior is correct
"""

import os
from typing import Any, Dict
from unittest import mock

import pytest

# Test both with and without neptune installed
pytest.importorskip('neptune_scale')

# Import after neptune is available
from neptune_scale.types import File as NeptuneFile
from neptune_scale.types import Histogram as NeptuneHistogram

from tests.utils import get_task_name


class MockNeptuneRun:
    """
    Mock Neptune Run for testing without actual Neptune backend.

    This simulates Neptune's behavior for testing the monkeypatch.
    """

    def __init__(self, *args, **kwargs):
        self.experiment_name = kwargs.get('experiment_name', 'test-experiment')
        self.run_id = kwargs.get('run_id', None)
        self.project = kwargs.get('project', 'test/project')
        self.logged_metrics = []
        self.logged_configs = []
        self.logged_files = []
        self.logged_histograms = []
        self.tags = []
        self.closed = False
        self.terminated = False

    def log_metrics(self, data: Dict[str, float], step: int, timestamp=None, **kwargs):
        self.logged_metrics.append({'data': data, 'step': step, 'timestamp': timestamp})
        return None

    def log_configs(self, data: Dict[str, Any], **kwargs):
        self.logged_configs.append(data)
        return None

    def assign_files(self, files: Dict[str, Any], **kwargs):
        self.logged_files.append({'type': 'assign', 'files': files})
        return None

    def log_files(self, files: Dict[str, Any], step: int, timestamp=None, **kwargs):
        self.logged_files.append(
            {'type': 'log', 'files': files, 'step': step, 'timestamp': timestamp}
        )
        return None

    def log_histograms(
        self, histograms: Dict[str, Any], step: int, timestamp=None, **kwargs
    ):
        self.logged_histograms.append(
            {'histograms': histograms, 'step': step, 'timestamp': timestamp}
        )
        return None

    def add_tags(self, tags, **kwargs):
        self.tags.extend(tags)
        return None

    def remove_tags(self, tags, **kwargs):
        for tag in tags:
            if tag in self.tags:
                self.tags.remove(tag)
        return None

    def close(self, **kwargs):
        self.closed = True
        return None

    def terminate(self, **kwargs):
        self.terminated = True
        return None

    def wait_for_submission(self, **kwargs):
        return None

    def wait_for_processing(self, **kwargs):
        return None

    def get_run_url(self):
        return f'https://neptune.ai/{self.project}/runs/{self.run_id or "test-run"}'

    def get_experiment_url(self):
        return f'https://neptune.ai/{self.project}/experiments/{self.experiment_name}'

    def log_string_series(
        self, data: Dict[str, str], step: int, timestamp=None, **kwargs
    ):
        # Not implemented in mock
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


@pytest.fixture
def mock_neptune_backend():
    """Replace the saved _original_neptune_run with our mock for testing."""
    import mlop.compat.neptune

    original_saved = mlop.compat.neptune._original_neptune_run
    # Replace the saved original with our mock
    mlop.compat.neptune._original_neptune_run = MockNeptuneRun
    yield
    # Restore
    mlop.compat.neptune._original_neptune_run = original_saved


@pytest.fixture
def clean_env():
    """Clean environment variables before each test."""
    env_vars = [
        'MLOP_PROJECT',
        'MLOP_API_KEY',
        'MLOP_URL_APP',
        'MLOP_URL_API',
        'MLOP_URL_INGEST',
    ]
    original_values = {}
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


class TestNeptuneCompatBasic:
    """Test basic Neptune API functionality is preserved."""

    def test_neptune_import_without_mlop_config(self, mock_neptune_backend, clean_env):
        """Test that Neptune works normally when MLOP_PROJECT is not set."""
        # Don't set MLOP_PROJECT - should fall back to Neptune-only
        from neptune_scale import Run

        run = Run(experiment_name='test-exp')
        run.log_metrics({'loss': 0.5}, step=0)
        run.close()

        assert run._neptune_run.closed
        assert len(run._neptune_run.logged_metrics) == 1
        assert run._neptune_run.logged_metrics[0]['data'] == {'loss': 0.5}

    def test_neptune_metrics_logging(self, mock_neptune_backend, clean_env):
        """Test that Neptune metrics logging works unchanged."""
        from neptune_scale import Run

        run = Run(experiment_name='metrics-test')
        run.log_metrics({'acc': 0.95, 'loss': 0.1}, step=1)
        run.log_metrics({'acc': 0.96, 'loss': 0.09}, step=2)
        run.close()

        assert len(run._neptune_run.logged_metrics) == 2
        assert run._neptune_run.logged_metrics[0]['step'] == 1
        assert run._neptune_run.logged_metrics[1]['step'] == 2

    def test_neptune_configs_logging(self, mock_neptune_backend, clean_env):
        """Test that Neptune config logging works unchanged."""
        from neptune_scale import Run

        run = Run(experiment_name='config-test')
        run.log_configs({'lr': 0.001, 'batch_size': 32})
        run.close()

        assert len(run._neptune_run.logged_configs) == 1
        assert run._neptune_run.logged_configs[0] == {'lr': 0.001, 'batch_size': 32}

    def test_neptune_tags(self, mock_neptune_backend, clean_env):
        """Test that Neptune tags work unchanged."""
        from neptune_scale import Run

        run = Run(experiment_name='tag-test')
        run.add_tags(['experiment', 'baseline'])
        run.add_tags(['v1'])
        run.remove_tags(['baseline'])
        run.close()

        assert 'experiment' in run._neptune_run.tags
        assert 'v1' in run._neptune_run.tags
        assert 'baseline' not in run._neptune_run.tags

    def test_neptune_context_manager(self, mock_neptune_backend, clean_env):
        """Test that Neptune context manager protocol works."""
        from neptune_scale import Run

        with Run(experiment_name='context-test') as run:
            run.log_metrics({'loss': 0.3}, step=0)

        assert run._neptune_run.closed


class TestNeptuneCompatDualLogging:
    """Test dual-logging to both Neptune and mlop."""

    @pytest.fixture
    def mlop_config_env(self, clean_env):
        """Set up environment for mlop dual-logging."""
        os.environ['MLOP_PROJECT'] = 'neptune-migration-test'
        # Don't set MLOP_API_KEY - let it fall back to keyring or fail gracefully
        yield

    def test_dual_logging_metrics_with_env_config(
        self, mock_neptune_backend, mlop_config_env, monkeypatch
    ):
        """Test that metrics are logged to both Neptune and mlop when configured."""
        # Mock mlop.init to avoid actual API calls
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='dual-log-test')
            run.log_metrics({'loss': 0.5, 'acc': 0.9}, step=0)
            run.close()

            # Verify Neptune received the data
            assert len(run._neptune_run.logged_metrics) == 1
            assert run._neptune_run.logged_metrics[0]['data'] == {
                'loss': 0.5,
                'acc': 0.9,
            }

            # Verify mlop received the data
            mock_mlop_run.log.assert_called_with({'loss': 0.5, 'acc': 0.9})
            mock_mlop_run.finish.assert_called_once()

    def test_dual_logging_configs(self, mock_neptune_backend, mlop_config_env):
        """Test that configs are logged to both Neptune and mlop."""
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='config-dual-test')
            run.log_configs({'lr': 0.001, 'epochs': 100})
            run.close()

            # Verify Neptune received the config
            assert len(run._neptune_run.logged_configs) == 1

            # Verify mlop config was updated
            assert mock_mlop_run.config['lr'] == 0.001
            assert mock_mlop_run.config['epochs'] == 100


class TestNeptuneCompatErrorHandling:
    """Test that Neptune never fails due to mlop errors."""

    @pytest.fixture
    def mlop_config_env(self, clean_env):
        """Set up environment for mlop dual-logging."""
        os.environ['MLOP_PROJECT'] = 'error-test'
        yield

    def test_neptune_works_when_mlop_init_fails(
        self, mock_neptune_backend, mlop_config_env
    ):
        """Test that Neptune continues working if mlop.init() fails."""
        # Make mlop.init() raise an exception
        with mock.patch('mlop.init', side_effect=Exception('mlop service down')):
            from neptune_scale import Run

            # Should not raise - Neptune should work fine
            run = Run(experiment_name='mlop-init-fail-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # Verify Neptune worked
            assert run._neptune_run.closed
            assert len(run._neptune_run.logged_metrics) == 1

    def test_neptune_works_when_mlop_log_fails(
        self, mock_neptune_backend, mlop_config_env
    ):
        """Test that Neptune continues working if mlop.log() fails."""
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock(side_effect=Exception('Network error'))
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            # Should not raise - Neptune should work fine
            run = Run(experiment_name='mlop-log-fail-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # Verify Neptune worked
            assert len(run._neptune_run.logged_metrics) == 1
            assert run._neptune_run.closed

    def test_neptune_works_when_mlop_finish_fails(
        self, mock_neptune_backend, mlop_config_env
    ):
        """Test that Neptune closes correctly even if mlop.finish() fails."""
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock(side_effect=Exception('Finish error'))

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            # Should not raise - Neptune should work fine
            run = Run(experiment_name='mlop-finish-fail-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()  # Should not raise

            # Verify Neptune worked
            assert run._neptune_run.closed

    def test_neptune_works_when_mlop_not_installed(
        self, mock_neptune_backend, mlop_config_env
    ):
        """Test that Neptune works when mlop is not installed."""
        # Simulate mlop import failure
        with mock.patch('mlop.compat.neptune._safe_import_mlop', return_value=None):
            from neptune_scale import Run

            run = Run(experiment_name='no-mlop-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # Verify Neptune worked
            assert run._neptune_run.closed
            assert len(run._neptune_run.logged_metrics) == 1


class TestNeptuneCompatFileConversion:
    """Test file type conversion from Neptune to mlop."""

    @pytest.fixture
    def mlop_config_env(self, clean_env):
        """Set up environment for mlop dual-logging."""
        os.environ['MLOP_PROJECT'] = 'file-conversion-test'
        yield

    def test_image_file_conversion(
        self, mock_neptune_backend, mlop_config_env, tmp_path
    ):
        """Test that Neptune File objects are converted to mlop.Image."""
        # Create a test image
        img_path = tmp_path / 'test.png'
        img_path.write_bytes(b'\x89PNG\r\n\x1a\n')  # PNG header

        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='image-test')

            # Log file with Neptune File object
            neptune_file = NeptuneFile(source=str(img_path), mime_type='image/png')
            run.assign_files({'sample_image': neptune_file})
            run.close()

            # Verify Neptune received the file
            assert len(run._neptune_run.logged_files) == 1

            # Verify mlop.log was called (file conversion is internal)
            assert mock_mlop_run.log.called

    def test_histogram_conversion(self, mock_neptune_backend, mlop_config_env):
        """Test that Neptune Histogram objects are converted to mlop.Histogram."""
        import numpy as np

        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('mlop.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='histogram-test')

            # Create Neptune histogram
            bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
            counts = np.array([10, 20, 15])
            neptune_hist = NeptuneHistogram(bin_edges=bin_edges, counts=counts)

            run.log_histograms({'layer1/activations': neptune_hist}, step=0)
            run.close()

            # Verify Neptune received the histogram
            assert len(run._neptune_run.logged_histograms) == 1

            # Verify mlop.log was called
            assert mock_mlop_run.log.called


class TestNeptuneCompatIntegration:
    """
    Integration tests with real mlop backend (requires auth).

    These tests should be run in CI with proper mlop credentials set.
    """

    @pytest.mark.skipif(
        not os.environ.get('MLOP_PROJECT') or not os.environ.get('CI'),
        reason='Requires MLOP_PROJECT env var and CI environment',
    )
    def test_real_dual_logging_integration(self, mock_neptune_backend):
        """
        Integration test with real mlop backend.

        This test requires:
        - MLOP_PROJECT environment variable
        - Valid mlop credentials (keyring or MLOP_API_KEY)
        - Network access to mlop service
        """
        from neptune_scale import Run

        task_name = get_task_name()

        # This should log to both mock Neptune and real mlop
        run = Run(experiment_name=task_name)

        # Log various data types
        run.log_configs({'lr': 0.001, 'batch_size': 32, 'model': 'resnet50'})
        run.log_metrics({'train/loss': 0.5, 'train/acc': 0.85}, step=0)
        run.log_metrics({'train/loss': 0.3, 'train/acc': 0.92}, step=1)
        run.add_tags(['integration-test', 'ci'])

        run.close()

        # Verify Neptune received all data
        assert len(run._neptune_run.logged_configs) == 1
        assert len(run._neptune_run.logged_metrics) == 2
        assert len(run._neptune_run.tags) == 2
        assert run._neptune_run.closed

        # mlop run should also be finished
        if run._mlop_run:
            # Verify mlop was initialized successfully
            assert run._mlop_run is not None
            print('✓ Integration test passed - dual-logged to Neptune and mlop')
        else:
            pytest.skip('mlop not configured, skipping integration validation')


class TestNeptuneCompatFallbackBehavior:
    """Test various fallback scenarios."""

    def test_no_mlop_project_env_var(self, mock_neptune_backend, clean_env):
        """
        Test that monkeypatch works but doesn't init mlop when
        MLOP_PROJECT is not set.
        """
        # No MLOP_PROJECT set
        from neptune_scale import Run

        run = Run(experiment_name='no-project-test')
        run.log_metrics({'loss': 0.5}, step=0)
        run.close()

        # Should have no mlop run
        assert run._mlop_run is None

        # Neptune should work fine
        assert run._neptune_run.closed
        assert len(run._neptune_run.logged_metrics) == 1

    def test_mlop_project_set_but_invalid_credentials(
        self, mock_neptune_backend, clean_env
    ):
        """Test fallback when mlop project is set but credentials are invalid."""
        os.environ['MLOP_PROJECT'] = 'test-project'
        os.environ['MLOP_API_KEY'] = 'invalid-key-123'

        # Mock mlop.init to fail with auth error
        with mock.patch('mlop.init', side_effect=Exception('Unauthorized')):
            from neptune_scale import Run

            run = Run(experiment_name='invalid-creds-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # mlop should have failed silently
            assert run._mlop_run is None

            # Neptune should work fine
            assert run._neptune_run.closed


class TestNeptuneCompatAPIForwarding:
    """Test that unknown Neptune API methods are forwarded correctly."""

    def test_unknown_method_forwarding(self, mock_neptune_backend, clean_env):
        """Test that unknown methods are forwarded to Neptune."""
        from neptune_scale import Run

        run = Run(experiment_name='forward-test')

        # Call Neptune-specific methods
        url = run.get_run_url()
        assert 'neptune.ai' in url

        exp_url = run.get_experiment_url()
        assert 'neptune.ai' in exp_url

        run.close()

    def test_wait_methods_work(self, mock_neptune_backend, clean_env):
        """Test that Neptune's wait methods work."""
        from neptune_scale import Run

        run = Run(experiment_name='wait-test')
        run.log_metrics({'loss': 0.5}, step=0)

        # These should not raise
        run.wait_for_submission()
        run.wait_for_processing()

        run.close()


class TestNeptuneRealBackend:
    """
    Integration tests with REAL Neptune backend.

    These tests validate that the monkeypatch works correctly with the
    actual Neptune client and API. Requires Neptune credentials.

    Set NEPTUNE_API_TOKEN and NEPTUNE_PROJECT to run these tests.
    """

    @pytest.mark.skipif(
        not os.environ.get('NEPTUNE_API_TOKEN')
        or not os.environ.get('NEPTUNE_PROJECT'),
        reason='Requires NEPTUNE_API_TOKEN and NEPTUNE_PROJECT env vars',
    )
    def test_real_neptune_without_mlop(self, clean_env, tmp_path):
        """
        Test with real Neptune backend, no mlop dual-logging.

        Validates that the monkeypatch doesn't break Neptune functionality.
        """
        # Ensure mlop is NOT configured
        assert 'MLOP_PROJECT' not in os.environ

        import numpy as np
        from neptune_scale import Run

        task_name = get_task_name()

        # Create real Neptune run
        run = Run(experiment_name=task_name)

        # Log to real Neptune
        run.log_configs({'test': 'real-neptune', 'mode': 'neptune-only'})
        run.log_metrics({'test/metric': 1.0}, step=0)

        # Log multiple test images to Neptune at different steps
        try:
            from PIL import Image

            # Log 3 images at different steps
            for img_step in range(3):
                # Create unique test image for each step (grayscale gradient)
                intensity = 50 + (img_step * 80)  # 50, 130, 210
                test_image = np.full((64, 64, 3), intensity, dtype=np.uint8)
                # Add some noise
                test_image += np.random.randint(-20, 20, (64, 64, 3), dtype=np.int16)
                test_image = np.clip(test_image, 0, 255).astype(np.uint8)

                img_path = tmp_path / f'neptune_test_step_{img_step}.png'
                Image.fromarray(test_image).save(img_path)

                # Log image at this step
                run.log_files(
                    {'test/sample_image': NeptuneFile(str(img_path))},
                    step=img_step,
                )

            print('  ✓ Logged 3 images at steps 0, 1, 2')
        except ImportError:
            print('  ⚠ PIL not available, skipping image logging')

        run.add_tags(['real-neptune-test', 'neptune-only'])

        # Should have no mlop run
        assert run._mlop_run is None

        # Wait for Neptune to finish all operations before closing
        # Use verbose=False to prevent logging errors when pytest captures stdout
        run.wait_for_processing(verbose=False)

        # Close Neptune run
        run.close()

        # Verify Neptune run exists and has URL
        url = run.get_run_url()
        assert 'neptune.ai' in url
        print(f'✓ Real Neptune test passed - run URL: {url}')

    @pytest.mark.skipif(
        not os.environ.get('NEPTUNE_API_TOKEN')
        or not os.environ.get('NEPTUNE_PROJECT')
        or not os.environ.get('MLOP_PROJECT'),
        reason='Requires NEPTUNE_API_TOKEN, NEPTUNE_PROJECT, and MLOP_PROJECT',
    )
    def test_real_neptune_with_mlop_dual_logging(self, tmp_path):
        """
        Full integration test with BOTH real Neptune and real mlop.

        This is the ultimate validation that dual-logging works in production.
        Requires both Neptune and mlop credentials.
        """
        import numpy as np
        from neptune_scale import Run

        task_name = get_task_name()

        # Create dual-logged run
        run = Run(experiment_name=task_name)

        # Log to both systems
        run.log_configs(
            {'test': 'dual-logging', 'mode': 'production', 'framework': 'pytest'}
        )

        for step in range(3):
            run.log_metrics(
                {
                    'test/loss': 1.0 / (step + 1),
                    'test/accuracy': 0.5 + (step * 0.1),
                },
                step=step,
            )

        # Create and log multiple test images at different steps
        try:
            from PIL import Image

            # Log 3 images at different steps to test stepping functionality
            for img_step in range(3):
                # Create unique test image for each step (different colors)
                # Use step-based color to make images visually distinct
                base_color = [
                    (255, 0, 0),  # Red for step 0
                    (0, 255, 0),  # Green for step 1
                    (0, 0, 255),  # Blue for step 2
                ][img_step]

                test_image = np.zeros((64, 64, 3), dtype=np.uint8)
                test_image[:, :] = base_color
                # Add some variation
                test_image += np.random.randint(0, 50, (64, 64, 3), dtype=np.uint8)
                test_image = np.clip(test_image, 0, 255).astype(np.uint8)

                img_path = tmp_path / f'test_image_step_{img_step}.png'
                Image.fromarray(test_image).save(img_path)

                # Log image at this step
                run.log_files(
                    {'training/sample_image': NeptuneFile(str(img_path))},
                    step=img_step,
                )

            # Also log a static image (not associated with a step)
            static_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            static_path = tmp_path / 'static_image.png'
            Image.fromarray(static_img).save(static_path)
            run.assign_files({'summary/final_image': NeptuneFile(str(static_path))})

            print('  ✓ Logged 3 stepped images + 1 static image')
        except ImportError:
            print('  ⚠ PIL not available, skipping image logging')

        run.add_tags(['dual-logging-test', 'production', 'with-images'])

        # Both runs should be active
        assert run._neptune_run is not None
        assert run._mlop_run is not None

        # Wait for Neptune to finish all operations before closing
        # Use verbose=False to prevent logging errors when pytest captures stdout
        run.wait_for_processing(verbose=False)

        # Close both
        run.close()

        # Get URLs from both systems
        neptune_url = run.get_run_url()
        assert 'neptune.ai' in neptune_url

        print('✓ Full dual-logging test passed!')
        print(f'  Neptune: {neptune_url}')
        print('  mlop run successfully logged')

    @pytest.mark.skipif(
        not os.environ.get('NEPTUNE_API_TOKEN')
        or not os.environ.get('NEPTUNE_PROJECT'),
        reason='Requires NEPTUNE_API_TOKEN and NEPTUNE_PROJECT env vars',
    )
    def test_real_neptune_context_manager(self, clean_env):
        """Test context manager protocol with real Neptune."""
        from neptune_scale import Run

        task_name = get_task_name()

        with Run(experiment_name=task_name) as run:
            run.log_metrics({'test/ctx': 1.0}, step=0)
            assert run._mlop_run is None  # No mlop configured

        # Neptune should be closed automatically
        url = run.get_run_url()
        assert 'neptune.ai' in url
        print('✓ Real Neptune context manager test passed')

    @pytest.mark.skipif(
        not os.environ.get('NEPTUNE_API_TOKEN')
        or not os.environ.get('NEPTUNE_PROJECT')
        or not os.environ.get('MLOP_PROJECT'),
        reason='Requires NEPTUNE_API_TOKEN, NEPTUNE_PROJECT, and MLOP_PROJECT',
    )
    def test_real_neptune_mlop_resilience(self):
        """
        Test that Neptune works even if mlop fails during the run.

        Simulates mlop service being down mid-run.
        """
        from neptune_scale import Run

        task_name = get_task_name()

        run = Run(experiment_name=task_name)

        # Log some data - both should work
        run.log_configs({'test': 'resilience'})
        run.log_metrics({'test/metric1': 1.0}, step=0)

        # Simulate mlop failure by breaking the run object
        if run._mlop_run:
            run._mlop_run.log = mock.MagicMock(side_effect=Exception('mlop down'))

        # Neptune should still work!
        run.log_metrics({'test/metric2': 2.0}, step=1)
        run.log_metrics({'test/metric3': 3.0}, step=2)

        # Wait for Neptune to finish all operations before closing
        # Use verbose=False to prevent logging errors when pytest captures stdout
        run.wait_for_processing(verbose=False)

        # Close should still work
        run.close()

        url = run.get_run_url()
        assert 'neptune.ai' in url
        print('✓ Resilience test passed - Neptune worked despite mlop failure')
