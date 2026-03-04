"""Tests for the standalone neptune_scale shim (pluto.compat.neptune_scale).

These tests verify that:
1. The shim types (File, Histogram) have the same API as neptune_scale.types
2. The shim Run class works without the real neptune-scale installed
3. The sys.modules registration works for transparent imports
4. The shim Run logs data to Pluto correctly
"""

import os
import sys
from unittest import mock

# ---------------------------------------------------------------------------
# Types tests (no external deps needed)
# ---------------------------------------------------------------------------


class TestFile:
    def test_basic_construction(self):
        from pluto.compat.neptune_scale.types import File

        f = File(source='image.png')
        assert f.source == 'image.png'
        assert f.mime_type is None

    def test_with_mime_type(self):
        from pluto.compat.neptune_scale.types import File

        f = File(source=b'\x89PNG', mime_type='image/png')
        assert f.source == b'\x89PNG'
        assert f.mime_type == 'image/png'

    def test_repr(self):
        from pluto.compat.neptune_scale.types import File

        f = File(source='test.png', mime_type='image/png')
        r = repr(f)
        assert 'File' in r
        assert 'test.png' in r


class TestHistogram:
    def test_basic_construction(self):
        from pluto.compat.neptune_scale.types import Histogram

        h = Histogram(bin_edges=[0, 1, 2, 3], counts=[10, 20, 30])
        assert h.bin_edges == [0, 1, 2, 3]
        assert h.counts == [10, 20, 30]
        assert h.densities is None

    def test_with_densities(self):
        from pluto.compat.neptune_scale.types import Histogram

        h = Histogram(bin_edges=[0, 1, 2], densities=[0.5, 0.5])
        assert h.densities == [0.5, 0.5]
        assert h.counts is None

    def test_as_list_methods(self):
        from pluto.compat.neptune_scale.types import Histogram

        h = Histogram(bin_edges=[0, 1, 2], counts=[5, 10])
        assert h.bin_edges_as_list() == [0, 1, 2]
        assert h.counts_as_list() == [5, 10]
        assert h.densities_as_list() == []

    def test_repr(self):
        from pluto.compat.neptune_scale.types import Histogram

        h = Histogram(bin_edges=[0, 1], counts=[5])
        r = repr(h)
        assert 'Histogram' in r


# ---------------------------------------------------------------------------
# Run class tests (mock Pluto internals)
# ---------------------------------------------------------------------------


class TestRunNoPlutoProject:
    """Run should be a safe no-op when PLUTO_PROJECT is not set."""

    def test_noop_without_project(self):
        from pluto.compat.neptune_scale.run import Run

        env = {k: v for k, v in os.environ.items() if not k.startswith('PLUTO_')}
        with mock.patch.dict(os.environ, env, clear=True):
            run = Run(experiment_name='test')
            # Should not raise
            run.log_metrics({'loss': 0.5}, step=0)
            run.log_configs({'lr': 0.001})
            run.add_tags(['test'])
            run.remove_tags(['test'])
            run.close()


class TestRunMockedPluto:
    """Run methods with mocked pluto.init to verify correct forwarding."""

    def _make_run(self, mock_pluto_init):
        """Create a Run with mocked pluto and PLUTO_PROJECT set."""
        from pluto.compat.neptune_scale.run import Run

        env = {
            'PLUTO_PROJECT': 'test-project',
            'PLUTO_API_KEY': 'test-key',
        }
        with mock.patch.dict(os.environ, env):
            with mock.patch('pluto.compat.neptune_scale.run._safe_import_pluto') as m:
                mock_pluto = mock.MagicMock()
                mock_pluto.init.return_value = mock_pluto_init
                m.return_value = mock_pluto
                run = Run(experiment_name='my-exp')
                return run, mock_pluto, mock_pluto_init

    def test_log_metrics(self):
        mock_pluto_run = mock.MagicMock()
        run, _, _ = self._make_run(mock_pluto_run)

        run.log_metrics({'loss': 0.5, 'acc': 0.9}, step=42)
        mock_pluto_run.log.assert_called_once_with(
            {'loss': 0.5, 'acc': 0.9}, step=42
        )

    def test_log_configs(self):
        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}
        run, _, _ = self._make_run(mock_pluto_run)

        run.log_configs({'lr': 0.001, 'epochs': 10})
        assert mock_pluto_run.config == {'lr': 0.001, 'epochs': 10}

    def test_add_tags(self):
        mock_pluto_run = mock.MagicMock()
        run, _, _ = self._make_run(mock_pluto_run)

        run.add_tags(['v1', 'baseline'])
        mock_pluto_run.add_tags.assert_called_once_with(['v1', 'baseline'])

    def test_remove_tags(self):
        mock_pluto_run = mock.MagicMock()
        run, _, _ = self._make_run(mock_pluto_run)

        run.remove_tags(['old'])
        mock_pluto_run.remove_tags.assert_called_once_with(['old'])

    def test_close(self):
        mock_pluto_run = mock.MagicMock()
        run, _, _ = self._make_run(mock_pluto_run)

        run.close()
        mock_pluto_run.finish.assert_called_once()

    def test_context_manager(self):
        mock_pluto_run = mock.MagicMock()
        run, _, _ = self._make_run(mock_pluto_run)

        with run:
            run.log_metrics({'x': 1.0}, step=0)

        mock_pluto_run.finish.assert_called_once()

    def test_double_close_is_safe(self):
        mock_pluto_run = mock.MagicMock()
        run, _, _ = self._make_run(mock_pluto_run)

        run.close()
        run.close()  # Should not raise
        # finish only called once (second close is a no-op)
        mock_pluto_run.finish.assert_called_once()

    def test_assign_files(self):
        from pluto.compat.neptune_scale.types import File

        mock_pluto_run = mock.MagicMock()
        run, mock_pluto, _ = self._make_run(mock_pluto_run)
        mock_pluto.Image.return_value = 'pluto_image'

        run.assign_files({'img': File(source='photo.png')})
        mock_pluto_run.log.assert_called_once()

    def test_log_histograms(self):
        from pluto.compat.neptune_scale.types import Histogram

        mock_pluto_run = mock.MagicMock()
        run, mock_pluto, _ = self._make_run(mock_pluto_run)
        mock_pluto.Histogram.return_value = 'pluto_hist'

        run.log_histograms(
            {'dist': Histogram(bin_edges=[0, 1, 2], counts=[5, 10])},
            step=1,
        )
        mock_pluto_run.log.assert_called_once()

    def test_get_run_url_with_pluto(self):
        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.url = 'https://pluto.trainy.ai/run/123'
        run, _, _ = self._make_run(mock_pluto_run)

        assert run.get_run_url() == 'https://pluto.trainy.ai/run/123'

    def test_wait_methods_are_noop(self):
        mock_pluto_run = mock.MagicMock()
        run, _, _ = self._make_run(mock_pluto_run)

        # Should not raise
        run.wait_for_submission()
        run.wait_for_processing()


# ---------------------------------------------------------------------------
# sys.modules shim registration tests
# ---------------------------------------------------------------------------


class TestShimRegistration:
    """Test that the compat layer registers the shim in sys.modules."""

    def test_shim_registered_when_neptune_missing(self):
        """When neptune-scale is missing, importing pluto.compat.neptune
        should register the shim in sys.modules['neptune_scale']."""
        import pluto.compat.neptune as compat_mod

        # Save state
        saved_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith('neptune_scale'):
                saved_modules[key] = sys.modules.pop(key)

        saved_original = compat_mod._original_neptune_run
        saved_applied = compat_mod._patch_applied

        try:
            # Reset state
            compat_mod._original_neptune_run = None
            compat_mod._patch_applied = False

            # Make neptune_scale import fail
            with mock.patch.dict(sys.modules, {'neptune_scale': None}):
                # Remove the None sentinel so our code can register the shim
                sys.modules.pop('neptune_scale', None)
                sys.modules.pop('neptune_scale.types', None)

                # Block the real neptune_scale from being found
                real_import = __import__

                def fake_import(name, *args, **kwargs):
                    if name == 'neptune_scale':
                        raise ImportError('fake')
                    return real_import(name, *args, **kwargs)

                with mock.patch(
                    'builtins.__import__',
                    side_effect=fake_import,
                ):
                    compat_mod._apply_monkeypatch()

                    assert 'neptune_scale' in sys.modules
                    ns = sys.modules['neptune_scale']
                    assert ns.__name__ == (
                        'pluto.compat.neptune_scale'
                    )
        finally:
            # Restore state
            compat_mod._original_neptune_run = saved_original
            compat_mod._patch_applied = saved_applied
            for key in list(sys.modules.keys()):
                if key.startswith('neptune_scale'):
                    sys.modules.pop(key, None)
            sys.modules.update(saved_modules)


# ---------------------------------------------------------------------------
# File conversion tests
# ---------------------------------------------------------------------------


class TestFileConversion:
    def test_image_by_extension(self):
        from pluto.compat.neptune_scale.run import _convert_file_to_pluto
        from pluto.compat.neptune_scale.types import File

        mock_pluto = mock.MagicMock()
        mock_pluto.Image.return_value = 'image_result'

        result = _convert_file_to_pluto(File(source='photo.jpg'), mock_pluto)
        mock_pluto.Image.assert_called_once_with('photo.jpg')
        assert result == 'image_result'

    def test_audio_by_extension(self):
        from pluto.compat.neptune_scale.run import _convert_file_to_pluto
        from pluto.compat.neptune_scale.types import File

        mock_pluto = mock.MagicMock()
        mock_pluto.Audio.return_value = 'audio_result'

        result = _convert_file_to_pluto(File(source='clip.wav'), mock_pluto)
        mock_pluto.Audio.assert_called_once_with('clip.wav')
        assert result == 'audio_result'

    def test_video_by_extension(self):
        from pluto.compat.neptune_scale.run import _convert_file_to_pluto
        from pluto.compat.neptune_scale.types import File

        mock_pluto = mock.MagicMock()
        mock_pluto.Video.return_value = 'video_result'

        result = _convert_file_to_pluto(File(source='video.mp4'), mock_pluto)
        mock_pluto.Video.assert_called_once_with('video.mp4')
        assert result == 'video_result'

    def test_image_by_mime_type(self):
        from pluto.compat.neptune_scale.run import _convert_file_to_pluto
        from pluto.compat.neptune_scale.types import File

        mock_pluto = mock.MagicMock()
        mock_pluto.Image.return_value = 'image_result'

        _convert_file_to_pluto(
            File(source=b'data', mime_type='image/png'), mock_pluto
        )
        mock_pluto.Image.assert_called_once_with(b'data')

    def test_fallback_to_artifact(self):
        from pluto.compat.neptune_scale.run import _convert_file_to_pluto
        from pluto.compat.neptune_scale.types import File

        mock_pluto = mock.MagicMock()
        mock_pluto.Artifact.return_value = 'artifact_result'

        result = _convert_file_to_pluto(
            File(source='data.bin'), mock_pluto
        )
        mock_pluto.Artifact.assert_called_once_with('data.bin')
        assert result == 'artifact_result'
