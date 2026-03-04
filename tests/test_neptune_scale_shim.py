"""Tests for the standalone neptune_scale shim (pluto.compat.neptune_scale).

These tests verify that:
1. The shim types (File, Histogram) have the same API as neptune_scale.types
2. The shim Run delegates to NeptuneRunWrapper with Neptune disabled
3. The sys.modules registration works for transparent imports
"""

import os
import sys
from unittest import mock

# ---------------------------------------------------------------------------
# Types tests
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
        assert 'Histogram' in repr(h)


# ---------------------------------------------------------------------------
# Run class tests — verifies it delegates to NeptuneRunWrapper
# ---------------------------------------------------------------------------


class TestRunReturnsWrapper:
    """Run() should produce a NeptuneRunWrapper instance."""

    def test_returns_wrapper_instance(self):
        from pluto.compat.neptune import NeptuneRunWrapper
        from pluto.compat.neptune_scale.run import Run

        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith('PLUTO_')
        }
        with mock.patch.dict(os.environ, env, clear=True):
            run = Run(experiment_name='test')
            assert isinstance(run, NeptuneRunWrapper)
            # Neptune should be disabled
            assert run._neptune_disabled is True
            assert run._neptune_run is None
            run.close()

    def test_env_var_restored_after_init(self):
        """DISABLE_NEPTUNE_LOGGING should not leak."""
        from pluto.compat.neptune_scale.run import Run

        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(('PLUTO_', 'DISABLE_'))
        }
        with mock.patch.dict(os.environ, env, clear=True):
            Run(experiment_name='test').close()
            assert 'DISABLE_NEPTUNE_LOGGING' not in os.environ


class TestRunMockedPluto:
    """Run methods with mocked pluto.init to verify correct forwarding."""

    def _make_run(self, mock_pluto_run):
        from pluto.compat.neptune_scale.run import Run

        env = {
            'PLUTO_PROJECT': 'test-project',
            'PLUTO_API_KEY': 'test-key',
        }
        with mock.patch.dict(os.environ, env):
            with mock.patch(
                'pluto.compat.neptune._safe_import_pluto'
            ) as m:
                mock_pluto = mock.MagicMock()
                mock_pluto.init.return_value = mock_pluto_run
                m.return_value = mock_pluto
                run = Run(experiment_name='my-exp')
                return run, mock_pluto

    def test_log_metrics(self):
        mock_pluto_run = mock.MagicMock()
        run, _ = self._make_run(mock_pluto_run)

        run.log_metrics({'loss': 0.5}, step=42)
        mock_pluto_run.log.assert_called_once_with(
            {'loss': 0.5}, step=42
        )

    def test_log_configs(self):
        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}
        run, _ = self._make_run(mock_pluto_run)

        run.log_configs({'lr': 0.001})
        assert mock_pluto_run.config == {'lr': 0.001}

    def test_add_tags(self):
        mock_pluto_run = mock.MagicMock()
        run, _ = self._make_run(mock_pluto_run)

        run.add_tags(['v1', 'baseline'])
        mock_pluto_run.add_tags.assert_called_once_with(
            ['v1', 'baseline']
        )

    def test_remove_tags(self):
        mock_pluto_run = mock.MagicMock()
        run, _ = self._make_run(mock_pluto_run)

        run.remove_tags(['old'])
        mock_pluto_run.remove_tags.assert_called_once_with(['old'])

    def test_close(self):
        mock_pluto_run = mock.MagicMock()
        run, _ = self._make_run(mock_pluto_run)

        run.close()
        mock_pluto_run.finish.assert_called_once()

    def test_context_manager(self):
        mock_pluto_run = mock.MagicMock()
        run, _ = self._make_run(mock_pluto_run)

        with run:
            run.log_metrics({'x': 1.0}, step=0)

        mock_pluto_run.finish.assert_called_once()

    def test_double_close_is_safe(self):
        mock_pluto_run = mock.MagicMock()
        run, _ = self._make_run(mock_pluto_run)

        run.close()
        run.close()
        mock_pluto_run.finish.assert_called_once()

    def test_wait_methods_are_noop(self):
        mock_pluto_run = mock.MagicMock()
        run, _ = self._make_run(mock_pluto_run)

        run.wait_for_submission()
        run.wait_for_processing()


class TestRunNoPlutoProject:
    """Run should be a safe no-op when PLUTO_PROJECT is not set."""

    def test_noop_without_project(self):
        from pluto.compat.neptune_scale.run import Run

        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith('PLUTO_')
        }
        with mock.patch.dict(os.environ, env, clear=True):
            run = Run(experiment_name='test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.log_configs({'lr': 0.001})
            run.add_tags(['test'])
            run.remove_tags(['test'])
            run.close()


# ---------------------------------------------------------------------------
# sys.modules shim registration
# ---------------------------------------------------------------------------


class TestShimRegistration:
    """Test that the compat layer registers the shim in sys.modules."""

    def test_shim_registered_when_neptune_missing(self):
        """When neptune-scale is missing, importing pluto.compat.neptune
        should register the shim in sys.modules['neptune_scale']."""
        import pluto.compat.neptune as compat_mod

        saved_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith('neptune_scale'):
                saved_modules[key] = sys.modules.pop(key)

        saved_original = compat_mod._original_neptune_run
        saved_applied = compat_mod._patch_applied

        try:
            compat_mod._original_neptune_run = None
            compat_mod._patch_applied = False

            with mock.patch.dict(
                sys.modules, {'neptune_scale': None}
            ):
                sys.modules.pop('neptune_scale', None)
                sys.modules.pop('neptune_scale.types', None)

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
            compat_mod._original_neptune_run = saved_original
            compat_mod._patch_applied = saved_applied
            for key in list(sys.modules.keys()):
                if key.startswith('neptune_scale'):
                    sys.modules.pop(key, None)
            sys.modules.update(saved_modules)
