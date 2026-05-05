"""
Unit tests for pluto._wandb_hook helpers.

These run in-process (no subprocess) so the lines actually get instrumented
by pytest-cov. The integration tests in test_wandb_pth_integration.py
validate the wired-up startup path; these cover the credential resolution,
discoverability hint, and finder/loader plumbing in isolation.
"""

import logging
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from pluto import _wandb_hook


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Reset hook state, finder list, and remove any cached wandb in sys.modules."""
    _wandb_hook.uninstall()
    _wandb_hook._hint_emitted = False
    monkeypatch.delitem(sys.modules, 'wandb', raising=False)
    yield
    _wandb_hook.uninstall()
    _wandb_hook._hint_emitted = False


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every Pluto/Wandb env var so tests start from a known baseline."""
    for k in list(os.environ):
        if k.startswith(('PLUTO_', 'WANDB_')) or k == 'DISABLE_WANDB_LOGGING':
            monkeypatch.delenv(k, raising=False)
    return monkeypatch


class TestKeyringCfgPath:
    def test_linux_uses_xdg_data_home(self, clean_env):
        clean_env.setattr(sys, 'platform', 'linux')
        clean_env.setenv('XDG_DATA_HOME', '/custom/xdg')
        path = _wandb_hook._keyring_cfg_path()
        assert path == '/custom/xdg/python_keyring/keyring_pass.cfg'

    def test_linux_falls_back_to_local_share(self, clean_env):
        clean_env.setattr(sys, 'platform', 'linux')
        clean_env.delenv('XDG_DATA_HOME', raising=False)
        path = _wandb_hook._keyring_cfg_path()
        assert path.endswith('/.local/share/python_keyring/keyring_pass.cfg')

    def test_windows_uses_localappdata(self, clean_env):
        # os.path.join uses the host's separator; assert components instead of literal.
        clean_env.setattr(sys, 'platform', 'win32')
        clean_env.setenv('LOCALAPPDATA', 'C:\\Users\\me\\AppData\\Local')
        path = _wandb_hook._keyring_cfg_path()
        assert path.startswith('C:\\Users\\me\\AppData\\Local')
        assert 'Python Keyring' in path
        assert path.endswith('keyring_pass.cfg')

    def test_windows_falls_back_to_programdata(self, clean_env):
        clean_env.setattr(sys, 'platform', 'win32')
        clean_env.delenv('LOCALAPPDATA', raising=False)
        clean_env.setenv('ProgramData', 'C:\\ProgramData')
        path = _wandb_hook._keyring_cfg_path()
        assert path.startswith('C:\\ProgramData')
        assert 'Python Keyring' in path
        assert path.endswith('keyring_pass.cfg')


class TestKeyringCfgHasPluto:
    def test_returns_false_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        assert _wandb_hook._keyring_cfg_has_pluto() is False

    def test_returns_true_when_pluto_section_present(self, tmp_path, monkeypatch):
        cfg = tmp_path / 'keyring_pass.cfg'
        cfg.write_text('[pluto]\napi_key = mlpi_xyz\n')
        monkeypatch.setattr(_wandb_hook, '_keyring_cfg_path', lambda: str(cfg))
        assert _wandb_hook._keyring_cfg_has_pluto() is True

    def test_returns_false_when_pluto_section_absent(self, tmp_path, monkeypatch):
        cfg = tmp_path / 'keyring_pass.cfg'
        cfg.write_text('[other]\nfoo = bar\n')
        monkeypatch.setattr(_wandb_hook, '_keyring_cfg_path', lambda: str(cfg))
        assert _wandb_hook._keyring_cfg_has_pluto() is False

    def test_returns_false_on_malformed_file(self, tmp_path, monkeypatch):
        cfg = tmp_path / 'keyring_pass.cfg'
        cfg.write_bytes(b'\xff\xfe\x00\x00garbage no section')
        monkeypatch.setattr(_wandb_hook, '_keyring_cfg_path', lambda: str(cfg))
        assert _wandb_hook._keyring_cfg_has_pluto() is False


class TestHasPlutoCredentials:
    def test_pluto_api_key_env_satisfies(self, clean_env, tmp_path):
        clean_env.setenv('PLUTO_API_KEY', 'mlpi_test')
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(tmp_path / 'absent'))
        clean_env.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        assert _wandb_hook._has_pluto_credentials() is True

    def test_login_marker_satisfies(self, clean_env, tmp_path):
        marker = tmp_path / '.login_ok'
        marker.touch()
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(marker))
        clean_env.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        assert _wandb_hook._has_pluto_credentials() is True

    def test_keyring_with_pluto_section_satisfies(self, clean_env, tmp_path):
        cfg = tmp_path / 'keyring_pass.cfg'
        cfg.write_text('[pluto]\napi_key = mlpi_xyz\n')
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(tmp_path / 'absent'))
        clean_env.setattr(_wandb_hook, '_keyring_cfg_path', lambda: str(cfg))
        assert _wandb_hook._has_pluto_credentials() is True

    def test_wandb_api_key_with_disable_flag_satisfies(self, clean_env, tmp_path):
        clean_env.setenv('WANDB_API_KEY', 'mlpi_via_wandb_var')
        clean_env.setenv('DISABLE_WANDB_LOGGING', 'true')
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(tmp_path / 'absent'))
        clean_env.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        assert _wandb_hook._has_pluto_credentials() is True

    def test_wandb_api_key_without_disable_flag_does_not_satisfy(
        self, clean_env, tmp_path
    ):
        clean_env.setenv('WANDB_API_KEY', 'wandb_only')
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(tmp_path / 'absent'))
        clean_env.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        assert _wandb_hook._has_pluto_credentials() is False

    def test_no_signals_means_no_credentials(self, clean_env, tmp_path):
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(tmp_path / 'absent'))
        clean_env.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        assert _wandb_hook._has_pluto_credentials() is False


class TestHasPartialPlutoSignal:
    @pytest.mark.parametrize(
        'var', ['PLUTO_PROJECT', 'PLUTO_URL_APP', 'PLUTO_URL_API', 'PLUTO_URL_INGEST']
    )
    def test_each_partial_var_triggers(self, clean_env, var):
        clean_env.setenv(var, 'value')
        assert _wandb_hook._has_partial_pluto_signal() is True

    def test_no_pluto_vars_means_no_signal(self, clean_env):
        assert _wandb_hook._has_partial_pluto_signal() is False


class TestEmitDiscoverabilityHint:
    def test_partial_config_message_when_pluto_var_set(self, clean_env, caplog):
        clean_env.setenv('PLUTO_PROJECT', 'p')
        with caplog.at_level(logging.WARNING, logger='pluto._wandb_hook'):
            _wandb_hook._emit_discoverability_hint()
        assert any(
            'Pluto config detected but no API key' in r.message for r in caplog.records
        )

    def test_generic_hint_when_no_pluto_vars(self, clean_env, caplog):
        with caplog.at_level(logging.WARNING, logger='pluto._wandb_hook'):
            _wandb_hook._emit_discoverability_hint()
        assert any('no Pluto credentials found' in r.message for r in caplog.records)

    def test_only_emits_once(self, clean_env, caplog):
        with caplog.at_level(logging.WARNING, logger='pluto._wandb_hook'):
            _wandb_hook._emit_discoverability_hint()
            _wandb_hook._emit_discoverability_hint()
            _wandb_hook._emit_discoverability_hint()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1


class TestPatchOrHint:
    def test_applies_patches_when_credentials_present(self, clean_env, tmp_path):
        clean_env.setenv('PLUTO_API_KEY', 'mlpi_test')
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(tmp_path / 'absent'))
        clean_env.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        wandb_mod = ModuleType('wandb')
        with mock.patch('pluto.compat.wandb.apply_wandb_patches') as apply:
            _wandb_hook._patch_or_hint(wandb_mod)
        apply.assert_called_once_with(wandb_mod)

    def test_emits_hint_when_no_credentials(self, clean_env, tmp_path, caplog):
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(tmp_path / 'absent'))
        clean_env.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        wandb_mod = ModuleType('wandb')
        with caplog.at_level(logging.WARNING, logger='pluto._wandb_hook'):
            _wandb_hook._patch_or_hint(wandb_mod)
        assert any('no Pluto credentials' in r.message for r in caplog.records)

    def test_logs_warning_when_patches_raise(self, clean_env, tmp_path, caplog):
        clean_env.setenv('PLUTO_API_KEY', 'mlpi_test')
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(tmp_path / 'absent'))
        clean_env.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        wandb_mod = ModuleType('wandb')
        with (
            mock.patch(
                'pluto.compat.wandb.apply_wandb_patches',
                side_effect=RuntimeError('boom'),
            ),
            caplog.at_level(logging.WARNING, logger='pluto._wandb_hook'),
        ):
            _wandb_hook._patch_or_hint(wandb_mod)
        assert any('Failed to apply wandb patches' in r.message for r in caplog.records)


class TestPatchingLoader:
    def test_exec_module_runs_real_loader_then_patches(self, clean_env, tmp_path):
        clean_env.setenv('PLUTO_API_KEY', 'mlpi_test')
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(tmp_path / 'absent'))
        clean_env.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        real_loader = mock.MagicMock()
        loader = _wandb_hook._PatchingLoader(real_loader)
        wandb_mod = ModuleType('wandb')
        with mock.patch('pluto.compat.wandb.apply_wandb_patches') as apply:
            loader.exec_module(wandb_mod)
        real_loader.exec_module.assert_called_once_with(wandb_mod)
        apply.assert_called_once_with(wandb_mod)

    def test_create_module_delegates_when_supported(self):
        real_loader = mock.MagicMock()
        sentinel = object()
        real_loader.create_module.return_value = sentinel
        loader = _wandb_hook._PatchingLoader(real_loader)
        spec = SimpleNamespace(name='wandb')
        assert loader.create_module(spec) is sentinel
        real_loader.create_module.assert_called_once_with(spec)

    def test_create_module_returns_none_when_real_has_no_create(self):
        class _LoaderWithoutCreate:
            def exec_module(self, module):
                pass

        loader = _wandb_hook._PatchingLoader(_LoaderWithoutCreate())
        assert loader.create_module(SimpleNamespace(name='wandb')) is None


class TestFinder:
    def test_returns_none_for_other_modules(self):
        finder = _wandb_hook._PlutoWandbFinder()
        assert finder.find_spec('numpy') is None

    def test_returns_none_when_re_entered(self):
        finder = _wandb_hook._PlutoWandbFinder()
        finder._patching = True
        assert finder.find_spec('wandb') is None

    def test_returns_none_when_real_wandb_not_findable(self):
        finder = _wandb_hook._PlutoWandbFinder()
        with mock.patch('importlib.util.find_spec', return_value=None):
            assert finder.find_spec('wandb') is None

    def test_returns_none_when_real_spec_has_no_loader(self):
        finder = _wandb_hook._PlutoWandbFinder()
        spec = SimpleNamespace(loader=None)
        with mock.patch('importlib.util.find_spec', return_value=spec):
            assert finder.find_spec('wandb') is None

    def test_wraps_real_loader_with_patching_loader(self):
        finder = _wandb_hook._PlutoWandbFinder()
        real_loader = mock.MagicMock()
        spec = SimpleNamespace(loader=real_loader)
        with mock.patch('importlib.util.find_spec', return_value=spec):
            wrapped = finder.find_spec('wandb')
        assert wrapped is spec
        assert isinstance(spec.loader, _wandb_hook._PatchingLoader)
        assert spec.loader._real_loader is real_loader


class TestInstall:
    def test_registers_finder_on_meta_path(self):
        _wandb_hook.install()
        assert any(isinstance(f, _wandb_hook._PlutoWandbFinder) for f in sys.meta_path)

    def test_idempotent(self):
        _wandb_hook.install()
        _wandb_hook.install()
        _wandb_hook.install()
        finders = [
            f for f in sys.meta_path if isinstance(f, _wandb_hook._PlutoWandbFinder)
        ]
        assert len(finders) == 1

    def test_already_imported_with_creds_patches_in_place(
        self, clean_env, tmp_path, monkeypatch
    ):
        clean_env.setenv('PLUTO_API_KEY', 'mlpi_test')
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(tmp_path / 'absent'))
        clean_env.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        fake = ModuleType('wandb')
        monkeypatch.setitem(sys.modules, 'wandb', fake)
        with mock.patch('pluto.compat.wandb.apply_wandb_patches') as apply:
            _wandb_hook.install()
        apply.assert_called_once_with(fake)
        # Finder should NOT be added when wandb is already imported.
        assert not any(
            isinstance(f, _wandb_hook._PlutoWandbFinder) for f in sys.meta_path
        )

    def test_already_imported_without_creds_emits_hint(
        self, clean_env, tmp_path, monkeypatch, caplog
    ):
        clean_env.setattr(_wandb_hook, '_LOGIN_MARKER_PATH', str(tmp_path / 'absent'))
        clean_env.setattr(
            _wandb_hook, '_keyring_cfg_path', lambda: str(tmp_path / 'absent.cfg')
        )
        fake = ModuleType('wandb')
        monkeypatch.setitem(sys.modules, 'wandb', fake)
        with caplog.at_level(logging.WARNING, logger='pluto._wandb_hook'):
            _wandb_hook.install()
        assert any('no Pluto credentials' in r.message for r in caplog.records)


class TestUninstall:
    def test_removes_finder_and_resets_state(self):
        _wandb_hook.install()
        assert any(isinstance(f, _wandb_hook._PlutoWandbFinder) for f in sys.meta_path)
        _wandb_hook.uninstall()
        assert not any(
            isinstance(f, _wandb_hook._PlutoWandbFinder) for f in sys.meta_path
        )
        assert _wandb_hook._hook_installed is False
        assert _wandb_hook._hint_emitted is False
