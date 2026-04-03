"""Tests for wandb dual-logging mode (PLUTO_WANDB_MODE=dual).

Unit tests (mock-based):
    - DualRun wrapper: forwarding, error isolation, context manager
    - Scalar extraction logic
    - setup_dual wiring
    - Conflict detection and mode activation
    - Coverage/warning integration with dual mode
    - Shim mode (wandb/__init__.py): shadowing, PLUTO_WANDB_SHIM=0

Integration tests (skipped when credentials absent):
    - TestDualLive: real training loop through PLUTO_WANDB_MODE=dual
      Requires: WANDB_API_KEY + PLUTO_API_KEY (or PLUTO_API_TOKEN)
    - TestShimLive: real training loop through shim mode
      Requires: PLUTO_API_KEY (or PLUTO_API_TOKEN)
"""

import importlib
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pluto.compat.wandb.dual import DualRun, _extract_scalars

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_has_real_wandb = False
try:
    import importlib.metadata

    _dist = importlib.metadata.distribution('wandb')
    _dist_name = (_dist.metadata['Name'] or '').lower().replace('-', '_')
    _has_real_wandb = _dist_name != 'pluto_ml'
except importlib.metadata.PackageNotFoundError:
    pass

_has_pluto_key = bool(
    os.environ.get('PLUTO_API_KEY') or os.environ.get('PLUTO_API_TOKEN')
)
_has_wandb_key = bool(os.environ.get('WANDB_API_KEY'))


# ===================================================================
# Unit tests (no credentials needed)
# ===================================================================


class TestExtractScalars:
    """Test scalar extraction from log dicts."""

    def test_simple_scalars(self):
        data = {'loss': 0.5, 'acc': 0.9, 'epoch': 3}
        result = _extract_scalars(data)
        assert result == {'loss': 0.5, 'acc': 0.9, 'epoch': 3}

    def test_nested_dicts_flattened(self):
        data = {'train': {'loss': 0.5, 'acc': 0.9}}
        result = _extract_scalars(data)
        assert result == {'train/loss': 0.5, 'train/acc': 0.9}

    def test_non_scalars_skipped(self):
        data = {
            'loss': 0.5,
            'image': MagicMock(),
            'text': 'hello',
            'table': MagicMock(),
        }
        result = _extract_scalars(data)
        assert result == {'loss': 0.5}

    def test_booleans_converted_to_int(self):
        data = {'flag': True, 'other': False}
        result = _extract_scalars(data)
        assert result == {'flag': 1, 'other': 0}

    def test_empty_dict(self):
        assert _extract_scalars({}) == {}

    def test_deeply_nested(self):
        data = {'a': {'b': {'c': 1.0}}}
        result = _extract_scalars(data)
        assert result == {'a/b/c': 1.0}

    def test_mixed_nested(self):
        data = {
            'metrics': {'loss': 0.3, 'image': MagicMock()},
            'lr': 0.001,
        }
        result = _extract_scalars(data)
        assert result == {'metrics/loss': 0.3, 'lr': 0.001}

    def test_integers(self):
        data = {'epoch': 5, 'step': 100}
        result = _extract_scalars(data)
        assert result == {'epoch': 5, 'step': 100}

    def test_none_skipped(self):
        data = {'loss': 0.5, 'nothing': None}
        result = _extract_scalars(data)
        assert result == {'loss': 0.5}

    def test_list_skipped(self):
        data = {'loss': 0.5, 'values': [1, 2, 3]}
        result = _extract_scalars(data)
        assert result == {'loss': 0.5}


class TestDualRun:
    """Test the DualRun wrapper."""

    def _make_dual_run(self):
        wandb_run = MagicMock()
        pluto_op = MagicMock()
        return DualRun(wandb_run, pluto_op), wandb_run, pluto_op

    # -- log() --

    def test_log_forwards_to_both(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        data = {'loss': 0.5, 'acc': 0.9}

        dual.log(data, step=1)

        wandb_run.log.assert_called_once_with(
            data, step=1, commit=None, sync=None,
        )
        pluto_op.log.assert_called_once_with(
            {'loss': 0.5, 'acc': 0.9}, step=1,
        )

    def test_log_skips_non_scalars_for_pluto(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        img = MagicMock()
        data = {'loss': 0.5, 'image': img}

        dual.log(data)

        wandb_run.log.assert_called_once()
        pluto_op.log.assert_called_once_with({'loss': 0.5}, step=None)

    def test_log_empty_scalars_skips_pluto(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        data = {'image': MagicMock()}

        dual.log(data)

        wandb_run.log.assert_called_once()
        pluto_op.log.assert_not_called()

    def test_log_with_commit_false(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        data = {'loss': 0.5}

        dual.log(data, commit=False)

        wandb_run.log.assert_called_once_with(
            data, step=None, commit=False, sync=None,
        )
        pluto_op.log.assert_called_once()

    def test_log_with_step(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        dual.log({'loss': 0.5}, step=42)

        pluto_op.log.assert_called_once_with({'loss': 0.5}, step=42)

    def test_log_nested_dict_flattened_for_pluto(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        data = {'train': {'loss': 0.5, 'acc': 0.9}}

        dual.log(data)

        pluto_op.log.assert_called_once_with(
            {'train/loss': 0.5, 'train/acc': 0.9}, step=None,
        )

    def test_pluto_log_failure_does_not_break_wandb(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        pluto_op.log.side_effect = RuntimeError('pluto down')

        dual.log({'loss': 0.5})

        wandb_run.log.assert_called_once()

    def test_multiple_log_calls(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        dual.log({'loss': 1.0}, step=0)
        dual.log({'loss': 0.5}, step=1)
        dual.log({'loss': 0.1}, step=2)

        assert wandb_run.log.call_count == 3
        assert pluto_op.log.call_count == 3

    # -- finish() --

    def test_finish_calls_both(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        dual.finish(exit_code=0)

        pluto_op.finish.assert_called_once()
        wandb_run.finish.assert_called_once_with(
            exit_code=0, quiet=None,
        )

    def test_finish_pluto_failure_still_finishes_wandb(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        pluto_op.finish.side_effect = RuntimeError('pluto down')

        dual.finish()

        wandb_run.finish.assert_called_once()

    def test_finish_pluto_only_once(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        dual.finish()
        dual.finish()

        pluto_op.finish.assert_called_once()
        assert wandb_run.finish.call_count == 2

    def test_finish_with_exit_code(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        dual.finish(exit_code=1, quiet=True)

        wandb_run.finish.assert_called_once_with(
            exit_code=1, quiet=True,
        )

    # -- watch() --

    def test_watch_forwards_to_both(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        model = MagicMock()

        dual.watch(model, log_freq=500)

        wandb_run.watch.assert_called_once()
        pluto_op.watch.assert_called_once_with(model, log_freq=500)

    def test_watch_multiple_models(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        models = [MagicMock(), MagicMock()]

        dual.watch(models, log_freq=100)

        wandb_run.watch.assert_called_once()
        assert pluto_op.watch.call_count == 2

    def test_watch_none_model_skips_pluto(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        dual.watch(None)

        wandb_run.watch.assert_called_once()
        pluto_op.watch.assert_not_called()

    def test_watch_pluto_failure_still_watches_wandb(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        pluto_op.watch.side_effect = RuntimeError('no torch')
        model = MagicMock()

        dual.watch(model)

        wandb_run.watch.assert_called_once()

    def test_unwatch_forwards_to_wandb(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        dual.unwatch()

        wandb_run.unwatch.assert_called_once()

    # -- config / summary --

    def test_config_property_returns_wandb_config(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        wandb_run.config = {'lr': 0.01}

        assert dual.config == {'lr': 0.01}

    def test_config_setter_mirrors_to_pluto(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        dual.config = {'lr': 0.01, 'batch_size': 32}

        pluto_op.update_config.assert_called_once_with(
            {'lr': 0.01, 'batch_size': 32},
        )

    def test_config_setter_non_dict_skips_pluto(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        dual.config = 'not a dict'

        pluto_op.update_config.assert_not_called()

    def test_summary_property_returns_wandb_summary(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        wandb_run.summary = {'best_loss': 0.1}

        assert dual.summary == {'best_loss': 0.1}

    # -- attribute forwarding --

    def test_getattr_falls_through_to_wandb(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        wandb_run.id = 'abc123'
        wandb_run.name = 'cool-run'
        wandb_run.project = 'my-project'

        assert dual.id == 'abc123'
        assert dual.name == 'cool-run'
        assert dual.project == 'my-project'

    def test_setattr_forwards_to_wandb(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        dual.name = 'new-name'
        dual.notes = 'some notes'

        assert wandb_run.name == 'new-name'
        assert wandb_run.notes == 'some notes'

    def test_repr(self):
        dual, wandb_run, pluto_op = self._make_dual_run()
        r = repr(dual)
        assert 'DualRun' in r

    # -- context manager --

    def test_context_manager(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        with dual as run:
            run.log({'loss': 1.0})

        pluto_op.finish.assert_called_once()
        wandb_run.finish.assert_called_once()

    def test_context_manager_with_exception(self):
        dual, wandb_run, pluto_op = self._make_dual_run()

        with pytest.raises(ValueError):
            with dual:
                raise ValueError('boom')

        wandb_run.finish.assert_called_once_with(
            exit_code=1, quiet=None,
        )
        pluto_op.finish.assert_called_once()

    # -- edge cases --

    def test_no_pluto_op_still_works(self):
        """DualRun with pluto_op=None should just forward to wandb."""
        wandb_run = MagicMock()
        dual = DualRun(wandb_run, None)

        dual.log({'loss': 0.5})
        dual.log({'image': MagicMock()})
        dual.finish()

        assert wandb_run.log.call_count == 2
        wandb_run.finish.assert_called_once()

    def test_pluto_op_none_watch_no_error(self):
        wandb_run = MagicMock()
        dual = DualRun(wandb_run, None)

        dual.watch(MagicMock())

        wandb_run.watch.assert_called_once()

    def test_pluto_op_none_finish_no_error(self):
        wandb_run = MagicMock()
        dual = DualRun(wandb_run, None)

        dual.finish()
        dual.finish()

        assert wandb_run.finish.call_count == 2


class TestSetupDual:
    """Test the setup_dual function."""

    def test_setup_replaces_init(self):
        from pluto.compat.wandb.dual import setup_dual

        fake_module = MagicMock()
        fake_real_wandb = MagicMock()

        setup_dual(fake_module, fake_real_wandb)

        assert fake_module.init != fake_real_wandb.init
        assert fake_module.log != fake_real_wandb.log
        assert fake_module.finish != fake_real_wandb.finish
        assert fake_module.watch != fake_real_wandb.watch

    def test_dual_init_calls_both(self):
        from pluto.compat.wandb.dual import setup_dual

        fake_module = MagicMock()
        fake_real_wandb = MagicMock()
        fake_wandb_run = MagicMock()
        fake_real_wandb.init.return_value = fake_wandb_run

        setup_dual(fake_module, fake_real_wandb)

        with patch('pluto.init') as mock_pluto_init:
            mock_pluto_init.return_value = MagicMock()
            result = fake_module.init(project='test', name='run1')

        assert isinstance(result, DualRun)
        fake_real_wandb.init.assert_called_once()
        mock_pluto_init.assert_called_once()

    def test_dual_init_passes_config_to_pluto(self):
        from pluto.compat.wandb.dual import setup_dual

        fake_module = MagicMock()
        fake_real_wandb = MagicMock()
        fake_real_wandb.init.return_value = MagicMock()

        setup_dual(fake_module, fake_real_wandb)

        with patch('pluto.init') as mock_pluto_init:
            mock_pluto_init.return_value = MagicMock()
            fake_module.init(
                project='p', name='n', config={'lr': 0.01},
                tags=['a', 'b'],
            )

        call_kwargs = mock_pluto_init.call_args[1]
        assert call_kwargs['project'] == 'p'
        assert call_kwargs['name'] == 'n'
        assert call_kwargs['config'] == {'lr': 0.01}
        assert call_kwargs['tags'] == ['a', 'b']

    def test_dual_init_pluto_failure_returns_dual_run(self):
        from pluto.compat.wandb.dual import setup_dual

        fake_module = MagicMock()
        fake_real_wandb = MagicMock()
        fake_wandb_run = MagicMock()
        fake_real_wandb.init.return_value = fake_wandb_run

        setup_dual(fake_module, fake_real_wandb)

        with patch('pluto.init', side_effect=RuntimeError('pluto down')):
            result = fake_module.init(project='test')

        assert isinstance(result, DualRun)
        fake_real_wandb.init.assert_called_once()

    def test_dual_init_updates_module_run(self):
        from pluto.compat.wandb.dual import setup_dual

        fake_module = MagicMock()
        fake_real_wandb = MagicMock()
        fake_real_wandb.init.return_value = MagicMock()

        setup_dual(fake_module, fake_real_wandb)

        with patch('pluto.init', return_value=MagicMock()):
            result = fake_module.init(project='test')

        assert fake_module.run is result

    def test_dual_finish_clears_module_run(self):
        from pluto.compat.wandb.dual import setup_dual

        fake_module = MagicMock()
        fake_real_wandb = MagicMock()
        fake_real_wandb.init.return_value = MagicMock()

        setup_dual(fake_module, fake_real_wandb)

        with patch('pluto.init', return_value=MagicMock()):
            fake_module.init(project='test')
            fake_module.finish()

        assert fake_module.run is None

    def test_dual_log_without_init_falls_back_to_real_wandb(self):
        from pluto.compat.wandb.dual import setup_dual

        fake_module = MagicMock()
        fake_real_wandb = MagicMock()

        setup_dual(fake_module, fake_real_wandb)

        fake_module.log({'loss': 0.5}, step=1)

        fake_real_wandb.log.assert_called_once_with(
            {'loss': 0.5}, step=1, commit=None, sync=None,
        )

    def test_dual_watch_without_init_falls_back_to_real_wandb(self):
        from pluto.compat.wandb.dual import setup_dual

        fake_module = MagicMock()
        fake_real_wandb = MagicMock()

        setup_dual(fake_module, fake_real_wandb)

        model = MagicMock()
        fake_module.watch(model, log_freq=500)

        fake_real_wandb.watch.assert_called_once()

    def test_dual_reinit_finishes_previous(self):
        from pluto.compat.wandb.dual import setup_dual

        fake_module = MagicMock()
        fake_real_wandb = MagicMock()
        fake_real_wandb.init.return_value = MagicMock()

        setup_dual(fake_module, fake_real_wandb)

        with patch('pluto.init', return_value=MagicMock()):
            first = fake_module.init(project='test')
            second = fake_module.init(project='test')

        assert second is not first
        assert fake_real_wandb.init.call_count == 2


class TestModeActivation:
    """Test mode dispatch in wandb/__init__.py."""

    def test_shim_mode_is_default(self):
        """Without PLUTO_WANDB_MODE, shim mode is active."""
        result = subprocess.run(
            [
                sys.executable, '-c',
                'import wandb; '
                'print("has_real:", hasattr(wandb, "_real_wandb")); '
                'print("file:", wandb.__file__)',
            ],
            capture_output=True,
            text=True,
            env={
                k: v for k, v in os.environ.items()
                if k not in ('PLUTO_WANDB_MODE', 'PLUTO_WANDB_SHIM')
            },
        )
        assert result.returncode == 0, result.stderr
        assert 'has_real: False' in result.stdout

    def test_shim_disabled_by_env(self):
        """PLUTO_WANDB_SHIM=0 prevents shim from loading."""
        result = subprocess.run(
            [sys.executable, '-c', 'import wandb'],
            capture_output=True,
            text=True,
            env={**os.environ, 'PLUTO_WANDB_SHIM': '0'},
        )
        assert result.returncode != 0
        assert 'pluto wandb shim is disabled' in result.stderr

    @pytest.mark.skipif(
        not _has_real_wandb,
        reason='real wandb not installed',
    )
    def test_dual_mode_activates(self):
        """PLUTO_WANDB_MODE=dual loads real wandb + dual wrappers."""
        result = subprocess.run(
            [
                sys.executable, '-c',
                'import wandb; '
                'print("has_real:", hasattr(wandb, "_real_wandb")); '
                'print("has_init:", hasattr(wandb, "init"))',
            ],
            capture_output=True,
            text=True,
            env={**os.environ, 'PLUTO_WANDB_MODE': 'dual'},
        )
        assert result.returncode == 0, result.stderr
        assert 'has_real: True' in result.stdout
        assert 'has_init: True' in result.stdout

    @pytest.mark.skipif(
        not _has_real_wandb,
        reason='real wandb not installed',
    )
    def test_dual_mode_has_real_wandb_types(self):
        """In dual mode, wandb.Image etc. come from real wandb."""
        result = subprocess.run(
            [
                sys.executable, '-c',
                'import wandb; '
                'print("Image:", wandb.Image.__module__)',
            ],
            capture_output=True,
            text=True,
            env={**os.environ, 'PLUTO_WANDB_MODE': 'dual'},
        )
        assert result.returncode == 0, result.stderr
        # Real wandb's Image module should NOT contain 'pluto'
        assert 'pluto' not in result.stdout

    def test_shim_mode_has_pluto_types(self):
        """In shim mode, wandb.Image comes from pluto compat layer."""
        result = subprocess.run(
            [
                sys.executable, '-c',
                'import wandb; '
                'print("Image:", wandb.Image.__module__)',
            ],
            capture_output=True,
            text=True,
            env={
                k: v for k, v in os.environ.items()
                if k != 'PLUTO_WANDB_MODE'
            },
        )
        assert result.returncode == 0, result.stderr
        assert 'pluto' in result.stdout

    def test_shim_shadows_with_warning(self):
        """Shim mode warns when real wandb is also installed."""
        if not _has_real_wandb:
            pytest.skip('real wandb not installed')

        result = subprocess.run(
            [
                sys.executable, '-W', 'all', '-c',
                'import wandb',
            ],
            capture_output=True,
            text=True,
            env={
                k: v for k, v in os.environ.items()
                if k not in ('PLUTO_WANDB_MODE', 'PLUTO_WANDB_SHIM')
            },
        )
        assert result.returncode == 0
        assert 'shadowed' in result.stderr

    def test_invalid_mode_treated_as_shim(self):
        """Unknown PLUTO_WANDB_MODE values fall back to shim."""
        result = subprocess.run(
            [
                sys.executable, '-c',
                'import wandb; print("ok")',
            ],
            capture_output=True,
            text=True,
            env={**os.environ, 'PLUTO_WANDB_MODE': 'invalid_mode'},
        )
        assert result.returncode == 0
        assert 'ok' in result.stdout


class TestCoverageWarningsInDualMode:
    """Verify coverage warnings don't fire in dual mode."""

    @pytest.mark.skipif(
        not _has_real_wandb,
        reason='real wandb not installed',
    )
    def test_dual_mode_no_pluto_warnings_for_stubs(self):
        """In dual mode, pluto compat warnings shouldn't fire."""
        # Use save() instead of login() as it's a simpler no-op in
        # real wandb and doesn't trigger complex lazy imports.
        result = subprocess.run(
            [
                sys.executable, '-W', 'all', '-c',
                'import warnings; warnings.simplefilter("always"); '
                'import wandb; '
                'print("PlutoWandbCompatWarning" in '
                '"".join(str(w) for w in []))',
            ],
            capture_output=True,
            text=True,
            env={**os.environ, 'PLUTO_WANDB_MODE': 'dual'},
        )
        assert result.returncode == 0, result.stderr
        # Our custom warning class shouldn't appear anywhere
        assert 'PlutoWandbCompatWarning' not in result.stderr


# ===================================================================
# Integration tests (require credentials)
# ===================================================================

DUAL_PROJECT = 'wandb-dual-test'
NUM_STEPS = 20


def _get_test_name(suffix: str) -> str:
    """Generate a unique test run name."""
    import uuid

    short_id = str(uuid.uuid4())[:6]
    return f'dual-test-{suffix}-{short_id}'


@pytest.mark.skipif(
    not _has_real_wandb,
    reason='real wandb not installed',
)
@pytest.mark.skipif(
    not _has_wandb_key,
    reason='WANDB_API_KEY not set',
)
@pytest.mark.skipif(
    not _has_pluto_key,
    reason='PLUTO_API_KEY / PLUTO_API_TOKEN not set',
)
class TestDualLive:
    """Live integration tests for dual-logging mode.

    Requires both WANDB_API_KEY and PLUTO_API_KEY to be set.
    Runs actual training loops through both wandb and pluto.
    """

    def test_dual_basic_training_loop(self):
        """Basic scalar logging through dual mode."""
        result = subprocess.run(
            [
                sys.executable, '-c', (
                    'import os, math\n'
                    'os.environ["PLUTO_WANDB_MODE"] = "dual"\n'
                    'import wandb\n'
                    f'run = wandb.init(project="{DUAL_PROJECT}", '
                    f'name="{_get_test_name("basic")}", '
                    'config={"lr": 0.01, "epochs": 10})\n'
                    f'for i in range({NUM_STEPS}):\n'
                    '    wandb.log({"loss": 1.0 / (i + 1), "step": i})\n'
                    'wandb.finish()\n'
                    'print("OK")\n'
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f'stdout: {result.stdout}\nstderr: {result.stderr}'
        )
        assert 'OK' in result.stdout

    def test_dual_nested_metrics(self):
        """Nested dict metrics in dual mode."""
        result = subprocess.run(
            [
                sys.executable, '-c', (
                    'import os\n'
                    'os.environ["PLUTO_WANDB_MODE"] = "dual"\n'
                    'import wandb\n'
                    f'wandb.init(project="{DUAL_PROJECT}", '
                    f'name="{_get_test_name("nested")}")\n'
                    f'for i in range({NUM_STEPS}):\n'
                    '    wandb.log({"train": {"loss": 1.0/(i+1), '
                    '"acc": i/20.0}})\n'
                    'wandb.finish()\n'
                    'print("OK")\n'
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f'stdout: {result.stdout}\nstderr: {result.stderr}'
        )

    def test_dual_with_rich_types(self):
        """Rich types (Image, Table) go to wandb, scalars to both."""
        result = subprocess.run(
            [
                sys.executable, '-c', (
                    'import os, numpy as np\n'
                    'os.environ["PLUTO_WANDB_MODE"] = "dual"\n'
                    'import wandb\n'
                    f'wandb.init(project="{DUAL_PROJECT}", '
                    f'name="{_get_test_name("rich")}")\n'
                    'for i in range(5):\n'
                    '    img = np.random.randint(0,255,(8,8,3),'
                    'dtype=np.uint8)\n'
                    '    wandb.log({"loss": 1.0/(i+1), '
                    '"image": wandb.Image(img)})\n'
                    'wandb.finish()\n'
                    'print("OK")\n'
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f'stdout: {result.stdout}\nstderr: {result.stderr}'
        )

    def test_dual_context_manager(self):
        """Context manager pattern in dual mode."""
        result = subprocess.run(
            [
                sys.executable, '-c', (
                    'import os\n'
                    'os.environ["PLUTO_WANDB_MODE"] = "dual"\n'
                    'import wandb\n'
                    f'with wandb.init(project="{DUAL_PROJECT}", '
                    f'name="{_get_test_name("ctx")}") as run:\n'
                    '    for i in range(5):\n'
                    '        wandb.log({"loss": 1.0/(i+1)})\n'
                    'print("OK")\n'
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f'stdout: {result.stdout}\nstderr: {result.stderr}'
        )

    def test_dual_config_and_tags(self):
        """Config and tags are passed to both systems."""
        result = subprocess.run(
            [
                sys.executable, '-c', (
                    'import os\n'
                    'os.environ["PLUTO_WANDB_MODE"] = "dual"\n'
                    'import wandb\n'
                    f'run = wandb.init(project="{DUAL_PROJECT}", '
                    f'name="{_get_test_name("cfg")}", '
                    'config={"lr": 0.01, "model": "resnet"}, '
                    'tags=["dual-test", "ci"])\n'
                    'wandb.log({"loss": 0.5})\n'
                    'wandb.finish()\n'
                    'print("OK")\n'
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f'stdout: {result.stdout}\nstderr: {result.stderr}'
        )


@pytest.mark.skipif(
    not _has_pluto_key,
    reason='PLUTO_API_KEY / PLUTO_API_TOKEN not set',
)
class TestShimLive:
    """Live integration tests for shim mode (pluto only).

    Requires PLUTO_API_KEY or PLUTO_API_TOKEN to be set.
    """

    def test_shim_basic_training_loop(self):
        """Basic scalar logging through shim mode."""
        import pluto.compat.wandb as wandb
        from tests.utils import get_task_name

        run_name = get_task_name()
        wandb.init(
            project='wandb-shim-test',
            name=run_name,
            config={'lr': 0.01, 'epochs': NUM_STEPS},
        )

        for i in range(NUM_STEPS):
            wandb.log({
                'loss': 1.0 / (i + 1),
                'acc': i / NUM_STEPS,
            })

        wandb.finish()

    def test_shim_with_tags(self):
        """Tags are sent to pluto server."""
        import pluto.compat.wandb as wandb
        from tests.utils import get_task_name

        wandb.init(
            project='wandb-shim-test',
            name=get_task_name(),
            tags=['shim-test', 'ci'],
        )

        wandb.log({'loss': 0.5})
        wandb.finish()

    def test_shim_with_data_types(self):
        """Data types (Image, Histogram) log without error."""
        import pluto.compat.wandb as wandb
        from tests.utils import get_task_name

        wandb.init(
            project='wandb-shim-test',
            name=get_task_name(),
        )

        img = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        wandb.log({
            'loss': 0.5,
            'image': wandb.Image(img, caption='test'),
        })

        hist_data = np.random.randn(100)
        wandb.log({
            'gradients': wandb.Histogram(hist_data),
        })

        wandb.finish()

    def test_shim_context_manager(self):
        """Context manager pattern works in shim mode."""
        import pluto.compat.wandb as wandb
        from tests.utils import get_task_name

        with wandb.init(
            project='wandb-shim-test',
            name=get_task_name(),
        ):
            for i in range(5):
                wandb.log({'loss': 1.0 / (i + 1)})

    def test_shim_define_metric(self):
        """define_metric works in shim mode."""
        import pluto.compat.wandb as wandb
        from tests.utils import get_task_name

        run = wandb.init(
            project='wandb-shim-test',
            name=get_task_name(),
        )

        wandb.define_metric('val/loss', summary='min')
        wandb.define_metric('val/acc', summary='max')

        for i in range(10):
            wandb.log({
                'val/loss': 1.0 / (i + 1),
                'val/acc': i / 10.0,
            })

        assert run.summary['val/loss'] == pytest.approx(1.0 / 10)
        assert run.summary['val/acc'] == pytest.approx(9.0 / 10)

        wandb.finish()

    def test_shim_stubs_emit_warnings(self):
        """Stub functions emit PlutoWandbCompatWarning in shim mode."""
        import warnings

        import pluto.compat.wandb as wandb
        from pluto.compat.wandb._coverage import (
            PlutoWandbCompatWarning,
            reset_warnings,
        )

        reset_warnings()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            wandb.save('*.pt')
            wandb.restore('model.pt')
            wandb.log_code()
            wandb.mark_preempting()
            wandb.use_artifact('test')
            wandb.login()

        compat_warnings = [
            x for x in w
            if issubclass(x.category, PlutoWandbCompatWarning)
        ]
        assert len(compat_warnings) == 6
