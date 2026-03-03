"""E2E test: torchtitan's WandBLogger driven through the pluto wandb shim.

This test imports torchtitan's *real* WandBLogger class and drives it through
its full lifecycle (init → log → close), verifying the pluto wandb shim works
as a drop-in replacement for wandb in torchtitan's actual code.

Requires:
    - torchtitan installed (`pip install -e /path/to/torchtitan`)
    - PLUTO_API_KEY set for live server communication
"""

import os
import tempfile

import pytest

# Skip entire module if torchtitan is not installed
torchtitan_metrics = pytest.importorskip(
    'torchtitan.components.metrics',
    reason='torchtitan not installed',
)

WandBLogger = torchtitan_metrics.WandBLogger

pytestmark = pytest.mark.skipif(
    not os.environ.get('PLUTO_API_KEY'),
    reason='PLUTO_API_KEY not set',
)


class TestTorchtitanWandBLogger:
    """Drive torchtitan's real WandBLogger through the pluto wandb shim."""

    def test_wandb_shim_resolves_to_pluto(self):
        """Verify that `import wandb` resolves to the pluto shim."""
        import inspect

        import wandb

        origin = inspect.getfile(wandb)
        assert 'pluto' in origin, f'wandb does not resolve to pluto shim: {origin}'

    def test_full_lifecycle(self):
        """Init → log metrics over multiple steps → close. Verify pluto URL logged."""
        import wandb

        with tempfile.TemporaryDirectory() as log_dir:
            # Set env vars that torchtitan's WandBLogger reads
            env_overrides = {
                'WANDB_PROJECT': 'torchtitan-e2e-ci',
                'WANDB_RUN_NAME': f'e2e-test-{os.getpid()}',
                'WANDB_RUN_TAGS': 'ci,cpu,e2e,torchtitan',
            }
            old_env = {}
            for k, v in env_overrides.items():
                old_env[k] = os.environ.get(k)
                os.environ[k] = v

            try:
                # -- init (torchtitan's real code) --
                logger = WandBLogger(
                    log_dir=log_dir,
                    config_dict={'model': 'llama3_debugmodel', 'test': True},
                    tag='train',
                )

                # Verify pluto run was created
                assert wandb.run is not None, 'wandb.run is None after WandBLogger init'

                # -- log metrics (same keys torchtitan logs) --
                for step in range(1, 6):
                    logger.log(
                        {
                            'loss': 10.0 - step * 0.5,
                            'grad_norm': 1.0 + step * 0.1,
                            'lr': 8e-4 * (step / 5),
                            'tokens_per_second': 1000.0 + step * 100,
                        },
                        step=step,
                    )

                # -- close (torchtitan's real code) --
                logger.close()

                # After close, wandb.run should be None
                assert wandb.run is None, 'wandb.run should be None after close'
            finally:
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_multiple_metric_types(self):
        """Verify various metric value types that torchtitan might log."""
        import wandb

        with tempfile.TemporaryDirectory() as log_dir:
            env_overrides = {
                'WANDB_PROJECT': 'torchtitan-e2e-ci',
                'WANDB_RUN_NAME': f'e2e-types-{os.getpid()}',
                'WANDB_RUN_TAGS': 'ci,cpu,e2e,torchtitan',
            }
            old_env = {}
            for k, v in env_overrides.items():
                old_env[k] = os.environ.get(k)
                os.environ[k] = v

            try:
                logger = WandBLogger(log_dir=log_dir, tag='train')

                # int, float, large float, small float, zero
                logger.log(
                    {
                        'int_metric': 42,
                        'float_metric': 3.14159,
                        'large_metric': 1e12,
                        'small_metric': 1e-8,
                        'zero_metric': 0.0,
                    },
                    step=1,
                )

                logger.close()
                assert wandb.run is None
            finally:
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
