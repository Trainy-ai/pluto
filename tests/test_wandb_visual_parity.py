"""
Visual parity test: runs the same training script through both the real
wandb SDK and the pluto wandb shim, then prints dashboard URLs for
side-by-side visual inspection.

Usage:
    # Run the pluto shim side (always available):
    python -m pytest tests/test_wandb_visual_parity.py -k pluto -v -s

    # Run the real wandb side (requires `pip install wandb`):
    python -m pytest tests/test_wandb_visual_parity.py -k real_wandb -v -s

    # Run both (requires wandb installed):
    python -m pytest tests/test_wandb_visual_parity.py -v -s

Requirements:
    - PLUTO_API_TOKEN must be set for the pluto side
    - WANDB_API_KEY must be set for the real wandb side
    - `pip install wandb` for the real wandb tests

The test names, configs, and logged data are identical so the resulting
dashboards should look the same.
"""

import importlib
import math
import os

import numpy as np
import pytest

from tests.utils import get_task_name

# ---------------------------------------------------------------------------
# Shared constants — identical between both sides
# ---------------------------------------------------------------------------

PLUTO_PROJECT = 'wandb-visual-parity'
WANDB_PROJECT = os.environ.get('WANDB_VISUAL_PARITY_PROJECT', 'wandb-visual-parity')
NUM_EPOCHS = 20
NUM_STEPS_PER_EPOCH = 50
CONFIG = {
    'lr': 0.001,
    'batch_size': 64,
    'optimizer': 'adam',
    'architecture': 'resnet18',
    'dataset': 'cifar10',
    'epochs': NUM_EPOCHS,
    'dropout': 0.1,
    'weight_decay': 1e-4,
}


# ---------------------------------------------------------------------------
# Shared training loop — parameterised by the wandb module
# ---------------------------------------------------------------------------


def _run_training_loop(wb, project, run_name):
    """Run a fake but realistic training loop through the given wandb module.

    Returns the dashboard URL (or None).
    """
    run = wb.init(
        project=project,
        name=run_name,
        config=CONFIG,
        tags=['visual-parity', 'automated'],
    )

    # Post-init config mutation (wandb pattern)
    wb.config.update({'scheduler': 'cosine_annealing'})
    wb.config.seed = 42

    for epoch in range(NUM_EPOCHS):
        # Simulated training metrics with realistic decay curves
        base_loss = 2.0 * math.exp(-0.15 * epoch) + 0.1
        base_acc = 1.0 - math.exp(-0.2 * epoch) * 0.6

        for step in range(NUM_STEPS_PER_EPOCH):
            noise = np.random.normal(0, 0.02)

            wb.log(
                {
                    'train/loss': base_loss + noise + 0.05 * math.sin(step * 0.3),
                    'train/accuracy': min(base_acc + noise * 0.5, 1.0),
                    'train/learning_rate': CONFIG['lr'] * (0.95**epoch),
                }
            )

        # Epoch-level validation metrics
        val_loss = base_loss * 1.1 + np.random.normal(0, 0.03)
        val_acc = base_acc * 0.98 + np.random.normal(0, 0.01)

        wb.log(
            {
                'val/loss': val_loss,
                'val/accuracy': min(val_acc, 1.0),
                'epoch': epoch,
            }
        )

        # Log a histogram every 5 epochs
        if epoch % 5 == 0:
            gradient_norms = np.random.lognormal(
                mean=-1.0 + epoch * 0.05,
                sigma=0.5,
                size=1000,
            )
            wb.log(
                {
                    'gradients/norm_distribution': wb.Histogram(gradient_norms),
                }
            )

        # Log a table at the midpoint
        if epoch == NUM_EPOCHS // 2:
            table = wb.Table(
                columns=['sample_id', 'predicted', 'actual', 'confidence'],
                data=[
                    [i, i % 10, (i + 1) % 10, round(np.random.uniform(0.5, 1.0), 3)]
                    for i in range(20)
                ],
            )
            wb.log({'predictions': table})

        # Log an image every 5 epochs
        if epoch % 5 == 0:
            img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            wb.log(
                {
                    'samples/random_image': wb.Image(
                        img_array, caption=f'epoch-{epoch}'
                    ),
                }
            )

    # Manual summary overrides
    run.summary['best_val_loss'] = 0.15
    run.summary['best_val_accuracy'] = 0.94
    run.summary['total_steps'] = NUM_EPOCHS * NUM_STEPS_PER_EPOCH

    wb.finish()

    # Extract URL
    url = getattr(run, 'url', None)
    if url is None:
        url = getattr(getattr(run, '_op', None), 'settings', None)
        if url is not None:
            url = getattr(url, 'url_view', None)

    return url


# ---------------------------------------------------------------------------
# Pluto shim side
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get('PLUTO_API_TOKEN'),
    reason='PLUTO_API_TOKEN not set — cannot run pluto side',
)
class TestPlutoShimSide:
    """Runs the training loop through ``import wandb`` (the pluto shim)."""

    def test_pluto_training_loop(self):
        import wandb

        run_name = f'pluto-shim-{get_task_name()}'
        url = _run_training_loop(wandb, PLUTO_PROJECT, run_name)

        print('\n')
        print('=' * 60)
        print('  PLUTO SHIM RUN')
        print(f'  Name: {run_name}')
        print(f'  URL:  {url or "(not available)"}')
        print('=' * 60)


# ---------------------------------------------------------------------------
# Real wandb side
# ---------------------------------------------------------------------------

_has_real_wandb = False
try:
    # Only consider real wandb available if it's NOT our shim
    spec = importlib.util.find_spec('wandb')
    if spec and spec.origin:
        # Our shim lives under the pluto repo; real wandb doesn't
        _has_real_wandb = 'pluto' not in (spec.origin or '')
except Exception:
    pass


@pytest.mark.skipif(
    not _has_real_wandb,
    reason='real wandb package not installed (only pluto shim found)',
)
@pytest.mark.skipif(
    not os.environ.get('WANDB_API_KEY'),
    reason='WANDB_API_KEY not set — cannot run real wandb side',
)
class TestRealWandbSide:
    """Runs the training loop through the real ``wandb`` SDK."""

    def test_real_wandb_training_loop(self):
        import wandb

        run_name = f'real-wandb-{get_task_name()}'
        url = _run_training_loop(wandb, WANDB_PROJECT, run_name)

        if url is None and wandb.run is None:
            # wandb.finish() clears wandb.run, but the URL was printed
            url = '(check wandb console output above)'

        print('\n')
        print('=' * 60)
        print('  REAL WANDB RUN')
        print(f'  Name: {run_name}')
        print(f'  URL:  {url or "(not available)"}')
        print('=' * 60)


# ---------------------------------------------------------------------------
# Combined runner — convenience for running both back-to-back
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get('PLUTO_API_TOKEN'),
    reason='PLUTO_API_TOKEN not set',
)
@pytest.mark.skipif(
    not _has_real_wandb,
    reason='real wandb not installed',
)
@pytest.mark.skipif(
    not os.environ.get('WANDB_API_KEY'),
    reason='WANDB_API_KEY not set',
)
class TestSideBySide:
    """Run both sides in one test and print URLs together."""

    def test_side_by_side(self):
        # Use a shared suffix so the runs are easy to find together
        suffix = get_task_name()

        # --- Pluto shim ---
        import wandb as pluto_wandb

        pluto_name = f'pluto-{suffix}'
        pluto_url = _run_training_loop(pluto_wandb, PLUTO_PROJECT, pluto_name)

        # --- Real wandb ---
        # We need to force-reimport real wandb. Since our shim took the
        # `wandb` namespace, this only works if real wandb is installed
        # in a separate venv. For CI, use subprocess isolation instead.
        # Here we just document the limitation.
        real_name = f'wandb-{suffix}'
        real_url = '(run separately: pytest -k real_wandb)'

        print('\n')
        print('=' * 60)
        print('  VISUAL PARITY COMPARISON')
        print('-' * 60)
        print(f'  Pluto shim:  {pluto_url or "(not available)"}')
        print(f'               name={pluto_name}')
        print(f'  Real wandb:  {real_url}')
        print(f'               name={real_name}')
        print('=' * 60)
        print()
        print('  To run real wandb side in a separate env:')
        print('    pip install wandb')
        print(
            f'    WANDB_API_KEY=<key> python -m pytest {__file__} -k real_wandb -v -s'
        )
        print('=' * 60)
