#!/usr/bin/env python3
"""
Standalone visual parity runner.

Run this script in two separate environments to compare dashboards:

  Environment A (pluto shim):
    pip install pluto-ml
    PLUTO_API_TOKEN=<token> python tests/wandb_visual_parity_runner.py

  Environment B (real wandb):
    pip install wandb
    WANDB_API_KEY=<key> python tests/wandb_visual_parity_runner.py

Both runs will use identical config, metrics, and data types.
Compare the resulting dashboard URLs visually.
"""

import argparse
import math
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Detect which backend we're running on
# ---------------------------------------------------------------------------


def _detect_backend():
    """Detect whether we're running on real wandb or pluto shim."""
    try:
        import wandb

        origin = getattr(getattr(wandb, '__spec__', None), 'origin', '') or ''
        if 'pluto' in origin:
            return 'pluto-shim', wandb
        # Check if it has the real wandb's internal modules
        if hasattr(wandb, 'sdk') and hasattr(wandb.sdk, 'wandb_run'):
            return 'real-wandb', wandb
        # Fallback: check for pluto re-export marker
        if hasattr(wandb, '_get_module'):
            return 'pluto-shim', wandb
        return 'real-wandb', wandb
    except ImportError:
        print('ERROR: neither wandb nor pluto-ml is installed')
        sys.exit(1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

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


def run_training(wb, project, run_name, seed=42):
    """Run a realistic training loop that exercises the wandb API surface."""
    np.random.seed(seed)

    run = wb.init(
        project=project,
        name=run_name,
        config=CONFIG,
        tags=['visual-parity', 'automated'],
    )

    # Post-init config mutations
    wb.config.update({'scheduler': 'cosine_annealing'})
    wb.config.seed = seed

    print(f'  Logging {NUM_EPOCHS} epochs x {NUM_STEPS_PER_EPOCH} steps...')

    for epoch in range(NUM_EPOCHS):
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

        val_loss = base_loss * 1.1 + np.random.normal(0, 0.03)
        val_acc = base_acc * 0.98 + np.random.normal(0, 0.01)
        wb.log(
            {
                'val/loss': val_loss,
                'val/accuracy': min(val_acc, 1.0),
                'epoch': epoch,
            }
        )

        # Histogram every 5 epochs
        if epoch % 5 == 0:
            gradient_norms = np.random.lognormal(-1.0 + epoch * 0.05, 0.5, 1000)
            wb.log(
                {
                    'gradients/norm_distribution': wb.Histogram(gradient_norms),
                }
            )

        # Table at midpoint
        if epoch == NUM_EPOCHS // 2:
            table = wb.Table(
                columns=['sample_id', 'predicted', 'actual', 'confidence'],
                data=[
                    [i, i % 10, (i + 1) % 10, round(np.random.uniform(0.5, 1.0), 3)]
                    for i in range(20)
                ],
            )
            wb.log({'predictions': table})

        # Image every 5 epochs
        if epoch % 5 == 0:
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            wb.log(
                {
                    'samples/random_image': wb.Image(img, caption=f'epoch-{epoch}'),
                }
            )

        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch + 1}/{NUM_EPOCHS} done')

    run.summary['best_val_loss'] = 0.15
    run.summary['best_val_accuracy'] = 0.94
    run.summary['total_steps'] = NUM_EPOCHS * NUM_STEPS_PER_EPOCH

    wb.finish()

    # Extract URL
    url = getattr(run, 'url', None)
    if url is None:
        settings = getattr(getattr(run, '_op', None), 'settings', None)
        if settings:
            url = getattr(settings, 'url_view', None)
    if url is None:
        url = getattr(run, '_get_run_url', lambda: None)()

    return url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Visual parity runner for wandb vs pluto shim'
    )
    parser.add_argument(
        '--project',
        default=None,
        help='Project name (default: wandb-visual-parity)',
    )
    parser.add_argument(
        '--name',
        default=None,
        help='Run name (default: auto-generated)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible noise',
    )
    args = parser.parse_args()

    backend_name, wb = _detect_backend()
    ts = int(time.time()) % 100000

    project = args.project or 'wandb-visual-parity'
    run_name = args.name or f'{backend_name}-{ts}'

    print()
    print('=' * 60)
    print(f'  Backend:  {backend_name}')
    print(f'  Project:  {project}')
    print(f'  Run:      {run_name}')
    print(f'  Seed:     {args.seed}')
    print('=' * 60)
    print()

    url = run_training(wb, project, run_name, seed=args.seed)

    print()
    print('=' * 60)
    print(f'  DONE — {backend_name}')
    print(f'  URL: {url or "(check console output)"}')
    print('=' * 60)
    print()


if __name__ == '__main__':
    main()
