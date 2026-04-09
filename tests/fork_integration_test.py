"""
End-to-end fork integration test for the Pluto platform.

This test validates the full fork workflow by:
1. Creating a parent run with config, tags, and metrics
2. Forking from the parent with various inheritance options
3. Logging metrics on forked runs
4. Chaining forks (fork from a fork)

Run names are suffixed with the git commit hash so results are
visible in the testing-ci-fork project UI.

Usage:
    python tests/fork_integration_test.py
"""

import subprocess
import time
from pathlib import Path

import pluto

FORK_PROJECT = 'testing-ci-fork'

PARENT_CONFIG = {
    'model': 'resnet50',
    'lr': 0.001,
    'optimizer': 'adam',
    'epochs': 100,
}
PARENT_TAGS = ['baseline', 'integration-test']
PARENT_STEPS = 10


def get_commit_hash() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(repo_root),
        )
        return result.decode().strip()
    except Exception:
        return 'unknown'


def wait_for_step(run_id: int, expected_step: int, timeout: float = 60.0) -> None:
    """Poll until ClickHouse has ingested metrics up to expected_step."""
    import pluto.query as pq

    deadline = time.monotonic() + timeout
    max_seen = -1
    while time.monotonic() < deadline:
        try:
            metrics = pq.get_metrics(FORK_PROJECT, run_id, metric_names=['train/loss'])
            if hasattr(metrics, 'to_dict'):
                steps = metrics['step'].tolist()
            else:
                steps = [m['step'] for m in metrics]
            max_seen = max(steps) if steps else -1
            if max_seen >= expected_step:
                return
        except Exception:
            pass
        time.sleep(3)
    print(
        f'  Warning: timed out waiting for step {expected_step} '
        f'on run {run_id} (max seen: {max_seen})'
    )


def main() -> None:
    commit = get_commit_hash()
    experiment_name = f'fork-experiment-{commit}'
    print(f'Fork integration test — commit {commit}')
    print(f'Project: {FORK_PROJECT}')
    print(f'Experiment name: {experiment_name}')
    print()

    # ------------------------------------------------------------------
    # 1. Parent run
    # ------------------------------------------------------------------
    print('[1/5] Creating parent run')
    parent = pluto.init(
        project=FORK_PROJECT,
        name=experiment_name,
        config=PARENT_CONFIG,
        tags=PARENT_TAGS,
    )
    parent_id = parent.settings._op_id

    for step in range(PARENT_STEPS):
        parent.log(
            {
                'train/loss': 1.0 - step * 0.08,
                'train/acc': step * 0.09,
            }
        )
        print(f'  step {step}: loss={1.0 - step * 0.08:.2f}  acc={step * 0.09:.2f}')
    parent.finish()
    print(f'  Parent run finished (id={parent_id})')

    # Wait for ClickHouse to ingest all parent steps so forkStep validation passes
    wait_for_step(parent_id, PARENT_STEPS - 1)

    # ------------------------------------------------------------------
    # 2. Fork with inherited config + overrides
    # ------------------------------------------------------------------
    print('\n[2/5] Forking with config inheritance + override')
    fork1 = pluto.init(
        project=FORK_PROJECT,
        name=experiment_name,
        fork_run_id=parent_id,
        fork_step=5,
        inherit_config=True,
        config={'lr': 0.01, 'scheduler': 'cosine'},
        tags=['fork', 'config-override'],
    )
    fork1_id = fork1.settings._op_id
    print(f'  fork_run_id={fork1.fork_run_id}  fork_step={fork1.fork_step}')

    for step in range(10):
        fork1.log(
            {
                'train/loss': 0.5 - step * 0.04,
                'train/acc': 0.5 + step * 0.05,
            }
        )
    fork1.finish()
    print(f'  Fork 1 finished (id={fork1_id})')

    wait_for_step(fork1_id, 9)

    # ------------------------------------------------------------------
    # 3. Fork with inherited tags
    # ------------------------------------------------------------------
    print('\n[3/5] Forking with tag inheritance')
    fork2 = pluto.init(
        project=FORK_PROJECT,
        name=experiment_name,
        fork_run_id=parent_id,
        fork_step=5,
        inherit_tags=True,
        tags=['fork', 'tag-inherit'],
    )
    fork2_id = fork2.settings._op_id
    print(f'  fork_run_id={fork2.fork_run_id}  fork_step={fork2.fork_step}')

    for step in range(8):
        fork2.log(
            {
                'train/loss': 0.6 - step * 0.06,
                'val/loss': 0.7 - step * 0.05,
            }
        )
    fork2.finish()
    print(f'  Fork 2 finished (id={fork2_id})')

    # ------------------------------------------------------------------
    # 4. Fork without inheritance
    # ------------------------------------------------------------------
    print('\n[4/5] Forking without inheritance')
    fork3 = pluto.init(
        project=FORK_PROJECT,
        name=experiment_name,
        fork_run_id=parent_id,
        fork_step=3,
        inherit_config=False,
        inherit_tags=False,
        config={'model': 'vit', 'lr': 0.0001},
        tags=['fork', 'fresh-start'],
    )
    fork3_id = fork3.settings._op_id
    print(f'  fork_run_id={fork3.fork_run_id}  fork_step={fork3.fork_step}')

    for step in range(15):
        fork3.log(
            {
                'train/loss': 0.8 - step * 0.05,
                'train/acc': 0.2 + step * 0.05,
            }
        )
    fork3.finish()
    print(f'  Fork 3 finished (id={fork3_id})')

    # ------------------------------------------------------------------
    # 5. Chain fork (fork from fork1)
    # ------------------------------------------------------------------
    print('\n[5/5] Chain fork (from fork 1)')
    fork4 = pluto.init(
        project=FORK_PROJECT,
        name=experiment_name,
        fork_run_id=fork1_id,
        fork_step=5,
        inherit_config=True,
        inherit_tags=True,
        config={'lr': 0.005},
        tags=['fork', 'chain'],
    )
    fork4_id = fork4.settings._op_id
    print(f'  fork_run_id={fork4.fork_run_id}  fork_step={fork4.fork_step}')

    for step in range(10):
        fork4.log(
            {
                'train/loss': 0.3 - step * 0.02,
                'train/acc': 0.7 + step * 0.03,
            }
        )
    fork4.finish()
    print(f'  Fork 4 finished (id={fork4_id})')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Fork integration test completed successfully!')
    print(f'Commit:     {commit}')
    print(f'Project:    {FORK_PROJECT}')
    print(f'Experiment: {experiment_name}')
    print('Runs created (all share the same name for experiments mode):')
    print(f'  Parent:            id={parent_id}')
    print(f'  Fork (config):     id={fork1_id}  fork_step=5')
    print(f'  Fork (tags):       id={fork2_id}  fork_step=5')
    print(f'  Fork (no inherit): id={fork3_id}  fork_step=3')
    print(f'  Fork (chain):      id={fork4_id}  fork_step=5 (from {fork1_id})')
    print('=' * 60)


if __name__ == '__main__':
    main()
