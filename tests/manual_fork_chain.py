"""
Manual fork chain test — simulates iterative LLM training over 5 runs.

Each run forks from the previous one at its last step, inheriting config
and adding its own metrics. All runs share the same name so they group
in experiments mode.

Usage:
    PLUTO_API_KEY=<token> python tests/manual_fork_chain.py

Then open the testing-ci-fork project in the UI and look for the
experiment named 'llm-pretrain-<commit>'.
"""

import subprocess
import time
from pathlib import Path

import pluto

PROJECT = 'testing-ci-fork'
NUM_RUNS = 5
STEPS_PER_RUN = 20


def get_commit_hash() -> str:
    try:
        result = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).resolve().parents[1]),
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
            metrics = pq.get_metrics(PROJECT, run_id, metric_names=['train/loss'])
            if hasattr(metrics, 'to_dict'):
                steps = metrics['step'].tolist()
            else:
                steps = [m['step'] for m in metrics]
            max_seen = max(steps) if steps else -1
            if max_seen >= expected_step:
                print(f'  ClickHouse has step {max_seen} (needed {expected_step})')
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
    experiment_name = f'llm-pretrain-{commit}'

    print(f'Chained fork test — {NUM_RUNS} runs, {STEPS_PER_RUN} steps each')
    print(f'Project:    {PROJECT}')
    print(f'Experiment: {experiment_name}')
    print()

    prev_id = None
    prev_last_step = None
    all_runs = []

    for i in range(NUM_RUNS):
        # Each run tweaks the learning rate (simulating LR decay across restarts)
        lr = 0.001 * (0.5**i)
        config = {
            'model': 'llama-7b',
            'lr': lr,
            'run_index': i,
            'total_steps': (i + 1) * STEPS_PER_RUN,
        }

        fork_kwargs = {}
        if prev_id is not None:
            fork_kwargs = {
                'fork_run_id': prev_id,
                'fork_step': prev_last_step,
                'inherit_config': True,
            }
            print(
                f'[{i + 1}/{NUM_RUNS}] Forking from run {prev_id}'
                f' at step {prev_last_step}  (lr={lr})'
            )
        else:
            print(f'[{i + 1}/{NUM_RUNS}] Starting root run  (lr={lr})')

        run = pluto.init(
            project=PROJECT,
            name=experiment_name,
            config=config,
            tags=['chain-test', f'run-{i}'],
            **fork_kwargs,
        )
        run_id = run.settings._op_id

        if prev_id is not None:
            print(
                f'  Server resolved: fork_run_id={run.fork_run_id}'
                f'  fork_step={run.fork_step}'
            )

        # Log metrics — global step is continuous across all runs
        global_step_offset = i * STEPS_PER_RUN
        for s in range(STEPS_PER_RUN):
            global_step = global_step_offset + s
            loss = 2.0 / (global_step + 1) + 0.01 * (0.5**i)
            run.log(
                {
                    'train/loss': loss,
                    'train/lr': lr,
                    'train/tokens_seen': (global_step + 1) * 4096,
                }
            )

        last_step = global_step_offset + STEPS_PER_RUN - 1
        run.finish()
        print(f'  Finished run {run_id}  (steps {global_step_offset}..{last_step})')

        # Wait for ClickHouse to ingest up to last_step before forking
        if i < NUM_RUNS - 1:
            wait_for_step(run_id, last_step)

        all_runs.append({'index': i, 'id': run_id, 'lr': lr})
        prev_id = run_id
        prev_last_step = last_step

    # Summary
    print()
    print('=' * 60)
    print('Chain complete!')
    print(f'Experiment: {experiment_name}')
    print(f'Project:    {PROJECT}')
    print()
    print('What to verify in the UI:')
    print('  1. Switch to experiments mode — all 5 runs group under one name')
    print('  2. Open a chart for train/loss — should be one continuous line')
    print(f'     from step 0 to step {NUM_RUNS * STEPS_PER_RUN - 1}')
    print('  3. Each fork point should be seamless (no gaps)')
    print()
    for r in all_runs:
        idx = r['index']
        fork_note = f'fork from {all_runs[idx - 1]["id"]}' if idx > 0 else 'root'
        print(f'  Run {r["index"]}: id={r["id"]}  lr={r["lr"]}  ({fork_note})')
    print('=' * 60)


if __name__ == '__main__':
    main()
