"""Verify that metric timestamps are consistent with run createdAt.

A run's first metric should have a relative time (metric.time - createdAt)
close to zero, not offset by hours due to timezone bugs.
"""

import time

import pluto
import pluto.query as pq
from tests.utils import get_task_name

TESTING_PROJECT_NAME = 'testing-ci'
MAX_OFFSET_SECONDS = 60  # createdAt and first metric should be within 60s


def test_time_offset():
    """End-to-end: create a run, log a metric, then verify timestamps align."""
    task_name = get_task_name()
    run = pluto.init(project=TESTING_PROJECT_NAME, name=task_name, config={})
    run_id = run.id

    run.log({'epoch': 0})
    time.sleep(2)  # let sync process flush
    run.finish()

    # Query the run and its metrics back from the server
    client = pq.Client()
    run_info = client.get_run(project=TESTING_PROJECT_NAME, run_id=run_id)
    metrics = client.get_metrics(
        project=TESTING_PROJECT_NAME,
        run_id=run_id,
        metric_names=['epoch'],
    )

    # Parse createdAt (ISO 8601 string) and first metric time (epoch ms)
    from datetime import datetime

    created_at_str = run_info['createdAt']
    # Handle both 'Z' suffix and '+00:00' formats
    created_at_str = created_at_str.replace('Z', '+00:00')
    created_at = datetime.fromisoformat(created_at_str)
    created_at_ms = created_at.timestamp() * 1000

    # Get first metric timestamp
    if hasattr(metrics, 'iloc'):
        first_metric_time_ms = float(metrics.iloc[0]['time'])
    else:
        first_metric_time_ms = float(metrics[0]['time'])

    offset_seconds = abs(first_metric_time_ms - created_at_ms) / 1000

    assert offset_seconds < MAX_OFFSET_SECONDS, (
        f'Time offset between createdAt and first metric is {offset_seconds:.1f}s '
        f'(max {MAX_OFFSET_SECONDS}s). '
        f'createdAt={created_at_str}, metric_time_ms={first_metric_time_ms}'
    )
