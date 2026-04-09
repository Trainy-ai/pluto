"""End-to-end tests for run forking.

Creates a parent run once per module, then forks from it in each test to verify
fork metadata, config/tag inheritance, and metric logging on forked runs.

Requires:
    - PLUTO_API_KEY environment variable
    - Network access to the Pluto server (staging with fork support)
"""

import time
from typing import Callable, Dict, List, TypeVar

import pytest

import pluto
import pluto.query as pq

FORK_PROJECT = 'testing-ci-fork'

_POLL_TIMEOUT = 15
_POLL_INTERVAL = 2

T = TypeVar('T')


def _poll(
    fn: Callable[[], T],
    check: Callable[[T], bool],
    timeout: float = _POLL_TIMEOUT,
    interval: float = _POLL_INTERVAL,
) -> T:
    """Call *fn* repeatedly until *check(result)* is truthy, then return result."""
    deadline = time.monotonic() + timeout
    last: T = fn()
    while not check(last):
        if time.monotonic() >= deadline:
            return last
        time.sleep(interval)
        last = fn()
    return last


def _poll_metric_names(
    project: str,
    run_id: int,
    expected: List[str],
    timeout: float = _POLL_TIMEOUT,
) -> List[str]:
    """Poll until all *expected* metric names are present on the server."""
    return _poll(
        fn=lambda: pq.get_metric_names(project, run_ids=[run_id]),
        check=lambda names: all(e in names for e in expected),
        timeout=timeout,
    )


def _poll_max_step(
    project: str,
    run_id: int,
    expected_step: int,
    metric: str = 'train/loss',
    timeout: float = 60,
) -> None:
    """Poll until ClickHouse has ingested metrics up to *expected_step*."""

    def _check() -> int:
        metrics = pq.get_metrics(project, run_id, metric_names=[metric])
        if hasattr(metrics, 'to_dict'):
            steps = metrics['step'].tolist()
        else:
            steps = [m['step'] for m in metrics]
        return max(steps) if steps else -1

    _poll(fn=_check, check=lambda s: s >= expected_step, timeout=timeout)


# ---------------------------------------------------------------------------
# Module-scoped parent run (created once, shared by all fork tests)
# ---------------------------------------------------------------------------

_PARENT_CONFIG = {'model': 'resnet50', 'lr': 0.001, 'epochs': 100}
_PARENT_TAGS = ['baseline', 'fork-parent']
_PARENT_STEPS = 10


@pytest.fixture(scope='module')
def parent_run() -> Dict:
    """Create a parent run with known config, tags, and metrics.

    Returns a dict with ``run_id``, ``config``, ``tags``, and ``max_step``.
    """
    run = pluto.init(
        project=FORK_PROJECT,
        name=f'fork-parent-{int(time.time())}',
        config=_PARENT_CONFIG,
        tags=_PARENT_TAGS,
    )
    run_id = run.settings._op_id

    for step in range(_PARENT_STEPS):
        run.log({'train/loss': 1.0 - step * 0.1, 'train/acc': step * 0.1})
    run.finish()

    # Wait for ClickHouse to ingest all steps so forkStep validation passes
    _poll_max_step(FORK_PROJECT, run_id, _PARENT_STEPS - 1)

    return {
        'run_id': run_id,
        'config': _PARENT_CONFIG,
        'tags': _PARENT_TAGS,
        'max_step': _PARENT_STEPS - 1,
    }


# ---------------------------------------------------------------------------
# Fork metadata
# ---------------------------------------------------------------------------


def test_fork_e2e_run_metadata(parent_run):
    """Forked run has correct fork_run_id and fork_step on the client."""
    fork_step = 5
    run = pluto.init(
        project=FORK_PROJECT,
        name=f'fork-meta-{int(time.time())}',
        fork_run_id=parent_run['run_id'],
        fork_step=fork_step,
    )
    run_id = run.settings._op_id

    assert run.fork_run_id == parent_run['run_id']
    assert run.fork_step == fork_step

    run.finish()

    server_run = pq.get_run(FORK_PROJECT, run_id)
    assert server_run['status'] == 'COMPLETED'


def test_fork_e2e_default_inherits_config(parent_run):
    """By default (inherit_config not set), forked run inherits parent config."""
    run = pluto.init(
        project=FORK_PROJECT,
        name=f'fork-cfg-default-{int(time.time())}',
        fork_run_id=parent_run['run_id'],
        fork_step=5,
    )
    run_id = run.settings._op_id
    run.finish()

    server_config = pq.get_run(FORK_PROJECT, run_id).get('config', {})
    assert server_config.get('model') == 'resnet50'
    assert server_config.get('lr') == 0.001
    assert server_config.get('epochs') == 100


def test_fork_e2e_inherit_config_with_override(parent_run):
    """Forked run merges explicit config on top of inherited parent config."""
    run = pluto.init(
        project=FORK_PROJECT,
        name=f'fork-cfg-override-{int(time.time())}',
        fork_run_id=parent_run['run_id'],
        fork_step=5,
        inherit_config=True,
        config={'lr': 0.01, 'scheduler': 'cosine'},
    )
    run_id = run.settings._op_id
    run.finish()

    server_config = pq.get_run(FORK_PROJECT, run_id).get('config', {})
    # Inherited from parent
    assert server_config.get('model') == 'resnet50'
    assert server_config.get('epochs') == 100
    # Overridden by child
    assert server_config.get('lr') == 0.01
    # New from child
    assert server_config.get('scheduler') == 'cosine'


def test_fork_e2e_no_inherit_config(parent_run):
    """inherit_config=False means forked run only has its own config."""
    run = pluto.init(
        project=FORK_PROJECT,
        name=f'fork-no-cfg-{int(time.time())}',
        fork_run_id=parent_run['run_id'],
        fork_step=5,
        inherit_config=False,
        config={'custom_only': True},
    )
    run_id = run.settings._op_id
    run.finish()

    server_config = pq.get_run(FORK_PROJECT, run_id).get('config', {})
    assert server_config.get('custom_only') is True
    # Parent config should NOT be inherited
    assert 'model' not in server_config


# ---------------------------------------------------------------------------
# Tag inheritance
# ---------------------------------------------------------------------------


def test_fork_e2e_inherit_tags(parent_run):
    """inherit_tags=True copies parent tags to forked run."""
    run = pluto.init(
        project=FORK_PROJECT,
        name=f'fork-tags-inherit-{int(time.time())}',
        fork_run_id=parent_run['run_id'],
        fork_step=5,
        inherit_tags=True,
        tags=['child-tag'],
    )
    run_id = run.settings._op_id
    run.finish()

    server_tags = pq.get_run(FORK_PROJECT, run_id).get('tags', [])
    # Inherited from parent
    for tag in _PARENT_TAGS:
        assert tag in server_tags, f"Inherited tag '{tag}' not found"
    # Child's own tag
    assert 'child-tag' in server_tags


def test_fork_e2e_no_inherit_tags_by_default(parent_run):
    """By default (inherit_tags not set), parent tags are NOT inherited."""
    run = pluto.init(
        project=FORK_PROJECT,
        name=f'fork-tags-default-{int(time.time())}',
        fork_run_id=parent_run['run_id'],
        fork_step=5,
        tags=['only-mine'],
    )
    run_id = run.settings._op_id
    run.finish()

    server_tags = pq.get_run(FORK_PROJECT, run_id).get('tags', [])
    assert 'only-mine' in server_tags
    # Parent tags should NOT appear (server default is inheritTags=false)
    assert 'fork-parent' not in server_tags


# ---------------------------------------------------------------------------
# Metrics on forked runs
# ---------------------------------------------------------------------------


def test_fork_e2e_log_metrics(parent_run):
    """Forked run can log its own metrics that appear on the server."""
    run = pluto.init(
        project=FORK_PROJECT,
        name=f'fork-metrics-{int(time.time())}',
        fork_run_id=parent_run['run_id'],
        fork_step=5,
    )
    run_id = run.settings._op_id

    for step in range(5):
        run.log({'fork/loss': 0.5 - step * 0.1})
    run.finish()

    metric_names = _poll_metric_names(FORK_PROJECT, run_id, ['fork/loss'])
    assert 'fork/loss' in metric_names

    metrics = pq.get_metrics(FORK_PROJECT, run_id, metric_names=['fork/loss'])
    if hasattr(metrics, 'to_dict'):
        values = metrics['value'].tolist()
    else:
        values = [m['value'] for m in metrics]
    assert len(values) == 5
    assert values[0] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Chain forking (fork from a fork)
# ---------------------------------------------------------------------------


def test_fork_e2e_chain_fork(parent_run):
    """Fork from a forked run (two-level lineage)."""
    # First fork
    run1 = pluto.init(
        project=FORK_PROJECT,
        name=f'fork-chain-1-{int(time.time())}',
        fork_run_id=parent_run['run_id'],
        fork_step=5,
    )
    run1_id = run1.settings._op_id
    for step in range(5):
        run1.log({'chain/metric': step})
    run1.finish()

    _poll_max_step(FORK_PROJECT, run1_id, 4, metric='chain/metric')

    # Second fork (from the first fork)
    run2 = pluto.init(
        project=FORK_PROJECT,
        name=f'fork-chain-2-{int(time.time())}',
        fork_run_id=run1_id,
        fork_step=3,
    )
    run2_id = run2.settings._op_id

    # Server may resolve lineage — fork_run_id could be run1 or parent
    assert run2.fork_run_id is not None
    assert run2.fork_step == 3

    run2.finish()

    server_run = pq.get_run(FORK_PROJECT, run2_id)
    assert server_run['status'] == 'COMPLETED'
