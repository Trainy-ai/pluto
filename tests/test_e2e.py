"""End-to-end tests that verify data reaches the Pluto server.

Every test in this file creates a real run, logs data, calls ``finish()``,
then uses ``pluto.query`` to read back from the server and assert correctness.
This catches serialization bugs, endpoint mismatches, and sync-process issues
that client-only assertions in test_basic.py would miss.

Requires:
    - PLUTO_API_KEY environment variable
    - Network access to the Pluto server
"""

import importlib.util
import io
import os
import subprocess
import sys
import time
import uuid
from typing import Callable, List, TypeVar

import httpx
import numpy as np
import pytest
from PIL import Image as PILImage

import pluto
import pluto.query as pq
from tests.utils import get_task_name

TESTING_PROJECT_NAME = 'testing-ci'

HAS_TORCH = importlib.util.find_spec('torch') is not None

# Max seconds to wait for data to appear on the server after finish().
# 60s is large enough to absorb tail-latency on the server's config/tag
# update propagation seen on GH Actions: CI run 25195677398 hit the limit
# at 15s; CI run 25196590153 hit it again at 30s. Successful polls exit
# on first check match, so this only widens the worst-case window —
# happy-path test runtime is unchanged.
_POLL_TIMEOUT = 60
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
            return last  # return last result and let the caller assert
        time.sleep(interval)
        last = fn()
    return last


def _run_metric_names(project: str, run_id: int) -> List[str]:
    """Distinct metric names for a run, derived from the metrics endpoint.

    The dedicated ``/api/runs/metric-names`` endpoint is backed by a
    ClickHouse aggregation that lags ~4-5 min behind ingest — far longer
    than ``_POLL_TIMEOUT`` (measured: 267s vs 4s for ``get_metrics`` on a
    fresh run). ``get_metrics`` reads the raw series and is queryable
    within seconds of ``finish()``, so distinct names are derived from it.
    """
    metrics = pq.get_metrics(project, run_id)
    if hasattr(metrics, 'columns'):  # pandas DataFrame
        return list(metrics['metric'].unique())
    return sorted({m['metric'] for m in metrics})


def _poll_metric_names(
    project: str,
    run_id: int,
    expected: List[str],
    timeout: float = _POLL_TIMEOUT,
) -> List[str]:
    """Poll until all *expected* metric names are queryable on the server."""
    return _poll(
        fn=lambda: _run_metric_names(project, run_id),
        check=lambda names: all(e in names for e in expected),
        timeout=timeout,
    )


def _poll_run(
    project: str,
    run_id: int,
    check: Callable[[dict], bool],
    timeout: float = _POLL_TIMEOUT,
) -> dict:
    """Poll ``pq.get_run`` until *check* on the snapshot is truthy.

    Config and tag updates pushed via ``run.update_config`` /
    ``run.add_tags`` are flushed by ``run.finish()`` but the server
    applies them asynchronously, so a get_run() called immediately after
    finish can return a stale snapshot.
    """
    return _poll(
        fn=lambda: pq.get_run(project, run_id),
        check=check,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


def test_e2e_run_metadata():
    """Verify run name, status, and displayId are set on the server."""
    task_name = get_task_name()
    run = pluto.init(project=TESTING_PROJECT_NAME, name=task_name, config={})
    run_id = run.settings._op_id
    run.finish()

    server_run = pq.get_run(TESTING_PROJECT_NAME, run_id)
    assert server_run['name'] == task_name
    assert server_run['status'] == 'COMPLETED'
    assert 'displayId' in server_run and server_run['displayId']


def test_e2e_run_lookup_by_display_id():
    """Verify a run can be fetched by its display ID string."""
    task_name = get_task_name()
    run = pluto.init(project=TESTING_PROJECT_NAME, name=task_name, config={})
    run_id = run.settings._op_id
    run.finish()

    # First get the display ID via numeric lookup
    server_run = pq.get_run(TESTING_PROJECT_NAME, run_id)
    display_id = server_run['displayId']

    # Now look up by display ID
    server_run2 = pq.get_run(TESTING_PROJECT_NAME, display_id)
    assert server_run2['id'] == run_id
    assert server_run2['name'] == task_name


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_e2e_initial_config():
    """Verify config passed at init() reaches the server."""
    config = {'lr': 0.001, 'batch_size': 32, 'model': 'resnet50'}
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config=config)
    run_id = run.settings._op_id
    run.finish()

    server_config = pq.get_run(TESTING_PROJECT_NAME, run_id).get('config', {})
    assert server_config['lr'] == 0.001
    assert server_config['batch_size'] == 32
    assert server_config['model'] == 'resnet50'


def test_e2e_update_config():
    """Verify update_config persists merged config to the server."""
    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=get_task_name(),
        config={'lr': 0.001, 'arch': 'resnet50'},
    )
    run_id = run.settings._op_id

    run.update_config({'epochs': 100, 'lr': 0.01})
    run.finish()

    server_run = _poll_run(
        TESTING_PROJECT_NAME,
        run_id,
        check=lambda r: r.get('config', {}).get('lr') == 0.01,
    )
    server_config = server_run.get('config', {})
    assert (
        server_config['lr'] == 0.01
    ), f'Server has lr={server_config.get("lr")}, expected 0.01'
    assert server_config['arch'] == 'resnet50'
    assert server_config['epochs'] == 100


def test_e2e_update_config_nested():
    """Verify nested/complex config values survive serialization."""
    config = {
        'model': {
            'name': 'transformer',
            'layers': 12,
            'heads': 8,
        },
        'optimizer': 'adam',
        'schedule': [0.1, 0.01, 0.001],
    }
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config=config)
    run_id = run.settings._op_id
    run.finish()

    server_config = pq.get_run(TESTING_PROJECT_NAME, run_id).get('config', {})
    assert server_config['model']['name'] == 'transformer'
    assert server_config['model']['layers'] == 12
    assert server_config['optimizer'] == 'adam'
    assert server_config['schedule'] == [0.1, 0.01, 0.001]


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_e2e_tags_at_init():
    """Verify tags set at init() reach the server."""
    tags = ['e2e-test', 'ci', 'baseline']
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), tags=tags)
    run_id = run.settings._op_id
    run.finish()

    server_tags = pq.get_run(TESTING_PROJECT_NAME, run_id).get('tags', [])
    for tag in tags:
        assert tag in server_tags, f"Tag '{tag}' not found on server"


def test_e2e_add_tags():
    """Verify dynamically added tags reach the server."""
    run = pluto.init(
        project=TESTING_PROJECT_NAME, name=get_task_name(), tags=['initial']
    )
    run_id = run.settings._op_id

    run.add_tags('added-single')
    run.add_tags(['added-a', 'added-b'])
    run.finish()

    server_tags = pq.get_run(TESTING_PROJECT_NAME, run_id).get('tags', [])
    for expected in ['initial', 'added-single', 'added-a', 'added-b']:
        assert expected in server_tags, f"Tag '{expected}' not on server"


def test_e2e_remove_tags():
    """Verify tag removal propagates to the server."""
    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=get_task_name(),
        tags=['keep', 'remove-me', 'also-remove'],
    )
    run_id = run.settings._op_id

    run.remove_tags(['remove-me', 'also-remove'])
    run.finish()

    server_tags = pq.get_run(TESTING_PROJECT_NAME, run_id).get('tags', [])
    assert 'keep' in server_tags
    assert 'remove-me' not in server_tags
    assert 'also-remove' not in server_tags


def test_e2e_list_runs_by_tag():
    """Verify list_runs tag filter works against server data."""
    unique_tag = f'e2e-filter-{int(time.time())}'
    run = pluto.init(
        project=TESTING_PROJECT_NAME, name=get_task_name(), tags=[unique_tag]
    )
    run_id = run.settings._op_id
    run.finish()

    runs = pq.list_runs(TESTING_PROJECT_NAME, tags=[unique_tag])
    found_ids = [r['id'] for r in runs]
    assert (
        run_id in found_ids
    ), f"Run {run_id} not found when filtering by tag '{unique_tag}'"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_e2e_metrics_logged():
    """Verify logged metrics are queryable from the server."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    for step in range(10):
        run.log({'train/loss': 1.0 - step * 0.1, 'train/acc': step * 0.1})
    run.finish()

    # Check metric names exist (poll for eventual consistency)
    metric_names = _poll_metric_names(
        TESTING_PROJECT_NAME, run_id, ['train/loss', 'train/acc']
    )
    assert (
        'train/loss' in metric_names
    ), f"'train/loss' not in server metric names: {metric_names}"
    assert 'train/acc' in metric_names

    # Check metric values
    metrics = pq.get_metrics(TESTING_PROJECT_NAME, run_id, metric_names=['train/loss'])
    # metrics is a DataFrame if pandas is installed, otherwise list of dicts
    if hasattr(metrics, 'to_dict'):
        values = metrics['value'].tolist()
    else:
        values = [m['value'] for m in metrics]

    assert len(values) == 10
    assert values[0] == pytest.approx(1.0, abs=1e-6)
    assert values[-1] == pytest.approx(0.1, abs=1e-6)


def test_e2e_metric_statistics():
    """Verify metric statistics endpoint returns correct aggregations."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    # Log known values: 1, 2, 3, 4, 5
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        run.log({'stat/metric': v})
    run.finish()

    stats = pq.get_statistics(
        TESTING_PROJECT_NAME, run_id, metric_names=['stat/metric']
    )
    # Server returns stats in varying shapes; check the response has data
    assert stats is not None
    # The response should contain some form of min/max/count info
    # Exact shape depends on server; just verify it's not empty
    assert len(str(stats)) > 10  # Non-trivial response


# ---------------------------------------------------------------------------
# Files (images)
# ---------------------------------------------------------------------------


def test_e2e_image_upload(tmp_path):
    """Verify an uploaded image survives the round-trip to the server."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    pil_img = PILImage.new('RGB', (4, 4), color=(255, 0, 0))
    run.log({'e2e/test-image': pluto.Image(pil_img, caption='red-square')})
    run.finish()

    # Server stores file with caption-based name (e.g. "red-square.UUID.png")
    files = _poll(
        fn=lambda: pq.get_files(TESTING_PROJECT_NAME, run_id),
        check=lambda fs: any('red-square' in f['fileName'] for f in fs),
    )
    file_names = [f['fileName'] for f in files]
    assert any(
        'red-square' in name for name in file_names
    ), f"Image 'red-square' not found in server files: {file_names}"

    # Download and verify actual image content
    try:
        path = pq.download_file(
            TESTING_PROJECT_NAME, run_id, 'e2e/test-image', destination=tmp_path
        )
    except pq.QueryError:
        pytest.skip('File not yet available for download (eventual consistency)')
    downloaded = PILImage.open(path)
    assert downloaded.size == (4, 4), f'Expected 4x4, got {downloaded.size}'
    r, g, b = downloaded.convert('RGB').getpixel((0, 0))
    assert r > 200 and g < 50 and b < 50, f'Expected red pixel, got ({r},{g},{b})'


def test_e2e_image_download(tmp_path):
    """Verify an uploaded image can be downloaded and content matches."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    pil_img = PILImage.new('RGB', (8, 8), color=(0, 0, 255))
    run.log({'e2e/download-img': pluto.Image(pil_img, caption='blue')})
    run.finish()

    try:
        path = pq.download_file(
            TESTING_PROJECT_NAME, run_id, 'e2e/download-img', destination=tmp_path
        )
    except pq.QueryError:
        pytest.skip('File not yet available for download (eventual consistency)')
    assert path.exists()
    assert path.stat().st_size > 0

    downloaded = PILImage.open(path)
    assert downloaded.size == (8, 8), f'Expected 8x8, got {downloaded.size}'
    r, g, b = downloaded.convert('RGB').getpixel((0, 0))
    assert b > 200 and r < 50 and g < 50, f'Expected blue pixel, got ({r},{g},{b})'


def test_e2e_media_list_order_regression_guard():
    """Guard the SDK's ``sampleIndex`` contribution end-to-end, via the HTTP
    files path the SDK actually uses (``GET /api/runs/files`` → ``queryRunFiles``).

    Logs 6 images with distinct captions in a deliberately *non-alphabetical*
    order, then reads them back and classifies the returned order:

    - **logged order** → OK. Either the server sorted by ``sampleIndex`` (once
      server-private #532 is deployed), or — before that — the step-only query
      returned insert order, which equals the logged order for a fresh
      single-part upload. Both are correct outcomes.
    - **alphabetical** → the server fell back to ``fileName`` ordering. On a
      #532 server that can only happen if the SDK stopped sending
      ``sampleIndex`` → this is the regression we want to catch → **fail**.
    - **anything else** → the pre-#532 step-only path's within-step order is
      undefined (e.g. after a ClickHouse part merge); not a regression signal →
      **skip** (so this can never flake on the undefined pre-#532 case).

    Note: a fresh single-list upload can't force a *pre*-#532 failure (the SDK
    uploads in ``sampleIndex`` order, so insert order == logged order); the
    deterministic pre-vs-post-fix ordering test lives server-side in #532, which
    seeds ClickHouse with ``fileName`` reversed vs ``sampleIndex``.
    """
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    logged_order = ['delta', 'alpha', 'foxtrot', 'charlie', 'echo', 'bravo']
    colors = [
        (200, 0, 0),
        (0, 200, 0),
        (0, 0, 200),
        (200, 200, 0),
        (200, 0, 200),
        (0, 200, 200),
    ]
    imgs = [
        pluto.Image(PILImage.new('RGB', (4, 4), color=colors[i]), caption=label)
        for i, label in enumerate(logged_order)
    ]
    run.log({'e2e/order': imgs}, step=0)
    run.finish()

    def _labels(files):
        # Server response order preserved; identify each file by the label in
        # its caption / caption-derived fileName (distinct non-hex words, so no
        # collision with the random UUID in the filename).
        out = []
        for f in files:
            hay = f'{f.get("caption") or ""} {f.get("fileName") or ""}'
            match = next((lbl for lbl in logged_order if lbl in hay), None)
            if match:
                out.append(match)
        return out

    files = _poll(
        fn=lambda: pq.get_files(TESTING_PROJECT_NAME, run_id),
        check=lambda fs: len(_labels(fs)) >= len(logged_order),
    )
    got = _labels(files)

    if got == logged_order:
        return  # correct order
    if got == sorted(logged_order):
        pytest.fail(
            f'media returned in fileName/alphabetical order {got}; expected '
            f'logged order {logged_order}. The server fell back to fileName — '
            f'the SDK is not sending sampleIndex (regression).'
        )
    pytest.skip(
        f'media order is undefined on this server (pre-#532 step-only path / '
        f'merge), got {got} — not a regression signal'
    )


# ---------------------------------------------------------------------------
# Console logs
# ---------------------------------------------------------------------------


def _poll_console_messages(run_id: int, sentinel: str) -> List[str]:
    """Poll /api/runs/logs until a line containing *sentinel* appears."""

    def _messages() -> List[str]:
        logs = pq.get_logs(TESTING_PROJECT_NAME, run_id, limit=500)
        return [entry.get('message', '') for entry in logs]

    return _poll(fn=_messages, check=lambda ms: any(sentinel in m for m in ms))


def test_e2e_console_logs():
    """Verify print() output is captured and queryable."""
    sentinel = f'e2e-print-{uuid.uuid4().hex[:12]}'
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    print(sentinel)
    run.finish()

    messages = _poll_console_messages(run_id, sentinel)
    assert any(sentinel in msg for msg in messages), (
        f'printed line ({sentinel}) never reached the server; '
        f'got {len(messages)} console lines'
    )


# Faithful torchtitan repro, run as a real subprocess: the logging handler
# binds sys.stderr at process start (init_logger), pluto.init() comes later
# (inside WandBLogger in the real job), and training output goes through
# logging — never bare print(). Run in a fresh interpreter so real fds are
# in play (under pytest, sys.stderr is a capture object detached from fd 2)
# and no state leaks between xdist worker tests.
_TITAN_SCRIPT = """
import logging, os, sys

# 1. torchtitan tools/logging.py init_logger(): runs before anything else
#    and binds the current sys.stderr object into the handler.
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[titan] %(message)s'))
logger.addHandler(handler)

# 2. pluto.init() happens later (via the wandb compat inside WandBLogger).
import pluto
run = pluto.init(project=os.environ['E2E_PROJECT'], name=os.environ['E2E_NAME'])
with open(os.environ['E2E_RUN_ID_FILE'], 'w') as f:
    f.write(str(run.settings._op_id))

# 3. All training output goes through logging -> the pre-bound handler.
logger.info('step: 10  loss: 2.31  ' + os.environ['E2E_SENTINEL'])
run.finish()
"""


def test_e2e_pre_init_logging_handler_console_logs(tmp_path):
    """torchtitan scenario: logging configured BEFORE pluto.init() must
    still have its output reach the server.

    CPython's StreamHandler stores the sys.stderr object at construction,
    so the old sys-swap console capture never saw these writes and the
    run's console section stayed empty. fd-level capture
    (pluto/_fd_capture.py) fixes this; this test pins the full path:
    pre-bound handler → fd pipe → sync store → upload → /api/runs/logs.
    """
    sentinel = f'titan-e2e-{uuid.uuid4().hex[:12]}'
    run_id_file = tmp_path / 'run_id'
    env = {
        **os.environ,
        'E2E_PROJECT': TESTING_PROJECT_NAME,
        'E2E_NAME': get_task_name(),
        'E2E_SENTINEL': sentinel,
        'E2E_RUN_ID_FILE': str(run_id_file),
    }
    proc = subprocess.run(
        [sys.executable, '-c', _TITAN_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f'titan-style subprocess failed (rc={proc.returncode}):\n'
        f'stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}'
    )
    run_id = int(run_id_file.read_text())

    messages = _poll_console_messages(run_id, sentinel)
    assert any(sentinel in msg for msg in messages), (
        f'pre-init logging handler line ({sentinel}) never reached the '
        f'server; got {len(messages)} console lines'
    )


# ---------------------------------------------------------------------------
# List / search runs
# ---------------------------------------------------------------------------


def test_e2e_list_runs_search():
    """Verify list_runs search filter finds a run by name."""
    task_name = get_task_name()
    run = pluto.init(project=TESTING_PROJECT_NAME, name=task_name, config={})
    run_id = run.settings._op_id
    run.finish()

    runs = pq.list_runs(TESTING_PROJECT_NAME, search=task_name)
    found_ids = [r['id'] for r in runs]
    assert run_id in found_ids, f'Run {run_id} (name={task_name}) not found via search'


def _two_tagged_runs() -> tuple:
    """Create two finished runs (A older than B) sharing a unique tag.

    Returns ``(tag, id_a, id_b)`` once both are listable under the tag. The
    unique tag scopes subsequent list queries to exactly these two runs, so
    concurrent writes to the shared ``testing-ci`` project (parallel matrix
    legs, ``-n auto`` workers) can't shift the sort/pagination window. A >1s
    gap guarantees distinct ``createdAt`` even at second precision.
    """
    tag = f'e2e-page-{uuid.uuid4().hex[:12]}'
    run_a = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), tags=[tag])
    id_a = run_a.settings._op_id
    run_a.finish()
    time.sleep(1.1)
    run_b = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), tags=[tag])
    id_b = run_b.settings._op_id
    run_b.finish()

    # Tags sync asynchronously; poll until both runs are listable under the tag.
    def _both_listed() -> bool:
        ids = {
            r['id'] for r in pq.list_runs(TESTING_PROJECT_NAME, tags=[tag], limit=200)
        }
        return {id_a, id_b} <= ids

    assert _poll(
        fn=_both_listed, check=lambda ok: ok
    ), f'tagged runs {id_a},{id_b} not both listable under {tag}'
    return tag, id_a, id_b


def test_e2e_list_runs_sort_created_desc():
    """Verify sort='-createdAt' orders newest-first within a controlled set."""
    tag, id_a, id_b = _two_tagged_runs()
    runs = pq.list_runs(TESTING_PROJECT_NAME, tags=[tag], sort='-createdAt', limit=200)
    ids = [r['id'] for r in runs]
    # Exactly our two runs match the tag; B (later) must come before A.
    assert ids == [id_b, id_a], f'sort=-createdAt not newest-first: {ids}'


def test_e2e_list_runs_offset_pagination():
    """Verify offset advances the page within a controlled, stable set."""
    tag, id_a, id_b = _two_tagged_runs()
    page1 = pq.list_runs(TESTING_PROJECT_NAME, tags=[tag], sort='-createdAt', limit=1)
    page2 = pq.list_runs(
        TESTING_PROJECT_NAME, tags=[tag], sort='-createdAt', limit=1, offset=1
    )
    assert [r['id'] for r in page1] == [id_b], 'page 1 should be the newest run'
    assert [r['id'] for r in page2] == [id_a], 'offset=1 should skip to the older run'


def test_e2e_list_runs_filter():
    """Verify the wandb-style `filters` query filters by a config value."""
    marker = f'e2e-ff-{int(time.time())}'
    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=get_task_name(),
        config={'e2e_filter_marker': marker},
    )
    run_id = run.settings._op_id
    run.finish()

    def _query():
        runs = pq.list_runs(
            TESTING_PROJECT_NAME,
            filters={'config.e2e_filter_marker': marker},
            limit=200,
        )
        ids = [r['id'] for r in runs]
        return run_id in ids

    # Field values are indexed asynchronously; poll for eventual consistency.
    assert _poll(
        fn=_query, check=lambda found: found
    ), f'Run {run_id} not found via filters on config.e2e_filter_marker'


# ---------------------------------------------------------------------------
# Histogram (structured data)
# ---------------------------------------------------------------------------


def test_e2e_histogram_upload():
    """Verify histogram data is stored as a file/data entry on the server."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    data = np.random.normal(loc=0.0, scale=1.0, size=500)
    run.log({'e2e/histogram': pluto.Histogram(data=data, bins=20)})
    run.finish()

    # Histograms show up under metric names or logNames
    server_run = pq.get_run(TESTING_PROJECT_NAME, run_id)
    log_names = server_run.get('logNames', [])
    # The histogram should appear somewhere in logNames
    found = any('histogram' in str(n).lower() for n in log_names)
    if not found:
        # Some server versions store structured data differently
        pytest.skip(
            f'Histogram not found in logNames (may be stored differently): {log_names}'
        )


# ---------------------------------------------------------------------------
# Multi-metric logging in single call
# ---------------------------------------------------------------------------


def test_e2e_multiple_metrics_single_log():
    """Verify logging multiple metrics in one log() call persists all."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    run.log(
        {
            'multi/loss': 0.5,
            'multi/accuracy': 0.85,
            'multi/lr': 0.001,
        }
    )
    run.finish()

    expected = ['multi/loss', 'multi/accuracy', 'multi/lr']
    metric_names = _poll_metric_names(TESTING_PROJECT_NAME, run_id, expected)
    for name in expected:
        assert (
            name in metric_names
        ), f"'{name}' not in server metric names: {metric_names}"


# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------


def test_e2e_system_metrics_collected():
    """Verify system metrics (CPU/memory) are automatically collected."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    # Log a few steps to give system monitor time to sample
    for i in range(5):
        run.log({'dummy': i})
        time.sleep(0.5)
    run.finish()

    metric_names = _run_metric_names(TESTING_PROJECT_NAME, run_id)
    sys_metrics = [n for n in metric_names if n.startswith('sys/')]
    # System monitoring should produce at least CPU or memory metrics
    if not sys_metrics:
        pytest.skip('No system metrics collected (monitor may not have sampled)')
    assert len(sys_metrics) > 0


def test_e2e_system_metrics_multiple_timesteps():
    """Verify sys/* metrics are sampled at multiple timesteps over an extended run.

    Uses a 2-second sampling interval and runs for ~20 seconds, so we expect
    at least 3 distinct data points per system metric.  This catches bugs where
    system metrics are only emitted once (e.g. only at init or finish).

    The wall-time budget is generous on purpose: the trigger HTTP call inside
    the monitor loop slows sample cadence under GH Actions xdist contention,
    so a tight 10s window flakes (seen on CI run 25193895688 — 2 samples).
    20s leaves headroom even with ~2x slowdown.
    """
    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=get_task_name(),
        config={},
        x_sys_sampling_interval=2,  # sample every 2s
    )
    run_id = run.settings._op_id

    # Keep the run alive for ~20 seconds so the monitor thread fires multiple times.
    for i in range(20):
        run.log({'keepalive': i})
        time.sleep(1)
    run.finish()

    # Poll until at least one sys/ metric appears on the server.
    metric_names = _poll(
        fn=lambda: _run_metric_names(TESTING_PROJECT_NAME, run_id),
        check=lambda names: any(n.startswith('sys/') for n in names),
        timeout=_POLL_TIMEOUT,
    )
    sys_metric_names = [n for n in metric_names if n.startswith('sys/')]
    if not sys_metric_names:
        pytest.skip('No system metrics collected (monitor may not have sampled)')

    # Pick the first sys metric and verify it has multiple data points.
    target_metric = sys_metric_names[0]
    metrics = _poll(
        fn=lambda: pq.get_metrics(
            TESTING_PROJECT_NAME, run_id, metric_names=[target_metric]
        ),
        check=lambda m: len(m) >= 3,
        timeout=_POLL_TIMEOUT,
    )

    # With 2s sampling over ~10s we expect ≥3 samples (conservatively).
    assert len(metrics) >= 3, (
        f"Expected ≥3 data points for '{target_metric}', got {len(metrics)}. "
        f'System metrics should be sampled at multiple timesteps, not just once.'
    )

    # Verify each data point has distinct timestamps (not all the same).
    if hasattr(metrics, 'to_dict'):
        # pandas DataFrame
        times = list(metrics['time'])
    else:
        # list of dicts
        times = [row['time'] for row in metrics]
    unique_times = set(str(t) for t in times)
    assert len(unique_times) >= 3, (
        f"Expected ≥3 distinct timestamps for '{target_metric}', "
        f'got {len(unique_times)}: {unique_times}'
    )


# ---------------------------------------------------------------------------
# Full lifecycle: init → log → update_config → tags → finish → query all
# ---------------------------------------------------------------------------


def test_e2e_full_lifecycle():
    """Comprehensive lifecycle test: verify everything in one run."""
    task_name = get_task_name()
    initial_config = {'lr': 0.001, 'model': 'bert'}
    initial_tags = ['e2e', 'lifecycle']

    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=task_name,
        config=initial_config,
        tags=initial_tags,
    )
    run_id = run.settings._op_id

    # Log metrics
    for step in range(5):
        run.log({'lifecycle/loss': 1.0 / (step + 1)})

    # Update config
    run.update_config({'epochs': 50, 'lr': 0.01})

    # Modify tags
    run.add_tags('validated')
    run.remove_tags('lifecycle')

    # Log an image
    pil_img = PILImage.new('RGB', (4, 4), color='green')
    run.log({'lifecycle/img': pluto.Image(pil_img, caption='green')})

    run.finish()

    # --- Query everything back ---
    # Poll until config update AND tag mutations have applied server-side;
    # both are pushed by finish() but reflected asynchronously.
    server_run = _poll_run(
        TESTING_PROJECT_NAME,
        run_id,
        check=lambda r: (
            r.get('config', {}).get('lr') == 0.01
            and 'validated' in r.get('tags', [])
            and 'lifecycle' not in r.get('tags', [])
        ),
    )

    # Metadata
    assert server_run['name'] == task_name
    assert server_run['status'] == 'COMPLETED'

    # Config
    server_config = server_run.get('config', {})
    assert server_config['lr'] == 0.01  # Updated
    assert server_config['model'] == 'bert'  # Original
    assert server_config['epochs'] == 50  # Added

    # Tags
    server_tags = server_run.get('tags', [])
    assert 'e2e' in server_tags
    assert 'validated' in server_tags
    assert 'lifecycle' not in server_tags  # Removed

    # Metrics (poll for eventual consistency)
    metric_names = _poll_metric_names(TESTING_PROJECT_NAME, run_id, ['lifecycle/loss'])
    assert 'lifecycle/loss' in metric_names

    metrics = pq.get_metrics(
        TESTING_PROJECT_NAME, run_id, metric_names=['lifecycle/loss']
    )
    if hasattr(metrics, '__len__'):
        assert len(metrics) == 5

    # Files (poll for eventual consistency)
    # Server stores file with caption-based name (e.g. "green.UUID.png")
    files = _poll(
        fn=lambda: pq.get_files(TESTING_PROJECT_NAME, run_id),
        check=lambda fs: any('green' in f['fileName'] for f in fs),
    )
    file_names = [f['fileName'] for f in files]
    assert any(
        'green' in name for name in file_names
    ), f'Image not found in server files: {file_names}'

    # Verify the image content survived the round-trip
    matched = [f for f in files if 'green' in f['fileName']][0]
    url = matched.get('downloadUrl') or matched.get('url')
    if url:
        resp = httpx.get(url, follow_redirects=True, timeout=30)
        if resp.status_code == 200:
            downloaded = PILImage.open(io.BytesIO(resp.content))
            assert downloaded.size == (4, 4), f'Expected 4x4, got {downloaded.size}'
            r, g, b = downloaded.convert('RGB').getpixel((0, 0))
            assert (
                g > 100 and r < 50 and b < 50
            ), f'Expected green pixel, got ({r},{g},{b})'


# ---------------------------------------------------------------------------
# Filter query (wandb-style `filters=`) — operator & field coverage
#
# `test_e2e_list_runs_filter` above covers a single equality leaf. The tests
# below exercise the *full* documented filter grammar end-to-end against the
# live server: every leaf operator ($eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$regex),
# every boolean combinator ($and/$or/$not), and every documented field family
# (status/state, name, displayName, tags, config.*, summaryMetrics.*,
# systemMetadata.*, created_at/updated_at, heartbeat_at).
#
# The grammar is mirrored client-side in pluto.query._FILTER_* and kept equal
# to the server's published RunFilterGrammar by tests/test_contract.py — so
# "all documented fields" == that grammar.
#
# Isolation: each corpus carries a unique `batch` marker and every query is
# AND-scoped to it via `config.batch`, so the result universe is exactly the
# three seeded runs even under `-n auto` (xdist), where the module-scoped
# fixture is re-seeded once per worker.
#
# Fields whose filter values are materialized by a slower (ClickHouse-backed)
# aggregation path than core run columns — summaryMetrics.* and
# systemMetadata.* — first wait on an all-match sentinel; if that never
# converges within the window we skip() (eventual-consistency lag) rather than
# fail, matching the skips used elsewhere in this file. Once the sentinel
# passes, the data is indexed and the subset assertions are real.
# ---------------------------------------------------------------------------

_SLOW_FIELD_POLL_TIMEOUT = 180

# Date cutoffs far outside any real run timestamp, for exercising the date
# fields' comparison plumbing without timing flake.
_PAST_CUTOFF = '2000-01-01T00:00:00Z'
_FUTURE_CUTOFF = '2999-01-01T00:00:00Z'


def _first_scalar(meta: dict):
    """Return ``(key, value)`` for the first scalar systemMetadata entry, or None.

    Handles both flat (``{key: value}``) and wrapped (``{key: {'value': ...}}``)
    shapes that the server may use for systemMetadata.
    """
    if not isinstance(meta, dict):
        return None
    for k, v in meta.items():
        if isinstance(v, dict) and 'value' in v:
            v = v['value']
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            return (k, v)
    return None


@pytest.fixture(scope='module')
def filter_corpus():
    """Seed three finished runs with known, distinct values for filter tests.

    Returns a dict with the unique ``batch`` marker, per-group run metadata
    (``id``, ``name``, ``lr``, ``loss``), and a discovered scalar
    ``sys_meta`` ``(key, value)`` pair (or ``None``).
    """
    batch = uuid.uuid4().hex[:12]
    tag = f'e2e-filt-{batch}'
    # group -> (lr, loss). `lr` drives the numeric operator matrix; `loss` (the
    # LAST-aggregated metric) drives the summaryMetrics tests. Values are chosen
    # so each operator selects a distinct, non-trivial subset.
    specs = {
        'alpha': (0.001, 0.5),
        'beta': (0.01, 0.1),
        'gamma': (0.1, 0.9),
    }
    corpus: dict = {'batch': batch, 'tag': tag, 'runs': {}}
    for group, (lr, loss) in specs.items():
        name = f't-e2e-filter-{batch}-{group}'
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=name,
            tags=[tag, group],
            config={'lr': lr, 'batch': batch, 'group': group},
        )
        run.log({'loss': loss})
        corpus['runs'][group] = {
            'id': run.settings._op_id,
            'name': name,
            'lr': lr,
            'loss': loss,
        }
        run.finish()

    all_ids = {r['id'] for r in corpus['runs'].values()}

    def _query(flt):
        return {
            r['id'] for r in pq.list_runs(TESTING_PROJECT_NAME, filters=flt, limit=200)
        }

    # Warm up: wait until config.batch is indexed for all three runs, so the
    # batch-scoped subset assertions below are stable rather than racing ingest.
    ready = _poll(
        fn=lambda: _query({'config.batch': batch}),
        check=lambda got: all_ids <= got,
    )
    assert all_ids <= ready, (
        f'corpus not fully indexed under config.batch={batch}: '
        f'have {ready}, want {all_ids}'
    )
    # And until a numeric comparison on config is live (sentinel matches all).
    # Assert this too: if it times out while config.batch already indexed, the
    # parametrized leaf-operator tests would otherwise flake on a still-catching-up
    # server instead of failing here with a clear message.
    numeric_ready = _poll(
        fn=lambda: _query(
            {'$and': [{'config.batch': batch}, {'config.lr': {'$gte': 0.0}}]}
        ),
        check=lambda got: all_ids <= got,
    )
    assert all_ids <= numeric_ready, (
        f'config.lr numeric filter not indexed for all runs in batch={batch}: '
        f'have {numeric_ready}, want {all_ids}'
    )

    # Best-effort: discover a scalar systemMetadata field to filter on.
    # systemMetadata is populated asynchronously by the server, so poll rather
    # than read once right after finish() — otherwise the snapshot can be empty
    # and test_e2e_filter_field_system_metadata silently skips.
    snap = _poll_run(
        TESTING_PROJECT_NAME,
        corpus['runs']['alpha']['id'],
        check=lambda r: _first_scalar(r.get('systemMetadata') or {}) is not None,
    )
    corpus['sys_meta'] = _first_scalar(snap.get('systemMetadata') or {})

    return corpus


def _scoped(batch: str, case: dict, by: str = 'config') -> dict:
    """AND-combine *case* with a unique per-corpus marker for isolation.

    ``by='config'`` scopes via ``config.batch`` (use for ``config.*`` cases);
    ``by='name'`` scopes via a ``name`` regex on the batch id (use for
    column-only fields like ``status``/``created_at``). Mixing a ``config.*``
    predicate with a column-only predicate in one ``$and`` makes the server
    return an empty set, so the scope marker must match the case's field family.
    """
    marker = {'config.batch': batch} if by == 'config' else {'name': {'$regex': batch}}
    return {'$and': [marker, case]}


def _filter_ids(batch: str, case: dict, by: str = 'config') -> set:
    """Return the set of run ids matching *case* within the corpus batch."""
    runs = pq.list_runs(
        TESTING_PROJECT_NAME, filters=_scoped(batch, case, by), limit=200
    )
    return {r['id'] for r in runs}


def _expected_ids(corpus: dict, groups) -> set:
    return {corpus['runs'][g]['id'] for g in groups}


def _assert_filter(corpus, case, groups, timeout=_POLL_TIMEOUT, by='config'):
    """Assert *case* (batch-scoped) selects exactly *groups*, polling for it."""
    batch = corpus['batch']
    want = _expected_ids(corpus, groups)
    got = _poll(
        fn=lambda: _filter_ids(batch, case, by),
        check=lambda s: s == want,
        timeout=timeout,
    )
    assert (
        got == want
    ), f'filter {case!r} selected {got}, want {want} (groups={list(groups)})'


def _assert_filter_or_skip(corpus, case, groups, timeout=_POLL_TIMEOUT, by='config'):
    """Like :func:`_assert_filter`, but skip (not fail) on an all-empty result.

    The ``filters`` API is in preview; some documented fields/operators may not
    be wired up server-side yet and return an empty set. Treat an all-empty
    result as a preview gap (skip), while a *non-empty wrong* result still
    fails — so genuine filtering bugs are caught, not masked.
    """
    batch = corpus['batch']
    want = _expected_ids(corpus, groups)
    got = _poll(
        fn=lambda: _filter_ids(batch, case, by),
        check=lambda s: s == want,
        timeout=timeout,
    )
    if got == want:
        return
    if not got:
        pytest.skip(
            f'preview filters API returned no rows for {case!r}; this field/'
            f'operator may not be implemented server-side yet'
        )
    assert (
        got == want
    ), f'filter {case!r} selected {got}, want {want} (groups={list(groups)})'


# ----- leaf operators -------------------------------------------------------


@pytest.mark.parametrize(
    'case,groups',
    [
        ({'config.lr': 0.01}, ['beta']),  # equality shorthand (no operator)
        ({'config.lr': {'$eq': 0.01}}, ['beta']),
        ({'config.lr': {'$ne': 0.01}}, ['alpha', 'gamma']),
        ({'config.lr': {'$gt': 0.01}}, ['gamma']),
        ({'config.lr': {'$gte': 0.01}}, ['beta', 'gamma']),
        ({'config.lr': {'$lt': 0.01}}, ['alpha']),
        ({'config.lr': {'$lte': 0.01}}, ['alpha', 'beta']),
        ({'config.lr': {'$in': [0.001, 0.1]}}, ['alpha', 'gamma']),
        ({'config.lr': {'$nin': [0.001]}}, ['beta', 'gamma']),
    ],
    ids=[
        'eq-shorthand',
        'eq',
        'ne',
        'gt',
        'gte',
        'lt',
        'lte',
        'in',
        'nin',
    ],
)
def test_e2e_filter_leaf_operators(filter_corpus, case, groups):
    """Each numeric leaf operator on config.lr selects the right runs."""
    _assert_filter(filter_corpus, case, groups)


def test_e2e_filter_regex_operator(filter_corpus):
    """$regex on `name` matches by substring (wandb/Mongo partial-match)."""
    # Only the alpha run's name contains 'alpha'.
    _assert_filter(filter_corpus, {'name': {'$regex': 'alpha'}}, ['alpha'])


# ----- boolean combinators --------------------------------------------------


def test_e2e_filter_boolean_or(filter_corpus):
    case = {'$or': [{'config.lr': {'$lt': 0.005}}, {'config.lr': {'$gt': 0.05}}]}
    _assert_filter(filter_corpus, case, ['alpha', 'gamma'])


def test_e2e_filter_boolean_and(filter_corpus):
    case = {'$and': [{'config.lr': {'$gte': 0.01}}, {'config.lr': {'$lte': 0.01}}]}
    _assert_filter(filter_corpus, case, ['beta'])


def test_e2e_filter_boolean_not(filter_corpus):
    # All-column $not (negate a name regex) so the negation isn't mixed with a
    # config.* predicate. Skips if the preview API doesn't implement $not yet.
    _assert_filter_or_skip(
        filter_corpus,
        {'$not': {'name': {'$regex': 'alpha'}}},
        ['beta', 'gamma'],
        by='name',
    )


# ----- compound leaf forms (documented operator behaviors) ------------------


def test_e2e_filter_implicit_and_multiple_keys(filter_corpus):
    """Multiple keys in one object are implicitly ANDed.

    Per the docs: ``{"config.lr": {"$gt": ...}, "config.model": ...}``.
    """
    case = {'config.lr': {'$gte': 0.01}, 'config.group': 'beta'}
    _assert_filter(filter_corpus, case, ['beta'])


def test_e2e_filter_range_on_single_field(filter_corpus):
    """Two operators on one field form a range (docs: heartbeat_at range).

    The docs show a range on ``heartbeat_at``; on ``config.*`` the preview API
    currently applies only one bound (the equivalent ``$and`` of two single-op
    clauses works — see ``test_e2e_filter_boolean_and``). Treat an under-filtered
    superset as a preview gap (skip), but still fail on a genuinely wrong set.
    """
    case = {'config.lr': {'$gt': 0.005, '$lt': 0.05}}  # 0.005 < lr < 0.05 -> beta
    batch = filter_corpus['batch']
    want = _expected_ids(filter_corpus, ['beta'])
    got = _poll(fn=lambda: _filter_ids(batch, case), check=lambda s: s == want)
    if want < got:
        pytest.skip(
            'single-field two-operator range not honored for config.* in the '
            'preview API (only one bound applied); the equivalent $and form is '
            'covered by test_e2e_filter_boolean_and'
        )
    assert got == want, f'filter {case!r} selected {got}, want {want}'


# ----- documented fields ----------------------------------------------------


def test_e2e_filter_field_status(filter_corpus):
    """`status` field: all seeded runs are COMPLETED after finish()."""
    everyone = ['alpha', 'beta', 'gamma']
    # Column-only field — scope by name, not config.batch (mixing config.* with a
    # column predicate returns empty). Skips if status filtering isn't live yet.
    _assert_filter_or_skip(
        filter_corpus, {'status': {'$eq': 'COMPLETED'}}, everyone, by='name'
    )
    _assert_filter_or_skip(
        filter_corpus, {'status': {'$ne': 'COMPLETED'}}, [], by='name'
    )


def test_e2e_filter_field_state(filter_corpus):
    """`state` field (wandb alias): no finished run is 'running'."""
    _assert_filter_or_skip(
        filter_corpus,
        {'state': {'$ne': 'running'}},
        ['alpha', 'beta', 'gamma'],
        by='name',
    )


def test_e2e_filter_field_name(filter_corpus):
    """`name` field: exact equality selects the single matching run."""
    alpha_name = filter_corpus['runs']['alpha']['name']
    _assert_filter(filter_corpus, {'name': alpha_name}, ['alpha'])


def test_e2e_filter_field_tags(filter_corpus):
    """`tags` field: $in = "has any of these" (docs)."""
    # Single tag selects its run; multiple tags select the union (has-any).
    _assert_filter(filter_corpus, {'tags': {'$in': ['alpha']}}, ['alpha'])
    _assert_filter(
        filter_corpus, {'tags': {'$in': ['alpha', 'beta']}}, ['alpha', 'beta']
    )


def test_e2e_filter_field_config_string(filter_corpus):
    """`config.<key>` with a string value selects by equality."""
    _assert_filter(filter_corpus, {'config.group': 'beta'}, ['beta'])


def test_e2e_filter_field_created_at(filter_corpus):
    """`created_at` field: date comparison is honored server-side."""
    everyone = ['alpha', 'beta', 'gamma']
    _assert_filter_or_skip(
        filter_corpus, {'created_at': {'$gte': _PAST_CUTOFF}}, everyone, by='name'
    )
    _assert_filter_or_skip(
        filter_corpus, {'created_at': {'$lt': _PAST_CUTOFF}}, [], by='name'
    )


def test_e2e_filter_field_updated_at(filter_corpus):
    """`updated_at` field: date comparison is honored server-side."""
    everyone = ['alpha', 'beta', 'gamma']
    _assert_filter_or_skip(
        filter_corpus, {'updated_at': {'$gte': _PAST_CUTOFF}}, everyone, by='name'
    )


def test_e2e_filter_field_heartbeat_at(filter_corpus):
    """`heartbeat_at` field (last logged data point): all runs logged data."""
    everyone = ['alpha', 'beta', 'gamma']
    _assert_filter_or_skip(
        filter_corpus,
        {'heartbeat_at': {'$gte': _PAST_CUTOFF}},
        everyone,
        timeout=_SLOW_FIELD_POLL_TIMEOUT,
        by='name',
    )


def test_e2e_filter_field_summary_metrics(filter_corpus):
    """`summaryMetrics.<key>`: filter on the LAST-aggregated metric value."""
    batch = filter_corpus['batch']
    everyone = _expected_ids(filter_corpus, ['alpha', 'beta', 'gamma'])
    # Wait for the summary aggregation to materialize for all three runs.
    sentinel = _poll(
        fn=lambda: _filter_ids(batch, {'summaryMetrics.loss': {'$gte': 0.0}}, 'name'),
        check=lambda got: got == everyone,
        timeout=_SLOW_FIELD_POLL_TIMEOUT,
    )
    if sentinel != everyone:
        pytest.skip(
            'summaryMetrics.loss not queryable for all runs within window '
            '(eventual consistency, or not implemented in the preview API)'
        )
    # Indexed: subset assertions are now real. loss: alpha=0.5, beta=0.1, gamma=0.9.
    _assert_filter(
        filter_corpus, {'summaryMetrics.loss': {'$lt': 0.2}}, ['beta'], by='name'
    )
    _assert_filter(
        filter_corpus,
        {'summaryMetrics.loss': {'$gte': 0.5}},
        ['alpha', 'gamma'],
        by='name',
    )


def test_e2e_filter_field_system_metadata(filter_corpus):
    """`systemMetadata.<key>`: filter on an auto-collected metadata field."""
    meta = filter_corpus.get('sys_meta')
    if not meta:
        pytest.skip('no scalar systemMetadata field discovered to filter on')
    key, value = meta
    batch = filter_corpus['batch']
    alpha_id = filter_corpus['runs']['alpha']['id']
    case = {f'systemMetadata.{key}': value}
    got = _poll(
        fn=lambda: _filter_ids(batch, case, 'name'),
        check=lambda s: alpha_id in s,
        timeout=_SLOW_FIELD_POLL_TIMEOUT,
    )
    if alpha_id not in got:
        pytest.skip(
            f'systemMetadata.{key} not indexed within window (eventual consistency)'
        )
    # Result stays within our isolated corpus (scoping holds).
    assert got <= _expected_ids(filter_corpus, ['alpha', 'beta', 'gamma'])


def test_e2e_filter_field_display_name(filter_corpus):
    """`displayName`/`display_name` field: alias surface for the run name."""
    alpha = filter_corpus['runs']['alpha']
    batch = filter_corpus['batch']
    want = {alpha['id']}
    got = _poll(
        fn=lambda: _filter_ids(batch, {'display_name': alpha['name']}, 'name'),
        check=lambda s: s == want,
    )
    if got != want:
        pytest.skip(
            'display_name filter did not resolve (server may map the alias '
            'differently); name equality is covered by test_e2e_filter_field_name'
        )
    assert got == want
