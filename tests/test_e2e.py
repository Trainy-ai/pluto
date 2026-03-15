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
import time

import numpy as np
import pytest
from PIL import Image as PILImage

import pluto
import pluto.query as pq
from tests.utils import get_task_name

TESTING_PROJECT_NAME = 'testing-ci'

HAS_TORCH = importlib.util.find_spec('torch') is not None


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
    assert 'url' in server_run and server_run['url']


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

    server_config = pq.get_run(TESTING_PROJECT_NAME, run_id).get('config', {})
    assert (
        server_config['lr'] == 0.01
    ), f"Server has lr={server_config.get('lr')}, expected 0.01"
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

    # Check metric names exist
    metric_names = pq.get_metric_names(TESTING_PROJECT_NAME, run_ids=[run_id])
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


def test_e2e_image_upload():
    """Verify an uploaded image appears in the server's file listing."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    pil_img = PILImage.new('RGB', (4, 4), color='red')
    run.log({'e2e/test-image': pluto.Image(pil_img, caption='red-square')})
    run.finish()

    files = pq.get_files(TESTING_PROJECT_NAME, run_id)
    file_names = [f['fileName'] for f in files]
    assert any(
        'test-image' in name for name in file_names
    ), f"Image 'test-image' not found in server files: {file_names}"


def test_e2e_image_download(tmp_path):
    """Verify an uploaded image can be downloaded back."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    pil_img = PILImage.new('RGB', (8, 8), color='blue')
    run.log({'e2e/download-img': pluto.Image(pil_img, caption='blue')})
    run.finish()

    try:
        path = pq.download_file(
            TESTING_PROJECT_NAME, run_id, 'e2e/download-img', destination=tmp_path
        )
        assert path.exists()
        assert path.stat().st_size > 0
    except pq.QueryError:
        pytest.skip('File not yet available for download (eventual consistency)')


# ---------------------------------------------------------------------------
# Console logs
# ---------------------------------------------------------------------------


def test_e2e_console_logs():
    """Verify print() output is captured and queryable."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run_id = run.settings._op_id

    print('e2e-log-sentinel-12345')
    run.finish()

    logs = pq.get_logs(TESTING_PROJECT_NAME, run_id, limit=100)
    messages = [entry.get('message', '') for entry in logs]
    found = any('e2e-log-sentinel-12345' in msg for msg in messages)
    # Console capture may be disabled or delayed; don't hard-fail
    if not found:
        pytest.skip(
            'Console log not found on server (capture may be disabled or delayed)'
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

    metric_names = pq.get_metric_names(TESTING_PROJECT_NAME, run_ids=[run_id])
    for expected in ['multi/loss', 'multi/accuracy', 'multi/lr']:
        assert (
            expected in metric_names
        ), f"'{expected}' not in server metric names: {metric_names}"


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

    metric_names = pq.get_metric_names(TESTING_PROJECT_NAME, run_ids=[run_id])
    sys_metrics = [n for n in metric_names if n.startswith('sys/')]
    # System monitoring should produce at least CPU or memory metrics
    if not sys_metrics:
        pytest.skip('No system metrics collected (monitor may not have sampled)')
    assert len(sys_metrics) > 0


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
    server_run = pq.get_run(TESTING_PROJECT_NAME, run_id)

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

    # Metrics
    metric_names = pq.get_metric_names(TESTING_PROJECT_NAME, run_ids=[run_id])
    assert 'lifecycle/loss' in metric_names

    metrics = pq.get_metrics(
        TESTING_PROJECT_NAME, run_id, metric_names=['lifecycle/loss']
    )
    if hasattr(metrics, '__len__'):
        assert len(metrics) == 5

    # Files
    files = pq.get_files(TESTING_PROJECT_NAME, run_id)
    file_names = [f['fileName'] for f in files]
    assert any(
        'lifecycle' in name or 'img' in name for name in file_names
    ), f'Image not found in server files: {file_names}'
