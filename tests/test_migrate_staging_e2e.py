"""
Client<->server integration test for the migration pipeline, run against
the STAGING (dev) environment.

Loads a hand-staged export (no wandb involved) into a real Pluto server
via PlutoLoader and verifies through the query API that the historical
data actually round-tripped: metric values/steps, original per-point
wall-clock timestamps, tags, and — once the server-side fix is deployed
— the run's backfilled createdAt.

Gated on PLUTO_STAGING_API_KEY so ordinary CI (which only has a
production PLUTO_API_KEY) skips it:

    PLUTO_STAGING_API_KEY=... poetry run pytest tests/test_migrate_staging_e2e.py

URL overrides (defaults point at the dev channel):
    PLUTO_STAGING_URL_APP     https://pluto-dev.trainy.ai
    PLUTO_STAGING_URL_API     https://pluto-api-dev.trainy.ai
    PLUTO_STAGING_URL_INGEST  https://pluto-ingest-dev.trainy.ai
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone

import pytest

pytest.importorskip('pyarrow')

from pluto.migrate.loader import PlutoLoader  # noqa: E402
from pluto.migrate.schema import PartWriter  # noqa: E402
from pluto.migrate.state import mark_run_exported, write_json_atomic  # noqa: E402

STAGING_API_KEY = os.environ.get('PLUTO_STAGING_API_KEY')
URL_APP = os.environ.get('PLUTO_STAGING_URL_APP', 'https://pluto-dev.trainy.ai')
URL_API = os.environ.get('PLUTO_STAGING_URL_API', 'https://pluto-api-dev.trainy.ai')
URL_INGEST = os.environ.get(
    'PLUTO_STAGING_URL_INGEST', 'https://pluto-ingest-dev.trainy.ai'
)
URL_PY = os.environ.get('PLUTO_STAGING_URL_PY', 'https://pluto-py-dev.trainy.ai')

pytestmark = pytest.mark.skipif(
    not STAGING_API_KEY,
    reason='PLUTO_STAGING_API_KEY not set; staging integration test skipped',
)

PROJECT = 'migrate-staging-e2e'
CREATED_AT_MS = 1600000000000  # 2020-09-13T12:26:40Z
T0_MS = CREATED_AT_MS + 60_000
METRIC_POINTS = [  # (step, timestamp_ms, loss)
    (0, T0_MS, 1.0),
    (1, T0_MS + 1000, 0.5),
    (2, T0_MS + 2000, 0.25),
]


def _stage_fixture_run(root, run_id: str):
    run_dir = root / 'acme' / 'vision' / 'runs' / run_id
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / 'run.json',
        {
            'entity': 'acme',
            'project': PROJECT,
            'run_id': run_id,
            'name': f'staging-e2e-{run_id}',
            'notes': 'staging integration fixture',
            'tags': ['fixture'],
            'state': 'finished',
            'config': {'lr': 0.1, 'optimizer': 'adamw'},
            'summary': {'loss': 0.25},
            'createdAt': CREATED_AT_MS,
            'updatedAt': CREATED_AT_MS + 3_600_000,
            'url': f'https://wandb.ai/acme/vision/runs/{run_id}',
        },
    )
    with PartWriter(run_dir) as w:
        for step, ts_ms, loss in METRIC_POINTS:
            w.write_row(
                project_id='acme/vision',
                run_id=run_id,
                attribute_path='loss',
                attribute_type='metric',
                step=step,
                timestamp_ms=ts_ms,
                float_value=loss,
            )
        w.write_row(
            project_id='acme/vision',
            run_id=run_id,
            attribute_path='console',
            attribute_type='console',
            step=1,
            timestamp_ms=T0_MS,
            string_value='hello from 2020',
        )
    mark_run_exported(run_dir, {'rows': len(METRIC_POINTS) + 1})
    return run_dir


def _parse_point_time_ms(value) -> int:
    if isinstance(value, (int, float)):
        return int(value if value > 1e11 else value * 1000)
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@pytest.fixture
def staging_env(monkeypatch, tmp_path):
    monkeypatch.setenv('PLUTO_API_KEY', STAGING_API_KEY)
    monkeypatch.setenv('PLUTO_URL_APP', URL_APP)
    monkeypatch.setenv('PLUTO_URL_API', URL_API)
    monkeypatch.setenv('PLUTO_URL_INGEST', URL_INGEST)
    monkeypatch.setenv('PLUTO_URL_PY', URL_PY)
    monkeypatch.setenv('PLUTO_DIR', str(tmp_path / 'staging'))


def test_migration_round_trip_against_staging(tmp_path, staging_env):
    from pluto import query

    run_id = uuid.uuid4().hex[:12]
    _stage_fixture_run(tmp_path, run_id)

    summary = PlutoLoader(tmp_path).load()
    assert summary['failed'] == []
    assert summary['loaded'] == 1

    client = query.Client(api_token=STAGING_API_KEY, host=URL_API)
    runs = client.list_runs(PROJECT, search=f'staging-e2e-{run_id}')
    match = [r for r in runs if r['name'] == f'staging-e2e-{run_id}']
    assert match, f'imported run staging-e2e-{run_id} not found on staging'
    run = client.get_run(PROJECT, match[0]['id'])

    assert 'import:wandb' in run['tags']
    assert 'fixture' in run['tags']

    # Metrics may take a moment to land in ClickHouse; poll get_metrics
    # (NOT get_metric_names, which lags minutes behind ingest).
    deadline = time.time() + 120
    rows = []
    while time.time() < deadline:
        data = client.get_metrics(PROJECT, match[0]['id'], metric_names=['loss'])
        rows = data.to_dict('records') if hasattr(data, 'to_dict') else list(data)
        if len(rows) >= len(METRIC_POINTS):
            break
        time.sleep(5)
    assert len(rows) == len(METRIC_POINTS), f'expected 3 points, got {rows}'

    by_step = {r['step']: r for r in rows}
    for step, ts_ms, loss in METRIC_POINTS:
        assert by_step[step]['value'] == pytest.approx(loss)
        assert _parse_point_time_ms(by_step[step]['time']) == ts_ms, (
            f'historical timestamp not preserved for step {step}: '
            f'{by_step[step]["time"]!r}'
        )

    # Run createdAt backfill needs the server-side fix; xfail until deployed.
    run_created_ms = _parse_point_time_ms(run['createdAt'])
    if abs(run_created_ms - CREATED_AT_MS) > 60_000:
        pytest.xfail(
            'server does not yet honor createdAt on the run row '
            '(fix pending deploy); run created at '
            f'{run["createdAt"]!r} instead of 2020-09-13'
        )
    assert run_created_ms == CREATED_AT_MS


def test_reload_is_idempotent_against_staging(tmp_path, staging_env):
    run_id = uuid.uuid4().hex[:12]
    _stage_fixture_run(tmp_path, run_id)

    first = PlutoLoader(tmp_path).load()
    assert first['loaded'] == 1

    # Local cache skip
    second = PlutoLoader(tmp_path).load()
    assert second == {'loaded': 0, 'skipped': 1, 'failed': []}

    # Server-side external-id dedup (fresh cache simulates another machine)
    (tmp_path / 'loaded_runs.json').unlink()
    third = PlutoLoader(tmp_path).load()
    assert third['loaded'] == 0
    assert third['skipped'] == 1
