#!/usr/bin/env python3
"""Reproduce / isolate the name-scoped ``filters`` empty-result behavior.

Background
----------
In ``tests/test_e2e.py`` the column-only filter tests (status/state/created_at/
updated_at/$not) scope by a ``{"name": {"$regex": <batch>}}`` marker rather than
by ``config.batch``. In some CI runs those queries returned an empty set and the
tests skipped ("preview filters API returned no rows"), even though the server
implements every one of these fields/operators.

Static analysis of the whole path (client ``_validate_filters`` → route →
``compileRunFilter``) shows the code is correct for these fields, so an empty
result should not be a missing feature. This script pins down *where* the empty
comes from by resolving each leaf independently and then the intersections —
run it against prod or staging (where you have credentials); this environment
cannot reach either host.

Usage
-----
    PLUTO_API_KEY=... python scripts/repro_filter_name_scope.py
    # against staging instead of prod:
    PLUTO_API_KEY=... PLUTO_URL_API=https://pluto-api-dev.trainy.ai \
        python scripts/repro_filter_name_scope.py

It creates three short-lived finished runs in the ``PLUTO_REPRO_PROJECT``
project (default ``pluto-filter-repro``), then reports the id set each query
resolves to. If the ``name~=<batch>`` marker resolves all three but a
name-scoped column query does not, the gap is in the intersection/visibility of
that column leaf — not in the marker. If the marker itself is short, the gap is
read-path/index propagation for freshly created run rows.
"""

from __future__ import annotations

import os
import time
import uuid

import pluto
import pluto.query as pq

PROJECT = os.environ.get('PLUTO_REPRO_PROJECT', 'pluto-filter-repro')
POLL_TIMEOUT = float(os.environ.get('PLUTO_REPRO_TIMEOUT', '120'))
POLL_INTERVAL = 2.0
PAST_CUTOFF = '2000-01-01T00:00:00Z'


def query_ids(flt: dict) -> set:
    """Run a filter and return the matched run-id set (capped at 200)."""
    return {r['id'] for r in pq.list_runs(PROJECT, filters=flt, limit=200)}


def scoped(batch: str, case: dict) -> dict:
    """AND the case with the name-regex batch marker (matches the e2e tests)."""
    return {'$and': [{'name': {'$regex': batch}}, case]}


def poll_ids(flt: dict, want: set) -> set:
    """Poll a filter until it covers *want* (or timeout); return the last set."""
    deadline = time.monotonic() + POLL_TIMEOUT
    last = query_ids(flt)
    while not (want <= last):
        if time.monotonic() >= deadline:
            return last
        time.sleep(POLL_INTERVAL)
        last = query_ids(flt)
    return last


def seed(batch: str) -> dict:
    """Create + finish three runs with known config/name; return group->id."""
    specs = {'alpha': 0.001, 'beta': 0.01, 'gamma': 0.1}
    ids: dict[str, int] = {}
    for group, lr in specs.items():
        run = pluto.init(
            project=PROJECT,
            name=f't-repro-filter-{batch}-{group}',
            tags=[f'repro-{batch}', group],
            config={'lr': lr, 'batch': batch, 'group': group},
        )
        run.log({'loss': lr})
        ids[group] = run.settings._op_id
        run.finish()
    return ids


def main() -> None:
    api_url = os.environ.get('PLUTO_URL_API', 'https://pluto-api.trainy.ai (default)')
    batch = uuid.uuid4().hex[:12]
    print(f'server (PLUTO_URL_API): {api_url}')
    print(f'project:                {PROJECT}')
    print(f'batch marker:           {batch}\n')

    ids = seed(batch)
    all_ids = set(ids.values())
    print(f'created run ids: {ids}\n')

    # Warm up on config.batch (the path the passing tests rely on).
    config_marker = poll_ids({'config.batch': batch}, all_ids)
    print(f'[config.batch == {batch}]          -> {config_marker}')

    # The name-regex marker on its own — the shared scope of every skipping test.
    name_marker = poll_ids({'name': {'$regex': batch}}, all_ids)
    print(f'[name ~= {batch}]                  -> {name_marker}')

    # Each column-only leaf, name-scoped exactly as the e2e tests build it,
    # paired with the run groups it should match. $not(name~=alpha) excludes the
    # alpha run, so it must be graded against {beta, gamma} — not all three, or
    # it would always time out and falsely report GAP.
    everyone = {'alpha', 'beta', 'gamma'}
    checks = {
        'status == COMPLETED': ({'status': {'$eq': 'COMPLETED'}}, everyone),
        'state != running': ({'state': {'$ne': 'running'}}, everyone),
        'created_at >= 2000': ({'created_at': {'$gte': PAST_CUTOFF}}, everyone),
        'updated_at >= 2000': ({'updated_at': {'$gte': PAST_CUTOFF}}, everyone),
        '$not(name ~= alpha)': (
            {'$not': {'name': {'$regex': 'alpha'}}},
            {'beta', 'gamma'},
        ),
    }
    print()
    for label, (case, groups) in checks.items():
        want = {ids[g] for g in groups}
        got = poll_ids(scoped(batch, case), want)
        ok = 'OK ' if want <= got else 'GAP'
        print(f'[{ok}] name-scoped {label:<22} -> {got}')

    print('\nInterpretation:')
    print('  * name marker short         => read-path/index lag on new run rows')
    print('  * name marker full, leaf GAP => intersection/visibility of that leaf')
    print('  * all OK                     => timing only; the e2e warmups fix it')


if __name__ == '__main__':
    main()
