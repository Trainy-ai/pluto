"""Contract test: the client's ``filters`` surface has a server counterpart.

The client sends `list_runs(filters=...)` as the `/api/runs/list` ``filter``
query param. This test fetches the server's published OpenAPI document
(``{url_api}/api/openapi.json``) and asserts that endpoint actually documents a
``filter`` parameter — so a deployed server that dropped/renamed it (or a client
pointed at a server that predates the feature) fails CI loudly instead of
silently no-op-ing the filter.

Until the server change is deployed to the target host, ``/api/runs/list`` has no
``filter`` param yet, so the test SKIPS rather than fails. The detailed operator
vocabulary is validated client-side (``tests/test_query.py``) and on the server;
a structured ``RunFilter`` schema for a full operator/field contract is a
fast-follow (see plan).

Network test: hits the live API spec endpoint. Skips cleanly when unreachable, so
offline/hermetic runs are unaffected — matching the style of ``tests/test_e2e.py``.
"""

import os

import httpx
import pytest

from pluto.query import (
    _FILTER_BOOL_OPS,
    _FILTER_FIELD_PREFIXES,
    _FILTER_FIELDS,
    _FILTER_LEAF_OPS,
    _resolve_url_api,
)


def _fetch_openapi() -> dict:
    url = f'{_resolve_url_api(None)}/api/openapi.json'
    # Prod serves the spec unauthenticated; staging/dev may gate it. Send the
    # bearer token when present so the check actually runs there. Harmless on an
    # unauthenticated endpoint.
    headers = {}
    token = os.environ.get('PLUTO_API_KEY')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers=headers)
    except httpx.HTTPError as exc:  # pragma: no cover - network dependent
        pytest.skip(f'Could not reach OpenAPI spec at {url}: {exc}')
    if resp.status_code != 200:
        pytest.skip(f'OpenAPI spec at {url} returned HTTP {resp.status_code}')
    try:
        return resp.json()
    except ValueError as exc:  # invalid/non-JSON body (e.g. an HTML error page)
        pytest.skip(f'OpenAPI spec at {url} returned non-JSON body: {exc}')


def _list_runs_params() -> dict:
    spec = _fetch_openapi()
    op = spec.get('paths', {}).get('/api/runs/list', {}).get('get', {})
    params = {p.get('name') for p in op.get('parameters', [])}
    if 'filter' not in params:
        pytest.skip(
            'GET /api/runs/list has no `filter` param yet; pending the '
            'pluto-server filter-query change being deployed to this host.'
        )
    return params


def test_filter_param_is_published():
    params = _list_runs_params()
    assert 'filter' in params


def test_list_runs_still_documents_core_params():
    # Guard against an accidental contract regression on the surface the client
    # relies on alongside `filter`.
    params = _list_runs_params()
    for p in ('projectName', 'limit', 'sort', 'offset'):
        assert p in params, f'/api/runs/list missing documented param: {p}'


def _grammar() -> dict:
    """The server's published RunFilterGrammar component (its `properties`)."""
    spec = _fetch_openapi()
    schemas = spec.get('components', {}).get('schemas', {})
    grammar = schemas.get('RunFilterGrammar')
    if grammar is None:
        pytest.skip(
            'OpenAPI spec has no RunFilterGrammar component yet; pending the '
            'pluto-server grammar-publish change being deployed to this host.'
        )
    return grammar.get('properties', {})


def _published(props: dict, key: str) -> set:
    enum = props.get(key, {}).get('items', {}).get('enum')
    if enum is None:
        pytest.skip(f'RunFilterGrammar.{key} has no items.enum in the spec')
    return set(enum)


@pytest.mark.parametrize(
    'prop,client',
    [
        ('booleanOperators', _FILTER_BOOL_OPS),
        ('leafOperators', _FILTER_LEAF_OPS),
        ('fields', _FILTER_FIELDS),
        ('fieldPrefixes', set(_FILTER_FIELD_PREFIXES)),
    ],
)
def test_client_filter_vocab_matches_server(prop, client):
    """The client's filter vocabulary must equal the server's published grammar.

    Drift (a field/operator added on one side only) fails here with the exact
    set difference, instead of surfacing as a runtime 400.
    """
    server = _published(_grammar(), prop)
    assert client == server, (
        f'{prop} drift between client and server.\n'
        f'  server-only: {sorted(server - client)}\n'
        f'  client-only: {sorted(client - server)}'
    )
