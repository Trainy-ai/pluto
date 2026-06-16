"""Contract test: keep the client's field-filter vocabulary in step with the
server's zod schema.

The server's zod schema for the ``/api/runs/list`` ``fieldFilters`` parameter is
the source of truth. Rather than reimplement validation, we assert the client's
hardcoded enums match the schema the server publishes in its OpenAPI document
(served at ``{url_api}/api/openapi.json``), so any drift fails CI loudly instead
of surfacing as opaque HTTP 400s for users.

This depends on the server exposing the inner filter-term schema as a structured
OpenAPI component (``components.schemas.FieldFilterTerm`` with real ``enum``s).
Until that companion server change lands, the spec carries the operators only as
prose in the parameter description, so this test SKIPS rather than fails.

Network test: hits the live API spec endpoint (no auth required for the public
spec). Skips cleanly when the spec is unreachable, so offline/hermetic runs are
unaffected — matching the network-dependent style of ``tests/test_e2e.py``.
"""

import httpx
import pytest

from pluto.query import (
    _FILTER_DATATYPES,
    _FILTER_OPERATORS,
    _FILTER_SOURCES,
    _resolve_url_api,
)

# Component name the companion server PR registers in its OpenAPI document.
_COMPONENT = 'FieldFilterTerm'


def _fetch_openapi() -> dict:
    url = f'{_resolve_url_api(None)}/api/openapi.json'
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
    except httpx.HTTPError as exc:  # pragma: no cover - network dependent
        pytest.skip(f'Could not reach OpenAPI spec at {url}: {exc}')
    if resp.status_code != 200:
        pytest.skip(f'OpenAPI spec at {url} returned HTTP {resp.status_code}')
    try:
        return resp.json()
    except ValueError as exc:  # invalid/non-JSON body (e.g. an HTML error page)
        pytest.skip(f'OpenAPI spec at {url} returned non-JSON body: {exc}')


def _term_schema() -> dict:
    spec = _fetch_openapi()
    schemas = spec.get('components', {}).get('schemas', {})
    if _COMPONENT not in schemas:
        pytest.skip(
            f'OpenAPI spec has no components.schemas.{_COMPONENT} yet; '
            'pending the companion pluto-server change that surfaces the '
            'field-filter zod schema as a structured component.'
        )
    return schemas[_COMPONENT]


def _enum_for(term: dict, field: str) -> set:
    prop = term.get('properties', {}).get(field, {})
    enum = prop.get('enum')
    if enum is None:
        pytest.skip(f'{_COMPONENT}.{field} has no enum in the OpenAPI spec')
    return set(enum)


def test_filter_operators_match_server():
    server = _enum_for(_term_schema(), 'operator')
    assert server == _FILTER_OPERATORS, (
        'operator enum drift between client and server.\n'
        f'  server-only: {sorted(server - _FILTER_OPERATORS)}\n'
        f'  client-only: {sorted(_FILTER_OPERATORS - server)}'
    )


def test_filter_sources_match_server():
    server = _enum_for(_term_schema(), 'source')
    assert server == _FILTER_SOURCES, (
        'source enum drift between client and server.\n'
        f'  server-only: {sorted(server - _FILTER_SOURCES)}\n'
        f'  client-only: {sorted(_FILTER_SOURCES - server)}'
    )


def test_filter_datatypes_match_server():
    server = _enum_for(_term_schema(), 'dataType')
    assert server == _FILTER_DATATYPES, (
        'dataType enum drift between client and server.\n'
        f'  server-only: {sorted(server - _FILTER_DATATYPES)}\n'
        f'  client-only: {sorted(_FILTER_DATATYPES - server)}'
    )
