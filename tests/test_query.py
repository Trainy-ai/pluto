"""Tests for pluto.query module."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from pluto.query import Client, QueryError, _resolve_api_token, _resolve_url_api

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove query-related env vars for test isolation."""
    for key in (
        'PLUTO_API_KEY',
        'MLOP_API_TOKEN',
        'PLUTO_URL_API',
        'MLOP_URL_API',
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def mock_response():
    """Factory for mock httpx.Response objects."""

    def _make(status_code=200, json_data=None, text=''):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.json.return_value = json_data if json_data is not None else {}
        resp.text = text or json.dumps(json_data or {})
        return resp

    return _make


@pytest.fixture()
def client(monkeypatch):
    """A Client with a mocked httpx.Client."""
    monkeypatch.setenv('PLUTO_API_KEY', 'test-token-123')
    c = Client()
    c._client = MagicMock(spec=httpx.Client)
    return c


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


class TestResolveApiToken:
    def test_from_pluto_env(self, monkeypatch):
        monkeypatch.setenv('PLUTO_API_KEY', 'plt_abc')
        assert _resolve_api_token() == 'plt_abc'

    def test_from_deprecated_mlop_env(self, monkeypatch):
        monkeypatch.setenv('MLOP_API_TOKEN', 'old_token')
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            token = _resolve_api_token()
        assert token == 'old_token'
        assert any('MLOP_API_TOKEN' in str(x.message) for x in w)

    def test_pluto_takes_precedence(self, monkeypatch):
        monkeypatch.setenv('PLUTO_API_KEY', 'new_token')
        monkeypatch.setenv('MLOP_API_TOKEN', 'old_token')
        assert _resolve_api_token() == 'new_token'

    def test_none_when_nothing_set(self):
        # keyring may or may not work, but at minimum no env var
        token = _resolve_api_token()
        # Token might come from keyring if configured, but in CI it should be None
        assert token is None or isinstance(token, str)


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------


class TestResolveUrlApi:
    def test_default(self):
        assert _resolve_url_api() == 'https://pluto-api.trainy.ai'

    def test_full_url(self):
        assert (
            _resolve_url_api('https://my-api.example.com')
            == 'https://my-api.example.com'
        )

    def test_full_url_trailing_slash(self):
        assert (
            _resolve_url_api('https://my-api.example.com/')
            == 'https://my-api.example.com'
        )

    def test_bare_host(self):
        assert _resolve_url_api('10.0.0.1') == 'http://10.0.0.1:3001'

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv('PLUTO_URL_API', 'https://env-api.example.com')
        assert _resolve_url_api() == 'https://env-api.example.com'

    def test_deprecated_env_var(self, monkeypatch):
        monkeypatch.setenv('MLOP_URL_API', 'https://old-api.example.com')
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            url = _resolve_url_api()
        assert url == 'https://old-api.example.com'
        assert any('MLOP_URL_API' in str(x.message) for x in w)

    def test_host_param_overrides_env(self, monkeypatch):
        monkeypatch.setenv('PLUTO_URL_API', 'https://env-api.example.com')
        assert (
            _resolve_url_api('https://param-api.example.com')
            == 'https://param-api.example.com'
        )


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------


class TestClientInit:
    def test_requires_token(self):
        with patch('pluto.query._resolve_api_token', return_value=None):
            with pytest.raises(QueryError, match='No API token'):
                Client()

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv('PLUTO_API_KEY', 'plt_abc')
        c = Client()
        assert c._api_token == 'plt_abc'
        c.close()

    def test_explicit_token(self, monkeypatch):
        c = Client(api_token='explicit_token')
        assert c._api_token == 'explicit_token'
        c.close()

    def test_context_manager(self, monkeypatch):
        monkeypatch.setenv('PLUTO_API_KEY', 'plt_abc')
        with Client() as c:
            assert c._api_token == 'plt_abc'

    def test_custom_host(self, monkeypatch):
        monkeypatch.setenv('PLUTO_API_KEY', 'plt_abc')
        c = Client(host='10.0.0.1')
        assert c._url_api == 'http://10.0.0.1:3001'
        c.close()


# ---------------------------------------------------------------------------
# list_projects
# ---------------------------------------------------------------------------


class TestListProjects:
    def test_success(self, client, mock_response):
        data = [
            {'id': 1, 'name': 'proj-a', 'runCount': 5},
            {'id': 2, 'name': 'proj-b', 'runCount': 3},
        ]
        client._client.get.return_value = mock_response(200, {'projects': data})
        result = client.list_projects()
        assert result == data
        client._client.get.assert_called_once()
        call_args = client._client.get.call_args
        assert '/api/runs/projects' in call_args[0][0]


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_basic(self, client, mock_response):
        data = [{'id': 1, 'name': 'run-1', 'displayId': 'MMP-1'}]
        client._client.get.return_value = mock_response(200, {'runs': data})
        result = client.list_runs('my-project')
        assert result == data
        call_args = client._client.get.call_args
        assert call_args[1]['params']['projectName'] == 'my-project'
        assert call_args[1]['params']['limit'] == 50

    def test_with_search_and_tags(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        client.list_runs('proj', search='experiment', tags=['v2', 'prod'])
        call_args = client._client.get.call_args
        assert call_args[1]['params']['search'] == 'experiment'
        assert call_args[1]['params']['tags'] == 'v2,prod'

    def test_limit_capped_at_200(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        client.list_runs('proj', limit=999)
        call_args = client._client.get.call_args
        assert call_args[1]['params']['limit'] == 200

    def test_sort(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        client.list_runs('proj', sort='-createdAt')
        call_args = client._client.get.call_args
        assert call_args[1]['params']['sort'] == '-createdAt'

    def test_offset(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        client.list_runs('proj', offset=100)
        call_args = client._client.get.call_args
        assert call_args[1]['params']['offset'] == 100

    def test_offset_zero_omitted(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        client.list_runs('proj')
        call_args = client._client.get.call_args
        assert 'offset' not in call_args[1]['params']

    def test_offset_clamped(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        client.list_runs('proj', offset=10**9)
        call_args = client._client.get.call_args
        assert call_args[1]['params']['offset'] == 100_000

    def test_negative_offset_omitted(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        client.list_runs('proj', offset=-5)
        call_args = client._client.get.call_args
        assert 'offset' not in call_args[1]['params']

    def test_filters_serialized_as_json(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        flt = {
            '$or': [
                {'state': 'running'},
                {'heartbeat_at': {'$gte': '2026-06-22T00:00:00Z'}},
            ]
        }
        client.list_runs('proj', filters=flt)
        params = client._client.get.call_args[1]['params']
        assert json.loads(params['filter']) == flt

    def test_filters_config_leaf(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        client.list_runs('proj', filters={'config.lr': {'$gt': 0.001}})
        params = client._client.get.call_args[1]['params']
        assert json.loads(params['filter']) == {'config.lr': {'$gt': 0.001}}

    def test_filters_unknown_field_raises(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        with pytest.raises(ValueError, match='unknown filter field'):
            client.list_runs('proj', filters={'bogus': 1})

    def test_filters_unknown_operator_raises(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        with pytest.raises(ValueError, match='unknown leaf operator'):
            client.list_runs('proj', filters={'config.lr': {'$bogus': 1}})

    def test_filters_unknown_boolean_op_raises(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        with pytest.raises(ValueError, match='unknown boolean operator'):
            client.list_runs('proj', filters={'$xor': [{'state': 'running'}]})

    def test_filters_validated_before_http(self, client, mock_response):
        # An invalid filter must raise before any request is issued.
        client._client.get.return_value = mock_response(200, {'runs': []})
        with pytest.raises(ValueError):
            client.list_runs('proj', filters={'bogus': 1})
        client._client.get.assert_not_called()

    def test_filters_combine_with_search_and_tags(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        client.list_runs(
            'proj', search='exp', tags=['a', 'b'], filters={'state': 'running'}
        )
        params = client._client.get.call_args[1]['params']
        assert params['search'] == 'exp'
        assert params['tags'] == 'a,b'
        assert json.loads(params['filter']) == {'state': 'running'}

    def test_filters_in_operator_list_roundtrips(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'runs': []})
        flt = {'config.lr': {'$in': [0.1, 0.01, 0.001]}}
        client.list_runs('proj', filters=flt)
        params = client._client.get.call_args[1]['params']
        assert json.loads(params['filter']) == flt


class TestValidateFilters:
    def test_accepts_nested_boolean(self):
        from pluto.query import _validate_filters

        _validate_filters(
            {
                '$and': [
                    {'$or': [{'status': 'RUNNING'}, {'status': 'FAILED'}]},
                    {'$not': {'config.lr': {'$lt': 0.1}}},
                    {'tags': {'$in': ['a', 'b']}},
                    {'summaryMetrics.loss': {'$lte': 0.5}},
                ]
            }
        )

    def test_and_or_require_list(self):
        from pluto.query import _validate_filters

        with pytest.raises(ValueError, match=r'\$or expects a list'):
            _validate_filters({'$or': {'state': 'running'}})


class TestFilterValidation:
    """Exhaustive structural validation of the wandb-style ``filters`` grammar.

    Covers every documented leaf operator, exact field, and field prefix, plus
    the guard branches in ``_validate_filters`` / ``_validate_filter_field``
    that the happy-path tests above don't reach. The vocabulary mirrored here
    is kept equal to the server's published grammar by ``test_contract.py``.
    """

    @pytest.mark.parametrize(
        'op', ['$eq', '$ne', '$gt', '$gte', '$lt', '$lte', '$in', '$nin', '$regex']
    )
    def test_each_leaf_operator_accepted(self, op):
        from pluto.query import _FILTER_LEAF_OPS, _validate_filters

        assert op in _FILTER_LEAF_OPS  # guard against silent vocab drift
        value = ['a', 'b'] if op in ('$in', '$nin') else 1
        _validate_filters({'config.x': {op: value}})

    @pytest.mark.parametrize(
        'field',
        [
            'state',
            'status',
            'heartbeat_at',
            'heartbeatAt',
            'created_at',
            'createdAt',
            'updated_at',
            'updatedAt',
            'name',
            'displayName',
            'display_name',
            'tags',
        ],
    )
    def test_each_exact_field_accepted(self, field):
        from pluto.query import _FILTER_FIELDS, _validate_filters

        assert field in _FILTER_FIELDS  # guard against silent vocab drift
        _validate_filters({field: 'x'})

    @pytest.mark.parametrize(
        'prefix', ['config.', 'systemMetadata.', 'summaryMetrics.', 'summary_metrics.']
    )
    def test_each_field_prefix_accepted(self, prefix):
        from pluto.query import _FILTER_FIELD_PREFIXES, _validate_filters

        assert prefix in _FILTER_FIELD_PREFIXES  # guard against silent vocab drift
        _validate_filters({f'{prefix}key': 1})

    def test_bare_prefix_without_suffix_rejected(self):
        from pluto.query import _validate_filters

        # A prefix with no key after it isn't a valid field.
        with pytest.raises(ValueError, match='unknown filter field'):
            _validate_filters({'config.': 1})

    def test_plain_equality_leaves_accepted(self):
        from pluto.query import _validate_filters

        _validate_filters({'name': 'foo'})
        _validate_filters({'state': 'running'})
        _validate_filters({'tags': ['a', 'b']})

    def test_depth_limit_rejected(self):
        from pluto.query import _validate_filters

        node: dict = {'config.x': 1}
        for _ in range(60):
            node = {'$not': node}
        with pytest.raises(ValueError, match='nested too deeply'):
            _validate_filters(node)

    def test_non_dict_node_in_list_rejected(self):
        from pluto.query import _validate_filters

        with pytest.raises(ValueError, match='must be a dict'):
            _validate_filters({'$and': [123]})

    def test_top_level_non_dict_rejected(self):
        from pluto.query import _validate_filters

        with pytest.raises(ValueError, match='must be a dict'):
            _validate_filters([{'state': 'running'}])

    def test_and_requires_list(self):
        from pluto.query import _validate_filters

        with pytest.raises(ValueError, match=r'\$and expects a list'):
            _validate_filters({'$and': {'state': 'running'}})

    def test_not_recurses_into_child(self):
        from pluto.query import _validate_filters

        with pytest.raises(ValueError, match='unknown filter field'):
            _validate_filters({'$not': {'bogus': 1}})

    def test_invalid_operator_deep_inside_or(self):
        from pluto.query import _validate_filters

        with pytest.raises(ValueError, match='unknown leaf operator'):
            _validate_filters({'$or': [{'config.x': {'$bogus': 1}}]})


# ---------------------------------------------------------------------------
# get_run
# ---------------------------------------------------------------------------


class TestGetRun:
    def test_by_numeric_id(self, client, mock_response):
        data = {'id': 42, 'name': 'run-42', 'config': {'lr': 0.001}}
        client._client.get.return_value = mock_response(200, data)
        result = client.get_run('proj', 42)
        assert result == data
        call_url = client._client.get.call_args[0][0]
        assert '/api/runs/details/42' in call_url

    def test_by_display_id(self, client, mock_response):
        data = {'id': 42, 'displayId': 'MMP-1', 'config': {}}
        client._client.get.return_value = mock_response(200, data)
        result = client.get_run('proj', 'MMP-1')
        assert result == data
        call_url = client._client.get.call_args[0][0]
        assert '/api/runs/details/by-display-id/MMP-1' in call_url


# ---------------------------------------------------------------------------
# get_metric_names
# ---------------------------------------------------------------------------


class TestGetMetricNames:
    def test_basic(self, client, mock_response):
        data = ['loss', 'accuracy', 'val/loss']
        client._client.get.return_value = mock_response(200, {'metricNames': data})
        result = client.get_metric_names('proj')
        assert result == data

    def test_with_run_ids(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'metricNames': []})
        client.get_metric_names('proj', run_ids=[1, 2, 3])
        params = client._client.get.call_args[1]['params']
        assert params['runIds'] == '1,2,3'


# ---------------------------------------------------------------------------
# get_metrics
# ---------------------------------------------------------------------------


class TestGetMetrics:
    def test_single_metric(self, client, mock_response):
        server_data = [
            {'logName': 'loss', 'step': 0, 'value': 1.0, 'time': '2025-01-01'},
            {'logName': 'loss', 'step': 1, 'value': 0.5, 'time': '2025-01-01'},
        ]
        client._client.get.return_value = mock_response(200, {'metrics': server_data})
        result = client.get_metrics('proj', 42, metric_names=['loss'])
        params = client._client.get.call_args[1]['params']
        assert params['logName'] == 'loss'
        # Should return a DataFrame if pandas available
        try:
            import pandas as pd

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            # logName should be renamed to metric
            assert 'metric' in result.columns
        except ImportError:
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]['metric'] == 'loss'

    def test_multiple_metrics(self, client, mock_response):
        data_loss = [{'logName': 'loss', 'step': 0, 'value': 1.0}]
        data_acc = [{'logName': 'acc', 'step': 0, 'value': 0.8}]
        client._client.get.side_effect = [
            mock_response(200, {'metrics': data_loss}),
            mock_response(200, {'metrics': data_acc}),
        ]
        client.get_metrics('proj', 42, metric_names=['loss', 'acc'])
        assert client._client.get.call_count == 2

    def test_all_metrics(self, client, mock_response):
        server_data = [{'logName': 'loss', 'step': 0, 'value': 1.0}]
        client._client.get.return_value = mock_response(200, {'metrics': server_data})
        client.get_metrics('proj', 42)
        params = client._client.get.call_args[1]['params']
        assert 'logName' not in params

    def test_step_range_sent_as_camel_case_params(self, client, mock_response):
        """step_min/step_max serialize to the endpoint's stepMin/stepMax."""
        client._client.get.return_value = mock_response(200, {'metrics': []})
        client.get_metrics(
            'proj', 42, metric_names=['loss'], step_min=4000, step_max=4500
        )
        params = client._client.get.call_args[1]['params']
        assert params['stepMin'] == 4000
        assert params['stepMax'] == 4500

    def test_step_range_omitted_when_unset(self, client, mock_response):
        """Neither bound is sent when not supplied — no empty params."""
        client._client.get.return_value = mock_response(200, {'metrics': []})
        client.get_metrics('proj', 42, metric_names=['loss'])
        params = client._client.get.call_args[1]['params']
        assert 'stepMin' not in params
        assert 'stepMax' not in params

    def test_step_range_bounds_are_independent(self, client, mock_response):
        """Either bound can be given on its own."""
        client._client.get.return_value = mock_response(200, {'metrics': []})
        client.get_metrics('proj', 42, step_min=100)
        params = client._client.get.call_args[1]['params']
        assert params['stepMin'] == 100
        assert 'stepMax' not in params

        client._client.get.reset_mock()
        client._client.get.return_value = mock_response(200, {'metrics': []})
        client.get_metrics('proj', 42, step_max=900)
        params = client._client.get.call_args[1]['params']
        assert params['stepMax'] == 900
        assert 'stepMin' not in params

    def test_step_range_applied_to_every_multi_metric_request(
        self, client, mock_response
    ):
        """The multi-metric path fans out to one request per metric; each one
        must still carry the bounds (regression guard: setting them only on the
        single-metric branch would silently drop them here and quietly return
        the full series)."""
        client._client.get.side_effect = [
            mock_response(200, {'metrics': [{'logName': 'loss', 'step': 4000}]}),
            mock_response(200, {'metrics': [{'logName': 'acc', 'step': 4000}]}),
        ]
        client.get_metrics(
            'proj', 42, metric_names=['loss', 'acc'], step_min=4000, step_max=4500
        )

        assert client._client.get.call_count == 2
        for call in client._client.get.call_args_list:
            params = call[1]['params']
            assert params['stepMin'] == 4000
            assert params['stepMax'] == 4500
        # ...and each request still targets its own metric.
        sent = [c[1]['params']['logName'] for c in client._client.get.call_args_list]
        assert sent == ['loss', 'acc']

    def test_negative_step_bounds_rejected(self, client):
        """Negative bounds fail fast client-side (the server would 400)."""
        with pytest.raises(ValueError, match='step_min must be non-negative'):
            client.get_metrics('proj', 42, step_min=-1)
        with pytest.raises(ValueError, match='step_max must be non-negative'):
            client.get_metrics('proj', 42, step_max=-1)
        client._client.get.assert_not_called()

    def test_reversed_step_range_rejected(self, client):
        """step_min > step_max raises instead of silently returning nothing.

        The server validates each bound independently but not their ordering,
        so a reversed range would match no rows and come back as an empty
        result with no error — a silently wrong answer.
        """
        with pytest.raises(ValueError, match='cannot be greater than'):
            client.get_metrics('proj', 42, step_min=100, step_max=50)
        client._client.get.assert_not_called()

    def test_equal_step_bounds_allowed(self, client, mock_response):
        """step_min == step_max is a valid single-step window."""
        client._client.get.return_value = mock_response(200, {'metrics': []})
        client.get_metrics('proj', 42, step_min=42, step_max=42)
        params = client._client.get.call_args[1]['params']
        assert params['stepMin'] == 42
        assert params['stepMax'] == 42

    def test_empty_returns_empty_dataframe(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'metrics': []})
        result = client.get_metrics('proj', 42)
        try:
            import pandas as pd

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
        except ImportError:
            assert result == []

    def test_display_id_resolves_to_numeric(self, client, mock_response):
        client._client.get.side_effect = [
            mock_response(200, {'id': 99, 'displayId': 'MMP-1'}),
            mock_response(200, {'metrics': []}),
        ]
        client.get_metrics('proj', 'MMP-1', metric_names=['loss'])
        # First call: get_run by display ID
        first_url = client._client.get.call_args_list[0][0][0]
        assert 'by-display-id/MMP-1' in first_url
        # Second call: /api/runs/metrics with resolved numeric ID
        second_params = client._client.get.call_args_list[1][1]['params']
        assert second_params['runId'] == 99

    def test_numeric_string_is_treated_as_server_id(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'metrics': []})
        client.get_metrics('proj', '42')
        # Only one call — no get_run lookup for numeric strings
        assert client._client.get.call_count == 1
        params = client._client.get.call_args[1]['params']
        assert params['runId'] == 42


# ---------------------------------------------------------------------------
# get_statistics
# ---------------------------------------------------------------------------


class TestGetStatistics:
    def test_basic(self, client, mock_response):
        data = {'loss': {'min': 0.1, 'max': 1.0, 'mean': 0.5}}
        client._client.get.return_value = mock_response(200, data)
        result = client.get_statistics('proj', 42)
        assert result == data


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def test_basic(self, client, mock_response):
        data = {'runs': [], 'bestRun': None}
        client._client.get.return_value = mock_response(200, data)
        client.compare_runs('proj', [1, 2, 3], 'loss')
        params = client._client.get.call_args[1]['params']
        assert params['runIds'] == '1,2,3'
        assert params['logName'] == 'loss'

    def test_mixed_display_and_numeric_ids(self, client, mock_response):
        client._client.get.side_effect = [
            mock_response(200, {'id': 7, 'displayId': 'MMP-1'}),
            mock_response(200, {'runs': [], 'bestRun': None}),
        ]
        client.compare_runs('proj', [1, 'MMP-1'], 'loss')
        params = client._client.get.call_args_list[-1][1]['params']
        assert params['runIds'] == '1,7'

    def test_duplicate_display_ids_resolved_once(self, client, mock_response):
        client._client.get.side_effect = [
            mock_response(200, {'id': 7, 'displayId': 'MMP-1'}),
            mock_response(200, {'id': 9, 'displayId': 'MMP-2'}),
            mock_response(200, {'runs': [], 'bestRun': None}),
        ]
        client.compare_runs('proj', ['MMP-1', 'MMP-2', 'MMP-1', 'MMP-2'], 'loss')
        # 2 get_run lookups + 1 compare = 3 calls (not 5)
        assert client._client.get.call_count == 3
        params = client._client.get.call_args_list[-1][1]['params']
        assert params['runIds'] == '7,9,7,9'


# ---------------------------------------------------------------------------
# leaderboard
# ---------------------------------------------------------------------------


class TestLeaderboard:
    def test_defaults(self, client, mock_response):
        client._client.get.return_value = mock_response(200, [])
        client.leaderboard('proj', 'val/loss')
        params = client._client.get.call_args[1]['params']
        assert params['aggregation'] == 'LAST'
        assert params['direction'] == 'ASC'
        assert params['limit'] == 50

    def test_custom_params(self, client, mock_response):
        client._client.get.return_value = mock_response(200, [])
        client.leaderboard('proj', 'acc', aggregation='MAX', direction='DESC', limit=10)
        params = client._client.get.call_args[1]['params']
        assert params['aggregation'] == 'MAX'
        assert params['direction'] == 'DESC'
        assert params['limit'] == 10


# ---------------------------------------------------------------------------
# get_files
# ---------------------------------------------------------------------------


class TestGetFiles:
    def test_basic(self, client, mock_response):
        data = [{'fileName': 'model.pt', 'downloadUrl': 'https://s3/model.pt'}]
        client._client.get.return_value = mock_response(200, {'files': data})
        result = client.get_files('proj', 42)
        assert result == data

    def test_with_filter(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'files': []})
        client.get_files('proj', 42, file_name='checkpoint')
        params = client._client.get.call_args[1]['params']
        assert params['logName'] == 'checkpoint'


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    def test_download(self, client, mock_response, tmp_path):
        file_data = [
            {
                'fileName': 'model.pt',
                'downloadUrl': 'https://s3/model.pt',
                'fileSize': 100,
            }
        ]
        client._client.get.return_value = mock_response(200, {'files': file_data})

        dl_response = MagicMock()
        dl_response.content = b'model-bytes'
        dl_response.raise_for_status = MagicMock()

        with patch('pluto.query.httpx.get', return_value=dl_response):
            path = client.download_file('proj', 42, 'model.pt', destination=tmp_path)

        assert path == tmp_path / 'model.pt'
        assert path.read_bytes() == b'model-bytes'

    def test_no_matching_file(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'files': []})
        with pytest.raises(QueryError, match='No file found'):
            client.download_file('proj', 42, 'missing.pt')


# ---------------------------------------------------------------------------
# get_logs
# ---------------------------------------------------------------------------


class TestGetLogs:
    def test_basic(self, client, mock_response):
        data = [{'message': 'hello', 'logType': 'info', 'lineNumber': 1}]
        client._client.get.return_value = mock_response(200, {'logs': data})
        result = client.get_logs('proj', 42)
        assert result == data

    def test_with_type_filter(self, client, mock_response):
        client._client.get.return_value = mock_response(200, {'logs': []})
        client.get_logs('proj', 42, log_type='error')
        params = client._client.get.call_args[1]['params']
        assert params['logType'] == 'error'


# ---------------------------------------------------------------------------
# Error handling & retries
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_client_error_raises(self, client, mock_response):
        client._client.get.return_value = mock_response(404, text='Not Found')
        with pytest.raises(QueryError, match='404'):
            client.list_projects()

    def test_server_error_retries_and_raises(self, client, mock_response):
        client._client.get.return_value = mock_response(
            500,
            text='Internal Server Error',
        )
        with pytest.raises(QueryError, match='500'):
            client.list_projects()
        # Should have retried: 1 initial + 4 retries = 5 calls
        assert client._client.get.call_count == 5

    def test_connection_error_retries(self, client):
        client._client.get.side_effect = httpx.ConnectError('connection refused')
        with pytest.raises(QueryError, match='Failed after'):
            client.list_projects()
        assert client._client.get.call_count == 5


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def test_list_runs_creates_default_client(self, monkeypatch, mock_response):
        monkeypatch.setenv('PLUTO_API_KEY', 'test-token')

        import pluto.query as pq

        # Reset default client
        pq._default_client = None

        mock_client_instance = MagicMock(spec=Client)
        mock_client_instance.list_runs.return_value = [{'id': 1}]

        with patch.object(pq, 'Client', return_value=mock_client_instance):
            result = pq.list_runs('my-project')

        assert result == [{'id': 1}]
        mock_client_instance.list_runs.assert_called_once_with(
            'my-project',
            search=None,
            tags=None,
            limit=50,
            sort=None,
            offset=0,
            filters=None,
        )

        # Clean up
        pq._default_client = None

    def test_import_as_submodule(self):
        import pluto.query as pq

        assert hasattr(pq, 'Client')
        assert hasattr(pq, 'list_runs')
        assert hasattr(pq, 'get_run')
        assert hasattr(pq, 'get_metrics')

    def test_accessible_from_pluto(self):
        import pluto

        assert hasattr(pluto, 'query')
        assert hasattr(pluto.query, 'Client')
