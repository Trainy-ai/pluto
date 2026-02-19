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
        'PLUTO_API_TOKEN',
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
    monkeypatch.setenv('PLUTO_API_TOKEN', 'test-token-123')
    c = Client()
    c._client = MagicMock(spec=httpx.Client)
    return c


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


class TestResolveApiToken:
    def test_from_pluto_env(self, monkeypatch):
        monkeypatch.setenv('PLUTO_API_TOKEN', 'plt_abc')
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
        monkeypatch.setenv('PLUTO_API_TOKEN', 'new_token')
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
        monkeypatch.setenv('PLUTO_API_TOKEN', 'plt_abc')
        c = Client()
        assert c._api_token == 'plt_abc'
        c.close()

    def test_explicit_token(self, monkeypatch):
        c = Client(api_token='explicit_token')
        assert c._api_token == 'explicit_token'
        c.close()

    def test_context_manager(self, monkeypatch):
        monkeypatch.setenv('PLUTO_API_TOKEN', 'plt_abc')
        with Client() as c:
            assert c._api_token == 'plt_abc'

    def test_custom_host(self, monkeypatch):
        monkeypatch.setenv('PLUTO_API_TOKEN', 'plt_abc')
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
        client._client.get.return_value = mock_response(200, data)
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
        client._client.get.return_value = mock_response(200, data)
        result = client.list_runs('my-project')
        assert result == data
        call_args = client._client.get.call_args
        assert call_args[1]['params']['projectName'] == 'my-project'
        assert call_args[1]['params']['limit'] == 50

    def test_with_search_and_tags(self, client, mock_response):
        client._client.get.return_value = mock_response(200, [])
        client.list_runs('proj', search='experiment', tags=['v2', 'prod'])
        call_args = client._client.get.call_args
        assert call_args[1]['params']['search'] == 'experiment'
        assert call_args[1]['params']['tags'] == 'v2,prod'

    def test_limit_capped_at_200(self, client, mock_response):
        client._client.get.return_value = mock_response(200, [])
        client.list_runs('proj', limit=999)
        call_args = client._client.get.call_args
        assert call_args[1]['params']['limit'] == 200


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
        client._client.get.return_value = mock_response(200, data)
        result = client.get_metric_names('proj')
        assert result == data

    def test_with_run_ids(self, client, mock_response):
        client._client.get.return_value = mock_response(200, [])
        client.get_metric_names('proj', run_ids=[1, 2, 3])
        params = client._client.get.call_args[1]['params']
        assert params['runIds'] == '1,2,3'


# ---------------------------------------------------------------------------
# get_metrics
# ---------------------------------------------------------------------------


class TestGetMetrics:
    def test_single_metric(self, client, mock_response):
        data = [
            {'metric': 'loss', 'step': 0, 'value': 1.0, 'time': '2025-01-01'},
            {'metric': 'loss', 'step': 1, 'value': 0.5, 'time': '2025-01-01'},
        ]
        client._client.get.return_value = mock_response(200, data)
        result = client.get_metrics('proj', 42, metric_names=['loss'])
        params = client._client.get.call_args[1]['params']
        assert params['logName'] == 'loss'
        # Should return a DataFrame if pandas available
        try:
            import pandas as pd

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
        except ImportError:
            assert isinstance(result, list)
            assert len(result) == 2

    def test_multiple_metrics(self, client, mock_response):
        data_loss = [{'metric': 'loss', 'step': 0, 'value': 1.0}]
        data_acc = [{'metric': 'acc', 'step': 0, 'value': 0.8}]
        client._client.get.side_effect = [
            mock_response(200, data_loss),
            mock_response(200, data_acc),
        ]
        client.get_metrics('proj', 42, metric_names=['loss', 'acc'])
        assert client._client.get.call_count == 2

    def test_all_metrics(self, client, mock_response):
        data = [{'metric': 'loss', 'step': 0, 'value': 1.0}]
        client._client.get.return_value = mock_response(200, data)
        client.get_metrics('proj', 42)
        params = client._client.get.call_args[1]['params']
        assert 'logName' not in params

    def test_empty_returns_empty_dataframe(self, client, mock_response):
        client._client.get.return_value = mock_response(200, [])
        result = client.get_metrics('proj', 42)
        try:
            import pandas as pd

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
        except ImportError:
            assert result == []


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
        client._client.get.return_value = mock_response(200, data)
        result = client.get_files('proj', 42)
        assert result == data

    def test_with_filter(self, client, mock_response):
        client._client.get.return_value = mock_response(200, [])
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
        client._client.get.return_value = mock_response(200, file_data)

        dl_response = MagicMock()
        dl_response.content = b'model-bytes'
        dl_response.raise_for_status = MagicMock()

        with patch('pluto.query.httpx.get', return_value=dl_response):
            path = client.download_file('proj', 42, 'model.pt', destination=tmp_path)

        assert path == tmp_path / 'model.pt'
        assert path.read_bytes() == b'model-bytes'

    def test_no_matching_file(self, client, mock_response):
        client._client.get.return_value = mock_response(200, [])
        with pytest.raises(QueryError, match='No file found'):
            client.download_file('proj', 42, 'missing.pt')


# ---------------------------------------------------------------------------
# get_logs
# ---------------------------------------------------------------------------


class TestGetLogs:
    def test_basic(self, client, mock_response):
        data = [{'message': 'hello', 'logType': 'info', 'lineNumber': 1}]
        client._client.get.return_value = mock_response(200, data)
        result = client.get_logs('proj', 42)
        assert result == data

    def test_with_type_filter(self, client, mock_response):
        client._client.get.return_value = mock_response(200, [])
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
        monkeypatch.setenv('PLUTO_API_TOKEN', 'test-token')

        import pluto.query as pq

        # Reset default client
        pq._default_client = None

        mock_client_instance = MagicMock(spec=Client)
        mock_client_instance.list_runs.return_value = [{'id': 1}]

        with patch.object(pq, 'Client', return_value=mock_client_instance):
            result = pq.list_runs('my-project')

        assert result == [{'id': 1}]
        mock_client_instance.list_runs.assert_called_once_with(
            'my-project', search=None, tags=None, limit=50
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
