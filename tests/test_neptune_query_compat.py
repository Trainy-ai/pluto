"""Tests for pluto.compat.neptune_query compatibility shim."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pluto.compat.neptune_query import runs as nq_runs
from pluto.compat.neptune_query.filters import (
    AttributeFilter,
    Filter,
    _resolve_attribute,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_RUN = {
    'id': 42,
    'displayId': 'MQT-1498',
    'name': 'my-training-run',
    'status': 'finished',
    'createdAt': '2025-06-01T00:00:00Z',
    'updatedAt': '2025-06-01T12:00:00Z',
    'tags': ['production'],
    'config': {
        'model/best_model_path': 's3://bucket/model.pt',
        'runtime/docker_tag': 'c9fed5d5',
        'global_step': 10000,
        'nested': {'key': 'value'},
    },
    'logNames': ['train/loss', 'test/raw/elec_H/MSE[mean]', 'hydra/config.yaml'],
}

SAMPLE_RUN_2 = {
    'id': 43,
    'displayId': 'MQT-1499',
    'name': 'another-run',
    'status': 'finished',
    'createdAt': '2025-06-02T00:00:00Z',
    'config': {},
    'logNames': [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level state between tests."""
    nq_runs._client = None
    nq_runs._run_cache.clear()
    nq_runs._last_project = None
    yield
    nq_runs._client = None
    nq_runs._run_cache.clear()
    nq_runs._last_project = None


@pytest.fixture()
def mock_client():
    """Return a mocked pluto.query.Client instance."""
    client = MagicMock()
    return client


@pytest.fixture()
def patch_client(mock_client):
    """Patch _get_client to return the mock client."""
    with patch.object(nq_runs, '_get_client', return_value=mock_client):
        yield mock_client


# ===========================================================================
# Filter tests
# ===========================================================================


class TestResolveAttribute:
    def test_sys_id(self):
        assert _resolve_attribute(SAMPLE_RUN, 'sys/id') == 'MQT-1498'

    def test_sys_name(self):
        assert _resolve_attribute(SAMPLE_RUN, 'sys/name') == 'my-training-run'

    def test_sys_creation_time(self):
        val = _resolve_attribute(SAMPLE_RUN, 'sys/creation_time')
        assert val == '2025-06-01T00:00:00Z'

    def test_bare_id(self):
        assert _resolve_attribute(SAMPLE_RUN, 'id') == 'MQT-1498'

    def test_flat_config_key(self):
        assert (
            _resolve_attribute(SAMPLE_RUN, 'model/best_model_path')
            == 's3://bucket/model.pt'
        )

    def test_nested_config_key(self):
        assert _resolve_attribute(SAMPLE_RUN, 'nested/key') == 'value'

    def test_log_name_exists(self):
        assert _resolve_attribute(SAMPLE_RUN, 'hydra/config.yaml') is True

    def test_missing_attribute(self):
        assert _resolve_attribute(SAMPLE_RUN, 'nonexistent/attr') is None

    def test_global_step_from_config(self):
        assert _resolve_attribute(SAMPLE_RUN, 'global_step') == 10000


class TestFilter:
    def test_exists_pass(self):
        f = Filter.exists('model/best_model_path')
        assert f.evaluate(SAMPLE_RUN) is True

    def test_exists_fail(self):
        f = Filter.exists('model/best_model_path')
        assert f.evaluate(SAMPLE_RUN_2) is False

    def test_matches_pass(self):
        f = Filter.matches('sys/id', 'MQT-1498')
        assert f.evaluate(SAMPLE_RUN) is True

    def test_matches_fail(self):
        f = Filter.matches('sys/id', 'MQT-9999')
        assert f.evaluate(SAMPLE_RUN) is False

    def test_and_both_pass(self):
        f = Filter.exists('model/best_model_path')
        f = f & Filter.matches('sys/id', 'MQT-1498')
        assert f.evaluate(SAMPLE_RUN) is True

    def test_and_one_fails(self):
        f = Filter.exists('model/best_model_path')
        f = f & Filter.matches('sys/id', 'MQT-9999')
        assert f.evaluate(SAMPLE_RUN) is False

    def test_iand(self):
        f = Filter.exists('model/best_model_path')
        f &= Filter.matches('sys/id', 'MQT-1498')
        assert f.evaluate(SAMPLE_RUN) is True

    def test_get_match_value(self):
        f = Filter.matches('sys/id', 'MQT-1498')
        assert f.get_match_value('sys/id') == 'MQT-1498'

    def test_get_match_value_none(self):
        f = Filter.exists('model/best_model_path')
        assert f.get_match_value('sys/id') is None


class TestAttributeFilter:
    def test_regex_match(self):
        af = AttributeFilter(name=r'test/raw/')
        assert af.matches_name('test/raw/elec_H/MSE[mean]') is True
        assert af.matches_name('train/loss') is False

    def test_list_exact_match(self):
        af = AttributeFilter(name=['loss', 'accuracy'])
        assert af.matches_name('loss') is True
        assert af.matches_name('accuracy') is True
        assert af.matches_name('val/loss') is False

    def test_type_ignored(self):
        af = AttributeFilter(name='test/', type='float_series')
        assert af.matches_name('test/loss') is True


# ===========================================================================
# runs.py helper tests
# ===========================================================================


class TestNormalizeProject:
    def test_with_workspace(self):
        assert nq_runs._normalize_project('tfc/MQTransformer') == 'MQTransformer'

    def test_without_workspace(self):
        assert nq_runs._normalize_project('MQTransformer') == 'MQTransformer'


class TestParseAttributes:
    def test_pipe_separated(self):
        assert nq_runs._parse_attributes('a | b | c') == ['a', 'b', 'c']

    def test_single(self):
        assert nq_runs._parse_attributes('runtime/docker_tag') == ['runtime/docker_tag']

    def test_extra_whitespace(self):
        assert nq_runs._parse_attributes(' a |  b ') == ['a', 'b']


# ===========================================================================
# list_runs tests
# ===========================================================================


class TestListRuns:
    def test_direct_lookup_by_sys_id(self, patch_client):
        patch_client.get_run.return_value = SAMPLE_RUN
        f = Filter.exists('model/best_model_path')
        f = f & Filter.matches('sys/id', 'MQT-1498')
        result = nq_runs.list_runs(project='tfc/MQTransformer', runs=f)
        assert result == ['MQT-1498']
        patch_client.get_run.assert_called_once_with('MQTransformer', 'MQT-1498')

    def test_direct_lookup_filter_rejects(self, patch_client):
        patch_client.get_run.return_value = SAMPLE_RUN_2
        f = Filter.exists('model/best_model_path')
        f = f & Filter.matches('sys/id', 'MQT-1499')
        result = nq_runs.list_runs(project='tfc/MQTransformer', runs=f)
        assert result == []

    def test_direct_lookup_not_found(self, patch_client):
        patch_client.get_run.side_effect = Exception('404 not found')
        f = Filter.matches('sys/id', 'MQT-9999')
        result = nq_runs.list_runs(project='tfc/MQTransformer', runs=f)
        assert result == []

    def test_general_filter(self, patch_client):
        patch_client.list_runs.return_value = [SAMPLE_RUN, SAMPLE_RUN_2]
        f = Filter.exists('model/best_model_path')
        result = nq_runs.list_runs(project='tfc/MQTransformer', runs=f)
        assert result == ['MQT-1498']
        patch_client.list_runs.assert_called_once_with('MQTransformer', limit=200)

    def test_no_filter(self, patch_client):
        patch_client.list_runs.return_value = [SAMPLE_RUN, SAMPLE_RUN_2]
        result = nq_runs.list_runs(project='MQTransformer')
        assert result == ['MQT-1498', 'MQT-1499']

    def test_sets_last_project(self, patch_client):
        patch_client.list_runs.return_value = []
        nq_runs.list_runs(project='tfc/MQTransformer')
        assert nq_runs._last_project == 'MQTransformer'


# ===========================================================================
# fetch_runs_table tests
# ===========================================================================


class TestFetchRunsTable:
    def test_single_run_multiple_attrs(self, patch_client):
        patch_client.get_run.return_value = SAMPLE_RUN
        df = nq_runs.fetch_runs_table(
            project='tfc/MQTransformer',
            runs=['MQT-1498'],
            attributes='runtime/docker_tag | model/best_model_path',
        )
        assert df.index.name == 'run'
        assert list(df.index) == ['MQT-1498']
        assert df.loc['MQT-1498', 'runtime/docker_tag'] == 'c9fed5d5'
        assert df.loc['MQT-1498', 'model/best_model_path'] == 's3://bucket/model.pt'

    def test_string_run_id(self, patch_client):
        patch_client.get_run.return_value = SAMPLE_RUN
        df = nq_runs.fetch_runs_table(
            project='MQTransformer',
            runs='MQT-1498',
            attributes='runtime/docker_tag',
        )
        assert list(df.index) == ['MQT-1498']

    def test_sort_by(self, patch_client):
        patch_client.get_run.side_effect = [SAMPLE_RUN, SAMPLE_RUN_2]
        df = nq_runs.fetch_runs_table(
            project='MQTransformer',
            runs=['MQT-1498', 'MQT-1499'],
            attributes='sys/creation_time',
            sort_by='sys/creation_time',
        )
        # MQT-1498 has earlier creation time, should come first
        assert list(df.index) == ['MQT-1498', 'MQT-1499']

    def test_empty_runs(self, patch_client):
        df = nq_runs.fetch_runs_table(
            project='MQTransformer',
            runs=[],
            attributes='a | b',
        )
        assert df.empty
        assert list(df.columns) == ['a', 'b']

    def test_sys_id_attribute(self, patch_client):
        patch_client.get_run.return_value = SAMPLE_RUN
        df = nq_runs.fetch_runs_table(
            project='MQTransformer',
            runs=['MQT-1498'],
            attributes='id | global_step',
        )
        assert df.loc['MQT-1498', 'id'] == 'MQT-1498'
        assert df.loc['MQT-1498', 'global_step'] == 10000


# ===========================================================================
# fetch_metrics tests
# ===========================================================================


class TestFetchMetrics:
    def test_regex_filter(self, patch_client):
        patch_client.get_run.return_value = SAMPLE_RUN
        patch_client.get_metric_names.return_value = [
            'test/raw/elec_H/MSE[mean]',
            'test/raw/elec_H/MAE[0.5]',
            'train/loss',
        ]
        long_data = pd.DataFrame(
            [
                {'metric': 'test/raw/elec_H/MSE[mean]', 'step': 0, 'value': 1.5},
                {'metric': 'test/raw/elec_H/MSE[mean]', 'step': 1, 'value': 1.2},
                {'metric': 'test/raw/elec_H/MAE[0.5]', 'step': 0, 'value': 0.8},
                {'metric': 'test/raw/elec_H/MAE[0.5]', 'step': 1, 'value': 0.7},
            ]
        )
        patch_client.get_metrics.return_value = long_data

        df = nq_runs.fetch_metrics(
            project='tfc/MQTransformer',
            runs='MQT-1498',
            attributes=AttributeFilter(name=r'test/raw/', type='float_series'),
        )

        assert df.index.name == 'step'
        assert 'test/raw/elec_H/MSE[mean]' in df.columns
        assert 'test/raw/elec_H/MAE[0.5]' in df.columns
        assert 'train/loss' not in df.columns
        # Verify metric names passed to get_metrics
        patch_client.get_metrics.assert_called_once_with(
            'MQTransformer',
            42,
            metric_names=['test/raw/elec_H/MSE[mean]', 'test/raw/elec_H/MAE[0.5]'],
        )

    def test_exact_name_list(self, patch_client):
        patch_client.get_run.return_value = SAMPLE_RUN
        patch_client.get_metric_names.return_value = [
            'test/eval_metrics/mean_weighted_sum_quantile_loss',
            'test/raw/elec_H/MSE[mean]',
        ]
        long_data = pd.DataFrame(
            [
                {
                    'metric': 'test/eval_metrics/mean_weighted_sum_quantile_loss',
                    'step': 0,
                    'value': 2.3,
                },
            ]
        )
        patch_client.get_metrics.return_value = long_data

        df = nq_runs.fetch_metrics(
            project='MQTransformer',
            runs=['MQT-1498'],
            attributes=AttributeFilter(
                name=['test/eval_metrics/mean_weighted_sum_quantile_loss'],
                type='float_series',
            ),
        )

        assert 'test/eval_metrics/mean_weighted_sum_quantile_loss' in df.columns
        assert len(df.columns) == 1

    def test_no_matching_metrics(self, patch_client):
        patch_client.get_run.return_value = SAMPLE_RUN
        patch_client.get_metric_names.return_value = ['train/loss']

        df = nq_runs.fetch_metrics(
            project='MQTransformer',
            runs='MQT-1498',
            attributes=AttributeFilter(name=r'nonexistent/'),
        )

        assert df.empty

    def test_reset_index_chain(self, patch_client):
        """Verify the .reset_index() pattern used in navi code works."""
        patch_client.get_run.return_value = SAMPLE_RUN
        patch_client.get_metric_names.return_value = ['train/loss']
        long_data = pd.DataFrame([{'metric': 'train/loss', 'step': 0, 'value': 1.0}])
        patch_client.get_metrics.return_value = long_data

        df = nq_runs.fetch_metrics(
            project='MQTransformer',
            runs='MQT-1498',
        ).reset_index()

        assert 'step' in df.columns


# ===========================================================================
# download_files tests
# ===========================================================================


class TestDownloadFiles:
    def test_download(self, patch_client, tmp_path):
        # Set up project context
        patch_client.get_run.return_value = SAMPLE_RUN
        nq_runs._last_project = 'MQTransformer'

        download_path = tmp_path / 'MQT-1498' / 'config.yaml'
        patch_client.download_file.return_value = download_path

        files_df = pd.DataFrame(
            [{'run': 'MQT-1498', 'hydra/config.yaml': True}]
        ).set_index('run')

        result = nq_runs.download_files(files=files_df, destination=tmp_path)

        assert result.loc['MQT-1498', 'hydra/config.yaml'] == str(download_path)
        patch_client.download_file.assert_called_once_with(
            'MQTransformer', 42, 'hydra/config.yaml', destination=tmp_path / 'MQT-1498'
        )

    def test_download_failure_graceful(self, patch_client, tmp_path):
        patch_client.get_run.return_value = SAMPLE_RUN
        patch_client.download_file.side_effect = Exception('download failed')
        nq_runs._last_project = 'MQTransformer'

        files_df = pd.DataFrame([{'run': 'MQT-1498', 'missing/file': True}]).set_index(
            'run'
        )

        # Should not raise
        result = nq_runs.download_files(files=files_df, destination=tmp_path)
        # Value unchanged on failure
        assert result.loc['MQT-1498', 'missing/file'] == True  # noqa: E712

    def test_infer_project_fails_without_prior_call(self, patch_client):
        nq_runs._last_project = None
        files_df = pd.DataFrame([{'run': 'MQT-1498', 'file': True}]).set_index('run')

        with pytest.raises(ValueError, match='Cannot infer project'):
            nq_runs.download_files(files=files_df, destination='/tmp')


# ===========================================================================
# Run cache tests
# ===========================================================================


class TestRunCache:
    def test_caches_run_lookups(self, patch_client):
        patch_client.get_run.return_value = SAMPLE_RUN
        nq_runs._get_run_cached('MQTransformer', 'MQT-1498')
        nq_runs._get_run_cached('MQTransformer', 'MQT-1498')
        # Only one API call despite two lookups
        patch_client.get_run.assert_called_once()


# ===========================================================================
# Import compatibility tests
# ===========================================================================


class TestImports:
    def test_import_runs(self):
        from pluto.compat.neptune_query import runs

        assert hasattr(runs, 'list_runs')
        assert hasattr(runs, 'fetch_runs_table')
        assert hasattr(runs, 'fetch_metrics')
        assert hasattr(runs, 'download_files')

    def test_import_filters(self):
        from pluto.compat.neptune_query.filters import AttributeFilter, Filter

        assert Filter is not None
        assert AttributeFilter is not None

    def test_import_from_package(self):
        from pluto.compat.neptune_query import filters, runs

        assert filters is not None
        assert runs is not None
