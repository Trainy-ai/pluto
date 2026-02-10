"""Tests for Konduktor integration: get_konduktor() and auto-tag."""

import os
from unittest.mock import patch

from pluto.sys import System


class TestGetKonduktor:
    """Tests for System.get_konduktor() method."""

    def _make_system(self):
        """Create a System instance with minimal settings."""
        from pluto.sets import setup

        settings = setup(None)
        return System(settings)

    def test_returns_empty_when_not_in_konduktor(self):
        """get_konduktor() returns {} when KONDUKTOR_JOB_NAME is not set."""
        env = {k: v for k, v in os.environ.items() if k != 'KONDUKTOR_JOB_NAME'}
        with patch.dict(os.environ, env, clear=True):
            sys = self._make_system()
            result = sys.get_konduktor()
            assert result == {}

    def test_detects_konduktor_env(self):
        """get_konduktor() detects KONDUKTOR_JOB_NAME and collects metadata."""
        konduktor_env = {
            'KONDUKTOR_JOB_NAME': 'my-training-run-abc1',
            'NUM_NODES': '4',
            'NUM_GPUS_PER_NODE': '8',
            'RANK': '0',
            'MASTER_ADDR': 'my-training-run-abc1-workers-0-0.my-training-run-abc1',
            'KONDUKTOR_ACCELERATOR_TYPE': 'H100',
            'KONDUKTOR_NODENAME': 'gke-a100-pool-abc12',
            'RESTART_ATTEMPT': '0',
            'KONDUKTOR_NAMESPACE': 'team-ml',
        }
        with patch.dict(os.environ, konduktor_env, clear=False):
            sys = self._make_system()
            result = sys.get_konduktor()

            assert result['job_name'] == 'my-training-run-abc1'
            assert result['num_nodes'] == '4'
            assert result['num_gpus_per_node'] == '8'
            assert result['rank'] == '0'
            assert result['accelerator_type'] == 'H100'
            assert result['node_name'] == 'gke-a100-pool-abc12'
            assert result['restart_attempt'] == '0'
            assert result['namespace'] == 'team-ml'

    def test_computes_total_gpus(self):
        """get_konduktor() computes total_gpus from num_gpus_per_node * num_nodes."""
        konduktor_env = {
            'KONDUKTOR_JOB_NAME': 'gpu-count-test',
            'NUM_NODES': '4',
            'NUM_GPUS_PER_NODE': '8',
        }
        with patch.dict(os.environ, konduktor_env, clear=False):
            sys = self._make_system()
            result = sys.get_konduktor()

            assert result['total_gpus'] == 32

    def test_total_gpus_zero_when_no_gpu_info(self):
        """get_konduktor() sets total_gpus=0 when GPU env vars are missing."""
        konduktor_env = {
            'KONDUKTOR_JOB_NAME': 'no-gpu-test',
        }
        # Clear any existing GPU-related env vars
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ('NUM_NODES', 'NUM_GPUS_PER_NODE')
        }
        env_clean.update(konduktor_env)
        with patch.dict(os.environ, env_clean, clear=True):
            sys = self._make_system()
            result = sys.get_konduktor()

            assert result['total_gpus'] == 0

    def test_handles_partial_env_vars(self):
        """get_konduktor() works with only KONDUKTOR_JOB_NAME set."""
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith('KONDUKTOR_')
            and k
            not in (
                'NUM_NODES',
                'NUM_GPUS_PER_NODE',
                'RANK',
                'MASTER_ADDR',
                'RESTART_ATTEMPT',
            )
        }
        env_clean['KONDUKTOR_JOB_NAME'] = 'minimal-test'
        with patch.dict(os.environ, env_clean, clear=True):
            sys = self._make_system()
            result = sys.get_konduktor()

            assert result['job_name'] == 'minimal-test'
            assert result['num_nodes'] is None
            assert result['accelerator_type'] is None
            assert result['node_name'] is None

    def test_get_info_includes_konduktor_when_detected(self):
        """get_info() includes 'konduktor' key when KONDUKTOR_JOB_NAME is set."""
        konduktor_env = {
            'KONDUKTOR_JOB_NAME': 'info-test',
            'NUM_NODES': '2',
            'NUM_GPUS_PER_NODE': '4',
        }
        with patch.dict(os.environ, konduktor_env, clear=False):
            sys = self._make_system()
            info = sys.get_info()

            assert 'konduktor' in info
            assert info['konduktor']['job_name'] == 'info-test'
            assert info['konduktor']['total_gpus'] == 8

    def test_get_info_excludes_konduktor_when_not_detected(self):
        """get_info() does not include 'konduktor' key when not in Konduktor env."""
        env = {k: v for k, v in os.environ.items() if k != 'KONDUKTOR_JOB_NAME'}
        with patch.dict(os.environ, env, clear=True):
            sys = self._make_system()
            info = sys.get_info()

            assert 'konduktor' not in info


def _apply_konduktor_auto_tag(tags, konduktor_job_name=None):
    """Simulate the auto-tag logic from init.py without server calls.

    This mirrors the exact logic in pluto/init.py lines 109-124.
    """
    if konduktor_job_name:
        if tags is None:
            tags = ['konduktor']
        elif isinstance(tags, str):
            tags = [tags, 'konduktor'] if tags != 'konduktor' else [tags]
        elif 'konduktor' not in tags:
            tags = list(tags) + ['konduktor']

    # Normalize (same as init.py)
    normalized_tags = None
    if tags:
        if isinstance(tags, str):
            normalized_tags = [tags]
        else:
            normalized_tags = list(tags)
    return normalized_tags


class TestKonduktorAutoTag:
    """Tests for auto-adding 'konduktor' tag logic (unit tests, no server needed)."""

    def test_auto_tag_added_when_in_konduktor_no_user_tags(self):
        """When in Konduktor with no user tags, ['konduktor'] is produced."""
        result = _apply_konduktor_auto_tag(tags=None, konduktor_job_name='my-job')
        assert result == ['konduktor']

    def test_no_auto_tag_when_not_in_konduktor(self):
        """When not in Konduktor, tags are unchanged."""
        result = _apply_konduktor_auto_tag(tags=None, konduktor_job_name=None)
        assert result is None

    def test_no_auto_tag_preserves_user_tags(self):
        """When not in Konduktor, user tags pass through unchanged."""
        result = _apply_konduktor_auto_tag(tags=['experiment'], konduktor_job_name=None)
        assert result == ['experiment']

    def test_auto_tag_merges_with_list_tags(self):
        """Auto 'konduktor' tag merges with user-provided list tags."""
        result = _apply_konduktor_auto_tag(
            tags=['experiment', 'v2'], konduktor_job_name='my-job'
        )
        assert 'konduktor' in result
        assert 'experiment' in result
        assert 'v2' in result
        assert len(result) == 3

    def test_auto_tag_merges_with_string_tag(self):
        """Auto 'konduktor' tag merges with single string tag."""
        result = _apply_konduktor_auto_tag(
            tags='experiment', konduktor_job_name='my-job'
        )
        assert 'konduktor' in result
        assert 'experiment' in result
        assert len(result) == 2

    def test_no_duplicate_konduktor_tag_in_list(self):
        """If user already provided 'konduktor' in list, no duplicate."""
        result = _apply_konduktor_auto_tag(
            tags=['konduktor', 'other'], konduktor_job_name='my-job'
        )
        konduktor_count = sum(1 for t in result if t == 'konduktor')
        assert konduktor_count == 1
        assert 'other' in result

    def test_no_duplicate_when_string_is_konduktor(self):
        """If user passes tags='konduktor', no duplicate."""
        result = _apply_konduktor_auto_tag(
            tags='konduktor', konduktor_job_name='my-job'
        )
        konduktor_count = sum(1 for t in result if t == 'konduktor')
        assert konduktor_count == 1

    def test_auto_tag_with_empty_list(self):
        """Empty list tags + Konduktor → only 'konduktor'."""
        result = _apply_konduktor_auto_tag(tags=[], konduktor_job_name='my-job')
        # Empty list is falsy, so no normalization; but konduktor check
        # sees it's not None and isinstance list, so adds 'konduktor'
        assert 'konduktor' in result
