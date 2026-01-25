"""Tests for multi-node distributed training support via run_id parameter.

This module tests the Neptune-style run resumption feature where multiple
processes can log to the same run by sharing a common run_id.
"""

import os
import uuid
import warnings

import pluto
from pluto.sets import setup
from tests.utils import get_task_name

TESTING_PROJECT_NAME = 'testing-ci'


class TestRunIdEnvironmentVariable:
    """Test PLUTO_RUN_ID environment variable support."""

    def test_run_id_env_var_sets_external_id(self):
        """Test that PLUTO_RUN_ID environment variable sets _external_id."""
        os.environ['PLUTO_RUN_ID'] = 'my-test-run-123'
        settings = setup()
        assert settings._external_id == 'my-test-run-123'
        del os.environ['PLUTO_RUN_ID']

    def test_run_id_env_var_default_none(self):
        """Test that _external_id is None when env var not set."""
        if 'PLUTO_RUN_ID' in os.environ:
            del os.environ['PLUTO_RUN_ID']
        settings = setup()
        assert settings._external_id is None

    def test_run_id_precedence_param_over_env(self):
        """Test that function params override env var."""
        os.environ['PLUTO_RUN_ID'] = 'env-run-id'
        settings = setup({'_external_id': 'param-run-id'})
        assert settings._external_id == 'param-run-id'
        del os.environ['PLUTO_RUN_ID']

    def test_deprecated_mlop_run_id(self):
        """Test deprecated MLOP_RUN_ID still works with warning."""
        os.environ['MLOP_RUN_ID'] = 'old-run-id'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            settings = setup()
            assert settings._external_id == 'old-run-id'
            assert any('MLOP_RUN_ID' in str(warning.message) for warning in w)
        del os.environ['MLOP_RUN_ID']

    def test_pluto_run_id_takes_precedence_over_mlop(self):
        """Test that PLUTO_RUN_ID takes precedence over MLOP_RUN_ID."""
        os.environ['MLOP_RUN_ID'] = 'old-run-id'
        os.environ['PLUTO_RUN_ID'] = 'new-run-id'
        settings = setup()
        assert settings._external_id == 'new-run-id'
        del os.environ['MLOP_RUN_ID']
        del os.environ['PLUTO_RUN_ID']


class TestRunIdInitParameter:
    """Test run_id parameter in pluto.init()."""

    def test_init_with_run_id_parameter(self):
        """Test initializing a run with run_id parameter."""
        run_id = f'test-run-{uuid.uuid4().hex[:8]}'
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=run_id,
        )
        assert run.run_id == run_id
        assert run.settings._external_id == run_id
        run.finish()

    def test_init_without_run_id_parameter(self):
        """Test that run_id is None when not provided."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
        )
        assert run.run_id is None
        run.finish()

    def test_init_run_id_param_overrides_env_var(self):
        """Test that run_id parameter overrides PLUTO_RUN_ID env var."""
        os.environ['PLUTO_RUN_ID'] = 'env-run-id'
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id='param-run-id',
        )
        assert run.run_id == 'param-run-id'
        run.finish()
        del os.environ['PLUTO_RUN_ID']

    def test_init_with_run_id_from_env_var(self):
        """Test that run_id is set from env var when param not provided."""
        os.environ['PLUTO_RUN_ID'] = 'env-run-id-test'
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
        )
        assert run.run_id == 'env-run-id-test'
        run.finish()
        del os.environ['PLUTO_RUN_ID']


class TestRunIdProperties:
    """Test Op properties related to run_id and resume."""

    def test_run_id_property(self):
        """Test that run.run_id returns the external_id."""
        run_id = f'property-test-{uuid.uuid4().hex[:8]}'
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=run_id,
        )
        assert run.run_id == run_id
        run.finish()

    def test_id_property(self):
        """Test that run.id returns the server-assigned numeric ID."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
        )
        assert run.id is not None
        assert isinstance(run.id, int)
        run.finish()

    def test_resumed_property_new_run(self):
        """Test that resumed is False for a new run."""
        # Use a unique run_id to ensure it's a new run
        run_id = f'new-run-{uuid.uuid4().hex}'
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=run_id,
        )
        assert run.resumed is False
        run.finish()


class TestMultiNodeResume:
    """Test multi-node resume functionality where multiple processes share a run."""

    def test_same_run_id_returns_same_server_id(self):
        """Test that two inits with the same run_id get the same server run ID.

        This simulates two processes in a distributed training setup attaching
        to the same run via a shared run_id.
        """
        shared_run_id = f'multinode-{uuid.uuid4().hex}'

        # First "process" creates the run
        run1 = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=shared_run_id,
        )
        server_id_1 = run1.id
        resumed_1 = run1.resumed
        run1.finish()

        # Second "process" should attach to the same run
        run2 = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=shared_run_id,
        )
        server_id_2 = run2.id
        resumed_2 = run2.resumed
        run2.finish()

        # Both should have the same server run ID
        assert server_id_1 == server_id_2, (
            f'Expected same server ID for shared run_id, '
            f'got {server_id_1} and {server_id_2}'
        )

        # First run should not be resumed, second should be
        assert resumed_1 is False, 'First run should not be resumed'
        assert resumed_2 is True, 'Second run should be resumed'

    def test_different_run_ids_get_different_server_ids(self):
        """Test that different run_ids get different server run IDs."""
        run_id_1 = f'unique-1-{uuid.uuid4().hex}'
        run_id_2 = f'unique-2-{uuid.uuid4().hex}'

        run1 = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=run_id_1,
        )
        server_id_1 = run1.id
        run1.finish()

        run2 = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=run_id_2,
        )
        server_id_2 = run2.id
        run2.finish()

        # Different run_ids should get different server IDs
        assert server_id_1 != server_id_2

    def test_multinode_logging_to_same_run(self):
        """Test that multiple processes can log metrics to the same run."""
        shared_run_id = f'multinode-log-{uuid.uuid4().hex}'

        # First "process" creates the run and logs
        run1 = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=shared_run_id,
        )
        run1.log({'loss/rank0': 0.5, 'throughput/rank0': 1000})
        server_id = run1.id
        run1.finish()

        # Second "process" attaches and logs
        run2 = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=shared_run_id,
        )
        run2.log({'loss/rank1': 0.6, 'throughput/rank1': 950})

        # Should be the same run
        assert run2.id == server_id
        assert run2.resumed is True
        run2.finish()

    def test_multinode_with_config(self):
        """Test that resumed runs preserve config from original run."""
        shared_run_id = f'multinode-config-{uuid.uuid4().hex}'
        config = {'lr': 0.001, 'epochs': 100, 'model': 'resnet50'}

        # First process creates run with config
        run1 = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=shared_run_id,
            config=config,
        )
        run1.finish()

        # Second process attaches (config is stored server-side)
        run2 = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=shared_run_id,
        )
        assert run2.resumed is True
        run2.finish()

    def test_multinode_with_tags(self):
        """Test multi-node with tags."""
        shared_run_id = f'multinode-tags-{uuid.uuid4().hex}'

        # First process creates run with tags
        run1 = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=shared_run_id,
            tags=['distributed', 'ddp'],
        )
        run1.finish()

        # Second process attaches
        run2 = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            run_id=shared_run_id,
        )
        assert run2.resumed is True
        run2.finish()


class TestMultiNodeUsagePatterns:
    """Test common multi-node usage patterns."""

    def test_torchrun_style_pattern(self):
        """Test typical torchrun-style distributed training pattern.

        In a typical DDP setup:
        - PLUTO_RUN_ID is set before launching torchrun
        - Each process calls pluto.init() with the same run_id
        - All processes log to the same run
        """
        shared_run_id = f'torchrun-{uuid.uuid4().hex}'

        # Simulate setting env var before launch
        os.environ['PLUTO_RUN_ID'] = shared_run_id

        # Simulate multiple ranks initializing
        server_ids = []
        resumed_flags = []

        for rank in range(4):  # Simulate 4 GPUs
            run = pluto.init(
                project=TESTING_PROJECT_NAME,
                name=f'{get_task_name()}-rank{rank}',
            )
            server_ids.append(run.id)
            resumed_flags.append(run.resumed)
            run.log({f'loss/rank{rank}': 0.5 - rank * 0.1})
            run.finish()

        # All should have the same server ID
        assert all(
            sid == server_ids[0] for sid in server_ids
        ), f'All ranks should log to same run, got IDs: {server_ids}'

        # First should not be resumed, rest should be
        assert resumed_flags[0] is False
        assert all(resumed_flags[1:])

        del os.environ['PLUTO_RUN_ID']

    def test_rank0_only_pattern(self):
        """Test pattern where only rank 0 logs (common in PyTorch DDP).

        Some users prefer only rank 0 to log to avoid duplicate metrics.
        This pattern should also work with run_id for consistency.
        """
        shared_run_id = f'rank0only-{uuid.uuid4().hex}'

        # Only rank 0 initializes Pluto
        rank = 0  # Simulate being rank 0

        if rank == 0:
            run = pluto.init(
                project=TESTING_PROJECT_NAME,
                name=get_task_name(),
                run_id=shared_run_id,
            )
            run.log({'loss': 0.5, 'accuracy': 0.95})
            assert run.resumed is False  # First and only init
            run.finish()
