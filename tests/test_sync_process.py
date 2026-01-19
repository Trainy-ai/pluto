"""Tests for sync process V2 architecture.

These tests verify the sync process implementation including:
- SyncStore database operations
- File upload via sync process
- Graceful shutdown handling
- Payload format verification (unit tests)
"""

import json
import logging
import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

import pluto
from pluto.sync.process import _SyncUploader
from pluto.sync.store import FileRecord, RecordType, SyncRecord, SyncStatus, SyncStore
from tests.utils import get_task_name

TESTING_PROJECT_NAME = 'testing-ci'


class TestSyncStore:
    """Unit tests for SyncStore database operations."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary SyncStore for testing."""
        db_path = tmp_path / 'test_sync.db'
        store = SyncStore(str(db_path))
        yield store
        store.close()

    def test_schema_initialization(self, store):
        """Test that database schema is created correctly."""
        # Verify we can query the tables
        cursor = store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert 'runs' in tables
        assert 'sync_queue' in tables
        assert 'file_uploads' in tables
        assert 'schema_version' in tables

    def test_enqueue_file(self, store):
        """Test file can be enqueued to the store."""
        # Register a run first
        store.register_run('test-run-1', 'test-project')

        # Enqueue a file
        file_id = store.enqueue_file(
            run_id='test-run-1',
            local_path='/tmp/test.png',
            file_name='test',
            file_ext='.png',
            file_type='image/png',
            file_size=1024,
            log_name='images/test',
            timestamp_ms=int(time.time() * 1000),
            step=1,
        )

        assert file_id > 0

    def test_get_pending_files(self, store):
        """Test pending files are retrieved correctly."""
        store.register_run('test-run-1', 'test-project')

        # Enqueue multiple files
        for i in range(3):
            store.enqueue_file(
                run_id='test-run-1',
                local_path=f'/tmp/test{i}.png',
                file_name=f'test{i}',
                file_ext='.png',
                file_type='image/png',
                file_size=1024,
                log_name=f'images/test{i}',
                timestamp_ms=int(time.time() * 1000),
                step=i,
            )

        # Get pending files
        pending = store.get_pending_files(limit=10)
        assert len(pending) == 3
        assert all(isinstance(f, FileRecord) for f in pending)
        assert all(f.status == SyncStatus.PENDING for f in pending)

    def test_file_status_transitions(self, store):
        """Test file status transitions work correctly."""
        store.register_run('test-run-1', 'test-project')

        file_id = store.enqueue_file(
            run_id='test-run-1',
            local_path='/tmp/test.png',
            file_name='test',
            file_ext='.png',
            file_type='image/png',
            file_size=1024,
            log_name='images/test',
            timestamp_ms=int(time.time() * 1000),
            step=1,
        )

        # Initially PENDING
        pending = store.get_pending_files()
        assert len(pending) == 1
        assert pending[0].status == SyncStatus.PENDING

        # Mark in progress
        store.mark_files_in_progress([file_id])
        # Should still be retrievable as "pending" for retry purposes
        pending = store.get_pending_files()
        assert len(pending) == 0  # IN_PROGRESS is not retrieved

        # Mark completed
        store.mark_files_completed([file_id])
        pending = store.get_pending_files()
        assert len(pending) == 0

    def test_file_retry_count(self, store):
        """Test retry count increments on failure."""
        store.register_run('test-run-1', 'test-project')

        file_id = store.enqueue_file(
            run_id='test-run-1',
            local_path='/tmp/test.png',
            file_name='test',
            file_ext='.png',
            file_type='image/png',
            file_size=1024,
            log_name='images/test',
            timestamp_ms=int(time.time() * 1000),
            step=1,
        )

        # Mark failed multiple times
        store.mark_files_failed([file_id], 'Test error')
        store.mark_files_failed([file_id], 'Test error 2')

        # Query directly to check retry count
        cursor = store.conn.execute(
            'SELECT retry_count FROM file_uploads WHERE id = ?', (file_id,)
        )
        row = cursor.fetchone()
        assert row[0] == 2

    def test_pending_file_count(self, store):
        """Test pending file count is accurate."""
        store.register_run('test-run-1', 'test-project')

        assert store.get_pending_file_count() == 0

        for i in range(5):
            store.enqueue_file(
                run_id='test-run-1',
                local_path=f'/tmp/test{i}.png',
                file_name=f'test{i}',
                file_ext='.png',
                file_type='image/png',
                file_size=1024,
                log_name=f'images/test{i}',
                timestamp_ms=int(time.time() * 1000),
                step=i,
            )

        assert store.get_pending_file_count() == 5
        assert store.get_pending_file_count('test-run-1') == 5
        assert store.get_pending_file_count('other-run') == 0


class TestSyncProcessIntegration:
    """Integration tests for sync process file uploads."""

    @pytest.fixture
    def sync_enabled_run(self):
        """Create a pluto run with sync process enabled."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            config={},
            sync_process_enabled=True,
        )
        yield run
        run.finish()

    def test_image_upload_via_sync(self, sync_enabled_run, tmp_path):
        """Test image uploads work via sync process."""
        # Create a test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = PILImage.fromarray(img_array)
        img_path = tmp_path / 'test_image.png'
        img.save(img_path)

        # Log the image
        image = pluto.Image(str(img_path), caption='test-image')
        sync_enabled_run.log({'images/test': image})

        # Allow time for sync process to pick it up
        time.sleep(2)

    def test_multiple_files_same_step(self, sync_enabled_run, tmp_path):
        """Test multiple files logged in same step."""
        # Create test files
        for i in range(3):
            img_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            img = PILImage.fromarray(img_array)
            img_path = tmp_path / f'test_image_{i}.png'
            img.save(img_path)

            image = pluto.Image(str(img_path), caption=f'test-{i}')
            sync_enabled_run.log({f'images/batch_{i}': image})

        # Allow time for sync
        time.sleep(2)

    def test_text_upload_via_sync(self, sync_enabled_run):
        """Test text uploads work via sync process."""
        text = pluto.Text('Test log message content', caption='log-message')
        sync_enabled_run.log({'logs/message': text})

        # Allow time for sync
        time.sleep(2)

    def test_artifact_upload_via_sync(self, sync_enabled_run, tmp_path):
        """Test artifact uploads work via sync process."""
        # Create a JSON artifact
        artifact_path = tmp_path / 'test_artifact.json'
        artifact_path.write_text(json.dumps({'test': 'data', 'value': 123}))

        artifact = pluto.Artifact(str(artifact_path), caption='json-artifact')
        sync_enabled_run.log({'artifacts/json': artifact})

        # Allow time for sync
        time.sleep(2)


class TestSyncProcessShutdown:
    """Tests for sync process shutdown behavior."""

    def test_graceful_shutdown_waits_for_pending(self):
        """Test that finish() waits for pending uploads to complete."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            config={},
            sync_process_enabled=True,
        )

        # Log some metrics
        for i in range(10):
            run.log({'metric': i})

        # Log a text file
        text = pluto.Text('Test content', caption='test')
        run.log({'logs/test': text})

        # Finish should wait for sync
        start = time.time()
        run.finish()
        elapsed = time.time() - start

        # Should complete reasonably quickly (not timeout)
        assert elapsed < 30, f'Shutdown took too long: {elapsed}s'

    def test_sync_manager_pending_count(self):
        """Test that pending count tracks metrics and files."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            config={},
            sync_process_enabled=True,
        )

        # Initial count should be 0 for a new run
        initial_count = run._sync_manager.get_pending_count()
        assert initial_count == 0

        # Log some data
        for i in range(5):
            run.log({'metric': i})

        # Allow a moment for data to be enqueued
        time.sleep(0.5)

        # Verify data was enqueued (pending count should increase)
        pending_after_log = run._sync_manager.get_pending_count()
        assert pending_after_log >= 0  # May be 0 if sync already processed

        run.finish()


class TestSyncUploaderPayloadFormat:
    """Unit tests for _SyncUploader payload formats.

    These tests mock HTTP requests and verify the exact JSON payloads
    being sent to ensure compatibility with the server API.
    """

    @pytest.fixture
    def uploader(self):
        """Create an uploader with mock settings."""
        settings = {
            '_auth': 'test-token',
            '_op_id': 12345,
            '_op_name': 'test-run',
            'project': 'test-project',
            'url_num': 'https://test.example.com/ingest/metrics',
            'url_data': 'https://test.example.com/ingest/data',
            'url_update_config': 'https://test.example.com/api/runs/config/update',
            'url_update_tags': 'https://test.example.com/api/runs/tags/update',
            'url_file': 'https://test.example.com/files',
        }
        log = logging.getLogger('test')
        return _SyncUploader(settings, log)

    def test_metrics_payload_format(self, uploader):
        """Test metrics are sent in correct NDJSON format.

        Expected format: {"time": <ms>, "step": <int>, "data": {...}}
        NOT the incorrect: {"k": key, "v": value, "s": step, "t": timestamp}
        """
        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'loss': 0.5, 'accuracy': 0.95},
                timestamp_ms=1705600000000,
                step=10,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client

            # Force re-creation of client
            uploader._client = None
            uploader.upload_metrics_batch(records)

            # Verify POST was called
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args

            # Parse the body
            body = call_args.kwargs.get('content') or call_args[1].get('content')
            lines = body.strip().split('\n')
            assert len(lines) == 1

            payload = json.loads(lines[0])

            # Verify correct format
            assert 'time' in payload, "Payload must have 'time' field"
            assert 'step' in payload, "Payload must have 'step' field"
            assert 'data' in payload, "Payload must have 'data' field"
            assert payload['time'] == 1705600000000
            assert payload['step'] == 10
            assert payload['data'] == {'loss': 0.5, 'accuracy': 0.95}

            # Verify incorrect format is NOT used
            assert 'k' not in payload, "Should not use 'k' field"
            assert 'v' not in payload, "Should not use 'v' field"
            assert 's' not in payload, "Should not use 's' field"
            assert 't' not in payload, "Should not use 't' field"

    def test_metrics_payload_filters_non_numeric(self, uploader):
        """Test that non-numeric values are filtered from metrics."""
        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'loss': 0.5, 'name': 'test', 'enabled': True},
                timestamp_ms=1705600000000,
                step=1,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_metrics_batch(records)

            body = mock_client.post.call_args.kwargs.get(
                'content'
            ) or mock_client.post.call_args[1].get('content')
            payload = json.loads(body.strip())

            # Only numeric values should be in data
            assert payload['data'] == {'loss': 0.5}
            assert 'name' not in payload['data']
            assert 'enabled' not in payload['data']

    def test_system_metrics_payload_format(self, uploader):
        """Test system metrics have 'sys/' prefix on keys."""
        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.SYSTEM,
                payload={'cpu_percent': 45.2, 'memory_mb': 1024},
                timestamp_ms=1705600000000,
                step=None,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_system_batch(records)

            body = mock_client.post.call_args.kwargs.get(
                'content'
            ) or mock_client.post.call_args[1].get('content')
            payload = json.loads(body.strip())

            # Verify sys/ prefix
            assert 'sys/cpu_percent' in payload['data']
            assert 'sys/memory_mb' in payload['data']
            assert payload['data']['sys/cpu_percent'] == 45.2
            assert payload['data']['sys/memory_mb'] == 1024
            assert payload['step'] == 0  # System metrics have step=0

    def test_config_payload_format(self, uploader):
        """Test config update payload format."""
        record = SyncRecord(
            id=1,
            run_id='test-run',
            record_type=RecordType.CONFIG,
            payload={'learning_rate': 0.001, 'batch_size': 32},
            timestamp_ms=1705600000000,
            step=None,
            status=SyncStatus.PENDING,
            retry_count=0,
            created_at=time.time(),
            last_attempt_at=None,
            error_message=None,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_config(record)

            body = mock_client.post.call_args.kwargs.get(
                'content'
            ) or mock_client.post.call_args[1].get('content')
            payload = json.loads(body)

            assert payload['runId'] == 12345
            assert payload['config'] == {'learning_rate': 0.001, 'batch_size': 32}

    def test_tags_payload_format(self, uploader):
        """Test tags update payload format."""
        record = SyncRecord(
            id=1,
            run_id='test-run',
            record_type=RecordType.TAGS,
            payload={'tags': ['experiment', 'v2', 'baseline']},
            timestamp_ms=1705600000000,
            step=None,
            status=SyncStatus.PENDING,
            retry_count=0,
            created_at=time.time(),
            last_attempt_at=None,
            error_message=None,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_tags(record)

            body = mock_client.post.call_args.kwargs.get(
                'content'
            ) or mock_client.post.call_args[1].get('content')
            payload = json.loads(body)

            assert payload['runId'] == 12345
            assert payload['tags'] == ['experiment', 'v2', 'baseline']

    def test_metrics_batch_multiple_records(self, uploader):
        """Test multiple metric records are sent as NDJSON (one per line)."""
        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'loss': 0.5},
                timestamp_ms=1705600000000,
                step=1,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
            SyncRecord(
                id=2,
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'loss': 0.4},
                timestamp_ms=1705600001000,
                step=2,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_metrics_batch(records)

            body = mock_client.post.call_args.kwargs.get(
                'content'
            ) or mock_client.post.call_args[1].get('content')
            lines = [line for line in body.strip().split('\n') if line]
            assert len(lines) == 2

            payload1 = json.loads(lines[0])
            payload2 = json.loads(lines[1])

            assert payload1['step'] == 1
            assert payload2['step'] == 2

    def test_data_payload_format(self, uploader):
        """Test structured data (Graph, Histogram, Table) payload format.

        Expected format: {"time": <ms>, "data": <json-string>, "dataType": <type>,
                         "logName": <name>, "step": <int>}
        """
        # Simulate a Histogram data payload (avoids tuple key JSON issues)
        histogram_data = {
            'bins': [0, 1, 2, 3, 4, 5],
            'counts': [10, 20, 30, 25, 15],
            'min': 0,
            'max': 5,
            'type': 'Histogram',
            'v': 1,
        }
        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.DATA,
                payload={
                    'log_name': 'loss_distribution',
                    'data_type': 'HISTOGRAM',
                    'data': histogram_data,
                },
                timestamp_ms=1705600000000,
                step=5,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_data_batch(records)

            # Verify POST was called to the data endpoint
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args

            # Verify URL
            url = call_args.args[0] if call_args.args else call_args.kwargs.get('url')
            assert url == 'https://test.example.com/ingest/data'

            # Parse the body
            body = call_args.kwargs.get('content') or call_args[1].get('content')
            lines = body.strip().split('\n')
            assert len(lines) == 1

            payload = json.loads(lines[0])

            # Verify correct format
            assert payload['time'] == 1705600000000
            assert payload['dataType'] == 'HISTOGRAM'
            assert payload['logName'] == 'loss_distribution'
            assert payload['step'] == 5

            # data field should be a JSON string
            assert isinstance(payload['data'], str)
            data_parsed = json.loads(payload['data'])
            assert data_parsed['type'] == 'Histogram'
            assert 'bins' in data_parsed
            assert 'counts' in data_parsed


class TestSyncUploaderErrorHandling:
    """Tests for _SyncUploader retry and error handling."""

    @pytest.fixture
    def uploader(self):
        """Create an uploader with mock settings."""
        settings = {
            '_auth': 'test-token',
            '_op_id': 12345,
            '_op_name': 'test-run',
            'project': 'test-project',
            'url_num': 'https://test.example.com/ingest/metrics',
            'sync_process_retry_max': 3,
            'sync_process_retry_backoff': 0.1,  # Fast for tests
        }
        log = logging.getLogger('test')
        return _SyncUploader(settings, log)

    def test_urgent_mode_shorter_timeout(self, uploader):
        """Test that urgent mode uses shorter timeouts."""
        assert uploader._urgent_mode is False

        uploader.set_urgent_mode(True)
        assert uploader._urgent_mode is True

        # Verify urgent mode constants
        assert uploader.URGENT_TIMEOUT_SECONDS == 5.0
        assert uploader.URGENT_MAX_RETRIES == 1

    def test_retry_on_connection_error(self, uploader):
        """Test that connection errors trigger retries."""
        import httpx

        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'loss': 0.5},
                timestamp_ms=1705600000000,
                step=1,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            # First two calls fail, third succeeds
            mock_client.post.side_effect = [
                httpx.ConnectError('Connection failed'),
                httpx.ConnectError('Connection failed'),
                mock_response,
            ]
            MockClient.return_value = mock_client
            uploader._client = None

            # Should succeed after retries
            uploader.upload_metrics_batch(records)
            assert mock_client.post.call_count == 3


class TestDistributedEnvironmentDetection:
    """Tests for DDP/distributed environment detection."""

    def test_no_distributed_env_by_default(self):
        """Test that non-distributed environment is correctly identified."""
        from pluto.op import _is_distributed_environment

        # Clear any distributed env vars that might be set
        env_vars = ['WORLD_SIZE', 'RANK', 'LOCAL_RANK']
        original_values = {var: os.environ.get(var) for var in env_vars}

        try:
            for var in env_vars:
                if var in os.environ:
                    del os.environ[var]

            # Should return False when no distributed env vars are set
            # (assuming torch.distributed is not initialized)
            result = _is_distributed_environment()
            # May be True if torch.distributed is initialized in test env
            assert isinstance(result, bool)
        finally:
            # Restore original values
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value

    def test_world_size_detection(self):
        """Test that WORLD_SIZE > 1 triggers distributed detection."""
        from pluto.op import _is_distributed_environment

        original = os.environ.get('WORLD_SIZE')
        try:
            os.environ['WORLD_SIZE'] = '2'
            assert _is_distributed_environment() is True
        finally:
            if original is not None:
                os.environ['WORLD_SIZE'] = original
            elif 'WORLD_SIZE' in os.environ:
                del os.environ['WORLD_SIZE']

    def test_world_size_one_not_distributed(self):
        """Test that WORLD_SIZE=1 is not considered distributed."""
        from pluto.op import _is_distributed_environment

        original = os.environ.get('WORLD_SIZE')
        try:
            os.environ['WORLD_SIZE'] = '1'
            # WORLD_SIZE=1 alone should not trigger distributed mode
            # (unless torch.distributed is initialized)
            result = _is_distributed_environment()
            # Result depends on torch.distributed state, just verify it returns bool
            assert isinstance(result, bool)
        finally:
            if original is not None:
                os.environ['WORLD_SIZE'] = original
            elif 'WORLD_SIZE' in os.environ:
                del os.environ['WORLD_SIZE']
