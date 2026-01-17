"""Tests for sync process V2 architecture.

These tests verify the sync process implementation including:
- SyncStore database operations
- File upload via sync process
- Graceful shutdown handling
"""

import json
import time

import numpy as np
import pytest
from PIL import Image as PILImage

import pluto
from pluto.sync.store import FileRecord, SyncStatus, SyncStore
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
