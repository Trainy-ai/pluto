"""
Sync process implementation for Pluto.

This module implements a separate process that handles all network I/O
to the Pluto backend. The training process writes to a local SQLite
database, and this sync process reads from it and uploads to the server.

Key design principles:
1. Use spawn (not fork) to avoid CUDA/threading issues
2. Monitor parent PID - exit gracefully if orphaned
3. Durable: all data goes to disk first
4. Never block the training process
"""

import json
import logging
import multiprocessing
import multiprocessing.process
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .store import RecordType, SyncRecord, SyncStore

# Type alias for process - SpawnProcess and Process have same interface
ProcessType = Union[multiprocessing.Process, multiprocessing.process.BaseProcess]

logger = logging.getLogger(__name__)

# Process coordination states
STATE_RUNNING = 'running'
STATE_FINISHING = 'finishing'
STATE_FINISHED = 'finished'
STATE_FAILED = 'failed'


class SyncProcessManager:
    """
    Manages the sync process lifecycle from the training process side.

    This class provides the interface for:
    - Starting/stopping the sync process
    - Enqueuing data for upload
    - Checking sync status
    - Waiting for sync completion
    """

    def __init__(
        self,
        run_id: str,
        project: str,
        settings_dict: Dict[str, Any],
        db_path: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self.project = project
        self.settings = settings_dict

        # Determine database path
        if db_path:
            self.db_path = db_path
        else:
            base_dir = settings_dict.get('dir', os.getcwd())
            tag = settings_dict.get('tag', 'pluto')
            run_dir = Path(base_dir) / f'.{tag}' / project / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(run_dir / 'sync.db')

        # Initialize store
        self.store = SyncStore(self.db_path, parent_pid=os.getpid())
        self.store.register_run(
            run_id=run_id,
            project=project,
            op_id=settings_dict.get('_op_id'),
            parent_pid=os.getpid(),
            config=settings_dict.get('_config'),
        )

        # Process handle
        self._process: Optional[ProcessType] = None
        self._started = False

    def start(self) -> None:
        """Start the sync process."""
        if self._started:
            return

        # Check for existing sync process (DDP coordination)
        existing_pid = self._get_existing_sync_pid()
        if existing_pid and _is_process_alive(existing_pid):
            logger.info(f'Using existing sync process (PID: {existing_pid})')
            self._started = True
            return

        # Start new sync process using spawn context
        ctx = multiprocessing.get_context('spawn')

        self._process = ctx.Process(
            target=_sync_main,
            args=(self.db_path, self.settings, os.getpid()),
            name='pluto-sync',
            daemon=False,  # Not daemon - allow it to outlive parent briefly
        )
        self._process.start()

        # Record PID for DDP coordination
        self._record_sync_pid(self._process.pid)
        self._started = True

        logger.info(f'Started sync process (PID: {self._process.pid})')

    def stop(self, timeout: Optional[float] = None) -> bool:
        """
        Stop the sync process gracefully.

        Args:
            timeout: Max time to wait for sync. None uses settings default.

        Returns:
            True if sync completed, False if timed out.
        """
        if not self._started:
            return True

        timeout = timeout or self.settings.get('sync_process_shutdown_timeout', 30.0)

        # Signal finish
        self.store.mark_run_finished(self.run_id)

        # Wait for sync to complete
        start = time.time()
        while time.time() - start < timeout:
            pending = self.store.get_pending_count(self.run_id)
            if pending == 0:
                self.store.mark_run_synced(self.run_id)
                logger.info('Sync process completed successfully')
                return True
            time.sleep(0.1)

        logger.warning(
            f'Sync process did not complete within {timeout}s, '
            f'{self.store.get_pending_count(self.run_id)} records pending'
        )
        return False

    def enqueue_metrics(
        self,
        metrics: Dict[str, Any],
        timestamp_ms: int,
        step: int,
    ) -> None:
        """Enqueue metrics for upload."""
        self.store.enqueue(
            run_id=self.run_id,
            record_type=RecordType.METRIC,
            payload=metrics,
            timestamp_ms=timestamp_ms,
            step=step,
        )
        # Update heartbeat to show we're alive
        self.store.heartbeat(self.run_id)

    def enqueue_config(self, config: Dict[str, Any], timestamp_ms: int) -> None:
        """Enqueue config update for upload."""
        self.store.enqueue(
            run_id=self.run_id,
            record_type=RecordType.CONFIG,
            payload=config,
            timestamp_ms=timestamp_ms,
        )

    def enqueue_tags(self, tags: List[str], timestamp_ms: int) -> None:
        """Enqueue tags update for upload."""
        self.store.enqueue(
            run_id=self.run_id,
            record_type=RecordType.TAGS,
            payload={'tags': tags},
            timestamp_ms=timestamp_ms,
        )

    def enqueue_system_metrics(
        self,
        metrics: Dict[str, Any],
        timestamp_ms: int,
    ) -> None:
        """Enqueue system metrics for upload."""
        self.store.enqueue(
            run_id=self.run_id,
            record_type=RecordType.SYSTEM,
            payload=metrics,
            timestamp_ms=timestamp_ms,
        )

    def get_pending_count(self) -> int:
        """Get count of pending records."""
        return self.store.get_pending_count(self.run_id)

    def heartbeat(self) -> None:
        """Send heartbeat to indicate training process is alive."""
        self.store.heartbeat(self.run_id)

    def close(self) -> None:
        """Close the store connection."""
        self.store.close()

    def _get_existing_sync_pid(self) -> Optional[int]:
        """Get PID of existing sync process from lock file."""
        lock_file = Path(self.db_path).parent / '.sync.pid'
        if lock_file.exists():
            try:
                pid = int(lock_file.read_text().strip())
                return pid
            except (ValueError, OSError):
                pass
        return None

    def _record_sync_pid(self, pid: Optional[int]) -> None:
        """Record sync process PID in lock file."""
        if pid is None:
            return
        lock_file = Path(self.db_path).parent / '.sync.pid'
        try:
            lock_file.write_text(str(pid))
        except OSError:
            pass


def _is_process_alive(pid: int) -> bool:
    """Check if process is alive."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ============================================================================
# Sync Process Main (runs in separate spawned process)
# ============================================================================


def _sync_main(
    db_path: str,
    settings_dict: Dict[str, Any],
    parent_pid: int,
) -> None:
    """
    Main entry point for sync process.

    This runs in a separate process (spawned, not forked).
    """
    # Set up logging
    log_level = settings_dict.get('x_log_level', logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [pluto-sync] %(levelname)s: %(message)s',
    )
    log = logging.getLogger('pluto-sync')

    log.info(f'Sync process started (parent PID: {parent_pid})')

    # Set up signal handlers
    shutdown_requested = {'value': False}

    def handle_signal(signum: int, frame: Any) -> None:
        sig_name = 'SIGTERM' if signum == signal.SIGTERM else 'SIGINT'
        log.info(f'Received {sig_name}, initiating graceful shutdown')
        shutdown_requested['value'] = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Connect to store
    try:
        store = SyncStore(db_path, parent_pid=parent_pid)
    except Exception as e:
        log.error(f'Failed to open sync store: {e}')
        sys.exit(1)

    # Create uploader
    uploader = _SyncUploader(settings_dict, log)

    # Timing settings
    flush_interval = settings_dict.get('sync_process_flush_interval', 1.0)
    orphan_timeout = settings_dict.get('sync_process_orphan_timeout', 10.0)
    max_retries = settings_dict.get('sync_process_retry_max', 5)
    shutdown_timeout = settings_dict.get('sync_process_shutdown_timeout', 30.0)

    parent_check_interval = 5.0
    last_parent_check = time.time()
    last_flush = time.time()

    try:
        while not shutdown_requested['value']:
            # Check if parent is alive
            if time.time() - last_parent_check > parent_check_interval:
                if not _is_process_alive(parent_pid):
                    log.warning('Parent process died, flushing and exiting')
                    _flush_remaining(
                        store, uploader, log, shutdown_timeout, max_retries
                    )
                    return
                last_parent_check = time.time()

            # Check for orphaned runs
            orphaned = store.get_orphaned_runs(orphan_timeout)
            for run_id in orphaned:
                log.info(f'Detected orphaned run: {run_id}')
                store.mark_run_finished(run_id)

            # Check for finished runs that need final flush
            unsynced = store.get_unsynced_runs()
            for run_info in unsynced:
                if run_info['finished'] and run_info['pending_count'] == 0:
                    store.mark_run_synced(run_info['run_id'])
                    log.info(f'Run {run_info["run_id"]} fully synced')

            # Periodic flush
            if time.time() - last_flush > flush_interval:
                synced = _sync_batch(store, uploader, log, max_retries)
                if synced:
                    log.debug(f'Synced batch of {synced} records')
                last_flush = time.time()

            # Short sleep to avoid busy loop
            time.sleep(0.1)

    except Exception as e:
        log.error(f'Sync process error: {e}', exc_info=True)
    finally:
        uploader.close()
        store.close()
        log.info('Sync process exiting')


def _sync_batch(
    store: SyncStore,
    uploader: '_SyncUploader',
    log: logging.Logger,
    max_retries: int,
    batch_size: int = 100,
) -> int:
    """
    Sync a batch of pending records.

    Returns count of records synced.
    """
    records = store.get_pending_records(limit=batch_size, max_retries=max_retries)
    if not records:
        return 0

    # Mark as in progress
    record_ids = [r.id for r in records]
    store.mark_in_progress(record_ids)

    # Group by type for efficient upload
    metrics_records: List[SyncRecord] = []
    config_records: List[SyncRecord] = []
    tags_records: List[SyncRecord] = []
    system_records: List[SyncRecord] = []

    for record in records:
        if record.record_type == RecordType.METRIC:
            metrics_records.append(record)
        elif record.record_type == RecordType.CONFIG:
            config_records.append(record)
        elif record.record_type == RecordType.TAGS:
            tags_records.append(record)
        elif record.record_type == RecordType.SYSTEM:
            system_records.append(record)

    success_ids: List[int] = []
    failed_ids: List[int] = []
    error_msg = ''

    # Upload metrics
    if metrics_records:
        try:
            uploader.upload_metrics_batch(metrics_records)
            success_ids.extend(r.id for r in metrics_records)
        except Exception as e:
            log.warning(f'Failed to upload metrics: {e}')
            failed_ids.extend(r.id for r in metrics_records)
            error_msg = str(e)

    # Upload config updates
    for record in config_records:
        try:
            uploader.upload_config(record)
            success_ids.append(record.id)
        except Exception as e:
            log.warning(f'Failed to upload config: {e}')
            failed_ids.append(record.id)
            error_msg = str(e)

    # Upload tags updates
    for record in tags_records:
        try:
            uploader.upload_tags(record)
            success_ids.append(record.id)
        except Exception as e:
            log.warning(f'Failed to upload tags: {e}')
            failed_ids.append(record.id)
            error_msg = str(e)

    # Upload system metrics
    if system_records:
        try:
            uploader.upload_system_batch(system_records)
            success_ids.extend(r.id for r in system_records)
        except Exception as e:
            log.warning(f'Failed to upload system metrics: {e}')
            failed_ids.extend(r.id for r in system_records)
            error_msg = str(e)

    # Update status
    store.mark_completed(success_ids)
    if failed_ids:
        store.mark_failed(failed_ids, error_msg)

    return len(success_ids)


def _flush_remaining(
    store: SyncStore,
    uploader: '_SyncUploader',
    log: logging.Logger,
    timeout: float,
    max_retries: int,
) -> None:
    """Flush all remaining records within timeout."""
    start = time.time()

    while time.time() - start < timeout:
        synced = _sync_batch(store, uploader, log, max_retries)
        if synced == 0:
            # Nothing left to sync
            pending = store.get_pending_count()
            if pending == 0:
                log.info('All records synced successfully')
                return
            else:
                log.warning(f'{pending} records failed to sync after retries')
                return

    log.warning(f'Flush timed out after {timeout}s')


# ============================================================================
# Uploader (HTTP Client for Sync Process)
# ============================================================================


class _SyncUploader:
    """
    HTTP client for uploading data to Pluto backend.

    Runs in the sync process and handles retries with backoff.
    """

    def __init__(self, settings_dict: Dict[str, Any], log: logging.Logger):
        self.settings = settings_dict
        self.log = log
        self._client: Any = None

        # Extract settings
        self.auth_token = settings_dict.get('_auth', '')
        self.op_id = settings_dict.get('_op_id')
        self.op_name = settings_dict.get('_op_name', '')
        self.project = settings_dict.get('project', '')

        # URLs
        self.url_num = settings_dict.get('url_num', '')
        self.url_update_config = settings_dict.get('url_update_config', '')
        self.url_update_tags = settings_dict.get('url_update_tags', '')

        # Retry settings
        self.retry_max = settings_dict.get('sync_process_retry_max', 5)
        self.retry_backoff = settings_dict.get('sync_process_retry_backoff', 2.0)

    @property
    def client(self) -> Any:
        """Lazy-init HTTP client."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=10),
                http2=True,
            )
        return self._client

    def _get_headers(self) -> Dict[str, str]:
        """Get common headers for requests."""
        return {
            'Authorization': f'Bearer {self.auth_token}',
            'Content-Type': 'application/x-ndjson',
            'X-Run-Id': str(self.op_id or ''),
            'X-Run-Name': self.op_name,
            'X-Project-Name': self.project,
        }

    def upload_metrics_batch(self, records: List[SyncRecord]) -> None:
        """Upload a batch of metric records."""
        if not self.url_num or not records:
            return

        # Convert to NDJSON format
        lines = []
        for record in records:
            payload = record.payload
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    lines.append(
                        json.dumps(
                            {
                                'k': key,
                                'v': value,
                                's': record.step,
                                't': record.timestamp_ms,
                            }
                        )
                    )

        if not lines:
            return

        body = '\n'.join(lines)
        self._post_with_retry(self.url_num, body, self._get_headers())

    def upload_config(self, record: SyncRecord) -> None:
        """Upload config update."""
        if not self.url_update_config or not self.op_id:
            return

        payload = {
            'runId': self.op_id,
            'config': record.payload,
        }

        headers = self._get_headers()
        headers['Content-Type'] = 'application/json'
        self._post_with_retry(
            self.url_update_config,
            json.dumps(payload),
            headers,
        )

    def upload_tags(self, record: SyncRecord) -> None:
        """Upload tags update."""
        if not self.url_update_tags or not self.op_id:
            return

        payload = {
            'runId': self.op_id,
            'tags': record.payload.get('tags', []),
        }

        headers = self._get_headers()
        headers['Content-Type'] = 'application/json'
        self._post_with_retry(
            self.url_update_tags,
            json.dumps(payload),
            headers,
        )

    def upload_system_batch(self, records: List[SyncRecord]) -> None:
        """Upload system metrics batch."""
        # System metrics use same endpoint as regular metrics
        # but with 'sys/' prefix on keys
        if not self.url_num or not records:
            return

        lines = []
        for record in records:
            payload = record.payload
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    lines.append(
                        json.dumps(
                            {
                                'k': f'sys/{key}',
                                'v': value,
                                's': 0,  # System metrics don't have steps
                                't': record.timestamp_ms,
                            }
                        )
                    )

        if not lines:
            return

        body = '\n'.join(lines)
        self._post_with_retry(self.url_num, body, self._get_headers())

    def _post_with_retry(
        self,
        url: str,
        body: str,
        headers: Dict[str, str],
    ) -> None:
        """POST with exponential backoff retry."""
        last_error = None

        for attempt in range(self.retry_max):
            try:
                response = self.client.post(url, content=body, headers=headers)
                response.raise_for_status()
                return
            except Exception as e:
                last_error = e
                if attempt < self.retry_max - 1:
                    wait = self.retry_backoff**attempt
                    self.log.debug(
                        f'Request failed (attempt {attempt + 1}), '
                        f'retrying in {wait}s: {e}'
                    )
                    time.sleep(wait)

        raise last_error or Exception('Request failed after retries')

    def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None


# ============================================================================
# Public API for process management
# ============================================================================


def start_sync_process(
    db_path: str,
    settings_dict: Dict[str, Any],
    parent_pid: int,
) -> ProcessType:
    """
    Start the background sync process.

    Uses spawn (not fork) to avoid issues with CUDA and threading.
    """
    ctx = multiprocessing.get_context('spawn')

    process = ctx.Process(
        target=_sync_main,
        args=(db_path, settings_dict, parent_pid),
        name='pluto-sync',
        daemon=False,
    )
    process.start()

    logger.info(f'Started sync process (PID: {process.pid})')
    return process


def get_existing_sync_process(db_path: str) -> Optional[int]:
    """Get PID of existing sync process from lock file."""
    lock_file = Path(db_path).parent / '.sync.pid'
    if lock_file.exists():
        try:
            pid = int(lock_file.read_text().strip())
            if _is_process_alive(pid):
                return pid
        except (ValueError, OSError):
            pass
    return None


def is_sync_process_alive(pid: int) -> bool:
    """Check if sync process is alive."""
    return _is_process_alive(pid)


def stop_sync_process(db_path: str, timeout: float = 30.0) -> bool:
    """
    Request sync process to stop gracefully.

    Returns True if process exited, False if timeout.
    """
    pid = get_existing_sync_process(db_path)
    if pid is None:
        return True

    # Signal graceful shutdown via SIGTERM
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return True  # Already dead

    # Wait for exit
    start = time.time()
    while time.time() - start < timeout:
        if not _is_process_alive(pid):
            return True
        time.sleep(0.1)

    logger.warning(f'Sync process (PID: {pid}) did not exit within {timeout}s')
    return False
