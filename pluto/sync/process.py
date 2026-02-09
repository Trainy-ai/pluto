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
import os
import signal
import sqlite3
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from filelock import FileLock
except ImportError:
    # Fallback for environments without filelock
    FileLock = None  # type: ignore[misc, assignment]

from .store import FileRecord, RecordType, SyncRecord, SyncStore

# Type alias for subprocess
ProcessType = subprocess.Popen

logger = logging.getLogger(__name__)


class SyncState(str, Enum):
    """Process coordination states."""

    RUNNING = 'running'
    FINISHING = 'finishing'
    FINISHED = 'finished'
    FAILED = 'failed'


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

        # Use file lock for DDP coordination to prevent race conditions
        # where multiple ranks might try to start sync processes simultaneously
        lock_file = Path(self.db_path).parent / '.sync.lock'

        if FileLock is not None:
            lock = FileLock(str(lock_file), timeout=10)
            with lock:
                self._start_or_attach()
        else:
            # Fallback without locking (less safe for DDP)
            logger.debug('filelock not available, proceeding without DDP lock')
            self._start_or_attach()

    def _start_or_attach(self) -> None:
        """Start a new sync process or attach to an existing one."""
        if self._started:
            return

        # Check for existing sync process (DDP coordination)
        existing_pid = self._get_existing_sync_pid()
        if existing_pid and _is_process_alive(existing_pid):
            logger.info(f'Using existing sync process (PID: {existing_pid})')
            self._started = True
            return

        # Start new sync process using subprocess.Popen
        # This avoids the __main__ re-import issue of multiprocessing.Process
        self._process = subprocess.Popen(
            [
                sys.executable,
                '-m',
                'pluto.sync',
                '--db-path',
                str(self.db_path),
                '--settings',
                json.dumps(self.settings),
                '--parent-pid',
                str(os.getpid()),
            ],
            stdin=subprocess.DEVNULL,  # Don't inherit stdin
            # stdout/stderr inherit from parent (visible in terminal)
        )

        # Record PID for DDP coordination
        self._record_sync_pid(self._process.pid)
        self._started = True

        logger.info(f'Started sync process (PID: {self._process.pid})')

    def stop(
        self,
        timeout: Optional[float] = None,
        wait: bool = True,
    ) -> bool:
        """
        Stop the sync process gracefully.

        Args:
            timeout: Max time to wait for sync. None uses settings default.
            wait: If True (default), block until sync completes or timeout.
                  If False, signal shutdown but don't wait (for preemption).

        Returns:
            True if sync completed (or wait=False), False if timed out.
        """
        if not self._started:
            return True

        # Signal finish to sync process
        self.store.mark_run_finished(self.run_id)

        if not wait:
            # Preemption/DDP mode: signal shutdown and terminate process
            # This prevents blocking in torchrun which waits for child processes
            pending = self.store.get_pending_count(self.run_id)
            logger.info(
                f'Signaled sync shutdown (not waiting). '
                f'{pending} records pending, data preserved in {self.db_path}'
            )
            # Send SIGTERM to sync process so it exits gracefully
            # This is critical for DDP where torchrun waits for all children
            if self._process is not None and self._process.poll() is None:
                try:
                    self._process.terminate()
                    logger.debug('Sent SIGTERM to sync process')
                except Exception as e:
                    logger.debug(f'Failed to terminate sync process: {e}')
            return True

        # Normal mode: wait for sync to complete
        timeout = timeout or self.settings.get('sync_process_shutdown_timeout', 30.0)

        start = time.time()
        while time.time() - start < timeout:
            pending = self.store.get_pending_count(self.run_id)
            if pending == 0:
                self.store.mark_run_synced(self.run_id)
                logger.info('Sync process completed successfully')
                return True
            time.sleep(0.1)

        pending = self.store.get_pending_count(self.run_id)
        logger.warning(
            f'Sync process did not complete within {timeout}s, '
            f'{pending} records pending. Data preserved in {self.db_path}'
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

    def enqueue_data(
        self,
        log_name: str,
        data_type: str,
        data_dict: Dict[str, Any],
        timestamp_ms: int,
        step: Optional[int] = None,
    ) -> None:
        """
        Enqueue structured data (Graph, Histogram, Table) for upload.

        Args:
            log_name: The key used in pluto.log()
            data_type: Type name (GRAPH, HISTOGRAM, TABLE)
            data_dict: Serialized data from Data.to_dict()
            timestamp_ms: Timestamp in milliseconds
            step: Optional step number
        """
        self.store.enqueue(
            run_id=self.run_id,
            record_type=RecordType.DATA,
            payload={
                'log_name': log_name,
                'data_type': data_type,
                'data': data_dict,
            },
            timestamp_ms=timestamp_ms,
            step=step,
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

    def enqueue_console_log(
        self,
        message: str,
        log_type: str,
        timestamp_ms: int,
        line_number: int,
    ) -> None:
        """Enqueue console log message for upload."""
        self.store.enqueue(
            run_id=self.run_id,
            record_type=RecordType.CONSOLE,
            payload={
                'message': message,
                'logType': log_type,
                'lineNumber': line_number,
            },
            timestamp_ms=timestamp_ms,
            step=line_number,
        )

    def enqueue_console_batch(
        self,
        lines: List[tuple],
    ) -> None:
        """Enqueue a batch of console log lines in a single transaction.

        Args:
            lines: List of (message, log_type, timestamp_ms, line_number) tuples.
        """
        if not lines:
            return
        records = [
            (
                self.run_id,
                RecordType.CONSOLE,
                {
                    'message': message,
                    'logType': log_type,
                    'lineNumber': line_number,
                },
                timestamp_ms,
                line_number,
            )
            for message, log_type, timestamp_ms, line_number in lines
        ]
        self.store.enqueue_batch(records)

    def enqueue_file(
        self,
        local_path: str,
        file_name: str,
        file_ext: str,
        file_type: str,
        file_size: int,
        log_name: str,
        timestamp_ms: int,
        step: Optional[int] = None,
    ) -> None:
        """
        Enqueue a file for upload.

        The file should already exist on disk at local_path.
        The sync process will handle getting presigned URLs and uploading.
        """
        self.store.enqueue_file(
            run_id=self.run_id,
            local_path=local_path,
            file_name=file_name,
            file_ext=file_ext,
            file_type=file_type,
            file_size=file_size,
            log_name=log_name,
            timestamp_ms=timestamp_ms,
            step=step,
        )
        # Update heartbeat to show we're alive
        self.store.heartbeat(self.run_id)

    def get_pending_count(self) -> int:
        """Get count of pending records (metrics + files)."""
        return self.store.get_pending_count(
            self.run_id
        ) + self.store.get_pending_file_count(self.run_id)

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

    # Suppress verbose httpx request logging (HTTP Request: POST ... "HTTP/2 200 OK")
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

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
    batch_size = settings_dict.get('sync_process_batch_size', 100)
    file_batch_size = settings_dict.get('sync_process_file_batch_size', 10)

    parent_check_interval = 5.0
    cleanup_interval = 300.0  # Clean up completed records every 5 minutes
    health_interval = 60.0  # Log health diagnostics every 60 seconds
    last_parent_check = time.time()
    last_flush = time.time()
    last_cleanup = time.time()
    last_health = time.time()

    try:
        while not shutdown_requested['value']:
            try:
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
                    synced = _sync_batch(
                        store, uploader, log, max_retries, batch_size, file_batch_size
                    )
                    if synced:
                        log.debug(f'Synced batch of {synced} records')
                    last_flush = time.time()

                # Periodic cleanup of completed records to prevent
                # unbounded table growth over long training runs
                if time.time() - last_cleanup > cleanup_interval:
                    deleted = store.cleanup_completed(
                        older_than_seconds=cleanup_interval
                    )
                    if deleted:
                        log.debug(f'Cleaned up {deleted} completed records')
                    last_cleanup = time.time()

                # Periodic health diagnostics — makes degradation
                # visible in logs before it escalates to failures
                if time.time() - last_health > health_interval:
                    try:
                        h = store.get_health_stats()
                        log.debug(
                            'Health: pending=%d in_progress=%d '
                            'completed=%d failed=%d '
                            'total_rows=%d lag_s=%.1f '
                            'wal_kb=%d db_kb=%d '
                            'writes=%d avg_ms=%.1f max_ms=%.1f '
                            'retries=%d retry_failures=%d',
                            h['pending'],
                            h['in_progress'],
                            h['completed'],
                            h['failed'],
                            h['total_rows'],
                            h['lag_s'],
                            h['wal_size_kb'],
                            h['db_size_kb'],
                            h['write_count'],
                            h['write_avg_ms'],
                            h['write_max_ms'],
                            h['retries'],
                            h['retry_failures'],
                        )
                        uploader.upload_health_stats(h)
                    except Exception:
                        pass  # Health check should never crash sync
                    last_health = time.time()

            except sqlite3.OperationalError as e:
                # Transient SQLite errors (e.g. "database is locked") should not
                # crash the sync process. Log and retry on the next iteration.
                log.warning(f'Transient SQLite error (will retry): {e}')
                time.sleep(1.0)  # Back off briefly before next attempt
                continue

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
    file_batch_size: int = 10,
) -> int:
    """
    Sync a batch of pending records (metrics + files).

    Returns count of records synced.
    """
    total_synced = 0

    # Sync regular records (metrics, config, tags, system)
    total_synced += _sync_records_batch(store, uploader, log, max_retries, batch_size)

    # Sync files
    total_synced += _sync_files_batch(
        store, uploader, log, max_retries, file_batch_size
    )

    return total_synced


def _sync_records_batch(
    store: SyncStore,
    uploader: '_SyncUploader',
    log: logging.Logger,
    max_retries: int,
    batch_size: int = 100,
) -> int:
    """
    Sync a batch of pending metric/config/tags/system records.

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
    data_records: List[SyncRecord] = []
    console_records: List[SyncRecord] = []

    for record in records:
        if record.record_type == RecordType.METRIC:
            metrics_records.append(record)
        elif record.record_type == RecordType.CONFIG:
            config_records.append(record)
        elif record.record_type == RecordType.TAGS:
            tags_records.append(record)
        elif record.record_type == RecordType.SYSTEM:
            system_records.append(record)
        elif record.record_type == RecordType.DATA:
            data_records.append(record)
        elif record.record_type == RecordType.CONSOLE:
            console_records.append(record)

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

    # Upload structured data (Graph, Histogram, Table)
    if data_records:
        try:
            uploader.upload_data_batch(data_records)
            success_ids.extend(r.id for r in data_records)
        except Exception as e:
            log.warning(f'Failed to upload structured data: {e}')
            failed_ids.extend(r.id for r in data_records)
            error_msg = str(e)

    # Upload console logs
    if console_records:
        try:
            uploader.upload_console_batch(console_records)
            success_ids.extend(r.id for r in console_records)
        except Exception as e:
            log.warning(f'Failed to upload console logs: {e}')
            failed_ids.extend(r.id for r in console_records)
            error_msg = str(e)

    # Update status
    store.mark_completed(success_ids)
    if failed_ids:
        store.mark_failed(failed_ids, error_msg)

    return len(success_ids)


def _sync_files_batch(
    store: SyncStore,
    uploader: '_SyncUploader',
    log: logging.Logger,
    max_retries: int,
    batch_size: int,
) -> int:
    """
    Sync a batch of pending file uploads.

    Returns count of files synced.
    """
    file_records = store.get_pending_files(limit=batch_size, max_retries=max_retries)
    if not file_records:
        return 0

    # Mark as in progress
    file_ids = [f.id for f in file_records]
    store.mark_files_in_progress(file_ids)

    # Upload files
    results = uploader.upload_files_batch(file_records)

    # Update status based on results
    success_ids = [fid for fid, success in results.items() if success]
    failed_ids = [fid for fid, success in results.items() if not success]

    store.mark_files_completed(success_ids)
    if failed_ids:
        store.mark_files_failed(failed_ids, 'Upload failed')

    if success_ids:
        log.debug(f'Uploaded {len(success_ids)} file(s)')
    if failed_ids:
        log.warning(f'Failed to upload {len(failed_ids)} file(s)')

    return len(success_ids)


def _flush_remaining(
    store: SyncStore,
    uploader: '_SyncUploader',
    log: logging.Logger,
    timeout: float,
    max_retries: int,
) -> None:
    """
    Flush all remaining records and files within timeout.

    Uses urgent mode to ensure we don't block on slow/failing uploads.
    Data that fails to sync remains in SQLite for later recovery.
    """
    # Enable urgent mode - short timeouts, minimal retries
    uploader.set_urgent_mode(True)

    start = time.time()
    batches_synced = 0
    total_records_synced = 0

    # Minimum time needed for a batch attempt (timeout + overhead)
    min_batch_time = 2.0

    def get_total_pending() -> int:
        return store.get_pending_count() + store.get_pending_file_count()

    while True:
        elapsed = time.time() - start
        remaining = timeout - elapsed

        # Hard timeout check - stop if not enough time for another batch
        if remaining < min_batch_time:
            log.debug(f'Flush: insufficient time for batch ({remaining:.1f}s)')
            break

        try:
            # Use minimal retries during flush - data is safe in SQLite
            synced = _sync_batch(store, uploader, log, max_retries=1)
            if synced == 0:
                # Nothing left to sync
                pending = get_total_pending()
                if pending == 0:
                    log.info(
                        f'All records synced successfully '
                        f'({total_records_synced} records in {batches_synced} batches)'
                    )
                    return
                else:
                    log.warning(
                        f'{pending} records/files failed to sync after retries. '
                        f'Data preserved in SQLite for recovery.'
                    )
                    return
            total_records_synced += synced
            batches_synced += 1
        except Exception as e:
            # Don't let a single batch failure stop the flush
            log.warning(f'Flush batch error (continuing): {e}')
            # Brief pause before next attempt
            time.sleep(0.5)

    # Timeout reached
    pending = get_total_pending()
    log.warning(
        f'Flush timed out after {timeout}s. '
        f'Synced {total_records_synced} records, {pending} remain in SQLite.'
    )


# ============================================================================
# Uploader (HTTP Client for Sync Process)
# ============================================================================


class _SyncUploader:
    """
    HTTP client for uploading data to Pluto backend.

    Runs in the sync process and handles retries with backoff.
    Supports "urgent mode" for shutdown scenarios with shorter timeouts.
    """

    # Timeouts for urgent mode (shutdown/preemption)
    URGENT_TIMEOUT_SECONDS = 5.0
    URGENT_MAX_RETRIES = 1
    URGENT_MAX_BACKOFF_SECONDS = 1.0

    # File upload settings
    FILE_UPLOAD_TIMEOUT_SECONDS = 120.0  # Larger timeout for file uploads

    def __init__(self, settings_dict: Dict[str, Any], log: logging.Logger):
        self.settings = settings_dict
        self.log = log
        self._client: Any = None
        self._storage_client: Any = None  # Separate client for S3 uploads
        self._urgent_mode = False

        # Extract settings
        self.auth_token = settings_dict.get('_auth', '')
        self.op_id = settings_dict.get('_op_id')
        self.op_name = settings_dict.get('_op_name', '')
        self.project = settings_dict.get('project', '')
        self.tag = settings_dict.get('tag', 'pluto')

        # URLs
        self.url_num = settings_dict.get('url_num', '')
        self.url_data = settings_dict.get('url_data', '')
        self.url_update_config = settings_dict.get('url_update_config', '')
        self.url_update_tags = settings_dict.get('url_update_tags', '')
        self.url_file = settings_dict.get('url_file', '')
        self.url_console = settings_dict.get('url_message', '')

        # Retry settings (normal mode)
        self.retry_max = settings_dict.get('sync_process_retry_max', 5)
        self.retry_backoff = settings_dict.get('sync_process_retry_backoff', 2.0)
        self.normal_timeout = 30.0

    def set_urgent_mode(self, urgent: bool) -> None:
        """
        Enable/disable urgent mode for shutdown scenarios.

        In urgent mode:
        - Shorter HTTP timeouts (5s vs 30s)
        - Fewer retries (1 vs 5)
        - Shorter backoff (max 1s)

        This prevents blocking during pod preemption/shutdown.
        """
        self._urgent_mode = urgent
        if urgent:
            self.log.debug('Uploader entering urgent mode (short timeouts)')

    @property
    def client(self) -> Any:
        """Lazy-init HTTP client."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                timeout=httpx.Timeout(self.normal_timeout),
                limits=httpx.Limits(max_connections=10),
            )
        return self._client

    @property
    def storage_client(self) -> Any:
        """Lazy-init HTTP client for S3 storage uploads."""
        if self._storage_client is None:
            import httpx

            self._storage_client = httpx.Client(
                timeout=httpx.Timeout(self.FILE_UPLOAD_TIMEOUT_SECONDS),
                limits=httpx.Limits(max_connections=5),
            )
        return self._storage_client

    def _get_headers(self) -> Dict[str, str]:
        """Get common headers for requests."""
        return {
            'Authorization': f'Bearer {self.auth_token}',
            'Content-Type': 'application/x-ndjson',
            'User-Agent': self.tag,
            'X-Run-Id': str(self.op_id or ''),
            'X-Run-Name': self.op_name,
            'X-Project-Name': self.project,
        }

    def upload_metrics_batch(self, records: List[SyncRecord]) -> None:
        """Upload a batch of metric records."""
        if not self.url_num or not records:
            return

        # Convert to NDJSON format expected by server:
        # {"time": <ms>, "step": <int>, "data": {...metrics...}}
        lines = []
        for record in records:
            # Filter to numeric values only (exclude bools, they're int subclass)
            numeric_data = {
                k: v
                for k, v in record.payload.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if numeric_data:
                lines.append(
                    json.dumps(
                        {
                            'time': record.timestamp_ms,
                            'step': record.step or 0,
                            'data': numeric_data,
                        }
                    )
                )

        if not lines:
            return

        body = ('\n'.join(lines) + '\n').encode('utf-8')
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
        # Keys already have 'sys/' prefix from the monitor
        if not self.url_num or not records:
            return

        # Convert to NDJSON format expected by server:
        # {"time": <ms>, "step": <int>, "data": {"sys/key": value, ...}}
        lines = []
        for record in records:
            # Filter to numeric values only (exclude bools)
            # Keys already have sys/ prefix from the monitor
            sys_data = {
                k: v
                for k, v in record.payload.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if sys_data:
                lines.append(
                    json.dumps(
                        {
                            'time': record.timestamp_ms,
                            'step': 0,  # System metrics don't have steps
                            'data': sys_data,
                        }
                    )
                )

        if not lines:
            return

        body = '\n'.join(lines) + '\n'
        self._post_with_retry(self.url_num, body, self._get_headers())

    def upload_health_stats(self, stats: Dict[str, Any]) -> None:
        """Upload sync health diagnostics as sys/ metrics.

        Posts directly to the numeric endpoint — bypasses SQLite entirely
        to avoid circular write pressure from the health check itself.
        """
        if not self.url_num:
            return

        data = {f'sys/sync.{k}': v for k, v in stats.items()}
        timestamp_ms = int(time.time() * 1000)
        body = json.dumps({'time': timestamp_ms, 'step': 0, 'data': data})
        body = body + '\n'
        self._post_with_retry(self.url_num, body, self._get_headers())

    def upload_data_batch(self, records: List[SyncRecord]) -> None:
        """
        Upload structured data batch (Graph, Histogram, Table).

        Each record payload contains:
        - log_name: The key used in pluto.log()
        - data_type: Type name (GRAPH, HISTOGRAM, TABLE)
        - data: The serialized data dict from Data.to_dict()
        """
        if not self.url_data or not records:
            return

        # Convert to NDJSON format expected by server:
        # {"time": ms, "data": json-str, "dataType": type, "logName": name, "step": int}
        lines = []
        for record in records:
            payload = record.payload
            lines.append(
                json.dumps(
                    {
                        'time': record.timestamp_ms,
                        'data': json.dumps(payload.get('data', {})),
                        'dataType': payload.get('data_type', 'UNKNOWN'),
                        'logName': payload.get('log_name', ''),
                        'step': record.step or 0,
                    }
                )
            )

        if not lines:
            return

        body = '\n'.join(lines) + '\n'
        self._post_with_retry(self.url_data, body, self._get_headers())

    def upload_console_batch(self, records: List[SyncRecord]) -> None:
        """
        Upload console log messages batch.

        Each record payload contains:
        - message: The log line content
        - logType: INFO, WARNING, ERROR, etc.
        - lineNumber: Line counter
        """
        if not self.url_console or not records:
            return

        # Convert to NDJSON format expected by server
        lines = []
        for record in records:
            payload = record.payload
            lines.append(
                json.dumps(
                    {
                        'time': record.timestamp_ms,
                        'message': payload.get('message', ''),
                        'lineNumber': payload.get('lineNumber', 0),
                        'logType': payload.get('logType', 'INFO'),
                    }
                )
            )

        if not lines:
            return

        body = ('\n'.join(lines) + '\n').encode('utf-8')
        self._post_with_retry(self.url_console, body, self._get_headers())

    def upload_files_batch(
        self,
        file_records: List[FileRecord],
    ) -> Dict[int, bool]:
        """
        Upload a batch of files.

        Returns dict mapping file_id -> success (True/False).

        Flow:
        1. Request presigned URLs for all files
        2. Upload each file to its presigned URL
        3. Return success/failure status for each
        """
        if not self.url_file or not file_records:
            return {}

        results: Dict[int, bool] = {}

        # Step 1: Request presigned URLs for all files
        try:
            presigned_urls = self._get_presigned_urls(file_records)
        except Exception as e:
            self.log.warning(f'Failed to get presigned URLs: {e}')
            # Mark all as failed
            for f in file_records:
                results[f.id] = False
            return results

        # Step 2: Upload each file to S3
        for file_record in file_records:
            file_key = f'{file_record.file_name}{file_record.file_ext}'
            url = presigned_urls.get(file_key)

            if not url:
                self.log.warning(
                    f'No presigned URL for file: {file_key} '
                    f'(available: {list(presigned_urls.keys())})'
                )
                results[file_record.id] = False
                continue

            try:
                self._upload_file_to_storage(file_record, url)
                results[file_record.id] = True
            except Exception as e:
                self.log.warning(f'Failed to upload file {file_key}: {e}')
                results[file_record.id] = False

        return results

    def _get_presigned_urls(
        self,
        file_records: List[FileRecord],
    ) -> Dict[str, str]:
        """
        Request presigned URLs for a batch of files.

        Returns dict mapping filename -> presigned_url.
        """
        # Build request payload (same format as make_compat_file_v1)
        batch = []
        for f in file_records:
            file_ext = f.file_ext
            file_type = file_ext[1:] if file_ext.startswith('.') else file_ext
            batch.append(
                {
                    'fileName': f'{f.file_name}{f.file_ext}',
                    'fileSize': f.file_size,
                    'fileType': file_type,
                    'time': f.timestamp_ms,
                    'logName': f.log_name,
                    'step': f.step,
                }
            )

        body = json.dumps({'files': batch})
        headers = self._get_headers()
        headers['Content-Type'] = 'application/json'

        # Use shorter timeout in urgent mode
        timeout = (
            self.URGENT_TIMEOUT_SECONDS if self._urgent_mode else self.normal_timeout
        )

        response = self.client.post(
            self.url_file,
            content=body,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()

        # Parse response - it's a dict mapping file type to list of url mappings
        # e.g., {"Image": [{"filename.png": "https://s3..."}]}
        result_data = response.json()
        presigned_urls: Dict[str, str] = {}

        # Flatten the nested structure
        for file_type_list in result_data.values():
            if isinstance(file_type_list, list):
                for url_mapping in file_type_list:
                    if isinstance(url_mapping, dict):
                        presigned_urls.update(url_mapping)

        return presigned_urls

    def _upload_file_to_storage(
        self,
        file_record: FileRecord,
        presigned_url: str,
    ) -> None:
        """Upload a single file to S3 using presigned URL with retry."""
        # Check file exists before attempting upload
        if not os.path.exists(file_record.local_path):
            raise FileNotFoundError(f'File not found: {file_record.local_path}')

        # Read file content
        with open(file_record.local_path, 'rb') as f:
            data = f.read()

        self.log.debug(
            f'Uploading file {file_record.file_name}{file_record.file_ext} '
            f'({len(data)} bytes) to S3'
        )

        # Use shorter timeout in urgent mode
        timeout = (
            self.URGENT_TIMEOUT_SECONDS
            if self._urgent_mode
            else self.FILE_UPLOAD_TIMEOUT_SECONDS
        )

        # Retry logic with exponential backoff
        max_retries = 1 if self._urgent_mode else 3
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                response = self.storage_client.put(
                    presigned_url,
                    content=data,
                    headers={'Content-Type': file_record.file_type},
                    timeout=timeout,
                )

                if response.status_code in [200, 201]:
                    return  # Success

                last_error = Exception(
                    f'S3 upload failed with status {response.status_code}: '
                    f'{response.text[:100]}'
                )
            except Exception as e:
                last_error = e

            # Retry with exponential backoff if not the last attempt
            if attempt < max_retries:
                wait = min(2**attempt, 10)  # Max 10s backoff
                self.log.debug(
                    f'S3 upload attempt {attempt + 1} failed, '
                    f'retrying in {wait}s: {last_error}'
                )
                time.sleep(wait)

        # All retries exhausted
        raise last_error or Exception('S3 upload failed')

    def _post_with_retry(
        self,
        url: str,
        body: Union[str, bytes],
        headers: Dict[str, str],
    ) -> None:
        """POST with exponential backoff retry. Respects urgent mode settings."""
        last_error = None

        # Use urgent mode settings if enabled
        if self._urgent_mode:
            max_retries = self.URGENT_MAX_RETRIES
            timeout = self.URGENT_TIMEOUT_SECONDS
            max_backoff = self.URGENT_MAX_BACKOFF_SECONDS
        else:
            max_retries = self.retry_max
            timeout = self.normal_timeout
            max_backoff = 60.0  # No cap in normal mode

        for attempt in range(max_retries):
            try:
                response = self.client.post(
                    url,
                    content=body,
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = min(self.retry_backoff**attempt, max_backoff)
                    self.log.debug(
                        f'Request failed (attempt {attempt + 1}/{max_retries}), '
                        f'retrying in {wait}s: {e}'
                    )
                    time.sleep(wait)

        raise last_error or Exception('Request failed after retries')

    def close(self) -> None:
        """Close HTTP clients."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._storage_client is not None:
            self._storage_client.close()
            self._storage_client = None


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

    Uses subprocess.Popen to avoid __main__ re-import issues.
    """
    process = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'pluto.sync',
            '--db-path',
            str(db_path),
            '--settings',
            json.dumps(settings_dict),
            '--parent-pid',
            str(parent_pid),
        ],
        stdin=subprocess.DEVNULL,
    )

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
