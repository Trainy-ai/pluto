"""
Sync Store - SQLite-based persistent storage for the sync process.

This module provides crash-safe storage for metrics, files, and sync state.
The database uses WAL mode and is designed to survive process crashes,
SIGKILL, and other failure modes.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'SyncStore'


class SyncStatus(IntEnum):
    """Status of a record in the sync queue."""

    PENDING = 0  # Not yet synced
    IN_PROGRESS = 1  # Currently being synced
    COMPLETED = 2  # Successfully synced
    FAILED = 3  # Failed after max retries


class RecordType(IntEnum):
    """Type of record in the sync queue."""

    METRIC = 0
    FILE = 1
    CONFIG = 2
    DATA = 3  # Structured data: Graph, Histogram, Table
    TAGS = 4
    SYSTEM = 5
    CONSOLE = 6  # stdout/stderr logs


@dataclass
class FileRecord:
    """A file pending upload to the server."""

    id: int
    run_id: str
    local_path: str
    file_name: str
    file_ext: str
    file_type: str  # MIME type
    file_size: int
    log_name: str  # Key used in pluto.log()
    timestamp_ms: int
    step: Optional[int]
    status: SyncStatus
    retry_count: int
    created_at: float
    last_attempt_at: Optional[float]
    error_message: Optional[str]
    presigned_url: Optional[str]


@dataclass
class SyncRecord:
    """A record pending sync to the server."""

    id: int
    run_id: str
    record_type: RecordType
    payload: Dict[str, Any]
    timestamp_ms: int
    step: Optional[int]
    status: SyncStatus
    retry_count: int
    created_at: float
    last_attempt_at: Optional[float]
    error_message: Optional[str]


class SyncStore:
    """
    SQLite-based persistent storage for sync process.

    Features:
    - WAL mode for crash safety and concurrent access
    - Separate tables for different record types
    - Sync status tracking with retry support
    - Parent process monitoring via heartbeat
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str,
        parent_pid: Optional[int] = None,
    ) -> None:
        self.db_path = db_path
        self.parent_pid = parent_pid
        self._lock = threading.Lock()

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect with WAL mode for crash safety
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode, we manage transactions
        )
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')  # Good balance of safety/speed
        self.conn.execute('PRAGMA busy_timeout=5000')  # Wait up to 5s for locks
        self.conn.row_factory = sqlite3.Row

        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._lock:
            cursor = self.conn.cursor()

            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            # Check schema version
            cursor.execute('SELECT version FROM schema_version LIMIT 1')
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    'INSERT INTO schema_version (version) VALUES (?)',
                    (self.SCHEMA_VERSION,),
                )
            elif row['version'] != self.SCHEMA_VERSION:
                # Handle migrations in future versions
                logger.warning(
                    f'{tag}: Schema version mismatch: {row["version"]} != '
                    f'{self.SCHEMA_VERSION}'
                )

            # Run metadata (which runs are active, their sync state)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    op_id INTEGER,
                    parent_pid INTEGER,
                    created_at REAL NOT NULL,
                    last_heartbeat REAL NOT NULL,
                    finished INTEGER DEFAULT 0,
                    finish_requested_at REAL,
                    fully_synced INTEGER DEFAULT 0,
                    config_json TEXT
                )
            """)

            # Sync queue - all records pending upload
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sync_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    record_type INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    step INTEGER,
                    status INTEGER DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_attempt_at REAL,
                    error_message TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

            # Indexes for efficient querying
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sync_queue_status
                ON sync_queue(status, created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sync_queue_run
                ON sync_queue(run_id, status)
            """)

            # File uploads - separate table for large file tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    remote_url TEXT,
                    file_type TEXT NOT NULL,
                    file_size INTEGER,
                    file_name TEXT,
                    file_ext TEXT,
                    log_name TEXT,
                    timestamp_ms INTEGER NOT NULL,
                    step INTEGER,
                    status INTEGER DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_attempt_at REAL,
                    error_message TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

            self.conn.commit()

    def register_run(
        self,
        run_id: str,
        project: str,
        op_id: Optional[int] = None,
        parent_pid: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new run or update existing."""
        now = time.time()
        config_json = json.dumps(config) if config else None

        with self._lock:
            self.conn.execute(
                """
                INSERT INTO runs (
                    run_id, project, op_id, parent_pid, created_at,
                    last_heartbeat, config_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    last_heartbeat = excluded.last_heartbeat,
                    parent_pid = COALESCE(excluded.parent_pid, parent_pid),
                    op_id = COALESCE(excluded.op_id, op_id),
                    config_json = COALESCE(excluded.config_json, config_json)
                """,
                (run_id, project, op_id, parent_pid, now, now, config_json),
            )

    def heartbeat(self, run_id: str) -> None:
        """Update heartbeat for a run (called by training process)."""
        with self._lock:
            self.conn.execute(
                'UPDATE runs SET last_heartbeat = ? WHERE run_id = ?',
                (time.time(), run_id),
            )

    def mark_run_finished(self, run_id: str) -> None:
        """Mark a run as finished (training complete, flush pending)."""
        with self._lock:
            self.conn.execute(
                """
                UPDATE runs
                SET finished = 1, finish_requested_at = ?
                WHERE run_id = ?
                """,
                (time.time(), run_id),
            )

    def mark_run_synced(self, run_id: str) -> None:
        """Mark a run as fully synced (all data uploaded)."""
        with self._lock:
            self.conn.execute(
                'UPDATE runs SET fully_synced = 1 WHERE run_id = ?',
                (run_id,),
            )

    def enqueue(
        self,
        run_id: str,
        record_type: RecordType,
        payload: Dict[str, Any],
        timestamp_ms: int,
        step: Optional[int] = None,
    ) -> int:
        """Add a record to the sync queue. Returns record ID."""
        now = time.time()
        payload_json = json.dumps(payload)

        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO sync_queue (
                    run_id, record_type, payload_json, timestamp_ms,
                    step, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, int(record_type), payload_json, timestamp_ms, step, now),
            )
            return cursor.lastrowid or 0

    def enqueue_batch(
        self,
        records: List[Tuple[str, RecordType, Dict[str, Any], int, Optional[int]]],
    ) -> List[int]:
        """Add multiple records to the sync queue. Returns record IDs."""
        now = time.time()
        ids = []

        with self._lock:
            self.conn.execute('BEGIN')
            try:
                for run_id, record_type, payload, timestamp_ms, step in records:
                    payload_json = json.dumps(payload)
                    cursor = self.conn.execute(
                        """
                        INSERT INTO sync_queue (
                            run_id, record_type, payload_json, timestamp_ms,
                            step, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            int(record_type),
                            payload_json,
                            timestamp_ms,
                            step,
                            now,
                        ),
                    )
                    ids.append(cursor.lastrowid or 0)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

        return ids

    def get_pending_records(
        self,
        limit: int = 100,
        max_retries: int = 5,
    ) -> List[SyncRecord]:
        """Get records pending sync (PENDING or FAILED with retries left)."""
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT * FROM sync_queue
                WHERE status IN (?, ?)
                AND retry_count < ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (int(SyncStatus.PENDING), int(SyncStatus.FAILED), max_retries, limit),
            )

            records = []
            for row in cursor.fetchall():
                records.append(
                    SyncRecord(
                        id=row['id'],
                        run_id=row['run_id'],
                        record_type=RecordType(row['record_type']),
                        payload=json.loads(row['payload_json']),
                        timestamp_ms=row['timestamp_ms'],
                        step=row['step'],
                        status=SyncStatus(row['status']),
                        retry_count=row['retry_count'],
                        created_at=row['created_at'],
                        last_attempt_at=row['last_attempt_at'],
                        error_message=row['error_message'],
                    )
                )
            return records

    def mark_in_progress(self, record_ids: List[int]) -> None:
        """Mark records as in-progress."""
        if not record_ids:
            return

        with self._lock:
            placeholders = ','.join('?' * len(record_ids))
            params: List[Any] = [int(SyncStatus.IN_PROGRESS), time.time()]
            params.extend(record_ids)
            self.conn.execute(
                f"""
                UPDATE sync_queue
                SET status = ?, last_attempt_at = ?
                WHERE id IN ({placeholders})
                """,
                params,
            )

    def mark_completed(self, record_ids: List[int]) -> None:
        """Mark records as successfully synced."""
        if not record_ids:
            return

        with self._lock:
            placeholders = ','.join('?' * len(record_ids))
            params: List[Any] = [int(SyncStatus.COMPLETED)]
            params.extend(record_ids)
            self.conn.execute(
                f"""
                UPDATE sync_queue
                SET status = ?
                WHERE id IN ({placeholders})
                """,
                params,
            )

    def mark_failed(
        self,
        record_ids: List[int],
        error_message: str,
    ) -> None:
        """Mark records as failed, increment retry count."""
        if not record_ids:
            return

        with self._lock:
            placeholders = ','.join('?' * len(record_ids))
            params: List[Any] = [int(SyncStatus.FAILED), error_message]
            params.extend(record_ids)
            self.conn.execute(
                f"""
                UPDATE sync_queue
                SET status = ?,
                    retry_count = retry_count + 1,
                    error_message = ?
                WHERE id IN ({placeholders})
                """,
                params,
            )

    def get_pending_count(self, run_id: Optional[str] = None) -> int:
        """Get count of pending records."""
        with self._lock:
            if run_id:
                cursor = self.conn.execute(
                    """
                    SELECT COUNT(*) FROM sync_queue
                    WHERE run_id = ? AND status IN (?, ?)
                    """,
                    (run_id, int(SyncStatus.PENDING), int(SyncStatus.IN_PROGRESS)),
                )
            else:
                cursor = self.conn.execute(
                    """
                    SELECT COUNT(*) FROM sync_queue
                    WHERE status IN (?, ?)
                    """,
                    (int(SyncStatus.PENDING), int(SyncStatus.IN_PROGRESS)),
                )
            return cursor.fetchone()[0]

    def get_orphaned_runs(self, timeout_seconds: float = 10.0) -> List[str]:
        """
        Get runs whose parent process has died (no heartbeat).

        A run is considered orphaned if:
        - It's not finished AND
        - Last heartbeat was more than timeout_seconds ago AND
        - Parent PID is no longer running
        """
        cutoff = time.time() - timeout_seconds
        orphaned = []

        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT run_id, parent_pid FROM runs
                WHERE finished = 0 AND last_heartbeat < ?
                """,
                (cutoff,),
            )

            for row in cursor.fetchall():
                parent_pid = row['parent_pid']
                if parent_pid is not None:
                    # Check if parent is still alive
                    try:
                        os.kill(parent_pid, 0)  # Signal 0 = check if exists
                    except OSError:
                        # Process doesn't exist
                        orphaned.append(row['run_id'])
                else:
                    # No parent PID recorded, consider orphaned based on heartbeat
                    orphaned.append(row['run_id'])

        return orphaned

    def get_unsynced_runs(self) -> List[Dict[str, Any]]:
        """Get all runs that have pending data to sync."""
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT r.*, COUNT(sq.id) as pending_count
                FROM runs r
                LEFT JOIN sync_queue sq ON r.run_id = sq.run_id
                    AND sq.status IN (?, ?)
                WHERE r.fully_synced = 0
                GROUP BY r.run_id
                """,
                (int(SyncStatus.PENDING), int(SyncStatus.FAILED)),
            )

            runs = []
            for row in cursor.fetchall():
                runs.append(dict(row))
            return runs

    def cleanup_completed(self, older_than_seconds: float = 3600.0) -> int:
        """Remove completed records older than threshold. Returns count deleted."""
        cutoff = time.time() - older_than_seconds

        with self._lock:
            cursor = self.conn.execute(
                """
                DELETE FROM sync_queue
                WHERE status = ? AND created_at < ?
                """,
                (int(SyncStatus.COMPLETED), cutoff),
            )
            return cursor.rowcount

    # ========================================================================
    # File upload methods
    # ========================================================================

    def enqueue_file(
        self,
        run_id: str,
        local_path: str,
        file_name: str,
        file_ext: str,
        file_type: str,
        file_size: int,
        log_name: str,
        timestamp_ms: int,
        step: Optional[int] = None,
    ) -> int:
        """Add a file to the upload queue. Returns file record ID."""
        now = time.time()

        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO file_uploads (
                    run_id, local_path, file_type, file_size,
                    timestamp_ms, step, created_at, file_name, file_ext, log_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    local_path,
                    file_type,
                    file_size,
                    timestamp_ms,
                    step,
                    now,
                    file_name,
                    file_ext,
                    log_name,
                ),
            )
            return cursor.lastrowid or 0

    def get_pending_files(
        self,
        limit: int = 10,
        max_retries: int = 5,
    ) -> List[FileRecord]:
        """Get files pending upload."""
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT * FROM file_uploads
                WHERE status IN (?, ?)
                AND retry_count < ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (int(SyncStatus.PENDING), int(SyncStatus.FAILED), max_retries, limit),
            )

            records = []
            for row in cursor.fetchall():
                records.append(
                    FileRecord(
                        id=row['id'],
                        run_id=row['run_id'],
                        local_path=row['local_path'],
                        file_name=row['file_name'] or '',
                        file_ext=row['file_ext'] or '',
                        file_type=row['file_type'],
                        file_size=row['file_size'] or 0,
                        log_name=row['log_name'] or '',
                        timestamp_ms=row['timestamp_ms'],
                        step=row['step'],
                        status=SyncStatus(row['status']),
                        retry_count=row['retry_count'],
                        created_at=row['created_at'],
                        last_attempt_at=row['last_attempt_at'],
                        error_message=row['error_message'],
                        presigned_url=row['remote_url'],
                    )
                )
            return records

    def mark_files_in_progress(self, file_ids: List[int]) -> None:
        """Mark file records as in-progress."""
        if not file_ids:
            return

        with self._lock:
            placeholders = ','.join('?' * len(file_ids))
            params: List[Any] = [int(SyncStatus.IN_PROGRESS), time.time()]
            params.extend(file_ids)
            self.conn.execute(
                f"""
                UPDATE file_uploads
                SET status = ?, last_attempt_at = ?
                WHERE id IN ({placeholders})
                """,
                params,
            )

    def mark_files_completed(self, file_ids: List[int]) -> None:
        """Mark file records as successfully uploaded."""
        if not file_ids:
            return

        with self._lock:
            placeholders = ','.join('?' * len(file_ids))
            params: List[Any] = [int(SyncStatus.COMPLETED)]
            params.extend(file_ids)
            self.conn.execute(
                f"""
                UPDATE file_uploads
                SET status = ?
                WHERE id IN ({placeholders})
                """,
                params,
            )

    def mark_files_failed(
        self,
        file_ids: List[int],
        error_message: str,
    ) -> None:
        """Mark file records as failed, increment retry count."""
        if not file_ids:
            return

        with self._lock:
            placeholders = ','.join('?' * len(file_ids))
            params: List[Any] = [int(SyncStatus.FAILED), error_message]
            params.extend(file_ids)
            self.conn.execute(
                f"""
                UPDATE file_uploads
                SET status = ?,
                    retry_count = retry_count + 1,
                    error_message = ?
                WHERE id IN ({placeholders})
                """,
                params,
            )

    def update_file_presigned_url(self, file_id: int, url: str) -> None:
        """Store presigned URL for a file."""
        with self._lock:
            self.conn.execute(
                'UPDATE file_uploads SET remote_url = ? WHERE id = ?',
                (url, file_id),
            )

    def get_pending_file_count(self, run_id: Optional[str] = None) -> int:
        """Get count of pending file uploads."""
        with self._lock:
            if run_id:
                cursor = self.conn.execute(
                    """
                    SELECT COUNT(*) FROM file_uploads
                    WHERE run_id = ? AND status IN (?, ?)
                    """,
                    (run_id, int(SyncStatus.PENDING), int(SyncStatus.IN_PROGRESS)),
                )
            else:
                cursor = self.conn.execute(
                    """
                    SELECT COUNT(*) FROM file_uploads
                    WHERE status IN (?, ?)
                    """,
                    (int(SyncStatus.PENDING), int(SyncStatus.IN_PROGRESS)),
                )
            return cursor.fetchone()[0]

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            self.conn.close()

    def __enter__(self) -> 'SyncStore':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
