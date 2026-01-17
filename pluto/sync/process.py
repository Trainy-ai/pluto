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
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Sync state keys in database
STATE_RUN_STATUS = "run_status"
STATE_SYNC_PID = "sync_process_pid"
STATE_LAST_UPLOAD_ID = "last_upload_id"

# Run status values
STATUS_RUNNING = "running"
STATUS_FINISHING = "finishing"
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"


def start_sync_process(
    db_path: str,
    settings_dict: Dict[str, Any],
    parent_pid: int,
) -> multiprocessing.Process:
    """
    Start the background sync process.

    Uses spawn (not fork) to avoid issues with:
    - CUDA contexts (don't survive fork)
    - Threading (inconsistent state after fork)
    - File descriptors (copied in bad state)

    Args:
        db_path: Path to SQLite database
        settings_dict: Serializable settings for sync process
        parent_pid: PID of parent process to monitor

    Returns:
        The started Process object
    """
    # Use spawn context - CRITICAL for CUDA compatibility
    ctx = multiprocessing.get_context("spawn")

    process = ctx.Process(
        target=_sync_main,
        args=(db_path, settings_dict, parent_pid),
        name="pluto-sync",
        daemon=False,  # NOT daemon - we want it to outlive parent briefly
    )
    process.start()

    # Record sync process PID in database for coordination
    _record_sync_pid(db_path, process.pid)

    logger.info(f"Started sync process (PID: {process.pid})")
    return process


def stop_sync_process(
    db_path: str,
    timeout: float = 30.0,
) -> bool:
    """
    Request sync process to stop gracefully.

    This sets the run status to "finishing" which signals the sync
    process to flush remaining data and exit.

    Args:
        db_path: Path to SQLite database
        timeout: Max time to wait for sync process to exit

    Returns:
        True if sync process exited, False if timeout
    """
    # Set finishing status
    _set_sync_state(db_path, STATE_RUN_STATUS, STATUS_FINISHING)

    # Get sync process PID
    pid = get_existing_sync_process(db_path)
    if pid is None:
        return True  # No process to stop

    # Wait for process to exit
    start = time.time()
    while time.time() - start < timeout:
        if not is_sync_process_alive(pid):
            return True
        time.sleep(0.1)

    logger.warning(f"Sync process (PID: {pid}) did not exit within {timeout}s")
    return False


def get_existing_sync_process(db_path: str) -> Optional[int]:
    """
    Get PID of existing sync process for this run.

    Returns:
        PID if sync process exists and is alive, None otherwise
    """
    pid_str = _get_sync_state(db_path, STATE_SYNC_PID)
    if pid_str is None:
        return None

    try:
        pid = int(pid_str)
        if is_sync_process_alive(pid):
            return pid
    except (ValueError, TypeError):
        pass

    return None


def is_sync_process_alive(pid: int) -> bool:
    """Check if a process with given PID is alive."""
    try:
        os.kill(pid, 0)  # Signal 0 = check existence
        return True
    except OSError:
        return False


# ============================================================================
# Sync Process Main (runs in separate process)
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
    # Set up logging for sync process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [pluto-sync] %(levelname)s: %(message)s",
    )
    log = logging.getLogger("pluto-sync")

    log.info(f"Sync process started (parent PID: {parent_pid})")

    # Set up signal handlers
    shutdown_requested = {"value": False}

    def handle_signal(signum, frame):
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        log.info(f"Received {sig_name}, initiating graceful shutdown")
        shutdown_requested["value"] = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Connect to database
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
    except Exception as e:
        log.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    # Ensure sync state table exists
    _ensure_sync_tables(conn)

    # Set run status to running
    _set_sync_state_conn(conn, STATE_RUN_STATUS, STATUS_RUNNING)

    # Create HTTP client for uploads
    uploader = _SyncUploader(settings_dict, log)

    # Main sync loop
    poll_interval = settings_dict.get("x_sync_process_poll_interval", 0.5)
    parent_check_interval = 5.0
    last_parent_check = time.time()

    try:
        while not shutdown_requested["value"]:
            # Periodically check if parent is alive
            if time.time() - last_parent_check > parent_check_interval:
                if not _is_parent_alive(parent_pid):
                    log.warning("Parent process died, flushing and exiting")
                    _flush_and_exit(conn, uploader, log, STATUS_FINISHED)
                    return
                last_parent_check = time.time()

            # Check for finish signal from training process
            status = _get_sync_state_conn(conn, STATE_RUN_STATUS)
            if status == STATUS_FINISHING:
                log.info("Received finish signal, flushing remaining data")
                _flush_and_exit(conn, uploader, log, STATUS_FINISHED)
                return

            # Sync a batch of data
            synced = _sync_batch(conn, uploader, settings_dict, log)

            # Sleep if nothing to sync
            if not synced:
                time.sleep(poll_interval)

    except Exception as e:
        log.error(f"Sync process error: {e}", exc_info=True)
        _set_sync_state_conn(conn, STATE_RUN_STATUS, STATUS_FAILED)
    finally:
        conn.close()
        log.info("Sync process exiting")


def _is_parent_alive(parent_pid: int) -> bool:
    """Check if parent process is still alive."""
    try:
        os.kill(parent_pid, 0)
        return True
    except OSError:
        return False


def _flush_and_exit(
    conn: sqlite3.Connection,
    uploader: "_SyncUploader",
    log: logging.Logger,
    final_status: str,
) -> None:
    """Flush all remaining data and mark run complete."""
    flush_timeout = 30.0
    start = time.time()

    while time.time() - start < flush_timeout:
        synced = _sync_batch(conn, uploader, {}, log)
        if not synced:
            break  # Nothing left to sync

    # Mark run as finished on server
    try:
        uploader.finish_run()
    except Exception as e:
        log.error(f"Failed to mark run as finished: {e}")

    _set_sync_state_conn(conn, STATE_RUN_STATUS, final_status)


def _sync_batch(
    conn: sqlite3.Connection,
    uploader: "_SyncUploader",
    settings_dict: Dict[str, Any],
    log: logging.Logger,
) -> bool:
    """
    Sync a batch of data from database to server.

    Returns:
        True if any data was synced, False otherwise
    """
    batch_size = settings_dict.get("x_sync_process_batch_size", 1000)

    # Get last uploaded ID
    last_id_str = _get_sync_state_conn(conn, STATE_LAST_UPLOAD_ID)
    last_id = int(last_id_str) if last_id_str else 0

    # Read batch of unuploaded metrics
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, time, step, key, value
        FROM num
        WHERE id > ?
        ORDER BY id
        LIMIT ?
        """,
        (last_id, batch_size),
    )
    rows = cursor.fetchall()

    if not rows:
        return False

    # Convert to metrics format
    metrics_by_step: Dict[int, Dict[str, float]] = {}
    max_id = last_id

    for row in rows:
        row_id, timestamp, step, key, value = row
        if step not in metrics_by_step:
            metrics_by_step[step] = {}
        metrics_by_step[step][key] = value
        max_id = max(max_id, row_id)

    # Upload batch
    try:
        for step, metrics in sorted(metrics_by_step.items()):
            uploader.upload_metrics(metrics, step)

        # Update last uploaded ID
        _set_sync_state_conn(conn, STATE_LAST_UPLOAD_ID, str(max_id))
        log.debug(f"Synced {len(rows)} metrics (up to ID {max_id})")
        return True

    except Exception as e:
        log.error(f"Failed to upload batch: {e}")
        return False


# ============================================================================
# Database Helpers
# ============================================================================


def _ensure_sync_tables(conn: sqlite3.Connection) -> None:
    """Ensure sync state table exists."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at INTEGER
        )
        """
    )
    conn.commit()


def _get_sync_state(db_path: str, key: str) -> Optional[str]:
    """Get sync state value from database."""
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        result = _get_sync_state_conn(conn, key)
        conn.close()
        return result
    except Exception:
        return None


def _set_sync_state(db_path: str, key: str, value: str) -> None:
    """Set sync state value in database."""
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        _set_sync_state_conn(conn, key, value)
        conn.close()
    except Exception:
        pass


def _get_sync_state_conn(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Get sync state value using existing connection."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM sync_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _set_sync_state_conn(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Set sync state value using existing connection."""
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO sync_state (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, value, int(time.time() * 1000)),
        )
        conn.commit()
    except Exception:
        pass


def _record_sync_pid(db_path: str, pid: int) -> None:
    """Record sync process PID in database."""
    _set_sync_state(db_path, STATE_SYNC_PID, str(pid))


# ============================================================================
# Uploader (HTTP Client for Sync Process)
# ============================================================================


class _SyncUploader:
    """
    HTTP client for uploading data to Pluto backend.

    This is a simplified version of ServerInterface, designed to run
    in the sync process.
    """

    def __init__(self, settings_dict: Dict[str, Any], log: logging.Logger):
        self.settings = settings_dict
        self.log = log

        # Import httpx here (in sync process)
        import httpx

        self.client = httpx.Client(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=10),
        )

        self.headers = {
            "Authorization": f"Bearer {settings_dict.get('_auth', '')}",
            "Content-Type": "application/x-ndjson",
            "X-Run-Id": str(settings_dict.get("_op_id", "")),
            "X-Run-Name": settings_dict.get("_op_name", ""),
            "X-Project-Name": settings_dict.get("project", ""),
        }

        self.url_num = settings_dict.get("url_num", "")

    def upload_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Upload metrics to Pluto backend."""
        if not self.url_num:
            return

        # Format as NDJSON
        timestamp = int(time.time() * 1000)
        lines = []
        for key, value in metrics.items():
            lines.append(json.dumps({"k": key, "v": value, "s": step, "t": timestamp}))

        body = "\n".join(lines)

        response = self.client.post(
            self.url_num,
            content=body,
            headers=self.headers,
        )
        response.raise_for_status()

    def finish_run(self) -> None:
        """Mark run as finished on server."""
        # This would call the status update endpoint
        # Simplified for now
        pass

    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()
