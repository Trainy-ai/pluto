"""
Pluto Sync Process Module.

This module provides a separate process architecture for uploading data
to the Pluto backend. The sync process:

1. Runs as a separate spawned process (not forked, for CUDA safety)
2. Reads from a local SQLite database (crash-safe, WAL mode)
3. Uploads data to the server with retries and backoff
4. Monitors parent process and handles orphan detection
5. Supports DDP coordination (multiple ranks share one sync process)

Usage:
    from pluto.sync import SyncProcessManager

    # In training process
    manager = SyncProcessManager(run_id, project, settings_dict)
    manager.start()

    # Log metrics (goes to disk, picked up by sync process)
    manager.enqueue_metrics({'loss': 0.5}, timestamp_ms, step)

    # On finish
    manager.stop()  # Waits for sync to complete
    manager.close()
"""

from .process import (
    SyncProcessManager,
    get_existing_sync_process,
    is_sync_process_alive,
    start_sync_process,
    stop_sync_process,
)
from .store import FileRecord, RecordType, SyncRecord, SyncStatus, SyncStore

__all__ = [
    # Manager class (primary interface)
    'SyncProcessManager',
    # Store classes
    'SyncStore',
    'SyncRecord',
    'FileRecord',
    'SyncStatus',
    'RecordType',
    # Low-level process functions
    'start_sync_process',
    'stop_sync_process',
    'get_existing_sync_process',
    'is_sync_process_alive',
]
