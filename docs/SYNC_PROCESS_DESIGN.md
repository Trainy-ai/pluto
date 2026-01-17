# Pluto Sync Process Architecture Design

## Overview

This document outlines the design for adding a separate sync process to Pluto,
similar to the architectures used by wandb (wandb-core) and Neptune Scale.

## Why a Separate Process?

### Current Architecture Problems

1. **Training crash = data loss**: If training process OOMs or crashes, buffered
   metrics in threads are lost
2. **GIL contention**: Network I/O threads compete with training
3. **Complex signal handling**: SIGINT/SIGTERM handling in multi-threaded
   Python + DDP is fragile
4. **Slow exits**: Must wait for thread joins/timeouts before process can exit
5. **DDP complexity**: Each rank has its own threads, coordinating shutdown is hard

### Benefits of Separate Process

1. **Crash resilience**: Training dies, sync process continues uploading
2. **Clean signal handling**: Sync process has its own handlers, isolated
3. **Fast training exits**: Training writes to disk, exits immediately
4. **No GIL contention**: Sync process is separate Python interpreter
5. **Simpler DDP**: All ranks write to shared disk location, one sync process

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Training Process (Rank 0, 1, 2, 3...)                    │
│                                                                              │
│  pluto.init() ──► Spawns sync process (once per node)                       │
│                                                                              │
│  pluto.log({"loss": 0.5})                                                   │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ DataStore (SQLite WAL)                                              │    │
│  │ - Append-only writes                                                │    │
│  │ - No network I/O                                                    │    │
│  │ - WAL mode allows concurrent reads                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  pluto.finish() ──► Writes "finish" marker, exits immediately              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ SQLite WAL (concurrent read)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Pluto Sync Process (1 per node)                          │
│                                                                              │
│  Lifecycle:                                                                  │
│  1. Spawned by first pluto.init() on node (using spawn, not fork)          │
│  2. Monitors parent PID - if orphaned, continues briefly then exits        │
│  3. Tails SQLite WAL, batches and uploads to Pluto backend                 │
│  4. On "finish" marker: flush remaining, mark run complete, exit           │
│  5. Has own SIGTERM handler for graceful shutdown                          │
│                                                                              │
│  Components:                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │ DB Reader Thread │  │ Uploader Thread  │  │ File Upload Pool │          │
│  │ - Polls SQLite   │  │ - Batches HTTP   │  │ - Parallel S3    │          │
│  │ - Marks uploaded │  │ - Retries        │  │ - Presigned URLs │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## IPC Mechanism

### Option 1: SQLite as the IPC (Recommended)

- Training process writes to SQLite with WAL mode
- Sync process reads from same SQLite file
- Add `uploaded` column to track what's been sent
- Simple, durable, already exists in Pluto

### Option 2: Unix Domain Socket

- Lower latency than disk
- More complex to implement
- Loses durability benefit

### Option 3: Shared Memory + Signal

- Highest performance
- Most complex
- Overkill for logging use case

**Recommendation**: Use SQLite (Option 1). It's already implemented, provides
durability, and the latency is acceptable for metrics logging.

## Database Schema Changes

```sql
-- Add to existing tables
ALTER TABLE num ADD COLUMN uploaded INTEGER DEFAULT 0;
ALTER TABLE num ADD COLUMN upload_attempts INTEGER DEFAULT 0;

-- Add sync state table
CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at INTEGER
);

-- Track run lifecycle
-- key='run_status', value='running'|'finishing'|'finished'|'failed'
-- key='last_upload_id', value='<id>'
-- key='sync_process_pid', value='<pid>'
```

## Implementation Plan

### Phase 1: Core Sync Process

**File: `pluto/sync/process.py`**

```python
"""
Sync process that runs separately from training.
Handles all network I/O to Pluto backend.
"""

import multiprocessing
import os
import signal
import sqlite3
import time
from typing import Optional

def start_sync_process(
    db_path: str,
    settings_dict: dict,
    parent_pid: int,
) -> multiprocessing.Process:
    """
    Start the background sync process.

    Uses spawn (not fork) to avoid CUDA/threading issues.
    """
    ctx = multiprocessing.get_context("spawn")

    process = ctx.Process(
        target=_sync_main,
        args=(db_path, settings_dict, parent_pid),
        name="pluto-sync",
        daemon=False,  # Want it to outlive parent briefly
    )
    process.start()
    return process


def _sync_main(db_path: str, settings_dict: dict, parent_pid: int):
    """Main entry point for sync process."""

    # Set up signal handlers
    shutdown = {"requested": False}

    def handle_signal(signum, frame):
        shutdown["requested"] = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Connect to SQLite (read-only)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")

    # Create HTTP client
    # ... (similar to current ServerInterface)

    # Main sync loop
    while not shutdown["requested"]:
        # Check parent alive
        if not _is_parent_alive(parent_pid):
            # Parent died - flush and exit
            _flush_remaining(conn, settings_dict)
            break

        # Check for finish marker
        status = _get_run_status(conn)
        if status == "finishing":
            _flush_remaining(conn, settings_dict)
            _mark_run_finished(conn, settings_dict)
            break

        # Sync batch
        _sync_batch(conn, settings_dict)

        time.sleep(0.5)

    conn.close()


def _is_parent_alive(parent_pid: int) -> bool:
    """Check if parent process is still alive."""
    try:
        os.kill(parent_pid, 0)
        return True
    except OSError:
        return False
```

### Phase 2: Training Process Changes

**Modify `pluto/op.py`**

```python
class Op:
    def __init__(self, ...):
        # ...existing code...

        if self.settings.x_use_sync_process:
            self._sync_process = self._start_or_attach_sync_process()
            self._iface = None  # Don't use direct interface
        else:
            self._sync_process = None
            self._iface = ServerInterface(...)  # Current behavior

    def _start_or_attach_sync_process(self):
        """Start sync process or attach to existing one."""
        from .sync.process import start_sync_process, get_existing_sync_process

        # Check if sync process already running for this run
        existing = get_existing_sync_process(self.settings.get_dir())
        if existing:
            return existing

        # Start new sync process
        return start_sync_process(
            db_path=self._store.db,
            settings_dict=self.settings.to_dict(),
            parent_pid=os.getpid(),
        )

    def finish(self, code=None):
        # ...existing code...

        if self.settings.x_use_sync_process:
            # Write finish marker to DB - sync process will handle the rest
            self._store.set_run_status("finishing")
            # Don't wait for sync process - it handles itself
        else:
            # Current behavior - wait for threads
            # ...
```

### Phase 3: Settings

**Add to `pluto/sets.py`**

```python
class Settings:
    # ...existing...

    # Sync process mode (experimental)
    x_use_sync_process: bool = False
    x_sync_process_flush_timeout: float = 30.0  # Max time to flush on finish
    x_sync_process_poll_interval: float = 0.5   # How often to check for new data
    x_sync_process_batch_size: int = 1000       # Max items per upload batch
```

### Phase 4: DDP Support

For DDP, the key insight is that **only one sync process per node** is needed:

```python
# In pluto.init() for DDP
def init(..., run_id=None):
    # ...

    if settings.x_use_sync_process:
        # Use run_id to coordinate - all ranks share same run_id
        # First rank to start creates sync process
        # Other ranks just attach
        sync_lock_file = f"{settings.get_dir()}/.sync_lock"

        with FileLock(sync_lock_file):
            existing = get_existing_sync_process(...)
            if not existing:
                start_sync_process(...)
```

## Migration Strategy

### Phase 1 (Current)
- Thread-based sync (existing)
- Neptune compat layer transparency (done!)

### Phase 2 (This Design)
- Add sync process behind `x_use_sync_process=True` flag
- Test extensively in DDP scenarios
- Keep thread mode as default

### Phase 3 (Future)
- Make sync process the default
- Deprecate thread mode
- Consider Rust/Go rewrite for better isolation (like wandb-core)

## Testing Plan

1. **Unit tests**: Sync process lifecycle, DB operations
2. **Integration tests**: Full training → sync → upload flow
3. **DDP tests**: Multi-rank coordination, signal handling
4. **Chaos tests**: Kill training process, verify data recovery
5. **Performance tests**: Compare latency/throughput vs thread mode

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Orphaned sync processes | Monitor parent PID, exit on orphan |
| DB corruption | SQLite WAL is crash-safe |
| Sync process crash | Training continues, data on disk |
| Deadlock on exit | Never wait for sync process from training |
| Resource exhaustion | Limit batch sizes, connection pools |

## Open Questions

1. **Should sync process be per-node or per-run?**
   - Per-run is simpler, per-node more efficient for many runs
   - Recommend: per-run for simplicity initially

2. **How to handle sync process crashes?**
   - Option A: Training restarts it automatically
   - Option B: Data stays on disk, `pluto sync` CLI command
   - Recommend: Option B for simplicity

3. **Should we use a lockfile or PID file for coordination?**
   - Recommend: Both - lockfile for startup, PID file for monitoring
