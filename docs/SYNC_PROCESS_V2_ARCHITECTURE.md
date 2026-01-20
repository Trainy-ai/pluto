# Pluto v2 Architecture: Sync Process First

## Design Philosophy

**Core Principle**: The training process should NEVER do network I/O.

```
Training Process                    Sync Process
================                    ============
pluto.log() → SQLite (local)   ←──  Tail & Upload
      ↓                                   ↓
   Fast, local                      Network I/O
   Never blocks                     Retries
   Crash-safe                       Crash-safe
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Node (GPU Server)                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Pluto Run Directory                                  │ │
│  │                    /tmp/pluto-runs/<run_id>/                           │ │
│  │                                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │ │
│  │  │  sync.db    │  │  files/     │  │  sync.pid   │  │  .sync.lock  │  │ │
│  │  │ (SQLite WAL)│  │  (uploads)  │  │  (PID file) │  │  (DDP coord) │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│         ▲                    ▲                                               │
│         │ Write              │ Write                                         │
│         │                    │                                               │
│  ┌──────┴────────────────────┴──────────────────────────────────────────┐   │
│  │                    Training Processes                                 │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │   │
│  │  │ Rank 0  │  │ Rank 1  │  │ Rank 2  │  │ Rank 3  │                 │   │
│  │  │ pluto   │  │ pluto   │  │ pluto   │  │ pluto   │                 │   │
│  │  │ .log()  │  │ .log()  │  │ .log()  │  │ .log()  │                 │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│         │ Read (tail)                                                        │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Pluto Sync Process (1 per node)                    │   │
│  │                                                                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │   │
│  │  │ DB Reader    │  │ HTTP Uploader│  │ File Uploader│                │   │
│  │  │ (poll SQLite)│  │ (metrics)    │  │ (S3 presign) │                │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │   │
│  │                                                                        │   │
│  │  Features:                                                             │   │
│  │  - Monitors parent PID (training)                                     │   │
│  │  - Orphan detection → flush & exit                                    │   │
│  │  - Own signal handlers (graceful shutdown)                            │   │
│  │  - Exponential backoff on failures                                    │   │
│  │  - Marks uploaded records in DB                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTPS
                                    ▼
                          ┌─────────────────────┐
                          │   Pluto Backend     │
                          │   (pluto.trainy.ai) │
                          └─────────────────────┘
```

## Database Schema

The actual implementation uses a unified queue approach in `pluto/sync/store.py`:

```sql
-- Schema version tracking
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY
);

-- Run metadata (which runs are active, their sync state)
CREATE TABLE runs (
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
);

-- Unified sync queue for all record types (metrics, config, tags, system, data, console)
CREATE TABLE sync_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    record_type INTEGER NOT NULL,  -- 0:METRIC, 1:FILE, 2:CONFIG, 3:DATA, 4:TAGS, 5:SYSTEM, 6:CONSOLE
    payload_json TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    step INTEGER,
    status INTEGER DEFAULT 0,      -- 0:PENDING, 1:IN_PROGRESS, 2:COMPLETED, 3:FAILED
    retry_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    last_attempt_at REAL,
    error_message TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX idx_sync_queue_status ON sync_queue(status, created_at);
CREATE INDEX idx_sync_queue_run ON sync_queue(run_id, status);

-- File uploads (separate table for large file tracking)
CREATE TABLE file_uploads (
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
    status INTEGER DEFAULT 0,      -- 0:PENDING, 1:IN_PROGRESS, 2:COMPLETED, 3:FAILED
    retry_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    last_attempt_at REAL,
    error_message TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

## Failure Handling

### 1. Training Process Crash (OOM, Segfault)

```
Timeline:
[T+0]     Training crashes
[T+0]     All data already on disk (SQLite WAL)
[T+5s]    Sync process detects parent died (os.kill(ppid, 0) fails)
[T+5s]    Sync process enters "orphan mode":
          - Flushes all remaining data
          - Marks run as 'crashed' on server
          - Exits cleanly
[T+30s]   All data uploaded, run shows as "Crashed" in UI
```

**Key**: No data loss because training never held data in memory queues.

### 2. Training Process SIGKILL (Spot Termination)

```
Timeline:
[T+0]     Cloud provider sends SIGKILL (no grace period)
[T+0]     Training dies instantly
[T+0]     SQLite WAL is crash-consistent (no corruption)
[T+5s]    Sync process detects orphan
[T+30s]   Data uploaded, run marked as terminated

Alternative (with SIGTERM warning):
[T+0]     Cloud provider sends SIGTERM (2min warning)
[T+0]     Training sets status='finishing' in DB
[T+1s]    Sync process sees 'finishing', enters flush mode
[T+30s]   Most data uploaded
[T+2min]  SIGKILL arrives, sync process dies
[Later]   `pluto sync` CLI recovers remaining data
```

### 3. Training Process Graceful Exit (SIGTERM, Ctrl+C)

```
Timeline:
[T+0]     User hits Ctrl+C
[T+0]     Training sets status='finishing' in DB
[T+0]     Training calls pluto.finish() (non-blocking)
[T+0.1s]  Training exits (doesn't wait for upload!)
[T+5s]    Sync process flushes remaining data
[T+10s]   Sync process marks run complete, exits
```

**Key**: Training exits IMMEDIATELY. No hanging.

### 4. Sync Process Crash

```
Timeline:
[T+0]     Sync process crashes (bug, OOM, etc.)
[T+0]     Training continues unaffected (writes to disk)
[T+0]     Data accumulates on disk
[Later]   Two recovery options:
          A) Training calls pluto.finish() → spawns new sync process
          B) User runs `pluto sync /path/to/run` CLI
```

**Key**: Training is decoupled. Sync crash doesn't affect training.

### 5. Network Failure

```
Timeline:
[T+0]     Network goes down
[T+0]     Sync process upload fails
[T+1s]    Retry with backoff (1s, 2s, 4s, 8s, 16s, 32s max)
[T+0]     Training continues, data accumulates on disk
[T+5min]  Network restored
[T+5min]  Sync process catches up, uploads backlog
```

**Key**: Unlimited local buffering. Network issues don't affect training.

### 6. Pluto Server Down

```
Timeline:
[T+0]     Pluto server returns 503
[T+0]     Sync process backs off exponentially
[T+0]     Training continues unaffected
[T+1hr]   Server restored
[T+1hr]   Sync process catches up
```

### 7. Disk Full

```
Timeline:
[T+0]     Disk full, SQLite write fails
[T+0]     pluto.log() catches error, logs warning
[T+0]     Training continues (log() is best-effort)
[T+0]     Metric is lost (acceptable degradation)
```

**Mitigation**: Monitor disk space, warn when low.

### 8. DDP - One Rank Crashes

```
Timeline:
[T+0]     Rank 2 crashes (OOM)
[T+0]     Rank 2's data is on disk
[T+0]     Ranks 0,1,3 continue (may hang on collective)
[T+30s]   NCCL timeout, other ranks fail
[T+30s]   All ranks' data is on disk
[T+35s]   Sync process (shared) detects all parents dead
[T+35s]   Sync process flushes all data, exits
```

**Key**: Sync process is per-node, not per-rank. One sync handles all ranks.

## API Design

```python
# pluto/__init__.py

def init(
    project: str,
    name: Optional[str] = None,
    config: Optional[dict] = None,
    tags: Optional[List[str]] = None,
    run_id: Optional[str] = None,  # For DDP resume

    # New options
    run_dir: Optional[str] = None,  # Override default /tmp/pluto-runs/
    sync_mode: str = "process",     # "process" (default) | "thread" (legacy) | "offline"
) -> Run:
    """
    Initialize a Pluto run.

    This creates a local run directory and optionally starts a sync process.
    The sync process handles all network I/O in the background.

    For DDP/FSDP:
    - All ranks should use the same run_id
    - First rank to init creates the sync process
    - Other ranks attach to existing sync process
    - All ranks write to shared SQLite (WAL mode supports this)
    """
    pass


class Run:
    def log(self, data: dict, step: Optional[int] = None) -> None:
        """
        Log metrics. This is a LOCAL operation (fast, never blocks).

        Data is written to SQLite immediately. The sync process
        handles uploading in the background.
        """
        # Write to SQLite - THAT'S IT
        self._store.insert(data, step)

    def finish(self, wait: bool = False, timeout: float = 30.0) -> None:
        """
        Mark run as finished.

        By default, this returns immediately. The sync process
        will flush remaining data and mark the run complete.

        Args:
            wait: If True, wait for sync to complete (up to timeout)
            timeout: Max seconds to wait if wait=True
        """
        # Set status in DB
        self._store.set_state("status", "finishing")

        if wait:
            # Optionally wait for sync
            self._wait_for_sync(timeout)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """
        Log a file artifact.

        The file is copied to the run directory. The sync process
        handles uploading in the background.
        """
        # Copy file to run dir
        # Insert record in files table
        pass
```

## DDP Coordination

```python
# How multiple ranks share one sync process

def init(..., run_id: str):
    run_dir = f"/tmp/pluto-runs/{run_id}"
    lock_file = f"{run_dir}/.sync.lock"

    with FileLock(lock_file):
        # Check if sync process already running
        existing_pid = _read_sync_pid(run_dir)

        if existing_pid and _is_alive(existing_pid):
            # Attach to existing sync process
            return Run(run_dir, sync_process=None)
        else:
            # Start new sync process
            sync_proc = start_sync_process(run_dir, ...)
            return Run(run_dir, sync_process=sync_proc)

# SQLite WAL mode allows concurrent writes from multiple ranks
# Each rank just appends to the same DB
# Sync process reads from all of them
```

## Recovery CLI

```bash
# If sync process died or was killed, recover data manually:

$ pluto sync /tmp/pluto-runs/my-run-id
Scanning run directory...
Found 15,432 unuploaded metrics
Found 3 unuploaded files
Uploading... [=================>] 100%
Run marked as complete.

# List local runs that need syncing:
$ pluto runs --pending
RUN_ID                    STATUS      PENDING_METRICS  PENDING_FILES
my-experiment-20240115    finishing   1,234            2
another-run-20240114      crashed     0                0

# Force-finish a stuck run:
$ pluto finish my-experiment-20240115 --force
```

## Configuration

```python
# Environment variables (for containerized training)
PLUTO_PROJECT=my-project
PLUTO_API_TOKEN=xxx
PLUTO_RUN_DIR=/tmp/pluto-runs       # Where to store local data
PLUTO_SYNC_MODE=process             # process | thread | offline

# Settings object
settings = {
    # Sync process settings
    "sync_poll_interval": 0.5,      # How often to check for new data
    "sync_batch_size": 1000,        # Max metrics per upload batch
    "sync_flush_timeout": 30.0,     # Max time to flush on finish
    "sync_retry_max": 5,            # Max upload retries
    "sync_retry_backoff": 2.0,      # Exponential backoff multiplier

    # Disk management
    "max_pending_metrics": 1_000_000,  # Start dropping if too many
    "max_pending_files_mb": 1000,      # Max file buffer size

    # Offline mode
    "offline": False,               # If True, never start sync process
}
```

## Migration from Thread-based Architecture

```python
# Phase 1: Sync process as opt-in (current)
pluto.init(settings={"sync_mode": "process"})

# Phase 2: Sync process as default, threads as fallback
pluto.init()  # Uses sync process
pluto.init(settings={"sync_mode": "thread"})  # Legacy

# Phase 3: Remove thread mode entirely
pluto.init()  # Only sync process
```

## Implementation Priority

1. **Core sync process** (process.py) - DONE (skeleton)
2. **Database schema** - Need to update store.py
3. **Integration with Op** - Wire up init/log/finish
4. **DDP coordination** - File locking, shared sync
5. **Recovery CLI** - `pluto sync` command
6. **File uploads** - Presigned URLs, S3
7. **Tests** - Unit, integration, chaos

## Open Questions

1. **Should sync process be long-lived (daemon) or per-run?**
   - Per-run is simpler, less state management
   - Daemon is more efficient for many short runs
   - Recommend: Per-run initially, optimize later

2. **How to handle very long training runs (days)?**
   - Sync process should be robust to restarts
   - Checkpoint sync state in DB
   - Allow manual intervention via CLI

3. **Should we support Windows?**
   - spawn works on Windows
   - File locking is different
   - Recommend: Linux/Mac first, Windows later
