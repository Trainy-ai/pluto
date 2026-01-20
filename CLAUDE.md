# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pluto** is a Machine Learning Operations (MLOps) framework providing experimental tracking and lifecycle management for ML models. This is the Python client library that communicates with the Pluto server infrastructure.

Note: The package was recently renamed from `mlop` to `pluto`. The `mlop` import is still supported for backward compatibility but is deprecated.

## Development Commands

### Setup
```bash
# Install dependencies (development mode)
pip install -e ".[dev,full]"

# Or using poetry
poetry install --with dev --extras full
```

### Testing
```bash
# Run all non-distributed tests
poetry run pytest -n auto -rs -m "not distributed" tests

# Run specific test file
poetry run pytest tests/test_basic.py

# Run specific test
poetry run pytest tests/test_basic.py::test_name

# Run distributed tests (DDP)
poetry run torchrun --standalone --nproc-per-node=2 -m pytest tests/test_pytorch.py -k test_pluto_watch_on_ddp_model -m distributed -rs
```

### Linting and Formatting
```bash
# Run all linting and formatting (recommended before commits)
bash format.sh

# Individual commands
poetry run ruff check --fix pluto mlop tests
poetry run ruff format pluto mlop tests
poetry run mypy pluto
```

### Authentication
```bash
# Login with API token
pluto login <token>

# Logout
pluto logout
```

## Architecture

### Core Components

**Op (Operation)**: Central abstraction representing a training run or experiment (pluto/op.py)
- Manages lifecycle: start, logging, finish
- Coordinates between data storage, server interface, and system monitoring
- Runs background workers for async data transmission and system monitoring

**Settings**: Configuration management (pluto/sets.py)
- Controls client behavior, API endpoints, and feature flags
- Default server: `https://pluto.trainy.ai` (production)
- Can be overridden with `host` parameter for self-hosted instances
- URL endpoints: `url_app`, `url_api`, `url_ingest`, `url_py`

**ServerInterface (iface.py)**: HTTP utility layer for Pluto server communication
- Provides HTTP client setup and retry logic
- Handles status updates, triggers, and direct API calls
- Note: Data upload (metrics, files, etc.) is handled by the sync process, not ServerInterface

**DataStore (store.py)**: Legacy local SQLite-based buffer for metrics
- Only used when sync process is disabled (legacy mode)
- Aggregates and batches data before transmission

**System (sys.py)**: System monitoring (CPU, GPU, memory, network)
- Samples hardware metrics at configurable intervals
- Supports NVIDIA GPU monitoring via nvidia-ml-py
- GPU metric format: `sys/gpu.{idx}.{device_name}.{metric}` where:
  - `idx` from `nvmlDeviceGetIndex()` - ensures unique keys for multiple GPUs of same model
  - `device_name` from `nvmlDeviceGetName()` - sanitized (lowercase, underscores)
  - Metrics: `utilization`, `memory.used`, `memory.utilization`, `power`
- Abbreviation expansion in `api.py` uses word boundaries to avoid issues (e.g., `mem` → `memory` but not inside `memory`)

### Data Types

**File types** (pluto/file.py): Image, Audio, Video, Text, Artifact
- All inherit from base `File` class
- Support both file paths and in-memory data
- Auto-detects format/extension

**Data types** (pluto/data.py): Graph, Histogram, Table
- Structured data for visualization
- Graph: network/tree visualizations
- Histogram: distribution data
- Table: tabular data with pandas-like API

### Compatibility Layer (pluto/compat/)

Integration hooks for popular ML frameworks:
- **torch.py**: PyTorch model watching (gradients, parameters, model graphs)
- **lightning.py**: PyTorch Lightning callback integration
- **transformers.py**: Hugging Face Transformers callback
- **neptune.py**: Neptune-to-Pluto migration compatibility layer

### API Communication (pluto/api.py)

Contains `make_compat_*_v1` functions that format data for server API v1:
- Converts Python objects to JSON payloads
- Handles timestamp formatting (ms conversion)
- Normalizes metric names (abbreviation expansion)

### Sync Process V2 Architecture (pluto/sync/)

The sync process is a separate spawned process that handles all network I/O for uploading data to the backend. This isolates the training process from network latency and failures.

**Key Components:**
- **SyncProcessManager** (pluto/sync/manager.py): Spawns and manages the sync child process
- **SyncProcessStore** (pluto/sync/store.py): SQLite-based storage with WAL mode for concurrent access
- **sync_worker** (pluto/sync/worker.py): Main loop running in the child process

**How it works:**
1. Training process writes metrics/files to SQLite database (fast, local)
2. Sync process reads from SQLite and uploads to backend (network I/O)
3. SQLite WAL mode allows concurrent reads/writes without blocking
4. On crash, data persists in SQLite for recovery

**Configuration:**
- `settings.sync_process_enabled` - Enable/disable sync process (default: True)
- `settings.sync_process_shutdown_timeout` - Max wait time for sync on finish (default: 30s)

**Supported Data Types:**
- Numeric metrics (int, float)
- System metrics (CPU, GPU, memory)
- Config updates
- Tags updates
- Structured data (Graph, Histogram, Table)
- Files (Image, Audio, Video, Text, Artifact)

**Critical Design Decisions:**

1. **Subprocess instead of multiprocessing**: Uses `subprocess.Popen` to start `python -m pluto.sync`:
   - Avoids the `__main__` re-import problem of `multiprocessing.Process` with spawn
   - No `if __name__ == '__main__':` guard required in user scripts
   - Settings passed as JSON via CLI args (not pickle)

2. **SQLite WAL mode**: Enables concurrent access between training and sync processes
   - Training writes don't block on sync reads
   - Sync reads don't block training writes
   - Data persists even if sync process crashes

3. **FileLock for DDP**: In distributed training, multiple ranks may initialize simultaneously
   - FileLock prevents race conditions when creating the sync process
   - Only rank 0 typically needs the sync process, but all ranks may call init()

## Typical Workflow

1. **Initialize**: `pluto.init(project="name")` creates an Op instance
2. **Log**: `pluto.log({"metric": value})` queues data for transmission
3. **Watch** (optional): `pluto.watch(model)` for PyTorch model tracking
4. **Finish**: `pluto.finish()` flushes buffers and marks run complete

The Op instance is stored in `pluto.ops` list and made available globally. Background threads handle async data transmission.

### Tags Support

Tags enable categorizing and filtering runs. Tags automatically sync to the server via tRPC:

```python
# Initialize with tags
run = pluto.init(project="name", tags="experiment")
run = pluto.init(project="name", tags=["production", "v2", "baseline"])

# Add tags dynamically (syncs to server)
run.add_tags("new-feature")
run.add_tags(["validated", "ready"])

# Remove tags (syncs to server)
run.remove_tags("old-tag")
run.remove_tags(["deprecated", "archived"])
```

**Implementation details**:
- **Client-side** (pluto/op.py, pluto/init.py, pluto/api.py):
  - Tags stored as `List[str]` on Op instance
  - Duplicate tags automatically prevented
  - Initial tags sent via `POST /api/runs/create` endpoint
  - Dynamic updates sent via HTTP POST to `/api/runs/tags/update`

- **Server synchronization** (pluto/iface.py, pluto/sets.py):
  - Full tags array sent on each update (not incremental)
  - Graceful error handling (logs debug, doesn't break on failure)
  - URL: `{url_api}/api/runs/tags/update`

- **Backend integration**:
  - HTTP POST endpoint: `/api/runs/tags/update`
  - Payload: `{ "runId": <numeric_id>, "tags": [...] }`
  - Uses numeric run ID (not SQID-encoded)
  - Replaces entire tags array (idempotent)
  - See: https://github.com/Trainy-ai/server-private/pull/15

- **Neptune compatibility**:
  - Neptune compat layer uses native tags API
  - Both `add_tags()` and `remove_tags()` work seamlessly

## Important Implementation Details

### Thread Safety
- OpMonitor runs a background thread for system metrics and heartbeats
- Clean shutdown via `_stop_event` and thread joining
- Data upload is handled by sync process (separate process, not threads)

### Subprocess-based Sync Process

The sync process uses `subprocess.Popen` instead of `multiprocessing.Process` to avoid the `__main__` re-import problem:

```python
# In pluto/sync/process.py - starts as independent process
process = subprocess.Popen([
    sys.executable, '-m', 'pluto.sync',
    '--db-path', db_path,
    '--settings', json.dumps(settings_dict),
    '--parent-pid', str(os.getpid()),
])
```

**Why subprocess instead of multiprocessing:**
- `multiprocessing.Process` with spawn context re-imports `__main__`, causing user scripts to run twice
- `subprocess.Popen` starts a completely independent Python interpreter
- No `if __name__ == '__main__':` guard required in user scripts
- Settings passed via CLI args (JSON), not pickle

**The CLI entry point** (`pluto/sync/__main__.py`):
- Allows sync process to be invoked as `python -m pluto.sync`
- Parses `--db-path`, `--settings`, and `--parent-pid` arguments
- Calls `_sync_main()` with the parsed parameters

### DDP/Distributed Training Considerations

**The blocking problem:**
In DDP (DistributedDataParallel), all ranks must progress together for collective operations. If any rank blocks in `finish()`, it causes deadlock.

**What didn't work:**
- Disabling sync process entirely in DDP - loses crash-safety benefits
- Using `wait=True` in `finish()` - blocks collective operations

**What works:**
Use `wait=False` in DDP environments (same approach as SIGTERM preemption):
```python
# In Op.finish():
if _is_distributed_environment():
    self._sync_manager.stop(wait=False)  # Signal but don't block
else:
    self._sync_manager.stop(wait=True)   # Normal: wait for completion
```

**Why `wait=False` is safe:**
1. Signals sync process to finish (marks run as "finished" in DB)
2. Returns immediately without blocking
3. Sync process continues flushing data in background
4. Data preserved in SQLite if process exits before flush completes

**DDP detection** (pluto/op.py `_is_distributed_environment()`):
- Checks `torch.distributed.is_initialized()`
- Checks environment variables: `WORLD_SIZE`, `RANK`, `LOCAL_RANK`
- Checks SLURM variables: `SLURM_PROCID`, `SLURM_NTASKS`

### Neptune Compatibility Layer Notes

The Neptune compat layer (pluto/compat/neptune.py) has special requirements:
- Uses 5-second cleanup timeout (Neptune API contract)
- Uses sync process with a short shutdown timeout (3s) to fit within Neptune's cleanup window
- Signal handlers disabled to preserve Neptune's exit behavior
- This ensures multi-GPU (DDP/FSDP) exit logic is identical to Neptune-only usage

### Configuration Precedence
Settings can be provided via:
1. Function parameters (highest priority)
2. Settings object/dict passed to `init()`
3. Environment variables (via `setup()` function)
4. Default values in Settings class

### Environment Variables
Environment variables use the `PLUTO_*` prefix. The old `MLOP_*` prefix is supported with deprecation warnings.

**Authentication & Project:**
- `PLUTO_API_TOKEN` - API token for authentication (alternative to `pluto login`)
- `PLUTO_PROJECT` - Default project name (alternative to `pluto.init(project="...")`)

**Configuration:**
- `PLUTO_DEBUG_LEVEL` - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `PLUTO_URL_APP`, `PLUTO_URL_API`, `PLUTO_URL_INGEST`, `PLUTO_URL_PY` - Server URLs

**Deprecated (still supported with warnings):**
- `MLOP_API_TOKEN`, `MLOP_PROJECT`, `MLOP_DEBUG_LEVEL`, `MLOP_URL_*`

### Testing Notes
- Tests run against production server by default
- Requires authentication via `PLUTO_API_TOKEN` environment variable
- Tests marked with `@pytest.mark.distributed` require multi-rank torch setup
- Use `HAS_TORCH`, `HAS_MATPLOTLIB` flags for optional dependency tests

### File Streaming
- Files are uploaded to pre-signed URLs obtained from server
- Chunked upload with retry logic
- Configurable via `x_file_stream_*` settings

### Versioning
- Version defined in `pyproject.toml` and `pluto/__init__.py`
- Git commit SHA embedded in builds for traceability

## Lessons Learned (PR #27 - Sync Process V2)

### What Worked Well

1. **SQLite WAL mode for IPC**: Using SQLite with WAL mode as the communication layer between training and sync processes works excellently. It's fast, reliable, and provides crash-safety for free.

2. **`wait=False` shutdown pattern**: For environments that can't block (DDP, preemption), signaling the sync process without waiting is the right approach. Data is preserved in SQLite.

3. **subprocess.Popen instead of multiprocessing.Process**: Using `subprocess.Popen` to start `python -m pluto.sync` avoids the `__main__` re-import problem entirely. No `if __name__ == '__main__':` guard required.

### What Didn't Work

1. **Disabling sync process in DDP**: Initial approach was to fall back to thread-based implementation in DDP. This loses the crash-safety benefits. Better solution: use `wait=False`.

2. **multiprocessing.Process with spawn**: The spawn context re-imports `__main__`, causing user scripts to run twice if they don't have `if __name__ == '__main__':` guards. Solution: use subprocess.Popen instead.

3. **Same shutdown timeout for all contexts**: Neptune compat has 5s timeout, sync process has 30s. These conflict. Solution: use a shorter shutdown timeout (3s) for Neptune compat.

### Performance Characteristics

- **Per-log overhead**: ~2-3ms for SQLite write (vs ~0ms for in-memory queue)
- **Worst-case behavior**: Much better - network issues don't block training
- **Memory**: Lower - data stored in SQLite, not Python memory
- **Crash recovery**: Data persists in SQLite for later upload

### Debugging Tips

1. **Hanging in DDP?** Check if `finish()` is blocking. Use `wait=False` for distributed environments.

2. **Data not uploading?** Check SQLite database in the run directory. Use `sqlite3` to inspect pending data.

3. **CI tests hanging?** Often caused by sync process shutdown blocking. Check timeout values and distributed detection.

4. **Sync process not starting?** Check if `python -m pluto.sync` can be invoked manually. Check subprocess stderr for errors.
