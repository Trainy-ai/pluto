# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mlop** is a Machine Learning Operations (MLOps) framework providing experimental tracking and lifecycle management for ML models. This is the Python client library that communicates with the mlop server infrastructure.

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
poetry run torchrun --standalone --nproc-per-node=2 -m pytest tests/test_pytorch.py -k test_mlop_watch_on_ddp_model -m distributed -rs
```

### Linting and Formatting
```bash
# Run all linting and formatting (recommended before commits)
bash format.sh

# Individual commands
poetry run ruff check --fix mlop tests
poetry run ruff format mlop tests
poetry run mypy mlop
```

### Authentication
```bash
# Login with API token
mlop login <token>

# Logout
mlop logout
```

## Architecture

### Core Components

**Op (Operation)**: Central abstraction representing a training run or experiment (mlop/op.py)
- Manages lifecycle: start, logging, finish
- Coordinates between data storage, server interface, and system monitoring
- Runs background workers for async data transmission and system monitoring

**Settings**: Configuration management (mlop/sets.py)
- Controls client behavior, API endpoints, and feature flags
- Default server: `https://trakkur.trainy.ai` (production)
- Can be overridden with `host` parameter for self-hosted instances
- URL endpoints: `url_app`, `url_api`, `url_ingest`, `url_py`

**ServerInterface (iface.py)**: HTTP communication layer with mlop server
- Uses httpx with HTTP/2 support for efficient data streaming
- Handles retries, timeouts, and connection pooling
- Publishes metrics, files, graphs, and system stats

**DataStore (store.py)**: Local SQLite-based buffer for metrics
- Aggregates and batches data before transmission
- Reduces network overhead during high-frequency logging

**System (sys.py)**: System monitoring (CPU, GPU, memory, network)
- Samples hardware metrics at configurable intervals
- Supports NVIDIA GPU monitoring via nvidia-ml-py

### Data Types

**File types** (mlop/file.py): Image, Audio, Video, Text, Artifact
- All inherit from base `File` class
- Support both file paths and in-memory data
- Auto-detects format/extension

**Data types** (mlop/data.py): Graph, Histogram, Table
- Structured data for visualization
- Graph: network/tree visualizations
- Histogram: distribution data
- Table: tabular data with pandas-like API

### Compatibility Layer (mlop/compat/)

Integration hooks for popular ML frameworks:
- **torch.py**: PyTorch model watching (gradients, parameters, model graphs)
- **lightning.py**: PyTorch Lightning callback integration
- **transformers.py**: Hugging Face Transformers callback

### API Communication (mlop/api.py)

Contains `make_compat_*_v1` functions that format data for server API v1:
- Converts Python objects to JSON payloads
- Handles timestamp formatting (ms conversion)
- Normalizes metric names (abbreviation expansion)

## Typical Workflow

1. **Initialize**: `mlop.init(project="name")` creates an Op instance
2. **Log**: `mlop.log({"metric": value})` queues data for transmission
3. **Watch** (optional): `mlop.watch(model)` for PyTorch model tracking
4. **Finish**: `mlop.finish()` flushes buffers and marks run complete

The Op instance is stored in `mlop.ops` list and made available globally. Background threads handle async data transmission.

### Tags Support

Tags enable categorizing and filtering runs. Tags automatically sync to the server via tRPC:

```python
# Initialize with tags
run = mlop.init(project="name", tags="experiment")
run = mlop.init(project="name", tags=["production", "v2", "baseline"])

# Add tags dynamically (syncs to server)
run.add_tags("new-feature")
run.add_tags(["validated", "ready"])

# Remove tags (syncs to server)
run.remove_tags("old-tag")
run.remove_tags(["deprecated", "archived"])
```

**Implementation details**:
- **Client-side** (mlop/op.py, mlop/init.py, mlop/api.py):
  - Tags stored as `List[str]` on Op instance
  - Duplicate tags automatically prevented
  - Initial tags sent via `POST /api/runs/create` endpoint
  - Dynamic updates sent via HTTP POST to `/api/runs/tags/update`

- **Server synchronization** (mlop/iface.py, mlop/sets.py):
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
- Op uses queues and threading for async data transmission
- OpMonitor runs two background threads: data worker and system monitor
- Clean shutdown via `_stop_event` and thread joining

### Configuration Precedence
Settings can be provided via:
1. Function parameters (highest priority)
2. Settings object/dict passed to `init()`
3. Environment variables (via `setup()` function)
4. Default values in Settings class

### Testing Notes
- Tests run against production server by default
- Requires authentication via `MLOP_API_TOKEN` environment variable
- Tests marked with `@pytest.mark.distributed` require multi-rank torch setup
- Use `HAS_TORCH`, `HAS_MATPLOTLIB` flags for optional dependency tests

### File Streaming
- Files are uploaded to pre-signed URLs obtained from server
- Chunked upload with retry logic
- Configurable via `x_file_stream_*` settings

### Versioning
- Version defined in `pyproject.toml` and `mlop/__init__.py`
- Git commit SHA embedded in builds for traceability
