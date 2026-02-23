# Design: Pluto Read/Query API for Neptune Migration

**Date:** 2025-02-17
**Status:** Proposal
**Author:** Andrew (Trainy)

## Context

A customer migrating from Neptune to Pluto needs programmatic read/query access to runs. Their workflow:

1. **Pull a run by ID** — after a training job completes, a separate eval pipeline opens the run
2. **Fetch metadata** — read config values (checkpoint paths, hyperparameters)
3. **Run async evaluation** — using the fetched checkpoint
4. **Append results to the same run** — upload eval metrics + plots at the checkpoint's step

The Pluto Python SDK (`pluto-ml`) is currently **write-only**. However, the Pluto server already exposes a full set of **REST read endpoints** at `/api/runs/*` with API-key authentication. These endpoints power the MCP server and the OpenAPI spec. We just need a Python client that wraps them.

## Goals

- Provide Neptune read/query parity sufficient for migration
- Thin wrapper over existing server REST endpoints (no new backend work)
- Return familiar Python types (dicts, lists, pandas DataFrames)
- Support the customer's two-phase workflow: read run → append eval results
- Consistent with existing SDK patterns (Settings, auth, error handling)

## Non-Goals

- NQL (Neptune Query Language) — Pluto has its own filtering model via query params
- Local-only reads from SQLite sync DB — server is the source of truth for reads
- Full `run["path/to/key"]` bracket-access API — too much surface area for V1
- Streaming/real-time subscriptions

---

## Existing Server Read Endpoints

The Pluto server already has these REST endpoints (API-key auth via `x-api-key` header):

| Endpoint | Description |
|---|---|
| `GET /api/runs/list` | List runs with search, tag filtering, pagination |
| `GET /api/runs/details/{runId}` | Full run details (config, tags, status, metadata) |
| `GET /api/runs/details/by-display-id/{displayId}` | Run details by display ID (e.g. "MMP-1") |
| `GET /api/runs/metrics` | Time-series metrics with reservoir sampling |
| `GET /api/runs/files` | File metadata with presigned download URLs |
| `GET /api/runs/logs` | Console logs |
| `GET /api/runs/projects` | List all projects |
| `GET /api/runs/metric-names` | Distinct metric names in a project |
| `GET /api/runs/statistics` | Min/max/mean/stddev + anomaly detection |
| `GET /api/runs/compare` | Compare metrics across runs |
| `GET /api/runs/leaderboard` | Rank runs by metric aggregation |
| `GET /api/runs/auth/validate` | Validate API key |

These are the same endpoints the Pluto MCP server calls. The Python SDK just needs to wrap them.

---

## Proposed API

### Module: `pluto.query`

New module at `pluto/query.py`. All functions are standalone (no global state). Users pass project name and authenticate via the same `PLUTO_API_TOKEN` used for logging.

### Initialization

```python
import pluto.query as pq

# Uses PLUTO_API_TOKEN env var and default server URL
runs = pq.list_runs("my-project")

# Explicit auth and custom server
client = pq.Client(
    api_token="plt_...",
    host="https://pluto.example.com",  # self-hosted
)
runs = client.list_runs("my-project")
```

The module-level functions (`pq.list_runs(...)`) are convenience wrappers that create a `Client` from environment/settings. The `Client` class holds auth + base URL and is reusable.

### Core API

#### `Client` class

```python
class Client:
    def __init__(
        self,
        api_token: str | None = None,   # Default: PLUTO_API_TOKEN env var
        host: str | None = None,         # Default: https://pluto.trainy.ai
    ): ...
```

Internally creates an `httpx.Client` with `x-api-key` header. Reuses the URL resolution logic from `Settings` (derives `url_api` from `host`).

#### List projects

```python
client.list_projects() -> list[dict]
```

Returns list of projects with `id`, `name`, `runCount`, `createdAt`, `updatedAt`.

Maps to: `GET /api/runs/projects`

#### List runs

```python
client.list_runs(
    project: str,
    search: str | None = None,        # Full-text search on run name
    tags: list[str] | None = None,    # Filter by tags (AND logic)
    limit: int = 50,                  # Max 200
) -> list[dict]
```

Returns list of run dicts with `id`, `name`, `displayId`, `status`, `tags`, `config`, `createdAt`, `updatedAt`, `url`.

Maps to: `GET /api/runs/list?projectName=...&search=...&tags=...&limit=...`

#### Get run details

```python
client.get_run(
    project: str,
    run_id: int | str,  # Numeric ID or display ID (e.g. "MMP-1")
) -> dict
```

Returns full run details: `id`, `name`, `displayId`, `status`, `tags`, `config`, `systemMetadata`, `logNames`, `createdAt`, `updatedAt`, `url`.

Maps to: `GET /api/runs/details/{runId}` or `GET /api/runs/details/by-display-id/{displayId}`

#### Fetch metrics

```python
client.get_metrics(
    project: str,
    run_id: int,
    metric_names: list[str] | None = None,  # None = all metrics
    limit: int = 10000,
) -> pd.DataFrame  # Columns: metric, step, value, time
```

Returns a pandas DataFrame with all requested metric series. If `pandas` is not installed, returns a list of dicts instead.

Maps to: `GET /api/runs/metrics?runId=...&projectName=...&logName=...`

#### List metric names

```python
client.get_metric_names(
    project: str,
    run_id: int | None = None,   # None = all metrics in project
    search: str | None = None,   # Filter by name substring
) -> list[str]
```

Maps to: `GET /api/runs/metric-names?projectName=...&runIds=...&search=...`

#### Fetch files

```python
client.get_files(
    project: str,
    run_id: int,
    file_name: str | None = None,  # Filter by log name
) -> list[dict]  # Each has: fileName, fileType, fileSize, step, time, downloadUrl
```

Returns file metadata with presigned download URLs. Users can download with `httpx`/`requests`/`urllib`.

Maps to: `GET /api/runs/files?runId=...&projectName=...&logName=...`

#### Download file

```python
client.download_file(
    project: str,
    run_id: int,
    file_name: str,
    destination: str | Path = ".",  # Directory or file path
) -> Path  # Path to downloaded file
```

Convenience method: calls `get_files()`, then downloads via presigned URL.

#### Fetch logs

```python
client.get_logs(
    project: str,
    run_id: int,
    log_type: str | None = None,  # "info", "error", "warning", "debug", "print"
    limit: int = 10000,
) -> list[dict]  # Each has: message, logType, time, lineNumber, step
```

Maps to: `GET /api/runs/logs?runId=...&projectName=...&logType=...&limit=...`

#### Statistics

```python
client.get_statistics(
    project: str,
    run_id: int,
    metric_names: list[str] | None = None,
) -> dict  # Per-metric: count, min, max, mean, stddev, first, last
```

Maps to: `GET /api/runs/statistics?runId=...&projectName=...`

#### Compare runs

```python
client.compare_runs(
    project: str,
    run_ids: list[int],
    metric_name: str,
) -> dict  # Per-run stats + best run recommendation
```

Maps to: `GET /api/runs/compare?runIds=...&projectName=...&logName=...`

#### Leaderboard

```python
client.leaderboard(
    project: str,
    metric_name: str,
    aggregation: str = "LAST",  # MIN, MAX, AVG, LAST, VARIANCE
    direction: str = "ASC",     # ASC or DESC
    limit: int = 50,
) -> list[dict]  # Ranked runs with metric values
```

Maps to: `GET /api/runs/leaderboard?projectName=...&logName=...&aggregation=...`

---

### Neptune Migration: Resume Run for Writing

The customer's workflow requires opening an existing run and appending data. This requires a change to `pluto.init()`, not the query module.

#### Current behavior

```python
# Creates a NEW run, or resumes if run_id matches an existing external_id
run = pluto.init(project="my-project", run_id="my-external-id")
```

The `run_id` parameter is the **external ID** — a user-provided string for multi-node coordination. The server returns `resumed=true` if a run with that `run_id` already exists.

#### Proposed: `with_id` parameter

```python
# Resume an existing run by its server-assigned numeric ID
run = pluto.init(project="my-project", with_id=12345)

# Resume by display ID
run = pluto.init(project="my-project", with_id="MMP-1")

# Read-only mode (no sync process, no system monitoring)
run = pluto.init(project="my-project", with_id="MMP-1", mode="read-only")
```

**Implementation:**

1. Add `with_id: int | str | None = None` and `mode: str = "async"` parameters to `pluto.init()`
2. When `with_id` is provided:
   - If `mode="read-only"`: return a lightweight `ReadOnlyRun` proxy that wraps `Client.get_run()`, `Client.get_metrics()`, etc. No background workers, no sync process, no system monitoring.
   - If `mode="async"` (default): call `/api/runs/create` with the server ID to resume the run for writing. Start sync process and background workers as usual. The `name` parameter is ignored on resume.
3. The server's `/api/runs/create` endpoint already supports resumption — it just needs to also accept the server-assigned run ID (not just external ID) as a lookup key. **This is the only backend change needed.**

#### `ReadOnlyRun` class

```python
class ReadOnlyRun:
    """Lightweight read-only run proxy. No background workers."""

    def __init__(self, client: Client, project: str, run_id: int): ...

    @property
    def id(self) -> int: ...

    @property
    def name(self) -> str: ...

    @property
    def config(self) -> dict: ...

    @property
    def tags(self) -> list[str]: ...

    @property
    def status(self) -> str: ...

    def fetch(self, key: str) -> Any:
        """Fetch a config value by key path. e.g. run.fetch('checkpoint_path')"""
        ...

    def fetch_metrics(self, metric_names=None, limit=10000) -> pd.DataFrame:
        """Fetch metric series as a DataFrame."""
        ...

    def fetch_files(self, file_name=None) -> list[dict]:
        """Fetch file metadata with download URLs."""
        ...

    def download(self, file_name: str, destination=".") -> Path:
        """Download a file artifact."""
        ...
```

---

### Neptune Compat Layer Update

Update `pluto/compat/neptune.py` to support read operations through the compat wrapper:

```python
# Neptune pattern:
run = neptune.init_run(with_id="PROJ-123", mode="read-only")
checkpoint = run["model/checkpoint_path"].fetch()

# With updated compat layer, this translates to:
# → pluto.init(with_id="PROJ-123", mode="read-only")
# → run.fetch("model/checkpoint_path")
```

This is a lower priority — the customer can use `pluto.query` directly during migration. The compat layer can be updated later for teams that want a drop-in replacement.

---

## Implementation Plan

### Phase 1: `pluto.query` module (no backend changes)

1. **`pluto/query.py`** — `Client` class wrapping all `GET /api/runs/*` endpoints
2. **Module-level convenience functions** — `pluto.query.list_runs(...)`, etc.
3. **Export from `pluto/__init__.py`** — `import pluto.query` works
4. **Tests** — unit tests with mocked HTTP, integration tests against staging
5. **Dependencies** — `httpx` (already a dependency), `pandas` optional

This phase requires **zero backend changes**. All endpoints already exist.

### Phase 2: Resume run by server ID

1. **Backend change** — `/api/runs/create` accepts server-assigned run ID for resumption
2. **`pluto.init(with_id=...)`** — add parameter, wire to backend
3. **`ReadOnlyRun` class** — lightweight proxy for read-only mode
4. **Tests** — resume by numeric ID, resume by display ID, read-only mode

### Phase 3: Neptune compat layer read support (optional)

1. **`NeptuneRunWrapper`** — support `run["key"].fetch()` bracket access
2. **`init_run(with_id=..., mode="read-only")`** interception
3. **`init_project().fetch_runs_table()`** interception

---

## Customer's Workflow with Proposed API

```python
import pluto
import pluto.query as pq

# ---- Phase 1: Read from completed training run ----

# Get the run (using display ID from dashboard URL)
run_details = pq.get_run("my-project", "MMP-42")

# Get checkpoint path from config
checkpoint_path = run_details["config"]["checkpoint_path"]
training_step = run_details["config"]["total_steps"]

# Get the best validation loss
metrics = pq.get_metrics("my-project", run_details["id"], metric_names=["val/loss"])
best_step = metrics.loc[metrics["value"].idxmin(), "step"]

# ---- Phase 2: Run async evaluation ----
# (load model from checkpoint_path, run eval, produce results)
eval_results = run_evaluation(checkpoint_path)

# ---- Phase 3: Append eval results to the same run ----

# Resume run for writing (Phase 2 feature, requires backend change)
run = pluto.init(project="my-project", with_id=run_details["id"])

# Log eval metrics at the training step
run.log({
    "eval/accuracy": eval_results["accuracy"],
    "eval/f1": eval_results["f1"],
}, step=training_step)

# Upload eval plots
run.log({
    "eval/heatmap": pluto.Image("heatmap.png"),
    "eval/forecast": pluto.Image("forecast.png"),
}, step=training_step)

run.finish()
```

---

## Alternatives Considered

### A. JSON-RPC 2.0 client wrapping MCP server

Claude (the customer's AI assistant) suggested building a JSON-RPC client to call the MCP server at `pluto-mcp.trainy.ai`. This was rejected because:

- The MCP server is in alpha with no public API docs
- The MCP server itself wraps the same REST endpoints we'd call directly
- Adding a JSON-RPC layer adds latency and complexity for no benefit
- The REST endpoints have a documented OpenAPI spec

### B. tRPC client

The server's tRPC routes have richer filtering (metric-based sorting, cross-DB joins). However:

- tRPC uses session auth (cookie-based), not API keys
- tRPC routes are an internal frontend API, not a stable public contract
- The REST OpenAPI endpoints provide sufficient functionality for V1
- If needed, we can add tRPC support later for advanced filtering

### C. GraphQL / custom query language

Over-engineered for V1. The REST endpoints with query parameters cover the customer's use case. A query language can be added later if demand exists.

---

## Open Questions

1. **Display ID format** — Is the display ID (e.g., "MMP-1") stable and unique within a project? Can we reliably use it for `with_id`?

2. **Run resume by server ID** — Does the `/api/runs/create` endpoint need modification to accept a server-assigned numeric ID for resumption, or can we use the existing `run_id` (external ID) path by setting `PLUTO_RUN_ID` to the server ID?

3. **Pagination for metrics** — The `/api/runs/metrics` endpoint uses reservoir sampling (2000 points per metric). Should `get_metrics()` support fetching the full unsampled series? This may require a new backend endpoint or a `full=true` parameter.

4. **File download auth** — Are the presigned download URLs from `/api/runs/files` sufficient, or do they expire too quickly for batch download workflows?

5. **Rate limiting** — Are the REST endpoints rate-limited? Should the `Client` class implement backoff?
