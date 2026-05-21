---
name: pluto-primary
description: "Primary skill for working with Pluto (Trainy's MLOps experiment-tracking platform). Covers the Pluto Python SDK (training runs, metrics, media, artifacts, tags, system stats) and the Pluto MCP server (querying runs, comparing experiments, fetching metrics/logs/files, leaderboards). Use this for: instrumenting a training script, inspecting or comparing runs, fetching metrics or logs, managing tags/notes, and answering broad 'what's going on with my experiments' questions."
---

# Pluto Primary Skill

This skill teaches agents how to (1) **instrument** ML training code with the Pluto Python SDK and (2) **query/analyze** existing runs through the Pluto MCP server. Reach for it whenever a task touches Pluto runs, metrics, or experiment-tracking patterns.

## Environment Defaults

- **Install:** `pip install "pluto[full]"` (minimal: `pip install pluto`).
- **Auth:** `pluto login <token>` or `PLUTO_API_KEY` env var. Project default: `PLUTO_PROJECT`.
- **Server:** defaults to `https://pluto.trainy.ai`. Self-hosted: pass `host=...` to `pluto.init()` or set `PLUTO_URL_*` env vars.
- **Backwards compat:** `import mlop` still works but is deprecated. `MLOP_*` env vars are honored with a warning. New code should use `pluto` / `PLUTO_*`.
- **Python:** 3.9+.

## When To Use What

| Goal | Use |
|------|-----|
| Add tracking to a training script | Python SDK (`pluto.init` / `log` / `finish`) |
| PyTorch model/grad/param watching | SDK: `pluto.watch(model)` |
| Lightning integration | SDK: `from pluto.compat.lightning import MLOPLogger` |
| HuggingFace Trainer | SDK: `from pluto.compat.transformers import PlutoCallback` |
| DDP / multi-node / SLURM | SDK with `wait=False` on `finish()` (see Distributed) |
| "What runs are in project X?" | MCP: `list_runs` |
| "Show me run R's metrics" | MCP: `get_run` → `query_metrics` |
| "Compare runs A, B, C" | MCP: `compare_runs` + `visualize_comparison` |
| "What's the best run by metric M?" | MCP: `get_leaderboard` |
| Stream stdout/stderr from a run | MCP: `query_logs` |
| Tag / untag / annotate a run | MCP: `add_tags`, `remove_tags`, `update_notes` |
| Shareable URL for a comparison | MCP: `generate_comparison_url` |

Rule of thumb: **mutating state on a *running* experiment → SDK.** **Inspecting or annotating *existing* runs → MCP.**

## Fast Recipes — SDK

### R1. Minimal training run

```python
import pluto

run = pluto.init(
    project="my-project",
    name="baseline-v1",         # optional; auto-generated if omitted
    config={"lr": 1e-3, "epochs": 10, "batch_size": 32},
    tags=["baseline", "v1"],     # str or list[str]
)
try:
    for epoch in range(10):
        # ... train step ...
        run.log({"train/loss": loss, "val/acc": acc, "epoch": epoch})
finally:
    run.finish()
```

Key facts:
- `pluto.init()` returns an `Op`. Assign it (`run` / `op`).
- `run.log()` takes a dict; values are numeric or Pluto data types (`Image`, `Audio`, `Histogram`, `Table`, ...).
- Use `/`-separated keys (`train/loss`, `val/loss`) for grouping in the UI.
- Always call `finish()` — wrap in `try/finally`.

### R2. PyTorch model watching

```python
import pluto, torch.nn as nn

run = pluto.init(project="cifar")
model = MyModel()
pluto.watch(model, log="all", log_freq=100)   # gradients + parameters every 100 steps
# ... train loop with run.log(...) ...
run.finish()
```

### R3. Lightning

```python
from pytorch_lightning import Trainer
from pluto.compat.lightning import MLOPLogger

logger = MLOPLogger(project="my-project", name="lit-run", config=hparams)
trainer = Trainer(logger=logger, max_epochs=10)
trainer.fit(model, dm)
# logger.finalize("success") is called automatically by Trainer
```

### R4. HuggingFace Transformers

```python
from transformers import Trainer, TrainingArguments
from pluto.compat.transformers import PlutoCallback

args = TrainingArguments(output_dir="out", report_to="none")  # disable HF's reporters
trainer = Trainer(model=model, args=args, callbacks=[PlutoCallback(project="hf-runs")])
trainer.train()
```

### R5. Logging media & structured data

```python
import pluto
run = pluto.init(project="vision")

run.log({"samples": pluto.Image(tensor_or_path, caption="epoch 3")})
run.log({"audio": pluto.Audio(waveform, sample_rate=16000)})
run.log({"loss_dist": pluto.Histogram(losses)})
run.log({"preds": pluto.Table(columns=["id", "label", "pred"], data=rows)})
run.log({"ckpt": pluto.Artifact("checkpoint.pt", type="model")})
```

### R6. Dynamic tags & notes (during a run)

```python
run.add_tags("converged")
run.add_tags(["candidate", "for-review"])
run.remove_tags("baseline")
```

Tags sync to the server on every call (idempotent, full-array replacement). Duplicates are dropped client-side.

### R7. Distributed / DDP

```python
import pluto, torch.distributed as dist

# Every rank can call init(); Pluto handles the rank-0-only behavior + FileLock
run = pluto.init(project="ddp-run", config=cfg)
try:
    # ... training ...
    if dist.get_rank() == 0:
        run.log({"loss": loss})
finally:
    run.finish()   # internally uses wait=False under DDP to avoid collective-op deadlock
```

Pluto detects DDP via `torch.distributed.is_initialized()` and env vars (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `SLURM_PROCID`, `SLURM_NTASKS`). The sync subprocess flushes asynchronously; data is persisted in SQLite even if a rank exits before the flush completes.

### R8. Hyperparameter sweep skeleton

```python
import itertools, pluto

grid = {"lr": [1e-3, 3e-4], "wd": [0.0, 1e-4]}
for lr, wd in itertools.product(*grid.values()):
    run = pluto.init(
        project="sweep-x",
        name=f"lr{lr}-wd{wd}",
        config={"lr": lr, "wd": wd},
        tags=["sweep", "grid-v1"],
    )
    try:
        train(lr=lr, wd=wd, log=run.log)
    finally:
        run.finish()
```

Use a shared tag (e.g. `grid-v1`) so the MCP side can pull the whole sweep with `list_runs`.

## Fast Recipes — MCP

The Pluto MCP server exposes these tools (names as registered; an installed server may prefix them). Use them for read/analysis tasks.

### M1. Find runs in a project

```
list_projects()
list_runs(project="my-project", tags=["sweep", "grid-v1"], limit=50)
```

### M2. Inspect a single run

```
get_run(run_id="r_abc123")
list_metric_names(run_id="r_abc123")
query_metrics(run_id="r_abc123", metric="train/loss")
get_statistics(run_id="r_abc123", metric="val/acc")     # min/max/last/mean
```

### M3. Compare runs

```
compare_runs(run_ids=["r_a", "r_b", "r_c"], metrics=["val/acc", "train/loss"])
visualize_comparison(run_ids=[...], metric="val/acc")   # returns an image
generate_comparison_url(run_ids=[...])                  # shareable link
```

### M4. Leaderboard

```
get_leaderboard(project="my-project", metric="val/acc", direction="max", limit=20)
```

### M5. Stream stdout/stderr

```
query_logs(run_id="r_abc123", stream="stdout", tail=200)
```

### M6. Files & artifacts

```
get_files(run_id="r_abc123")           # lists logged files + URLs
```

### M7. Annotate runs

```
add_tags(run_id="r_abc123", tags=["validated"])
remove_tags(run_id="r_abc123", tags=["wip"])
update_notes(run_id="r_abc123", notes="Beats baseline by 1.4pp; promote to v2.")
```

## Critical Rules

1. **Always call `finish()`.** Without it, the run never transitions to a terminal state and buffered data may not flush. Use `try/finally`.
2. **`log()` keys must be stable.** Don't change a metric's name between epochs — agents downstream rely on consistent series.
3. **Numeric values only for raw `log()`.** Use Pluto data types (`Image`, `Audio`, `Histogram`, `Table`, `Artifact`) for everything else. Strings/None will be rejected or coerced.
4. **DDP: do not wrap `finish()` in a barrier.** Pluto already uses `wait=False` under DDP. Calling `dist.barrier()` immediately after `finish()` on rank 0 can deadlock.
5. **Don't disable the sync process to "speed up" training.** Per-log overhead is ~2–3 ms (SQLite write). Disabling it removes crash-safety and gives you network latency on the critical path.
6. **Server-side filtering in MCP.** Prefer `list_runs(tags=...)` over `list_runs()` + client-side filter — projects can have 10K+ runs.
7. **Tags replace, not merge, on the server.** `add_tags` / `remove_tags` are convenience methods that compute the new full array and POST it. Don't race two clients updating the same run's tags.
8. **`pluto.init()` re-entry.** Calling `pluto.init()` again before `finish()` starts a *second* run, it does not return the existing one. To access the active run, use `pluto.ops[-1]` or the variable you assigned.

## Gotchas

- **`import mlop` works but is deprecated.** New code: `import pluto`. The `mlop` shim emits a `DeprecationWarning`.
- **`MLOP_API_TOKEN` vs `PLUTO_API_KEY`.** Old name still works (with a warning), but the new name is `PLUTO_API_KEY` (note: `KEY`, not `TOKEN`).
- **`report_to="none"` for HF Trainer.** Otherwise Transformers' built-in W&B/TensorBoard reporters fight with `PlutoCallback`.
- **Neptune compat is best-effort.** `neptune-scale` is no longer a dev dep; the compat layer in `pluto/compat/neptune.py` is for in-place migration only.
- **`pluto.watch` before the first forward pass.** Graph capture needs the model to have been called at least once; if you watch before any forward, the graph payload will be empty until step 1.
- **System metrics keys.** GPU metrics are namespaced `sys/gpu.{idx}.{device_name}.{metric}` — don't try to grep for plain `gpu.utilization`. Use `list_metric_names` if you're unsure.
- **MCP `query_metrics` returns a series, not a scalar.** For "what was the final val/acc?" use `get_statistics(metric=..., agg="last")` instead.

## Bundled Files

This skill currently ships only `SKILL.md`. Helper scripts (e.g. a `pluto_helpers.py` mirroring `wandb-helpers`) may be added later under `skills/pluto-primary/scripts/`. Until then, prefer the SDK and MCP recipes above directly.
