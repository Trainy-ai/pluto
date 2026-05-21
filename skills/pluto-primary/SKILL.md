---
name: pluto-primary
description: "Primary skill for working with Pluto (Trainy's experiment-tracking platform). Covers the documented Pluto Python SDK surface (init / log / finish, config, init-time tags) and the Pluto MCP server (querying runs, comparing experiments, fetching metrics/logs/files, leaderboards, tags/notes). Use this for: instrumenting a training script with Pluto tracking, inspecting or comparing runs, fetching metrics or logs, and answering broad 'what's going on with my experiments' questions."
---

# Pluto Primary Skill

This skill teaches agents how to (1) **instrument** ML training code with the Pluto Python SDK and (2) **query/analyze** existing runs through the Pluto MCP server. Reach for it whenever a task touches Pluto runs or experiment-tracking patterns.

Scope note: this skill covers only the **documented, supported** SDK surface (`init` / `log` / `finish`). The codebase contains additional integrations (Lightning logger, HF callback, `watch()`, media types, dynamic tag mutation) but those are **not part of the documented public API** — do not generate code that relies on them unless the user explicitly asks and accepts they may change.

## Environment Defaults

- **Install:** `pip install -Uq "pluto-ml[full]"` (minimal: `pip install pluto-ml`). The PyPI package is **`pluto-ml`**; the import name is **`pluto`**.
- **Auth:** `pluto login <token>` or `PLUTO_API_KEY` env var. Default project: `PLUTO_PROJECT`.
- **Server:** defaults to `https://pluto.trainy.ai`. Self-hosted: pass `host=...` to `pluto.init()` or set `PLUTO_URL_*` env vars. Self-host server: https://github.com/Trainy-ai/pluto-server
- **Backwards compat:** `import mlop` still works but is deprecated; `MLOP_*` env vars are honored with a warning. New code uses `pluto` / `PLUTO_*`.
- **Telemetry opt-out:** `export PLUTO_DISABLE_TELEMETRY=1` before importing `pluto`.
- **Python:** 3.9+.

## When To Use What

| Goal | Use |
|------|-----|
| Add tracking to a training script | Python SDK (`pluto.init` / `pluto.log` / `pluto.finish`) |
| Migrate from Neptune | Neptune compat module — https://docs.trainy.ai/pluto/neptune-migration |
| "What runs are in project X?" | MCP: `list_runs` |
| "Show me run R's metrics" | MCP: `get_run` → `query_metrics` |
| "Compare runs A, B, C" | MCP: `compare_runs` + `visualize_comparison` |
| "What's the best run by metric M?" | MCP: `get_leaderboard` |
| Stream stdout/stderr from a run | MCP: `query_logs` |
| Tag / untag / annotate an existing run | MCP: `add_tags`, `remove_tags`, `update_notes` |
| Shareable URL for a comparison | MCP: `generate_comparison_url` |

Rule of thumb: **logging from a *running* experiment → SDK.** **Inspecting or annotating *existing* runs → MCP.**

## Fast Recipes — SDK

### R1. Minimal run (the documented API)

```python
import pluto

pluto.init(project="hello-world")
pluto.log({"e": 2.718})
pluto.finish()
```

`init` / `log` / `finish` are module-level functions. `init()` also returns an `Op` you may keep a reference to (e.g. `run = pluto.init(...)`), but the module-level form above is the documented usage.

### R2. Training loop with config + tags

```python
import pluto

run = pluto.init(
    project="my-project",
    name="baseline-v1",                      # optional; auto-generated if omitted
    config={"lr": 1e-3, "epochs": 10, "batch_size": 32},
    tags=["baseline", "v1"],                  # set at init time; str or list[str]
)
try:
    for epoch in range(10):
        # ... train step ...
        pluto.log({"train/loss": loss, "val/acc": acc, "epoch": epoch})
finally:
    pluto.finish()
```

Key facts:
- `pluto.log()` takes a dict; values are numeric (int/float).
- Use `/`-separated keys (`train/loss`, `val/loss`) for grouping in the UI.
- Keep metric names stable across steps.
- Always call `pluto.finish()` — wrap in `try/finally` so it runs on error too.

### R3. Hyperparameter sweep skeleton

```python
import itertools, pluto

grid = {"lr": [1e-3, 3e-4], "wd": [0.0, 1e-4]}
for lr, wd in itertools.product(*grid.values()):
    pluto.init(
        project="sweep-x",
        name=f"lr{lr}-wd{wd}",
        config={"lr": lr, "wd": wd},
        tags=["sweep", "grid-v1"],
    )
    try:
        train(lr=lr, wd=wd)   # call pluto.log(...) inside
    finally:
        pluto.finish()
```

Use a shared init-time tag (e.g. `grid-v1`) so the MCP side can pull the whole sweep with `list_runs`.

## Fast Recipes — MCP

The Pluto MCP server exposes the tools below for read/analysis tasks. **Tool argument names below are illustrative** — confirm the exact parameters against the installed server's tool schema (an installed server may also prefix the tool names).

### M1. Find runs

```
list_projects()
list_runs(project="my-project", tags=["sweep", "grid-v1"])
```

### M2. Inspect a single run

```
get_run(...)
list_metric_names(...)
query_metrics(...)        # time-series for a metric
get_statistics(...)       # min/max/last/mean for a metric
```

### M3. Compare runs

```
compare_runs(...)
visualize_comparison(...) # returns an image
generate_comparison_url(...)  # shareable link
```

### M4. Leaderboard / logs / files

```
get_leaderboard(...)
query_logs(...)           # console stdout/stderr
get_files(...)            # logged files + URLs
```

### M5. Annotate existing runs

```
add_tags(...)
remove_tags(...)
update_notes(...)
```

## Critical Rules

1. **Always call `pluto.finish()`.** Without it the run never reaches a terminal state and buffered data may not flush. Use `try/finally`.
2. **Keep `log()` keys stable.** Don't rename a metric between steps — downstream tooling relies on consistent series.
3. **Numeric values for `log()`.** Stick to int/float for the documented surface.
4. **Server-side filtering in MCP.** Prefer `list_runs(tags=...)` over fetching all runs and filtering client-side — projects can have many runs.
5. **Don't invent SDK features.** Lightning logger, HF callback, `pluto.watch`, media types (`Image`/`Audio`/`Table`/...), and dynamic `add_tags`/`remove_tags` exist in the source but are **not documented/supported**. Don't emit code using them unless the user explicitly opts in.

## Gotchas

- **PyPI name is `pluto-ml`, import name is `pluto`.** `pip install pluto` installs the wrong package.
- **`import mlop` works but is deprecated.** New code: `import pluto`.
- **`PLUTO_API_KEY`, not `PLUTO_API_TOKEN`.** The old `MLOP_API_TOKEN` still works with a warning.
- **MCP `query_metrics` returns a series, not a scalar.** For "what was the final val/acc?" prefer `get_statistics` over grabbing the last point of the series yourself.

## Bundled Files

This skill currently ships only `SKILL.md`. Helper scripts may be added later under `skills/pluto-primary/scripts/`.
