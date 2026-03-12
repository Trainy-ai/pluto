# Skill: experiment-setup

Generate boilerplate code for a new Pluto experiment tracking setup.

## When to Use

Trigger when the user wants to:
- Set up a new ML experiment with tracking
- Create a basic training script with Pluto logging
- Initialize experiment tracking for a new project
- Scaffold a training script from scratch

## Instructions

Ask the user for:
1. **Project name** (required) - the Pluto project to log to
2. **Framework** (optional) - PyTorch, Lightning, Transformers, or plain Python (default: plain Python)
3. **What to track** (optional) - metrics, config, tags, files, system stats

Then generate a complete, runnable training script scaffold with Pluto integration using these patterns:

### Core Pattern

```python
import pluto

# 1. Define hyperparameters as a config dict
config = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32,
    # ... user-specific params
}

# 2. Initialize the run
run = pluto.init(
    project="<project-name>",
    name="<descriptive-run-name>",  # optional, auto-generated if omitted
    config=config,
    tags=["experiment", "v1"],      # optional
)

# 3. Training loop with logging
for epoch in range(config["epochs"]):
    # ... training code ...
    run.log({
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "epoch": epoch,
    })

# 4. Finish the run
run.finish()
```

### Key API Details

- `pluto.init()` returns an `Op` object — assign it to a variable (typically `run` or `op`)
- `run.log()` accepts a dict; keys become metric names, values must be numeric (int/float) or Pluto data types
- Use `/`-separated metric names for grouping (e.g., `"train/loss"`, `"val/loss"`)
- `config` can be a flat dict or nested dict — it's sent to the server for hyperparameter tracking
- Tags can be a single string or list of strings
- Always call `run.finish()` at the end to flush data and mark the run complete
- Use `try/finally` to ensure `finish()` is called even on errors

### Environment Setup

Remind the user they need:
- `pip install "pluto[full]"` (or `pip install pluto` for minimal)
- Authentication via `pluto login <token>` or `PLUTO_API_KEY` env var

### Error-Safe Pattern

For production scripts, wrap in try/finally:

```python
run = pluto.init(project="my-project", config=config)
try:
    # training loop
    for epoch in range(epochs):
        run.log({"loss": loss})
finally:
    run.finish()
```
