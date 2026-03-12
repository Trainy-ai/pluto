---
name: hyperparam-sweep
description: Generate code for hyperparameter grid/random search with Pluto-tracked runs and tag organization
user-invocable: true
---

# Skill: hyperparam-sweep

Generate code for running hyperparameter sweeps with Pluto experiment tracking.

## When to Use

Trigger when the user wants to:
- Run a hyperparameter search with experiment tracking
- Compare multiple training runs with different configs
- Set up grid search or random search with Pluto logging
- Organize related experiments with tags

## Instructions

### Grid Search Pattern

```python
import itertools
import pluto

# Define the search space
search_space = {
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "batch_size": [32, 64, 128],
    "dropout": [0.1, 0.3, 0.5],
}

# Generate all combinations
keys = search_space.keys()
combinations = list(itertools.product(*search_space.values()))

for combo in combinations:
    config = dict(zip(keys, combo))

    run = pluto.init(
        project="my-project",
        config=config,
        tags=["sweep", "grid-search"],
    )

    try:
        # Build model with these hyperparameters
        model = build_model(
            lr=config["learning_rate"],
            dropout=config["dropout"],
        )

        # Train
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, config)
            val_loss, val_acc = evaluate(model, val_loader)

            run.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "epoch": epoch,
            })

    finally:
        run.finish()
```

### Random Search Pattern

```python
import random
import pluto

num_trials = 20

for trial in range(num_trials):
    config = {
        "learning_rate": 10 ** random.uniform(-5, -2),  # log-uniform
        "batch_size": random.choice([16, 32, 64, 128]),
        "dropout": random.uniform(0.0, 0.5),
        "hidden_dim": random.choice([128, 256, 512]),
        "weight_decay": 10 ** random.uniform(-6, -3),
    }

    run = pluto.init(
        project="my-project",
        config=config,
        tags=["sweep", "random-search", f"trial-{trial}"],
    )

    try:
        best_val_acc = train_and_evaluate(config)
        run.log({"best/val_accuracy": best_val_acc})
    finally:
        run.finish()
```

### Using Tags for Organization

Tags are powerful for filtering and comparing sweep runs:

```python
run = pluto.init(
    project="my-project",
    config=config,
    tags=[
        "sweep",                           # all sweep runs
        "sweep-2024-03",                   # this sweep batch
        f"lr-{config['learning_rate']}",   # quick filtering
        "baseline" if is_baseline else "experiment",
    ],
)

# Add tags dynamically based on results
if val_acc > 0.95:
    run.add_tags("high-accuracy")

if val_loss < best_known_loss:
    run.add_tags("new-best")
    run.remove_tags("experiment")
    run.add_tags("champion")
```

### Querying Past Sweep Results

```python
import pluto

# Use the query API to find past runs
client = pluto.query.Client()
runs = client.list_runs(project="my-project")

# Filter for sweep runs, compare configs and results
for run_info in runs:
    print(f"Run: {run_info['name']}, Config: {run_info.get('config')}")
```

### Key Guidelines

- Each `pluto.init()` / `run.finish()` pair creates a separate tracked run
- Use `config` dict to record all hyperparameters — they're searchable on the dashboard
- Use `tags` to group related runs within a sweep for filtering
- Always use `try/finally` to ensure `run.finish()` is called even if training fails
- For large sweeps, consider parallel execution with shared `PLUTO_API_KEY`
- Log a summary "best" metric at the end of each run for easy comparison
- Use `run.add_tags()` / `run.remove_tags()` to dynamically tag runs based on results
