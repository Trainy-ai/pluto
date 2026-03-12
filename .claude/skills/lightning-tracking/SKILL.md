---
name: lightning-tracking
description: Integrate Pluto with PyTorch Lightning using the MLOPLogger
user-invocable: true
---

# Skill: lightning-tracking

Generate code to integrate Pluto experiment tracking with PyTorch Lightning.

## When to Use

Trigger when the user wants to:
- Add Pluto logging to a PyTorch Lightning training setup
- Use Pluto as a Lightning logger
- Track Lightning training metrics with Pluto

## Instructions

### Lightning Logger Integration

Pluto provides a native Lightning logger via `pluto.compat.lightning.MLOPLogger`:

```python
import pytorch_lightning as pl
from pluto.compat.lightning import MLOPLogger

# Create the Pluto logger
logger = MLOPLogger(
    project="my-project",
    name="lightning-experiment",      # optional
    config={"lr": 1e-3, "epochs": 10},  # optional
    tags=["lightning", "v1"],         # optional
)

# Use it with your Trainer
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    logger=logger,
)

trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
```

### What Gets Tracked Automatically

When using `MLOPLogger`, Lightning automatically logs:
- All metrics from `self.log()` and `self.log_dict()` in your LightningModule
- Training loss, validation loss, and any custom metrics
- Hyperparameters (if `log_hyperparams()` is called or config is passed)

### Custom Logging in LightningModule

```python
class MyModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # These metrics are automatically sent to Pluto
        self.log("train/loss", loss)
        self.log("train/accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("val/loss", loss)
        self.log("val/accuracy", accuracy)
```

### Logging Files via the Logger

```python
# Access the underlying Pluto experiment
logger.log_file("predictions", pluto.Image(data=img_array, caption="sample"))
```

### Key Properties

The `MLOPLogger` exposes these Lightning-compatible properties:
- `logger.name` — run name
- `logger.version` — run version/ID
- `logger.experiment` — the underlying `Op` object (for direct `run.log()` calls)
- `logger.save_dir` / `logger.log_dir` / `logger.root_dir` — artifact directories

### Multiple Loggers

You can combine Pluto with other loggers:

```python
from pytorch_lightning.loggers import TensorBoardLogger

tb_logger = TensorBoardLogger("logs/")
pluto_logger = MLOPLogger(project="my-project")

trainer = pl.Trainer(logger=[tb_logger, pluto_logger])
```

### Key Guidelines

- Import from `pluto.compat.lightning`, not from `pluto` directly
- The logger handles `finish()` automatically when training completes
- Pass hyperparameters via the `config` parameter to `MLOPLogger` for best results
- For direct Pluto API access, use `logger.experiment` to get the `Op` object
