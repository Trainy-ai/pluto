---
name: pytorch-tracking
description: Add Pluto experiment tracking to a PyTorch training script (model watching, metrics, images, checkpoints)
user-invocable: true
---

# Skill: pytorch-tracking

Generate code to integrate Pluto experiment tracking into a PyTorch training script.

## When to Use

Trigger when the user wants to:
- Add experiment tracking to an existing PyTorch training script
- Track gradients, parameters, or model architecture with Pluto
- Log training and validation metrics from PyTorch training
- Watch a PyTorch model with Pluto

## Instructions

Analyze the user's existing PyTorch code (or ask about their model setup) and generate the Pluto integration. Focus on minimal, non-intrusive changes.

### Model Watching

```python
import pluto

run = pluto.init(project="my-project", config={
    "lr": 1e-3,
    "batch_size": 64,
    "optimizer": "Adam",
    "architecture": "ResNet50",
})

model = MyModel()

# Watch model gradients, parameters, and architecture
run.watch(model)
```

`run.watch(model)` automatically tracks:
- Gradient distributions (histograms)
- Parameter statistics
- Model graph / architecture visualization

### Training Loop Integration

Insert `run.log()` calls at the end of each training step or epoch:

```python
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        # Per-step logging (optional, for high-frequency tracking)
        if batch_idx % log_interval == 0:
            run.log({
                "train/step_loss": loss.item(),
                "train/step": epoch * len(train_loader) + batch_idx,
            })

    # Per-epoch logging
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Validation
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    run.log({
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "epoch": epoch,
        "lr": optimizer.param_groups[0]["lr"],
    })

run.finish()
```

### Logging Images

```python
# Log sample predictions as images
run.log({
    "predictions": pluto.Image(
        data=img_tensor.cpu().numpy(),
        caption=f"pred={pred}, true={target}",
    )
})
```

### Logging Model Checkpoints

```python
# Save and log model checkpoint
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": val_loss,
}, "checkpoint.pt")

run.log({
    "model/checkpoint": pluto.Artifact(
        path="checkpoint.pt",
        name=f"checkpoint-epoch-{epoch}",
    )
})
```

### Key Guidelines

- Place `run.watch(model)` AFTER model creation but BEFORE the training loop
- Use `/`-separated keys to group metrics: `"train/loss"`, `"val/loss"`, `"lr/scheduler"`
- Log the learning rate if using a scheduler — it's valuable for debugging
- Tensors must be converted to Python scalars before logging: use `.item()` for scalar tensors
- For image logging, convert tensors to numpy arrays first
- Always call `run.finish()` — use `try/finally` for robustness
