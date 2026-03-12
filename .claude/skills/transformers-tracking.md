# Skill: transformers-tracking

Generate code to integrate Pluto experiment tracking with HuggingFace Transformers.

## When to Use

Trigger when the user wants to:
- Add Pluto logging to a HuggingFace Transformers training script
- Track fine-tuning runs with Pluto
- Use Pluto callbacks with the HuggingFace Trainer

## Instructions

### Basic Transformers Integration

```python
import pluto
from pluto.compat.transformers import PlutoCallback
from transformers import Trainer, TrainingArguments

# Initialize Pluto
run = pluto.init(
    project="my-project",
    name="bert-finetune",
    config={
        "model": "bert-base-uncased",
        "dataset": "imdb",
        "lr": 2e-5,
        "epochs": 3,
        "batch_size": 16,
    },
    tags=["transformers", "fine-tuning"],
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    logging_steps=10,
)

# Create the Trainer with Pluto callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[PlutoCallback()],
)

# Train
trainer.train()

# Finish
run.finish()
```

### What Gets Tracked

The `PlutoCallback` automatically captures:
- Training loss at each logging step
- Evaluation metrics (loss, accuracy, etc.) at each eval step
- Learning rate schedule
- Training progress (epoch, step)

### Manual Metric Logging Alongside Trainer

For custom metrics not captured by the callback:

```python
# After evaluation, log additional metrics
eval_results = trainer.evaluate()
run.log({
    "eval/custom_f1": compute_f1(predictions, labels),
    "eval/custom_precision": compute_precision(predictions, labels),
})
```

### Key Guidelines

- Initialize `pluto.init()` BEFORE creating the `Trainer`
- Add `PlutoCallback()` to the `callbacks` list in `Trainer`
- The callback hooks into Trainer events automatically
- Call `run.finish()` after `trainer.train()` completes
- Log hyperparameters via `config` in `pluto.init()` rather than inside the training loop
- Use `try/finally` around training to ensure `run.finish()` is called
