# Skill: metrics-and-media

Generate code for logging custom metrics, media files, and structured data with Pluto.

## When to Use

Trigger when the user wants to:
- Log images, audio, video, or text artifacts to Pluto
- Create custom visualizations (graphs, histograms, tables)
- Log model artifacts or checkpoints
- Track non-standard metrics or structured data

## Instructions

### Numeric Metrics

```python
# Simple scalar metrics
run.log({
    "train/loss": 0.42,
    "train/accuracy": 0.89,
    "learning_rate": 1e-4,
})

# Use step parameter for explicit step control
run.log({"train/loss": 0.42}, step=100)
```

### Image Logging

```python
import pluto
import numpy as np

# From numpy array (H, W) or (H, W, C)
run.log({"sample": pluto.Image(data=np.random.rand(28, 28) * 255, caption="random noise")})

# From PIL image
from PIL import Image as PILImage
img = PILImage.open("output.png")
run.log({"result": pluto.Image(data=img, caption="model output")})

# From file path
run.log({"photo": pluto.Image(path="./generated.png", caption="generated sample")})

# From PyTorch tensor
run.log({"features": pluto.Image(data=feature_map.cpu().numpy(), caption="feature map")})
```

### Audio Logging

```python
import numpy as np

# From numpy array (1D waveform)
waveform = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))  # 440Hz tone
run.log({"audio/sample": pluto.Audio(data=waveform, sample_rate=22050, caption="sine wave")})

# From file path
run.log({"audio/speech": pluto.Audio(path="./output.wav", caption="synthesized speech")})
```

### Video Logging

```python
# From file path
run.log({"video/rollout": pluto.Video(path="./episode.mp4", caption="agent rollout")})

# From numpy array (T, H, W, C)
frames = np.random.randint(0, 255, (30, 64, 64, 3), dtype=np.uint8)
run.log({"video/generated": pluto.Video(data=frames, caption="generated video")})
```

### Text Logging

```python
# Log text content
run.log({"output/summary": pluto.Text(data="The model predicted class 'cat' with 97% confidence.")})

# From file
run.log({"output/report": pluto.Text(path="./eval_report.txt")})
```

### Artifact Logging (Model Checkpoints, Files)

```python
import torch

# Save and log a model checkpoint
torch.save(model.state_dict(), "model_best.pt")
run.log({
    "model/best": pluto.Artifact(
        path="model_best.pt",
        name="best-checkpoint",
    )
})

# Log any generic file
run.log({
    "data/predictions": pluto.Artifact(
        path="predictions.csv",
        name="test-predictions",
    )
})
```

### Histogram

```python
# Log weight distributions
for name, param in model.named_parameters():
    run.log({
        f"weights/{name}": pluto.Histogram(data=param.detach().cpu().numpy())
    })
```

### Table

```python
# Log tabular data
run.log({
    "eval/results": pluto.Table(
        columns=["input", "predicted", "actual", "correct"],
        data=[
            ["image_001.png", "cat", "cat", True],
            ["image_002.png", "dog", "cat", False],
            ["image_003.png", "bird", "bird", True],
        ]
    )
})

# From pandas DataFrame
import pandas as pd
df = pd.DataFrame({"metric": ["precision", "recall", "f1"], "value": [0.92, 0.88, 0.90]})
run.log({"eval/summary": pluto.Table(dataframe=df)})
```

### Graph (Network Visualization)

```python
# Log a computation graph or network topology
run.log({
    "architecture": pluto.Graph(
        nodes=[
            {"id": "input", "label": "Input (784)"},
            {"id": "hidden1", "label": "Linear (256)"},
            {"id": "relu", "label": "ReLU"},
            {"id": "output", "label": "Output (10)"},
        ],
        edges=[
            {"source": "input", "target": "hidden1"},
            {"source": "hidden1", "target": "relu"},
            {"source": "relu", "target": "output"},
        ],
    )
})
```

### Config Updates

```python
# Update config dynamically during training
run.update_config({
    "early_stop_epoch": epoch,
    "best_val_loss": best_loss,
})
```

### Alerts

```python
# Send an alert (e.g., training complete, anomaly detected)
run.alert(title="Training Complete", text=f"Final accuracy: {final_acc:.4f}")
```

### Key Guidelines

- All media types (Image, Audio, Video, Text, Artifact) accept either `data=` (in-memory) or `path=` (file path)
- Use `caption=` for descriptive labels on media
- Metric keys with `/` separators are grouped in the dashboard (e.g., `"train/loss"` and `"val/loss"`)
- `run.log()` can mix numeric metrics and media in one call
- Large files are uploaded asynchronously via the sync process — they won't block training
- Histograms accept numpy arrays; they auto-bin the data
- Tables accept either `columns`+`data` or a `dataframe` parameter
- Import all data types from `pluto` directly: `pluto.Image`, `pluto.Audio`, `pluto.Table`, etc.
