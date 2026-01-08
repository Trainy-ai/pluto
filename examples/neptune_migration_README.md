# Neptune to mlop Migration Guide

## Overview

This document describes the Neptune-to-mlop dual-logging compatibility layer that enables a seamless migration from Neptune experiment tracking to mlop's Trakkur platform.

## Quick Start (TL;DR)

**For Engineers:**
```python
# 1. Add ONE line to your existing Neptune script
import mlop.compat.neptune

# 2. Your existing Neptune code works unchanged
from neptune_scale import Run
run = Run(experiment_name="my-exp")
run.log_metrics({"loss": 0.5}, step=0)
run.close()

# 3. Logs now go to BOTH Neptune AND mlop!
```

**Configuration:**
```bash
export MLOP_PROJECT="your-project-name"  # Required
export MLOP_API_KEY="your-api-key"       # Optional (uses keyring)
```

## Table of Contents

1. [Why This Exists](#why-this-exists)
2. [How It Works](#how-it-works)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Supported Features](#supported-features)
7. [Error Handling](#error-handling)
8. [Testing](#testing)
9. [Migration Timeline](#migration-timeline)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

## Why This Exists

**Problem**: Your team uses Neptune for experiment tracking, but needs to migrate to mlop/Trakkur.

**Challenge**: Updating hundreds of training scripts would:
- Take weeks of engineering time
- Risk breaking existing workflows
- Require coordinating changes across the team
- Cause downtime during the migration

**Solution**: The Neptune compatibility layer provides:
- ✅ **Zero code changes** (just add one import)
- ✅ **Dual-logging** to both Neptune and mlop
- ✅ **Gradual migration** over 2 months
- ✅ **Risk-free** - Neptune never fails due to mlop errors
- ✅ **Drop-in replacement** - works with existing scripts

## How It Works

### Architecture

The compatibility layer uses **monkeypatching** to intercept Neptune API calls:

```
Your Code                  Neptune Compatibility Layer              Backends
─────────────────         ───────────────────────────────         ──────────────

Run(...)          ──→     NeptuneRunWrapper.__init__()    ──→     Neptune API
                            │                                      ↓
                            └──→ Try: mlop.init()          ──→    mlop/Trakkur
                                 Catch: Continue silently           (optional)

run.log_metrics() ──→     NeptuneRunWrapper.log_metrics() ──→     Neptune API
                            │                                      (always works)
                            └──→ Try: mlop_run.log()       ──→    mlop/Trakkur
                                 Catch: Log warning                 (best effort)

run.close()       ──→     NeptuneRunWrapper.close()       ──→     Neptune API
                            │                                      ↓ THEN ↓
                            └──→ Try: mlop_run.finish()    ──→    mlop/Trakkur
```

### Safety Guarantees

1. **Neptune always works** - All mlop calls are wrapped in try-except
2. **No API changes** - Neptune's return values and exceptions are preserved
3. **Silent failures** - mlop errors are logged as warnings, never raised
4. **Lazy initialization** - mlop is only used if `MLOP_PROJECT` is set

## Installation

### Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **Neptune Scale**: `>= 0.30.0` (tested with `0.30.0`)
- **mlop package**: Latest version (`pip install trainy-mlop`)

**Compatibility Matrix**:

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.10 | ✅ Tested in CI |
| Python | 3.11 | ✅ Tested in CI |
| Python | 3.12 | ✅ Tested in CI |
| neptune-scale | 0.30.0 | ✅ Tested in CI |
| mlop | Latest | ✅ Compatible |

### Install mlop

```bash
pip install trainy-mlop
```

### Verify Installation

```bash
python -c "import mlop.compat.neptune; print('✓ Neptune compat ready')"
```

## Usage

### Basic Usage

**Step 1**: Add the import at the top of your script:

```python
import mlop.compat.neptune  # Enable dual-logging
```

**Step 2**: Use your existing Neptune code unchanged:

```python
from neptune_scale import Run

run = Run(experiment_name="my-training-run")
run.log_configs({"lr": 0.001, "batch_size": 32})

for epoch in range(100):
    run.log_metrics({"loss": 0.5, "acc": 0.9}, step=epoch)

run.close()
```

**That's it!** Your code now logs to both Neptune and mlop.

### Advanced Usage

#### With Images

```python
import mlop.compat.neptune
from neptune_scale import Run
from neptune_scale.types import File

run = Run(experiment_name="vision-model")

# Log images (auto-converted to mlop.Image)
image_file = File(source="sample.png", mime_type="image/png")
run.assign_files({"samples/image1": image_file})

run.close()
```

#### With Histograms

```python
import mlop.compat.neptune
import numpy as np
from neptune_scale import Run
from neptune_scale.types import Histogram

run = Run(experiment_name="histogram-logging")

# Log layer activations
activations = np.random.randn(1000)
counts, bin_edges = np.histogram(activations, bins=50)
hist = Histogram(bin_edges=bin_edges, counts=counts)

run.log_histograms({"layer1/activations": hist}, step=0)
run.close()
```

#### Context Manager

```python
import mlop.compat.neptune
from neptune_scale import Run

with Run(experiment_name="context-test") as run:
    run.log_metrics({"loss": 0.3}, step=0)
    # Both Neptune and mlop runs closed automatically
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description | Code Location |
|----------|----------|---------|-------------|---------------|
| `MLOP_PROJECT` | **Yes** | None | mlop project name (e.g., `"my-team-project"`). **Required for dual-logging**. Passed to `mlop.init(project=...)`. | `neptune.py:49` |
| `MLOP_API_KEY` | No | Keyring | API key for mlop authentication. Falls back to keyring if not set. Passed as `settings['_auth']` to mlop. | `neptune.py:56` |
| `MLOP_URL_APP` | No | `trakkur.trainy.ai` | Custom mlop app URL for self-hosted instances. Passed as `settings['url_app']`. | `neptune.py:60` |
| `MLOP_URL_API` | No | `trakkur-api.trainy.ai` | Custom mlop API URL for self-hosted instances. Passed as `settings['url_api']`. | `neptune.py:62` |
| `MLOP_URL_INGEST` | No | `trakkur-ingest.trainy.ai` | Custom mlop ingest URL for self-hosted instances. Passed as `settings['url_ingest']`. | `neptune.py:64` |
| `DISABLE_NEPTUNE_LOGGING` | No | `false` | **Post-sunset kill switch**. Set to `true`, `1`, or `yes` to disable all Neptune API calls. Only logs to mlop. | `neptune.py:198` |

#### How Environment Variables Work

**MLOP_PROJECT** (Required):
- Read at `mlop/compat/neptune.py:49`
- If not set: Logs INFO message, dual-logging disabled, Neptune-only mode
- If set: Passed to `mlop.init(project=...)` to create mlop run
- **This is the master switch for dual-logging**

**MLOP_API_KEY** (Optional):
- Read at `mlop/compat/neptune.py:56`
- If not set: Falls back to keyring (from `mlop login <token>`)
- If set: Passed as `settings['_auth']` to mlop
- Used in HTTP headers: `Authorization: Bearer {MLOP_API_KEY}`
- **Verified**: Works correctly (see `mlop/auth.py:26`, `mlop/iface.py:39`)

**MLOP_URL_* Variables** (Optional):
- All three URLs are read and passed to mlop settings
- Used for self-hosted mlop instances
- Default to production Trakkur URLs if not set
- **Verified**: All three work correctly (see `mlop/sets.py`)

**DISABLE_NEPTUNE_LOGGING** (Optional):
- Read at `mlop/compat/neptune.py:198-200`
- Accepts: `"true"`, `"1"`, `"yes"` (case-insensitive)
- When enabled: All Neptune API calls become no-ops
- **Use case**: Post-Neptune-sunset to avoid errors from dead Neptune API

### Configuration Methods

#### Method 1: Environment Variables (Recommended for CI/CD)

```bash
export MLOP_PROJECT="my-project"
export MLOP_API_KEY="mlop_api_xxxxx"

python train.py  # Dual-logging enabled
```

#### Method 2: Keyring (Recommended for Development)

```bash
# Store credentials once
mlop login <api-key>

# Set only the project
export MLOP_PROJECT="my-project"

python train.py  # Uses stored credentials
```

#### Method 3: No Configuration (Neptune Only)

```bash
# Don't set MLOP_PROJECT
python train.py  # Works with Neptune only, no dual-logging
```

## Disabling Neptune (Post-Sunset)

After Neptune's sunset, you can disable all Neptune API calls while keeping mlop logging active. This prevents errors from failed Neptune requests.

### Usage

Set the `DISABLE_NEPTUNE_LOGGING` environment variable:

```bash
export DISABLE_NEPTUNE_LOGGING=true  # or "1" or "yes"
```

### What Happens

- ✅ **mlop logging continues normally** - All metrics, configs, and files go to mlop
- ✅ **Neptune calls are no-ops** - No API requests to Neptune servers
- ✅ **No code changes needed** - Your existing Neptune code still works
- ✅ **No errors** - Silent fallback, one INFO log at startup

### Example

```python
import mlop.compat.neptune
from neptune_scale import Run

# With DISABLE_NEPTUNE_LOGGING=true, this only logs to mlop
run = Run(experiment_name='post-sunset-training')
run.log_metrics({'loss': 0.5}, step=0)  # → mlop only
run.close()
```

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| **Neptune sunset completed** | Set `DISABLE_NEPTUNE_LOGGING=true` |
| **During migration** | Keep it disabled (dual-logging) |
| **Testing mlop-only mode** | Temporarily enable to test post-sunset behavior |

### Notes

- Methods like `get_run_url()` return placeholder values when Neptune is disabled
- This is a kill switch for the transition period after Neptune sunset
- Eventually, you should migrate to the mlop API directly

## Supported Features

### Full Compatibility Matrix

| Neptune API | mlop Equivalent | Support Status | Notes |
|-------------|-----------------|----------------|-------|
| `Run(experiment_name, ...)` | `mlop.init(name, ...)` | ✅ Full | Experiment name → run name |
| `log_metrics(data, step)` | `run.log(data, step)` | ✅ Full | Step parameter passed through |
| `log_configs(data)` | `config={...}` in init | ✅ Full | Also logged as metrics |
| `assign_files(files)` | `run.log({k: mlop.Image(v)})` | ✅ Full | Auto type conversion |
| `log_files(files, step)` | `run.log({k: mlop.Image(v)})` | ✅ Full | Auto type conversion |
| `log_histograms(hists, step)` | `run.log({k: mlop.Histogram(v)})` | ✅ Full | Format conversion |
| `add_tags(tags)` | `run.add_tags(tags)` | ✅ Full | Native tags support |
| `remove_tags(tags)` | `run.remove_tags(tags)` | ✅ Full | Native tags support |
| `close()` | `run.finish()` | ✅ Full | Closes both runs |
| `terminate()` | `run.finish()` | ✅ Full | Terminates both runs |
| `wait_for_submission()` | N/A | ✅ Passthrough | Neptune-only method |
| `wait_for_processing()` | N/A | ✅ Passthrough | Neptune-only method |
| `get_run_url()` | N/A | ✅ Passthrough | Returns Neptune URL |
| `get_experiment_url()` | N/A | ✅ Passthrough | Returns Neptune URL |
| `log_string_series()` | N/A | ⚠️ Neptune-only | Not supported in mlop |

### Automatic Type Conversions

The compatibility layer automatically converts Neptune types:

| Neptune Type | mlop Type | Detection Method |
|--------------|-----------|------------------|
| `File(..., mime_type="image/*")` | `mlop.Image()` | MIME type |
| `File(..., mime_type="audio/*")` | `mlop.Audio()` | MIME type |
| `File(..., mime_type="video/*")` | `mlop.Video()` | MIME type |
| `File("*.png")` | `mlop.Image()` | File extension |
| `File("*.mp3")` | `mlop.Audio()` | File extension |
| `File("*.mp4")` | `mlop.Video()` | File extension |
| `File(...)` (other) | `mlop.Artifact()` | Default fallback |
| `Histogram(bin_edges, counts)` | `mlop.Histogram(data=(counts, bins))` | Data structure conversion |

### Tags Support

Tags are now fully supported in mlop through the native tags API. Both Neptune and mlop tags work identically:

```python
import mlop.compat.neptune
from neptune_scale import Run

# Initialize with tags
run = Run(experiment_name='my-training', tags=['experiment', 'baseline'])

# Add tags dynamically
run.add_tags(['production', 'v2'])  # Works in both Neptune and mlop

# Remove tags
run.remove_tags('baseline')  # Works in both Neptune and mlop

run.close()
```

**Implementation**:
- Neptune tags → mlop native tags (not config array)
- Uses `mlop_run.add_tags()` and `mlop_run.remove_tags()`
- Duplicate prevention automatic
- Tags automatically sync to server via tRPC `runs.updateTags`
- Initial tags sent during run creation, updates sent dynamically

**Direct mlop usage**:
```python
# When ready to migrate away from Neptune
import mlop

run = mlop.init(project='my-project', tags=['production', 'v2'])
run.add_tags('validated')
run.remove_tags('v2')
```

### Step Parameter Handling

**Important**: When using dual-logging, the `step` parameter you provide to Neptune methods is **automatically passed through to mlop**. This ensures charts and metrics stay aligned between both systems:

```python
import mlop.compat.neptune
from neptune_scale import Run

run = Run(experiment_name='my-training')

# Step values are preserved across both systems
run.log_metrics({"loss": 0.5}, step=0)
run.log_metrics({"loss": 0.4}, step=100)
run.log_metrics({"loss": 0.3}, step=200)

# Both Neptune AND mlop receive step=0, step=100, step=200
# Charts will be perfectly aligned ✅
run.close()
```

**Methods that pass step through to mlop:**
- ✅ `log_metrics(data, step)` - metric values at specific steps
- ✅ `log_files(files, step)` - files logged at specific steps
- ✅ `log_histograms(histograms, step)` - histogram data at specific steps

**Behavior details:**
- Neptune's explicit `step` parameter is passed directly to `mlop.log(data, step=step)`
- This ensures x-axis alignment in charts when viewing metrics in either system
- If you don't provide a step, mlop auto-increments (but you should always provide step with Neptune)
- Step values can be any integer (not required to be sequential)

**Direct mlop usage:**
```python
# When ready to migrate away from Neptune
import mlop

run = mlop.init(project='my-project')

# mlop also supports explicit step parameter
run.log({"loss": 0.5}, step=0)
run.log({"loss": 0.4}, step=100)

# Or auto-increment (if step=None)
run.log({"loss": 0.3})  # step auto-increments to 101
```

### Logging Configuration

To see mlop errors during development:

```python
import logging

# Configure mlop logger to show warnings
logging.getLogger('mlop').setLevel(logging.WARNING)

# See detailed debug info from Neptune compatibility layer
logging.getLogger('mlop.compat.neptune').setLevel(logging.DEBUG)

# Ensure logs are displayed (adds console handler if needed)
mlop_logger = logging.getLogger('mlop')
if not mlop_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(name)s: %(levelname)s - %(message)s'))
    mlop_logger.addHandler(handler)
```

**Note**: If you're using mlop for dual-logging, the logging system is automatically configured when you initialize a run. The above is only needed if you want to see debug messages before initialization or if mlop initialization fails.

## Testing

### Manual Testing

Use the provided test script:

```bash
./scripts/test_neptune_migration.sh
```

This validates:
- Prerequisites are installed
- Configuration is correct
- Compatibility layer loads
- Neptune API works
- Dual-logging is functional

### Automated Testing

Run the test suite:

```bash
# All Neptune compatibility tests
pytest tests/test_neptune_compat.py -v

# Specific test categories
pytest tests/test_neptune_compat.py::TestNeptuneCompatBasic -v
pytest tests/test_neptune_compat.py::TestNeptuneCompatErrorHandling -v
pytest tests/test_neptune_compat.py::TestNeptuneCompatDualLogging -v
```

### CI Integration

The Neptune compatibility tests run automatically on:
- Every PR to `main`
- Every push to `main`
- Daily at 8 AM UTC (during transition period)
- Manual workflow dispatch

See `.github/workflows/neptune-compat.yml` for details.

### Test Your Own Scripts

```bash
# Test with Neptune-only (no mlop config)
unset MLOP_PROJECT
python your_training_script.py  # Should work normally

# Test with dual-logging
export MLOP_PROJECT="test-project"
python your_training_script.py  # Should log to both

# Test with mlop unavailable (simulate service down)
export MLOP_PROJECT="test-project"
export MLOP_API_KEY="invalid-key-to-force-failure"
python your_training_script.py  # Should still work with Neptune
```

### Migration Checklist

**For Each Training Script:**
- [ ] Add `import mlop.compat.neptune` at the top
- [ ] Set `MLOP_PROJECT` environment variable
- [ ] Run script and verify logs appear in Neptune
- [ ] Verify logs also appear in mlop/Trakkur
- [ ] Test error case (invalid mlop config) to ensure Neptune still works
- [ ] Update CI/CD pipelines with mlop env vars
- [ ] Document the change for your team

## Troubleshooting

### Common Issues

#### Issue: "mlop not receiving logs"

**Symptoms**: Logs appear in Neptune but not mlop

**Debug steps**:
```bash
# 1. Check MLOP_PROJECT is set
echo $MLOP_PROJECT

# 2. Check credentials (MLOP_API_KEY or keyring)
if [ -n "$MLOP_API_KEY" ]; then
    echo "✓ MLOP_API_KEY is set"
else
    echo "⚠ MLOP_API_KEY not set, checking keyring..."
    python -c "import keyring; key = keyring.get_password('mlop', 'mlop'); print('✓ Keyring has credentials' if key else '✗ No keyring credentials')"
fi

# 3. Enable debug logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import mlop.compat.neptune
"

# 4. Check network connectivity
curl https://trakkur-api.trainy.ai/health
```

**Common causes**:
- `MLOP_PROJECT` not set
- Invalid API key
- Network/firewall blocking mlop endpoints
- mlop service temporarily down

#### Issue: "Step numbers don't match between Neptune and mlop"

**Expected behavior**: This is normal and acceptable during migration

**Explanation**:
- Neptune uses explicit `step` parameter
- mlop auto-increments steps
- Alignment is not guaranteed during dual-logging

**After migration**: Use mlop's step system consistently

#### Issue: "ImportError: No module named 'neptune_scale'"

**Solution**:
```bash
pip install neptune-scale
```

#### Issue: "TypeError: Unknown file type"

**Cause**: File type couldn't be auto-detected

**Solution**: Specify MIME type explicitly:
```python
file = File(source="myfile.dat", mime_type="image/png")
```

### Getting Help

- **Documentation**: `examples/neptune_migration_README.md`
- **Test Script**: `./scripts/test_neptune_migration.sh`
- **Examples**: `examples/neptune_migration_example.py`
- **Tests**: `tests/test_neptune_compat.py`
- **Issues**: File a GitHub issue with `[neptune-compat]` tag

## FAQ

**Q: Do I need to change my existing Neptune code?**
A: No! Just add `import mlop.compat.neptune` at the top.

**Q: What if the mlop service is down during training?**
A: Your training continues normally. Logs go to Neptune only.

**Q: Will this slow down my training?**
A: Minimal impact. mlop uses async logging and batching, similar to Neptune.

**Q: Can I disable dual-logging temporarily?**
A: Yes, unset `MLOP_PROJECT`: `unset MLOP_PROJECT`

**Q: Do I need neptune-scale installed?**
A: Yes, the compatibility layer wraps Neptune's API, so Neptune must be installed.

## See Also

- **Examples**: `examples/neptune_migration_example.py`
- **Detailed Guide**: `examples/neptune_migration_README.md`
- **Test Suite**: `tests/test_neptune_compat.py`
- **CI Workflow**: `.github/workflows/neptune-compat.yml`
- **Test Script**: `scripts/test_neptune_migration.sh`

---

**Questions or Issues?** File a GitHub issue

## Testing with Real Neptune

The test suite includes integration tests that validate the monkeypatch works with the actual Neptune client. These tests are skipped by default but can be enabled by setting Neptune credentials:

```bash
# Set Neptune credentials
export NEPTUNE_API_TOKEN="your-neptune-token"
export NEPTUNE_PROJECT="workspace/project"

# Run Neptune integration tests
pytest tests/test_neptune_compat.py::TestNeptuneRealBackend -v

# Test Neptune-only (no dual-logging)
pytest tests/test_neptune_compat.py::TestNeptuneRealBackend::test_real_neptune_without_mlop -v

# Test full dual-logging (requires both Neptune and mlop credentials)
export MLOP_PROJECT="your-mlop-project"
pytest tests/test_neptune_compat.py::TestNeptuneRealBackend::test_real_neptune_with_mlop_dual_logging -v
```

These tests will:
- Create real runs in your Neptune workspace
- Validate that the monkeypatch correctly forwards calls to Neptune
- Test dual-logging to both Neptune and mlop (if configured)
- Verify error resilience (Neptune works even if mlop fails)

**Note**: These tests create actual Neptune runs in your workspace. They are tagged with `real-neptune-test` and `dual-logging-test` for easy identification and cleanup.
