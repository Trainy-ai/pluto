# Neptune to mlop Migration Examples

This directory contains example scripts demonstrating how to migrate from Neptune to mlop using the dual-logging compatibility layer.

## Quick Start

See the main [Neptune Migration Guide](../NEPTUNE_MIGRATION.md) for comprehensive documentation.

### Basic Example - 3 Steps

1. **Add the compatibility import** at the top of your Neptune script:

```python
import mlop.compat.neptune
```

2. **Configure mlop credentials**:

```bash
export MLOP_PROJECT="your-project-name"
export MLOP_API_KEY="your-mlop-api-key"  # Optional, uses keyring if not set
```

3. **Run your script** - it will now log to both Neptune and mlop!

```bash
python your_training_script.py
```

## Example Files

### `neptune_migration_example.py`

Comprehensive example showing:
- Basic metric logging (original vs migrated)
- Image logging with automatic conversion
- Histogram logging
- How to run dual-logging tests

Run the example:

```bash
# Make sure credentials are configured
export NEPTUNE_PROJECT="workspace/project"
export NEPTUNE_API_TOKEN="your-neptune-token"
export MLOP_PROJECT="your-mlop-project"

# Run the example
python examples/neptune_migration_example.py
```

The script includes several example functions:
- `original_neptune_script()` - Shows original Neptune code
- `migrated_dual_logging_script()` - Same code with one import line added
- `image_logging_example()` - Demonstrates image logging
- `histogram_logging_example()` - Demonstrates histogram logging

## Key Features

- ✅ **Zero code changes** - just add one import line
- ✅ **Dual-logging** - logs to both Neptune and mlop simultaneously
- ✅ **Automatic fallback** - if mlop is unavailable, Neptune continues
- ✅ **Drop-in replacement** - works with existing Neptune Scale code

## What Gets Logged to Both Systems?

| Neptune API | mlop Equivalent | Status |
|------------|----------------|--------|
| `log_metrics()` | `mlop.log()` | ✅ Supported |
| `log_configs()` | `mlop.log()` | ✅ Supported |
| `log_files()` (images) | `mlop.Image` | ✅ Supported |
| `log_histograms()` | `mlop.Histogram` | ✅ Supported |
| `add_tags()` | `mlop.log()` (as metadata) | ✅ Supported |
| `assign_files()` | `mlop.Image` | ✅ Supported |

## Troubleshooting

For detailed troubleshooting, migration strategies, and advanced usage, see the [main migration guide](../NEPTUNE_MIGRATION.md).

**Common issues:**

- **Import order matters**: The compatibility import must come *before* importing Neptune classes
- **Credentials required**: Both `NEPTUNE_API_TOKEN` and `MLOP_PROJECT` must be set for dual-logging
- **Silent fallback**: If mlop credentials aren't set, only Neptune logging occurs (with INFO message)

## Testing Your Migration

Run the test script to validate your setup:

```bash
bash scripts/test_neptune_migration.sh
```

## Next Steps

1. Read the [full migration guide](../NEPTUNE_MIGRATION.md)
2. Try the examples in this directory
3. Add the compatibility import to one of your scripts
4. Monitor both Neptune and mlop dashboards during testing

## Support

For issues or questions:
- See [Neptune Migration Guide - Troubleshooting](../NEPTUNE_MIGRATION.md#troubleshooting)
- Check [Neptune Migration Guide - FAQ](../NEPTUNE_MIGRATION.md#faq)
- Review test suite: `tests/test_neptune_compat.py`
