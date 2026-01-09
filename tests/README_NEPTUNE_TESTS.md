# Neptune Compatibility Tests

This directory contains comprehensive tests for the Neptune-to-mlop compatibility layer.

## Test Structure

### Unit Tests (with Mocks)
These tests use `MockNeptuneRun` to validate the wrapper logic without requiring real backends:

- `TestNeptuneCompatBasic` - Basic Neptune API preservation
- `TestNeptuneCompatDualLogging` - Dual-logging logic (mocked backends)
- `TestNeptuneCompatErrorHandling` - Error handling and resilience
- `TestNeptuneCompatFileConversion` - Type conversion logic
- `TestNeptuneCompatFallbackBehavior` - Configuration fallback
- `TestNeptuneCompatAPIForwarding` - Method forwarding

**Run unit tests:**
```bash
pytest tests/test_neptune_compat.py -v -m "not neptune"
```

### Integration Tests (Real Neptune)
These tests use the **actual Neptune client** to validate the monkeypatch works in production:

- `TestNeptuneRealBackend` - Real Neptune integration tests

**Requirements:**
- Neptune API token: `NEPTUNE_API_TOKEN`
- Neptune project: `NEPTUNE_PROJECT` (format: `workspace/project`)

**Run integration tests:**
```bash
export NEPTUNE_API_TOKEN="your-token-here"
export NEPTUNE_PROJECT="asai/test"

# Run all real Neptune tests
pytest tests/test_neptune_compat.py::TestNeptuneRealBackend -v

# Run specific tests
pytest tests/test_neptune_compat.py::TestNeptuneRealBackend::test_real_neptune_without_mlop -v
```

### Full Dual-Logging Tests (Both Backends)
Test with **both real Neptune and real mlop**:

**Requirements:**
- Neptune credentials (as above)
- mlop credentials: `MLOP_PROJECT` and mlop auth (keyring or `MLOP_API_KEY`)

**Run full integration:**
```bash
export NEPTUNE_API_TOKEN="your-neptune-token"
export NEPTUNE_PROJECT="asai/test"
export MLOP_PROJECT="testing-ci"

pytest tests/test_neptune_compat.py::TestNeptuneRealBackend::test_real_neptune_with_mlop_dual_logging -v
```

## What Gets Tested

### Mock Tests (No Credentials Required)
✅ Neptune API calls work unchanged  
✅ Dual-logging logic (when enabled)  
✅ Error handling (mlop failures don't break Neptune)  
✅ File type conversions  
✅ Configuration fallback  
✅ Method forwarding  

### Real Neptune Tests (Neptune Credentials Required)
✅ Monkeypatch works with actual Neptune client  
✅ Real Neptune runs are created successfully  
✅ Context managers work correctly  
✅ Neptune continues working when mlop fails mid-run  

### Full Integration (Both Credentials Required)
✅ Data logs to both Neptune and mlop simultaneously  
✅ Both systems receive the same metrics, configs, tags  
✅ URLs are accessible from both systems  

## CI Integration

In CI, set environment variables as secrets:

```yaml
env:
  NEPTUNE_API_TOKEN: ${{ secrets.NEPTUNE_API_TOKEN }}
  NEPTUNE_PROJECT: "asai/test"
  MLOP_PROJECT: "testing-ci"
  CI: true
```

The GitHub Actions workflow automatically runs:
- Mock tests (always)
- Real Neptune tests (on push/schedule/manual trigger with NEPTUNE_API_TOKEN secret)
- Full dual-logging tests (when both Neptune and mlop credentials are available)

## Notes

- Real Neptune tests create actual runs in your Neptune workspace
- Runs are tagged with `real-neptune-test` or `dual-logging-test` for easy identification
- Consider creating a dedicated Neptune project for testing to avoid cluttering production
- Tests are designed to be safe - they don't delete or modify existing runs

## Troubleshooting

**Tests skip with "Requires NEPTUNE_API_TOKEN":**
- Set the environment variables shown above
- Ensure your Neptune token has access to the specified project

**Tests fail with "NeptuneProjectNotProvided":**
- Check `NEPTUNE_PROJECT` format is `workspace/project-name`
- Verify the project exists in your Neptune workspace

**Dual-logging tests skip:**
- Ensure both Neptune AND mlop credentials are set
- Verify mlop authentication: `mlop auth status`
