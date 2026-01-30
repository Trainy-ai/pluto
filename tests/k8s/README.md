# Kubernetes Multinode E2E Tests

This directory contains end-to-end tests for Pluto's multinode distributed training support using Kubernetes and JobSet.

## Overview

These tests verify that multiple Kubernetes pods can log to the same Pluto run using a shared `PLUTO_RUN_ID`. This simulates real-world distributed training scenarios where:

1. Multiple nodes (pods) are launched via JobSet
2. All nodes share the same `PLUTO_RUN_ID` environment variable
3. Each node logs metrics concurrently to the same run
4. All metrics appear in a single unified run on the Pluto server

## Prerequisites

- Docker
- [kind](https://kind.sigs.k8s.io/) (Kubernetes in Docker)
- kubectl
- `PLUTO_API_TOKEN` environment variable

## Running Locally

```bash
# Set your API token
export PLUTO_API_TOKEN="your-token-here"

# Run the test
./tests/k8s/run_multinode_test.sh
```

### Options

- `PLUTO_RUN_ID`: Set a custom run ID (default: auto-generated)
- `KEEP_CLUSTER`: Set to `true` to keep the kind cluster after the test

```bash
# Keep the cluster for debugging
KEEP_CLUSTER=true ./tests/k8s/run_multinode_test.sh
```

## Files

- `multinode_worker.py` - Python script that runs on each node
- `Dockerfile` - Container image for the test
- `jobset-multinode-test.yaml` - JobSet manifest template
- `run_multinode_test.sh` - Test orchestration script

## How It Works

1. Creates a kind cluster
2. Installs the JobSet controller
3. Builds and loads the test image
4. Applies a JobSet with 2 replicas (simulating 2 nodes)
5. Each pod runs `multinode_worker.py` which:
   - Reads `PLUTO_RUN_ID` from environment
   - Initializes Pluto with the shared run ID
   - Logs node-specific metrics
   - Verifies all nodes get the same server run ID
6. Validates the JobSet completes successfully

## Integration with konduktor

This test pattern mirrors how [konduktor](https://github.com/Trainy-ai/konduktor) launches distributed training jobs. In a konduktor YAML:

```yaml
name: my-training
num_nodes: 2
run: |
  torchrun --nnodes=$NUM_NODES --node_rank=$RANK ...
```

The environment variables `PLUTO_RUN_ID`, `NUM_NODES`, `RANK`, etc. are automatically set, allowing all nodes to log to the same Pluto run.

## CI

The test runs automatically on:
- Push to `main` or `releases/**` branches
- Pull requests to `main` or `releases/**` branches
- Manual trigger via `workflow_dispatch`

See `.github/workflows/k8s-multinode-test.yml` for the CI configuration.
