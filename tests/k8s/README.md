# Kubernetes Multinode E2E Tests with Konduktor

This directory contains end-to-end tests for Pluto's multinode distributed training support using [konduktor](https://pypi.org/project/konduktor-nightly/) and Kubernetes JobSet.

## Overview

These tests verify that multiple Kubernetes pods can log to the same Pluto run using a shared `PLUTO_RUN_ID`. This demonstrates the real-world distributed training workflow where:

1. A distributed job is launched via `konduktor launch`
2. Konduktor creates a JobSet with multiple replicas (nodes)
3. All nodes receive the same `PLUTO_RUN_ID` environment variable
4. Each node logs metrics concurrently to the same Pluto run
5. All metrics appear in a single unified run on the Pluto server

## Konduktor Job Format

The test uses konduktor's YAML job definition format:

```yaml
# multinode-job.yaml
name: pluto-multinode-e2e-test
num_nodes: 2

resources:
  cpus: 2
  memory: 4
  cloud: kubernetes
  image_id: pluto-multinode-test:latest

envs:
  PLUTO_PROJECT: testing-ci

run: |
  python /workspace/multinode_worker.py
```

Konduktor automatically injects environment variables:
- `$MASTER_ADDR` - Address of the master node (for torchrun rendezvous)
- `$NUM_NODES` - Total number of nodes in the job
- `$RANK` - Global rank of this node (0, 1, 2, ...)

## Prerequisites

- Docker
- [kind](https://kind.sigs.k8s.io/) (Kubernetes in Docker)
- kubectl
- konduktor CLI: `pip install konduktor-nightly`
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

| File | Purpose |
|------|---------|
| `multinode_worker.py` | Python script that runs on each node |
| `multinode-job.yaml` | Konduktor job definition (2 nodes) |
| `Dockerfile` | Container image for the test |
| `run_multinode_test.sh` | Test orchestration script |

## How It Works

1. **Setup**: Creates a kind cluster and installs JobSet controller
2. **Build**: Builds the test Docker image and loads it into kind
3. **Launch**: Uses `konduktor launch multinode-job.yaml` to start the job
4. **Execute**: Each pod runs `multinode_worker.py` which:
   - Reads `PLUTO_RUN_ID`, `RANK`, `NUM_NODES` from environment
   - Initializes Pluto with the shared run ID
   - Logs node-specific metrics (e.g., `loss/node0`, `loss/node1`)
   - Verifies all nodes get the same server run ID
5. **Verify**: Checks job completion status via `konduktor queue`
6. **Cleanup**: Deletes the kind cluster

## Integration with Pluto Distributed Logging

This test pattern mirrors the [Pluto distributed logging documentation](https://docs.trainy.ai/pluto/distributed-logging):

```python
import os
import pluto

# All nodes share the same run_id
run_id = os.environ.get('PLUTO_RUN_ID')
rank = int(os.environ.get('RANK', '0'))

run = pluto.init(
    project='my-project',
    name=f'training-node{rank}',
    run_id=run_id,  # Shared across all nodes
)

# Log metrics - all appear in the same run
run.log({'loss': 0.5, 'rank': rank})
```

## CI

The test runs automatically on:
- Push to `main` or `releases/**` branches
- Pull requests to `main` or `releases/**` branches
- Manual trigger via `workflow_dispatch`

Uses **ubicloud-standard-4** runner for better performance.

See `.github/workflows/k8s-multinode-test.yml` for the CI configuration.

## Troubleshooting

### Job not starting
Check konduktor queue status:
```bash
konduktor queue
```

### View job logs
```bash
konduktor logs pluto-multinode-e2e-test
```

### Debug locally
Keep the cluster running:
```bash
KEEP_CLUSTER=true ./tests/k8s/run_multinode_test.sh
kubectl get pods -A
kubectl logs -l jobset.sigs.k8s.io/jobset-name=pluto-multinode-e2e-test
```

## References

- [Konduktor Documentation](https://trainy.mintlify.app/overview)
- [Pluto Distributed Logging](https://docs.trainy.ai/pluto/distributed-logging)
- [Kubernetes JobSet](https://github.com/kubernetes-sigs/jobset)
