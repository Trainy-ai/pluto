# Kubernetes Multinode E2E Tests with Konduktor

This directory contains end-to-end tests for Pluto's multinode distributed training support using [konduktor](https://pypi.org/project/konduktor-nightly/) CLI with Kubernetes JobSet and Kueue.

## Overview

These tests verify that multiple Kubernetes pods can log to the same Pluto run using a shared `PLUTO_RUN_ID`. This demonstrates the real-world distributed training workflow as used in konduktor CI:

1. A distributed job is launched via `konduktor launch`
2. Konduktor creates a JobSet with multiple replicas (nodes)
3. Each node runs `torchrun` for distributed coordination
4. All nodes receive the same `PLUTO_RUN_ID` environment variable
5. Each node logs metrics concurrently to the same Pluto run
6. All metrics appear in a single unified run on the Pluto server

## Konduktor Job Format

The test uses konduktor's YAML job definition format (based on konduktor CI):

```yaml
# multinode-job.yaml
name: pluto-multinode-e2e-test
num_nodes: 2

resources:
  cpus: 1
  memory: 2
  image_id: pluto-multinode-test:latest
  labels:
    kueue.x-k8s.io/queue-name: user-queue
    maxRunDurationSeconds: "3200"

run: |
  set -e
  torchrun \
    --rdzv_id=123 \
    --nnodes=$NUM_NODES \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=1234 \
    --node_rank=$RANK \
    --rdzv_conf=$RDZV_CONF \
    /workspace/multinode_worker.py
```

Konduktor automatically injects environment variables:
- `$MASTER_ADDR` - Address of the master node (for torchrun rendezvous)
- `$NUM_NODES` - Total number of nodes in the job
- `$RANK` - Global rank of this node (0, 1, 2, ...)
- `$RDZV_CONF` - Rendezvous configuration for torchrun

## Prerequisites

- Docker
- [kind](https://kind.sigs.k8s.io/) (Kubernetes in Docker)
- kubectl
- konduktor CLI: `pip install konduktor-nightly`
- `PLUTO_API_TOKEN` environment variable

## Required Kubernetes Components

The test script automatically installs:
- **JobSet controller** - For managing multi-pod jobs
- **Kueue** - For queue-based job scheduling
- **LocalQueue** (`user-queue`) - Required by konduktor job labels

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
| `multinode-job.yaml` | Konduktor job definition (2 nodes with torchrun) |
| `kueue-config.yaml` | Kueue ResourceFlavor, ClusterQueue, and LocalQueue |
| `Dockerfile` | Container image with PyTorch + Pluto |
| `run_multinode_test.sh` | Test orchestration script |

## How It Works

1. **Setup**: Creates a kind cluster
2. **Install Controllers**: Installs JobSet and Kueue controllers
3. **Create Queues**: Creates ClusterQueue and LocalQueue (`user-queue`)
4. **Build**: Builds the test Docker image and loads it into kind
5. **Launch**: Uses `konduktor launch multinode-job.yaml` to start the job
6. **Execute**: Each pod runs `torchrun` which executes `multinode_worker.py`:
   - Reads `PLUTO_RUN_ID`, `RANK`, `NUM_NODES` from environment
   - Initializes Pluto with the shared run ID
   - Logs node-specific metrics (e.g., `loss/node0`, `loss/node1`)
   - Verifies all nodes get the same server run ID
7. **Verify**: Checks job completion status via `konduktor queue`
8. **Cleanup**: Deletes the kind cluster

## Integration with Pluto Distributed Logging

This test pattern mirrors the [Pluto distributed logging documentation](https://docs.trainy.ai/pluto/distributed-logging):

```python
import os
import pluto

# All nodes share the same run_id (set by konduktor)
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

Check Kueue workload status:
```bash
kubectl get workloads -A
kubectl describe clusterqueue cluster-queue
```

### View job logs
```bash
konduktor logs pluto-multinode-e2e-test
# or
kubectl logs -l jobset.sigs.k8s.io/jobset-name=pluto-multinode-e2e-test --all-containers=true
```

### Debug locally
Keep the cluster running:
```bash
KEEP_CLUSTER=true ./tests/k8s/run_multinode_test.sh
kubectl get pods -A
kubectl get jobsets
kubectl get workloads
```

## References

- [Konduktor Documentation](https://trainy.mintlify.app/overview)
- [Konduktor PyPI](https://pypi.org/project/konduktor-nightly/)
- [Pluto Distributed Logging](https://docs.trainy.ai/pluto/distributed-logging)
- [Kubernetes JobSet](https://github.com/kubernetes-sigs/jobset)
- [Kubernetes Kueue](https://github.com/kubernetes-sigs/kueue)
