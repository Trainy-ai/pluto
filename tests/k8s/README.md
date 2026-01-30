# Kubernetes Multinode E2E Test

E2E test for Pluto's multinode distributed training using the [konduktor CLI](https://pypi.org/project/konduktor-nightly/).

## What It Tests

Verifies that multiple nodes in a konduktor job can log to the same Pluto run via shared `PLUTO_RUN_ID`.

## Running

```bash
export PLUTO_API_TOKEN="your-token"
./tests/k8s/run_multinode_test.sh
```

Options:
- `KEEP_CLUSTER=true` - Keep the kind cluster after test for debugging

## Files

| File | Purpose |
|------|---------|
| `multinode-job.yaml` | Konduktor job definition |
| `multinode_worker.py` | Worker script for each node |
| `run_multinode_test.sh` | Test runner |

## Konduktor Job

```yaml
name: pluto-multinode-e2e-test
num_nodes: 2

resources:
  cpus: 1
  memory: 2
  image_id: pluto-multinode-test:latest
  labels:
    kueue.x-k8s.io/queue-name: user-queue

run: |
  torchrun \
    --nnodes=$NUM_NODES \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    /workspace/multinode_worker.py
```

Launch with: `konduktor launch multinode-job.yaml`
