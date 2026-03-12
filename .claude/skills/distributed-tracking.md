# Skill: distributed-tracking

Generate code for experiment tracking in distributed (DDP/multi-node) training setups.

## When to Use

Trigger when the user wants to:
- Add Pluto tracking to a DistributedDataParallel (DDP) training script
- Track experiments across multiple GPUs or nodes
- Handle multi-rank logging with Pluto
- Set up experiment tracking for SLURM / torchrun jobs

## Instructions

### Key DDP Concepts

In distributed training, multiple processes (ranks) run the same script. Pluto handles this by:
1. Using a shared `run_id` so all ranks log to the same run
2. Using `wait=False` on finish to avoid DDP deadlocks
3. Detecting distributed environments automatically

### Basic DDP Pattern

```python
import os
import pluto
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = setup_distributed()

    # Use a shared run_id so all ranks log to the same run.
    # Set PLUTO_RUN_ID in your launch script, or generate one and broadcast.
    run_id = os.environ.get("PLUTO_RUN_ID", pluto.generate_run_id())

    run = pluto.init(
        project="distributed-training",
        name="ddp-experiment",
        run_id=run_id,
        config={
            "world_size": world_size,
            "model": "ResNet50",
            "lr": 0.001 * world_size,  # linear scaling rule
        },
        tags=["ddp", f"gpus-{world_size}"],
    )

    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * world_size)

    for epoch in range(num_epochs):
        # ... training loop ...

        # Log from all ranks with rank-prefixed metrics
        run.log({
            f"train/loss_rank{rank}": loss.item(),
            f"train/throughput_rank{rank}": samples_per_sec,
        })

        # Log aggregate metrics from rank 0 only
        if rank == 0:
            run.log({
                "train/avg_loss": avg_loss,
                "train/epoch": epoch,
            })

    # finish() automatically uses wait=False in distributed environments
    # to avoid blocking collective operations
    run.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### Launch Script

```bash
# Single-node, multi-GPU
export PLUTO_RUN_ID="ddp-$(date +%Y%m%d-%H%M%S)"
torchrun --standalone --nproc-per-node=4 train.py

# Multi-node (e.g., 2 nodes x 4 GPUs)
export PLUTO_RUN_ID="ddp-$(date +%Y%m%d-%H%M%S)"
torchrun \
    --nnodes=2 \
    --nproc-per-node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py
```

### SLURM Integration

```bash
#!/bin/bash
#SBATCH --job-name=ddp-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

export PLUTO_RUN_ID="slurm-${SLURM_JOB_ID}"
export PLUTO_API_KEY="<your-api-key>"

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    train.py
```

### Key Guidelines

- **Always set `run_id`** in DDP so all ranks log to the same run
- Use `PLUTO_RUN_ID` environment variable or `pluto.generate_run_id()` to create a shared ID
- `run.finish()` automatically detects DDP and uses non-blocking shutdown
- Pluto detects distributed via `torch.distributed.is_initialized()`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, and SLURM env vars
- The `if __name__ == "__main__":` guard is NOT required (Pluto uses subprocess.Popen, not multiprocessing)
- For rank-specific metrics, prefix with `rank{N}/`; for aggregated metrics, log only from rank 0
- The `resume` parameter allows re-attaching to an existing run (useful for preemption recovery)
