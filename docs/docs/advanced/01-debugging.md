---
sidebar_position: 1
---

# Debugging

## NCCL environment capture

Distributed jobs live and die by their NCCL configuration: one missing
`NCCL_SOCKET_IFNAME` and a job that should saturate InfiniBand quietly falls
back to TCP. Pluto records that configuration automatically on every run — no
code change required.

At `pluto.init()` the client collects the NCCL-relevant environment of the
calling process:

| Matched | Examples |
| --- | --- |
| `NCCL_*` | `NCCL_DEBUG`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_ALGO` |
| `TORCH_NCCL_*` | `TORCH_NCCL_ASYNC_ERROR_HANDLING`, `TORCH_NCCL_BLOCKING_WAIT` |
| `FI_*`, `OFI_*` | `FI_PROVIDER`, `FI_EFA_USE_DEVICE_RDMA` (libfabric / aws-ofi-nccl) |
| `UCX_*` | `UCX_TLS`, `UCX_NET_DEVICES` |
| Extras | `TORCH_DISTRIBUTED_DEBUG`, `TORCH_CPP_LOG_LEVEL`, `TORCH_SHOW_CPP_STACKTRACES`, `GLOO_SOCKET_IFNAME` |

Values whose key looks credential-bearing (`*_TOKEN`, `*_SECRET`,
`*_PASSWORD`, `*_API_KEY`, `*_ACCESS_KEY`, …) are replaced with `<redacted>`
before anything is stored or sent.

The capture lands in two places:

**1. Run metadata**, under `systemMetadata.nccl.nccl_env`, sent with the
run-create request. `systemMetadata.*` is an accepted filter field, so runs can
be selected by the settings they ran with:

```python
import pluto.query as pq

runs = pq.list_runs(
    "my-project",
    filters={"systemMetadata.nccl.nccl_env.NCCL_ALGO": "Tree"},
)
```

The same payload carries the NCCL versions Pluto could detect —
`nccl_pytorch` (from `torch.cuda.nccl.version()`) and `nccl_system` (from
`ncclGetVersion()` in `libnccl`) — alongside CUDA/cuDNN versions under
`systemMetadata.cuda` and adapter details under
`systemMetadata.infiniband`.

**2. The run's console log**, as a single line emitted at startup:

```
Operation: NCCL environment (4 vars): NCCL_DEBUG=INFO, NCCL_IB_DISABLE=0, ...
```

Every rank logs its own line, which is what makes it useful in multi-node
runs: only the rank that creates the run sends `systemMetadata`, so the log
line is where a misconfigured *worker* node shows up. Nothing is logged when
no NCCL-relevant variables are set.

:::note
The environment is read at `pluto.init()` time. Variables exported after
that — for example by a launcher that configures NCCL immediately before
`torch.distributed.init_process_group()` — will not appear. Set them before
initializing the run (or before launching the process) to have them recorded.
:::

Capture is skipped entirely for runs created with
`disable_system_metrics=True`, such as backfills through `pluto.migrate`,
where the importing host's environment says nothing about the run being
written.
