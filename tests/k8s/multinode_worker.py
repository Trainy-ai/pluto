#!/usr/bin/env python3
"""Worker script for k8s multinode E2E test.

This script runs on each node in a konduktor-launched distributed training job.
All nodes share the same PLUTO_RUN_ID and log metrics to the same Pluto run.

Environment variables (set by konduktor):
- PLUTO_RUN_ID: Shared run ID for all nodes
- PLUTO_API_TOKEN: API token for authentication
- MASTER_ADDR: Address of the master node (for torchrun rendezvous)
- NUM_NODES: Total number of nodes
- RANK: Global rank of this node (0, 1, 2, ...)
- SKYPILOT_NODE_RANK: Same as RANK (for compatibility)

Konduktor docs: https://trainy.mintlify.app/overview
"""

import os
import sys
import time

# Add pluto to path if running from source
sys.path.insert(0, '/workspace')

import pluto


def get_node_info():
    """Extract node information from konduktor environment variables."""
    # Konduktor provides these environment variables
    rank = int(os.environ.get('RANK', os.environ.get('SKYPILOT_NODE_RANK', '0')))
    num_nodes = int(os.environ.get('NUM_NODES', '1'))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')

    # Job identifiers
    job_name = os.environ.get('SKYPILOT_TASK_ID', os.environ.get('JOBSET_NAME', 'unknown'))

    return {
        'rank': rank,
        'num_nodes': num_nodes,
        'master_addr': master_addr,
        'job_name': job_name,
    }


def main():
    """Run the multinode E2E test worker."""
    node_info = get_node_info()

    print(f"[Node {node_info['rank']}] Starting multinode E2E test worker")
    print(f"[Node {node_info['rank']}] Node info: {node_info}")
    print(f"[Node {node_info['rank']}] PLUTO_RUN_ID: {os.environ.get('PLUTO_RUN_ID', 'NOT SET')}")

    # Verify PLUTO_RUN_ID is set
    run_id = os.environ.get('PLUTO_RUN_ID')
    if not run_id:
        print(f"[Node {node_info['rank']}] ERROR: PLUTO_RUN_ID not set!")
        sys.exit(1)

    # Verify PLUTO_API_TOKEN is set
    if not os.environ.get('PLUTO_API_TOKEN'):
        print(f"[Node {node_info['rank']}] ERROR: PLUTO_API_TOKEN not set!")
        sys.exit(1)

    # Initialize Pluto with shared run_id
    # All nodes use the same run_id, so they all log to the same run
    print(f"[Node {node_info['rank']}] Initializing Pluto with shared run_id...")
    run = pluto.init(
        project=os.environ.get('PLUTO_PROJECT', 'testing-ci'),
        name=f'k8s-multinode-node{node_info["rank"]}',
        run_id=run_id,
        config={
            'test': 'k8s-multinode-e2e',
            'rank': node_info['rank'],
            'num_nodes': node_info['num_nodes'],
            'master_addr': node_info['master_addr'],
            'job_name': node_info['job_name'],
        },
    )

    print(f"[Node {node_info['rank']}] Pluto initialized:")
    print(f"  - Server ID: {run.id}")
    print(f"  - Resumed: {run.resumed}")
    print(f"  - Run ID: {run_id}")

    # Log metrics from this node
    print(f"[Node {node_info['rank']}] Logging metrics...")
    for step in range(5):
        metrics = {
            f'loss/node{node_info["rank"]}': 1.0 - (step * 0.1) - (node_info['rank'] * 0.01),
            f'throughput/node{node_info["rank"]}': 1000 + node_info['rank'] * 100 + step * 10,
            'step': step,
        }
        run.log(metrics)
        print(f"[Node {node_info['rank']}] Step {step}: logged {metrics}")
        time.sleep(0.2)  # Small delay between logs

    # Write result to file for verification
    result_dir = '/tmp/pluto-test-results'
    os.makedirs(result_dir, exist_ok=True)
    result_file = f'{result_dir}/result_node{node_info["rank"]}.txt'
    with open(result_file, 'w') as f:
        f.write(f'server_id={run.id}\n')
        f.write(f'resumed={run.resumed}\n')
        f.write(f'run_id={run_id}\n')
        f.write(f'rank={node_info["rank"]}\n')
        f.write(f'num_nodes={node_info["num_nodes"]}\n')
    print(f"[Node {node_info['rank']}] Wrote results to {result_file}")

    # Finish the run
    print(f"[Node {node_info['rank']}] Finishing run...")
    run.finish()

    print(f"[Node {node_info['rank']}] SUCCESS - All metrics logged to run_id={run_id}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
