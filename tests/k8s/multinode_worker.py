#!/usr/bin/env python3
"""Worker script for k8s multinode E2E test.

This script runs on each node in a JobSet-based distributed training job.
All nodes share the same PLUTO_RUN_ID and log metrics to the same Pluto run.

Environment variables (set by JobSet/konduktor):
- PLUTO_RUN_ID: Shared run ID for all nodes
- PLUTO_API_TOKEN: API token for authentication
- JOB_COMPLETION_INDEX: Node index (0, 1, 2, ...)
- JOBSET_NAME: Name of the JobSet
- NUM_NODES: Total number of nodes (set by konduktor or manually)
- MASTER_ADDR: Address of the master node (for torchrun)
- RANK: Global rank of this node
"""

import os
import sys
import time

# Add pluto to path if running from source
sys.path.insert(0, '/workspace')

import pluto


def get_node_info():
    """Extract node information from environment variables."""
    # JobSet provides JOB_COMPLETION_INDEX
    node_index = int(os.environ.get('JOB_COMPLETION_INDEX', '0'))

    # konduktor style
    num_nodes = int(os.environ.get('NUM_NODES', os.environ.get('NNODES', '1')))
    rank = int(os.environ.get('RANK', os.environ.get('NODE_RANK', str(node_index))))

    # Get job identifiers
    jobset_name = os.environ.get('JOBSET_NAME', 'unknown')

    return {
        'node_index': node_index,
        'num_nodes': num_nodes,
        'rank': rank,
        'jobset_name': jobset_name,
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
    print(f"[Node {node_info['rank']}] Initializing Pluto...")
    run = pluto.init(
        project='testing-ci',
        name=f'k8s-multinode-node{node_info["rank"]}',
        run_id=run_id,
        config={
            'test': 'k8s-multinode-e2e',
            'node_index': node_info['node_index'],
            'num_nodes': node_info['num_nodes'],
            'rank': node_info['rank'],
            'jobset_name': node_info['jobset_name'],
        },
    )

    print(f"[Node {node_info['rank']}] Pluto initialized. Server ID: {run.id}, Resumed: {run.resumed}")

    # Log metrics from this node
    print(f"[Node {node_info['rank']}] Logging metrics...")
    for step in range(5):
        metrics = {
            f'loss/node{node_info["rank"]}': 1.0 - (step * 0.1) - (node_info['rank'] * 0.01),
            f'throughput/node{node_info["rank"]}': 1000 + node_info['rank'] * 100 + step * 10,
            'step': step,
        }
        run.log(metrics)
        print(f"[Node {node_info['rank']}] Step {step}: {metrics}")
        time.sleep(0.2)  # Small delay between logs

    # Write result to file for verification
    result_file = f'/tmp/result_node{node_info["rank"]}.txt'
    with open(result_file, 'w') as f:
        f.write(f'server_id={run.id}\n')
        f.write(f'resumed={run.resumed}\n')
        f.write(f'run_id={run_id}\n')
    print(f"[Node {node_info['rank']}] Wrote results to {result_file}")

    # Finish the run
    print(f"[Node {node_info['rank']}] Finishing run...")
    run.finish()

    print(f"[Node {node_info['rank']}] Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
