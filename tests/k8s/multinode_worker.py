#!/usr/bin/env python3
"""Worker script for k8s multinode E2E test.

This script runs on each node in a konduktor-launched distributed training job.
All nodes share the same PLUTO_RUN_ID and log metrics to the same Pluto run.

Environment variables (set by konduktor):
- PLUTO_RUN_ID: Shared run ID for all nodes
- PLUTO_API_TOKEN: API token for authentication
- MASTER_ADDR: Address of the master node
- NUM_NODES: Total number of nodes
- RANK: Global rank of this node (0, 1, 2, ...)
"""

import os
import sys
import time

sys.path.insert(0, '/workspace')

import pluto


def get_node_info():
    """Extract node information from environment variables."""
    rank = int(os.environ.get('RANK', '0'))
    num_nodes = int(os.environ.get('NUM_NODES', '1'))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')

    return {
        'rank': rank,
        'num_nodes': num_nodes,
        'master_addr': master_addr,
    }


def main():
    """Run the multinode E2E test worker."""
    node_info = get_node_info()
    rank = node_info['rank']

    print(f"[Node {rank}] Starting multinode E2E test")
    print(f"[Node {rank}] PLUTO_RUN_ID: {os.environ.get('PLUTO_RUN_ID', 'NOT SET')}")

    run_id = os.environ.get('PLUTO_RUN_ID')
    if not run_id:
        print(f"[Node {rank}] ERROR: PLUTO_RUN_ID not set!")
        sys.exit(1)

    if not os.environ.get('PLUTO_API_TOKEN'):
        print(f"[Node {rank}] ERROR: PLUTO_API_TOKEN not set!")
        sys.exit(1)

    # Initialize Pluto with shared run_id
    run = pluto.init(
        project=os.environ.get('PLUTO_PROJECT', 'testing-ci'),
        name=f'k8s-multinode-node{rank}',
        run_id=run_id,
        config={
            'test': 'k8s-multinode-e2e',
            'rank': rank,
            'num_nodes': node_info['num_nodes'],
        },
    )

    print(f"[Node {rank}] Pluto initialized: server_id={run.id}, resumed={run.resumed}")

    # Log metrics
    for step in range(5):
        run.log({
            f'loss/node{rank}': 1.0 - (step * 0.1) - (rank * 0.01),
            f'throughput/node{rank}': 1000 + rank * 100 + step * 10,
            'step': step,
        })
        time.sleep(0.2)

    run.finish()
    print(f"[Node {rank}] Done")
    return 0


if __name__ == '__main__':
    sys.exit(main())
