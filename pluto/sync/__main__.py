"""
CLI entry point for sync worker subprocess.

This module allows the sync process to be started via subprocess.Popen
without re-importing the user's __main__ module, avoiding the double-print
issue that occurs with multiprocessing.Process and spawn context.

Usage (internal):
    python -m pluto.sync --db-path /path/to/sync.db --settings '{"..."}'
        --parent-pid 12345
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Pluto sync worker process',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--db-path',
        required=True,
        help='Path to SQLite database for sync state',
    )
    parser.add_argument(
        '--settings',
        required=True,
        help='JSON-encoded settings dictionary',
    )
    parser.add_argument(
        '--parent-pid',
        type=int,
        required=True,
        help='PID of parent training process (for orphan detection)',
    )

    args = parser.parse_args()

    # Parse settings from JSON
    try:
        settings_dict = json.loads(args.settings)
    except json.JSONDecodeError as e:
        print(f'Error: Invalid JSON in --settings: {e}', file=sys.stderr)
        sys.exit(1)

    # Import here to avoid circular imports and ensure clean startup
    from .process import _sync_main

    # Run the sync worker
    _sync_main(args.db_path, settings_dict, args.parent_pid)


if __name__ == '__main__':
    main()
