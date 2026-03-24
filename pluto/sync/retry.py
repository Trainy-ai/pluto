"""
CLI entry point for background sync retry.

Invoked as: python -m pluto.sync.retry --db-path ... --settings ... --timeout ...

This is used by `pluto sync --background` to spawn a detached retry process.
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description='Pluto sync retry worker')
    parser.add_argument('--db-path', required=True, help='Path to sync.db')
    parser.add_argument('--settings', required=True, help='JSON-encoded settings dict')
    parser.add_argument(
        '--timeout', type=float, default=60.0, help='Max sync time in seconds'
    )
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    try:
        settings_dict = json.loads(args.settings)
    except json.JSONDecodeError as e:
        print(f'Error: Invalid JSON in --settings: {e}', file=sys.stderr)
        sys.exit(1)

    from .process import retry_sync

    success = retry_sync(
        db_path=args.db_path,
        settings_dict=settings_dict,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
