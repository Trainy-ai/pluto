#!/usr/bin/env python3

import argparse
import sys

from . import __version__, _get_git_commit
from .auth import login, logout


def _get_auth_token(tag: str = 'pluto') -> str | None:
    """Resolve auth token from env var or keyring."""
    import os

    token = os.environ.get('PLUTO_API_KEY') or os.environ.get('MLOP_API_TOKEN')
    if token:
        return token

    try:
        import keyring

        try:
            assert sys.platform == 'darwin'
            return keyring.get_password(tag, tag)
        except (keyring.errors.NoKeyringError, AssertionError):
            from .util import import_lib

            keyring.set_keyring(import_lib('keyrings.alt.file').PlaintextKeyring())
            return keyring.get_password(tag, tag)
    except ImportError:
        return None


def _cmd_sync(args: argparse.Namespace) -> None:
    """Handle the `pluto sync` command."""
    import glob
    import os
    import subprocess

    from .sets import Settings

    # Find sync databases
    if args.path:
        if os.path.basename(args.path) != 'sync.db':
            print('Error: Path must point to a sync.db file.', file=sys.stderr)
            sys.exit(1)
        db_paths = [args.path]
        if not os.path.isfile(db_paths[0]):
            print(f'Error: {db_paths[0]} not found.', file=sys.stderr)
            sys.exit(1)
    else:
        base_dir = args.dir or os.getcwd()
        pattern = os.path.join(base_dir, '.pluto', '**', 'sync.db')
        db_paths = glob.glob(pattern, recursive=True)
        if not db_paths:
            print(f'No sync databases found under {base_dir}/.pluto/')
            return

    # Resolve auth
    auth = _get_auth_token()
    if not auth:
        print(
            'Error: No auth token found. Set PLUTO_API_KEY or run `pluto login` first.',
            file=sys.stderr,
        )
        sys.exit(1)

    # Build settings dict with URLs
    settings = Settings()
    settings.update_host()
    settings_dict = settings.to_dict()
    settings_dict['_auth'] = auth

    # Read run info from each database and populate settings
    import sqlite3

    dbs_with_pending = []
    for db_path in db_paths:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                'SELECT run_id, project, op_id FROM runs LIMIT 1'
            ).fetchone()
            if row is None:
                conn.close()
                continue

            # Check for pending records
            pending_statuses = (0, 1, 3)  # PENDING, IN_PROGRESS, FAILED
            pending = conn.execute(
                'SELECT COUNT(*) FROM sync_queue WHERE status IN (?, ?, ?)',
                pending_statuses,
            ).fetchone()[0]
            pending_files = conn.execute(
                'SELECT COUNT(*) FROM file_uploads WHERE status IN (?, ?, ?)',
                pending_statuses,
            ).fetchone()[0]
            conn.close()

            if pending + pending_files > 0:
                dbs_with_pending.append(
                    {
                        'db_path': db_path,
                        'run_id': row['run_id'],
                        'project': row['project'],
                        'op_id': row['op_id'],
                        'pending': pending,
                        'pending_files': pending_files,
                    }
                )
        except sqlite3.Error as e:
            print(f'Warning: Could not read {db_path}: {e}', file=sys.stderr)

    if not dbs_with_pending:
        print('No pending records found in any sync database.')
        return

    # Show summary
    for db_info in dbs_with_pending:
        total = db_info['pending'] + db_info['pending_files']
        print(
            f'  {db_info["project"]}/{db_info["run_id"]}: '
            f'{total} pending '
            f'({db_info["pending"]} records, '
            f'{db_info["pending_files"]} files)'
        )

    if args.background:
        # Non-blocking: spawn a subprocess for each database
        for db_info in dbs_with_pending:
            run_settings = dict(settings_dict)
            run_settings['project'] = db_info['project']
            run_settings['_op_id'] = db_info['op_id']
            run_settings['_op_name'] = db_info['run_id']

            import json

            cmd = [
                sys.executable,
                '-m',
                'pluto.sync.retry',
                '--db-path',
                db_info['db_path'],
                '--settings',
                json.dumps(run_settings),
                '--timeout',
                str(args.timeout),
            ]
            if args.verbose:
                cmd.append('--verbose')

            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=None if args.verbose else subprocess.DEVNULL,
                stderr=None if args.verbose else subprocess.DEVNULL,
            )
        print(f'Started {len(dbs_with_pending)} background sync process(es).')
    else:
        # Blocking: sync inline
        from .sync.process import retry_sync

        for db_info in dbs_with_pending:
            run_settings = dict(settings_dict)
            run_settings['project'] = db_info['project']
            run_settings['_op_id'] = db_info['op_id']
            run_settings['_op_name'] = db_info['run_id']

            print(f'\nSyncing {db_info["project"]}/{db_info["run_id"]}...')
            success = retry_sync(
                db_path=db_info['db_path'],
                settings_dict=run_settings,
                timeout=args.timeout,
                verbose=args.verbose,
            )
            if not success:
                print(f'  Some records failed to sync for {db_info["run_id"]}.')


def main():
    parser = argparse.ArgumentParser(description='pluto')
    parser.add_argument(
        '-v',
        '--version',
        action='store_true',
        help='show the installed pluto version',
    )
    parser.add_argument(
        '-c',
        '--commit',
        action='store_true',
        help='show the current git commit hash',
    )
    subparsers = parser.add_subparsers(dest='command', help='commands')

    p_login = subparsers.add_parser('login', help='login to pluto')
    p_login.add_argument('key', nargs='?', help='login key')
    subparsers.add_parser('logout', help='logout from pluto')

    p_sync = subparsers.add_parser(
        'sync', help='retry syncing pending/failed records from crashed runs'
    )
    p_sync.add_argument(
        '--path',
        help='path to a specific sync.db file (default: scan .pluto/ recursively)',
    )
    p_sync.add_argument(
        '--dir',
        help='base directory to scan for .pluto/ (default: current directory)',
    )
    p_sync.add_argument(
        '--background',
        action='store_true',
        help='run sync in background (non-blocking)',
    )
    p_sync.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='max seconds to spend syncing per run (default: 60)',
    )
    p_sync.add_argument(
        '--verbose',
        action='store_true',
        help='show detailed sync progress',
    )

    args = parser.parse_args()

    if args.version:
        print(__version__)
        return

    if args.commit:
        print(_get_git_commit())
        return

    if args.command == 'login':
        if args.key:
            login(settings={'_auth': args.key})
        else:
            login()
    elif args.command == 'logout':
        logout()
    elif args.command == 'sync':
        _cmd_sync(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
