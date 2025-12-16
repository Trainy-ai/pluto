#!/usr/bin/env python3

import argparse
import sys

from . import __version__, _get_git_commit
from .auth import login, logout


def main():
    parser = argparse.ArgumentParser(description='mlop')
    parser.add_argument(
        '-v',
        '--version',
        action='store_true',
        help='show the installed mlop version',
    )
    parser.add_argument(
        '-c',
        '--commit',
        action='store_true',
        help='show the current git commit hash',
    )
    subparsers = parser.add_subparsers(dest='command', help='commands')

    p_login = subparsers.add_parser('login', help='login to mlop')
    p_login.add_argument('key', nargs='?', help='login key')
    subparsers.add_parser('logout', help='logout from mlop')

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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
