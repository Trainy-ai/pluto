#!/usr/bin/env python3

import argparse
import sys

from .auth import login, logout


def main():
    parser = argparse.ArgumentParser(description="mlop")
    subparsers = parser.add_subparsers(dest="command", help="commands")

    p_login = subparsers.add_parser("login", help="login to mlop")
    p_logout = subparsers.add_parser("logout", help="logout from mlop")

    args = parser.parse_args()

    if args.command == "login":
        login()
    elif args.command == "logout":
        logout()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
