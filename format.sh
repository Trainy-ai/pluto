#!/usr/bin/env bash
set -eo pipefail

RUFF_VERSION=$(poetry run ruff --version | head -n 1 | awk '{print $2}')
MYPY_VERSION=$(poetry run mypy --version | awk '{print $2}')

echo "ruff ver $RUFF_VERSION"
echo "mypy ver $MYPY_VERSION"

# Run ruff
echo 'pluto ruff:'
poetry run ruff check --fix pluto mlop tests
poetry run ruff format pluto mlop tests


# Run mypy
echo 'pluto mypy:'
poetry run mypy pluto