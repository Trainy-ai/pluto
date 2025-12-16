#!/usr/bin/env bash
set -eo pipefail

RUFF_VERSION=$(poetry run ruff --version | head -n 1 | awk '{print $2}')
MYPY_VERSION=$(poetry run mypy --version | awk '{print $2}')

echo "ruff ver $RUFF_VERSION"
echo "mypy ver $MYPY_VERSION"

# Run ruff
echo 'mlop ruff:'
poetry run ruff check --fix mlop tests
poetry run ruff format mlop tests


# Run mypy
echo 'mlop mypy:'
poetry run mypy mlop