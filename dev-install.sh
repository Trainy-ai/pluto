#!/usr/bin/env bash
# Deploy zzzz_pluto_wandb_hook.pth into a Python env's site-packages.
#
# poetry-core's editable install only emits a single pluto_ml.pth pointing
# at the source dir; it doesn't copy other .pth files declared in
# tool.poetry.packages. Without this symlink the wandb-import hook never
# fires in editable installs, so `import wandb` is not patched and dev
# behavior diverges from production wheels.
#
# Defaults to `poetry run python` so the standard poetry workflow works
# with no args. Override with PYTHON to target a different interpreter:
#
#   bash dev-install.sh                   # poetry venv
#   PYTHON=python bash dev-install.sh     # currently active env (conda etc.)
#   PYTHON=/path/to/python bash dev-install.sh
#
# Run once per env you test from. Safe to re-run; idempotent.
set -eo pipefail

PYTHON_CMD="${PYTHON:-poetry run python}"
SRC="$(cd "$(dirname "$0")" && pwd)/zzzz_pluto_wandb_hook.pth"
# sysconfig.get_path('purelib') is more reliable than site.getsitepackages()[0]
# — the latter's first entry isn't guaranteed to be the active install target
# on all platforms (e.g. user-site or sometimes Windows/conda layouts).
SITE=$($PYTHON_CMD -c "import sysconfig; print(sysconfig.get_path('purelib'))")
DEST="$SITE/zzzz_pluto_wandb_hook.pth"

ln -sf "$SRC" "$DEST"
echo "Linked $DEST -> $SRC (via: $PYTHON_CMD)"
