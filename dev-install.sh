#!/usr/bin/env bash
# Deploy zzzz_pluto_wandb_hook.pth into the active venv's site-packages.
#
# poetry-core's editable install only emits a single pluto_ml.pth pointing
# at the source dir; it doesn't copy other .pth files declared in
# tool.poetry.packages. Without this symlink the wandb-import hook never
# fires in editable installs, so `import wandb` is not patched and dev
# behavior diverges from production wheels.
#
# Run this once after `poetry install`. Safe to re-run; idempotent.
set -eo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/zzzz_pluto_wandb_hook.pth"
# sysconfig.get_path('purelib') is more reliable than site.getsitepackages()[0]
# — the latter's first entry isn't guaranteed to be the active install target
# on all platforms (e.g. user-site or sometimes Windows/conda layouts).
SITE=$(poetry run python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
DEST="$SITE/zzzz_pluto_wandb_hook.pth"

ln -sf "$SRC" "$DEST"
echo "Linked $DEST -> $SRC"
