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
SITE=$(poetry run python -c "import site; print(site.getsitepackages()[0])")
DEST="$SITE/zzzz_pluto_wandb_hook.pth"

ln -sf "$SRC" "$DEST"
echo "Linked $DEST -> $SRC"
