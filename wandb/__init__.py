"""wandb drop-in replacement backed by pluto.

This package allows ``import wandb`` to resolve to pluto's wandb
compatibility layer. Users swap ``wandb`` for ``pluto-ml`` in their
dependencies and keep all source code unchanged.

All public wandb API symbols (init, log, finish, watch, config, summary,
Image, Table, etc.) are re-exported from pluto.compat.wandb.
"""

# Re-export everything from the compat layer
from pluto.compat.wandb import *  # noqa: F401, F403
from pluto.compat.wandb import (  # noqa: F401
    __all__,
    config,
    run,
    summary,
)

# Additional symbols that wandb exposes at top level
from wandb.apis import Api  # noqa: F401

# wandb exposes a few submodule-level imports that users rely on.
# We handle the most common ones (wandb.sdk, wandb.data_types, etc.)
# via sub-packages defined alongside this __init__.py.
