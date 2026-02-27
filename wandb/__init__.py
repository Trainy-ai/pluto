"""wandb drop-in replacement backed by pluto.

This package allows ``import wandb`` to resolve to pluto's wandb
compatibility layer. Users swap ``wandb`` for ``pluto-ml`` in their
dependencies and keep all source code unchanged.

All public wandb API symbols (init, log, finish, watch, config, summary,
Image, Table, etc.) are re-exported from pluto.compat.wandb.
"""

import pluto.compat.wandb as _compat

# Re-export everything from the compat layer
from pluto.compat.wandb import *  # noqa: F401, F403
from pluto.compat.wandb import __all__  # noqa: F401

# Additional symbols that wandb exposes at top level
from wandb.apis import Api  # noqa: F401

# Remove the static copies of mutable state so __getattr__ proxies them
# dynamically from pluto.compat.wandb (where init/finish update them).
del run, config, summary  # noqa: F821

# ---------------------------------------------------------------------------
# Mutable module-level state (run, config, summary) must be proxied
# dynamically. A plain `from pluto.compat.wandb import run` copies the
# initial None value; subsequent mutations in pluto.compat.wandb are
# invisible. __getattr__ solves this by forwarding lookups at access time.
# ---------------------------------------------------------------------------

# Attributes that are mutable module-level state on pluto.compat.wandb
# and must always be read live from there.
_MUTABLE_ATTRS = frozenset({'run', 'config', 'summary'})


def __getattr__(name):
    if name in _MUTABLE_ATTRS:
        return getattr(_compat, name)
    raise AttributeError(f"module 'wandb' has no attribute {name!r}")
