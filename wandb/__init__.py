"""wandb drop-in replacement backed by pluto.

This package allows ``import wandb`` to resolve to pluto's wandb
compatibility layer. Users swap ``wandb`` for ``pluto-ml`` in their
dependencies and keep all source code unchanged.

All public wandb API symbols (init, log, finish, watch, config, summary,
Image, Table, etc.) are re-exported from pluto.compat.wandb.

Modes (``PLUTO_WANDB_MODE`` env var)
-------------------------------------
``shim`` (default)
    Replaces wandb entirely.  All calls route through pluto.
    Real wandb is not used even if installed.

``dual``
    Logs to both real wandb AND pluto simultaneously.
    Requires real wandb to be installed.  Real wandb is the primary
    system; pluto mirrors scalar metrics, config, and lifecycle.

Set ``PLUTO_WANDB_SHIM=0`` to disable the shim entirely and let the
real wandb package load (only meaningful in shim mode).
"""

import importlib
import os
import sys
import warnings
from pathlib import Path


def _real_wandb_is_installed() -> bool:
    """Check whether the *real* wandb package is also installed."""
    try:
        import importlib.metadata

        dist = importlib.metadata.distribution('wandb')
        dist_name = dist.metadata['Name'] or ''
        if dist_name.lower().replace('-', '_') == 'pluto_ml':
            return False
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _load_real_wandb():
    """Import real wandb from site-packages, bypassing our shim.

    Returns the real wandb module object.  Raises ImportError if the
    real wandb package is not installed.
    """
    this_dir = str(Path(__file__).resolve().parent.parent)

    # Save our wandb modules from sys.modules
    our_modules = {}
    for key in list(sys.modules):
        if key == 'wandb' or key.startswith('wandb.'):
            our_modules[key] = sys.modules.pop(key)

    # Filter our package from sys.path
    original_path = sys.path[:]
    sys.path = [
        p for p in sys.path
        if os.path.realpath(p) != os.path.realpath(this_dir)
    ]

    try:
        real_wandb = importlib.import_module('wandb')
        # Snapshot all real wandb submodules that got loaded
        real_modules = {
            k: v for k, v in sys.modules.items()
            if k == 'wandb' or k.startswith('wandb.')
        }
        return real_wandb, real_modules
    except ImportError:
        raise ImportError(
            'PLUTO_WANDB_MODE=dual requires the real wandb package. '
            'Install it: pip install wandb'
        )
    finally:
        # Restore sys.path
        sys.path = original_path
        # Clear real wandb from sys.modules
        for key in list(sys.modules):
            if key == 'wandb' or key.startswith('wandb.'):
                del sys.modules[key]
        # Restore our modules
        sys.modules.update(our_modules)


def _setup_shim_mode():
    """Activate shim mode — replace wandb entirely with pluto."""
    env = os.environ.get('PLUTO_WANDB_SHIM', '').strip().lower()

    if env == '0':
        raise ImportError(
            'The pluto wandb shim is disabled (PLUTO_WANDB_SHIM=0). '
            'Unset this variable or install the real wandb package.'
        )

    if _real_wandb_is_installed():
        warnings.warn(
            'Both pluto-ml and wandb are installed. The pluto wandb '
            'shim is active and the real wandb package is being '
            'shadowed. Set PLUTO_WANDB_SHIM=0 to use real wandb.',
            UserWarning,
            stacklevel=3,
        )


def _setup_dual_mode():
    """Activate dual mode — log to both real wandb and pluto.

    Imports real wandb, re-exports everything from it, then installs
    dual-logging wrappers for init/log/finish/watch.
    """
    real_wandb, real_modules = _load_real_wandb()

    # Put real wandb's submodules back into sys.modules so that
    # real wandb's lazy internal imports (e.g. from wandb.sdk.wandb_settings
    # import Settings) resolve correctly instead of hitting our shim stubs.
    for key, mod in real_modules.items():
        sys.modules[key] = mod

    # Re-export everything from real wandb into this module
    this_module = sys.modules[__name__]
    for attr in dir(real_wandb):
        if not attr.startswith('_'):
            setattr(this_module, attr, getattr(real_wandb, attr))

    # Also copy __all__ if present
    if hasattr(real_wandb, '__all__'):
        this_module.__all__ = list(real_wandb.__all__)  # type: ignore[attr-defined]

    # Store reference so dual.py can use it
    this_module._real_wandb = real_wandb  # type: ignore[attr-defined]

    # Install dual-logging wrappers (overrides init/log/finish/watch)
    from pluto.compat.wandb.dual import setup_dual

    setup_dual(this_module, real_wandb)


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

_WANDB_MODE = os.environ.get('PLUTO_WANDB_MODE', 'shim').strip().lower()

if _WANDB_MODE == 'dual':
    _setup_dual_mode()
else:
    _setup_shim_mode()

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
