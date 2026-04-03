"""
Import hook that intercepts `import wandb` to enable dual-logging to Pluto.

This module is designed to be loaded via a .pth file at Python startup.
It registers a sys.meta_path finder that, when `import wandb` is executed,
loads the real wandb package and then monkey-patches it to dual-log to Pluto.

The hook is controlled by environment variables:
    - PLUTO_PROJECT: Pluto project name (required)
    - PLUTO_API_KEY: Pluto API key (required)
    - DISABLE_WANDB_LOGGING=true: Skip real wandb, Pluto-only mode

The hook is a no-op unless both PLUTO_PROJECT and PLUTO_API_KEY are set.
"""

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

_hook_installed = False


class _PlutoWandbFinder:
    """
    Meta path finder that intercepts `import wandb` to apply dual-logging patches.

    On first `import wandb`, this finder:
    1. Temporarily removes itself from sys.meta_path (to avoid recursion)
    2. Loads the real wandb package via normal import machinery
    3. Applies monkey-patches to wandb.init/wandb.log/etc. for dual-logging
    4. Re-inserts itself (for future imports, though wandb is now cached in sys.modules)
    """

    _patching = False

    def find_module(self, fullname, path=None):
        # Only intercept top-level `import wandb`, and only once
        if fullname == 'wandb' and not self._patching:
            return self
        return None

    def load_module(self, fullname):
        # If wandb is already in sys.modules, it's been loaded — don't patch again
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Prevent re-entrant calls
        self._patching = True
        try:
            # Remove ourselves so the real import machinery finds the real wandb
            sys.meta_path.remove(self)
            try:
                real_wandb = importlib.import_module('wandb')
            finally:
                # Always re-insert ourselves
                sys.meta_path.insert(0, self)

            # Apply the dual-logging patches
            try:
                from pluto.compat.wandb import apply_wandb_patches

                apply_wandb_patches(real_wandb)
                logger.info(
                    'pluto._wandb_hook: Successfully patched wandb for dual-logging'
                )
            except Exception as e:
                logger.warning(
                    f'pluto._wandb_hook: Failed to apply wandb patches: {e}. '
                    f'wandb will work normally without Pluto dual-logging.'
                )

            return real_wandb
        finally:
            self._patching = False


def install():
    """
    Register the wandb import hook on sys.meta_path.

    Only installs if PLUTO_WANDB=1 is set. Safe to call multiple times.
    """
    import os

    global _hook_installed

    if _hook_installed:
        return

    if not os.environ.get('PLUTO_PROJECT') or not os.environ.get('PLUTO_API_KEY'):
        return

    # Don't install if wandb is already imported (too late to intercept)
    if 'wandb' in sys.modules:
        logger.warning(
            'pluto._wandb_hook: wandb already imported before hook installation. '
            'Attempting to patch existing wandb module.'
        )
        try:
            from pluto.compat.wandb import apply_wandb_patches

            apply_wandb_patches(sys.modules['wandb'])
        except Exception as e:
            logger.warning(
                f'pluto._wandb_hook: Failed to patch already-imported wandb: {e}'
            )
        _hook_installed = True
        return

    # Install the finder
    finder = _PlutoWandbFinder()
    sys.meta_path.insert(0, finder)
    _hook_installed = True


def uninstall():
    """Remove the wandb import hook (for testing)."""
    global _hook_installed
    sys.meta_path[:] = [
        f for f in sys.meta_path if not isinstance(f, _PlutoWandbFinder)
    ]
    _hook_installed = False
