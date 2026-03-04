"""Standalone ``neptune_scale`` drop-in replacement backed by Pluto.

This package provides the same public API as the ``neptune_scale`` PyPI
package so that existing training scripts continue to work **even if
neptune-scale is removed from PyPI**.

Usage -- direct import::

    # Replace:
    from neptune_scale import Run
    from neptune_scale.types import File, Histogram

    # With:
    from pluto.compat.neptune_scale import Run
    from pluto.compat.neptune_scale.types import File, Histogram

    # The rest of your code stays exactly the same.

Usage -- automatic (via compat layer)::

    import pluto.compat.neptune  # patches or installs the shim

    # If neptune-scale is installed, it is monkeypatched to dual-log.
    # If neptune-scale is NOT installed, this shim is registered as
    # ``neptune_scale`` in sys.modules so ``from neptune_scale import Run``
    # transparently uses Pluto.

Configuration is read from the same environment variables as the dual-
logging compat layer:

- ``PLUTO_PROJECT`` (required)
- ``PLUTO_API_KEY`` (optional, falls back to keyring)
- ``PLUTO_URL_APP``, ``PLUTO_URL_API``, ``PLUTO_URL_INGEST`` (optional)
"""

from pluto.compat.neptune_scale import types
from pluto.compat.neptune_scale.run import Run

__all__ = ['Run', 'types']
