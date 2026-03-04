"""Standalone ``neptune_scale.Run`` replacement backed by Pluto.

This is a thin wrapper around :class:`~pluto.compat.neptune.NeptuneRunWrapper`
with Neptune forcibly disabled — all data goes to Pluto only.
"""

from __future__ import annotations

import os
from typing import Any


class Run:
    """Drop-in replacement for ``neptune_scale.Run`` that logs to Pluto.

    Delegates to :class:`~pluto.compat.neptune.NeptuneRunWrapper` with
    ``DISABLE_NEPTUNE_LOGGING=true`` so that no real Neptune calls are
    made.  Accepts the same constructor kwargs as ``neptune_scale.Run``.
    """

    def __new__(cls, **kwargs: Any) -> Any:
        # Import here to avoid circular import at module-load time.
        from pluto.compat.neptune import NeptuneRunWrapper

        # Temporarily force Neptune disabled so the wrapper skips all
        # real Neptune calls (the package isn't installed anyway).
        old = os.environ.get('DISABLE_NEPTUNE_LOGGING')
        os.environ['DISABLE_NEPTUNE_LOGGING'] = 'true'
        try:
            return NeptuneRunWrapper(**kwargs)
        finally:
            if old is None:
                os.environ.pop('DISABLE_NEPTUNE_LOGGING', None)
            else:
                os.environ['DISABLE_NEPTUNE_LOGGING'] = old
