"""
Backward compatibility shim for mlop.compat -> pluto.compat migration.
This module is deprecated. Use `import pluto.compat` instead.
"""

import warnings

warnings.warn(
    "The 'mlop.compat' module is deprecated. Please use 'import pluto.compat' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pluto.compat import *  # noqa: E402, F401, F403
