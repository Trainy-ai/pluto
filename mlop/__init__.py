"""
Backward compatibility shim for mlop -> pluto migration.
This module is deprecated. Use `import pluto` instead.
"""

import warnings

warnings.warn(
    "The 'mlop' package is deprecated and will be removed in a future release. "
    "Please use 'import pluto' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from pluto
from pluto import *  # noqa: E402, F401, F403
from pluto import __all__, __version__  # noqa: E402, F401
