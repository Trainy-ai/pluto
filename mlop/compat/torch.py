"""
Backward compatibility shim for mlop.compat.torch -> pluto.compat.torch migration.
This module is deprecated. Use `import pluto.compat.torch` instead.
"""

import warnings

warnings.warn(
    "The 'mlop.compat.torch' module is deprecated. "
    "Please use 'import pluto.compat.torch' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pluto.compat.torch import *  # noqa: E402, F401, F403
