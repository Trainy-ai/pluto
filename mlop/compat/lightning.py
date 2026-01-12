"""
Backward compatibility shim for mlop.compat.lightning.
This module is deprecated. Use `import pluto.compat.lightning` instead.
"""

import warnings

warnings.warn(
    "The 'mlop.compat.lightning' module is deprecated. "
    "Please use 'import pluto.compat.lightning' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pluto.compat.lightning import *  # noqa: E402, F401, F403
