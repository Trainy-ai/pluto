"""
Backward compatibility shim for mlop.compat.transformers -> pluto.compat.transformers.
This module is deprecated. Use `import pluto.compat.transformers` instead.
"""

import warnings

warnings.warn(
    "The 'mlop.compat.transformers' module is deprecated. "
    "Please use 'import pluto.compat.transformers' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pluto.compat.transformers import *  # noqa: E402, F401, F403
