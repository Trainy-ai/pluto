"""
Backward compatibility shim for mlop.compat.neptune -> pluto.compat.neptune migration.
This module is deprecated. Use `import pluto.compat.neptune` instead.
"""

import warnings

warnings.warn(
    "The 'mlop.compat.neptune' module is deprecated. "
    "Please use 'import pluto.compat.neptune' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pluto.compat.neptune import *  # noqa: E402, F401, F403
