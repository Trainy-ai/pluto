"""Deprecated CLI entry point."""

import warnings

warnings.warn(
    "The 'mlop' command is deprecated. Use 'pluto' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pluto.__main__ import main  # noqa: E402

if __name__ == '__main__':
    main()
