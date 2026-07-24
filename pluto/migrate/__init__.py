"""
pluto.migrate — import historical experiment data into Pluto.

Two-phase pipeline: export a source platform's runs to on-disk parquet
(``pluto migrate wandb export``), then load the staged data into Pluto
(``pluto migrate wandb load``). Both phases are resumable.

Requires the ``migrate`` extra: ``pip install 'pluto-ml[migrate]'``.
"""

from typing import Any

_INSTALL_HINT = (
    "pluto.migrate requires the 'migrate' extra. "
    "Install it with: pip install 'pluto-ml[migrate]'"
)


def __getattr__(name: str) -> Any:
    # Lazy so `import pluto` never pays for (or requires) wandb/pyarrow.
    if name == 'WandbExporter':
        from pluto.migrate.wandb_export import WandbExporter

        return WandbExporter
    if name == 'PlutoLoader':
        from pluto.migrate.loader import PlutoLoader

        return PlutoLoader
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
