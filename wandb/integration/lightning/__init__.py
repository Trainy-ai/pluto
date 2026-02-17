"""Stub for wandb.integration.lightning — provides WandbLogger.

Maps ``from wandb.integration.lightning import WandbLogger`` to pluto's
own Lightning logger (pluto.compat.lightning.MLOPLogger).
"""

from pluto.compat.lightning import MLOPLogger as WandbLogger  # noqa: F401

__all__ = ['WandbLogger']
