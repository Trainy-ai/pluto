"""
Pluto sync process module.

This module provides a separate process for handling network I/O to the
Pluto backend, enabling crash-resilient and DDP-friendly logging.

Usage:
    # Enable sync process mode
    pluto.init(project="x", settings={"x_use_sync_process": True})

Architecture:
    - Training process writes metrics to SQLite (fast, local)
    - Sync process (spawned) tails SQLite and uploads to backend
    - Training can exit immediately; sync process handles the rest
    - Crash-safe: data on disk survives training crashes
"""

from .process import (
    get_existing_sync_process,
    is_sync_process_alive,
    start_sync_process,
    stop_sync_process,
)

__all__ = [
    "start_sync_process",
    "stop_sync_process",
    "get_existing_sync_process",
    "is_sync_process_alive",
]
