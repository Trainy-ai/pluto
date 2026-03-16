"""wandb API coverage registry and warning infrastructure.

Tracks which wandb APIs are supported, partially supported, stubbed,
or missing in the pluto compatibility layer. Emits user-visible warnings
when unsupported APIs are called.
"""

import enum
import warnings
from dataclasses import dataclass
from typing import Dict, Set


class SupportLevel(enum.Enum):
    """How well a wandb API is supported in the pluto compat layer."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    STUB = "stub"
    NOT_IMPLEMENTED = "not_implemented"
    MISSING = "missing"


@dataclass
class ApiEntry:
    """Registry entry for a single wandb API."""

    level: SupportLevel
    notes: str = ""


class PlutoWandbCompatWarning(UserWarning):
    """Warning emitted when unsupported wandb APIs are called.

    Silence with:
        warnings.filterwarnings("ignore", category=PlutoWandbCompatWarning)
    """

    pass


# Set of API names that have already been warned about (once per process)
_warned: Set[str] = set()


def warn_unsupported(api_name: str) -> None:
    """Issue a warning if *api_name* is a stub/unsupported/partial API.

    Each API name is warned about at most once per process.
    """
    if api_name in _warned:
        return

    entry = WANDB_API_REGISTRY.get(api_name)
    if entry is None:
        return

    msg: str | None = None
    if entry.level == SupportLevel.STUB:
        msg = f"{api_name}() is a no-op in the pluto wandb compatibility layer"
    elif entry.level == SupportLevel.NOT_IMPLEMENTED:
        msg = f"{api_name}() is not implemented in the pluto wandb compatibility layer"
    elif entry.level == SupportLevel.PARTIAL:
        msg = (
            f"{api_name}() has limited support in the pluto wandb compatibility layer"
        )
        if entry.notes:
            msg += f": {entry.notes}"

    if msg is not None:
        _warned.add(api_name)
        warnings.warn(msg, PlutoWandbCompatWarning, stacklevel=3)


def reset_warnings() -> None:
    """Reset warned set — useful for testing."""
    _warned.clear()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

WANDB_API_REGISTRY: Dict[str, ApiEntry] = {
    # ---- Top-level functions (from wandb.__all__) ----
    "wandb.init": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.log": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.finish": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.watch": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.unwatch": ApiEntry(SupportLevel.STUB),
    "wandb.alert": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.define_metric": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.save": ApiEntry(SupportLevel.STUB),
    "wandb.restore": ApiEntry(SupportLevel.STUB),
    "wandb.login": ApiEntry(
        SupportLevel.PARTIAL,
        notes="Always returns True; use pluto login",
    ),
    "wandb.log_artifact": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.use_artifact": ApiEntry(SupportLevel.STUB),
    "wandb.log_code": ApiEntry(SupportLevel.STUB),
    "wandb.mark_preempting": ApiEntry(SupportLevel.STUB),
    "wandb.setup": ApiEntry(SupportLevel.MISSING),
    "wandb.teardown": ApiEntry(SupportLevel.MISSING),
    "wandb.attach": ApiEntry(SupportLevel.MISSING),
    "wandb.sweep": ApiEntry(SupportLevel.MISSING),
    "wandb.controller": ApiEntry(SupportLevel.MISSING),
    "wandb.agent": ApiEntry(SupportLevel.MISSING),
    "wandb.require": ApiEntry(SupportLevel.MISSING),
    "wandb.jupyter": ApiEntry(SupportLevel.MISSING),
    # Top-level classes / data types
    "wandb.Run": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Config": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Settings": ApiEntry(SupportLevel.PARTIAL, notes="Accepts kwargs but no-op"),
    "wandb.AlertLevel": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Image": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Audio": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Video": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Table": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Histogram": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Html": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Graph": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Artifact": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Api": ApiEntry(
        SupportLevel.PARTIAL,
        notes="Instantiation works but query methods raise NotImplementedError",
    ),
    # Module-level state
    "wandb.run": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.config": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.summary": ApiEntry(SupportLevel.SUPPORTED),
    # ---- wandb.Run public methods ----
    "wandb.Run.log": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.finish": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.watch": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.unwatch": ApiEntry(SupportLevel.STUB),
    "wandb.Run.alert": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.define_metric": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.save": ApiEntry(SupportLevel.STUB),
    "wandb.Run.restore": ApiEntry(SupportLevel.STUB),
    "wandb.Run.log_artifact": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.use_artifact": ApiEntry(SupportLevel.STUB),
    "wandb.Run.log_code": ApiEntry(SupportLevel.STUB),
    "wandb.Run.mark_preempting": ApiEntry(SupportLevel.STUB),
    "wandb.Run.status": ApiEntry(SupportLevel.PARTIAL, notes="Always returns synced"),
    # Run properties
    "wandb.Run.id": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.name": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.entity": ApiEntry(
        SupportLevel.PARTIAL, notes="Always returns empty string"
    ),
    "wandb.Run.project": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.group": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.job_type": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.tags": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.notes": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.config": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.summary": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.url": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.dir": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.step": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.offline": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.disabled": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.resumed": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.path": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.settings": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.start_time": ApiEntry(SupportLevel.SUPPORTED),
    "wandb.Run.sweep_id": ApiEntry(SupportLevel.PARTIAL, notes="Always returns None"),
    "wandb.Run.project_url": ApiEntry(SupportLevel.SUPPORTED),
    # Run methods not in our shim
    "wandb.Run.link_artifact": ApiEntry(SupportLevel.MISSING),
    "wandb.Run.use_model": ApiEntry(SupportLevel.MISSING),
    "wandb.Run.link_model": ApiEntry(SupportLevel.MISSING),
    "wandb.Run.log_model": ApiEntry(SupportLevel.MISSING),
    "wandb.Run.join": ApiEntry(SupportLevel.MISSING),
    "wandb.Run.to_html": ApiEntry(SupportLevel.MISSING),
    "wandb.Run.display": ApiEntry(SupportLevel.MISSING),
    "wandb.Run.get_url": ApiEntry(SupportLevel.MISSING),
    "wandb.Run.get_project_url": ApiEntry(SupportLevel.MISSING),
    # ---- wandb.plot functions ----
    "wandb.plot.line_series": ApiEntry(SupportLevel.STUB),
    "wandb.plot.scatter": ApiEntry(SupportLevel.STUB),
    "wandb.plot.bar": ApiEntry(SupportLevel.STUB),
    "wandb.plot.histogram": ApiEntry(SupportLevel.STUB),
    "wandb.plot.line": ApiEntry(SupportLevel.STUB),
    "wandb.plot.confusion_matrix": ApiEntry(SupportLevel.STUB),
    "wandb.plot.roc_curve": ApiEntry(SupportLevel.STUB),
    "wandb.plot.pr_curve": ApiEntry(SupportLevel.STUB),
    # ---- wandb.Api methods ----
    "wandb.Api.runs": ApiEntry(SupportLevel.NOT_IMPLEMENTED),
    "wandb.Api.run": ApiEntry(SupportLevel.NOT_IMPLEMENTED),
    "wandb.Api.artifact": ApiEntry(SupportLevel.NOT_IMPLEMENTED),
    "wandb.Api.artifacts": ApiEntry(SupportLevel.NOT_IMPLEMENTED),
    "wandb.Api.sweep": ApiEntry(SupportLevel.NOT_IMPLEMENTED),
}
