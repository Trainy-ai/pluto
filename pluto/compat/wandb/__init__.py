"""wandb-compatible drop-in replacement backed by pluto.

Usage:
    Replace ``import wandb`` with ``import pluto.compat.wandb as wandb``
    and your existing wandb code will route through pluto.

Supported API:
    - wandb.init(), wandb.log(), wandb.finish()
    - wandb.watch(), wandb.unwatch()
    - wandb.config, wandb.summary, wandb.run
    - wandb.alert(), wandb.define_metric()
    - wandb.Image, wandb.Audio, wandb.Video, wandb.Table, wandb.Histogram
    - wandb.Html, wandb.Graph, wandb.Artifact, wandb.AlertLevel
    - Context manager: ``with wandb.init() as run: ...``
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union

from .config import Config
from .data_types import (
    AlertLevel,
    Artifact,
    Audio,
    Graph,
    Histogram,
    Html,
    Image,
    Table,
    Video,
)
from .run import Run
from .summary import Summary

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'WandbCompat'

__all__ = [
    # Core API
    'init',
    'log',
    'finish',
    'watch',
    'unwatch',
    'alert',
    'define_metric',
    'save',
    'restore',
    'login',
    'log_artifact',
    'use_artifact',
    'log_code',
    'mark_preempting',
    # Module-level state
    'run',
    'config',
    'summary',
    # Classes
    'Run',
    'Config',
    'Settings',
    'AlertLevel',
    # Data types
    'Image',
    'Audio',
    'Video',
    'Table',
    'Histogram',
    'Html',
    'Graph',
    'Artifact',
]

# ---------------------------------------------------------------------------
# Module-level state (mirrors wandb.run, wandb.config, wandb.summary)
# ---------------------------------------------------------------------------

run: Optional[Run] = None
config: Config = Config()
summary: Summary = Summary()


class Settings:
    """Stub for wandb.Settings — accepts kwargs and stores them."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def init(
    entity: Optional[str] = None,
    project: Optional[str] = None,
    dir: Optional[str] = None,
    id: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    config: Union[Dict[str, Any], str, None] = None,
    config_exclude_keys: Optional[List[str]] = None,
    config_include_keys: Optional[List[str]] = None,
    allow_val_change: Optional[bool] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    mode: Optional[str] = None,
    force: Optional[bool] = None,
    anonymous: Optional[str] = None,
    reinit: Optional[Union[bool, str]] = None,
    resume: Optional[Union[bool, str]] = None,
    resume_from: Optional[str] = None,
    fork_from: Optional[str] = None,
    save_code: Optional[bool] = None,
    tensorboard: Optional[bool] = None,
    sync_tensorboard: Optional[bool] = None,
    monitor_gym: Optional[bool] = None,
    settings: Optional[Any] = None,
    **kwargs: Any,
) -> Run:
    """Initialize a new run. Compatible with ``wandb.init()``."""
    import pluto as _pluto

    global run
    global summary

    # Finish any previous run before creating a new one
    if run is not None:
        if not reinit:
            logger.debug('%s: init called with existing run, finishing previous', tag)
        try:
            run.finish()
        except Exception:
            pass

    # Resolve project from env if not provided
    project = (
        project or os.environ.get('WANDB_PROJECT') or os.environ.get('PLUTO_PROJECT')
    )

    # Resolve mode
    mode = mode or os.environ.get('WANDB_MODE', 'online')
    if os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1'):
        mode = 'disabled'

    # Resolve name from env if not provided
    name = name or os.environ.get('WANDB_NAME')

    # Resolve tags from env if not provided
    if tags is None:
        env_tags = os.environ.get('WANDB_TAGS')
        if env_tags:
            tags = [t.strip() for t in env_tags.split(',') if t.strip()]

    # Filter config keys if requested
    config_dict: Dict[str, Any] = {}
    if config is not None:
        if isinstance(config, dict):
            config_dict = dict(config)
        elif hasattr(config, '__dict__'):
            config_dict = vars(config)

        if config_dict and config_include_keys:
            config_dict = {
                k: v for k, v in config_dict.items() if k in config_include_keys
            }
        if config_dict and config_exclude_keys:
            config_dict = {
                k: v for k, v in config_dict.items() if k not in config_exclude_keys
            }

    # Build pluto settings
    pluto_settings: Dict[str, Any] = {}
    if mode == 'disabled':
        pluto_settings['mode'] = 'noop'
    elif mode == 'offline':
        pluto_settings['sync_process_enabled'] = False

    # Map wandb run_id / resume
    run_id = id

    # Store wandb-only metadata in config
    extra_config: Dict[str, Any] = {}
    if notes:
        extra_config['_wandb_notes'] = notes
    if group:
        extra_config['_wandb_group'] = group
    if job_type:
        extra_config['_wandb_job_type'] = job_type

    merged_config = {**config_dict, **extra_config} or None

    # Initialize pluto
    try:
        op = _pluto.init(
            project=project,
            name=name,
            config=merged_config,
            tags=list(tags) if tags else None,
            dir=dir,
            settings=pluto_settings or None,
            run_id=run_id,
        )
    except Exception as e:
        logger.warning(
            '%s: pluto.init() failed (%s), creating disabled run',
            tag,
            e,
            exc_info=True,
        )
        # Return a disabled run that no-ops everything
        return _create_disabled_run(
            name=name,
            notes=notes,
            group=group,
            job_type=job_type,
            config_dict=config_dict,
        )

    # Create the Run wrapper
    _run = Run(
        op=op,
        name=name,
        notes=notes,
        group=group,
        job_type=job_type,
        mode=mode or 'online',
    )

    # Load config into the Config object
    _run.config._load(config_dict)

    # Set module-level state
    run = _run

    # Replace module-level config and summary proxies
    _module = _get_module()
    _module.config = _run.config
    _module.summary = _run.summary

    return _run


def log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None,
) -> None:
    """Log metrics. Compatible with ``wandb.log()``."""
    _require_run('log')
    assert run is not None
    run.log(data, step=step, commit=commit, sync=sync)


def finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None,
) -> None:
    """Finish the current run. Compatible with ``wandb.finish()``."""
    global run

    if run is not None:
        run.finish(exit_code=exit_code, quiet=quiet)
        run = None

        # Reset module-level proxies
        _module = _get_module()
        _module.config = Config()
        _module.summary = Summary()


def watch(
    models: Any = None,
    criterion: Any = None,
    log: Optional[str] = 'gradients',
    log_freq: int = 1000,
    idx: Optional[int] = None,
    log_graph: bool = False,
) -> None:
    """Watch a model. Compatible with ``wandb.watch()``."""
    _require_run('watch')
    assert run is not None
    run.watch(
        models=models,
        criterion=criterion,
        log=log,
        log_freq=log_freq,
        idx=idx,
        log_graph=log_graph,
    )


def unwatch(models: Any = None) -> None:
    """Remove model watch hooks. Compatible with ``wandb.unwatch()``."""
    if run is not None:
        run.unwatch(models)


def alert(
    title: str = '',
    text: str = '',
    level: Optional[str] = None,
    wait_duration: Optional[Union[int, float]] = None,
) -> None:
    """Send an alert. Compatible with ``wandb.alert()``."""
    _require_run('alert')
    assert run is not None
    run.alert(title=title, text=text, level=level, wait_duration=wait_duration)


def define_metric(
    name: str,
    step_metric: Optional[str] = None,
    step_sync: Optional[bool] = None,
    hidden: Optional[bool] = None,
    summary: Optional[str] = None,
    goal: Optional[str] = None,
    overwrite: Optional[bool] = None,
) -> Any:
    """Define metric behavior. No-op in pluto compat layer."""
    if run is not None:
        return run.define_metric(
            name,
            step_metric=step_metric,
            step_sync=step_sync,
            hidden=hidden,
            summary=summary,
            goal=goal,
            overwrite=overwrite,
        )
    from .run import _MetricStub

    return _MetricStub(name)


def save(
    glob_str: Optional[str] = None,
    base_path: Optional[str] = None,
    policy: str = 'live',
) -> None:
    """Sync files. Not supported — no-op."""
    logger.debug('%s: save is not supported', tag)


def restore(
    name: str = '',
    run_path: Optional[str] = None,
    replace: bool = False,
    root: Optional[str] = None,
) -> None:
    """Restore a file. Not supported — no-op."""
    logger.debug('%s: restore is not supported', tag)


def log_artifact(
    artifact_or_path: Any,
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
) -> Any:
    """Log an artifact. Compatible with ``wandb.log_artifact()``."""
    if run is not None:
        return run.log_artifact(artifact_or_path, name=name, type=type, aliases=aliases)
    logger.debug('%s: log_artifact called without active run', tag)
    return artifact_or_path


def use_artifact(
    artifact_or_name: Any,
    type: Optional[str] = None,
) -> Any:
    """Declare artifact as input. Not supported — no-op."""
    logger.debug('%s: use_artifact is not supported', tag)
    return artifact_or_name


def log_code(
    root: Optional[str] = None,
    name: Optional[str] = None,
    include_fn: Any = None,
    exclude_fn: Any = None,
) -> None:
    """Save source code. Not supported — no-op."""
    logger.debug('%s: log_code is not supported', tag)


def mark_preempting() -> None:
    """Mark run as preempted. No-op."""
    logger.debug('%s: mark_preempting is a no-op', tag)


def login(
    anonymous: Optional[str] = None,
    key: Optional[str] = None,
    relogin: Optional[bool] = None,
    host: Optional[str] = None,
    force: Optional[bool] = None,
    timeout: Optional[int] = None,
    verify: bool = False,
) -> bool:
    """Login stub. Pluto uses its own auth (``pluto login``).

    Returns True to indicate "logged in" so callers don't block.
    """
    logger.debug('%s: login is a no-op (use pluto login)', tag)
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_run(fn_name: str) -> None:
    if run is None:
        raise RuntimeError(
            f'wandb.{fn_name}() called before wandb.init(). Call wandb.init() first.'
        )


def _get_module() -> Any:
    """Return this module object for setting module-level attributes."""
    import sys

    return sys.modules[__name__]


def _create_disabled_run(
    name: Optional[str] = None,
    notes: Optional[str] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> Run:
    """Create a Run that wraps a no-op Op for disabled/error cases."""
    global run

    import pluto as _pluto

    op = _pluto.init(project='disabled', settings={'mode': 'noop'})
    _run = Run(
        op=op,
        name=name,
        notes=notes,
        group=group,
        job_type=job_type,
        mode='disabled',
    )
    if config_dict:
        _run.config._load(config_dict)

    run = _run

    _module = _get_module()
    _module.config = _run.config
    _module.summary = _run.summary

    return _run
