"""Dual-logging mode: log to both real wandb AND pluto simultaneously.

Activated by setting ``PLUTO_WANDB_MODE=dual``.  Real wandb is the
primary system; pluto mirrors scalar metrics, config, and lifecycle
events.  If pluto fails at any point, wandb continues unaffected.

Rich data types (Image, Table, Histogram, etc.) are sent to real
wandb only — pluto receives the scalar subset.
"""

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger('pluto')
tag = 'WandbDual'


class DualRun:
    """Wraps a real wandb.Run *and* a pluto Op, forwarding to both.

    Attribute access falls through to the real wandb Run, so the full
    wandb API is available.  Only the core logging/lifecycle methods
    are intercepted to also push data to pluto.
    """

    def __init__(self, wandb_run: Any, pluto_op: Any) -> None:
        # Use object.__setattr__ to avoid triggering __setattr__
        object.__setattr__(self, '_wandb_run', wandb_run)
        object.__setattr__(self, '_pluto_op', pluto_op)
        object.__setattr__(self, '_pluto_finished', False)
        object.__setattr__(self, '_lock', threading.Lock())

    # -- Core methods (dual-logged) ------------------------------------

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: Optional[bool] = None,
        sync: Optional[bool] = None,
    ) -> None:
        """Log to real wandb, then mirror scalars to pluto."""
        self._wandb_run.log(data, step=step, commit=commit, sync=sync)
        self._mirror_log(data, step=step)

    def finish(
        self,
        exit_code: Optional[int] = None,
        quiet: Optional[bool] = None,
    ) -> None:
        """Finish both runs (pluto first, then wandb)."""
        self._finish_pluto()
        self._wandb_run.finish(exit_code=exit_code, quiet=quiet)

    def watch(
        self,
        models: Any = None,
        criterion: Any = None,
        log: Optional[str] = 'gradients',
        log_freq: int = 1000,
        idx: Optional[int] = None,
        log_graph: bool = False,
    ) -> None:
        """Watch model in both systems."""
        self._wandb_run.watch(
            models, criterion=criterion, log=log,
            log_freq=log_freq, idx=idx, log_graph=log_graph,
        )
        if models is not None and self._pluto_op:
            try:
                model_list = (
                    models if isinstance(models, (list, tuple))
                    else [models]
                )
                for model in model_list:
                    self._pluto_op.watch(model, log_freq=log_freq)
            except Exception as e:
                logger.debug('%s: pluto watch failed: %s', tag, e)

    def unwatch(self, models: Any = None) -> None:
        self._wandb_run.unwatch(models)

    # -- Config mirroring ----------------------------------------------

    @property
    def config(self) -> Any:
        return self._wandb_run.config

    @config.setter
    def config(self, value: Any) -> None:
        self._wandb_run.config = value
        if self._pluto_op and isinstance(value, dict):
            try:
                self._pluto_op.update_config(value)
            except Exception as e:
                logger.debug('%s: pluto config update failed: %s', tag, e)

    @property
    def summary(self) -> Any:
        return self._wandb_run.summary

    # -- Context manager -----------------------------------------------

    def __enter__(self) -> 'DualRun':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        exit_code = 1 if exc_type else 0
        self.finish(exit_code=exit_code)

    # -- Transparent forwarding ----------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wandb_run, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'config':
            # Use the property setter
            type(self).config.fset(self, value)  # type: ignore[union-attr]
            return
        # Forward attribute sets to real wandb run
        setattr(self._wandb_run, name, value)

    def __repr__(self) -> str:
        return f'<DualRun wandb={self._wandb_run!r}>'

    # -- Internal helpers ----------------------------------------------

    def _mirror_log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """Mirror scalar values from a log() call to pluto."""
        if not self._pluto_op:
            return
        try:
            scalars = _extract_scalars(data)
            if scalars:
                self._pluto_op.log(scalars, step=step)
        except Exception as e:
            logger.debug('%s: pluto log failed: %s', tag, e)

    def _finish_pluto(self) -> None:
        """Finish the pluto run (at most once, non-blocking)."""
        with self._lock:
            if self._pluto_finished or not self._pluto_op:
                return
            object.__setattr__(self, '_pluto_finished', True)
        try:
            self._pluto_op.finish()
        except Exception as e:
            logger.debug('%s: pluto finish failed: %s', tag, e)


def _extract_scalars(
    data: Dict[str, Any],
    prefix: str = '',
) -> Dict[str, Any]:
    """Recursively extract scalar values from a log dict.

    Nested dicts are flattened with '/' separator.  Non-scalar values
    (wandb.Image, wandb.Table, etc.) are skipped — they go to real
    wandb only.
    """
    scalars: Dict[str, Any] = {}
    for k, v in data.items():
        key = f'{prefix}/{k}' if prefix else k
        if isinstance(v, dict):
            scalars.update(_extract_scalars(v, prefix=key))
        elif isinstance(v, (int, float)):
            scalars[key] = v
        elif isinstance(v, bool):
            scalars[key] = int(v)
        # Everything else (images, tables, strings, etc.) is skipped
    return scalars


def setup_dual(
    wandb_module: Any,
    real_wandb: Any,
) -> None:
    """Install dual-logging wrappers on the wandb module.

    Called from ``wandb/__init__.py`` when ``PLUTO_WANDB_MODE=dual``.
    Replaces ``init``, ``log``, ``finish``, and ``watch`` with
    versions that forward to both real wandb and pluto.
    """
    import pluto as _pluto

    # Module-level state
    _state: Dict[str, Any] = {'run': None}

    def dual_init(
        *,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        dir: Optional[str] = None,
        id: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Any = None,
        config: Any = None,
        mode: Optional[str] = None,
        **kwargs: Any,
    ) -> DualRun:
        """Initialize both real wandb and pluto runs."""
        # Finish previous dual run if any
        if _state['run'] is not None:
            try:
                _state['run'].finish()
            except Exception:
                pass

        # Real wandb init (always — this is the primary)
        wandb_run = real_wandb.init(
            entity=entity, project=project, dir=dir,
            id=id, name=name, notes=notes, tags=tags,
            config=config, mode=mode, **kwargs,
        )

        # Pluto init (best effort)
        pluto_op = None
        try:
            # Extract config dict for pluto
            pluto_config = None
            if config is not None:
                if isinstance(config, dict):
                    pluto_config = config
                elif hasattr(config, '__dict__'):
                    pluto_config = vars(config)

            pluto_op = _pluto.init(
                project=project,
                name=name,
                config=pluto_config,
                tags=list(tags) if tags else None,
                dir=dir,
                run_id=id,
            )
            logger.info(
                '%s: dual logging active — wandb + pluto', tag,
            )
        except Exception as e:
            logger.warning(
                '%s: pluto init failed (%s), continuing with '
                'wandb-only logging',
                tag, e,
            )

        dual_run = DualRun(wandb_run, pluto_op)
        _state['run'] = dual_run

        # Update module-level state
        wandb_module.run = dual_run

        return dual_run

    def dual_log(
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: Optional[bool] = None,
        sync: Optional[bool] = None,
    ) -> None:
        """Log to both systems via the active dual run."""
        if _state['run'] is not None:
            _state['run'].log(
                data, step=step, commit=commit, sync=sync,
            )
        else:
            real_wandb.log(
                data, step=step, commit=commit, sync=sync,
            )

    def dual_finish(
        exit_code: Optional[int] = None,
        quiet: Optional[bool] = None,
    ) -> None:
        """Finish both runs."""
        if _state['run'] is not None:
            _state['run'].finish(
                exit_code=exit_code, quiet=quiet,
            )
            _state['run'] = None
            wandb_module.run = None

    def dual_watch(
        models: Any = None,
        criterion: Any = None,
        log: Optional[str] = 'gradients',
        log_freq: int = 1000,
        idx: Optional[int] = None,
        log_graph: bool = False,
    ) -> None:
        """Watch model in both systems."""
        if _state['run'] is not None:
            _state['run'].watch(
                models, criterion=criterion, log=log,
                log_freq=log_freq, idx=idx, log_graph=log_graph,
            )
        else:
            real_wandb.watch(
                models, criterion=criterion, log=log,
                log_freq=log_freq, idx=idx, log_graph=log_graph,
            )

    # Install wrappers
    wandb_module.init = dual_init
    wandb_module.log = dual_log
    wandb_module.finish = dual_finish
    wandb_module.watch = dual_watch
