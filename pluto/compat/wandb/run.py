"""wandb.Run-compatible wrapper around pluto.Op."""

import logging
import time as _time
from typing import Any, Dict, List, Optional, Sequence, Union

from .config import Config
from .data_types import Artifact, Audio, Graph, Histogram, Html, Image, Table, Video
from .summary import Summary

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'WandbCompat.Run'

# Sentinel for data types that have _to_pluto()
_WANDB_DATA_TYPES = (Image, Audio, Video, Table, Histogram, Html, Graph)


def _convert_value(v: Any) -> Any:
    """Convert a wandb data type wrapper to its pluto equivalent."""
    if isinstance(v, _WANDB_DATA_TYPES):
        return v._to_pluto()
    return v


def _flatten_dict(d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Flatten nested dicts using '/' separator (wandb convention)."""
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        key = f'{prefix}/{k}' if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, prefix=key))
        else:
            flat[key] = v
    return flat


class Run:
    """A wandb.Run-compatible object wrapping a pluto Op.

    Provides the same interface as wandb.Run so user code like
    ``run.log()``, ``run.config``, ``run.summary``, etc. works seamlessly.
    """

    def __init__(
        self,
        op: Any,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        mode: str = 'online',
    ) -> None:
        self._op = op
        self._name = name
        self._notes = notes
        self._group = group
        self._job_type = job_type
        self._mode = mode

        self._config = Config(op=op)
        self._summary = Summary()
        self._pending_data: Dict[str, Any] = {}
        self._step = 0
        self._watched_models: List[Any] = []
        self._start_time: float = _time.time()

    # -- Properties matching wandb.Run --

    @property
    def id(self) -> str:
        if self._op.run_id:
            return self._op.run_id
        op_id = self._op.id
        return str(op_id) if op_id is not None else ''

    @property
    def name(self) -> Optional[str]:
        return self._name or getattr(self._op.settings, '_op_name', None)

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def entity(self) -> str:
        return ''

    @property
    def project(self) -> Optional[str]:
        return getattr(self._op.settings, 'project', None)

    @property
    def group(self) -> Optional[str]:
        return self._group

    @property
    def job_type(self) -> Optional[str]:
        return self._job_type

    @property
    def tags(self) -> tuple:
        return tuple(self._op.tags)

    @tags.setter
    def tags(self, value: Union[tuple, list, Sequence[str]]) -> None:
        current = set(self._op.tags)
        new = set(value)
        to_add = new - current
        to_remove = current - new
        if to_remove:
            self._op.remove_tags(list(to_remove))
        if to_add:
            self._op.add_tags(list(to_add))

    @property
    def notes(self) -> Optional[str]:
        return self._notes

    @notes.setter
    def notes(self, value: str) -> None:
        self._notes = value

    @property
    def config(self) -> Config:
        return self._config

    @config.setter
    def config(self, value: Any) -> None:
        if isinstance(value, dict):
            self._config.update(value)
        elif isinstance(value, Config):
            self._config = value

    @property
    def summary(self) -> Summary:
        return self._summary

    @property
    def url(self) -> Optional[str]:
        return getattr(self._op.settings, 'url_view', None)

    @property
    def dir(self) -> str:
        return self._op.settings.get_dir()

    @property
    def step(self) -> int:
        return self._step

    @property
    def offline(self) -> bool:
        return self._mode == 'offline'

    @property
    def disabled(self) -> bool:
        return self._mode == 'disabled'

    @property
    def resumed(self) -> bool:
        return self._op.resumed

    @property
    def path(self) -> str:
        return f'{self.entity}/{self.project}/{self.id}'

    @property
    def settings(self) -> Any:
        return self._op.settings

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def sweep_id(self) -> Optional[str]:
        return None

    @property
    def project_url(self) -> Optional[str]:
        url = self.url
        if url:
            # Strip the run-specific part to get project URL
            parts = url.rsplit('/', 1)
            return parts[0] if len(parts) > 1 else url
        return None

    # -- Core methods --

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: Optional[bool] = None,
        sync: Optional[bool] = None,
    ) -> None:
        """Log metrics/data to the run."""
        # Flatten nested dicts
        flat = _flatten_dict(data)

        # Convert wandb data types to pluto equivalents
        converted = {k: _convert_value(v) for k, v in flat.items()}

        if commit is False:
            # Buffer data, don't send yet
            self._pending_data.update(converted)
            return

        # Merge with any pending data
        if self._pending_data:
            merged = {**self._pending_data, **converted}
            self._pending_data = {}
        else:
            merged = converted

        # Update summary with scalar values
        self._summary._update_from_log(merged)

        # Track step
        if step is not None:
            self._step = step
        else:
            self._step += 1

        # Forward to pluto
        try:
            self._op.log(merged, step=step)
        except Exception as e:
            logger.debug('%s: log failed: %s', tag, e)

    def finish(
        self,
        exit_code: Optional[int] = None,
        quiet: Optional[bool] = None,
    ) -> None:
        """Mark the run as finished."""
        try:
            self._op.finish(code=exit_code)
        except Exception as e:
            logger.debug('%s: finish failed: %s', tag, e)

    def watch(
        self,
        models: Any = None,
        criterion: Any = None,
        log: Optional[str] = 'gradients',
        log_freq: int = 1000,
        idx: Optional[int] = None,
        log_graph: bool = False,
    ) -> None:
        """Watch a PyTorch model for gradient/parameter logging."""
        if models is None:
            return

        model_list = models if isinstance(models, (list, tuple)) else [models]
        for model in model_list:
            try:
                self._op.watch(model, log_freq=log_freq)
                self._watched_models.append(model)
            except Exception as e:
                logger.debug('%s: watch failed: %s', tag, e)

    def unwatch(self, models: Any = None) -> None:
        """Remove model watching hooks (best-effort)."""
        logger.debug('%s: unwatch is a no-op in pluto compat layer', tag)

    def alert(
        self,
        title: str = '',
        text: str = '',
        level: Optional[str] = None,
        wait_duration: Optional[Union[int, float]] = None,
    ) -> None:
        """Send an alert."""
        try:
            kwargs: Dict[str, Any] = {}
            if wait_duration is not None:
                kwargs['wait_duration'] = wait_duration
            self._op.alert(
                title=title,
                message=text,
                level=level or 'INFO',
                **kwargs,
            )
        except Exception as e:
            logger.debug('%s: alert failed: %s', tag, e)

    def define_metric(
        self,
        name: str,
        step_metric: Optional[str] = None,
        step_sync: Optional[bool] = None,
        hidden: Optional[bool] = None,
        summary: Optional[str] = None,
        goal: Optional[str] = None,
        overwrite: Optional[bool] = None,
    ) -> Any:
        """Define metric behavior (aggregation, custom x-axis)."""
        # Build definition dict for client-side aggregation
        definition: dict = {'name': name}
        if step_metric is not None:
            definition['step_metric'] = step_metric
        if summary is not None:
            definition['summary'] = summary
        if goal is not None:
            definition['goal'] = goal
        if hidden is not None:
            definition['hidden'] = hidden

        # Register with Summary for client-side aggregation
        self._summary._set_metric_definition(name, definition)

        # Forward to Op for server sync
        try:
            self._op.define_metric(
                name,
                step_metric=step_metric,
                summary=summary,
                goal=goal,
                hidden=hidden,
            )
        except Exception as e:
            logger.debug('%s: define_metric server sync failed: %s', tag, e)

        return _MetricStub(name)

    def save(
        self,
        glob_str: Optional[str] = None,
        base_path: Optional[str] = None,
        policy: str = 'live',
    ) -> None:
        """Sync files to the run. No-op in pluto compat layer."""
        logger.debug('%s: save is not supported', tag)

    def restore(
        self,
        name: str,
        run_path: Optional[str] = None,
        replace: bool = False,
        root: Optional[str] = None,
    ) -> None:
        """Download a file from a run. Not supported."""
        logger.debug('%s: restore is not supported', tag)

    def log_artifact(
        self,
        artifact_or_path: Any,
        name: Optional[str] = None,
        type: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> Any:
        """Log an artifact (file collection) to the run."""
        if isinstance(artifact_or_path, Artifact):
            pluto_files = artifact_or_path._to_pluto_files()
            for i, pf in enumerate(pluto_files):
                if hasattr(pf, '_name'):
                    log_name = f'{artifact_or_path.name}/{pf._name}'
                else:
                    log_name = f'{artifact_or_path.name}/{i}'
                try:
                    self._op.log({log_name: pf})
                except Exception as e:
                    logger.debug('%s: log_artifact file failed: %s', tag, e)
            return artifact_or_path
        elif isinstance(artifact_or_path, str):
            import os

            from pluto.file import Artifact as PlutoArtifact

            art = PlutoArtifact(data=artifact_or_path, caption=name)
            log_name = name or os.path.basename(artifact_or_path)
            try:
                self._op.log({log_name: art})
            except Exception as e:
                logger.debug('%s: log_artifact path failed: %s', tag, e)
        return artifact_or_path

    def use_artifact(
        self,
        artifact_or_name: Any,
        type: Optional[str] = None,
    ) -> Any:
        """Declare an artifact as input. Not supported."""
        logger.debug('%s: use_artifact is not supported', tag)
        return artifact_or_name

    def log_code(
        self,
        root: Optional[str] = None,
        name: Optional[str] = None,
        include_fn: Any = None,
        exclude_fn: Any = None,
    ) -> None:
        """Save source code. Not supported."""
        logger.debug('%s: log_code is not supported', tag)

    def mark_preempting(self) -> None:
        """Mark run as preempted. No-op (pluto handles this via signals)."""
        logger.debug('%s: mark_preempting is a no-op', tag)

    def status(self) -> Dict[str, Any]:
        return {'synced': True}

    # -- Context manager --

    def __enter__(self) -> 'Run':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        exit_code = 1 if exc_type else 0
        self.finish(exit_code=exit_code)

    def __repr__(self) -> str:
        return f'<Run {self.project}/{self.id} ({self.name})>'


class _MetricStub:
    """Stub returned by define_metric()."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f'<Metric {self.name}>'
