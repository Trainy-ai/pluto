"""Standalone ``neptune_scale.Run`` replacement backed by Pluto.

This module provides a :class:`Run` class with the same public API as
``neptune_scale.Run``.  All data is logged to Pluto; no Neptune
dependency is required.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with neptune.py compat layer)
# ---------------------------------------------------------------------------


def _get_env_with_deprecation(new_key: str, old_key: str) -> Optional[str]:
    import warnings

    value = os.environ.get(new_key)
    if value is None:
        old_value = os.environ.get(old_key)
        if old_value is not None:
            warnings.warn(
                f'Environment variable {old_key} is deprecated. Use {new_key} instead.',
                DeprecationWarning,
                stacklevel=3,
            )
            return old_value
    return value


def _get_pluto_config_from_env() -> Optional[Dict[str, Any]]:
    project = _get_env_with_deprecation('PLUTO_PROJECT', 'MLOP_PROJECT')
    if not project:
        return None
    config: Dict[str, Any] = {'project': project}
    if api_key := _get_env_with_deprecation('PLUTO_API_KEY', 'MLOP_API_KEY'):
        config['api_key'] = api_key
    if url_app := _get_env_with_deprecation('PLUTO_URL_APP', 'MLOP_URL_APP'):
        config['url_app'] = url_app
    if url_api := _get_env_with_deprecation('PLUTO_URL_API', 'MLOP_URL_API'):
        config['url_api'] = url_api
    if url_ingest := _get_env_with_deprecation('PLUTO_URL_INGEST', 'MLOP_URL_INGEST'):
        config['url_ingest'] = url_ingest
    return config


def _safe_import_pluto():
    try:
        import pluto

        return pluto
    except ImportError:
        logger.warning(
            'pluto.compat.neptune_scale: pluto package not available'
        )
        return None


def _convert_file_to_pluto(file_obj: Any, pluto_module: Any) -> Any:
    """Convert a neptune_scale File to a Pluto file type."""
    if hasattr(file_obj, 'source'):
        source = file_obj.source
        mime_type = getattr(file_obj, 'mime_type', None)
    else:
        source = file_obj
        mime_type = None

    if mime_type:
        if mime_type.startswith('image/'):
            return pluto_module.Image(source)
        elif mime_type.startswith('audio/'):
            return pluto_module.Audio(source)
        elif mime_type.startswith('video/'):
            return pluto_module.Video(source)

    if isinstance(source, (str, os.PathLike)):
        source_str = str(source).lower()
        if any(
            source_str.endswith(ext)
            for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
        ):
            return pluto_module.Image(source)
        elif any(source_str.endswith(ext) for ext in ['.wav', '.mp3', '.ogg', '.flac']):
            return pluto_module.Audio(source)
        elif any(source_str.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.webm']):
            return pluto_module.Video(source)

    if isinstance(source, (bytes, bytearray)):
        from pluto.compat.neptune import _detect_media_type_from_bytes

        detected = _detect_media_type_from_bytes(source)
        if detected == 'image':
            return pluto_module.Image(source)
        elif detected == 'audio':
            return pluto_module.Audio(source)
        elif detected == 'video':
            return pluto_module.Video(source)

    return pluto_module.Artifact(source)


def _convert_histogram_to_pluto(hist_obj: Any, pluto_module: Any) -> Any:
    """Convert a :class:`~pluto.compat.neptune_scale.types.Histogram` to Pluto."""
    if hasattr(hist_obj, 'bin_edges') and hasattr(hist_obj, 'counts'):
        bin_edges = (
            hist_obj.bin_edges_as_list()
            if hasattr(hist_obj, 'bin_edges_as_list')
            else list(hist_obj.bin_edges)
        )
        if hasattr(hist_obj, 'counts') and hist_obj.counts is not None:
            counts = (
                hist_obj.counts_as_list()
                if hasattr(hist_obj, 'counts_as_list')
                else list(hist_obj.counts)
            )
        elif hasattr(hist_obj, 'densities') and hist_obj.densities is not None:
            counts = (
                hist_obj.densities_as_list()
                if hasattr(hist_obj, 'densities_as_list')
                else list(hist_obj.densities)
            )
        else:
            counts = None

        if counts:
            return pluto_module.Histogram(data=(counts, bin_edges), bins=None)

    return pluto_module.Histogram(data=hist_obj)


# ---------------------------------------------------------------------------
# Standalone Run
# ---------------------------------------------------------------------------


class Run:
    """Drop-in replacement for ``neptune_scale.Run`` that logs to Pluto.

    Accepts the same constructor arguments as ``neptune_scale.Run`` so that
    existing code works without changes.  Neptune is **not** called; all
    data goes directly to Pluto.

    Configuration is read from environment variables (see module docstring).
    """

    _CLEANUP_TIMEOUT_SECONDS: float = 5.0

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        api_token: Optional[str] = None,
        run_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        # Accept (and ignore) any other Neptune kwargs for compatibility
        **kwargs: Any,
    ) -> None:
        self._closed = False
        self._close_lock = threading.Lock()
        self._pluto_run: Optional[Any] = None
        self._pluto: Optional[Any] = _safe_import_pluto()

        if self._pluto is None:
            return

        try:
            pluto_config = _get_pluto_config_from_env()
            if pluto_config is None:
                logger.info(
                    'pluto.compat.neptune_scale: PLUTO_PROJECT not set, '
                    'run will be a no-op'
                )
                return

            name = experiment_name or 'neptune-migration'
            if run_id:
                name = f'{name}-{run_id}'

            pluto_init_kwargs: Dict[str, Any] = {
                'project': pluto_config['project'],
                'name': name,
                'config': {},
            }
            if run_id:
                pluto_init_kwargs['run_id'] = run_id

            settings: Dict[str, Any] = {
                'sync_process_enabled': True,
                'sync_process_shutdown_timeout': 3.0,
            }
            if 'url_app' in pluto_config:
                settings['url_app'] = pluto_config['url_app']
            if 'url_api' in pluto_config:
                settings['url_api'] = pluto_config['url_api']
            if 'url_ingest' in pluto_config:
                settings['url_ingest'] = pluto_config['url_ingest']
            if 'api_key' in pluto_config:
                settings['_auth'] = pluto_config['api_key']

            pluto_init_kwargs['settings'] = settings

            self._pluto_run = self._pluto.init(**pluto_init_kwargs)
            logger.info(
                'pluto.compat.neptune_scale: Initialized Pluto run '
                'for project=%s, name=%s',
                pluto_config['project'],
                name,
            )
            atexit.register(self._atexit_cleanup)

        except Exception as e:
            logger.warning(
                'pluto.compat.neptune_scale: Failed to initialize Pluto run: %s. '
                'Run will be a no-op.',
                e,
            )
            self._pluto_run = None

    # ---- atexit / cleanup --------------------------------------------------

    def _atexit_cleanup(self) -> None:
        self._finish_with_timeout(timeout=self._CLEANUP_TIMEOUT_SECONDS)

    def _finish_with_timeout(self, timeout: float) -> None:
        with self._close_lock:
            if self._pluto_run is None:
                return
            pluto_run = self._pluto_run
            self._pluto_run = None

        done = threading.Event()

        def _do_finish() -> None:
            try:
                pluto_run.finish()
            except Exception:
                pass
            finally:
                done.set()

        t = threading.Thread(target=_do_finish, daemon=False)
        t.start()
        completed = done.wait(timeout=timeout)
        if completed:
            t.join(timeout=1.0)

    # ---- Neptune-compatible API --------------------------------------------

    def log_metrics(
        self,
        data: Dict[str, float],
        step: int,
        timestamp: Any = None,
        **kwargs: Any,
    ) -> None:
        if self._pluto_run:
            try:
                self._pluto_run.log(data, step=step)
            except Exception as e:
                logger.debug('neptune_scale shim: log_metrics failed: %s', e)

    def log_configs(
        self,
        data: Dict[str, Any],
        flatten: bool = False,
        cast_unsupported: bool = False,
        **kwargs: Any,
    ) -> None:
        if data is None or self._pluto_run is None:
            return
        try:
            pluto_data = data
            if is_dataclass(pluto_data) and not isinstance(pluto_data, type):
                pluto_data = asdict(pluto_data)
            if flatten:
                pluto_data = _flatten_nested(pluto_data)
            if cast_unsupported:
                pluto_data = _cast_unsupported(pluto_data)

            if hasattr(self._pluto_run, 'config'):
                if self._pluto_run.config is None:
                    self._pluto_run.config = {}
                self._pluto_run.config.update(pluto_data)
            if hasattr(self._pluto_run, '_iface') and self._pluto_run._iface:
                self._pluto_run._iface._update_config(pluto_data)
        except Exception as e:
            logger.debug('neptune_scale shim: log_configs failed: %s', e)

    def assign_files(self, files: Dict[str, Any], **kwargs: Any) -> None:
        if self._pluto_run and self._pluto:
            try:
                pluto_files = {}
                for key, file_obj in files.items():
                    try:
                        pluto_files[key] = _convert_file_to_pluto(
                            file_obj, self._pluto
                        )
                    except Exception as e:
                        logger.debug(
                            'neptune_scale shim: file convert failed for %s: %s', key, e
                        )
                if pluto_files:
                    self._pluto_run.log(pluto_files)
            except Exception as e:
                logger.debug('neptune_scale shim: assign_files failed: %s', e)

    def log_files(
        self,
        files: Dict[str, Any],
        step: int,
        timestamp: Any = None,
        **kwargs: Any,
    ) -> None:
        if self._pluto_run and self._pluto:
            try:
                pluto_files = {}
                for key, file_obj in files.items():
                    try:
                        pluto_files[key] = _convert_file_to_pluto(
                            file_obj, self._pluto
                        )
                    except Exception as e:
                        logger.debug(
                            'neptune_scale shim: file convert failed for %s: %s', key, e
                        )
                if pluto_files:
                    self._pluto_run.log(pluto_files, step=step)
            except Exception as e:
                logger.debug('neptune_scale shim: log_files failed: %s', e)

    def log_histograms(
        self,
        histograms: Dict[str, Any],
        step: int,
        timestamp: Any = None,
        **kwargs: Any,
    ) -> None:
        if self._pluto_run and self._pluto:
            try:
                pluto_hists = {}
                for key, hist_obj in histograms.items():
                    try:
                        pluto_hists[key] = _convert_histogram_to_pluto(
                            hist_obj, self._pluto
                        )
                    except Exception as e:
                        logger.debug(
                            'neptune_scale shim: histogram convert failed for %s: %s',
                            key,
                            e,
                        )
                if pluto_hists:
                    self._pluto_run.log(pluto_hists, step=step)
            except Exception as e:
                logger.debug('neptune_scale shim: log_histograms failed: %s', e)

    def add_tags(self, tags: Union[List[str], str], **kwargs: Any) -> None:
        if self._pluto_run:
            try:
                self._pluto_run.add_tags(tags)
            except Exception as e:
                logger.debug('neptune_scale shim: add_tags failed: %s', e)

    def remove_tags(self, tags: Union[List[str], str], **kwargs: Any) -> None:
        if self._pluto_run:
            try:
                self._pluto_run.remove_tags(tags)
            except Exception as e:
                logger.debug('neptune_scale shim: remove_tags failed: %s', e)

    def log_string_series(
        self,
        data: Dict[str, str],
        step: int,
        timestamp: Any = None,
        **kwargs: Any,
    ) -> None:
        logger.debug(
            'neptune_scale shim: string series not supported, skipping'
        )

    def close(self, **kwargs: Any) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._finish_with_timeout(timeout=self._CLEANUP_TIMEOUT_SECONDS)

    def terminate(self, **kwargs: Any) -> None:
        with self._close_lock:
            self._closed = True
        self._finish_with_timeout(timeout=2.0)

    def wait_for_submission(self, **kwargs: Any) -> None:
        pass

    def wait_for_processing(self, **kwargs: Any) -> None:
        pass

    def get_run_url(self) -> str:
        if self._pluto_run and hasattr(self._pluto_run, 'url'):
            return self._pluto_run.url
        return 'pluto://run'

    def get_experiment_url(self) -> str:
        return self.get_run_url()

    def __enter__(self) -> 'Run':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_nested(data: Any) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}

    def _inner(d: Any, prefix: str = '') -> None:
        if is_dataclass(d) and not isinstance(d, type):
            d = asdict(d)
        if not isinstance(d, Mapping):
            raise TypeError(
                f'Cannot flatten value of type {type(d)}. Try flatten=False.'
            )
        for key, value in d.items():
            new_key = f'{prefix}/{key}' if prefix else str(key)
            if isinstance(value, Mapping) or (
                is_dataclass(value) and not isinstance(value, type)
            ):
                _inner(value, prefix=new_key)
            else:
                flattened[new_key] = value

    _inner(data)
    return flattened


def _cast_unsupported(
    data: Dict[str, Any],
) -> Dict[str, Union[str, float, int, bool, datetime, List[str], set, tuple]]:
    result: Dict[
        str, Union[str, float, int, bool, datetime, List[str], set, tuple]
    ] = {}
    for k, v in data.items():
        if isinstance(v, (float, bool, int, str, datetime)) or (
            isinstance(v, (list, set, tuple))
            and all(isinstance(item, str) for item in v)
        ):
            result[k] = v
        else:
            result[k] = '' if v is None else str(v)
    return result
