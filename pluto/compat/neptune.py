"""
Neptune-to-pluto compatibility layer for seamless migration.

This module provides a monkeypatch that allows existing Neptune API calls
to simultaneously log to both Neptune and pluto, enabling a gradual migration
without breaking existing workflows.

Usage:
    import pluto.compat.neptune  # This automatically patches Neptune

    # Your existing Neptune code continues to work
    from neptune_scale import Run
    run = Run(experiment_name="my-experiment")
    run.log_metrics({"loss": 0.5}, step=0)
    run.close()

    # Now logs to BOTH Neptune and pluto!

Configuration:
    Set environment variables:
    - PLUTO_PROJECT: pluto project name (required)
    - PLUTO_API_KEY: pluto API key (optional, falls back to keyring)
    - PLUTO_URL_APP: pluto app URL (optional, uses default)
    - PLUTO_URL_API: pluto API URL (optional, uses default)
    - PLUTO_URL_INGEST: pluto ingest URL (optional, uses default)

    Legacy MLOP_* environment variables are also supported with deprecation warnings.

Hard Requirements:
    - MUST NOT break existing Neptune functionality under ANY condition
    - If pluto is down/misconfigured, silently continue with Neptune only
    - Zero impact on Neptune's behavior, return values, or exceptions
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_original_neptune_run = None
_patch_applied = False


def _get_env_with_deprecation(new_key: str, old_key: str) -> Optional[str]:
    """Get env var with fallback to deprecated MLOP_* name."""
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
    """
    Extract pluto configuration from environment variables.

    Returns:
        Config dict if PLUTO_PROJECT is set, None otherwise
    """
    project = _get_env_with_deprecation('PLUTO_PROJECT', 'MLOP_PROJECT')
    if not project:
        return None

    config = {'project': project}

    # Optional: API key (will fall back to keyring if not provided)
    if api_key := _get_env_with_deprecation('PLUTO_API_KEY', 'MLOP_API_KEY'):
        config['api_key'] = api_key

    # Optional: Custom URLs
    if url_app := _get_env_with_deprecation('PLUTO_URL_APP', 'MLOP_URL_APP'):
        config['url_app'] = url_app
    if url_api := _get_env_with_deprecation('PLUTO_URL_API', 'MLOP_URL_API'):
        config['url_api'] = url_api
    if url_ingest := _get_env_with_deprecation('PLUTO_URL_INGEST', 'MLOP_URL_INGEST'):
        config['url_ingest'] = url_ingest

    return config


def _safe_import_pluto():
    """
    Safely import pluto, returning None if unavailable.

    Returns:
        pluto module or None if import fails
    """
    try:
        import pluto

        return pluto
    except ImportError:
        logger.warning(
            'pluto.compat.neptune: pluto not installed, '
            'continuing with Neptune-only logging'
        )
        return None


def _convert_neptune_file_to_pluto(file_obj, pluto_module):
    """
    Convert Neptune File object to appropriate pluto file type.

    Args:
        file_obj: Neptune File object or file path
        pluto_module: The pluto module

    Returns:
        pluto.Image, pluto.Audio, pluto.Video, or pluto.Artifact
    """
    # Handle neptune_scale.types.File objects
    if hasattr(file_obj, 'source'):
        source = file_obj.source
        mime_type = getattr(file_obj, 'mime_type', None)
    else:
        # Assume it's a file path or data
        source = file_obj
        mime_type = None

    # Infer type from mime_type or file extension
    if mime_type:
        if mime_type.startswith('image/'):
            return pluto_module.Image(source)
        elif mime_type.startswith('audio/'):
            return pluto_module.Audio(source)
        elif mime_type.startswith('video/'):
            return pluto_module.Video(source)

    # Try to infer from file path
    if isinstance(source, (str, os.PathLike)):
        source_str = str(source).lower()
        if any(
            source_str.endswith(ext)
            for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        ):
            return pluto_module.Image(source)
        elif any(source_str.endswith(ext) for ext in ['.wav', '.mp3', '.ogg', '.flac']):
            return pluto_module.Audio(source)
        elif any(source_str.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.webm']):
            return pluto_module.Video(source)

    # Default to generic artifact
    return pluto_module.Artifact(source)


def _convert_neptune_histogram_to_pluto(hist_obj, pluto_module):
    """
    Convert Neptune Histogram object to pluto Histogram.

    Args:
        hist_obj: Neptune Histogram object
        pluto_module: The pluto module

    Returns:
        pluto.Histogram
    """
    if hasattr(hist_obj, 'bin_edges') and hasattr(hist_obj, 'counts'):
        # Neptune Histogram has bin_edges and counts/densities
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

        # pluto.Histogram expects (counts, bins) format
        if counts:
            return pluto_module.Histogram(data=(counts, bin_edges), bins=None)

    # Fallback: return as-is and let pluto handle it
    return pluto_module.Histogram(data=hist_obj)


class NeptuneRunWrapper:
    """
    Wrapper around Neptune's Run class that dual-logs to pluto.

    This wrapper intercepts Neptune API calls and forwards them to both
    the original Neptune Run and to pluto. All pluto operations are wrapped
    in try-except blocks to ensure Neptune functionality is never impacted.
    """

    _neptune_run: Any
    _pluto_run: Optional[Any]
    _pluto: Optional[Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize both Neptune and pluto runs.

        Neptune args/kwargs are passed through unchanged.
        pluto is configured via environment variables.
        """
        # Check if Neptune logging is disabled
        self._neptune_disabled = os.environ.get(
            'DISABLE_NEPTUNE_LOGGING', ''
        ).lower() in ('true', '1', 'yes')

        if self._neptune_disabled:
            logger.info(
                'pluto.compat.neptune: DISABLE_NEPTUNE_LOGGING=true, '
                'skipping Neptune initialization. Only pluto logging will occur.'
            )
            self._neptune_run = None
            # Store args/kwargs for compatibility (e.g., get_run_url)
            self._neptune_args = args
            self._neptune_kwargs = kwargs
        else:
            # Use the saved original Run class (not the wrapper!)
            global _original_neptune_run
            try:
                if _original_neptune_run is None:
                    raise RuntimeError('Neptune monkeypatch not applied correctly')
                self._neptune_run = _original_neptune_run(*args, **kwargs)
            except Exception as e:
                # If Neptune itself fails, we can't do anything
                logger.error(
                    f'pluto.compat.neptune: Failed to initialize Neptune Run: {e}'
                )
                raise

        # Try to initialize pluto (silent failure)
        self._pluto_run = None
        self._pluto = _safe_import_pluto()

        if self._pluto is None:
            return

        try:
            pluto_config = _get_pluto_config_from_env()
            if pluto_config is None:
                logger.info(
                    'pluto.compat.neptune: PLUTO_PROJECT not set, '
                    'continuing with Neptune-only logging'
                )
                return

            # Extract Neptune parameters for mapping
            experiment_name = kwargs.get('experiment_name', 'neptune-migration')
            run_id = kwargs.get('run_id', None)

            # Build pluto init parameters
            pluto_init_kwargs = {
                'project': pluto_config['project'],
                'name': experiment_name
                if not run_id
                else f'{experiment_name}-{run_id}',
                'config': {},  # Will be populated by log_configs()
            }

            # Add custom URLs if provided
            settings = {}
            if 'url_app' in pluto_config:
                settings['url_app'] = pluto_config['url_app']
            if 'url_api' in pluto_config:
                settings['url_api'] = pluto_config['url_api']
            if 'url_ingest' in pluto_config:
                settings['url_ingest'] = pluto_config['url_ingest']

            # If API key provided via env var, pass it directly to settings
            if 'api_key' in pluto_config:
                settings['_auth'] = pluto_config['api_key']

            if settings:
                pluto_init_kwargs['settings'] = settings

            # Initialize pluto run
            self._pluto_run = self._pluto.init(**pluto_init_kwargs)
            logger.info(
                f'pluto.compat.neptune: Successfully initialized pluto run '
                f'for project={pluto_config["project"]}, '
                f'name={pluto_init_kwargs["name"]}'
            )

        except Exception as e:
            logger.warning(
                f'pluto.compat.neptune: Failed to initialize pluto run: {e}. '
                f'Continuing with Neptune-only logging.'
            )
            self._pluto_run = None
            # Clean up any partially initialized Pluto resources
            self._cleanup_pluto_state()

    def _cleanup_pluto_state(self):
        """Clean up any lingering Pluto state after initialization failure."""
        if not self._pluto:
            return
        try:
            import pluto

            if hasattr(pluto, 'ops') and pluto.ops:
                for op in pluto.ops[:]:
                    try:
                        op.finish()
                    except Exception as e:
                        logger.debug(f'Error finishing op during cleanup: {e}')
        except Exception as e:
            logger.debug(f'Error during pluto state cleanup: {e}')

    def log_metrics(self, data: Dict[str, float], step: int, timestamp=None, **kwargs):
        """
        Log metrics to both Neptune and pluto.

        Neptune's explicit step is passed through to pluto to maintain alignment.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.log_metrics(
                data=data, step=step, timestamp=timestamp, **kwargs
            )

        # Try to log to pluto with the same step value
        if self._pluto_run:
            try:
                # Pass Neptune's explicit step to pluto to maintain alignment
                self._pluto_run.log(data, step=step)
            except Exception as e:
                logger.debug(
                    f'pluto.compat.neptune: Failed to log metrics to pluto: {e}'
                )

        return result

    def log_configs(self, data: Dict[str, Any], **kwargs):
        """
        Log configuration/hyperparameters to both Neptune and pluto.

        Config updates are synced to the pluto server via the config update endpoint.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.log_configs(data=data, **kwargs)

        # Try to log to pluto
        if self._pluto_run:
            try:
                # Update pluto's config locally
                if hasattr(self._pluto_run, 'config'):
                    if self._pluto_run.config is None:
                        self._pluto_run.config = {}
                    self._pluto_run.config.update(data)
                # Sync to server
                if hasattr(self._pluto_run, '_iface') and self._pluto_run._iface:
                    self._pluto_run._iface._update_config(data)
            except Exception as e:
                logger.debug(
                    f'pluto.compat.neptune: Failed to log configs to pluto: {e}'
                )

        return result

    def assign_files(self, files: Dict[str, Any], **kwargs):
        """
        Assign files (single values) to both Neptune and pluto.

        Converts Neptune File objects to appropriate pluto types.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.assign_files(files=files, **kwargs)

        # Try to log to pluto
        if self._pluto_run and self._pluto:
            try:
                pluto_files = {}
                for key, file_obj in files.items():
                    try:
                        pluto_file = _convert_neptune_file_to_pluto(
                            file_obj, self._pluto
                        )
                        pluto_files[key] = pluto_file
                        pluto_type = type(pluto_file).__name__
                        logger.info(
                            f'pluto.compat.neptune: Converted file {key} '
                            f'to {pluto_type}'
                        )
                    except Exception as e:
                        logger.warning(
                            f'pluto.compat.neptune: Failed to convert file {key}: {e}'
                        )

                if pluto_files:
                    self._pluto_run.log(pluto_files)
                    logger.info(
                        f'pluto.compat.neptune: Logged {len(pluto_files)} files '
                        f'to pluto'
                    )
            except Exception as e:
                logger.warning(
                    f'pluto.compat.neptune: Failed to assign files to pluto: {e}'
                )

        return result

    def log_files(self, files: Dict[str, Any], step: int, timestamp=None, **kwargs):
        """
        Log files as a series to both Neptune and pluto.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.log_files(
                files=files, step=step, timestamp=timestamp, **kwargs
            )

        # Try to log to pluto
        if self._pluto_run and self._pluto:
            try:
                pluto_files = {}
                for key, file_obj in files.items():
                    try:
                        pluto_file = _convert_neptune_file_to_pluto(
                            file_obj, self._pluto
                        )
                        pluto_files[key] = pluto_file
                        pluto_type = type(pluto_file).__name__
                        logger.info(
                            f'pluto.compat.neptune: Converted {key} at step '
                            f'{step} to {pluto_type}'
                        )
                    except Exception as e:
                        logger.warning(
                            f'pluto.compat.neptune: Failed to convert file {key}: {e}'
                        )

                if pluto_files:
                    self._pluto_run.log(pluto_files, step=step)
                    logger.info(
                        f'pluto.compat.neptune: Logged {len(pluto_files)} files '
                        f'to pluto at step {step}'
                    )
            except Exception as e:
                logger.warning(
                    f'pluto.compat.neptune: Failed to log files to pluto: {e}'
                )

        return result

    def log_histograms(
        self, histograms: Dict[str, Any], step: int, timestamp=None, **kwargs
    ):
        """
        Log histograms to both Neptune and pluto.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.log_histograms(
                histograms=histograms, step=step, timestamp=timestamp, **kwargs
            )

        # Try to log to pluto
        if self._pluto_run and self._pluto:
            try:
                pluto_histograms = {}
                for key, hist_obj in histograms.items():
                    try:
                        pluto_hist = _convert_neptune_histogram_to_pluto(
                            hist_obj, self._pluto
                        )
                        pluto_histograms[key] = pluto_hist
                    except Exception as e:
                        logger.debug(
                            f'pluto.compat.neptune: Failed to convert '
                            f'histogram {key}: {e}'
                        )

                if pluto_histograms:
                    self._pluto_run.log(pluto_histograms, step=step)
            except Exception as e:
                logger.debug(
                    f'pluto.compat.neptune: Failed to log histograms to pluto: {e}'
                )

        return result

    def add_tags(self, tags: List[str], **kwargs):
        """
        Add tags to Neptune run.

        pluto now has native tag support, so we use the native API.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.add_tags(tags=tags, **kwargs)

        # Add to pluto using native tags API
        if self._pluto_run:
            try:
                self._pluto_run.add_tags(tags)
            except Exception as e:
                logger.debug(f'pluto.compat.neptune: Failed to add tags to pluto: {e}')

        return result

    def remove_tags(self, tags: List[str], **kwargs):
        """Remove tags from Neptune run."""
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.remove_tags(tags=tags, **kwargs)

        # Remove from pluto using native tags API
        if self._pluto_run:
            try:
                self._pluto_run.remove_tags(tags)
            except Exception as e:
                logger.debug(
                    f'pluto.compat.neptune: Failed to remove tags from pluto: {e}'
                )

        return result

    def close(self, **kwargs):
        """
        Close both Neptune and pluto runs.
        """
        # Close pluto first (less critical)
        if self._pluto_run:
            try:
                self._pluto_run.finish()
            except Exception as e:
                logger.warning(f'pluto.compat.neptune: Failed to close pluto run: {e}')

        # Close Neptune (unless disabled)
        if not self._neptune_disabled:
            return self._neptune_run.close(**kwargs)
        return None

    def terminate(self, **kwargs):
        """Terminate both runs immediately."""
        if self._pluto_run:
            try:
                self._pluto_run.finish()
            except Exception as e:
                logger.debug(
                    f'pluto.compat.neptune: Failed to terminate pluto run: {e}'
                )

        if not self._neptune_disabled:
            return self._neptune_run.terminate(**kwargs)
        return None

    def wait_for_submission(self, **kwargs):
        """Wait for Neptune submission (pluto not applicable)."""
        if not self._neptune_disabled:
            return self._neptune_run.wait_for_submission(**kwargs)
        return None

    def wait_for_processing(self, **kwargs):
        """Wait for Neptune processing (pluto not applicable)."""
        if not self._neptune_disabled:
            return self._neptune_run.wait_for_processing(**kwargs)
        return None

    def get_run_url(self):
        """Get Neptune run URL."""
        if not self._neptune_disabled:
            return self._neptune_run.get_run_url()
        # Return placeholder when Neptune is disabled
        return 'neptune://disabled'

    def get_experiment_url(self):
        """Get Neptune experiment URL."""
        if not self._neptune_disabled:
            return self._neptune_run.get_experiment_url()
        # Return placeholder when Neptune is disabled
        return 'neptune://disabled'

    def log_string_series(
        self, data: Dict[str, str], step: int, timestamp=None, **kwargs
    ):
        """Log string series to Neptune (pluto doesn't support this directly)."""
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.log_string_series(
                data=data, step=step, timestamp=timestamp, **kwargs
            )

            # pluto doesn't have string series support, skip silently
            logger.debug(
                'pluto.compat.neptune: String series not supported in pluto, '
                'logged to Neptune only'
            )

        return result

    @staticmethod
    def _flatten_dict(
        d: Dict[str, Any], parent_key: str = '', sep: str = '/'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for logging."""
        items: List[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    NeptuneRunWrapper._flatten_dict(v, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)

    def __getattr__(self, name):
        """
        Forward any unknown attributes/methods to the original Neptune run.

        This ensures 100% API compatibility even for methods we haven't wrapped.
        """
        return getattr(self._neptune_run, name)

    def __enter__(self):
        """Support context manager protocol."""
        if not self._neptune_disabled:
            self._neptune_run.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager protocol."""
        if self._pluto_run:
            try:
                self._pluto_run.finish()
            except Exception as e:
                logger.warning(
                    f'pluto.compat.neptune: Failed to close pluto run on exit: {e}'
                )

        if self._neptune_disabled:
            return False

        # Wait for processing with verbose=False before exit to prevent
        # logging errors when pytest or other tools capture stdout/stderr
        try:
            self._neptune_run.wait_for_processing(verbose=False)
        except Exception:
            pass  # Ignore errors during wait, __exit__ will handle cleanup
        return self._neptune_run.__exit__(exc_type, exc_val, exc_tb)


def _apply_monkeypatch():
    """
    Apply the monkeypatch to neptune_scale.Run.

    This function is called automatically when this module is imported.
    """
    global _original_neptune_run, _patch_applied

    if _patch_applied:
        logger.debug('pluto.compat.neptune: Monkeypatch already applied')
        return

    try:
        import neptune_scale

        # Save original Run class
        _original_neptune_run = neptune_scale.Run

        # Replace with our wrapper
        neptune_scale.Run = NeptuneRunWrapper

        _patch_applied = True
        logger.info(
            'pluto.compat.neptune: Monkeypatch applied successfully. '
            'Neptune API calls will now dual-log to pluto (if configured).'
        )

    except ImportError:
        logger.warning(
            'pluto.compat.neptune: neptune-scale not installed, monkeypatch not applied'
        )
    except Exception as e:
        logger.error(f'pluto.compat.neptune: Failed to apply monkeypatch: {e}')


def restore_neptune():
    """
    Restore the original Neptune Run class (for testing).

    This reverses the monkeypatch.
    """
    global _original_neptune_run, _patch_applied

    if not _patch_applied:
        return

    try:
        import neptune_scale

        if _original_neptune_run:
            neptune_scale.Run = _original_neptune_run
            _patch_applied = False
            logger.info('pluto.compat.neptune: Monkeypatch restored')
    except Exception as e:
        logger.error(f'pluto.compat.neptune: Failed to restore monkeypatch: {e}')


# Apply monkeypatch on module import
_apply_monkeypatch()
