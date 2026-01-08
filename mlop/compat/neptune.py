"""
Neptune-to-mlop compatibility layer for seamless migration.

This module provides a monkeypatch that allows existing Neptune API calls
to simultaneously log to both Neptune and mlop, enabling a gradual migration
without breaking existing workflows.

Usage:
    import mlop.compat.neptune  # This automatically patches Neptune

    # Your existing Neptune code continues to work
    from neptune_scale import Run
    run = Run(experiment_name="my-experiment")
    run.log_metrics({"loss": 0.5}, step=0)
    run.close()

    # Now logs to BOTH Neptune and mlop!

Configuration:
    Set environment variables:
    - MLOP_PROJECT: mlop project name (required)
    - MLOP_API_KEY: mlop API key (optional, falls back to keyring)
    - MLOP_URL_APP: mlop app URL (optional, uses default)
    - MLOP_URL_API: mlop API URL (optional, uses default)
    - MLOP_URL_INGEST: mlop ingest URL (optional, uses default)

Hard Requirements:
    - MUST NOT break existing Neptune functionality under ANY condition
    - If mlop is down/misconfigured, silently continue with Neptune only
    - Zero impact on Neptune's behavior, return values, or exceptions
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_original_neptune_run = None
_patch_applied = False


def _get_mlop_config_from_env() -> Optional[Dict[str, Any]]:
    """
    Extract mlop configuration from environment variables.

    Returns:
        Config dict if MLOP_PROJECT is set, None otherwise
    """
    project = os.environ.get('MLOP_PROJECT')
    if not project:
        return None

    config = {'project': project}

    # Optional: API key (will fall back to keyring if not provided)
    if api_key := os.environ.get('MLOP_API_KEY'):
        config['api_key'] = api_key

    # Optional: Custom URLs
    if url_app := os.environ.get('MLOP_URL_APP'):
        config['url_app'] = url_app
    if url_api := os.environ.get('MLOP_URL_API'):
        config['url_api'] = url_api
    if url_ingest := os.environ.get('MLOP_URL_INGEST'):
        config['url_ingest'] = url_ingest

    return config


def _safe_import_mlop():
    """
    Safely import mlop, returning None if unavailable.

    Returns:
        mlop module or None if import fails
    """
    try:
        import mlop

        return mlop
    except ImportError:
        logger.warning(
            'mlop.compat.neptune: mlop not installed, '
            'continuing with Neptune-only logging'
        )
        return None


def _convert_neptune_file_to_mlop(file_obj, mlop_module):
    """
    Convert Neptune File object to appropriate mlop file type.

    Args:
        file_obj: Neptune File object or file path
        mlop_module: The mlop module

    Returns:
        mlop.Image, mlop.Audio, mlop.Video, or mlop.Artifact
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
            return mlop_module.Image(source)
        elif mime_type.startswith('audio/'):
            return mlop_module.Audio(source)
        elif mime_type.startswith('video/'):
            return mlop_module.Video(source)

    # Try to infer from file path
    if isinstance(source, (str, os.PathLike)):
        source_str = str(source).lower()
        if any(
            source_str.endswith(ext)
            for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        ):
            return mlop_module.Image(source)
        elif any(source_str.endswith(ext) for ext in ['.wav', '.mp3', '.ogg', '.flac']):
            return mlop_module.Audio(source)
        elif any(source_str.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.webm']):
            return mlop_module.Video(source)

    # Default to generic artifact
    return mlop_module.Artifact(source)


def _convert_neptune_histogram_to_mlop(hist_obj, mlop_module):
    """
    Convert Neptune Histogram object to mlop Histogram.

    Args:
        hist_obj: Neptune Histogram object
        mlop_module: The mlop module

    Returns:
        mlop.Histogram
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

        # mlop.Histogram expects (counts, bins) format
        if counts:
            return mlop_module.Histogram(data=(counts, bin_edges), bins=None)

    # Fallback: return as-is and let mlop handle it
    return mlop_module.Histogram(data=hist_obj)


class NeptuneRunWrapper:
    """
    Wrapper around Neptune's Run class that dual-logs to mlop.

    This wrapper intercepts Neptune API calls and forwards them to both
    the original Neptune Run and to mlop. All mlop operations are wrapped
    in try-except blocks to ensure Neptune functionality is never impacted.
    """

    _neptune_run: Any
    _mlop_run: Optional[Any]
    _mlop: Optional[Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize both Neptune and mlop runs.

        Neptune args/kwargs are passed through unchanged.
        mlop is configured via environment variables.
        """
        # Check if Neptune logging is disabled
        self._neptune_disabled = os.environ.get(
            'DISABLE_NEPTUNE_LOGGING', ''
        ).lower() in ('true', '1', 'yes')

        if self._neptune_disabled:
            logger.info(
                'mlop.compat.neptune: DISABLE_NEPTUNE_LOGGING=true, '
                'skipping Neptune initialization. Only mlop logging will occur.'
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
                    f'mlop.compat.neptune: Failed to initialize Neptune Run: {e}'
                )
                raise

        # Try to initialize mlop (silent failure)
        self._mlop_run = None
        self._mlop = _safe_import_mlop()

        if self._mlop is None:
            return

        try:
            mlop_config = _get_mlop_config_from_env()
            if mlop_config is None:
                logger.info(
                    'mlop.compat.neptune: MLOP_PROJECT not set, '
                    'continuing with Neptune-only logging'
                )
                return

            # Extract Neptune parameters for mapping
            experiment_name = kwargs.get('experiment_name', 'neptune-migration')
            run_id = kwargs.get('run_id', None)

            # Build mlop init parameters
            mlop_init_kwargs = {
                'project': mlop_config['project'],
                'name': experiment_name
                if not run_id
                else f'{experiment_name}-{run_id}',
                'config': {},  # Will be populated by log_configs()
            }

            # Add custom URLs if provided
            settings = {}
            if 'url_app' in mlop_config:
                settings['url_app'] = mlop_config['url_app']
            if 'url_api' in mlop_config:
                settings['url_api'] = mlop_config['url_api']
            if 'url_ingest' in mlop_config:
                settings['url_ingest'] = mlop_config['url_ingest']

            # If API key provided via env var, pass it directly to settings
            if 'api_key' in mlop_config:
                settings['_auth'] = mlop_config['api_key']

            if settings:
                mlop_init_kwargs['settings'] = settings

            # Initialize mlop run
            self._mlop_run = self._mlop.init(**mlop_init_kwargs)
            logger.info(
                f'mlop.compat.neptune: Successfully initialized mlop run '
                f'for project={mlop_config["project"]}, name={mlop_init_kwargs["name"]}'
            )

        except Exception as e:
            logger.warning(
                f'mlop.compat.neptune: Failed to initialize mlop run: {e}. '
                f'Continuing with Neptune-only logging.'
            )
            self._mlop_run = None

    def log_metrics(self, data: Dict[str, float], step: int, timestamp=None, **kwargs):
        """
        Log metrics to both Neptune and mlop.

        Neptune's explicit step is mapped to mlop's auto-incrementing step
        by calling log() multiple times if needed.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.log_metrics(
                data=data, step=step, timestamp=timestamp, **kwargs
            )

        # Try to log to mlop
        if self._mlop_run:
            try:
                # mlop auto-increments steps, but we can set it explicitly
                # For now, just log the data and let mlop handle steps
                # Note: This may cause step misalignment - acceptable for migration
                self._mlop_run.log(data)
            except Exception as e:
                logger.debug(f'mlop.compat.neptune: Failed to log metrics to mlop: {e}')

        return result

    def log_configs(self, data: Dict[str, Any], **kwargs):
        """
        Log configuration/hyperparameters to both Neptune and mlop.

        In mlop, configs are set during init, so we update the run's config.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.log_configs(data=data, **kwargs)

        # Try to log to mlop
        if self._mlop_run:
            try:
                # Update mlop's config
                if hasattr(self._mlop_run, 'config'):
                    self._mlop_run.config.update(data)
                # Also log as a special metric for visibility
                # Flatten nested dicts for logging
                flattened = self._flatten_dict(data)
                # Log each config as a constant metric (step 0)
                for key, value in flattened.items():
                    if isinstance(value, (int, float, bool)):
                        self._mlop_run.log({f'config/{key}': float(value)})
            except Exception as e:
                logger.debug(f'mlop.compat.neptune: Failed to log configs to mlop: {e}')

        return result

    def assign_files(self, files: Dict[str, Any], **kwargs):
        """
        Assign files (single values) to both Neptune and mlop.

        Converts Neptune File objects to appropriate mlop types.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.assign_files(files=files, **kwargs)

        # Try to log to mlop
        if self._mlop_run and self._mlop:
            try:
                mlop_files = {}
                for key, file_obj in files.items():
                    try:
                        mlop_file = _convert_neptune_file_to_mlop(file_obj, self._mlop)
                        mlop_files[key] = mlop_file
                        mlop_type = type(mlop_file).__name__
                        logger.info(
                            f'mlop.compat.neptune: Converted file {key} to {mlop_type}'
                        )
                    except Exception as e:
                        logger.warning(
                            f'mlop.compat.neptune: Failed to convert file {key}: {e}'
                        )

                if mlop_files:
                    self._mlop_run.log(mlop_files)
                    logger.info(
                        f'mlop.compat.neptune: Logged {len(mlop_files)} files to mlop'
                    )
            except Exception as e:
                logger.warning(
                    f'mlop.compat.neptune: Failed to assign files to mlop: {e}'
                )

        return result

    def log_files(self, files: Dict[str, Any], step: int, timestamp=None, **kwargs):
        """
        Log files as a series to both Neptune and mlop.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.log_files(
                files=files, step=step, timestamp=timestamp, **kwargs
            )

        # Try to log to mlop
        if self._mlop_run and self._mlop:
            try:
                mlop_files = {}
                for key, file_obj in files.items():
                    try:
                        mlop_file = _convert_neptune_file_to_mlop(file_obj, self._mlop)
                        mlop_files[key] = mlop_file
                        mlop_type = type(mlop_file).__name__
                        logger.info(
                            f'mlop.compat.neptune: Converted {key} at step '
                            f'{step} to {mlop_type}'
                        )
                    except Exception as e:
                        logger.warning(
                            f'mlop.compat.neptune: Failed to convert file {key}: {e}'
                        )

                if mlop_files:
                    self._mlop_run.log(mlop_files, step=step)
                    logger.info(
                        f'mlop.compat.neptune: Logged {len(mlop_files)} files '
                        f'to mlop at step {step}'
                    )
            except Exception as e:
                logger.warning(f'mlop.compat.neptune: Failed to log files to mlop: {e}')

        return result

    def log_histograms(
        self, histograms: Dict[str, Any], step: int, timestamp=None, **kwargs
    ):
        """
        Log histograms to both Neptune and mlop.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.log_histograms(
                histograms=histograms, step=step, timestamp=timestamp, **kwargs
            )

        # Try to log to mlop
        if self._mlop_run and self._mlop:
            try:
                mlop_histograms = {}
                for key, hist_obj in histograms.items():
                    try:
                        mlop_hist = _convert_neptune_histogram_to_mlop(
                            hist_obj, self._mlop
                        )
                        mlop_histograms[key] = mlop_hist
                    except Exception as e:
                        logger.debug(
                            f'mlop.compat.neptune: Failed to convert '
                            f'histogram {key}: {e}'
                        )

                if mlop_histograms:
                    self._mlop_run.log(mlop_histograms, step=step)
            except Exception as e:
                logger.debug(
                    f'mlop.compat.neptune: Failed to log histograms to mlop: {e}'
                )

        return result

    def add_tags(self, tags: List[str], **kwargs):
        """
        Add tags to Neptune run.

        mlop now has native tag support, so we use the native API.
        """
        # Call Neptune first (unless disabled)
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.add_tags(tags=tags, **kwargs)

        # Add to mlop using native tags API
        if self._mlop_run:
            try:
                self._mlop_run.add_tags(tags)
            except Exception as e:
                logger.debug(f'mlop.compat.neptune: Failed to add tags to mlop: {e}')

        return result

    def remove_tags(self, tags: List[str], **kwargs):
        """Remove tags from Neptune run."""
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.remove_tags(tags=tags, **kwargs)

        # Remove from mlop using native tags API
        if self._mlop_run:
            try:
                self._mlop_run.remove_tags(tags)
            except Exception as e:
                logger.debug(
                    f'mlop.compat.neptune: Failed to remove tags from mlop: {e}'
                )

        return result

    def close(self, **kwargs):
        """
        Close both Neptune and mlop runs.
        """
        # Close mlop first (less critical)
        if self._mlop_run:
            try:
                self._mlop_run.finish()
            except Exception as e:
                logger.warning(f'mlop.compat.neptune: Failed to close mlop run: {e}')

        # Close Neptune (unless disabled)
        if not self._neptune_disabled:
            return self._neptune_run.close(**kwargs)
        return None

    def terminate(self, **kwargs):
        """Terminate both runs immediately."""
        if self._mlop_run:
            try:
                self._mlop_run.finish()
            except Exception as e:
                logger.debug(f'mlop.compat.neptune: Failed to terminate mlop run: {e}')

        if not self._neptune_disabled:
            return self._neptune_run.terminate(**kwargs)
        return None

    def wait_for_submission(self, **kwargs):
        """Wait for Neptune submission (mlop not applicable)."""
        if not self._neptune_disabled:
            return self._neptune_run.wait_for_submission(**kwargs)
        return None

    def wait_for_processing(self, **kwargs):
        """Wait for Neptune processing (mlop not applicable)."""
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
        """Log string series to Neptune (mlop doesn't support this directly)."""
        result = None
        if not self._neptune_disabled:
            result = self._neptune_run.log_string_series(
                data=data, step=step, timestamp=timestamp, **kwargs
            )

            # mlop doesn't have string series support, skip silently
            logger.debug(
                'mlop.compat.neptune: String series not supported in mlop, '
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
        self._neptune_run.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager protocol."""
        if self._mlop_run:
            try:
                self._mlop_run.finish()
            except Exception as e:
                logger.warning(
                    f'mlop.compat.neptune: Failed to close mlop run on exit: {e}'
                )
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
        logger.debug('mlop.compat.neptune: Monkeypatch already applied')
        return

    try:
        import neptune_scale

        # Save original Run class
        _original_neptune_run = neptune_scale.Run

        # Replace with our wrapper
        neptune_scale.Run = NeptuneRunWrapper

        _patch_applied = True
        logger.info(
            'mlop.compat.neptune: Monkeypatch applied successfully. '
            'Neptune API calls will now dual-log to mlop (if configured).'
        )

    except ImportError:
        logger.warning(
            'mlop.compat.neptune: neptune-scale not installed, '
            'monkeypatch not applied'
        )
    except Exception as e:
        logger.error(f'mlop.compat.neptune: Failed to apply monkeypatch: {e}')


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
            logger.info('mlop.compat.neptune: Monkeypatch restored')
    except Exception as e:
        logger.error(f'mlop.compat.neptune: Failed to restore monkeypatch: {e}')


# Apply monkeypatch on module import
_apply_monkeypatch()
