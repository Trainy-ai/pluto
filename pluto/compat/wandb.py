"""
Wandb-to-Pluto compatibility layer for seamless dual-logging.

This module monkey-patches wandb.init() so that every wandb Run also logs
to Pluto. It can be activated in two ways:

1. Automatic (zero code changes): Set PLUTO_PROJECT + PLUTO_API_KEY env vars
   and pip install pluto-ml. The .pth file triggers the import hook which
   calls apply_wandb_patches().

2. Explicit import: `import pluto.compat.wandb` at the top of your script.
   This patches wandb directly (like the Neptune compat layer).

Configuration:
    Required environment variables:
    - PLUTO_PROJECT: Pluto project name
    - PLUTO_API_KEY: Pluto API key

    Optional:
    - PLUTO_URL_APP: Pluto app URL (for self-hosted)
    - PLUTO_URL_API: Pluto API URL (for self-hosted)
    - PLUTO_URL_INGEST: Pluto ingest URL (for self-hosted)
    - DISABLE_WANDB_LOGGING=true: Skip real wandb entirely, Pluto-only mode

Hard Requirements:
    - MUST NOT break existing wandb functionality under ANY condition
    - If Pluto is down/misconfigured, silently continue with wandb only
    - Zero impact on wandb's behavior, return values, or exceptions
"""

import atexit
import logging
import os
import threading
from typing import Any, Dict

from ._utils import (
    get_pluto_config_from_env as _get_pluto_config_from_env,
)
from ._utils import (
    safe_import_pluto as _safe_import_pluto,
)

logger = logging.getLogger(__name__)

_original_wandb_init = None
_original_wandb_log = None
_original_wandb_finish = None
_patch_applied = False

# Maps wandb run IDs to Pluto numeric run IDs.
# Populated when pluto.init() succeeds in patched_init, used by fork_from.
_wandb_to_pluto_run_ids: Dict[str, int] = {}


class WandbRunWrapper:
    """
    Wrapper around wandb.Run that dual-logs to Pluto.

    Intercepts key wandb Run methods and forwards them to both the original
    wandb Run and to a Pluto run. All Pluto operations are wrapped in
    try-except blocks to ensure wandb functionality is never impacted.
    """

    # Single-GPU cleanup timeout — short, since Pluto's own finish() drains
    # the sync process synchronously in non-distributed mode.
    _PLUTO_CLEANUP_TIMEOUT_SECONDS: float = 5.0

    # Multi-rank cleanup timeout — longer, because in distributed mode we
    # explicitly drain the sync manager before calling pluto.finish(), which
    # Pluto's own finish() would skip to avoid collective-op deadlocks.
    _PLUTO_DISTRIBUTED_CLEANUP_TIMEOUT_SECONDS: float = 30.0

    def __init__(self, wandb_run, pluto_run, pluto_module, wandb_disabled=False):
        self._wandb_run = wandb_run
        self._pluto_run = pluto_run
        self._pluto = pluto_module
        self._wandb_disabled = wandb_disabled
        self._fallback_step = 0  # Used when wandb is disabled (_step won't increment)
        self._closed = False
        self._close_lock = threading.Lock()

        if self._pluto_run:
            atexit.register(self._atexit_cleanup_pluto)

    def _atexit_cleanup_pluto(self) -> None:
        """Atexit handler for Pluto cleanup."""
        self._finish_pluto_with_timeout(timeout=self._get_cleanup_timeout())

    def _get_cleanup_timeout(self) -> float:
        """
        Return the cleanup timeout, longer in distributed mode.

        In distributed mode (DDP/FSDP), Pluto's finish() skips the drain
        to avoid collective-op deadlocks. We compensate here by running
        an explicit drain before finish(), which needs a longer timeout.
        """
        if _is_torch_distributed():
            return self._PLUTO_DISTRIBUTED_CLEANUP_TIMEOUT_SECONDS
        return self._PLUTO_CLEANUP_TIMEOUT_SECONDS

    def _finish_pluto_with_timeout(self, timeout: float) -> None:
        """
        Finish the Pluto run with a timeout to prevent blocking.

        In distributed mode, explicitly drains the sync manager before
        calling finish(), because Pluto's own finish() would skip the
        drain (to avoid collective-op deadlocks in normal Pluto usage).
        In single-GPU mode, Pluto's finish() handles the drain itself.
        """
        with self._close_lock:
            if self._pluto_run is None:
                return
            pluto_run = self._pluto_run
            self._pluto_run = None

        is_distributed = _is_torch_distributed()
        done_event = threading.Event()

        def _do_finish():
            try:
                # In distributed mode, explicitly drain the sync manager
                # BEFORE finish() so we don't lose tail records. This
                # bypasses Pluto's "is_distributed → don't wait" logic.
                # Safe to wait here because training is already done and
                # no more collective ops will happen on this process.
                if is_distributed and hasattr(pluto_run, '_sync_manager'):
                    sync_mgr = pluto_run._sync_manager
                    if sync_mgr is not None:
                        try:
                            sync_mgr.stop(timeout=timeout - 2.0, wait=True)
                        except Exception as e:
                            logger.debug(f'pluto.compat.wandb: sync drain failed: {e}')
                pluto_run.finish()
            except Exception:
                pass
            finally:
                done_event.set()

        thread = threading.Thread(target=_do_finish, daemon=False)
        thread.start()
        completed = done_event.wait(timeout=timeout)
        if completed:
            thread.join(timeout=1.0)
        else:
            logger.debug(f'pluto.compat.wandb: Pluto finish timed out after {timeout}s')

    def log(self, data: Dict[str, Any], step=None, commit=None, **kwargs):
        """Log metrics to both wandb and Pluto."""
        # Determine the step to use for Pluto.
        # When step is explicit, use it. Otherwise:
        # - Normal mode: read wandb's _step before log() increments it
        # - Disabled mode: wandb._step never increments, use our own counter
        if step is not None:
            actual_step = step
        elif self._wandb_disabled:
            actual_step = self._fallback_step
            self._fallback_step += 1
        else:
            actual_step = getattr(self._wandb_run, '_step', None)

        result = self._wandb_run.log(data, step=step, commit=commit, **kwargs)

        if self._pluto_run:
            try:
                # Separate numeric values from rich data types
                pluto_data = {}
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        pluto_data[key] = value
                    elif _is_torch_tensor_scalar(value):
                        pluto_data[key] = value.item()
                    else:
                        # Try to convert wandb data types to pluto equivalents
                        converted = _convert_wandb_to_pluto(key, value, self._pluto)
                        if converted is not None:
                            pluto_data[key] = converted

                if pluto_data:
                    log_kwargs = {}
                    if actual_step is not None:
                        log_kwargs['step'] = actual_step
                    self._pluto_run.log(pluto_data, **log_kwargs)
            except Exception as e:
                logger.debug(f'pluto.compat.wandb: Failed to log metrics to Pluto: {e}')

        return result

    def finish(self, exit_code=None, quiet=None):
        """Finish both wandb and Pluto runs."""
        with self._close_lock:
            if self._closed:
                return self._wandb_run.finish(exit_code=exit_code, quiet=quiet)
            self._closed = True

        # Finish Pluto first (non-blocking, bounded timeout)
        self._finish_pluto_with_timeout(timeout=self._get_cleanup_timeout())

        # Finish wandb (critical path)
        return self._wandb_run.finish(exit_code=exit_code, quiet=quiet)

    def define_metric(self, *args, **kwargs):
        """Forward to wandb (no Pluto equivalent)."""
        return self._wandb_run.define_metric(*args, **kwargs)

    def watch(self, *args, **kwargs):
        """Forward to wandb (Pluto watch is separate)."""
        return self._wandb_run.watch(*args, **kwargs)

    def unwatch(self, *args, **kwargs):
        """Forward to wandb."""
        return self._wandb_run.unwatch(*args, **kwargs)

    def save(self, *args, **kwargs):
        """Forward to wandb."""
        return self._wandb_run.save(*args, **kwargs)

    def alert(self, title, text, level=None, wait_duration=None):
        """Alert on both wandb and Pluto."""
        result = self._wandb_run.alert(
            title=title, text=text, level=level, wait_duration=wait_duration
        )

        if self._pluto_run:
            try:
                self._pluto_run.alert(title=title, text=text)
            except Exception as e:
                logger.debug(f'pluto.compat.wandb: Failed to send alert to Pluto: {e}')

        return result

    @property
    def config(self):
        return self._wandb_run.config

    @config.setter
    def config(self, value):
        self._wandb_run.config = value

    @property
    def summary(self):
        return self._wandb_run.summary

    @summary.setter
    def summary(self, value):
        self._wandb_run.summary = value

    @property
    def name(self):
        return self._wandb_run.name

    @name.setter
    def name(self, value):
        self._wandb_run.name = value

    @property
    def id(self):
        return self._wandb_run.id

    @property
    def dir(self):
        return self._wandb_run.dir

    @property
    def tags(self):
        return self._wandb_run.tags

    @tags.setter
    def tags(self, value):
        self._wandb_run.tags = value
        if self._pluto_run:
            try:
                self._pluto_run.add_tags(list(value))
            except Exception as e:
                logger.debug(f'pluto.compat.wandb: Failed to sync tags to Pluto: {e}')

    def get_url(self):
        return self._wandb_run.get_url()

    def get_project_url(self):
        return self._wandb_run.get_project_url()

    def __getattr__(self, name):
        """Forward any unknown attributes to the real wandb run."""
        return getattr(self._wandb_run, name)

    def __enter__(self):
        self._wandb_run.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._close_lock:
            self._closed = True
        self._finish_pluto_with_timeout(timeout=self._get_cleanup_timeout())
        return self._wandb_run.__exit__(exc_type, exc_val, exc_tb)


def _parse_wandb_fork_from(fork_from):
    """Parse wandb's fork_from string into (run_id, step).

    wandb format: "{run_id}?_step={step}"
    Returns (run_id, step) tuple or None if unparseable.
    """
    try:
        if '?_step=' in fork_from:
            run_id, step_str = fork_from.split('?_step=', 1)
            return run_id.strip(), int(step_str)
    except (ValueError, AttributeError):
        pass
    logger.debug(f'pluto.compat.wandb: Could not parse fork_from: {fork_from}')
    return None


def _resolve_wandb_to_pluto_run(wandb_run_id, project):
    """Look up a Pluto run ID from a wandb run ID.

    First checks the in-process mapping (populated when pluto.init succeeds).
    Falls back to searching the Pluto API by run name.

    Returns the Pluto numeric run ID, or None if not found.
    """
    # Fast path: in-process mapping from earlier wandb.init() calls
    if wandb_run_id in _wandb_to_pluto_run_ids:
        return _wandb_to_pluto_run_ids[wandb_run_id]

    # Slow path: search Pluto API
    try:
        import pluto.query as pq

        runs = pq.list_runs(project, search=wandb_run_id, limit=10)
        for run in runs:
            if run.get('id'):
                return run['id']
    except Exception as e:
        logger.debug(
            f'pluto.compat.wandb: Failed to resolve wandb run {wandb_run_id}: {e}'
        )
    return None


def _is_torch_tensor_scalar(value):
    """Check if value is a scalar torch tensor."""
    try:
        import torch

        return isinstance(value, torch.Tensor) and value.dim() == 0
    except ImportError:
        return False


def _is_torch_distributed() -> bool:
    """
    Check if we're running in a torch.distributed environment.

    Used to decide whether to explicitly drain the Pluto sync manager
    before finish() (needed in distributed mode because Pluto skips the
    drain there to avoid collective-op deadlocks).
    """
    # Cheap env-var check first — avoids importing torch if it's not used.
    if (
        os.environ.get('RANK') is not None
        or os.environ.get('LOCAL_RANK') is not None
        or os.environ.get('WORLD_SIZE') is not None
    ):
        return True
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except ImportError:
        return False


def _convert_wandb_to_pluto(key, value, pluto_module):
    """
    Convert wandb data types to Pluto equivalents.

    Returns the converted value, or None if conversion is not possible.
    """
    try:
        type_name = type(value).__name__

        if type_name == 'Image':
            # wandb.Image -> pluto.Image. wandb.Image has both `.image`
            # (public) and `._image` (legacy alias) pointing to the same
            # PIL.Image (typically a PIL.PngImagePlugin.PngImageFile
            # subclass, not the base PIL.Image.Image class).
            #
            # pluto.Image's class-name check uses startswith('PIL.Image.Image')
            # which does NOT match subclasses, so we can't pass the PIL
            # object directly. Instead, use the file path — wandb.Image
            # always writes to _path on construction.
            if getattr(value, '_path', None):
                return pluto_module.Image(value._path)
            # Fallback: convert PIL to numpy (which pluto.Image handles)
            pil_img = getattr(value, 'image', None) or getattr(value, '_image', None)
            if pil_img is not None:
                try:
                    import numpy as np

                    return pluto_module.Image(np.asarray(pil_img))
                except Exception:
                    return None
            return None

        if type_name == 'Histogram':
            # wandb.Histogram -> pluto.Histogram. wandb stores the
            # pre-binned counts as `.histogram` (list) and edges as
            # `.bins` (list of len N+1). pluto.Histogram takes a
            # (counts, bin_edges) tuple.
            if hasattr(value, 'histogram') and hasattr(value, 'bins'):
                return pluto_module.Histogram(
                    data=(list(value.histogram), list(value.bins)), bins=None
                )
            return None

        if type_name == 'Audio':
            # wandb.Audio always writes to _path on construction
            # (whether from numpy, file path, or bytes).
            if getattr(value, '_path', None):
                return pluto_module.Audio(value._path)
            return None

        if type_name == 'Video':
            # wandb.Video always writes to _path on construction (after
            # encoding). This can take a few seconds for numpy input.
            if getattr(value, '_path', None):
                return pluto_module.Video(value._path)
            return None

        if type_name == 'Table':
            # wandb.Table -> pluto.Table (best-effort — pluto.Table has
            # a different API, this may need adjustment).
            if hasattr(value, 'data') and hasattr(value, 'columns'):
                return pluto_module.Table(data=value.data, columns=value.columns)
            return None

    except Exception as e:
        logger.debug(f'pluto.compat.wandb: Failed to convert {key} ({type_name}): {e}')

    return None


def _make_patched_init(original_init, wandb_module):
    """
    Create a patched wandb.init that wraps the returned Run with WandbRunWrapper.
    """

    def patched_init(*args, **kwargs):
        wandb_disabled = os.environ.get('DISABLE_WANDB_LOGGING', '').lower() in (
            'true',
            '1',
            'yes',
        )

        if wandb_disabled:
            # Wandb-disabled mode: only log to Pluto
            # Set wandb to disabled mode so it doesn't actually connect
            kwargs['mode'] = 'disabled'

        # Call real wandb.init
        wandb_run = original_init(*args, **kwargs)

        # Try to initialize Pluto
        pluto = _safe_import_pluto()
        if pluto is None:
            return wandb_run

        pluto_config = _get_pluto_config_from_env()
        if pluto_config is None:
            logger.info(
                'pluto.compat.wandb: PLUTO_PROJECT not set, '
                'continuing with wandb-only logging'
            )
            return wandb_run

        try:
            # Build Pluto init kwargs from wandb args
            # NOTE: Everything below is Pluto-only. wandb_run is already
            # fully initialized and functional. If anything here fails
            # or hangs, we return wandb_run unmodified (see except block).
            #
            # Resolution order for each field:
            #   1. Explicit kwarg to wandb.init(...)
            #   2. Attribute on the resolved wandb_run (wandb reads WANDB_*
            #      env vars during init and populates these)
            #   3. WANDB_* env var direct fallback (in case wandb_run
            #      attribute is missing)
            # See https://docs.wandb.ai/guides/track/environment-variables
            name = (
                kwargs.get('name')
                or getattr(wandb_run, 'name', None)
                or os.environ.get('WANDB_NAME')
            )
            wandb_config = kwargs.get('config') or getattr(wandb_run, 'config', None)
            tags = (
                kwargs.get('tags')
                or getattr(wandb_run, 'tags', None)
                or (
                    os.environ.get('WANDB_TAGS', '').split(',')
                    if os.environ.get('WANDB_TAGS')
                    else None
                )
            )
            # Wandb notes have no direct Pluto equivalent, but we can
            # stash them in the config so they're still visible.
            notes = (
                kwargs.get('notes')
                or getattr(wandb_run, 'notes', None)
                or os.environ.get('WANDB_NOTES')
            )
            # Always use the wandb run ID as Pluto's externalId so we can
            # look up the Pluto run from the wandb ID later (e.g. for forking).
            # PLUTO_RUN_ID env var takes precedence (for distributed training).
            wandb_run_id = getattr(wandb_run, 'id', None)
            run_id = os.environ.get('PLUTO_RUN_ID') or wandb_run_id

            pluto_settings = {
                'sync_process_enabled': True,
                'sync_process_shutdown_timeout': 3.0,
            }
            if 'url_app' in pluto_config:
                pluto_settings['url_app'] = pluto_config['url_app']
            if 'url_api' in pluto_config:
                pluto_settings['url_api'] = pluto_config['url_api']
            if 'url_ingest' in pluto_config:
                pluto_settings['url_ingest'] = pluto_config['url_ingest']
            if 'api_key' in pluto_config:
                pluto_settings['_auth'] = pluto_config['api_key']

            pluto_init_kwargs = {
                'project': pluto_config['project'],
                'name': name or 'wandb-migration',
                'settings': pluto_settings,
            }
            if wandb_config:
                pluto_init_kwargs['config'] = (
                    dict(wandb_config)
                    if not isinstance(wandb_config, dict)
                    else wandb_config
                )
            # Stash wandb notes in config since Pluto has no native
            # "notes" field on init. Users can still see it.
            if notes:
                pluto_init_kwargs.setdefault('config', {})
                pluto_init_kwargs['config']['_wandb_notes'] = notes
            if tags:
                pluto_init_kwargs['tags'] = list(tags)
            if run_id:
                pluto_init_kwargs['run_id'] = run_id

            # Handle wandb fork_from — translate to Pluto fork parameters
            fork_from = kwargs.get('fork_from')
            if fork_from:
                fork_info = _parse_wandb_fork_from(fork_from)
                if fork_info:
                    wandb_source_id, fork_step = fork_info
                    pluto_source_id = _resolve_wandb_to_pluto_run(
                        wandb_source_id, pluto_config['project']
                    )
                    if pluto_source_id is not None:
                        pluto_init_kwargs['fork_run_id'] = pluto_source_id
                        pluto_init_kwargs['fork_step'] = fork_step

            # Use a thread with timeout so a slow/unreachable Pluto server
            # never blocks the user's training script from starting.
            pluto_run = None
            init_error = {}

            def _do_pluto_init():
                nonlocal pluto_run
                try:
                    pluto_run = pluto.init(**pluto_init_kwargs)
                except Exception as e:
                    init_error['err'] = e

            init_thread = threading.Thread(target=_do_pluto_init, daemon=True)
            init_thread.start()
            init_thread.join(timeout=10.0)  # 10s max for Pluto init

            if init_thread.is_alive():
                logger.warning(
                    'pluto.compat.wandb: Pluto init timed out after 10s. '
                    'Continuing with wandb-only logging.'
                )
                return wandb_run

            if 'err' in init_error:
                raise init_error['err']

            if pluto_run is None:
                return wandb_run

            logger.info(
                f'pluto.compat.wandb: Successfully initialized Pluto run '
                f'for project={pluto_config["project"]}, name={name}'
            )

            # Store wandb→pluto ID mapping for fork resolution
            if wandb_run_id and hasattr(pluto_run, 'id') and pluto_run.id:
                _wandb_to_pluto_run_ids[wandb_run_id] = pluto_run.id

            # Wrap the wandb run
            wrapper = WandbRunWrapper(wandb_run, pluto_run, pluto, wandb_disabled)

            # Update the wandb module's global `run` reference
            wandb_module.run = wrapper

            # CRITICAL: wandb.init() overwrites wandb.log/wandb.finish with
            # bound methods from the Run instance, clobbering our patches.
            # We must re-patch them after init to point at the wrapper.
            # Tested against wandb 0.25.1 — if wandb changes this behavior,
            # the re-patch becomes unnecessary (but harmless).
            wandb_module.log = wrapper.log
            wandb_module.finish = wrapper.finish

            return wrapper

        except Exception as e:
            # Loud failure — users should know dual-logging is off
            _msg = (
                f'pluto.compat.wandb: DUAL-LOGGING DISABLED. Failed to '
                f'initialize Pluto run: {type(e).__name__}: {e}. '
                f'wandb will continue to work normally, but NO DATA will be '
                f'sent to Pluto. To fix, resolve the error above and retry.'
            )
            logger.error(_msg)
            # Also print to stderr so it shows up even if logging is not configured
            import sys

            print(f'[pluto.compat.wandb] {_msg}', file=sys.stderr, flush=True)
            return wandb_run

    return patched_init


def _make_patched_log(wandb_module):
    """
    Create a patched wandb.log that delegates to the current run's log method.

    This ensures that module-level wandb.log() calls go through the wrapper
    when a WandbRunWrapper is the active run.
    """
    original_log = wandb_module.log

    def patched_log(data, step=None, commit=None, **kwargs):
        current_run = wandb_module.run
        if isinstance(current_run, WandbRunWrapper):
            return current_run.log(data, step=step, commit=commit, **kwargs)
        return original_log(data, step=step, commit=commit, **kwargs)

    return patched_log


def _make_patched_finish(wandb_module):
    """
    Create a patched wandb.finish that delegates to the current run's finish.
    """
    original_finish = wandb_module.finish

    def patched_finish(exit_code=None, quiet=None):
        current_run = wandb_module.run
        if isinstance(current_run, WandbRunWrapper):
            return current_run.finish(exit_code=exit_code, quiet=quiet)
        return original_finish(exit_code=exit_code, quiet=quiet)

    return patched_finish


def apply_wandb_patches(wandb_module):
    """
    Apply dual-logging monkey-patches to the wandb module.

    This is the main entry point called by either:
    - The .pth import hook (_wandb_hook.py)
    - Direct import of this module (import pluto.compat.wandb)

    Args:
        wandb_module: The real wandb module to patch.
    """
    global _original_wandb_init, _original_wandb_log, _original_wandb_finish
    global _patch_applied

    if _patch_applied:
        logger.debug('pluto.compat.wandb: Patches already applied')
        return

    _original_wandb_init = wandb_module.init
    _original_wandb_log = wandb_module.log
    _original_wandb_finish = wandb_module.finish

    wandb_module.init = _make_patched_init(_original_wandb_init, wandb_module)
    wandb_module.log = _make_patched_log(wandb_module)
    wandb_module.finish = _make_patched_finish(wandb_module)

    _patch_applied = True
    logger.info(
        'pluto.compat.wandb: Patches applied. wandb.init/log/finish now dual-log '
        'to Pluto (when PLUTO_PROJECT is set).'
    )


def restore_wandb():
    """Restore the original wandb functions (for testing)."""
    global _original_wandb_init, _original_wandb_log, _original_wandb_finish
    global _patch_applied

    if not _patch_applied:
        return

    try:
        import wandb

        if _original_wandb_init:
            wandb.init = _original_wandb_init
        if _original_wandb_log:
            wandb.log = _original_wandb_log
        if _original_wandb_finish:
            wandb.finish = _original_wandb_finish
        _patch_applied = False
        logger.info('pluto.compat.wandb: Patches restored')
    except Exception as e:
        logger.error(f'pluto.compat.wandb: Failed to restore patches: {e}')


# When imported directly (import pluto.compat.wandb), auto-patch wandb
def _apply_on_import():
    try:
        import wandb

        apply_wandb_patches(wandb)
    except ImportError:
        logger.warning('pluto.compat.wandb: wandb not installed, patches not applied')


_apply_on_import()
