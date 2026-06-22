"""
Wandb-to-Pluto compatibility layer for seamless dual-logging.

This module monkey-patches wandb.init() so that every wandb Run also logs
to Pluto. It can be activated in two ways:

1. Automatic (zero code changes): pip install pluto-ml. The .pth file
   triggers the import hook which calls apply_wandb_patches() once Pluto
   credentials are available (see Configuration below).

2. Explicit import: `import pluto.compat.wandb` at the top of your script.
   This patches wandb directly (like the Neptune compat layer).

Configuration:
    Authentication (one of the following):
    - Run `pluto login` to store a token in the system keyring.
    - Set PLUTO_API_KEY (Pluto API token).
    - In DISABLE_WANDB_LOGGING=true mode only, WANDB_API_KEY may be
      reused as the Pluto token (migration shortcut).

    Project name (one of the following, checked in order):
    - PLUTO_PROJECT env var
    - the `project` kwarg passed to wandb.init()
    - WANDB_PROJECT env var
    - the project attribute on the resolved wandb run
    If you already pass project= to wandb.init() (or via a framework
    wrapper like Lightning's WandbLogger) or have WANDB_PROJECT set,
    you don't need to set PLUTO_PROJECT separately.

    Optional:
    - PLUTO_URL_APP: Pluto app URL (for self-hosted)
    - PLUTO_URL_API: Pluto API URL (for self-hosted)
    - PLUTO_URL_INGEST: Pluto ingest URL (for self-hosted)
    - DISABLE_WANDB_LOGGING=true: Skip real wandb entirely, Pluto-only mode

Hard Requirements:
    - MUST NOT break existing wandb functionality under ANY condition
    - If Pluto is down/misconfigured, log a warning and continue with
      wandb only — never raise.
    - Zero impact on wandb's behavior, return values, or exceptions
"""

import atexit
import copy
import json
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

# Distinct from None so config dedup can tell "never logged" from "logged None".
_MISSING = object()

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
        # Keys we've already warned about being unforwardable to Pluto, so a
        # value logged every step warns once rather than spamming the logs.
        self._unforwardable_warned: set = set()
        # Last config values we synced to Pluto, keyed by log key. Lets us skip
        # redundant update_config() calls when a str/bool/config value is logged
        # unchanged every step (a common pattern: phase/status/checkpoint paths).
        self._last_logged_config: Dict[str, Any] = {}
        # (context, exc-type) pairs already surfaced at error, so a failure that
        # recurs every step is shouted once then drops to debug (see
        # _log_pluto_failure).
        self._pluto_failure_warned: set = set()

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

    def _log_pluto_failure(self, context: str, exc: Exception) -> None:
        """Report a swallowed Pluto-side failure at a severity matching its kind.

        The shim must never let a Pluto problem break wandb logging, so callers
        catch broadly. But severity should depend on the cause:

        - Transient/expected network errors (httpx) are noisy and already
          retried by the sync layer -> keep them at debug.
        - Anything else (e.g. an OSError from an oversized staging filename, or
          a bug) means the user's data was silently dropped and they'd otherwise
          be left guessing -> surface it at error, once per (context, type) so a
          per-step failure is shouted once then drops to debug.
        """
        import httpx

        if isinstance(exc, httpx.HTTPError):
            logger.debug(f'pluto.compat.wandb: {context}: {exc}')
            return

        detail = f'{context}: {type(exc).__name__}: {exc}'
        key = (context, type(exc).__name__)
        if key in self._pluto_failure_warned:
            logger.debug(f'pluto.compat.wandb: {detail}')
            return
        self._pluto_failure_warned.add(key)
        logger.error(
            f'pluto.compat.wandb: {detail} '
            '(Pluto copy dropped; wandb unaffected; further occurrences at debug)'
        )
        # Telemetry: report the failure once per (context, type) so we see these
        # in the wild. Self-gates on PLUTO_DISABLE_TELEMETRY / CI; never raises.
        try:
            from pluto import sentry

            sentry.capture_exception(exc)
        except Exception:
            pass

    def log(self, data: Dict[str, Any], step=None, commit=None, **kwargs):
        """Log metrics to both wandb and Pluto.

        Value routing for the Pluto side:
        - int/float and any scalar exposing .item() (numpy/torch/etc.)
          -> Pluto metrics (time-series), matching Pluto core's own log()
        - wandb media (Image/Video/Audio/Histogram/Table), and lists
          thereof -> converted Pluto media
        - str and bool -> Pluto config (latest-wins). Pluto has no
          string/bool time-series metric, so these mirror wandb's
          summary/overview placement and stay queryable via
          get_run().config.
        - anything else with no metric/media mapping -> preserved as
          config if it survives update_config's normalization (incl.
          OmegaConf), otherwise dropped and reported to Sentry telemetry
          once per key (a maintainer-coverage signal, not a user-facing
          warning). See _handle_unforwardable.

        str/bool/config values are deduped against the last synced value, so
        logging an unchanged value every step doesn't spam update_config.
        """
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
                # Separate numeric values from rich data types.
                #
                # Values can be: scalars, tensors, single wandb media
                # objects (Image/Video/Audio/Histogram/Table), or LISTS
                # of wandb media objects. Lists are a common pattern
                # from Composer/HuggingFace loggers that batch multiple
                # generated samples under one key:
                #     wandb.log({"gens": [wandb.Image(t) for t in imgs]})
                # Pluto.log() natively supports lists, so we just need
                # to convert each element and pass the list through.
                pluto_data: Dict[str, Any] = {}
                # String values have no time-series metric equivalent in
                # Pluto (op._process_log_item_sync only keeps int/float/
                # tensor/File/Data). wandb puts loose strings in the run
                # summary/overview; the closest Pluto analogue is config,
                # which is latest-wins and queryable via get_run().config.
                # This is what lets e.g. a resume skill read back the most
                # recent checkpoint/r2_path for a run.
                pluto_config: Dict[str, Any] = {}
                for key, value in data.items():
                    if isinstance(value, bool):
                        # bool is a subclass of int, but Pluto drops bool
                        # metrics — surface it as config so it isn't lost.
                        # Skip if unchanged since last log (avoid redundant
                        # config writes when logged every step).
                        if self._last_logged_config.get(key, _MISSING) != value:
                            pluto_config[key] = value
                    elif isinstance(value, (int, float)):
                        pluto_data[key] = value
                    elif (num := _as_scalar_number(value)) is not None:
                        pluto_data[key] = num
                    elif isinstance(value, str):
                        if self._last_logged_config.get(key, _MISSING) != value:
                            pluto_config[key] = value
                    elif isinstance(value, (list, tuple)):
                        # List of wandb media — convert each element.
                        converted_items = []
                        for item in value:
                            c = _convert_wandb_to_pluto(key, item, self._pluto)
                            if c is not None:
                                converted_items.append(c)
                        if converted_items:
                            pluto_data[key] = converted_items
                        else:
                            # Not a media list (e.g. list of primitives) —
                            # preserve as config if possible, else warn.
                            self._handle_unforwardable(key, value, pluto_config)
                    else:
                        # Try to convert wandb data types to pluto equivalents
                        converted = _convert_wandb_to_pluto(key, value, self._pluto)
                        if converted is not None:
                            pluto_data[key] = converted
                        else:
                            # No metric/media mapping — last-resort handling
                            # so the value is never silently dropped.
                            self._handle_unforwardable(key, value, pluto_config)

                # Metrics and config are sent in independent try blocks: a
                # failure logging metrics must NOT skip the config update (or
                # vice versa) — str/bool from the same wandb.log() call live in
                # config and would otherwise be silently lost.
                if pluto_data:
                    try:
                        log_kwargs = {}
                        if actual_step is not None:
                            log_kwargs['step'] = actual_step
                        self._pluto_run.log(pluto_data, **log_kwargs)
                    except Exception as e:
                        self._log_pluto_failure('Failed to log media/metrics', e)

                if pluto_config:
                    try:
                        self._pluto_run.update_config(pluto_config)
                        # Only remember as synced once the update succeeds.
                        # deepcopy so the dedup snapshot can't share a reference
                        # with a caller-owned mutable: today pluto_config holds
                        # only immutable str/bool or a fresh to_native_config
                        # rebuild, but copying keeps the != comparison correct
                        # even if a future branch stores a user object directly.
                        self._last_logged_config.update(copy.deepcopy(pluto_config))
                    except Exception as e:
                        self._log_pluto_failure('Failed to sync config', e)
            except Exception as e:
                self._log_pluto_failure('Failed to prepare Pluto data', e)

        return result

    def _handle_unforwardable(self, key, value, pluto_config: Dict[str, Any]) -> None:
        """Last-resort handling for a value with no metric/media mapping.

        Pluto only stores numbers (metrics), media/structured data, and
        config — so values outside those (dicts, None, raw/multi-element
        tensors, numpy arrays, unconvertible wandb media like Html/Object3D,
        custom objects) have nowhere to go. Rather than dropping them
        silently — which is what made missing data so hard to diagnose —
        we:

        1. Preserve the value as config if it survives update_config's own
           normalization (mirrors how wandb keeps loose values in the run
           summary). This covers nested dicts/lists of primitives, None, and
           OmegaConf DictConfig/ListConfig nodes (which to_native_config
           deep-converts). Skipped if unchanged since the last log.
        2. Otherwise drop the Pluto copy (it still reached W&B) and report
           it as a maintainer-coverage signal via Sentry telemetry — once
           per key. This is a gap in OUR type handling, not a user error,
           so we deliberately do NOT emit a user-facing warning: people
           migrating away from wandb shouldn't be nagged about types only
           we can fix. The local log stays at debug for self-host
           debugging.
        """
        storable, native = _config_storable_value(value)
        if storable:
            if self._last_logged_config.get(key, _MISSING) != native:
                pluto_config[key] = native
            return
        if key in self._unforwardable_warned:
            return
        self._unforwardable_warned.add(key)
        type_name = type(value).__name__
        # Quiet locally (debug only) — not a user-actionable problem.
        logger.debug(
            'pluto.compat.wandb: not forwarding %r to Pluto — type %s has no '
            'metric/media/config mapping (still logged to W&B).',
            key,
            type_name,
        )
        # Alert us (the maintainers) so we can add coverage for the type.
        # Message is keyed on the type (not the run-specific key) so Sentry
        # groups all occurrences of the same unhandled type together.
        try:
            from pluto import sentry

            sentry.capture_message(
                f'wandb compat: unforwardable Pluto log value of type '
                f'{type_name!r} (no metric/media/config mapping)',
                level='warning',
            )
        except Exception:
            pass

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

    def save(self, glob_str=None, base_path=None, policy='live'):
        """
        Save files to wandb AND log them to Pluto as artifacts.

        wandb.save() accepts a glob pattern and uploads matching files
        to the wandb run. Our dual-logging version expands the glob, and
        for each matching regular file, logs it to Pluto via
        pluto.Artifact. The key in Pluto is the file's basename.

        Safety: wandb.save() is called first and its result is always
        returned. Any failure in the Pluto path is logged at debug level
        and swallowed — wandb behavior is never affected.

        Policy semantics ('live', 'now', 'end') are handled by wandb;
        Pluto logs the file once, at the time save() is called.
        """
        result = self._wandb_run.save(glob_str, base_path=base_path, policy=policy)

        if self._pluto_run and glob_str:
            try:
                import glob as _glob_module

                # wandb.save() allows relative globs resolved against
                # cwd or base_path. Mirror that behavior.
                pattern = glob_str
                if base_path and not os.path.isabs(pattern):
                    pattern = os.path.join(base_path, pattern)

                matches = _glob_module.glob(pattern, recursive=True)
                pluto_files = {}
                for match in matches:
                    if not os.path.isfile(match):
                        continue
                    basename = os.path.basename(match)
                    # Namespace under 'save/' so these don't collide
                    # with metric / media keys.
                    log_name = f'save/{basename}'
                    pluto_files[log_name] = self._pluto.Artifact(
                        match, caption=basename
                    )

                if pluto_files:
                    self._pluto_run.log(pluto_files)
                    logger.info(
                        f'pluto.compat.wandb: save() forwarded '
                        f'{len(pluto_files)} file(s) to Pluto'
                    )
            except Exception as e:
                logger.debug(f'pluto.compat.wandb: save() Pluto forward failed: {e}')

        return result

    def log_artifact(
        self, artifact_or_path, name=None, type=None, aliases=None, tags=None
    ):
        """
        Log a wandb.Artifact to wandb AND forward its local files to Pluto.

        wandb.Artifact is a versioned bundle of files/dirs/references.
        Pluto has no equivalent versioning/aliasing, but we can still
        log the local file contents as Pluto artifacts so the data is
        preserved. Reference entries (S3 URLs, etc.) are skipped —
        we only forward entries that have a local file path.

        Accepts either:
        - A wandb.Artifact instance (iterates its manifest entries)
        - A file path string (wandb.Artifact with a single file)

        Safety: wandb.log_artifact() is called first. Any failure in
        the Pluto path is swallowed — wandb behavior is never affected.
        """
        # Call the real wandb log_artifact first — it finalizes the
        # artifact's manifest, which we need to read local_path from.
        try:
            forward_kwargs = {}
            if name is not None:
                forward_kwargs['name'] = name
            if type is not None:
                forward_kwargs['type'] = type
            if aliases is not None:
                forward_kwargs['aliases'] = aliases
            if tags is not None:
                forward_kwargs['tags'] = tags
            result = self._wandb_run.log_artifact(artifact_or_path, **forward_kwargs)
        except Exception:
            # Re-raise — wandb errors must surface to the user.
            raise

        if self._pluto_run:
            try:
                pluto_files = {}

                # Path-string form: single file/dir reference.
                if isinstance(artifact_or_path, str):
                    if os.path.isfile(artifact_or_path):
                        basename = os.path.basename(artifact_or_path)
                        art_name = name or basename
                        log_name = f'artifacts/{art_name}'
                        pluto_files[log_name] = self._pluto.Artifact(
                            artifact_or_path, caption=art_name
                        )
                else:
                    # wandb.Artifact form: iterate manifest entries.
                    art = artifact_or_path
                    art_name = getattr(art, 'name', None) or 'artifact'
                    manifest = getattr(art, 'manifest', None)
                    entries = {}
                    if manifest is not None:
                        entries = getattr(manifest, 'entries', {}) or {}

                    for entry_name, entry in entries.items():
                        # Skip reference entries (S3 URLs etc.) — we
                        # can't upload something we don't have locally.
                        local_path = getattr(entry, 'local_path', None)
                        if not local_path or not os.path.isfile(local_path):
                            continue
                        log_name = f'artifacts/{art_name}/{entry_name}'
                        pluto_files[log_name] = self._pluto.Artifact(
                            local_path, caption=entry_name
                        )

                if pluto_files:
                    self._pluto_run.log(pluto_files)
                    logger.info(
                        f'pluto.compat.wandb: log_artifact() forwarded '
                        f'{len(pluto_files)} file(s) to Pluto'
                    )
            except Exception as e:
                logger.debug(
                    f'pluto.compat.wandb: log_artifact() Pluto forward failed: {e}'
                )

        return result

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


def _as_scalar_number(value):
    """Return value as a python int/float if it's a scalar number, else None.

    Mirrors Pluto's own log() (op._process_log_item_sync), which forwards
    anything exposing a callable ``.item()``. The shim previously only
    accepted plain int/float and torch scalar tensors, so a value logged as
    a numpy scalar (``np.int64``), a 0-d numpy array, or a non-torch 0-d
    tensor was dropped here even though Pluto core would have kept it — e.g.
    an ``epoch`` that is ``np.int64`` rather than a plain ``int``.

    bool and str are excluded (Pluto drops bool metrics; str routes to
    config). ``.item()`` on a multi-element array/tensor raises — we treat
    that as "not a scalar" and return None, same as Pluto would fail it.
    """
    if isinstance(value, (bool, str)):
        return None
    item = getattr(value, 'item', None)
    if not callable(item):
        return None
    try:
        result = item()
    except Exception:
        return None
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        return None
    return result


def _config_storable_value(value):
    """Return ``(storable, native)`` for the config fallback.

    Mirrors what ``update_config`` actually does — normalize via
    ``to_native_config`` (which deep-converts OmegaConf ``DictConfig`` /
    ``ListConfig`` to native containers), then check JSON-serializability.
    Keeping the gate in lockstep with ``update_config`` means a logged
    ``DictConfig`` is correctly stored as config, even though plain
    ``json.dumps`` would reject it. Tensors / ndarrays / custom objects still
    fail (``to_native_config`` leaves them as-is) and fall through to the
    Sentry path.

    Returns ``(True, native_value)`` when storable, else ``(False, None)``.
    """
    try:
        from pluto.util import to_native_config

        native = to_native_config(value)
        json.dumps(native)
        return True, native
    except Exception:
        return False, None


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


def _wandb_caption(value):
    """Extract a user-provided caption from a wandb media object.

    wandb.Image/Audio/Video store the ``caption=`` kwarg on ``_caption``.
    Returns a non-empty string or None (ignores wandb's list-of-captions
    grouping form, which has no single-file equivalent here).
    """
    cap = getattr(value, '_caption', None)
    return cap if isinstance(cap, str) and cap else None


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
            caption = _wandb_caption(value)
            if getattr(value, '_path', None):
                return pluto_module.Image(value._path, caption=caption)
            # Fallback: convert PIL to numpy (which pluto.Image handles)
            pil_img = getattr(value, 'image', None) or getattr(value, '_image', None)
            if pil_img is not None:
                try:
                    import numpy as np

                    return pluto_module.Image(np.asarray(pil_img), caption=caption)
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
                return pluto_module.Audio(value._path, caption=_wandb_caption(value))
            return None

        if type_name == 'Video':
            # wandb.Video always writes to _path on construction (after
            # encoding). This can take a few seconds for numpy input.
            if getattr(value, '_path', None):
                return pluto_module.Video(value._path, caption=_wandb_caption(value))
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

        # Project name fallback (works in ALL modes): if PLUTO_PROJECT
        # isn't set, fall back to (in order): the explicit `project`
        # kwarg passed to wandb.init(), the `WANDB_PROJECT` env var,
        # or finally the project attribute on the resolved wandb run.
        # This makes PLUTO_PROJECT fully optional — frameworks like
        # Lightning's WandbLogger pass project as a kwarg and may never
        # set WANDB_PROJECT, so kwargs must be consulted too.
        #
        # If pluto_config is None here, it means PLUTO_PROJECT wasn't set
        # (that's the only reason _get_pluto_config_from_env returns None).
        # We build a fresh config from the resolved project and re-read
        # the other PLUTO_* env vars (api key, urls) since the helper
        # bailed before reading them.
        if pluto_config is None:
            resolved_project = (
                kwargs.get('project')
                or os.environ.get('WANDB_PROJECT')
                or getattr(wandb_run, 'project', None)
            )
            if resolved_project:
                pluto_config = {'project': resolved_project}
                if api_key := os.environ.get('PLUTO_API_KEY'):
                    pluto_config['api_key'] = api_key
                for env_var, cfg_key in (
                    ('PLUTO_URL_APP', 'url_app'),
                    ('PLUTO_URL_API', 'url_api'),
                    ('PLUTO_URL_INGEST', 'url_ingest'),
                ):
                    if v := os.environ.get(env_var):
                        pluto_config[cfg_key] = v
                logger.info(
                    f'pluto.compat.wandb: using "{resolved_project}" as Pluto '
                    f'project (PLUTO_PROJECT not set)'
                )

        # Migration shortcut (disabled-mode only): in DISABLE_WANDB_LOGGING
        # mode, the user can reuse WANDB_API_KEY to hold a Pluto API token
        # so they don't need a separate PLUTO_API_KEY env var.
        if wandb_disabled and pluto_config and 'api_key' not in pluto_config:
            wandb_api_key = os.environ.get('WANDB_API_KEY')
            if wandb_api_key:
                pluto_config['api_key'] = wandb_api_key
                logger.info(
                    'pluto.compat.wandb: using WANDB_API_KEY as Pluto API key '
                    '(DISABLE_WANDB_LOGGING fallback)'
                )

        if pluto_config is None:
            logger.warning(
                'pluto.compat.wandb: cannot dual-log to Pluto — no project '
                'name resolvable (none of: PLUTO_PROJECT, project= kwarg, '
                'WANDB_PROJECT, wandb run project). Continuing with wandb-'
                'only logging.'
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
            #   2. WANDB_* env var (authoritative — user-set)
            #   3. Attribute on the resolved wandb_run (fallback)
            #
            # Env vars come before wandb_run attributes because in
            # wandb's disabled mode (DISABLE_WANDB_LOGGING=true → wandb
            # mode=disabled), wandb generates a fake "dummy-xxx" name
            # that ignores WANDB_NAME. We want the user-specified name.
            #
            # See https://docs.wandb.ai/guides/track/environment-variables
            name = (
                kwargs.get('name')
                or os.environ.get('WANDB_NAME')
                or getattr(wandb_run, 'name', None)
            )
            wandb_config = kwargs.get('config') or getattr(wandb_run, 'config', None)
            wandb_tags_env = os.environ.get('WANDB_TAGS', '')
            tags = (
                kwargs.get('tags')
                or (wandb_tags_env.split(',') if wandb_tags_env else None)
                or getattr(wandb_run, 'tags', None)
            )
            notes = (
                kwargs.get('notes')
                or os.environ.get('WANDB_NOTES')
                or getattr(wandb_run, 'notes', None)
            )
            # WANDB_RUN_GROUP and WANDB_JOB_TYPE: wandb-specific concepts
            # with no direct Pluto equivalent. We surface them as tags so
            # they're searchable in the Pluto UI. These are documented at
            # https://docs.wandb.ai/guides/track/environment-variables
            wandb_group = kwargs.get('group') or os.environ.get('WANDB_RUN_GROUP')
            wandb_job_type = kwargs.get('job_type') or os.environ.get('WANDB_JOB_TYPE')
            extra_tags = []
            if wandb_group:
                extra_tags.append(f'group:{wandb_group}')
            if wandb_job_type:
                extra_tags.append(f'job_type:{wandb_job_type}')
            if extra_tags:
                tags = (list(tags) if tags else []) + extra_tags
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
            # wandb.save and wandb.log_artifact also get rebound to
            # Run instance methods after init, same as log/finish.
            # Route them through our wrapper so dual-logging works.
            wandb_module.save = wrapper.save
            wandb_module.log_artifact = wrapper.log_artifact

            return wrapper

        except Exception as e:
            # Loud failure — users should know dual-logging is off
            _msg = (
                f'pluto.compat.wandb: DUAL-LOGGING DISABLED. Failed to '
                f'initialize Pluto run: {type(e).__name__}: {e}. '
                f'wandb will continue to work normally, but NO DATA will be '
                f'sent to Pluto. To fix, resolve the error above and retry.'
            )
            # exc_info=True attaches the traceback so the log points at the
            # raise site (e.g. the failing json.dumps), not just this handler.
            logger.error(_msg, exc_info=True)
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
