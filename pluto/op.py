import atexit
import builtins
import logging
import os
import queue
import signal
import sqlite3
import sys
import threading
import time
import traceback
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse

import pluto

from . import sentry as _sentry
from .api import (
    make_compat_alert_v1,
    make_compat_monitor_v1,
    make_compat_resume_v1,
    make_compat_start_v1,
    make_compat_status_v1,
    make_compat_trigger_v1,
    make_compat_webhook_v1,
)
from .auth import login
from .data import Data
from .file import Artifact, Audio, File, Image, Text, Video
from .iface import PlutoRequestError, ServerInterface
from .log import flush_console_buffers, setup_logger, teardown_logger
from .store import DataStore
from .sync import SyncProcessManager
from .sync.store import HEALTH_METRIC_KEYS
from .sys import System
from .util import (
    ANSI,
    deep_merge,
    get_char,
    get_val,
    print_url,
    to_json,
    to_native_config,
)

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'Operation'


def _is_distributed_environment() -> bool:
    """Check if running in a distributed (DDP/FSDP) environment."""
    # Check environment variables set by torchrun/torch.distributed.launch
    world_size = os.environ.get('WORLD_SIZE', '1')
    if world_size.isdigit() and int(world_size) > 1:
        return True

    # Check SLURM environment variables
    if 'SLURM_PROCID' in os.environ:
        slurm_ntasks = os.environ.get('SLURM_NTASKS', '1')
        if slurm_ntasks.isdigit() and int(slurm_ntasks) > 1:
            return True

    # Check if torch.distributed is initialized
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return True
    except ImportError:
        pass

    return False


# Exception hook state for detecting unhandled exceptions (FAILED status)
_original_excepthook = None
_excepthook_registered = False


def _module_is_pluto(module: str) -> bool:
    return module == 'pluto' or module.startswith('pluto.')


def _exception_touches_pluto(
    exc: Optional[BaseException],
    _seen: Optional[set] = None,
) -> bool:
    """
    Return True if an exception, its traceback, chained causes/contexts, or
    ExceptionGroup members involve the pluto package.

    Used to decide whether an unhandled exception should be forwarded to
    Pluto's internal Sentry telemetry. We only want to report errors that
    actually originated in (or passed through) Pluto's own code, not
    unrelated user/framework exceptions that merely happened while a Pluto
    run was active.

    The check covers:
      * the exception class's defining module (e.g. ``pluto.errors.X``)
      * every frame in the exception's own traceback
      * ``__cause__`` (explicit ``raise X from Y``)
      * ``__context__`` (implicit chaining, unless ``__suppress_context__``)
      * ``BaseExceptionGroup.exceptions`` members on Python 3.11+

    An id-based seen set guards against cyclic exception chains.
    """
    if exc is None:
        return False

    if _seen is None:
        _seen = set()
    if id(exc) in _seen:
        return False
    _seen.add(id(exc))

    # Check the exception class's defining module
    module = getattr(type(exc), '__module__', '') or ''
    if _module_is_pluto(module):
        return True

    # Walk the traceback frames
    tb = exc.__traceback__
    while tb is not None:
        frame_module = tb.tb_frame.f_globals.get('__name__', '')
        if _module_is_pluto(frame_module):
            return True
        tb = tb.tb_next

    # Follow the explicit `raise X from Y` chain
    if _exception_touches_pluto(exc.__cause__, _seen):
        return True

    # Follow the implicit `except: raise Y` chain, unless suppressed by
    # `raise Y from None`.
    if not getattr(exc, '__suppress_context__', False):
        if _exception_touches_pluto(exc.__context__, _seen):
            return True

    # ExceptionGroup members (Python 3.11+). BaseExceptionGroup does not
    # exist on 3.10, so look it up defensively.
    base_group = getattr(builtins, 'BaseExceptionGroup', None)
    if base_group is not None and isinstance(exc, base_group):
        for inner in exc.exceptions:
            if _exception_touches_pluto(inner, _seen):
                return True

    return False


def _excepthook(exc_type, exc_value, exc_traceback):
    """
    Custom sys.excepthook to mark active runs as FAILED on unhandled exceptions.

    When user training code raises an unhandled exception, Python calls sys.excepthook
    before interpreter shutdown. We mark all active runs as FAILED (status=1) so the
    atexit-registered finish() sends the correct status to the server.
    """
    # Mark all active ops as FAILED
    if pluto.ops:
        for op in list(pluto.ops):
            if op.settings._op_status == -1:  # Only if still RUNNING
                op.settings._op_status = 1
                logger.debug(
                    f'{tag}: Marked run {op.settings._op_id} as FAILED '
                    f'due to unhandled {exc_type.__name__}'
                )

    # Report to Sentry APM — but only if the exception actually touches pluto.
    # Otherwise we'd capture unrelated errors from user/framework code (e.g.
    # NCCL/CUDA failures in torch.distributed) that have nothing to do with
    # the SDK. The check walks the exception's traceback, chained causes and
    # contexts, and any ExceptionGroup members.
    if _exception_touches_pluto(exc_value):
        _sentry.capture_exception(exc_value)
        _sentry.flush()

    # Call the original excepthook to preserve default behavior (print traceback)
    if _original_excepthook is not None:
        _original_excepthook(exc_type, exc_value, exc_traceback)


def _register_excepthook():
    """Register custom excepthook if not already registered."""
    global _excepthook_registered, _original_excepthook

    if _excepthook_registered:
        return

    _original_excepthook = sys.excepthook
    sys.excepthook = _excepthook
    _excepthook_registered = True
    logger.debug(f'{tag}: Registered sys.excepthook for FAILED status detection')


def _unregister_excepthook():
    """Restore original excepthook when no more Ops are active."""
    global _excepthook_registered, _original_excepthook

    if not _excepthook_registered:
        return

    if _original_excepthook is not None:
        sys.excepthook = _original_excepthook
    _excepthook_registered = False
    logger.debug(f'{tag}: Restored original sys.excepthook')


# SIGTERM handler state for reporting a terminal status on preemption/eviction.
# Spot reclaim and k8s eviction deliver SIGTERM, which by default terminates the
# process WITHOUT running atexit — so the atexit-registered finish() never fires
# and the run would linger as RUNNING until the server-side heartbeat reaps it
# (~90s). We install a handler that best-effort pushes TERMINATED, then chains to
# whatever handler was previously installed so the process's actual exit
# behaviour is unchanged (this is also what keeps the Neptune compat contract of
# "MUST NOT affect signal handling" intact — the prior handler still runs).
_original_sigterm_handler = None
_sigterm_handler_registered = False


def _sigterm_handler(signum, frame):
    """Best-effort push TERMINATED for active runs, then chain to the previous
    handler so the process terminates exactly as it otherwise would."""
    for op in list(pluto.ops or []):
        try:
            if op.settings._op_status != -1:
                continue  # already terminal/finishing — don't overwrite
            # signum maps to TERMINATED via api.STATUS (SIGTERM/SIGINT).
            op.settings._op_status = signum
            iface = op._iface
            if iface is None:
                continue
            # Bounded, no-retry POST: a hung network must not eat the finite
            # eviction grace window before SIGKILL lands.
            iface._post_v1(
                op.settings.url_stop,
                iface.headers,
                make_compat_status_v1(op.settings),
                client=iface.client_api,
                max_retries=0,
                timeout=op.settings.x_sigterm_status_timeout_seconds,
                suppress_httpx_logs=True,
            )
        except Exception as e:
            logger.debug('%s: SIGTERM status push failed: %s', tag, e)

    # Chain to the previously-installed handler so we preserve exit behaviour.
    prev = _original_sigterm_handler
    if callable(prev):
        prev(signum, frame)
    elif prev == signal.SIG_IGN:
        return  # process was ignoring SIGTERM — keep doing so
    else:
        # SIG_DFL, or None (handler installed outside Python): re-apply the
        # default terminating action so we never accidentally swallow SIGTERM.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.raise_signal(signum)


def _register_sigterm_handler(settings) -> None:
    """Install the SIGTERM status handler once, if enabled and possible.

    signal.signal only works in the main thread; if init() runs off the main
    thread (some frameworks/launchers), skip silently rather than raise.
    """
    global _sigterm_handler_registered, _original_sigterm_handler

    if _sigterm_handler_registered:
        return
    if not getattr(settings, 'x_sigterm_status_enabled', True):
        return
    if threading.current_thread() is not threading.main_thread():
        logger.debug(f'{tag}: not on main thread, skipping SIGTERM handler')
        return
    try:
        _original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _sigterm_handler)
        _sigterm_handler_registered = True
        logger.debug(f'{tag}: Registered SIGTERM handler for TERMINATED status')
    except (ValueError, OSError) as e:
        logger.debug(f'{tag}: could not register SIGTERM handler: {e}')


def _unregister_sigterm_handler() -> None:
    """Restore the previous SIGTERM handler when no more Ops are active."""
    global _sigterm_handler_registered, _original_sigterm_handler

    if not _sigterm_handler_registered:
        return
    try:
        if _original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, _original_sigterm_handler)
    except (ValueError, OSError):
        pass
    _sigterm_handler_registered = False
    logger.debug(f'{tag}: Restored original SIGTERM handler')


MetaNames = List[str]
MetaFiles = Dict[str, List[str]]
LoggedNumbers = Dict[str, Any]
LoggedData = Dict[str, List[Data]]
LoggedFiles = Dict[str, List[File]]
QueueItem = Tuple[Dict[str, Any], Optional[int]]


class OpMonitor:
    def __init__(self, op) -> None:
        self.op = op
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._thread_monitor: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self.op._worker, args=(self._stop_event.is_set,), daemon=True
            )
            self._thread.start()
        if self._thread_monitor is None:
            self._thread_monitor = threading.Thread(
                target=self._worker_monitor,
                args=(self._stop_event.is_set,),
                daemon=True,
            )
            self._thread_monitor.start()

    def stop(self, code: Union[int, None] = None) -> None:
        self._stop_event.set()
        timeout = self.op.settings.x_thread_join_timeout_seconds
        for attr in ['_thread', '_thread_monitor']:
            thread = getattr(self, attr)
            if thread is not None:
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(
                        f'{tag}: Thread {thread.name} did not terminate, '
                        'continuing anyway'
                    )
                setattr(self, attr, None)
        if isinstance(code, int):
            self.op.settings._op_status = code
        elif self.op.settings._op_status == -1:
            # Only set COMPLETED if still RUNNING; preserve FAILED (1) if
            # set by sys.excepthook due to an unhandled exception.
            self.op.settings._op_status = 0

    def _worker_monitor(self, stop):
        while not stop():
            try:
                # Collect system metrics
                sys_metrics = make_compat_monitor_v1(self.op.settings._sys.monitor())
                timestamp_ms = int(time.time() * 1000)

                # Send system metrics via sync process if enabled
                if self.op._sync_manager is not None:
                    self.op._sync_manager.enqueue_system_metrics(
                        metrics=sys_metrics,
                        timestamp_ms=timestamp_ms,
                    )

                # Send heartbeat/trigger to server
                # Use short timeout and no retries: if it fails, the next
                # cycle will try again. This prevents the monitor thread
                # from blocking when the server is returning errors,
                # which would cause finish() to hang.
                if self.op._iface:
                    r = self.op._iface._post_v1(
                        self.op.settings.url_trigger,
                        self.op._iface.headers,
                        make_compat_trigger_v1(self.op.settings),
                        client=self.op._iface.client,
                        max_retries=0,
                        timeout=5.0,
                        suppress_httpx_logs=True,
                    )
                    if hasattr(r, 'json') and r.json()['status'] == 'CANCELLED':
                        logger.critical(f'{tag}: server finished run')
                        os._exit(signal.SIGINT.value)  # TODO: do a more graceful exit
            except sqlite3.OperationalError as e:
                logger.warning('%s: transient database error (will retry): %s', tag, e)
            except Exception as e:
                logger.critical('%s: failed: %s', tag, e)
            # Use event.wait() instead of time.sleep() so the thread wakes
            # immediately when _stop_event is set during shutdown.
            self._stop_event.wait(timeout=self.op.settings.x_sys_sampling_interval)


class Op:
    def __init__(self, config, settings, tags=None, resume=False) -> None:
        self.config = config
        self.settings = settings
        self.tags: List[str] = tags if tags else []  # Use provided tags or empty list
        self._monitor = OpMonitor(op=self)
        self._resumed: bool = False  # Whether this run was resumed (multi-node)
        self._resume: bool = resume  # Whether resume was explicitly requested
        self._sync_manager: Optional[SyncProcessManager] = None
        self._fork_run_id: Optional[int] = None  # Resolved parent run ID from response
        self._fork_step: Optional[int] = None  # Fork step from response

        # Determine if sync process should be used
        self._use_sync_process = settings.sync_process_enabled

        if self.settings.mode == 'noop':
            self.settings.disable_iface = True
            self.settings.disable_store = True
        else:
            # TODO: set up tmp dir
            login(settings=self.settings)
            if self.settings._sys == {}:
                self.settings._sys = System(self.settings)
            tmp_iface = ServerInterface(config=config, settings=settings)
            try:
                if (
                    settings._resume_run_id is not None
                    or settings._resume_display_id is not None
                ):
                    # Resume existing run via /api/runs/resume
                    r = tmp_iface._post_v1(
                        self.settings.url_resume,
                        tmp_iface.headers,
                        make_compat_resume_v1(self.settings),
                        client=tmp_iface.client_api,
                        raise_on_error=True,
                    )
                else:
                    # Create new run (or resume via externalId)
                    r = tmp_iface._post_v1(
                        self.settings.url_start,  # create-run
                        tmp_iface.headers,
                        make_compat_start_v1(
                            self.config,
                            self.settings,
                            self.settings._sys.get_info(),
                            self.tags,
                        ),
                        client=tmp_iface.client_api,
                        raise_on_error=True,
                    )
            except PlutoRequestError as e:
                # The server rejected the request (e.g. a validation error such
                # as "A run can have at most one group:* tag."). Surface the
                # server's reason rather than a misleading connection error.
                raise RuntimeError(f'Failed to create run: {e}') from e
            if not r:
                # No response after retries → the server was unreachable (a
                # 4xx/5xx would have raised PlutoRequestError above).
                raise ConnectionError(
                    'Failed to create or resume run. Check connection to Pluto server.'
                )
            response_data = r.json()
            self.settings.url_view = response_data['url']
            self.settings._op_id = response_data['runId']
            self.settings._display_id = response_data.get('displayId')
            self._resumed = response_data.get('resumed', False)
            self._fork_run_id = response_data.get('forkedFromRunId')
            self._fork_step = response_data.get('forkStep')
            if self._resumed:
                # Collision detection: prevent accidental data interleaving
                if self._resume:
                    pass  # User explicitly opted in to resume
                elif (
                    self.settings._resume_run_id is not None
                    or self.settings._resume_display_id is not None
                ):
                    pass  # Explicit resume via numeric/display ID
                elif self.settings._external_id_from_env:
                    logger.info(
                        f'{tag}: Run ID from PLUTO_RUN_ID env var matched '
                        f'existing run, allowing resume.'
                    )
                else:
                    external_id = self.settings._external_id
                    raise RuntimeError(
                        f"Run with externalId '{external_id}' already exists. "
                        f'This often happens when random.seed() or '
                        f'L.seed_everything() makes run IDs deterministic. '
                        f'Pass resume=True to pluto.init() to intentionally '
                        f'reattach, or use a unique run_id.'
                    )
                logger.info(f'{tag}: resumed run {str(self.settings._op_id)}')
                self._print_run_banner('resumed')
                logger.warning(
                    f'{tag}: Run was resumed via run_id. The `name` parameter '
                    f'is ignored for resumed runs - the original run name is '
                    f'preserved. For multi-node, use the same name across all ranks.'
                )
            else:
                logger.info(f'{tag}: started run {str(self.settings._op_id)}')
                self._print_run_banner('started')

            os.makedirs(f'{self.settings.get_dir()}/files', exist_ok=True)

            # Initialize sync process manager if enabled (before logger setup
            # so console logs can be captured via sync process)
            if self._use_sync_process:
                self._init_sync_manager()

            setup_logger(
                settings=self.settings,
                logger=logger,
                console=logging.getLogger('console'),
                sync_manager=self._sync_manager,
            )  # global logger
            to_json(
                [self.settings._sys.get_info()], f'{self.settings.get_dir()}/sys.json'
            )

        # DataStore is only used when sync process is disabled (legacy mode)
        self._store: Optional[DataStore] = (
            DataStore(config=config, settings=settings)
            if not settings.disable_store and not self._use_sync_process
            else None
        )
        # ServerInterface is always created for HTTP utilities (status updates,
        # triggers). Data upload is handled by sync process, not ServerInterface.
        self._iface: Optional[ServerInterface] = (
            ServerInterface(config=config, settings=settings)
            if not settings.disable_iface
            else None
        )
        self._step = 0
        self._queue: queue.Queue[QueueItem] = queue.Queue()
        self._finished = False
        self._finish_lock = threading.Lock()
        # (log-key, exc-type) pairs already surfaced at error from a dropped log
        # item, so a per-step failure is shouted once then drops to debug.
        self._dropped_item_warned: set = set()
        atexit.register(self.finish)

    def _init_sync_manager(self) -> None:
        """Initialize the sync process manager."""
        # Generate run_id for sync process
        run_id = self.settings._external_id or str(self.settings._op_id)

        # Build settings dict for sync process (must be serializable)
        settings_dict = {
            'dir': self.settings.dir,
            'tag': self.settings.tag,
            'project': self.settings.project,
            '_auth': self.settings._auth,
            '_op_id': self.settings._op_id,
            '_op_name': self.settings._op_name,
            '_config': self.config,
            'url_num': self.settings.url_num,
            'url_data': self.settings.url_data,  # For structured data
            'url_update_config': self.settings.url_update_config,
            'url_update_tags': self.settings.url_update_tags,
            'url_file': self.settings.url_file,  # For file uploads
            'url_message': self.settings.url_message,  # For console logs
            'x_log_level': self.settings.x_log_level,
            'pluto_version': self.settings.pluto_version,
            'pluto_commit': self.settings.pluto_commit,
            'sync_process_flush_interval': (self.settings.sync_process_flush_interval),
            'sync_process_shutdown_timeout': (
                self.settings.sync_process_shutdown_timeout
            ),
            'sync_process_orphan_timeout': (self.settings.sync_process_orphan_timeout),
            'sync_process_retry_max': self.settings.sync_process_retry_max,
            'sync_process_retry_backoff': self.settings.sync_process_retry_backoff,
            'sync_process_batch_size': self.settings.sync_process_batch_size,
            'sync_process_file_batch_size': self.settings.sync_process_file_batch_size,
        }

        self._sync_manager = SyncProcessManager(
            run_id=run_id,
            project=self.settings.project,
            settings_dict=settings_dict,
            db_path=self.settings.sync_process_db_path,
        )
        logger.debug(f'{tag}: initialized sync process manager')

    def _print_run_banner(self, verb: str) -> None:
        """Print a stable, greppable run banner to stdout.

        Emits one line in a fixed format so external tooling can reverse-look
        up a run from a training process's stdout, e.g.::

            pluto: run LV3-12 started (external_id=dhyecrvx)

        The display ID (e.g. ``LV3-12``) comes from the server's create/resume
        response; the ``external_id`` is the sqid slug (the last path segment
        of the run URL). This is intentionally a plain ``print`` to stdout,
        independent of the logging system, so it can't be suppressed by log
        levels or console-capture settings and always lands on stdout.

        The line is colored green only when stdout is a TTY. When stdout is
        piped or redirected (the case where tooling scrapes the banner), it is
        emitted as plain text so the ANSI codes never land in captured logs and
        greppability is preserved.
        """
        display_id = self.settings._display_id
        if not display_id:
            return  # server didn't return a display ID; nothing stable to print
        external_id = None
        if self.settings.url_view:
            # Parse the path so a host-only URL (no run slug) doesn't yield the
            # hostname as a bogus external_id.
            path = urlparse(self.settings.url_view).path.strip('/')
            if path:
                external_id = path.split('/')[-1]
        suffix = f' (external_id={external_id})' if external_id else ''
        msg = f'pluto: run {display_id} {verb}{suffix}'
        # Wrap the whole line (not just the ID) so the codes sit at the very
        # start/end and the matchable token stays contiguous even in a TTY.
        if sys.stdout is not None and sys.stdout.isatty():
            msg = f'\033[32m{msg}\033[0m'  # green
        print(msg, flush=True)

    def _view_run_message(self) -> str:
        """Build the 'View run [<id>] at <url>' log message.

        Includes the display ID (green) when the server returned one. ANSI
        codes come from ``util.ANSI``, which blanks them on non-TTY output, so
        this matches how ``print_url`` colors the URL.
        """
        url = print_url(self.settings.url_view)
        display_id = self.settings._display_id
        if display_id:
            # Return to cyan (the INFO message color) after the green ID rather
            # than a full reset, so the trailing "at <url>" stays cyan like the
            # rest of the line.
            return f'View run {ANSI.green}{display_id}{ANSI.cyan} at {url}'
        return f'View run at {url}'

    def start(self) -> None:
        # Start sync process if enabled
        if self._sync_manager is not None:
            self._sync_manager.start()
            logger.debug(f'{tag}: sync process started')

        # Always start the monitor for system metrics and heartbeats
        self._monitor.start()

        # Register system metric names with server (required for dashboard display)
        if self._iface:
            sys_metric_names = list(
                make_compat_monitor_v1(self.settings._sys.monitor()).keys()
            )
            # Include sync health metrics so they appear on the dashboard
            if self._sync_manager is not None:
                sys_metric_names += [f'sys/pluto.{k}' for k in HEALTH_METRIC_KEYS]
            self._iface._update_meta(sys_metric_names)

        # Print URL where users can view the run
        logger.info(f'{tag}: {self._view_run_message()}')

        # Register excepthook to detect unhandled exceptions and mark runs as FAILED
        _register_excepthook()

        # Register SIGTERM handler to report TERMINATED on preemption/eviction
        _register_sigterm_handler(self.settings)

        # set globals
        if pluto.ops is None:
            pluto.ops = []
        pluto.ops.append(self)
        pluto.log, pluto.alert, pluto.watch = self.log, self.alert, self.watch

    def log(
        self,
        data: Dict[str, Any],
        step: Union[int, None] = None,
        commit: Union[bool, None] = None,
    ) -> None:
        """Log run data"""
        # Use sync process if enabled (default: uploads data to server)
        if self._sync_manager is not None:
            try:
                self._log_via_sync(data=data, step=step)
            except sqlite3.OperationalError as e:
                # Never let a transient SQLite error crash the user's training.
                # The data for this step is lost, but training continues.
                logger.warning(
                    '%s: dropping log data due to database error: %s', tag, e
                )
        elif self.settings.mode == 'perf':
            self._queue.put((data, step), block=False)
        else:
            # Legacy offline mode (sync_process_enabled=False)
            # Data stored locally in SQLite only, not uploaded to server
            self._log(data=data, step=step)

    def _log_via_sync(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """Log data via sync process (writes to SQLite, picked up by sync)."""
        if self._sync_manager is None:
            return

        self._step = self._step + 1 if step is None else step
        timestamp_ms = int(time.time() * 1000)

        metrics: Dict[str, Any] = {}
        new_metric_names: List[str] = []
        new_file_meta: Dict[str, List[str]] = defaultdict(list)

        for k, v in data.items():
            k = get_char(k)
            items = v if isinstance(v, list) else [v]

            # Register first item for metadata (only need one per key)
            if items:
                self._register_meta_sync(k, items[0], new_metric_names, new_file_meta)

            # enumerate() gives each item its 0-based position within this
            # (key, step) log() call — the sampleIndex the server uses to
            # restore logged order for media lists. A scalar is wrapped in a
            # 1-element list above, so it naturally gets sampleIndex 0.
            for sample_index, item in enumerate(items):
                try:
                    self._process_log_item_sync(
                        k, item, metrics, timestamp_ms, sample_index
                    )
                except Exception as e:
                    # One bad item (e.g. media that can't be encoded/staged)
                    # must not abort the rest of the batch or crash the caller's
                    # training loop -- logging is best-effort. Drop just this
                    # item, loudly, and keep going so sibling metrics/files in
                    # the same call still reach the server.
                    self._warn_dropped_item(k, item, e)

        if metrics:
            self._sync_manager.enqueue_metrics(metrics, timestamp_ms, self._step)

        # Register new metric/file names with server (required for dashboard display)
        if (new_metric_names or new_file_meta) and self._iface:
            self._iface._update_meta(num=new_metric_names, df=dict(new_file_meta))

    def _warn_dropped_item(self, key: str, value: Any, exc: Exception) -> None:
        """Report a log item dropped due to an unexpected error.

        Surfaced at the native API so every entry point (direct log(), and the
        wandb/neptune/lightning shims alike) gets the same visibility -- a shim
        catching exceptions is then only defense-in-depth, not the sole signal.
        Logged once per (key, exc-type) at error, then at debug, so a failure
        that recurs every step is shouted once rather than flooding.
        """
        seen = (key, type(exc).__name__)
        detail = (
            f'{tag}: dropped {key!r} ({type(value).__name__}): '
            f'{type(exc).__name__}: {exc}'
        )
        if seen in self._dropped_item_warned:
            logger.debug(detail)
            return
        self._dropped_item_warned.add(seen)
        logger.error(f'{detail} (further occurrences at debug)')
        # Telemetry: report once per (key, exc-type) so we can spot and fix
        # these in the wild (ships the exception + traceback). Self-gates on
        # PLUTO_DISABLE_TELEMETRY / CI and never raises.
        try:
            from pluto import sentry

            sentry.capture_exception(exc)
        except Exception:
            pass

    def _is_numeric_value(self, value: Any) -> bool:
        """Check if value is a numeric type (int, float, or tensor)."""
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        # Check for tensor-like objects with .item() method
        return hasattr(value, 'item') and callable(value.item)

    def _register_meta_sync(
        self,
        key: str,
        value: Any,
        new_metric_names: List[str],
        new_file_meta: Dict[str, List[str]],
    ) -> None:
        """Register new metric/file names for metadata update."""
        if key in self.settings.meta:
            return

        self.settings.meta.append(key)
        logger.debug(f'{tag}: added {key} at step {self._step}')

        if isinstance(value, (File, Data)):
            new_file_meta[value.__class__.__name__].append(key)
        elif self._is_numeric_value(value):
            new_metric_names.append(key)

    def _process_log_item_sync(
        self,
        key: str,
        value: Any,
        metrics: Dict[str, Any],
        timestamp_ms: int,
        sample_index: int = 0,
    ) -> None:
        """Process a single log item for sync process mode."""
        if self._sync_manager is None:
            return

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = value
        elif hasattr(value, 'item'):  # Tensor
            metrics[key] = value.item()
        elif isinstance(value, File):
            # Handle file types (Image, Audio, Video, Text, Artifact)
            self._enqueue_file_sync(key, value, timestamp_ms, sample_index)
        elif isinstance(value, Data):
            # Handle structured data types (Graph, Histogram, Table)
            self._sync_manager.enqueue_data(
                log_name=key,
                data_type=type(value).__name__.upper(),
                data_dict=value.to_dict(),
                timestamp_ms=timestamp_ms,
                step=self._step,
            )

    def _enqueue_file_sync(
        self,
        log_name: str,
        file_obj: File,
        timestamp_ms: int,
        sample_index: int = 0,
    ) -> None:
        """
        Process and enqueue a file for upload via sync process.

        This mirrors the file handling in _log/_op but for sync process mode.
        """
        if self._sync_manager is None:
            return

        # Load the file (converts in-memory data to disk if needed)
        if isinstance(file_obj, (Artifact, Text, Image, Audio, Video)):
            file_obj.load(self.settings.get_dir())

        # Copy to run directory (if not already there)
        file_obj._mkcopy(self.settings.get_dir())

        # Enqueue for upload
        if file_obj._path is not None:
            self._sync_manager.enqueue_file(
                local_path=file_obj._path,
                file_name=file_obj._name,
                file_ext=file_obj._ext,
                file_type=file_obj._type,
                file_size=file_obj._stat.st_size,
                log_name=log_name,
                timestamp_ms=timestamp_ms,
                step=self._step,
                caption=file_obj._caption,
                sample_index=sample_index,
            )
            logger.debug(
                f'{tag}: enqueued file {file_obj._name}{file_obj._ext} for sync'
            )
        else:
            logger.warning(
                f'{tag}: Cannot enqueue file for sync - path is None after load'
            )

    def finish(self, code: Union[int, None] = None) -> None:
        """Finish logging and mark the run as a terminal status on the server.

        Flushes pending metrics, tears down local resources, and calls
        ``update_status`` on the server so the run transitions out of the
        active state. Idempotent and registered via ``atexit`` in
        ``__init__``.

        Callers attaching to an already-active run from a short-lived process
        (e.g., writing eval metrics to an ongoing training run) should prefer
        :meth:`close`, which performs the same local teardown without
        changing the server-side run status.
        """
        with self._finish_lock:
            if self._finished:
                return
            self._finished = True

        self._teardown(code, update_status=True)

    def close(self, code: Union[int, None] = None) -> None:
        """Release local resources without changing the server-side run status.

        Performs the same local cleanup as :meth:`finish` (stops the monitor,
        drains the sync manager, closes HTTP clients) but does **not** call
        ``update_status`` on the server — the run stays in its current state
        (typically "running") rather than being marked as a terminal status.

        Also unregisters the ``atexit``-registered ``finish`` hook so
        interpreter shutdown won't implicitly mark the run complete after
        the caller has released the op.

        Useful when a short-lived process resumes an active run just to
        append data (e.g., an eval job writing metrics to an ongoing
        training run) and must not interfere with its lifecycle.
        Idempotent.
        """
        with self._finish_lock:
            if self._finished:
                return
            self._finished = True

        self._teardown(code, update_status=False)

    def _teardown(self, code: Union[int, None], update_status: bool) -> None:
        """Shared teardown used by :meth:`finish` and :meth:`close`.

        When ``update_status`` is True, notifies the server that the run has
        reached a terminal state (and marks it FAILED if teardown itself
        raises). When False, the run's server-side status is left untouched
        and local errors are merely logged.
        """
        # Once teardown is underway, the atexit-registered finish() is no
        # longer needed. Unregistering here centralises the cleanup so both
        # finish() and close() self-unregister; it also prevents the atexit
        # hook from firing an idempotent (but wasteful) second finish() at
        # interpreter shutdown after close() has already detached the op.
        try:
            atexit.unregister(self.finish)
        except Exception:
            pass

        # In DDP/distributed, don't block waiting for sync - it causes deadlocks
        # because all ranks must progress together for collective operations
        is_distributed = _is_distributed_environment()

        try:
            # Stop the monitor (system metrics and heartbeats)
            self._monitor.stop(code)

            # Push pending console-capture lines into the sync store while
            # the sync process can still upload them — anything enqueued
            # after the drain below stays in SQLite unuploaded (the capture
            # layers batch up to 0.2s of output).
            if self._sync_manager is not None:
                flush_console_buffers()

            # Handle sync process shutdown
            if self._sync_manager is not None:
                if is_distributed:
                    # DDP mode: signal shutdown but don't wait
                    # This prevents deadlocks in collective operations
                    logger.debug(
                        f'{tag}: DDP environment detected, '
                        f'signaling sync shutdown without waiting'
                    )
                    self._sync_manager.stop(wait=False)
                else:
                    # Normal mode: wait for sync to complete
                    logger.debug(f'{tag}: stopping sync process manager')
                    sync_completed = self._sync_manager.stop(
                        timeout=self.settings.sync_process_shutdown_timeout,
                        wait=True,
                    )
                    if not sync_completed:
                        pending = self._sync_manager.get_pending_count()
                        logger.warning(
                            f'{tag}: Sync did not complete within timeout, '
                            f'{pending} records may not have been uploaded. '
                            f'Data is preserved in {self._sync_manager.db_path}'
                        )
                self._sync_manager.close()
                self._sync_manager = None

            # Update run status on server (only when finishing)
            if update_status and self._iface:
                self._iface.update_status()

            # Clean up data store if used (legacy mode)
            if self._store:
                self._store.stop()

            # Close HTTP clients
            if self._iface:
                self._iface.close()

            if update_status:
                # Print URL where users can view the completed run
                logger.info(f'{tag}: {self._view_run_message()}')
            else:
                logger.debug(f'{tag}: closed (run status unchanged)')
        except (Exception, KeyboardInterrupt) as e:
            _sentry.capture_exception(e)
            if update_status:
                self.settings._op_status = signal.SIGINT.value
                if self._iface:
                    self._iface._update_status(
                        self.settings,
                        trace={
                            'type': e.__class__.__name__,
                            'message': str(e),
                            'frames': [
                                {
                                    'filename': frame.filename,
                                    'lineno': frame.lineno,
                                    'name': frame.name,
                                    'line': frame.line,
                                }
                                for frame in traceback.extract_tb(e.__traceback__)
                            ],
                            'trace': traceback.format_exc(),
                        },
                    )
            logger.critical('%s: interrupted %s', tag, e)
            # Re-raise user-initiated termination so the process actually
            # exits as the user expects. Post-cleanup (sentry flush,
            # teardown_logger, pluto.ops mutation) still runs via the
            # finally block below.
            if isinstance(e, KeyboardInterrupt):
                raise
        finally:
            _sentry.flush()
            logger.debug(f'{tag}: finished' if update_status else f'{tag}: closed')
            teardown_logger(logger, console=logging.getLogger('console'))

            self.settings.meta = []
            if pluto.ops is not None:
                pluto.ops = [
                    op for op in pluto.ops if op.settings._op_id != self.settings._op_id
                ]  # TODO: make more efficient
                if not pluto.ops:
                    _unregister_excepthook()
                    _unregister_sigterm_handler()

    def watch(self, module, **kwargs):
        from .compat.torch import _watch_torch

        if any(
            b.__module__.startswith(
                (
                    'torch.nn',
                    'lightning.pytorch',
                    'pytorch_lightning.core.module',
                    'transformers.models',
                )
            )
            for b in module.__class__.__bases__
        ):
            return _watch_torch(module, op=self, **kwargs)
        else:
            logger.error(f'{tag}: unsupported module type {module.__class__.__name__}')
            return None

    def add_tags(self, tags: Union[str, List[str]]) -> None:
        """
        Add tags to the current run.

        Args:
            tags: Single tag string or list of tag strings to add

        Example:
            run.add_tags('experiment')
            run.add_tags(['production', 'v2'])
        """
        if isinstance(tags, str):
            tags = [tags]

        for tag_item in tags:
            if tag_item not in self.tags:
                self.tags.append(tag_item)

        logger.debug(f'{tag}: added tags: {tags}')

        # Sync full tags array to server
        if self._sync_manager is not None:
            timestamp_ms = int(time.time() * 1000)
            self._sync_manager.enqueue_tags(self.tags, timestamp_ms)
        elif self._iface:
            try:
                self._iface._update_tags(self.tags)
            except PlutoRequestError as e:
                # Server rejected the tags update (validation error) — this is
                # not a transient failure, so surface it rather than hide it.
                logger.warning(f'{tag}: server rejected tags update: {e}')
            except Exception as e:
                logger.debug(f'{tag}: failed to sync tags to server: {e}')

    def remove_tags(self, tags: Union[str, List[str]]) -> None:
        """
        Remove tags from the current run.

        Args:
            tags: Single tag string or list of tag strings to remove

        Example:
            run.remove_tags('experiment')
            run.remove_tags(['v1', 'old'])
        """
        if isinstance(tags, str):
            tags = [tags]

        for tag_item in tags:
            if tag_item in self.tags:
                self.tags.remove(tag_item)

        logger.debug(f'{tag}: removed tags: {tags}')

        # Sync full tags array to server
        if self._sync_manager is not None:
            timestamp_ms = int(time.time() * 1000)
            self._sync_manager.enqueue_tags(self.tags, timestamp_ms)
        elif self._iface:
            try:
                self._iface._update_tags(self.tags)
            except PlutoRequestError as e:
                # Server rejected the tags update (validation error) — this is
                # not a transient failure, so surface it rather than hide it.
                logger.warning(f'{tag}: server rejected tags update: {e}')
            except Exception as e:
                logger.debug(f'{tag}: failed to sync tags to server: {e}')

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update config on the current run.

        Config updates are merged with existing config (new keys override existing).

        Args:
            config: Dictionary of config key-value pairs to add/update

        Example:
            run.update_config({'epochs': 100})
            run.update_config({'lr': 0.01, 'model': 'resnet50'})
        """
        config = to_native_config(config)
        if self.config is None:
            self.config = {}
        self.config = deep_merge(self.config, config)

        logger.debug(f'{tag}: updated config: {config}')

        # Sync config to server
        if self._sync_manager is not None:
            timestamp_ms = int(time.time() * 1000)
            self._sync_manager.enqueue_config(config, timestamp_ms)
        elif self._iface:
            try:
                self._iface._update_config(config)
            except Exception as e:
                logger.debug(f'{tag}: failed to sync config to server: {e}')

    @property
    def resumed(self) -> bool:
        """
        Whether this run was resumed from an existing run.

        Returns True if a run with the same run_id already existed and this
        process attached to it (Neptune-style multi-node resume).
        """
        return self._resumed

    @property
    def run_id(self) -> Optional[str]:
        """
        The user-provided run ID for multi-node distributed training.

        This is the external ID that can be shared across multiple processes
        to log to the same run. Returns None if no run_id was provided.
        """
        return self.settings._external_id

    @property
    def id(self) -> Optional[int]:
        """
        The server-assigned numeric run ID.
        """
        return self.settings._op_id

    @property
    def fork_run_id(self) -> Optional[int]:
        """
        The resolved parent run ID this run was forked from.

        May differ from the requested fork_run_id due to lineage resolution.
        Returns None if this run was not forked.
        """
        return self._fork_run_id

    @property
    def fork_step(self) -> Optional[int]:
        """
        The step at which this run was forked from its parent.

        Returns None if this run was not forked.
        """
        return self._fork_step

    def alert(
        self,
        message=None,
        title=__name__.split('.')[0],
        level='INFO',
        wait=0,
        url=None,
        remote=True,
        **kwargs,
    ):
        # TODO: remove legacy compat
        message = kwargs.get('text', message)
        wait = kwargs.get('wait_duration', wait)
        kwargs['email'] = kwargs.get('email', True)

        url = url or self.settings.url_webhook or None

        t = time.time()
        time.sleep(wait)
        if logging._nameToLevel.get(level) is not None:
            logger.log(logging._nameToLevel[level], f'{tag}: {title}: {message}')
        if remote or not url:  # force remote alert
            self._iface._post_v1(
                self.settings.url_alert,
                self._iface.headers,
                make_compat_alert_v1(
                    self.settings, t, message, title, level, url, **kwargs
                ),
                client=self._iface.client,
            ) if self._iface else None
        else:
            self._iface._post_v1(
                url,
                {'Content-Type': 'application/json'},
                make_compat_webhook_v1(
                    t, level, title, message, self._step, self.settings.url_view
                ),
                self._iface.client,  # TODO: check client
            ) if self._iface else logger.warning(
                f'{tag}: alert not sent since interface is disabled'
            )

    def _worker(self, stop: Callable[[], bool]) -> None:
        while not stop() or not self._queue.empty():
            try:
                # if queue seems empty, wait for x_internal_check_process before it
                # considers it empty to save compute
                self._log(
                    *self._queue.get(
                        block=True, timeout=self.settings.x_internal_check_process
                    )
                )
            except queue.Empty:
                continue
            except sqlite3.OperationalError as e:
                logger.warning('%s: transient database error (will retry): %s', tag, e)
                time.sleep(self.settings.x_internal_check_process)  # debounce
            except Exception as e:
                time.sleep(self.settings.x_internal_check_process)  # debounce
                logger.critical('%s: failed: %s', tag, e)

    def _log(
        self,
        data: Mapping[str, Any],
        step: Optional[int],
        t: Optional[float] = None,
    ) -> None:
        if not isinstance(data, Mapping):
            e = ValueError(
                'unsupported type for logged data: '
                f'{type(data).__name__}, expected dict'
            )
            logger.critical('%s: failed: %s', tag, e)
            raise e
        if any(not isinstance(k, str) for k in data.keys()):
            e = ValueError('unsupported type for key in dict of logged data')
            logger.critical('%s: failed: %s', tag, e)
            raise e

        self._step = self._step + 1 if step is None else step
        t = time.time() if t is None else t

        numbers: LoggedNumbers = {}
        datasets: LoggedData = {}
        files: LoggedFiles = {}
        nm: MetaNames = []
        fm: MetaFiles = {}
        for k, v in data.items():
            k = get_char(k)  # TODO: remove validation

            if isinstance(v, list):
                nm, fm = self._m(nm, fm, k, v[0])
                for e in v:
                    numbers, datasets, files = self._op(numbers, datasets, files, k, e)
            else:
                nm, fm = self._m(nm, fm, k, v)
                numbers, datasets, files = self._op(numbers, datasets, files, k, v)

        # d = dict_to_json(d)  # TODO: add serialisation
        # Store data locally (legacy mode when sync process is disabled)
        self._store.insert(
            num=numbers, data=datasets, file=files, timestamp=t, step=self._step
        ) if self._store else None

    def _m(
        self, nm: MetaNames, fm: MetaFiles, k: str, v: Any
    ) -> Tuple[MetaNames, MetaFiles]:
        if k not in self.settings.meta:
            if isinstance(v, File) or isinstance(v, Data):
                if v.__class__.__name__ not in fm:
                    fm[v.__class__.__name__] = []
                fm[v.__class__.__name__].append(k)
            elif isinstance(v, (int, float)) or v.__class__.__name__ == 'Tensor':
                nm.append(k)
            self.settings.meta.append(k)
            # d[f"{self.settings.x_meta_label}{k}"] = 0
            logger.debug(f'{tag}: added {k} at step {self._step}')
        return nm, fm

    def _op(
        self,
        n: LoggedNumbers,
        d: LoggedData,
        f: LoggedFiles,
        k: str,
        v: Any,
    ) -> Tuple[LoggedNumbers, LoggedData, LoggedFiles]:
        if isinstance(v, File):
            if (
                isinstance(v, Artifact)
                or isinstance(v, Text)
                or isinstance(v, Image)
                or isinstance(v, Audio)
                or isinstance(v, Video)
            ):
                v.load(self.settings.get_dir())
            # TODO: add step to serialise data for files
            v._mkcopy(self.settings.get_dir())  # key independent
            # d[k] = int(v._id, 16)
            if k not in f:
                f[k] = []
            f[k].append(v)
        elif isinstance(v, Data):
            if k not in d:
                d[k] = []
            d[k].append(v)
        else:
            n[k] = get_val(v)
        return n, d, f
