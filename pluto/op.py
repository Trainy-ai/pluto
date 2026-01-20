import atexit
import logging
import os
import queue
import signal
import sys
import threading
import time
import traceback
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import pluto

from .api import (
    make_compat_alert_v1,
    make_compat_monitor_v1,
    make_compat_start_v1,
    make_compat_trigger_v1,
    make_compat_webhook_v1,
)
from .auth import login
from .data import Data
from .file import Artifact, Audio, File, Image, Text, Video
from .iface import ServerInterface
from .log import setup_logger, teardown_logger
from .store import DataStore
from .sync import SyncProcessManager
from .sys import System
from .util import get_char, get_val, print_url, to_json

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


# Signal handling state for graceful shutdown (Ctrl+C / SIGTERM)
_signal_count = 0
_signal_lock = threading.Lock()
_original_sigint_handler = None
_original_sigterm_handler = None
_signal_handler_registered = False

# Map signal numbers to names for logging
_SIGNAL_NAMES = {
    signal.SIGINT: 'SIGINT',
    signal.SIGTERM: 'SIGTERM',
}


def _shutdown_handler(signum, frame):
    """
    Handle SIGINT (Ctrl+C) and SIGTERM (K8s termination) with two-stage shutdown:
    - First signal: Graceful shutdown - finish all active runs
    - Second signal: Force exit immediately
    """
    global _signal_count

    with _signal_lock:
        _signal_count += 1
        count = _signal_count

    sig_name = _SIGNAL_NAMES.get(signum, f'signal {signum}')

    if count == 1:
        msg = f'{tag}: Received {sig_name}, shutting down gracefully...'
        if signum == signal.SIGINT:
            msg += ' (press Ctrl+C again to force exit)'
        logger.warning(msg)
        # Print to stderr as well in case logging is not visible
        print(f'\n{msg}', file=sys.stderr)
        # Finish all active ops
        if pluto.ops:
            # Copy list to avoid modification during iteration
            for op in list(pluto.ops):
                try:
                    op.finish(code=signum)
                except Exception as e:
                    logger.debug(f'{tag}: Error during graceful shutdown: {e}')
        # Exit with appropriate status code
        sys.exit(128 + signum)
    else:
        # Second signal - force exit immediately
        print(f'\n{tag}: Force exiting...', file=sys.stderr)
        os._exit(128 + signum)


def _register_signal_handler():
    """Register SIGINT and SIGTERM handlers if in main thread."""
    global _signal_handler_registered
    global _original_sigint_handler, _original_sigterm_handler

    if _signal_handler_registered:
        return

    # Signal handlers can only be registered from the main thread
    if threading.current_thread() is not threading.main_thread():
        logger.debug(f'{tag}: Skipping signal handler registration (not main thread)')
        return

    try:
        _original_sigint_handler = signal.signal(signal.SIGINT, _shutdown_handler)
        _original_sigterm_handler = signal.signal(signal.SIGTERM, _shutdown_handler)
        _signal_handler_registered = True
        logger.debug(f'{tag}: Registered SIGINT/SIGTERM handlers for graceful shutdown')
    except (ValueError, OSError) as e:
        # ValueError: signal only works in main thread
        # OSError: can happen in some embedded environments
        logger.debug(f'{tag}: Could not register signal handler: {e}')


def _unregister_signal_handler():
    """Restore original signal handlers when no more Ops are active."""
    global _signal_handler_registered
    global _original_sigint_handler, _original_sigterm_handler

    if not _signal_handler_registered:
        return

    # Signal handlers can only be modified from the main thread
    if threading.current_thread() is not threading.main_thread():
        return

    try:
        if _original_sigint_handler is not None:
            signal.signal(signal.SIGINT, _original_sigint_handler)
        if _original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, _original_sigterm_handler)
        _signal_handler_registered = False
        logger.debug(f'{tag}: Restored original SIGINT/SIGTERM handlers')
    except (ValueError, OSError) as e:
        logger.debug(f'{tag}: Could not restore signal handler: {e}')


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
        for attr in ['_thread', '_thread_monitor']:
            thread = getattr(self, attr)
            if thread is not None:
                thread.join(timeout=self.op.settings.x_thread_join_timeout_seconds)
                if thread.is_alive():
                    logger.warning(
                        f'{tag}: Thread {thread.name} did not terminate, '
                        'continuing anyway'
                    )
                setattr(self, attr, None)
        if isinstance(code, int):
            self.op.settings._op_status = code
        elif self.op.settings._op_status == -1:
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
                if self.op._iface:
                    r = self.op._iface._post_v1(
                        self.op.settings.url_trigger,
                        self.op._iface.headers,
                        make_compat_trigger_v1(self.op.settings),
                        client=self.op._iface.client,
                    )
                    if hasattr(r, 'json') and r.json()['status'] == 'CANCELLED':
                        logger.critical(f'{tag}: server finished run')
                        os._exit(signal.SIGINT.value)  # TODO: do a more graceful exit
            except Exception as e:
                logger.critical('%s: failed: %s', tag, e)
            time.sleep(self.op.settings.x_sys_sampling_interval)


class Op:
    def __init__(self, config, settings, tags=None) -> None:
        self.config = config
        self.settings = settings
        self.tags: List[str] = tags if tags else []  # Use provided tags or empty list
        self._monitor = OpMonitor(op=self)
        self._resumed: bool = False  # Whether this run was resumed (multi-node)
        self._sync_manager: Optional[SyncProcessManager] = None

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
            r = tmp_iface._post_v1(
                self.settings.url_start,  # create-run
                tmp_iface.headers,
                make_compat_start_v1(
                    self.config, self.settings, self.settings._sys.get_info(), self.tags
                ),
                client=tmp_iface.client_api,
            )
            if not r:
                raise ConnectionError(
                    'Failed to create or resume run. Check connection to Pluto server.'
                )
            response_data = r.json()
            self.settings.url_view = response_data['url']
            self.settings._op_id = response_data['runId']
            self._resumed = response_data.get('resumed', False)
            if self._resumed:
                logger.info(f'{tag}: resumed run {str(self.settings._op_id)}')
                logger.warning(
                    f'{tag}: Run was resumed via run_id. The `name` parameter '
                    f'is ignored for resumed runs - the original run name is '
                    f'preserved. For multi-node, use the same name across all ranks.'
                )
            else:
                logger.info(f'{tag}: started run {str(self.settings._op_id)}')

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
            self._iface._update_meta(sys_metric_names)

        # Print URL where users can view the run
        logger.info(f'{tag}: View run at {print_url(self.settings.url_view)}')

        # Register signal handler for graceful Ctrl+C shutdown
        # (unless disabled, e.g., when running under a compat layer like Neptune)
        if not self.settings.x_disable_signal_handlers:
            _register_signal_handler()

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
            self._log_via_sync(data=data, step=step)
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

            for item in items:
                self._process_log_item_sync(k, item, metrics, timestamp_ms)

        if metrics:
            self._sync_manager.enqueue_metrics(metrics, timestamp_ms, self._step)

        # Register new metric/file names with server (required for dashboard display)
        if (new_metric_names or new_file_meta) and self._iface:
            self._iface._update_meta(num=new_metric_names, df=dict(new_file_meta))

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
            self._enqueue_file_sync(key, value, timestamp_ms)
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
            )
            logger.debug(
                f'{tag}: enqueued file {file_obj._name}{file_obj._ext} for sync'
            )
        else:
            logger.warning(
                f'{tag}: Cannot enqueue file for sync - path is None after load'
            )

    def finish(self, code: Union[int, None] = None) -> None:
        """Finish logging"""
        # Make finish() idempotent - can be called multiple times safely
        # (e.g., from signal handler and atexit)
        with self._finish_lock:
            if self._finished:
                return
            self._finished = True

        # Detect if we're being preempted (SIGTERM) vs normal exit
        # During preemption, don't block - let sync process handle it
        is_preemption = code == signal.SIGTERM

        # In DDP/distributed, don't block waiting for sync - it causes deadlocks
        # because all ranks must progress together for collective operations
        is_distributed = _is_distributed_environment()

        try:
            # Stop the monitor (system metrics and heartbeats)
            self._monitor.stop(code)

            # Handle sync process shutdown
            if self._sync_manager is not None:
                if is_preemption:
                    # Preemption mode: signal shutdown but don't wait
                    # This prevents blocking during pod termination
                    logger.debug(
                        f'{tag}: preemption detected (SIGTERM), '
                        f'signaling sync shutdown without waiting'
                    )
                    self._sync_manager.stop(wait=False)
                elif is_distributed:
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

            # Update run status on server
            if self._iface:
                self._iface.update_status()

            # Clean up data store if used (legacy mode)
            if self._store:
                self._store.stop()

            # Close HTTP clients
            if self._iface:
                self._iface.close()

            # Print URL where users can view the completed run
            logger.info(f'{tag}: View run at {print_url(self.settings.url_view)}')
        except (Exception, KeyboardInterrupt) as e:
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
        logger.debug(f'{tag}: finished')
        teardown_logger(logger, console=logging.getLogger('console'))

        self.settings.meta = []
        if pluto.ops is not None:
            pluto.ops = [
                op for op in pluto.ops if op.settings._op_id != self.settings._op_id
            ]  # TODO: make more efficient
            # Restore original signal handlers when last op finishes
            if not pluto.ops:
                _unregister_signal_handler()

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
        if self.config is None:
            self.config = {}
        self.config.update(config)

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
