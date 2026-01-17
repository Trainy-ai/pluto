import atexit
import logging
import os
import queue
import signal
import sys
import threading
import time
import traceback
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
from .util import get_char, get_val, to_json

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'Operation'

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
                self.op._iface.publish(
                    num=make_compat_monitor_v1(self.op.settings._sys.monitor()),
                    timestamp=time.time(),
                    step=self.op._step,
                ) if self.op._iface else None
                r = (
                    self.op._iface._post_v1(
                        self.op.settings.url_trigger,
                        self.op._iface.headers,
                        make_compat_trigger_v1(self.op.settings),
                        client=self.op._iface.client,
                    )
                    if self.op._iface
                    else None
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
            setup_logger(
                settings=self.settings,
                logger=logger,
                console=logging.getLogger('console'),
            )  # global logger
            to_json(
                [self.settings._sys.get_info()], f'{self.settings.get_dir()}/sys.json'
            )

            # Initialize sync process manager if enabled
            if settings.sync_process_enabled:
                self._init_sync_manager()

        self._store: Optional[DataStore] = (
            DataStore(config=config, settings=settings)
            if not settings.disable_store and not settings.sync_process_enabled
            else None
        )
        self._iface: Optional[ServerInterface] = (
            ServerInterface(config=config, settings=settings)
            if not settings.disable_iface and not settings.sync_process_enabled
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
            'url_update_config': self.settings.url_update_config,
            'url_update_tags': self.settings.url_update_tags,
            'x_log_level': self.settings.x_log_level,
            'sync_process_flush_interval': (self.settings.sync_process_flush_interval),
            'sync_process_shutdown_timeout': (
                self.settings.sync_process_shutdown_timeout
            ),
            'sync_process_orphan_timeout': (self.settings.sync_process_orphan_timeout),
            'sync_process_retry_max': self.settings.sync_process_retry_max,
            'sync_process_retry_backoff': self.settings.sync_process_retry_backoff,
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
        else:
            # Use traditional thread-based approach
            self._iface.start() if self._iface else None
            self._iface._update_meta(
                list(make_compat_monitor_v1(self.settings._sys.monitor()).keys())
            ) if self._iface else None
            self._monitor.start()

        logger.debug(f'{tag}: started')

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
        # Use sync process if enabled
        if self._sync_manager is not None:
            self._log_via_sync(data=data, step=step)
        elif self.settings.mode == 'perf':
            self._queue.put((data, step), block=False)
        else:  # bypass queue
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

        # Extract numeric values for metrics
        metrics: Dict[str, Any] = {}
        for k, v in data.items():
            k = get_char(k)
            if isinstance(v, (int, float)):
                metrics[k] = v
            elif hasattr(v, 'item'):  # Tensor
                metrics[k] = v.item()
            # TODO: Handle File, Data types via separate queue

        if metrics:
            self._sync_manager.enqueue_metrics(metrics, timestamp_ms, self._step)

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

        try:
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
            else:
                # Traditional thread-based shutdown
                self._monitor.stop(code)
                # Wait for queue to drain with timeout
                drain_timeout = self.settings.x_thread_join_timeout_seconds
                drain_start = time.time()
                initial_size = self._queue.qsize()
                last_log_time = drain_start
                logger.debug(f'{tag}: waiting for queue to drain, {initial_size} items')
                while not self._queue.empty():
                    elapsed = time.time() - drain_start
                    if elapsed > drain_timeout:
                        remaining = self._queue.qsize()
                        logger.warning(
                            f'{tag}: Queue drain timeout after {drain_timeout}s, '
                            f'{remaining} items remaining (started with {initial_size})'
                        )
                        break
                    # Log progress every 5 seconds
                    if time.time() - last_log_time > 5:
                        current_size = self._queue.qsize()
                        logger.debug(
                            f'{tag}: queue drain progress: {current_size} items '
                            f'remaining ({initial_size - current_size} processed '
                            f'in {elapsed:.1f}s)'
                        )
                        last_log_time = time.time()
                    time.sleep(self.settings.x_internal_check_process)
                logger.debug(f'{tag}: queue drained, stopping store')
                self._store.stop() if self._store else None
                logger.debug(f'{tag}: store stopped, stopping interface')
                self._iface.stop() if self._iface else None  # fixed order
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
        self._store.insert(
            num=numbers, data=datasets, file=files, timestamp=t, step=self._step
        ) if self._store else None
        self._iface.publish(
            num=numbers, data=datasets, file=files, timestamp=t, step=self._step
        ) if self._iface else None
        self._iface._update_meta(num=nm, df=fm) if (nm or fm) and self._iface else None

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
