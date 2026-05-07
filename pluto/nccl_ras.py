import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'NcclRAS'

DEFAULT_RAS_HOST = '127.0.0.1'
DEFAULT_RAS_PORT = 28028
SOCKET_TIMEOUT = 10.0
NCCLRAS_TIMEOUT = 10.0
SOCKET_TEXT_COMMAND = b'verbose status\n'

MODE_NCCLRAS_JSON = 'ncclras-json'
MODE_SOCKET_TEXT = 'socket-text'


def _is_rank_zero() -> bool:
    """Return True on the head process. Defaults to True when not distributed."""
    for var in ('RANK', 'SLURM_PROCID'):
        v = os.environ.get(var)
        if v is not None and v.lstrip('-').isdigit():
            return int(v) == 0
    v = os.environ.get('LOCAL_RANK')
    if v is not None and v.lstrip('-').isdigit():
        return int(v) == 0
    return True


def _parse_addr(addr: str) -> Tuple[str, int]:
    if addr and ':' in addr:
        host, _, port_str = addr.rpartition(':')
        try:
            return (host or DEFAULT_RAS_HOST), int(port_str)
        except ValueError:
            pass
    return DEFAULT_RAS_HOST, DEFAULT_RAS_PORT


def _resolve_addr(settings) -> Tuple[str, int]:
    # NCCL_RAS_ADDR env wins, matching NCCL's own behavior.
    env = os.environ.get('NCCL_RAS_ADDR')
    if env:
        return _parse_addr(env)
    return _parse_addr(getattr(settings, 'x_nccl_ras_addr', ''))


def _detect_ncclras_json() -> Optional[str]:
    """If `ncclras` is on PATH and supports `-f json`, return its absolute path.
    Otherwise return None and we'll fall back to raw socket text mode.
    """
    binary = shutil.which('ncclras')
    if not binary:
        return None
    try:
        r = subprocess.run(
            [binary, '--help'],
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug(f'{tag}: ncclras --help failed: {e}')
        return None
    blob = (r.stdout + r.stderr).decode('utf-8', errors='replace').lower()
    # JSON output was added in NCCL 2.28.7 via `-f json`.
    if '-f' in blob and 'json' in blob:
        return binary
    return None


def _query_ncclras_json(binary: str, host: str, port: int) -> str:
    """Run `ncclras -f json -v` and return its stdout. JSON mode dumps the
    full RAS state; -v requests verbose detail.
    """
    cmd = [binary, '-f', 'json', '-v', '-a', f'{host}:{port}']
    r = subprocess.run(cmd, capture_output=True, timeout=NCCLRAS_TIMEOUT)
    if r.returncode != 0:
        stderr = r.stderr.decode('utf-8', errors='replace')[:200]
        raise RuntimeError(f'ncclras exited {r.returncode}: {stderr}')
    return r.stdout.decode('utf-8', errors='replace')


def _query_socket_text(host: str, port: int, timeout: float = SOCKET_TIMEOUT) -> str:
    """Fallback: open the RAS socket directly and request `verbose status`."""
    chunks: List[bytes] = []
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(SOCKET_TEXT_COMMAND)
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        while True:
            try:
                data = sock.recv(8192)
            except socket.timeout:
                break
            if not data:
                break
            chunks.append(data)
    return b''.join(chunks).decode('utf-8', errors='replace')


class NcclRasMonitor:
    """Polls the local NCCL RAS socket on rank 0 and ships output through the
    same console-log pipeline used for stdout/stderr.

    RAS gossips OOB across all ranks, so a single rank's view is global.
    Prefers `ncclras -f json -v` (NCCL >= 2.28.7) and falls back to the raw
    socket with `verbose status` on older NCCL.
    """

    def __init__(self, op) -> None:
        self.op = op
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._line_count = 0
        self._consecutive_failures = 0

    def _enabled(self) -> bool:
        if not getattr(self.op.settings, 'x_nccl_ras_enabled', False):
            return False
        return _is_rank_zero()

    def start(self) -> None:
        if not self._enabled() or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._worker,
            name='pluto-nccl-ras',
            daemon=True,
        )
        self._thread.start()
        logger.debug(f'{tag}: monitor started')

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        timeout = getattr(self.op.settings, 'x_thread_join_timeout_seconds', 30)
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(f'{tag}: thread did not terminate within {timeout}s')
        self._thread = None

    def _worker(self) -> None:
        host, port = _resolve_addr(self.op.settings)
        log_type = getattr(self.op.settings, 'x_nccl_ras_log_type', 'RAS')
        interval = self.op.settings.x_sys_sampling_interval

        ncclras_bin = _detect_ncclras_json()
        if ncclras_bin:
            mode = MODE_NCCLRAS_JSON
            logger.debug(f'{tag}: using {mode} via {ncclras_bin}')
        else:
            mode = MODE_SOCKET_TEXT
            logger.debug(f'{tag}: ncclras JSON unavailable; using {mode}')

        while not self._stop_event.is_set():
            try:
                if mode == MODE_NCCLRAS_JSON:
                    output = _query_ncclras_json(ncclras_bin, host, port)
                else:
                    output = _query_socket_text(host, port)
                self._consecutive_failures = 0
                if output:
                    self._enqueue_output(output, log_type, mode)
            except OSError as e:
                # ConnectionRefused / unreachable = RAS disabled or not yet up.
                self._consecutive_failures += 1
                if (
                    self._consecutive_failures == 1
                    or self._consecutive_failures % 10 == 0
                ):
                    logger.debug(
                        f'{tag}: poll failed ({host}:{port}): {e} '
                        f'(failure #{self._consecutive_failures})'
                    )
            except subprocess.TimeoutExpired as e:
                self._consecutive_failures += 1
                logger.debug(f'{tag}: ncclras timed out: {e}')
            except Exception as e:
                self._consecutive_failures += 1
                logger.debug(f'{tag}: unexpected error: {e}')
            self._stop_event.wait(timeout=interval)

    def _enqueue_output(self, output: str, log_type: str, mode: str) -> None:
        sync_manager = getattr(self.op, '_sync_manager', None)
        if sync_manager is None:
            return
        timestamp_ms = int(time.time() * 1000)
        batch = []
        if mode == MODE_NCCLRAS_JSON:
            # Treat the entire JSON document as a single record so the UI can
            # parse it whole rather than reassembling split lines.
            self._line_count += 1
            batch.append((output, log_type, timestamp_ms, self._line_count))
        else:
            for line in output.splitlines():
                if not line.strip():
                    continue
                self._line_count += 1
                batch.append((line, log_type, timestamp_ms, self._line_count))
        if not batch:
            return
        try:
            sync_manager.enqueue_console_batch(batch)
        except Exception as e:
            logger.debug(f'{tag}: enqueue failed: {e}')
