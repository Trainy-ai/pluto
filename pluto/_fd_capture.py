"""
File-descriptor-level console capture (the wandb console="redirect" approach).

pluto.log.ConsoleHandler captures console output by swapping the
``sys.stdout``/``sys.stderr`` *variables* — which misses every writer that
bound the original stream object before ``pluto.init()`` ran. The dominant
real-world case is ``logging.StreamHandler``: CPython stores
``self.stream = sys.stderr`` at handler construction, so frameworks that
configure logging at process start (torchtitan, most torch trainers) write
straight to the raw fd and the run's console section stays empty.

``FdCapture`` intercepts at the OS level instead: it ``dup2``s a pipe over
fd 1/2, and a daemon reader thread tees every byte back to the saved real
fd (so the terminal stays live) while line-buffering the captured copy into
``SyncProcessManager.enqueue_console_batch`` — same tuple contract, rank
prefixing, and sanitization as ``ConsoleHandler``. This catches logging
handlers, C extensions, and child processes that inherit the fd.

Shutdown notes (``stop()``):
- A sentinel byte-sequence is written into the pipe before the real fd is
  restored. When the reader sees it, everything written before ``stop()``
  has been enqueued — ``stop()`` waits on that (bounded), giving a
  deterministic flush without racing the reader.
- The reader thread is NOT joined/killed. Child processes forked while
  capture was active (e.g. DataLoader workers) inherit the pipe write end;
  if nobody drained it, their writes would block once the pipe buffer
  fills. The daemon reader stays behind in tee-only mode, forwarding any
  such stragglers to the real fd until EOF or interpreter exit.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(f'{__name__.split(".")[0]}')

# Written into the pipe by stop() to mark "everything before this is
# pre-stop output". Control bytes make a collision with real output
# effectively impossible; writes <= PIPE_BUF are atomic so it can't
# interleave with a concurrent writer's bytes.
_STOP_SENTINEL = b'\x00\x1dpluto:fdcap:flush\x1d\x00'


class FdCapture:
    # Mirrors pluto.log.ConsoleHandler's batching to keep SQLite write
    # pressure low while still delivering logs promptly.
    _FLUSH_SIZE = 50
    _FLUSH_INTERVAL = 0.2  # seconds
    _READ_CHUNK = 8192

    def __init__(
        self,
        fd: int,
        level: int,
        sync_manager: Any,
        sanitizer: Optional[Any] = None,
    ):
        self.fd = fd
        self.level = level
        self.sync_manager = sync_manager
        self.sanitizer = sanitizer
        self.count = 0

        self._orig_fd: Optional[int] = None
        self._read_fd: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stopped = False

        # Guards line/batch state shared between the reader thread and a
        # stop() caller doing a last-resort flush.
        self._state_lock = threading.Lock()
        self._enqueue_enabled = True
        self._partial_line = ''
        self._log_buffer: List[Tuple[str, str, int, int]] = []
        self._last_flush = 0.0
        # Set once the reader has flushed everything written before stop().
        self._flushed = threading.Event()

        # Same rank tagging as ConsoleHandler: only the captured copy is
        # prefixed, never the tee (torchrun prefixes the terminal itself).
        rank = os.environ.get('RANK')
        self._rank_prefix = f'[rank{rank}] ' if rank is not None else ''

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Redirect self.fd into a pipe drained by a daemon reader thread."""
        if self._started or self._stopped:
            return
        self._orig_fd = os.dup(self.fd)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, self.fd)
        os.close(write_fd)
        self._read_fd = read_fd
        self._started = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f'pluto-fdcap-{self.fd}',
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Restore the real fd and flush everything captured so far.

        Idempotent. Never raises. The reader thread is left running in
        tee-only drain mode (see module docstring).
        """
        if not self._started or self._stopped:
            return
        self._stopped = True

        # 1. Sentinel into the pipe while self.fd still points at it — it
        #    lands after all pre-stop output. Written from a helper thread:
        #    if the pipe buffer is full (reader wedged on a blocked tee), a
        #    direct write would hang stop() — and stop() runs in finish()
        #    paths that must never block (DDP).
        def _write_sentinel() -> None:
            try:
                os.write(self.fd, _STOP_SENTINEL)
            except OSError:
                pass

        writer = threading.Thread(target=_write_sentinel, daemon=True)
        writer.start()
        writer.join(timeout=min(0.5, timeout))

        # 2. Point the fd back at the real destination — new writes bypass
        #    the pipe from here on.
        try:
            if self._orig_fd is not None:
                os.dup2(self._orig_fd, self.fd)
        except OSError as e:
            logger.debug('fdcap: failed to restore fd %d: %s', self.fd, e)

        # 3. Wait for the reader to reach the sentinel and flush. On
        #    timeout, flush whatever state we can see ourselves.
        if not self._flushed.wait(timeout):
            with self._state_lock:
                self._flush_locked(drain_partial=True)
                self._enqueue_enabled = False
        # _orig_fd stays open: the drain-mode reader tees stragglers to it.

    # -- reader thread -----------------------------------------------------

    def _reader_loop(self) -> None:
        assert self._read_fd is not None
        held = b''  # possible partial sentinel at a chunk boundary
        while True:
            try:
                chunk = os.read(self._read_fd, self._READ_CHUNK)
            except OSError:
                break
            if not chunk:  # EOF: every write end (incl. children's) closed
                break
            data = held + chunk
            held = b''

            idx = data.find(_STOP_SENTINEL)
            if idx != -1:
                before = data[:idx]
                after = data[idx + len(_STOP_SENTINEL) :]
                self._tee(before)
                self._ingest(before)
                with self._state_lock:
                    self._flush_locked(drain_partial=True)
                    self._enqueue_enabled = False
                self._flushed.set()
                self._tee(after)
                continue  # drain mode: _ingest below is a no-op now

            # A sentinel prefix at the very end of the chunk may be the
            # sentinel split across reads — hold those bytes back until the
            # next read resolves it. Real output essentially never starts
            # with \x00, so this doesn't delay the tee in practice.
            cut = self._partial_sentinel_suffix(data)
            if cut:
                held = data[-cut:]
                data = data[:-cut]
            self._tee(data)
            self._ingest(data)
        if held:
            self._tee(held)
            self._ingest(held)
        with self._state_lock:
            self._flush_locked(drain_partial=True)
        self._flushed.set()
        try:
            os.close(self._read_fd)
        except OSError:
            pass

    @staticmethod
    def _partial_sentinel_suffix(data: bytes) -> int:
        """Length of the longest proper sentinel prefix that ends ``data``."""
        max_k = min(len(_STOP_SENTINEL) - 1, len(data))
        for k in range(max_k, 0, -1):
            if data.endswith(_STOP_SENTINEL[:k]):
                return k
        return 0

    def _tee(self, data: bytes) -> None:
        """Forward captured bytes to the real destination, completely."""
        if not data or self._orig_fd is None:
            return
        view = memoryview(data)
        try:
            while view:
                written = os.write(self._orig_fd, view)
                view = view[written:]
        except OSError:
            pass  # real terminal gone (e.g. closed pty) — keep capturing

    # -- line buffering (mirrors pluto.log.ConsoleHandler) ------------------

    def _ingest(self, data: bytes) -> None:
        if not data:
            return
        text = data.decode('utf-8', errors='replace')
        with self._state_lock:
            if not self._enqueue_enabled:
                return
            self._partial_line += text
            if '\n' in self._partial_line:
                *complete, self._partial_line = self._partial_line.split('\n')
                for line in complete:
                    self._emit_line_locked(line)
            if self._log_buffer and (
                len(self._log_buffer) >= self._FLUSH_SIZE
                or time.time() - self._last_flush >= self._FLUSH_INTERVAL
            ):
                self._flush_locked()

    def _emit_line_locked(self, line: str) -> None:
        if not line:  # do not log empty lines (matches ConsoleHandler)
            return
        self.count += 1
        if self._rank_prefix:
            line = self._rank_prefix + line
        if self.sanitizer:
            line = self.sanitizer.sanitize(line)
        log_type = logging._levelToName.get(self.level, 'INFO')
        self._log_buffer.append((line, log_type, int(time.time() * 1000), self.count))

    def _flush_locked(self, drain_partial: bool = False) -> None:
        if drain_partial and self._partial_line:
            self._emit_line_locked(self._partial_line)
            self._partial_line = ''
        if not self._log_buffer:
            return
        try:
            self.sync_manager.enqueue_console_batch(self._log_buffer)
        except Exception as e:
            logger.debug('fdcap: failed to flush console batch: %s', e)
        self._log_buffer = []
        self._last_flush = time.time()
