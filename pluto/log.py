import builtins
import logging
import os
import sys
import time

from .util import ANSI

logger = logging.getLogger(f'{__name__.split(".")[0]}')

_input = builtins.input
_stdout = sys.stdout
_stderr = sys.stderr

# Active fd-level captures (see pluto._fd_capture). Populated by
# setup_logger_file, stopped in teardown_logger.
_fd_captures: list = []

colors = {
    'DEBUG': ANSI.green,
    'INFO': ANSI.cyan,
    'WARNING': ANSI.yellow,
    'ERROR': ANSI.red,
    'CRITICAL': ANSI.purple,
}
styles = {
    'DEBUG': ' 💬 ',
    'INFO': ' 🚀 ',
    'WARNING': ' 🚨 ',
    'ERROR': ' ⛔ ',
    'CRITICAL': ' 🚫 ',
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        prefix = ANSI.bold + ANSI.cyan + f'{__name__.split(".")[0]}:' + ANSI.reset
        color = colors.get(record.levelname, '')
        style = styles.get(record.levelname, '')

        # TODO: remove legacy compat
        if record.msg.startswith('Operation'):
            prefix = '\n' + prefix

        return f'{prefix}{color}{style}{super().format(record)}{ANSI.reset}'


class ConsoleHandler:
    # Flush buffered console logs after this many lines or this many seconds,
    # whichever comes first.  Keeps SQLite write pressure low while still
    # delivering logs promptly.
    _FLUSH_SIZE = 50
    _FLUSH_INTERVAL = 0.2  # seconds

    def __init__(
        self,
        logger,
        sync_manager=None,
        level=logging.INFO,
        stream=sys.stdout,
        type='stdout',
        sanitizer=None,
    ):
        self.logger = logger
        self.sync_manager = sync_manager
        self.level = level
        self.stream = stream
        self.type = type
        self.count = 0
        self.sanitizer = sanitizer
        self._log_buffer: list = []
        self._last_flush = 0.0
        # Carry-over for partial writes that don't end at a line boundary.
        # Python's traceback printer (and rich) call write() with chunks
        # like just whitespace or a single character, so treating each
        # call as a complete line shreds tracebacks into one-char "lines".
        self._partial_line: str = ''
        # When running under torchrun, prepend the rank to captured lines
        # so the Pluto UI can distinguish rank N from rank M without the
        # user having to wrap every print() themselves. RANK is set by
        # torchrun in every child process; absent → no prefix (so single-
        # process and non-torch jobs keep their existing log format).
        # Note: only the captured copy is prefixed, not the pass-through
        # to self.stream — that lets torchrun add its own [defaultN]:
        # prefix to the terminal stream without double-prefixing.
        rank = os.environ.get('RANK')
        self._rank_prefix = f'[rank{rank}] ' if rank is not None else ''

    def _flush_log_buffer(self) -> None:
        """Flush buffered console log lines to the sync store in one batch."""
        if not self._log_buffer or self.sync_manager is None:
            self._log_buffer.clear()
            return
        try:
            self.sync_manager.enqueue_console_batch(self._log_buffer)
        except Exception as e:
            logger.debug('Failed to flush console log buffer: %s', e)
        self._log_buffer.clear()
        self._last_flush = time.time()

    def _emit_line(self, line: str) -> None:
        """Log one complete line through the sync buffer + the python logger."""
        if not line:  # do not log empty lines
            return
        self.count += 1
        timestamp_ms = int(time.time() * 1000)
        if self._rank_prefix:
            line = self._rank_prefix + line
        if self.sync_manager is not None:
            sanitized_line = self.sanitizer.sanitize(line) if self.sanitizer else line
            log_type = logging._levelToName.get(self.level, 'INFO')
            self._log_buffer.append(
                (sanitized_line, log_type, timestamp_ms, self.count)
            )
        self.logger.log(self.level, line)

    def write(self, buf: str) -> None:
        # Pass-through to the real stream first so terminal output is not
        # delayed by our line buffering.
        self.stream.write(buf)
        self.stream.flush()

        # Accumulate partial writes and only emit on real '\n' boundaries.
        # Splitting on '\n' specifically (not splitlines()) avoids breaking
        # on \v, \f, \x1c-\x1e, \x85, U+2028, U+2029 — chars that rich and
        # other styled-output libs use as internal segment separators.
        self._partial_line += buf
        if '\n' not in self._partial_line:
            return
        *complete, self._partial_line = self._partial_line.split('\n')
        for line in complete:
            self._emit_line(line)
        # Flush the buffer if it's large enough or old enough
        if self._log_buffer and (
            len(self._log_buffer) >= self._FLUSH_SIZE
            or time.time() - self._last_flush >= self._FLUSH_INTERVAL
        ):
            self._flush_log_buffer()

    def flush(self):
        # Emit any trailing partial line so we don't drop output that
        # never got a terminating newline (e.g. a final print(..., end='')
        # before interpreter shutdown).
        if self._partial_line:
            self._emit_line(self._partial_line)
            self._partial_line = ''
        self._flush_log_buffer()
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def stream_formatter(settings):
    if settings.x_log_level <= logging.DEBUG:
        return ColorFormatter(
            '%(asctime)s.%(msecs)03d %(levelname)-8s %(threadName)-10s:%(process)d '
            '[%(filename)s:%(funcName)s():%(lineno)s] %(message)s',
            datefmt='%H:%M:%S',
        )
    else:
        return ColorFormatter(
            '%(asctime)s | %(message)s',
            # "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt='%H:%M:%S',
        )


def input_hook(prompt='', logger=None):
    content = _input(prompt)
    logger.warn(f'{prompt}{content}')
    return content


def setup_logger(settings, logger, console=None, sync_manager=None) -> None:
    # TODO: capture stdout through rich
    if settings._nb_colab():
        rlogger = logging.getLogger()
        for h in rlogger.handlers[:]:  # iter root handlers
            rlogger.removeHandler(h)

    logger.setLevel(settings.x_log_level)

    if len(logger.handlers) == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter(settings))
        logger.addHandler(stream_handler)

    if settings._op_id and not settings.disable_console:
        if len(console.handlers) > 0:  # full logger
            return
        logger, console = setup_logger_file(settings, logger, console, sync_manager)


def teardown_logger(logger, console=None):
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    if console:
        _stop_fd_captures()
        builtins.input = _input
        sys.stdout = _stdout  # global _stdout
        sys.stderr = _stderr
        teardown_logger(logger=console)


def _stop_fd_captures() -> None:
    """Stop and forget all active fd-level captures. Never raises."""
    while _fd_captures:
        capture = _fd_captures.pop()
        try:
            capture.stop()
        except Exception as e:
            logger.debug('Failed to stop fd capture: %s', e)


def flush_console_buffers() -> None:
    """Flush every console-capture layer into the sync store. Never raises.

    Op._teardown() calls this BEFORE draining/stopping the sync manager:
    lines still sitting in the fd reader's batch buffer (up to 0.2s of
    output) or a ConsoleHandler's buffer are otherwise enqueued at
    teardown_logger time — after the uploader is gone — and sit in SQLite
    unuploaded. Observed on CI as a line logged immediately before
    finish() never reaching the server.

    Stopping the fd captures here (rather than just flushing) is
    deliberate: output written after the sync drain below can't be
    uploaded in this run anyway, and stop() is the only deterministic
    flush point the reader thread offers. teardown_logger's later
    _stop_fd_captures() is then a no-op.
    """
    _stop_fd_captures()
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, ConsoleHandler):
            try:
                stream.flush()
            except Exception as e:
                logger.debug('Failed to flush console stream: %s', e)


def setup_logger_file(settings, logger, console, sync_manager=None):
    console.setLevel(logging.DEBUG)
    # The console logger is internal capture plumbing. If it propagated,
    # every captured line would re-emit through the user's root handlers —
    # duplicating terminal output, and under fd capture feeding the line
    # straight back into the capture pipe (root handler → pre-init stderr
    # object → fd 2).
    console.propagate = False

    file_handler = logging.FileHandler(f'{settings.get_dir()}/{settings.tag}.log')
    file_formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(threadName)-10s:%(process)d '
        '[%(filename)s:%(funcName)s():%(lineno)s] %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    file_handler = logging.FileHandler(f'{settings.get_dir()}/sys.log')
    file_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
        datefmt='%H:%M:%S',
    )
    file_handler.setFormatter(file_formatter)
    console.addHandler(file_handler)  # TODO: fix slow file writes

    sanitizer = None
    if getattr(settings, 'sanitize_logs', True):
        from .sanitize import SecretSanitizer

        sanitizer = SecretSanitizer()

    # fd-level capture (dup2 tee over fds 1/2) catches writers that bound
    # the original stream objects before init — logging.StreamHandlers set
    # up by frameworks like torchtitan, C extensions, forked children. The
    # sys.stdout/sys.stderr swap below can't see any of those (a handler
    # holds the old object), which left the console section empty for any
    # job that configures logging before pluto.init().
    fd_capture_active = _start_fd_captures(settings, sync_manager, sanitizer)

    if settings.mode == 'debug':
        builtins.input = lambda prompt='': input_hook(prompt, logger=console)
    # The ConsoleHandler wrappers stay even with fd capture active: they
    # flush Python-level writes through to the fd promptly (the original
    # sys.stdout is block-buffered when not a tty) and feed the local
    # sys.log file. Exactly ONE layer may enqueue a given stream to the
    # sync manager, or every Python-level line would be uploaded twice.
    # The fd layer owns a stream only when that stream actually writes to
    # the captured fd; when it doesn't (Jupyter's ZMQ OutStream, pytest's
    # capture objects), its writes never reach the pipe, so the wrapper
    # must keep enqueueing them.
    stdout_owned_by_fd = fd_capture_active and _stream_writes_to_fd(sys.stdout, 1)
    stderr_owned_by_fd = fd_capture_active and _stream_writes_to_fd(sys.stderr, 2)
    sys.stdout = ConsoleHandler(
        console,
        None if stdout_owned_by_fd else sync_manager,
        logging.INFO,
        sys.stdout,
        'stdout',
        sanitizer,
    )
    sys.stderr = ConsoleHandler(
        console,
        None if stderr_owned_by_fd else sync_manager,
        logging.ERROR,
        sys.stderr,
        'stderr',
        sanitizer,
    )

    return logger, console


def _stream_writes_to_fd(stream, fd: int) -> bool:
    """True if writes to `stream` land on OS file descriptor `fd`."""
    try:
        return stream.fileno() == fd
    except Exception:  # no fileno / io.UnsupportedOperation / detached
        return False


def _start_fd_captures(settings, sync_manager, sanitizer) -> bool:
    """Start fd-level console capture on fds 1/2. Returns True if active.

    Fails soft: any OS-level problem (exotic platform, fds not real) falls
    back to the legacy wrapper-based capture so logging never breaks a run.
    """
    if sync_manager is None or not getattr(settings, 'x_console_fd_capture', True):
        return False
    from ._fd_capture import FdCapture

    started = []
    try:
        # Flush Python-level buffers so pre-init output goes to the real
        # terminal instead of being captured into this run's console.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        for fd, level in ((1, logging.INFO), (2, logging.ERROR)):
            capture = FdCapture(
                fd=fd, level=level, sync_manager=sync_manager, sanitizer=sanitizer
            )
            capture.start()
            started.append(capture)
    except Exception as e:
        logger.debug('fd-level console capture unavailable, using fallback: %s', e)
        for capture in started:
            try:
                capture.stop()
            except Exception:
                pass
        return False
    _fd_captures.extend(started)
    return True
