import builtins
import logging
import sys
import time

from .util import ANSI

logger = logging.getLogger(f'{__name__.split(".")[0]}')

_input = builtins.input
_stdout = sys.stdout
_stderr = sys.stderr

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

    def write(self, buf: str) -> None:
        for line in buf.splitlines():
            if line:  # do not log empty lines
                self.count += 1
                timestamp_ms = int(time.time() * 1000)
                if self.sync_manager is not None:
                    sanitized_line = (
                        self.sanitizer.sanitize(line) if self.sanitizer else line
                    )
                    log_type = logging._levelToName.get(self.level, 'INFO')
                    self._log_buffer.append(
                        (sanitized_line, log_type, timestamp_ms, self.count)
                    )
                self.logger.log(self.level, line)
        # Flush the buffer if it's large enough or old enough
        if self._log_buffer and (
            len(self._log_buffer) >= self._FLUSH_SIZE
            or time.time() - self._last_flush >= self._FLUSH_INTERVAL
        ):
            self._flush_log_buffer()
        self.stream.write(buf)
        self.stream.flush()

    def flush(self):
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
        builtins.input = _input
        sys.stdout = _stdout  # global _stdout
        sys.stderr = _stderr
        teardown_logger(logger=console)


def setup_logger_file(settings, logger, console, sync_manager=None):
    console.setLevel(logging.DEBUG)

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

    if settings.mode == 'debug':
        builtins.input = lambda prompt='': input_hook(prompt, logger=console)
    sys.stdout = ConsoleHandler(
        console, sync_manager, logging.INFO, sys.stdout, 'stdout', sanitizer
    )
    sys.stderr = ConsoleHandler(
        console, sync_manager, logging.ERROR, sys.stderr, 'stderr', sanitizer
    )

    return logger, console
