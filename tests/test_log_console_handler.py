"""
Unit tests for pluto.log.ConsoleHandler line buffering.

The previous implementation called write() once per call and ran
splitlines() over the buffer, which:
  1. Treated each write() as a complete line — so a traceback emitted
     via many partial writes (one for the whitespace, one for the code,
     one for the carets, etc.) showed up as one log entry per write,
     producing per-character "lines" in the UI.
  2. Used splitlines(), which splits on \v \f \x1c \x1d \x1e \x85
     U+2028 U+2029 in addition to \n — chars that rich's styled output
     emits as internal segment separators, so rich-styled tracebacks
     also got chopped at non-newline boundaries.

These tests pin both behaviors.
"""

from __future__ import annotations

import io
import logging
from unittest import mock

import pytest

from pluto.log import ConsoleHandler


@pytest.fixture
def harness():
    """Build a ConsoleHandler wired to fakes that capture every emit.

    The handler clears its internal buffer right after passing it to
    enqueue_console_batch, so a plain MagicMock would record an empty
    list. We snapshot via side_effect into a separate sink instead.
    """
    sync_manager = mock.MagicMock()
    enqueued: list[list[tuple]] = []

    def _snapshot(batch):
        enqueued.append(list(batch))

    sync_manager.enqueue_console_batch = mock.MagicMock(side_effect=_snapshot)
    sync_manager._enqueued = enqueued

    underlying = io.StringIO()
    log = logging.getLogger('test_console_handler')
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    handler = ConsoleHandler(
        logger=log,
        sync_manager=sync_manager,
        level=logging.INFO,
        stream=underlying,
        type='stdout',
    )
    return handler, sync_manager, underlying


def _emitted_lines(sync_manager) -> list[str]:
    """Return every line text that was enqueued, across all batches."""
    return [line for batch in sync_manager._enqueued for (line, *_rest) in batch]


def _drain(handler) -> None:
    """Force any pending buffer to be emitted (mirrors interpreter shutdown)."""
    handler.flush()


class TestPartialWriteCoalescing:
    def test_three_partial_writes_become_one_line(self, harness):
        handler, sync_manager, _ = harness
        # This is the exact pattern Python's traceback printer / rich produces:
        # leading whitespace, then the code, then the newline as separate writes.
        handler.write('    ')
        handler.write('main()')
        handler.write('\n')
        _drain(handler)
        assert _emitted_lines(sync_manager) == ['    main()']

    def test_carets_arriving_per_character_become_one_line(self, harness):
        handler, sync_manager, _ = harness
        # Rich/3.12 enhanced tracebacks can emit the caret-marker line
        # one character at a time.  Pre-fix this produced one log entry
        # per character, the bug the user reported.
        for ch in '           ^^^^^^^^^^^^^^^':
            handler.write(ch)
        handler.write('\n')
        _drain(handler)
        assert _emitted_lines(sync_manager) == ['           ^^^^^^^^^^^^^^^']

    def test_newline_in_middle_of_buffer_still_works(self, harness):
        handler, sync_manager, _ = harness
        handler.write('abc\ndef')  # one complete line, one partial
        handler.write('ghi\n')  # finishes the partial
        _drain(handler)
        assert _emitted_lines(sync_manager) == ['abc', 'defghi']

    def test_consecutive_newlines_only_skip_empties(self, harness):
        handler, sync_manager, _ = harness
        handler.write('one\n\ntwo\n')
        _drain(handler)
        # The empty line between 'one' and 'two' is dropped (existing behavior).
        assert _emitted_lines(sync_manager) == ['one', 'two']

    def test_trailing_partial_emitted_on_flush(self, harness):
        handler, sync_manager, _ = harness
        handler.write('no_newline_here')
        # Without flush, nothing emits — we don't know if more is coming.
        assert _emitted_lines(sync_manager) == []
        handler.flush()
        assert _emitted_lines(sync_manager) == ['no_newline_here']

    def test_underlying_stream_gets_full_buf_immediately(self, harness):
        handler, _sync, underlying = harness
        # Pass-through to the real terminal must NOT wait for a newline,
        # otherwise progress bars / partial prints freeze the terminal.
        handler.write('partial')
        assert underlying.getvalue() == 'partial'
        handler.write(' more\n')
        assert underlying.getvalue() == 'partial more\n'


class TestNonNewlineSeparatorsArePreserved:
    """splitlines() splits on too many chars; we only want \\n.

    The old impl called buf.splitlines(), which silently broke output on
    \v \f \x1c-\x1e \x85 U+2028 U+2029. Rich uses some of those as
    segment separators when emitting styled output, so styled tracebacks
    got chopped mid-line. Switching to '\\n'-only split fixes that.
    """

    @pytest.mark.parametrize(
        'sep_char,name',
        [
            ('\v', 'VT'),
            ('\f', 'FF'),
            ('\x1c', 'FS'),
            ('\x1d', 'GS'),
            ('\x1e', 'RS'),
            ('\x85', 'NEL'),
            (' ', 'LSEP'),
            (' ', 'PSEP'),
        ],
    )
    def test_unicode_or_control_separator_does_not_split(self, harness, sep_char, name):
        handler, sync_manager, _ = harness
        handler.write(f'left{sep_char}right\n')
        _drain(handler)
        assert _emitted_lines(sync_manager) == [
            f'left{sep_char}right'
        ], f'{name} ({sep_char!r}) should not split a logical line'


class TestSyncManagerBatching:
    def test_complete_lines_only_enqueued_after_newline(self, harness):
        handler, sync_manager, _ = harness
        handler.write('hello')
        # No newline yet → nothing enqueued
        sync_manager.enqueue_console_batch.assert_not_called()
        handler.write(' world\n')
        # The batch flush is gated on size/time too, so force-flush:
        handler.flush()
        sync_manager.enqueue_console_batch.assert_called()
        assert _emitted_lines(sync_manager) == ['hello world']

    def test_no_sync_manager_still_logs_to_python_logger(self):
        underlying = io.StringIO()
        log = logging.getLogger('test_console_handler_no_sync')
        log.handlers.clear()
        log.setLevel(logging.DEBUG)
        captured: list[logging.LogRecord] = []
        log.addHandler(_ListHandler(captured))
        handler = ConsoleHandler(
            logger=log,
            sync_manager=None,
            level=logging.INFO,
            stream=underlying,
            type='stdout',
        )
        handler.write('hello\n')
        handler.flush()
        assert [r.getMessage() for r in captured] == ['hello']


class TestSanitizerStillRuns:
    def test_sanitizer_is_invoked_on_complete_lines(self, harness):
        handler, sync_manager, _ = harness
        handler.sanitizer = mock.MagicMock()
        handler.sanitizer.sanitize.side_effect = lambda s: s.replace('mlpi_xxx', '***')
        handler.write('api_key=mlpi_xxx ')
        handler.write('here\n')
        handler.flush()
        assert _emitted_lines(sync_manager) == ['api_key=*** here']


class TestRankPrefix:
    """When RANK env is set (torchrun), captured lines get a [rankN] prefix.

    Pluto's stdout capture sits inside the child process before lines
    reach the OS pipe that torchrun reads, so the [defaultN]: prefix
    torchrun adds in the parent never makes it to the SyncStore. Reading
    RANK directly from the child's env gives every captured line a tag
    so DDP runs aren't a confusing single stream of unattributed lines
    in the Pluto UI. The pass-through to the underlying stream stays
    unprefixed so torchrun's own prefixing isn't doubled in the
    terminal.
    """

    def _build_handler(self, sync_manager, stream):
        log = logging.getLogger('test_console_handler_rank')
        log.handlers.clear()
        log.setLevel(logging.DEBUG)
        return ConsoleHandler(
            logger=log,
            sync_manager=sync_manager,
            level=logging.INFO,
            stream=stream,
            type='stdout',
        )

    def _capturing_sync_manager(self):
        sync_manager = mock.MagicMock()
        enqueued: list[list[tuple]] = []
        sync_manager.enqueue_console_batch = mock.MagicMock(
            side_effect=lambda batch: enqueued.append(list(batch))
        )
        sync_manager._enqueued = enqueued
        return sync_manager

    def test_rank_env_var_prefixes_captured_lines(self, monkeypatch):
        monkeypatch.setenv('RANK', '1')
        sync_manager = self._capturing_sync_manager()
        underlying = io.StringIO()
        handler = self._build_handler(sync_manager, underlying)
        handler.write('starting fit\n')
        handler.flush()
        assert _emitted_lines(sync_manager) == ['[rank1] starting fit']

    def test_no_rank_env_var_means_no_prefix(self, monkeypatch):
        """Single-process runs (or anything not under torchrun) keep their format."""
        monkeypatch.delenv('RANK', raising=False)
        sync_manager = self._capturing_sync_manager()
        underlying = io.StringIO()
        handler = self._build_handler(sync_manager, underlying)
        handler.write('hello\n')
        handler.flush()
        assert _emitted_lines(sync_manager) == ['hello']

    def test_underlying_stream_is_not_prefixed(self, monkeypatch):
        """The pass-through to terminal/OS-pipe must stay unprefixed.

        torchrun reads from the OS pipe and adds its own [defaultN]:
        prefix in the parent. If we prefixed the underlying stream too,
        the terminal would show [default0]: [rank0] ... — double tag.
        """
        monkeypatch.setenv('RANK', '0')
        sync_manager = self._capturing_sync_manager()
        underlying = io.StringIO()
        handler = self._build_handler(sync_manager, underlying)
        handler.write('hello\n')
        handler.flush()
        assert underlying.getvalue() == 'hello\n'  # exactly the input, no prefix
        assert _emitted_lines(sync_manager) == ['[rank0] hello']

    def test_prefix_applied_per_line_not_per_write(self, monkeypatch):
        """Multiple lines in one write each get their own prefix."""
        monkeypatch.setenv('RANK', '2')
        sync_manager = self._capturing_sync_manager()
        underlying = io.StringIO()
        handler = self._build_handler(sync_manager, underlying)
        handler.write('one\ntwo\nthree\n')
        handler.flush()
        assert _emitted_lines(sync_manager) == [
            '[rank2] one',
            '[rank2] two',
            '[rank2] three',
        ]


class _ListHandler(logging.Handler):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        self.sink.append(record)
