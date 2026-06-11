"""Unit tests for the stdout run banner (Op._print_run_banner).

The banner is a stable, greppable line printed to stdout when a run starts or
resumes, so external tooling can reverse-look up a run from a training
process's stdout. It must:
  - go to stdout (not stderr / the logging system),
  - use the server display ID (e.g. "LV3-12"), not the numeric run ID,
  - include the sqid slug parsed from the run URL as external_id,
  - be a fixed format matching ``pluto: run <id> <verb>``.
"""

import io
from contextlib import redirect_stderr, redirect_stdout

from pluto.op import Op
from pluto.sets import Settings


def _make_op(display_id, url):
    """Build an Op without running __init__ (no server contact)."""
    op = Op.__new__(Op)
    op.settings = Settings()
    op.settings._display_id = display_id
    op.settings.url_view = url
    return op


def test_banner_started_to_stdout():
    op = _make_op(
        'LV3-12',
        'https://pluto.trainy.ai/o/linum-n/projects/linum-v3/dhyecrvx',
    )
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        op._print_run_banner('started')

    assert out.getvalue() == 'pluto: run LV3-12 started (external_id=dhyecrvx)\n'
    # Must not leak onto stderr.
    assert err.getvalue() == ''


def test_banner_resumed_verb():
    op = _make_op(
        'LV3-12',
        'https://pluto.trainy.ai/o/linum-n/projects/linum-v3/dhyecrvx',
    )
    out = io.StringIO()
    with redirect_stdout(out):
        op._print_run_banner('resumed')

    assert out.getvalue() == 'pluto: run LV3-12 resumed (external_id=dhyecrvx)\n'


def test_banner_matches_consumer_regex():
    """The format must satisfy the documented reverse-lookup regex."""
    import re

    op = _make_op(
        'LV3-12',
        'https://pluto.trainy.ai/o/linum-n/projects/linum-v3/dhyecrvx',
    )
    out = io.StringIO()
    with redirect_stdout(out):
        op._print_run_banner('started')

    m = re.search(r'pluto:\s*run\s+(LV3-\d+)', out.getvalue())
    assert m is not None
    assert m.group(1) == 'LV3-12'


def test_banner_trailing_slash_url():
    op = _make_op('LV3-12', 'https://x/o/n/projects/p/dhyecrvx/')
    out = io.StringIO()
    with redirect_stdout(out):
        op._print_run_banner('started')

    assert out.getvalue() == 'pluto: run LV3-12 started (external_id=dhyecrvx)\n'


def test_banner_no_display_id_is_silent():
    """No display ID -> nothing stable to print, emit nothing."""
    op = _make_op(None, 'https://x/o/n/projects/p/dhyecrvx')
    out = io.StringIO()
    with redirect_stdout(out):
        op._print_run_banner('started')

    assert out.getvalue() == ''


def test_banner_no_url_omits_external_id():
    op = _make_op('LV3-12', None)
    out = io.StringIO()
    with redirect_stdout(out):
        op._print_run_banner('started')

    assert out.getvalue() == 'pluto: run LV3-12 started\n'


def test_banner_host_only_url_omits_external_id():
    """A host-only URL has no path segment, so external_id is omitted rather
    than falling back to the hostname."""
    op = _make_op('LV3-12', 'https://pluto.trainy.ai')
    out = io.StringIO()
    with redirect_stdout(out):
        op._print_run_banner('started')

    assert out.getvalue() == 'pluto: run LV3-12 started\n'


class _FakeTTY(io.StringIO):
    """A StringIO that claims to be a TTY, to exercise the colored path."""

    def isatty(self):
        return True


def test_banner_non_tty_is_plain_no_ansi():
    """Piped/redirected stdout (isatty() is False) -> no ANSI codes, so the
    captured output stays byte-clean for downstream greppers."""
    op = _make_op(
        'LV3-12', 'https://pluto.trainy.ai/o/linum-n/projects/linum-v3/dhyecrvx'
    )
    out = io.StringIO()  # StringIO.isatty() -> False
    with redirect_stdout(out):
        op._print_run_banner('started')

    assert '\033' not in out.getvalue()
    assert out.getvalue() == 'pluto: run LV3-12 started (external_id=dhyecrvx)\n'


def test_banner_tty_is_green_and_still_greppable():
    """On a TTY the line is green-wrapped, but the codes sit at the start/end
    so the matchable token stays contiguous and the consumer regex works."""
    import re

    op = _make_op(
        'LV3-12', 'https://pluto.trainy.ai/o/linum-n/projects/linum-v3/dhyecrvx'
    )
    out = _FakeTTY()
    with redirect_stdout(out):
        op._print_run_banner('started')

    value = out.getvalue()
    assert value.startswith('\033[32m')
    assert value.endswith('\033[0m\n')
    # The reverse-lookup regex still extracts the display ID from the colored line.
    m = re.search(r'pluto:\s*run\s+([A-Z0-9]+-\d+)', value)
    assert m is not None and m.group(1) == 'LV3-12'


def test_view_run_message_includes_green_display_id(monkeypatch):
    """The 'View run' log line names the run, with the display ID colored."""
    from pluto import util

    monkeypatch.setattr(util.ANSI, 'green', '<G>')
    monkeypatch.setattr(util.ANSI, 'cyan', '<C>')
    url = 'https://pluto.trainy.ai/o/trainy/projects/testing-ci/OgiAJ'
    op = _make_op('TCI-144405', url)

    msg = op._view_run_message()

    # ID is green, then back to cyan so the trailing "at <url>" stays cyan.
    assert 'View run <G>TCI-144405<C> at ' in msg
    assert url in msg


def test_view_run_message_without_display_id_falls_back():
    op = _make_op(None, 'https://pluto.trainy.ai/o/trainy/projects/testing-ci/OgiAJ')

    msg = op._view_run_message()

    assert msg.startswith('View run at ')
    assert 'TCI-' not in msg
