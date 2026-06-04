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
