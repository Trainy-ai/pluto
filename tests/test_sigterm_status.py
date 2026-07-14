"""
Unit tests for the SIGTERM -> TERMINATED status handler (pluto/op.py).

Spot reclaim / k8s eviction deliver SIGTERM, which by default terminates the
process without running atexit, so the run would linger RUNNING until the
server-side heartbeat reaps it. The handler pushes a terminal status first,
then chains to the previously-installed handler.

These tests exercise the handler and (un)register helpers directly — they do
NOT raise a real SIGTERM (the SIG_DFL chain would terminate the test runner),
so every case uses a callable or SIG_IGN previous handler.
"""

import signal
import threading
import types
from unittest.mock import MagicMock

import pytest

import pluto
from pluto import op as op_module


def _make_op(status=-1, op_id=123):
    settings = types.SimpleNamespace(
        _op_id=op_id,
        _op_status=status,
        url_stop='http://server/api/runs/status/update',
        x_sigterm_status_timeout_seconds=5.0,
        # Read when the handler builds its fresh httpx client.
        insecure_disable_ssl=False,
        http_proxy=None,
        https_proxy=None,
    )
    return types.SimpleNamespace(settings=settings, _iface=MagicMock())


@pytest.fixture(autouse=True)
def _restore_state():
    """Snapshot/restore module + process signal state so tests never leak a
    handler into the runner or each other."""
    prev_registered = op_module._sigterm_handler_registered
    prev_original = op_module._original_sigterm_handler
    prev_ops = pluto.ops
    prev_sig = signal.getsignal(signal.SIGTERM)
    # Start from a clean baseline. Other test files run a real pluto.init(),
    # which installs `_sigterm_handler` on SIGTERM and can leave it there; if a
    # test captured that as the "previous" handler, the recursion guard in
    # _register_sigterm_handler would (correctly) refuse to re-store it and skew
    # the assertions. Resetting here isolates us from that leakage.
    op_module._sigterm_handler_registered = False
    op_module._original_sigterm_handler = None
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        yield
    finally:
        op_module._sigterm_handler_registered = prev_registered
        op_module._original_sigterm_handler = prev_original
        pluto.ops = prev_ops
        signal.signal(signal.SIGTERM, prev_sig)


def test_handler_pushes_terminated_and_chains():
    o = _make_op(status=-1)
    pluto.ops = [o]
    prev = MagicMock()
    op_module._original_sigterm_handler = prev
    op_module._sigterm_handler_registered = True

    op_module._sigterm_handler(signal.SIGTERM, None)

    assert o.settings._op_status == signal.SIGTERM.value  # -> TERMINATED
    o._iface._post_v1.assert_called_once()
    kwargs = o._iface._post_v1.call_args.kwargs
    assert kwargs['max_retries'] == 0
    assert kwargs['timeout'] == 5.0
    assert kwargs['suppress_httpx_logs'] is True
    prev.assert_called_once_with(signal.SIGTERM, None)  # chained


def test_handler_skips_terminal_ops():
    o = _make_op(status=0)  # already COMPLETED
    pluto.ops = [o]
    op_module._original_sigterm_handler = MagicMock()
    op_module._sigterm_handler_registered = True

    op_module._sigterm_handler(signal.SIGTERM, None)

    assert o.settings._op_status == 0  # untouched
    o._iface._post_v1.assert_not_called()


def test_handler_survives_push_failure_and_still_chains():
    o = _make_op(status=-1)
    o._iface._post_v1.side_effect = RuntimeError('network down')
    pluto.ops = [o]
    prev = MagicMock()
    op_module._original_sigterm_handler = prev
    op_module._sigterm_handler_registered = True

    op_module._sigterm_handler(signal.SIGTERM, None)  # must not raise

    prev.assert_called_once()


def test_handler_sig_ign_returns_without_reraising():
    o = _make_op(status=-1)
    pluto.ops = [o]
    op_module._original_sigterm_handler = signal.SIG_IGN
    op_module._sigterm_handler_registered = True

    # Previous handler ignored SIGTERM: still push status, but do not re-raise.
    op_module._sigterm_handler(signal.SIGTERM, None)

    o._iface._post_v1.assert_called_once()


def test_handler_no_iface_is_safe():
    o = _make_op(status=-1)
    o._iface = None
    pluto.ops = [o]
    op_module._original_sigterm_handler = MagicMock()
    op_module._sigterm_handler_registered = True

    op_module._sigterm_handler(signal.SIGTERM, None)  # must not raise

    assert o.settings._op_status == signal.SIGTERM.value


def test_register_and_unregister_restores_previous_handler():
    op_module._sigterm_handler_registered = False
    original = signal.getsignal(signal.SIGTERM)

    op_module._register_sigterm_handler(
        types.SimpleNamespace(x_sigterm_status_enabled=True)
    )
    assert op_module._sigterm_handler_registered is True
    assert signal.getsignal(signal.SIGTERM) is op_module._sigterm_handler
    assert op_module._original_sigterm_handler == original

    op_module._unregister_sigterm_handler()
    assert op_module._sigterm_handler_registered is False
    assert signal.getsignal(signal.SIGTERM) == original


def test_register_is_idempotent():
    op_module._sigterm_handler_registered = False
    op_module._register_sigterm_handler(
        types.SimpleNamespace(x_sigterm_status_enabled=True)
    )
    first_original = op_module._original_sigterm_handler
    # A second call must not clobber the saved original with our own handler.
    op_module._register_sigterm_handler(
        types.SimpleNamespace(x_sigterm_status_enabled=True)
    )
    assert op_module._original_sigterm_handler == first_original


def test_register_respects_disabled_setting():
    op_module._sigterm_handler_registered = False
    before = signal.getsignal(signal.SIGTERM)

    op_module._register_sigterm_handler(
        types.SimpleNamespace(x_sigterm_status_enabled=False)
    )

    assert op_module._sigterm_handler_registered is False
    assert signal.getsignal(signal.SIGTERM) == before


def test_register_never_stores_self_as_original():
    """Recursion guard: if our handler is already installed but the flag was
    cleared (e.g. an off-thread unregister), re-registering must NOT record our
    own handler as `_original` (which would recurse when chaining on SIGTERM)."""
    signal.signal(signal.SIGTERM, op_module._sigterm_handler)
    op_module._sigterm_handler_registered = False
    op_module._original_sigterm_handler = signal.SIG_DFL  # sentinel "real" prior

    op_module._register_sigterm_handler(
        types.SimpleNamespace(x_sigterm_status_enabled=True)
    )

    assert op_module._sigterm_handler_registered is True
    assert op_module._original_sigterm_handler is not op_module._sigterm_handler


def test_unregister_off_main_thread_is_noop():
    """Off the main thread, signal.signal can't restore the handler, so state
    must stay consistent (still registered, handler still installed) rather
    than clearing the flag while the handler remains live."""
    signal.signal(signal.SIGTERM, op_module._sigterm_handler)
    op_module._sigterm_handler_registered = True
    op_module._original_sigterm_handler = signal.SIG_DFL

    t = threading.Thread(target=op_module._unregister_sigterm_handler)
    t.start()
    t.join()

    assert op_module._sigterm_handler_registered is True
    assert signal.getsignal(signal.SIGTERM) is op_module._sigterm_handler
    assert op_module._original_sigterm_handler == signal.SIG_DFL
