"""Unit tests for ServerInterface write-error handling (pluto/iface.py).

These exercise the retry/raise policy of ``_try`` without a real server:
- 4xx responses are terminal (no retry) and surface the server's message.
- 5xx responses are retried.
- Network exceptions never raise PlutoRequestError (caller sees None).
"""

import logging

import httpx
import pytest

from pluto.iface import (
    PlutoAuthError,
    PlutoRequestError,
    ServerInterface,
    _server_error_message,
    http_401_message,
)
from pluto.sets import Settings


def _make_iface():
    settings = Settings()
    settings.mode = 'noop'
    settings.update_host()
    # Keep retry backoff instant so the 5xx test doesn't actually sleep long.
    settings.x_file_stream_retry_max = 2
    settings.x_file_stream_retry_wait_min_seconds = 0
    settings.x_file_stream_retry_wait_max_seconds = 0
    return ServerInterface(config={}, settings=settings)


def _resp(status_code, json_body=None, text=''):
    if json_body is not None:
        return httpx.Response(status_code, json=json_body)
    return httpx.Response(status_code, text=text)


def test_server_error_message_prefers_error_field():
    r = _resp(400, json_body={'error': 'A run can have at most one group:* tag.'})
    assert _server_error_message(r) == 'A run can have at most one group:* tag.'
    # Falls back to raw body when not the expected shape.
    assert _server_error_message(_resp(400, text='plain boom')) == 'plain boom'


def test_try_400_does_not_retry_and_raises_server_message():
    iface = _make_iface()
    calls = {'n': 0}

    def fake_method(url, content=None, headers=None, **kwargs):
        calls['n'] += 1
        return _resp(
            400, json_body={'error': 'A run can have at most one group:* tag.'}
        )

    with pytest.raises(PlutoRequestError) as excinfo:
        iface._try(
            fake_method,
            'https://example/api/runs/create',
            {},
            b'{}',
            name='create',
            raise_on_error=True,
        )

    assert calls['n'] == 1, 'a 400 must not be retried'
    assert 'at most one group' in str(excinfo.value)
    assert not isinstance(excinfo.value, ConnectionError)
    assert excinfo.value.status_code == 400


def test_try_400_without_raise_returns_none_no_retry():
    iface = _make_iface()
    calls = {'n': 0}

    def fake_method(url, content=None, headers=None, **kwargs):
        calls['n'] += 1
        return _resp(400, json_body={'error': 'bad'})

    r = iface._try(fake_method, 'https://x', {}, b'{}', name='tags')
    assert r is None
    assert calls['n'] == 1, 'a 400 must not be retried even without raise_on_error'


def test_try_500_is_retried_then_gives_up():
    iface = _make_iface()  # x_file_stream_retry_max = 2
    calls = {'n': 0}

    def fake_method(url, content=None, headers=None, **kwargs):
        calls['n'] += 1
        return _resp(500, text='boom')

    # raise_on_error → persistent 5xx raises after exhausting retries.
    with pytest.raises(PlutoRequestError) as excinfo:
        iface._try(
            fake_method, 'https://x', {}, b'{}', name='create', raise_on_error=True
        )
    # initial attempt + 2 retries = 3 calls
    assert calls['n'] == 3, calls['n']
    # The raised error carries the real status code, not None.
    assert excinfo.value.status_code == 500


def test_try_transient_4xx_is_retried_then_recovers():
    """401/408/429 are transient (auth blips, timeouts, rate limits) and must
    be retried — a 401 that clears on retry should succeed, not hard-fail."""
    iface = _make_iface()
    calls = {'n': 0}

    def fake_method(url, content=None, headers=None, **kwargs):
        calls['n'] += 1
        # First two attempts flake with 401, third succeeds.
        if calls['n'] < 3:
            return _resp(401, text='Unauthorized')
        return _resp(200, json_body={'ok': True})

    r = iface._try(
        fake_method, 'https://x', {}, b'{}', name='create', raise_on_error=True
    )
    assert r is not None and r.status_code == 200
    assert calls['n'] == 3, 'a transient 401 must be retried, not terminal'


def test_try_persistent_401_raises_after_retries():
    """A 401 that never clears still raises (after exhausting retries), carrying
    the real status code — not a first-attempt hard fail, not None."""
    iface = _make_iface()  # x_file_stream_retry_max = 2
    calls = {'n': 0}

    def fake_method(url, content=None, headers=None, **kwargs):
        calls['n'] += 1
        return _resp(401, text='Unauthorized')

    with pytest.raises(PlutoRequestError) as excinfo:
        iface._try(
            fake_method, 'https://x', {}, b'{}', name='create', raise_on_error=True
        )
    assert calls['n'] == 3, 'a 401 is retried (initial + 2 retries)'
    assert excinfo.value.status_code == 401
    # ...and it is the auth-specific subclass, carrying the actionable message
    # rather than a bare "HTTP 401" that reads like a server outage.
    assert isinstance(excinfo.value, PlutoAuthError)
    msg = str(excinfo.value)
    assert 'expired' in msg and 'not a server outage' in msg
    assert 'api-keys' in msg


def test_message_names_env_var_key_source(monkeypatch):
    """The fix differs by key source, so the message must name the right one."""
    monkeypatch.setenv('PLUTO_API_KEY', 'k')
    msg = http_401_message(page_url='https://self.hosted/api-keys')
    assert 'PLUTO_API_KEY' in msg
    assert 'update PLUTO_API_KEY' in msg
    # Self-hosted deployments must not be pointed at the SaaS key page.
    assert 'https://self.hosted/api-keys' in msg

    monkeypatch.delenv('PLUTO_API_KEY')
    monkeypatch.delenv('MLOP_API_TOKEN', raising=False)
    msg = http_401_message()
    assert 'pluto login' in msg


def test_message_resolves_api_page_from_env(monkeypatch):
    """Log paths can't read the key page off Settings (it holds the API key,
    and log lines must not be built from it), so an omitted page_url resolves
    from the environment — self-hosted still gets its own page."""
    monkeypatch.delenv('PLUTO_URL_APP', raising=False)
    monkeypatch.delenv('MLOP_URL_APP', raising=False)
    assert 'https://pluto.trainy.ai/api-keys' in http_401_message()

    monkeypatch.setenv('PLUTO_URL_APP', 'https://self.hosted/')
    assert 'https://self.hosted/api-keys' in http_401_message()


def test_message_keeps_specific_server_reason(monkeypatch):
    monkeypatch.delenv('PLUTO_API_KEY', raising=False)
    monkeypatch.delenv('MLOP_API_TOKEN', raising=False)
    # A specific reason is worth surfacing...
    assert 'token expired' in http_401_message(server_msg='token expired')
    # ...but a bare restatement of the status code adds nothing.
    assert 'Server said' not in http_401_message(server_msg='Unauthorized')


def test_try_persistent_401_without_raise_logs_once(monkeypatch, caplog):
    """Fire-and-forget paths (heartbeat, status update, logName registration)
    swallow failures, so the only way the user learns their key expired is the
    log — but the heartbeat fires every ~4 s, so it must not repeat."""
    monkeypatch.setattr('pluto.iface._logged_401', False)
    iface = _make_iface()

    def fake_method(url, content=None, headers=None, **kwargs):
        return _resp(401, text='Unauthorized')

    with caplog.at_level(logging.CRITICAL, logger='pluto'):
        for _ in range(3):
            r = iface._try(fake_method, 'https://x', {}, b'{}', name='trigger')
            assert r is None

    auth_logs = [r for r in caplog.records if 'authentication failed' in r.message]
    assert len(auth_logs) == 1, 'the persistent-401 message must be logged once'
    assert 'expired' in auth_logs[0].message


def test_try_network_error_returns_none_not_request_error():
    iface = _make_iface()

    def fake_method(url, content=None, headers=None, **kwargs):
        raise httpx.ConnectError('no route')

    # Even with raise_on_error, a pure network failure is NOT a server
    # validation error — caller gets None and decides (ConnectionError).
    r = iface._try(
        fake_method, 'https://x', {}, b'{}', name='create', raise_on_error=True
    )
    assert r is None


# --- sync-process uploader (pluto/sync/process.py) -------------------------


class _FakeClient:
    """Stand-in httpx client that returns a canned response and counts posts."""

    def __init__(self, response):
        self._response = response
        self.calls = 0

    def post(self, url, content=None, headers=None, timeout=None):
        self.calls += 1
        return self._response


def _make_uploader(response, retry_max=4):
    import logging

    from pluto.sync.process import _SyncUploader

    uploader = _SyncUploader(
        {'sync_process_retry_max': retry_max, 'sync_process_retry_backoff': 0.0},
        logging.getLogger('test'),
    )
    uploader._client = _FakeClient(response)
    return uploader


# raise_for_status() needs a request attached to build its error, so give the
# canned responses one (real httpx.Client always sets it).
_REQ = httpx.Request('POST', 'https://x')


def test_sync_post_400_is_terminal_and_surfaces_reason():
    """The sync uploader must not retry a 400 and must raise the server's
    reason (which the plain raise_for_status message would omit)."""
    resp = httpx.Response(
        400,
        json={'error': 'A run can have at most one group:* tag.'},
        request=_REQ,
    )
    uploader = _make_uploader(resp)

    with pytest.raises(PlutoRequestError) as excinfo:
        uploader._post_with_retry('https://x/api/runs/tags/update', b'{}', {})

    assert uploader.client.calls == 1, 'a 400 must not be retried'
    assert 'at most one group' in str(excinfo.value)
    assert excinfo.value.status_code == 400


def test_sync_post_500_is_retried():
    """5xx stays retryable in the sync uploader."""
    resp = httpx.Response(500, text='boom', request=_REQ)
    uploader = _make_uploader(resp, retry_max=2)

    with pytest.raises(Exception) as excinfo:
        uploader._post_with_retry('https://x', b'{}', {})

    # retry_max = 2 → 2 attempts total; never a PlutoRequestError (that's 4xx).
    assert uploader.client.calls == 2
    assert not isinstance(excinfo.value, PlutoRequestError)
