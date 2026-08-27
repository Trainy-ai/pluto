"""Unit tests for write-error handling (pluto/iface.py) without a real server.

These exercise the retry/raise policy of ``_try``:
- 4xx responses are terminal (no retry) and surface the server's message.
- 5xx responses are retried.
- Network exceptions never raise PlutoRequestError (caller sees None).

Plus how a persistent 401 is reported. An expired API key is the most common
cause, and ``login()`` (pluto/auth.py) is the first place it can be caught, so
its wording is pinned here too — it shares the message builder with ``_try``.
"""

import logging
from unittest.mock import MagicMock

import httpx
import pytest

import pluto.auth as auth
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


def test_server_error_message_reads_message_when_error_is_generic():
    """Auth failures put the status phrase in `error` and the real reason in
    `message`, so reading `error` alone throws away the only useful part."""
    # Shape returned by the web API's withApiKey middleware.
    r = _resp(
        401, json_body={'error': 'Unauthorized', 'message': 'API key has expired'}
    )
    assert _server_error_message(r) == 'API key has expired'
    # Shape returned by the ingest service (numeric code, no `error` field).
    r = _resp(401, json_body={'code': 1002, 'message': 'API key has been revoked'})
    assert _server_error_message(r) == 'API key has been revoked'
    # A generic `error` with nothing else still comes back, rather than ''.
    assert _server_error_message(_resp(401, json_body={'error': 'Unauthorized'})) == (
        'Unauthorized'
    )


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


def test_message_states_server_reason_as_fact(monkeypatch):
    """The server knows whether the key expired, was revoked, or was never
    valid — when it says so, report it rather than guessing."""
    monkeypatch.delenv('PLUTO_API_KEY', raising=False)
    monkeypatch.delenv('MLOP_API_TOKEN', raising=False)

    msg = http_401_message(server_msg='API key has expired')
    assert 'The server says: API key has expired.' in msg
    assert 'most likely' not in msg, 'do not hedge when the server told us'

    # With nothing useful from the server, the hedge is the honest wording.
    for uninformative in ('', 'Unauthorized', '401'):
        msg = http_401_message(server_msg=uninformative)
        assert 'most likely expired or been revoked' in msg
        assert 'The server says' not in msg


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


def test_persistent_401_notice_resets_per_run(monkeypatch, caplog):
    """ "Once" is once per run, not once per process: a sweep creating many runs
    in one process must not have run N's bad key silence run N+1's uploads."""
    monkeypatch.setattr('pluto.iface._logged_401', False)

    def fake_method(url, content=None, headers=None, **kwargs):
        return _resp(401, text='Unauthorized')

    with caplog.at_level(logging.CRITICAL, logger='pluto'):
        for _ in range(2):
            # A new ServerInterface is built per run.
            iface = _make_iface()
            for _ in range(2):
                iface._try(fake_method, 'https://x', {}, b'{}', name='trigger')

    auth_logs = [r for r in caplog.records if 'authentication failed' in r.message]
    assert len(auth_logs) == 2, 'each run gets its own one-time notice'


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


def test_try_connection_reset_before_shutdown_is_retried():
    """A connection reset during normal operation is transient — retry it.

    Regression test: _try treated every connection error as an atexit
    shutdown signal and returned None with zero retries. In the field the
    same exception fires during container startup (seen in production), where
    giving up silently drops the request.
    """
    iface = _make_iface()
    calls = {'n': 0}

    def fake_method(url, content=None, headers=None, **kwargs):
        calls['n'] += 1
        if calls['n'] < 3:
            raise ConnectionResetError('connection reset by peer')
        return _resp(200)

    r = iface._try(fake_method, 'https://x', {}, b'{}', name='create')
    assert r is not None and r.status_code == 200
    assert calls['n'] == 3, 'connection resets before shutdown must be retried'


def test_try_connection_reset_during_shutdown_returns_none_without_retry():
    """Once shutdown has begun, connection errors must not hang atexit
    (heartbeat / trigger / streaming-upload spam)."""
    iface = _make_iface()
    iface.mark_shutting_down()
    calls = {'n': 0}

    def fake_method(url, content=None, headers=None, **kwargs):
        calls['n'] += 1
        raise ConnectionResetError('connection reset by peer')

    r = iface._try(fake_method, 'https://x', {}, b'{}', name='create')
    assert r is None
    assert calls['n'] == 1, 'no retries once shutdown has begun'


def test_try_connection_reset_retried_when_flagged_even_during_shutdown():
    """For critical one-shot requests (retry_connection_errors=True, e.g. the
    terminal status update) a dropped socket is transient and IS retried — even
    once shutdown has begun, since that's exactly when the status update runs."""
    iface = _make_iface()
    iface.mark_shutting_down()
    calls = {'n': 0}

    def fake_method(url, content=None, headers=None, **kwargs):
        calls['n'] += 1
        if calls['n'] < 3:
            raise ConnectionResetError('peer reset')
        return _resp(200, json_body={'ok': True})

    r = iface._try(
        fake_method,
        'https://x/api/runs/status/update',
        {},
        b'{}',
        name='status',
        raise_on_error=True,
        retry_connection_errors=True,
    )
    assert r is not None and r.status_code == 200
    assert calls['n'] == 3, 'a connection reset must be retried when flagged'


def test_try_connection_reset_raises_after_retries_when_flagged():
    """A reset that never clears raises (not None) so the caller can't mistake a
    dropped terminal-status POST for success."""
    iface = _make_iface()  # x_file_stream_retry_max = 2
    calls = {'n': 0}

    def fake_method(url, content=None, headers=None, **kwargs):
        calls['n'] += 1
        raise ConnectionResetError('peer reset')

    with pytest.raises(PlutoRequestError):
        iface._try(
            fake_method,
            'https://x/api/runs/status/update',
            {},
            b'{}',
            name='status',
            raise_on_error=True,
            retry_connection_errors=True,
        )
    assert calls['n'] == 3, 'initial attempt + 2 retries'


def test_update_status_retries_reset_and_raises_on_persistent_failure(monkeypatch):
    """update_status() is the run's terminal transition: it must retry a dropped
    socket and ultimately raise, never silently swallow it (the finish-strands-
    run bug). Wires the retry_connection_errors + raise_on_error flags end-to-end."""
    iface = _make_iface()
    calls = {'n': 0}

    def fake_post(url, content=None, headers=None, **kwargs):
        calls['n'] += 1
        raise ConnectionResetError('peer reset')

    monkeypatch.setattr(iface.client_api, 'post', fake_post)

    with pytest.raises(PlutoRequestError):
        iface.update_status()
    assert calls['n'] >= 2, 'update_status must retry the reset before failing'


def test_log_failed_request_written_at_default_log_level(tmp_path):
    """Permanent failures must be written to failed_requests.log always.

    Regression test: the write was gated on x_log_level <= DEBUG, which no
    production configuration uses — so the post-mortem artifact never
    existed exactly when it was needed.
    """
    iface = _make_iface()
    iface.settings.dir = str(tmp_path)
    log_dir = iface.settings.get_dir()
    import os

    os.makedirs(log_dir, exist_ok=True)
    assert iface.settings.x_log_level > 10  # default config, not DEBUG

    iface._log_failed_request(
        request_type='file',
        url='https://x/files',
        payload_info='1 items',
        error_info='ConnectionResetError: reset',
        retry_count=5,
    )

    log_path = os.path.join(log_dir, 'failed_requests.log')
    assert os.path.exists(log_path), 'failure log must be written outside DEBUG'
    with open(log_path) as f:
        entry = f.read()
    assert 'ConnectionResetError' in entry


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


# --- login()-time reporting (pluto/auth.py) ---------------------------------
#
# login() runs before create-run, so it is the earliest place a rejected key
# can be reported. It builds its message with the same http_401_message() the
# _try path uses, which is why these live alongside the tests above.


def _make_login_settings(token='expired-key'):
    settings = Settings()
    settings.mode = 'noop'
    settings.update_host()
    settings._auth = token
    return settings


@pytest.fixture
def no_keyring(monkeypatch):
    """Keep login() away from the real keyring (and the darwin/fallback fork)."""
    monkeypatch.setattr(auth.sys, 'platform', 'darwin')
    monkeypatch.setattr(auth.keyring, 'get_password', lambda *a, **k: None)
    monkeypatch.setattr(auth.keyring, 'set_password', lambda *a, **k: None)


def _patch_login_response(monkeypatch, response):
    client = MagicMock()
    client.post.return_value = response
    monkeypatch.setattr(auth.httpx, 'Client', lambda **kwargs: client)
    return client


def _login_response(
    status_code, url='https://pluto-api.trainy.ai/api/slug', text='', json_body=None
):
    request = httpx.Request('POST', url)
    if json_body is not None:
        return httpx.Response(status_code, json=json_body, request=request)
    return httpx.Response(status_code, text=text, request=request)


def test_login_401_reports_expired_key_not_maybe_valid(monkeypatch, caplog, no_keyring):
    """The server saw the key and rejected it — that is definitive, and it says
    which problem it is, so login() must pass that reason through."""
    monkeypatch.delenv('PLUTO_API_KEY', raising=False)
    monkeypatch.delenv('MLOP_API_TOKEN', raising=False)
    _patch_login_response(
        monkeypatch,
        _login_response(
            401, json_body={'error': 'Unauthorized', 'message': 'API key has expired'}
        ),
    )

    with caplog.at_level(logging.CRITICAL, logger='auth'):
        auth.login(settings=_make_login_settings())

    joined = '\n'.join(rec.getMessage() for rec in caplog.records)
    # The server's own words, not the generic fallback.
    assert 'The server says: API key has expired.' in joined
    assert 'most likely' not in joined
    assert 'not a server outage' in joined
    assert 'pluto login' in joined
    assert (
        'may still be valid' not in joined
    ), 'a 401 is definitive — never suggest the key might be fine'


def test_login_401_without_a_reason_falls_back_to_the_hedge(
    monkeypatch, caplog, no_keyring
):
    monkeypatch.delenv('PLUTO_API_KEY', raising=False)
    monkeypatch.delenv('MLOP_API_TOKEN', raising=False)
    _patch_login_response(monkeypatch, _login_response(401, text='Unauthorized'))

    with caplog.at_level(logging.CRITICAL, logger='auth'):
        auth.login(settings=_make_login_settings())

    joined = '\n'.join(rec.getMessage() for rec in caplog.records)
    assert 'most likely expired or been revoked' in joined
    assert 'The server says' not in joined


def test_login_5xx_keeps_may_still_be_valid_wording(monkeypatch, caplog, no_keyring):
    """A server-side failure genuinely says nothing about the key's validity,
    so that path must keep its softer wording."""
    _patch_login_response(monkeypatch, _login_response(500, text='boom'))

    with caplog.at_level(logging.WARNING, logger='auth'):
        auth.login(settings=_make_login_settings())

    joined = '\n'.join(rec.getMessage() for rec in caplog.records)
    assert 'may still be valid' in joined
    assert 'expired' not in joined


def test_login_unreachable_server_does_not_blame_the_key(
    monkeypatch, caplog, no_keyring
):
    client = MagicMock()
    client.post.side_effect = httpx.ConnectError('no route')
    monkeypatch.setattr(auth.httpx, 'Client', lambda **kwargs: client)

    with caplog.at_level(logging.WARNING, logger='auth'):
        auth.login(settings=_make_login_settings())

    joined = '\n'.join(rec.getMessage() for rec in caplog.records)
    assert 'server not reachable' in joined
    assert 'expired' not in joined
