"""Unit tests for ServerInterface write-error handling (pluto/iface.py).

These exercise the retry/raise policy of ``_try`` without a real server:
- 4xx responses are terminal (no retry) and surface the server's message.
- 5xx responses are retried.
- Network exceptions never raise PlutoRequestError (caller sees None).
"""

import httpx
import pytest

from pluto.iface import PlutoRequestError, ServerInterface, _server_error_message
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


def test_sync_post_400_is_terminal_and_surfaces_reason():
    """The sync uploader must not retry a 400 and must raise the server's
    reason (which the plain raise_for_status message would omit)."""
    resp = httpx.Response(
        400, json={'error': 'A run can have at most one group:* tag.'}
    )
    uploader = _make_uploader(resp)

    with pytest.raises(PlutoRequestError) as excinfo:
        uploader._post_with_retry('https://x/api/runs/tags/update', b'{}', {})

    assert uploader.client.calls == 1, 'a 400 must not be retried'
    assert 'at most one group' in str(excinfo.value)
    assert excinfo.value.status_code == 400


def test_sync_post_500_is_retried():
    """5xx stays retryable in the sync uploader."""
    resp = httpx.Response(500, text='boom')
    uploader = _make_uploader(resp, retry_max=2)

    with pytest.raises(Exception) as excinfo:
        uploader._post_with_retry('https://x', b'{}', {})

    # retry_max = 2 → 2 attempts total; never a PlutoRequestError (that's 4xx).
    assert uploader.client.calls == 2
    assert not isinstance(excinfo.value, PlutoRequestError)
