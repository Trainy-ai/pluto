"""Unit tests for login-time auth error reporting (pluto/auth.py).

An expired API key is the most common cause of a 401, and ``login()`` is the
first place it can be caught — before create-run, before any metric is logged.
These tests pin the wording so a rejected key never gets reported as a
"may still be valid" server hiccup.
"""

import logging
from unittest.mock import MagicMock

import httpx
import pytest

import pluto.auth as auth
from pluto.sets import Settings


def _make_settings(token='expired-key'):
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


def _response(status_code, url='https://pluto-api.trainy.ai/api/slug', text=''):
    return httpx.Response(status_code, text=text, request=httpx.Request('POST', url))


def test_login_401_reports_expired_key_not_maybe_valid(monkeypatch, caplog, no_keyring):
    """The server saw the key and rejected it — that is definitive."""
    monkeypatch.delenv('PLUTO_API_KEY', raising=False)
    monkeypatch.delenv('MLOP_API_TOKEN', raising=False)
    _patch_login_response(monkeypatch, _response(401, text='Unauthorized'))

    with caplog.at_level(logging.CRITICAL, logger='auth'):
        auth.login(settings=_make_settings())

    messages = [rec.getMessage() for rec in caplog.records]
    joined = '\n'.join(messages)
    assert 'expired' in joined
    assert 'not a server outage' in joined
    assert 'pluto login' in joined
    assert (
        'may still be valid' not in joined
    ), 'a 401 is definitive — never suggest the key might be fine'


def test_login_5xx_keeps_may_still_be_valid_wording(monkeypatch, caplog, no_keyring):
    """A server-side failure genuinely says nothing about the key's validity,
    so that path must keep its softer wording."""
    _patch_login_response(monkeypatch, _response(500, text='boom'))

    with caplog.at_level(logging.WARNING, logger='auth'):
        auth.login(settings=_make_settings())

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
        auth.login(settings=_make_settings())

    joined = '\n'.join(rec.getMessage() for rec in caplog.records)
    assert 'server not reachable' in joined
    assert 'expired' not in joined
