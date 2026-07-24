"""
Regression tests for transient login-validation failures.

login() posts the token to /api/slug as a best-effort validation. A
TRANSIENT network failure of that single request (timeout, reset — CI
under load) must not corrupt a token that was explicitly provided via
PLUTO_API_KEY/keyring: overwriting it with the '_key' sentinel made
every subsequent request send 'Bearer _key', which fails the server's
prefix check as 401 "Invalid API key" for the run's entire lifetime.
"""

from __future__ import annotations

from unittest import mock

import httpx

from pluto import auth
from pluto.sets import Settings

PROVIDED_TOKEN = 'mlpi_env_provided_token'


def _settings_with_token(token) -> Settings:
    settings = Settings()
    settings._auth = token
    settings.update_host()
    return settings


def test_transient_login_failure_keeps_provided_token():
    settings = _settings_with_token(PROVIDED_TOKEN)
    with mock.patch.object(auth.httpx, 'Client') as client_cls:
        client_cls.return_value.post.side_effect = httpx.ConnectTimeout('boom')
        auth.login(settings=settings)
    assert settings._auth == PROVIDED_TOKEN


def test_unreachable_server_without_token_still_marks_unauthenticated():
    # Preserve the interactive-flow sentinel when no token was provided.
    settings = _settings_with_token(None)
    with (
        mock.patch.object(auth.httpx, 'Client') as client_cls,
        mock.patch.object(auth.keyring, 'get_password', return_value=None),
        mock.patch.object(auth.sys, 'stdin', None),  # no interactive prompt
    ):
        client_cls.return_value.post.side_effect = httpx.ConnectTimeout('boom')
        auth.login(settings=settings)
    assert settings._auth == '_key'
