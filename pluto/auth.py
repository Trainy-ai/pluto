import getpass
import logging
import os
import sys
import webbrowser

import httpx
import keyring

from .log import setup_logger, teardown_logger
from .sets import get_console, setup
from .util import ANSI, import_lib, print_url

tlogger = logging.getLogger('auth')
tag = 'Authentication'

# Marker file written after a successful `pluto login`. The wandb compat
# layer's import hook (pluto/_wandb_hook.py) checks for this so a user who
# has only run `pluto login` (no PLUTO_API_KEY env var) still gets dual-
# logging activated. Stat-only check; never read.
LOGIN_MARKER_PATH = os.path.expanduser('~/.pluto/.login_ok')


def _write_login_marker() -> None:
    try:
        os.makedirs(os.path.dirname(LOGIN_MARKER_PATH), exist_ok=True)
        with open(LOGIN_MARKER_PATH, 'w'):
            pass
    except OSError as e:
        tlogger.debug('%s: failed to write login marker: %s', tag, e)


def _remove_login_marker() -> None:
    try:
        os.remove(LOGIN_MARKER_PATH)
    except FileNotFoundError:
        pass
    except OSError as e:
        tlogger.debug('%s: failed to remove login marker: %s', tag, e)


def login(settings=None, retry=False):
    settings = setup(settings)
    setup_logger(settings=settings, logger=tlogger)
    try:
        assert sys.platform == 'darwin'
        auth = keyring.get_password(f'{settings.tag}', f'{settings.tag}')
    except (keyring.errors.NoKeyringError, AssertionError):  # fallback
        keyring.set_keyring(import_lib('keyrings.alt.file').PlaintextKeyring())
        auth = keyring.get_password(f'{settings.tag}', f'{settings.tag}')
    if settings._auth is None:
        if auth == '':
            keyring.delete_password(f'{settings.tag}', f'{settings.tag}')
        elif auth is not None:
            settings._auth = auth
    if settings._auth == '':
        tlogger.critical(
            '%s: authentication failed: the provided token cannot be empty', tag
        )
        settings._auth = '_key'
    # Remember whether auth was pre-provided (env var, keyring) so we never
    # fall through to an interactive prompt when it was.
    auth_was_provided = settings._auth is not None and settings._auth != '_key'
    client = httpx.Client(
        http2=True,
        verify=True if not settings.insecure_disable_ssl else False,
        proxy=settings.http_proxy or settings.https_proxy or None,
    )
    try:
        r = client.post(
            url=settings.url_login,
            headers={
                'Authorization': f'Bearer {settings._auth}',
            },
        )
    except Exception as e:
        tlogger.warning(f'{tag}: server not reachable; reason: {e}')
        settings._auth = '_key'
    try:
        r.raise_for_status()
        body = r.json()
        tlogger.info(f'{tag}: logged in as {body["organization"]["slug"]}')
        keyring.set_password(f'{settings.tag}', f'{settings.tag}', f'{settings._auth}')
        _write_login_marker()
        teardown_logger(tlogger)
    except Exception as e:
        # If _auth was already provided (e.g. via env var or keyring), don't
        # prompt for interactive input — just warn and continue.  This also
        # covers the NameError case where `r` is undefined because the
        # network request itself failed.
        if auth_was_provided:
            tlogger.warning(
                '%s: server validation failed (token may still be valid); reason: %s',
                tag,
                e,
            )
            teardown_logger(tlogger)
            return
        if retry:
            tlogger.warning('%s: authentication failed; reason: %s', tag, e)
        hint1 = (
            f'{ANSI.cyan}- Please copy the API key provided in the web portal '
            'and paste it below'
        )
        hint2 = f'- You can alternatively manually open {print_url(settings.url_token)}'
        hint3 = f'{ANSI.green}- You may exit at any time by pressing CTRL+C / ⌃+C'
        tlogger.info(
            f'{tag}: initializing authentication\n\n {hint1}\n\n {hint2}\n\n {hint3}\n'
        )
        if (
            hasattr(settings._sys, 'monitor') and settings._sys.monitor() == {}
        ):  # migrate mode
            return
        else:
            webbrowser.open(url=settings.url_token)
        if not sys.stdin or not sys.stdin.isatty():
            tlogger.warning(
                '%s: no interactive terminal available for authentication; '
                'set the PLUTO_API_KEY environment variable or run `pluto login`',
                tag,
            )
            teardown_logger(tlogger)
            return
        if get_console() == 'jupyter':
            settings._auth = getpass.getpass(prompt='Enter API key: ')
        else:
            settings._auth = input(f'{ANSI.yellow}Enter API key: ')
        try:
            keyring.set_password(
                f'{settings.tag}', f'{settings.tag}', f'{settings._auth}'
            )
            _write_login_marker()
        except Exception as e:
            tlogger.critical(
                '%s: failed to save key to system keyring service: %s', tag, e
            )
        teardown_logger(tlogger)
        login(settings=settings, retry=True)


def logout(settings=None):
    settings = setup(settings)
    setup_logger(settings=settings, logger=tlogger)
    try:
        assert sys.platform == 'darwin'
        keyring.delete_password(f'{settings.tag}', f'{settings.tag}')
    except (keyring.errors.NoKeyringError, AssertionError):
        keyring.set_keyring(import_lib('keyrings.alt.file').PlaintextKeyring())
        keyring.delete_password(f'{settings.tag}', f'{settings.tag}')
    except Exception as e:
        tlogger.warning(
            '%s: failed to delete key from system keyring service: %s', tag, e
        )
    _remove_login_marker()
    tlogger.info(f'{tag}: logged out')
    teardown_logger(tlogger)
