"""
Internal Sentry APM for the Pluto SDK.

Uses an isolated Sentry client so it never interferes with a user's own
Sentry configuration. Opt out by setting PLUTO_DISABLE_TELEMETRY=1.
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(f'{__name__.split(".")[0]}')

_SENTRY_DSN = (
    'https://564fa041c398f8f54db48b830a8437df'
    '@o4511152987766784.ingest.us.sentry.io/4511152989405184'
)

_client: Any = None
_scope: Any = None
_disabled: bool = False


def _is_telemetry_disabled() -> bool:
    val = os.environ.get('PLUTO_DISABLE_TELEMETRY', '').lower()
    return val in ('1', 'true', 'yes')


def _init_sentry() -> None:
    """Lazily initialize an isolated Sentry client for SDK telemetry."""
    global _client, _scope, _disabled

    if _client is not None or _disabled:
        return

    if _is_telemetry_disabled():
        _disabled = True
        return

    try:
        import sentry_sdk
        from sentry_sdk.transport import HttpTransport

        from pluto import __version__, _get_git_commit

        _client = sentry_sdk.Client(
            dsn=_SENTRY_DSN,
            release=f'pluto-sdk@{__version__}+{_get_git_commit()}',
            traces_sample_rate=0.2,
            environment=os.environ.get('PLUTO_ENV', 'production'),
            transport=HttpTransport,
            # Only capture events from our own package
            in_app_include=['pluto'],
            # Minimal integrations — we don't want to hook user code
            default_integrations=False,
        )
        _scope = sentry_sdk.Scope(client=_client)
    except Exception:
        _disabled = True


def set_tag(key: str, value: str) -> None:
    _init_sentry()
    if _scope is not None:
        _scope.set_tag(key, value)


def set_context(name: str, data: Dict[str, Any]) -> None:
    _init_sentry()
    if _scope is not None:
        _scope.set_context(name, data)


def set_user(user_id: Optional[str]) -> None:
    _init_sentry()
    if _scope is not None and user_id:
        _scope.set_user({'id': user_id})


def capture_exception(error: Optional[BaseException] = None) -> None:
    _init_sentry()
    if _scope is not None:
        _scope.capture_exception(error)


def capture_message(message: str, level: str = 'info') -> None:
    _init_sentry()
    if _scope is not None:
        _scope.capture_message(message, level=level)


def flush() -> None:
    if _client is not None:
        try:
            _client.flush(timeout=2.0)
        except Exception:
            pass
