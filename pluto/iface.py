import json
import logging
import os
import queue
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import httpx

from .api import (
    make_compat_meta_v1,
    make_compat_status_v1,
    make_compat_update_config_v1,
    make_compat_update_tags_v1,
)
from .sets import Settings

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'Interface'

# Set once a persistent 401 has been reported, so the ~4 s heartbeat doesn't
# repeat the same (static) message for the life of the run.
_auth_error_logged = False

# 4xx status codes that are commonly *transient* and worth retrying like 5xx:
#   401 Unauthorized     — auth/token validation races (e.g. the token service
#                          timing out mid-request), which clear on retry.
#   408 Request Timeout  — server-side timeout, not a bad request.
#   429 Too Many Requests — rate limited; back off and retry.
# Every other 4xx (400, 403, 404, 409, 422, ...) is a permanent client/
# validation error — e.g. 400 "A run can have at most one group:* tag." —
# where retrying the identical request would just fail again, so it's terminal.
RETRYABLE_STATUS_CODES = frozenset({401, 408, 429})

# Fallback for the API-key page when the caller has no Settings handy (e.g. the
# sync process, which only receives a settings dict). Mirrors sets.url_token.
DEFAULT_URL_TOKEN = 'https://pluto.trainy.ai/api-keys'


def _auth_key_source() -> Tuple[str, str]:
    """Describe *which* key the server rejected, so the fix is unambiguous.

    An expired key is the single most common cause of a persistent 401, and the
    fix differs depending on where the key came from (env var vs. keyring), so
    name the source rather than making the user guess.
    """
    # Membership tests, not reads: this only needs to know *whether* a key was
    # provided via the environment, so never pull the value itself into scope.
    if 'PLUTO_API_KEY' in os.environ:
        return (
            'the API key in the PLUTO_API_KEY environment variable',
            'update PLUTO_API_KEY with the new key',
        )
    if 'MLOP_API_TOKEN' in os.environ:
        return (
            'the API key in the MLOP_API_TOKEN environment variable '
            '(deprecated — use PLUTO_API_KEY)',
            'update PLUTO_API_KEY with the new key',
        )
    return (
        'the API key saved by `pluto login`',
        'run `pluto login <key>` with the new key',
    )


def auth_error_message(server_msg: str = '', url_token: Optional[str] = None) -> str:
    """Build the user-facing message for a 401 that never cleared.

    A persistent 401 reads like a server outage in the logs ("response code 401
    ... from https://pluto-api...") even though it is almost always a local
    problem: the key expired or was revoked. Say that outright, name the key
    source, and give the exact command that fixes it.
    """
    source, fix = _auth_key_source()
    msg = (
        f'authentication failed (HTTP 401): the Pluto server rejected {source}. '
        'This is an authentication failure, not a server outage — the key has '
        'most likely expired or been revoked. Create a new key at '
        f'{url_token or DEFAULT_URL_TOKEN}, then {fix}.'
    )
    # The body sometimes carries a more specific reason ("token expired",
    # "revoked"); keep it, but only when it adds something over bare "401".
    server_msg = (server_msg or '').strip()
    uninformative = ('unauthorized', '401', '401 unauthorized')
    if server_msg and server_msg.lower() not in uninformative:
        msg += f' Server said: {server_msg[:200]}'
    return msg


class PlutoRequestError(Exception):
    """Raised when a Pluto write request fails with a server validation error.

    Carries the server-provided reason (parsed from the JSON ``error`` field
    when present) so callers can surface *why* the request was rejected — e.g.
    ``"A run can have at most one group:* tag."`` — instead of a generic
    connection error. Mirrors ``query.PlutoQueryError`` for the read path.
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class PlutoAuthError(PlutoRequestError):
    """Raised when a 401 never clears — i.e. the API key is bad, not the server.

    A subclass of ``PlutoRequestError`` so existing ``except PlutoRequestError``
    handlers keep working; callers that want to say something more specific
    (``init()`` does) can catch this first.
    """

    def __init__(self, message: str):
        super().__init__(message, status_code=401)


def _server_error_message(r: httpx.Response) -> str:
    """Best-effort extraction of the human-readable reason from a response.

    The backend returns ``{"error": "<reason>"}`` on validation failures; fall
    back to the raw (truncated) body when the payload isn't the expected shape.
    """
    try:
        body = r.json()
        if isinstance(body, dict) and body.get('error'):
            return str(body['error'])
    except Exception:
        pass
    return r.text[:500]


@contextmanager
def _suppress_sentry_breadcrumbs():
    """Prevent the host app's Sentry from capturing Pluto's internal HTTP traffic.

    When a user calls ``sentry_sdk.init()`` in their own code, global HTTP
    integrations monkey-patch urllib3/httpx at the module level.  Every HTTP
    call Pluto makes (heartbeats every ~4 s, status updates, etc.) gets
    recorded as a breadcrumb on the *user's* Sentry scope — eating their
    quota with Pluto's internal API chatter.

    This context manager forks the current Sentry isolation scope so that
    any breadcrumbs added during the ``with`` block are captured on the
    fork and silently discarded when it exits.
    """
    try:
        import sentry_sdk

        # sentry_sdk 2.x
        if hasattr(sentry_sdk, 'isolation_scope'):
            with sentry_sdk.isolation_scope():
                yield
                return

        # sentry_sdk 1.x
        if hasattr(sentry_sdk, 'Hub'):
            with sentry_sdk.Hub.current.push_scope():
                yield
                return
    except ImportError:
        pass

    # sentry_sdk not installed or API unrecognised — nothing to suppress.
    yield


@contextmanager
def _suppress_httpx_logging():
    """Temporarily raise httpx/httpcore log level to WARNING for the call.

    Used to silence the noisy INFO-level log lines from high-frequency
    endpoints (e.g. the trigger/heartbeat that fires every ~4 s).  Only
    affects the duration of the ``with`` block — the user's own httpx log
    level is saved and restored afterwards.
    """
    httpx_logger = logging.getLogger('httpx')
    httpcore_logger = logging.getLogger('httpcore')
    original_httpx = httpx_logger.level
    original_httpcore = httpcore_logger.level
    httpx_logger.setLevel(logging.WARNING)
    httpcore_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        httpx_logger.setLevel(original_httpx)
        httpcore_logger.setLevel(original_httpcore)


class ServerInterface:
    """
    HTTP interface for communicating with the Pluto backend.

    This class provides HTTP utilities for:
    - Creating runs (via create_run API)
    - Updating run status (start/stop)
    - Direct API calls (alerts, etc.)

    Note: Data upload (metrics, files, structured data) is handled by the
    sync process (pluto/sync/process.py), not by this class.
    """

    def __init__(self, config: dict, settings: Settings) -> None:
        self.config = config
        self.settings = settings

        self.headers = {
            'Authorization': f'Bearer {self.settings._auth}',
            'Content-Type': 'application/json',
            'User-Agent': f'{self.settings.tag}',
            'X-Run-Id': f'{self.settings._op_id}',
            'X-Run-Name': f'{self.settings._op_name}',
            'X-Project-Name': f'{self.settings.project}',
        }
        self.headers_num = self.headers.copy()
        self.headers_num.update({'Content-Type': 'application/x-ndjson'})

        self.client = httpx.Client(
            http2=True,
            verify=not self.settings.insecure_disable_ssl,
            proxy=self.settings.http_proxy or self.settings.https_proxy or None,
            limits=httpx.Limits(
                max_keepalive_connections=self.settings.x_file_stream_max_conn,
                max_connections=self.settings.x_file_stream_max_conn,
            ),
            timeout=httpx.Timeout(
                self.settings.x_file_stream_timeout_seconds,
            ),
        )
        self.client_api = httpx.Client(
            http2=True,
            verify=not self.settings.insecure_disable_ssl,
            proxy=self.settings.http_proxy or self.settings.https_proxy or None,
            timeout=httpx.Timeout(
                self.settings.x_file_stream_timeout_seconds,
            ),
        )

    def close(self) -> None:
        """Close HTTP clients."""
        if self.client:
            self.client.close()
        if self.client_api:
            self.client_api.close()

    def update_status(self, trace: Union[Any, None] = None) -> None:
        """Update run status on the server (called at finish)."""
        self._post_v1(
            self.settings.url_stop,
            self.headers,
            make_compat_status_v1(self.settings, trace),
            client=self.client_api,
        )

    def update_tags(self, tags: List[str]) -> None:
        """Update tags on the server via HTTP API.

        Raises ``PlutoRequestError`` if the server rejects the update (e.g. a
        validation error), so a failed ``run.update_tags(...)`` surfaces the
        reason instead of silently no-op'ing.
        """
        self._post_v1(
            self.settings.url_update_tags,
            self.headers,
            make_compat_update_tags_v1(self.settings, tags),
            client=self.client_api,
            raise_on_error=True,
        )

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update config on the server via HTTP API."""
        self._post_v1(
            self.settings.url_update_config,
            self.headers,
            make_compat_update_config_v1(self.settings, config),
            client=self.client_api,
        )

    # Keep legacy underscore methods for backwards compatibility
    def _update_status(self, settings, trace: Union[Any, None] = None):
        """Legacy method - use update_status() instead."""
        self.update_status(trace)

    def _update_tags(self, tags: List[str]):
        """Legacy method - use update_tags() instead."""
        self.update_tags(tags)

    def _update_config(self, config: Dict[str, Any]):
        """Legacy method - use update_config() instead."""
        self.update_config(config)

    def _update_meta(
        self,
        num: Optional[List[str]] = None,
        df: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Register log names (metrics/files) with the server.

        This tells the server what metric/file names to expect so it can
        properly index and display them in dashboards.

        Args:
            num: List of numeric metric names
            df: Dict mapping file type names to lists of log names
        """
        # Suppress the per-request httpx INFO line ("HTTP Request: POST
        # .../api/runs/logName/add ..."). One POST fires per new metric/file
        # name, so this is noisy; the heartbeat/status path suppresses it too.
        if num:
            self._post_v1(
                self.settings.url_meta,
                self.headers,
                make_compat_meta_v1(num, 'num', self.settings),
                client=self.client_api,
                suppress_httpx_logs=True,
            )
        if df:
            for type_name, names in df.items():
                self._post_v1(
                    self.settings.url_meta,
                    self.headers,
                    make_compat_meta_v1(names, type_name, self.settings),
                    client=self.client_api,
                    suppress_httpx_logs=True,
                )

    def _log_auth_error_once(self, message: str) -> None:
        """Log a persistent-401 message at most once per process."""
        global _auth_error_logged
        if _auth_error_logged:
            return
        _auth_error_logged = True
        # `message` is generated text plus the public API-key page URL — never
        # key material. CodeQL flags it because its taint tracking is
        # field-insensitive: the URL is read off a Settings object that also
        # holds `_auth` (which can come from getpass()), so it treats any
        # attribute of that object as a password.
        logger.critical(  # codeql[py/clear-text-logging-sensitive-data]
            '%s: %s',
            tag,
            message,  # codeql[py/clear-text-logging-sensitive-data]
        )

    def _log_failed_request(
        self,
        request_type: str,
        url: str,
        payload_info: str,
        error_info: str,
        retry_count: int,
    ) -> None:
        """Log failed requests to file after all retries exhausted."""

        # Only log failures in DEBUG mode
        if self.settings.x_log_level > logging.DEBUG:
            return

        failure_log_path = f'{self.settings.get_dir()}/failed_requests.log'

        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'request_type': request_type,
            'url': url,
            'payload_info': payload_info,
            'error_info': error_info,
            'retries_attempted': retry_count,
        }

        try:
            with open(failure_log_path, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
        except Exception as e:
            logger.debug(f'{tag}: failed to write failure log: {e}')

    def _try(
        self,
        method,
        url,
        headers,
        content,
        name: Union[str, None] = None,
        drained: Optional[List[Any]] = None,
        retry: int = 0,
        error_info: str = '',
        last_status: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        suppress_httpx_logs: bool = False,
        raise_on_error: bool = False,
    ):
        effective_max_retries = (
            max_retries
            if max_retries is not None
            else self.settings.x_file_stream_retry_max
        )

        if retry == 0:
            if isinstance(content, bytes):
                content_info = f'{len(content)} bytes'
            else:
                content_info = 'stream'
            logger.debug(
                f'{tag}: {name}: {method.__name__.upper()} '
                f'{url[:80]}... ({content_info})'
            )
        if retry > effective_max_retries:
            if retry > 0:
                logger.critical(f'{tag}: {name}: failed after {retry} retries')

            # Log failure details to file
            payload_info = f'{len(drained)} items' if drained else 'single request'
            self._log_failed_request(
                request_type=name or 'unknown',
                url=url,
                payload_info=payload_info,
                error_info=error_info,
                retry_count=retry,
            )

            # A 401 that survived every retry is not the transient auth race
            # RETRYABLE_STATUS_CODES is there for — it's an expired/revoked key.
            # Report it as such instead of as a generic "HTTP 401" that reads
            # like a server outage.
            # (Only when the *final* attempt was the 401 — if it ended on a
            # network error instead, that's a connectivity story, not an auth
            # one, and error_info carries no server message.)
            if last_status == 401 and error_info.startswith('HTTP 401'):
                auth_msg = auth_error_message(
                    server_msg=error_info.partition(': ')[2],
                    url_token=getattr(self.settings, 'url_token', None),
                )
                if raise_on_error:
                    raise PlutoAuthError(auth_msg)
                # Paths that swallow failures (heartbeats, status updates, log
                # name registration) still need to tell the user *why* their
                # data stopped flowing — but the heartbeat fires every ~4 s, so
                # say it once per process rather than on every cycle.
                self._log_auth_error_once(auth_msg)
                return None

            # Distinguish a persistent server error (we got HTTP responses but
            # they never succeeded) from an unreachable server (network
            # exceptions → error_info doesn't start with "HTTP"). Only the
            # former carries a server-provided reason worth raising. last_status
            # is threaded down from the last HTTP response so the exception
            # reports the real code (e.g. 500) rather than None.
            if raise_on_error and error_info.startswith('HTTP '):
                raise PlutoRequestError(error_info, status_code=last_status)

            return None

        try:
            kwargs: Dict[str, Any] = {}
            if timeout is not None:
                kwargs['timeout'] = timeout
            with _suppress_sentry_breadcrumbs():
                if suppress_httpx_logs and self.settings.x_log_level > logging.DEBUG:
                    with _suppress_httpx_logging():
                        r = method(url, content=content, headers=headers, **kwargs)
                else:
                    r = method(url, content=content, headers=headers, **kwargs)
            if r.status_code in [200, 201]:
                return r

            # Capture error info for potential failure logging
            server_msg = _server_error_message(r)
            error_info = f'HTTP {r.status_code}: {server_msg[:200]}'
            # Remember the status so a later retry-exhaustion raise carries the
            # real code, not None (network errors below leave this untouched).
            last_status = r.status_code

            target = len(drained) if drained else 'request'
            # High-frequency endpoints (the trigger/heartbeat that fires
            # every ~4 s) set suppress_httpx_logs; route their non-200
            # responses to DEBUG so a flaky server doesn't spam WARNING.
            log_response = logger.debug if suppress_httpx_logs else logger.warning
            log_response(
                '%s: %s: attempt %s/%s: response code %s for %s from %s: %s',
                tag,
                name,
                retry + 1,
                effective_max_retries + 1,
                r.status_code,
                target,
                url,
                server_msg,
            )

            # A permanent 4xx client/validation error — retrying the identical
            # payload will never succeed, so stop immediately (no wasted backoff)
            # and, when asked, surface the server's reason to the caller.
            # Transient 4xx (RETRYABLE_STATUS_CODES: 401/408/429) fall through to
            # the retry/backoff path below, same as 5xx.
            if (
                400 <= r.status_code < 500
                and r.status_code not in RETRYABLE_STATUS_CODES
            ):
                if raise_on_error:
                    raise PlutoRequestError(server_msg, status_code=r.status_code)
                return None
        except PlutoRequestError:
            # A deliberate terminal error (4xx) — propagate, don't treat it as
            # a transient network failure to be retried by the except below.
            raise
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            httpx.RemoteProtocolError,
            httpx.LocalProtocolError,
        ) as e:
            # Treat connection errors as shutdown signals - don't retry
            # This prevents hanging during atexit when sockets are being torn down
            logger.debug(
                '%s: %s: connection error (likely shutdown): %s: %s',
                tag,
                name,
                type(e).__name__,
                e,
            )
            return None
        except Exception as e:
            # Capture error info for potential failure logging
            error_info = f'{type(e).__name__}: {str(e)}'

            logger.debug(
                '%s: %s: attempt %s/%s: no response from %s: %s: %s',
                tag,
                name,
                retry + 1,
                effective_max_retries + 1,
                url,
                type(e).__name__,
                e,
            )
        time.sleep(
            min(
                self.settings.x_file_stream_retry_wait_min_seconds * (2 ** (retry + 1)),
                self.settings.x_file_stream_retry_wait_max_seconds,
            )
        )

        return self._try(
            method,
            url,
            headers,
            content,
            name=name,
            drained=drained,
            retry=retry + 1,
            error_info=error_info,
            last_status=last_status,
            max_retries=effective_max_retries,
            timeout=timeout,
            suppress_httpx_logs=suppress_httpx_logs,
            raise_on_error=raise_on_error,
        )

    def _put_v1(
        self,
        url,
        headers,
        content,
        client,
        name='put',
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        suppress_httpx_logs: bool = False,
    ):
        return self._try(
            client.put,
            url,
            headers,
            content,
            name=name,
            max_retries=max_retries,
            timeout=timeout,
            suppress_httpx_logs=suppress_httpx_logs,
        )

    def _post_v1(
        self,
        url,
        headers,
        q,
        client,
        name: Union[str, None] = 'post',
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        suppress_httpx_logs: bool = False,
        raise_on_error: bool = False,
    ):
        # Support both queue and direct content
        if isinstance(q, queue.Queue):
            b: List[Any] = []
            content = self._queue_iter(q, b)
            drained = b
        else:
            content = q
            drained = None

        s = time.time()
        r = self._try(
            client.post,
            url,
            headers,
            content,
            name=name,
            drained=drained,
            max_retries=max_retries,
            timeout=timeout,
            suppress_httpx_logs=suppress_httpx_logs,
            raise_on_error=raise_on_error,
        )

        if (
            r
            and r.status_code in [200, 201]
            and name is not None
            and drained is not None
        ):
            logger.debug(
                f'{tag}: {name}: sent {len(drained)} line(s) at '
                f'{len(drained) / (time.time() - s):.2f} lines/s to {url}'
            )
        return r

    def _queue_iter(self, q: queue.Queue[Any], b: List[Any]) -> Iterable[Any]:
        """Iterate over queue items for streaming upload."""
        s = time.time()
        while (
            len(b) < self.settings.x_file_stream_max_size
            and (time.time() - s) < self.settings.x_file_stream_transmit_interval
        ):
            try:
                v = q.get(timeout=self.settings.x_internal_check_process)
                b.append(v)
                yield v
            except queue.Empty:
                break
