import json
import logging
import queue
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Union

import httpx

from .api import (
    make_compat_status_v1,
    make_compat_update_config_v1,
    make_compat_update_tags_v1,
)
from .sets import Settings

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'Interface'


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
        """Update tags on the server via HTTP API."""
        self._post_v1(
            self.settings.url_update_tags,
            self.headers,
            make_compat_update_tags_v1(self.settings, tags),
            client=self.client_api,
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
    ):
        if retry == 0:
            if isinstance(content, bytes):
                content_info = f'{len(content)} bytes'
            else:
                content_info = 'stream'
            logger.debug(
                f'{tag}: {name}: {method.__name__.upper()} '
                f'{url[:80]}... ({content_info})'
            )
        if retry >= self.settings.x_file_stream_retry_max:
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

            return None

        try:
            r = method(url, content=content, headers=headers)
            if r.status_code in [200, 201]:
                return r

            # Capture error info for potential failure logging
            error_info = f'HTTP {r.status_code}: {r.text[:100]}'

            max_retry = self.settings.x_file_stream_retry_max
            status_code = r.status_code if r else 'N/A'
            target = len(drained) if drained else 'request'
            response = r.text if r else 'N/A'
            logger.warning(
                '%s: %s: retry %s/%s: response code %s for %s from %s: %s',
                tag,
                name,
                retry + 1,
                max_retry,
                status_code,
                target,
                url,
                response,
            )
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
                '%s: %s: retry %s/%s: no response from %s: %s: %s',
                tag,
                name,
                retry + 1,
                self.settings.x_file_stream_retry_max,
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
        )

    def _put_v1(self, url, headers, content, client, name='put'):
        return self._try(
            client.put,
            url,
            headers,
            content,
            name=name,
        )

    def _post_v1(self, url, headers, q, client, name: Union[str, None] = 'post'):
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
