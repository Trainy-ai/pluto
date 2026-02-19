"""
Read/query API for Pluto runs.

Provides programmatic access to runs, metrics, files, and logs stored in Pluto.
Wraps the Pluto server REST API (``/api/runs/*``) with API-key authentication.

Usage::

    import pluto.query as pq

    # Uses PLUTO_API_TOKEN env var and default server
    runs = pq.list_runs("my-project")
    run = pq.get_run("my-project", "MMP-42")
    metrics = pq.get_metrics("my-project", run["id"], metric_names=["val/loss"])

    # Explicit auth / custom server
    client = pq.Client(api_token="plt_...", host="https://pluto.example.com")
    runs = client.list_runs("my-project")
"""

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'Query'

_DEFAULT_URL_API = 'https://pluto-api.trainy.ai'
_DEFAULT_TIMEOUT = 30
_RETRY_MAX = 4
_RETRY_WAIT_MIN = 0.5
_RETRY_WAIT_MAX = 4.0


class QueryError(Exception):
    """Raised when a query to the Pluto server fails."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class Client:
    """HTTP client for reading data from the Pluto server.

    Args:
        api_token: API token for authentication. Defaults to ``PLUTO_API_TOKEN``
            environment variable.
        host: Server URL (e.g. ``https://pluto.trainy.ai`` or
            ``https://pluto-api.trainy.ai``). When a bare hostname or
            ``host:port`` is given (matching the ``pluto.init(host=...)``
            pattern), the API URL is derived as ``http://{host}:3001``.
            Defaults to ``PLUTO_URL_API`` env var or the production API URL.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        host: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._api_token = api_token or _resolve_api_token()
        if not self._api_token:
            raise QueryError(
                'No API token provided. Set PLUTO_API_TOKEN environment variable '
                'or pass api_token to Client().'
            )

        self._url_api = _resolve_url_api(host)
        self._client = httpx.Client(
            headers={
                'x-api-key': self._api_token,
                'User-Agent': 'pluto-query',
            },
            timeout=httpx.Timeout(timeout),
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> 'Client':
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects in the organization.

        Returns:
            List of project dicts with keys: ``id``, ``name``, ``runCount``,
            ``createdAt``, ``updatedAt``.
        """
        return self._get('/api/runs/projects')

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def list_runs(
        self,
        project: str,
        search: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List runs in a project.

        Args:
            project: Project name.
            search: Full-text search on run name.
            tags: Filter by tags (AND logic). Only runs matching *all*
                specified tags are returned.
            limit: Maximum number of runs to return (max 200).

        Returns:
            List of run dicts with keys: ``id``, ``name``, ``displayId``,
            ``status``, ``tags``, ``config``, ``createdAt``, ``updatedAt``,
            ``url``.
        """
        params: Dict[str, Any] = {'projectName': project, 'limit': min(limit, 200)}
        if search is not None:
            params['search'] = search
        if tags is not None:
            params['tags'] = ','.join(tags)
        return self._get('/api/runs/list', params=params)

    def get_run(
        self,
        project: str,
        run_id: Union[int, str],
    ) -> Dict[str, Any]:
        """Get full details for a single run.

        Args:
            project: Project name.
            run_id: Numeric server ID (``int``) or display ID string
                (e.g. ``"MMP-1"``).

        Returns:
            Run dict with keys: ``id``, ``name``, ``displayId``, ``status``,
            ``tags``, ``config``, ``systemMetadata``, ``logNames``,
            ``createdAt``, ``updatedAt``, ``url``.
        """
        if isinstance(run_id, int):
            return self._get(
                f'/api/runs/details/{run_id}',
                params={'projectName': project},
            )
        else:
            return self._get(
                f'/api/runs/details/by-display-id/{run_id}',
                params={'projectName': project},
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metric_names(
        self,
        project: str,
        run_ids: Optional[List[int]] = None,
        search: Optional[str] = None,
        limit: int = 500,
    ) -> List[str]:
        """List distinct metric names.

        Args:
            project: Project name.
            run_ids: Restrict to these run IDs. ``None`` returns metrics
                across the whole project.
            search: Substring filter on metric name.
            limit: Maximum number of names (max 500).

        Returns:
            List of metric name strings.
        """
        params: Dict[str, Any] = {'projectName': project, 'limit': min(limit, 500)}
        if run_ids is not None:
            params['runIds'] = ','.join(str(r) for r in run_ids)
        if search is not None:
            params['search'] = search
        return self._get('/api/runs/metric-names', params=params)

    def get_metrics(
        self,
        project: str,
        run_id: int,
        metric_names: Optional[List[str]] = None,
        limit: int = 10000,
    ) -> Any:
        """Fetch time-series metric data for a run.

        The server returns up to ``limit`` data points per metric, sampled
        via reservoir sampling when the full series exceeds the limit.

        When *pandas* is installed the return value is a
        :class:`~pandas.DataFrame` with columns ``metric``, ``step``,
        ``value``, ``time``. Otherwise a list of dicts is returned.

        Args:
            project: Project name.
            run_id: Numeric server ID.
            metric_names: Metric names to fetch. ``None`` fetches all.
            limit: Max data points per metric (max 10 000).

        Returns:
            ``pandas.DataFrame`` or ``list[dict]``.
        """
        params: Dict[str, Any] = {
            'runId': run_id,
            'projectName': project,
            'limit': min(limit, 10000),
        }

        if metric_names is not None and len(metric_names) > 1:
            # Endpoint only supports a single logName filter, so fetch
            # each metric individually and merge the results.
            raw: list = []
            for name in metric_names:
                p = dict(params)
                p['logName'] = name
                raw.extend(self._get('/api/runs/metrics', params=p))
        else:
            if metric_names is not None and len(metric_names) == 1:
                params['logName'] = metric_names[0]
            raw = self._get('/api/runs/metrics', params=params)

        return _to_dataframe(raw)

    # ------------------------------------------------------------------
    # Statistics / comparison
    # ------------------------------------------------------------------

    def get_statistics(
        self,
        project: str,
        run_id: int,
        metric_names: Optional[List[str]] = None,
    ) -> Any:
        """Compute statistics for run metrics.

        Returns per-metric aggregations: ``count``, ``min``, ``max``,
        ``mean``, ``stddev``, as well as anomaly detection data.

        Args:
            project: Project name.
            run_id: Numeric server ID.
            metric_names: Restrict to these metrics.

        Returns:
            Server response dict.
        """
        params: Dict[str, Any] = {'runId': run_id, 'projectName': project}
        if metric_names is not None and len(metric_names) == 1:
            params['logName'] = metric_names[0]
        return self._get('/api/runs/statistics', params=params)

    def compare_runs(
        self,
        project: str,
        run_ids: List[int],
        metric_name: str,
    ) -> Dict[str, Any]:
        """Compare a metric across multiple runs.

        Args:
            project: Project name.
            run_ids: List of numeric run IDs (max 100).
            metric_name: The metric to compare.

        Returns:
            Dict with per-run statistics and a ``bestRun`` recommendation.
        """
        params: Dict[str, Any] = {
            'runIds': ','.join(str(r) for r in run_ids[:100]),
            'projectName': project,
            'logName': metric_name,
        }
        return self._get('/api/runs/compare', params=params)

    def leaderboard(
        self,
        project: str,
        metric_name: str,
        aggregation: str = 'LAST',
        direction: str = 'ASC',
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Rank runs by a metric aggregation.

        Args:
            project: Project name.
            metric_name: Metric to rank by.
            aggregation: One of ``MIN``, ``MAX``, ``AVG``, ``LAST``,
                ``VARIANCE``.
            direction: ``ASC`` or ``DESC``.
            limit: Max results (max 100).
            offset: Pagination offset.

        Returns:
            List of run dicts with metric values.
        """
        params: Dict[str, Any] = {
            'projectName': project,
            'logName': metric_name,
            'aggregation': aggregation,
            'direction': direction,
            'limit': min(limit, 100),
            'offset': offset,
        }
        return self._get('/api/runs/leaderboard', params=params)

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def get_files(
        self,
        project: str,
        run_id: int,
        file_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get file metadata with presigned download URLs.

        Args:
            project: Project name.
            run_id: Numeric server ID.
            file_name: Filter by log name / file name.

        Returns:
            List of file dicts with keys: ``fileName``, ``fileType``,
            ``fileSize``, ``step``, ``time``, ``downloadUrl``.
        """
        params: Dict[str, Any] = {'runId': run_id, 'projectName': project}
        if file_name is not None:
            params['logName'] = file_name
        return self._get('/api/runs/files', params=params)

    def download_file(
        self,
        project: str,
        run_id: int,
        file_name: str,
        destination: Union[str, Path] = '.',
    ) -> Path:
        """Download a file artifact to local disk.

        Args:
            project: Project name.
            run_id: Numeric server ID.
            file_name: Log name of the file to download.
            destination: Directory or file path.

        Returns:
            Path to the downloaded file.

        Raises:
            QueryError: If no matching file is found.
        """
        files = self.get_files(project, run_id, file_name=file_name)
        if not files:
            raise QueryError(f'No file found matching "{file_name}" for run {run_id}')

        file_info = files[0]
        url = file_info.get('downloadUrl') or file_info.get('url')
        if not url:
            raise QueryError(f'No download URL for file "{file_name}"')

        dest = Path(destination)
        if dest.is_dir():
            dest = dest / file_info.get('fileName', file_name)

        resp = httpx.get(url, follow_redirects=True, timeout=120)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def get_logs(
        self,
        project: str,
        run_id: int,
        log_type: Optional[str] = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch console logs for a run.

        Args:
            project: Project name.
            run_id: Numeric server ID.
            log_type: Filter by type: ``"info"``, ``"error"``, ``"warning"``,
                ``"debug"``, ``"print"``.
            limit: Max lines (max 10 000).
            offset: Pagination offset.

        Returns:
            List of log dicts with keys: ``message``, ``logType``, ``time``,
            ``lineNumber``, ``step``.
        """
        params: Dict[str, Any] = {
            'runId': run_id,
            'projectName': project,
            'limit': min(limit, 10000),
            'offset': offset,
        }
        if log_type is not None:
            params['logType'] = log_type
        return self._get('/api/runs/logs', params=params)

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        retry: int = 0,
    ) -> Any:
        """Issue a GET request with retry logic."""
        url = f'{self._url_api}{path}'

        try:
            resp = self._client.get(url, params=params)
        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
        ) as exc:
            return self._retry_or_raise(
                path, params, retry, f'{type(exc).__name__}: {exc}'
            )

        if resp.status_code in (200, 201):
            return resp.json()

        # Retryable server errors
        if resp.status_code >= 500 and retry < _RETRY_MAX:
            return self._retry_or_raise(
                path, params, retry, f'HTTP {resp.status_code}: {resp.text[:200]}'
            )

        # Client errors — don't retry
        raise QueryError(
            f'{url} returned {resp.status_code}: {resp.text[:500]}',
            status_code=resp.status_code,
        )

    def _retry_or_raise(
        self,
        path: str,
        params: Optional[Dict[str, Any]],
        retry: int,
        error_info: str,
    ) -> Any:
        if retry >= _RETRY_MAX:
            raise QueryError(f'Failed after {retry} retries: {error_info}')
        wait = min(_RETRY_WAIT_MIN * (2 ** (retry + 1)), _RETRY_WAIT_MAX)
        logger.debug(
            '%s: retry %d/%d for %s: %s',
            tag,
            retry + 1,
            _RETRY_MAX,
            path,
            error_info,
        )
        time.sleep(wait)
        return self._get(path, params=params, retry=retry + 1)


# ======================================================================
# Module-level convenience functions
# ======================================================================

_default_client: Optional[Client] = None


def _get_client() -> Client:
    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


def list_projects() -> List[Dict[str, Any]]:
    """List all projects. See :meth:`Client.list_projects`."""
    return _get_client().list_projects()


def list_runs(
    project: str,
    search: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List runs in a project. See :meth:`Client.list_runs`."""
    return _get_client().list_runs(project, search=search, tags=tags, limit=limit)


def get_run(project: str, run_id: Union[int, str]) -> Dict[str, Any]:
    """Get run details. See :meth:`Client.get_run`."""
    return _get_client().get_run(project, run_id)


def get_metric_names(
    project: str,
    run_ids: Optional[List[int]] = None,
    search: Optional[str] = None,
    limit: int = 500,
) -> List[str]:
    """List metric names. See :meth:`Client.get_metric_names`."""
    return _get_client().get_metric_names(
        project,
        run_ids=run_ids,
        search=search,
        limit=limit,
    )


def get_metrics(
    project: str,
    run_id: int,
    metric_names: Optional[List[str]] = None,
    limit: int = 10000,
) -> Any:
    """Fetch metric data. See :meth:`Client.get_metrics`."""
    return _get_client().get_metrics(
        project,
        run_id,
        metric_names=metric_names,
        limit=limit,
    )


def get_statistics(
    project: str,
    run_id: int,
    metric_names: Optional[List[str]] = None,
) -> Any:
    """Compute metric statistics. See :meth:`Client.get_statistics`."""
    return _get_client().get_statistics(project, run_id, metric_names=metric_names)


def compare_runs(
    project: str,
    run_ids: List[int],
    metric_name: str,
) -> Dict[str, Any]:
    """Compare runs by metric. See :meth:`Client.compare_runs`."""
    return _get_client().compare_runs(project, run_ids, metric_name)


def leaderboard(
    project: str,
    metric_name: str,
    aggregation: str = 'LAST',
    direction: str = 'ASC',
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Rank runs by metric. See :meth:`Client.leaderboard`."""
    return _get_client().leaderboard(
        project,
        metric_name,
        aggregation=aggregation,
        direction=direction,
        limit=limit,
        offset=offset,
    )


def get_files(
    project: str,
    run_id: int,
    file_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get file metadata. See :meth:`Client.get_files`."""
    return _get_client().get_files(project, run_id, file_name=file_name)


def download_file(
    project: str,
    run_id: int,
    file_name: str,
    destination: Union[str, Path] = '.',
) -> Path:
    """Download a file. See :meth:`Client.download_file`."""
    return _get_client().download_file(project, run_id, file_name, destination)


def get_logs(
    project: str,
    run_id: int,
    log_type: Optional[str] = None,
    limit: int = 10000,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch console logs. See :meth:`Client.get_logs`."""
    return _get_client().get_logs(
        project,
        run_id,
        log_type=log_type,
        limit=limit,
        offset=offset,
    )


# ======================================================================
# Helpers
# ======================================================================


def _resolve_api_token() -> Optional[str]:
    """Resolve API token from environment or keyring."""
    token = os.environ.get('PLUTO_API_TOKEN')
    if token:
        return token

    # Deprecated env var
    token = os.environ.get('MLOP_API_TOKEN')
    if token:
        warnings.warn(
            'MLOP_API_TOKEN is deprecated. Use PLUTO_API_TOKEN instead.',
            DeprecationWarning,
            stacklevel=3,
        )
        return token

    # Try keyring (same logic as auth.py login)
    try:
        import keyring

        try:
            assert __import__('sys').platform == 'darwin'
            token = keyring.get_password('pluto', 'pluto')
        except (keyring.errors.NoKeyringError, AssertionError):
            from keyrings.alt.file import PlaintextKeyring

            keyring.set_keyring(PlaintextKeyring())
            token = keyring.get_password('pluto', 'pluto')
        return token if token else None
    except Exception:
        return None


def _resolve_url_api(host: Optional[str] = None) -> str:
    """Resolve the API base URL."""
    if host is not None:
        # If it looks like a full URL, strip to derive url_api
        if host.startswith('http://') or host.startswith('https://'):
            # e.g. "https://pluto.trainy.ai" → "https://pluto-api.trainy.ai"
            # or   "https://pluto-api.trainy.ai" → keep as-is
            if '/api/' in host or host.rstrip('/').endswith((':3001',)):
                return host.rstrip('/')
            # Best guess: user passed the app URL. We can't reliably derive
            # the API URL, so use it as-is and let the server redirect.
            return host.rstrip('/')
        else:
            # Bare host like "10.0.0.1" or "my-host:3001" — same as Settings.update_host
            return f'http://{host}:3001'

    # Environment variable
    url = os.environ.get('PLUTO_URL_API')
    if url:
        return url.rstrip('/')

    url = os.environ.get('MLOP_URL_API')
    if url:
        warnings.warn(
            'MLOP_URL_API is deprecated. Use PLUTO_URL_API instead.',
            DeprecationWarning,
            stacklevel=3,
        )
        return url.rstrip('/')

    return _DEFAULT_URL_API


def _to_dataframe(data: Any) -> Any:
    """Convert raw metric data to a pandas DataFrame if pandas is available."""
    if not data:
        try:
            import pandas as pd

            return pd.DataFrame(columns=['metric', 'step', 'value', 'time'])
        except ImportError:
            return []

    try:
        import pandas as pd

        return pd.DataFrame(data)
    except ImportError:
        return data
