"""
Read/query API for Pluto runs.

Provides programmatic access to runs, metrics, files, and logs stored in Pluto.
Wraps the Pluto server REST API (``/api/runs/*``) with API-key authentication.

Usage::

    import pluto.query as pq

    # Uses PLUTO_API_KEY env var and default server
    runs = pq.list_runs("my-project")
    run = pq.get_run("my-project", "MMP-42")
    metrics = pq.get_metrics("my-project", run["id"], metric_names=["val/loss"])

    # Explicit auth / custom server
    client = pq.Client(api_token="plt_...", host="https://pluto.example.com")
    runs = client.list_runs("my-project")
"""

import json
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


# wandb-compatible vocabulary for the ``filters`` query (the ``/api/runs/list``
# ``filter`` param). The server defines this grammar canonically in
# ``lib/queries/run-filter-grammar.ts`` and publishes it as the
# ``RunFilterGrammar`` OpenAPI component; these constants are the client mirror,
# kept honest by ``tests/test_contract.py`` (drift fails CI). The server is the
# authority on detailed semantics; the client does a light structural check so
# obvious mistakes fail fast with a clear ``ValueError`` instead of an HTTP 400.
_FILTER_BOOL_OPS = {'$and', '$or', '$not'}
_FILTER_LEAF_OPS = {
    '$eq',
    '$ne',
    '$gt',
    '$gte',
    '$lt',
    '$lte',
    '$in',
    '$nin',
    '$regex',
}
# Recognised leaf field names (exact) plus dotted-prefix families.
_FILTER_FIELDS = {
    'state',
    'status',
    'heartbeat_at',
    'heartbeatAt',
    'created_at',
    'createdAt',
    'updated_at',
    'updatedAt',
    'name',
    'displayName',
    'display_name',
    'tags',
}
_FILTER_FIELD_PREFIXES = (
    'config.',
    'systemMetadata.',
    'summaryMetrics.',
    'summary_metrics.',
)

# Server clamps offset to this range (MAX_JSON_SORT_OFFSET).
_MAX_OFFSET = 100_000


def _validate_filters(node: Any, _depth: int = 0) -> None:
    """Light structural validation of a wandb-style ``filters`` dict.

    Recursively checks boolean operators (``$and``/``$or``/``$not``), leaf
    operators, and field names against the supported vocabulary, raising
    ``ValueError`` on anything unrecognised. The server enforces full semantics.
    """
    if _depth > 50:
        raise ValueError('filters nested too deeply')
    if not isinstance(node, dict):
        raise ValueError(f'filters node must be a dict, got {type(node).__name__}')
    for key, value in node.items():
        if key in ('$and', '$or'):
            if not isinstance(value, list):
                raise ValueError(f'{key} expects a list')
            for child in value:
                _validate_filters(child, _depth + 1)
        elif key == '$not':
            _validate_filters(value, _depth + 1)
        elif key.startswith('$'):
            raise ValueError(f'unknown boolean operator: {key!r}')
        else:
            _validate_filter_field(key)
            if isinstance(value, dict) and any(k.startswith('$') for k in value):
                for op in value:
                    if op not in _FILTER_LEAF_OPS:
                        raise ValueError(
                            f'unknown leaf operator {op!r} on field {key!r}; '
                            f'expected one of {sorted(_FILTER_LEAF_OPS)}'
                        )


def _validate_filter_field(field_name: str) -> None:
    if field_name in _FILTER_FIELDS:
        return
    if any(
        field_name.startswith(p) and len(field_name) > len(p)
        for p in _FILTER_FIELD_PREFIXES
    ):
        return
    raise ValueError(f'unknown filter field: {field_name!r}')


class Client:
    """HTTP client for reading data from the Pluto server.

    Args:
        api_token: API token for authentication. Defaults to ``PLUTO_API_KEY``
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
                'No API token provided. Set PLUTO_API_KEY environment variable '
                'or pass api_token to Client().'
            )

        self._url_api = _resolve_url_api(host)
        self._client = httpx.Client(
            http2=True,
            headers={
                'Authorization': f'Bearer {self._api_token}',
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
        return self._get('/api/runs/projects')['projects']

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def list_runs(
        self,
        project: str,
        search: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        sort: Optional[str] = None,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """List runs in a project.

        Args:
            project: Project name.
            search: Full-text search on run name.
            tags: Filter by tags (AND logic). Only runs matching *all*
                specified tags are returned.
            limit: Maximum number of runs to return (max 200).
            sort: Server-side ordering as ``[+|-]<field>`` (``-`` = descending).
                Accepts built-in columns (``createdAt``, ``updatedAt``,
                ``name``, ``status`` and their snake_case aliases),
                ``config.<key>.value`` / ``systemMetadata.<key>.value``,
                ``summary_metrics.<name>`` (LAST aggregation), and
                ``heartbeat_at`` (last logged data point). Defaults to
                ``createdAt desc`` server-side.
            offset: Skip-based pagination offset (0–100,000). Combine with
                ``limit`` to page; advance ``offset`` until a short page is
                returned.
            filters: A wandb-compatible MongoDB-style query (the OR-capable
                filter surface). Boolean ``$and``/``$or``/``$not`` combine leaf
                terms; each leaf is ``{field: value}`` (equality) or
                ``{field: {$op: value}}`` with ops ``$eq``, ``$ne``, ``$gt``,
                ``$gte``, ``$lt``, ``$lte``, ``$in``, ``$nin``, ``$regex``.
                Fields: ``state``/``status``, ``heartbeat_at``,
                ``created_at``/``updated_at``, ``name``, ``tags``,
                ``config.<key>``, ``summaryMetrics.<key>``.
                ``$or``/``$not`` and ``heartbeat_at``/``summaryMetrics.*`` leaves
                require ``project``.

        ``filters`` AND-combines with ``search``/``tags``. To find interrupted
        spot jobs to retry, find the *alive* set and exclude it — or directly
        select non-completed runs that have gone stale::

            # wandb-style "alive" set (running OR reported data in the last hour):
            list_runs(project, filters={"$or": [
                {"state": "running"},
                {"heartbeat_at": {"$gte": cutoff_iso}},
            ]})

            # retry candidates directly (not completed AND stale):
            list_runs(project, filters={"$and": [
                {"status": {"$ne": "COMPLETED"}},
                {"heartbeat_at": {"$lt": cutoff_iso}},
            ]})

        Returns:
            List of run dicts with keys: ``id``, ``name``, ``displayId``,
            ``status``, ``tags``, ``config``, ``createdAt``, ``updatedAt``,
            ``url``.

        Raises:
            ValueError: If ``filters`` uses an unrecognised operator or field.
        """
        params: Dict[str, Any] = {'projectName': project, 'limit': min(limit, 200)}
        if search is not None:
            params['search'] = search
        if tags is not None:
            params['tags'] = ','.join(tags)
        if filters is not None:
            _validate_filters(filters)
            params['filter'] = json.dumps(filters)
        if sort is not None:
            params['sort'] = sort
        if offset:
            clamped = max(0, min(offset, _MAX_OFFSET))
            if clamped != offset:
                logger.debug(
                    '%s: offset %d clamped to %d (max %d)',
                    tag,
                    offset,
                    clamped,
                    _MAX_OFFSET,
                )
            # A negative offset clamps to 0; don't send a redundant offset=0.
            if clamped:
                params['offset'] = clamped
        return self._get('/api/runs/list', params=params)['runs']

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
        return self._get('/api/runs/metric-names', params=params)['metricNames']

    def get_metrics(
        self,
        project: str,
        run_id: Union[int, str],
        metric_names: Optional[List[str]] = None,
        limit: int = 10000,
        step_min: Optional[int] = None,
        step_max: Optional[int] = None,
    ) -> Any:
        """Fetch time-series metric data for a run.

        The server returns up to ``limit`` data points per metric, sampled
        via reservoir sampling when the full series exceeds the limit.

        When *pandas* is installed the return value is a
        :class:`~pandas.DataFrame` with columns ``metric``, ``step``,
        ``value``, ``time``. Otherwise a list of dicts is returned.

        ``step_min`` / ``step_max`` scope the query to a step window instead
        of pulling the whole series — useful for zooming in around a spike or
        anomaly (e.g. ``step_min=4000, step_max=4500``). Because the server
        samples down to ``limit`` points, narrowing the range also gives you
        finer resolution within it.

        Args:
            project: Project name.
            run_id: Numeric server ID (``int``) or display ID string
                (e.g. ``"MMP-1"``).
            metric_names: Metric names to fetch. ``None`` fetches all.
            limit: Max data points per metric (max 10 000).
            step_min: Lowest step to return, **inclusive**. ``None`` for no
                lower bound.
            step_max: Highest step to return, **inclusive**. ``None`` for no
                upper bound.

        Returns:
            ``pandas.DataFrame`` or ``list[dict]``.
        """
        params: Dict[str, Any] = {
            'runId': self._resolve_run_id(project, run_id),
            'projectName': project,
            'limit': min(limit, 10000),
        }
        # Set on the base params (not per-branch) so the multi-metric path
        # below — which copies these into one request per metric — carries the
        # bounds too. Omit entirely when unset rather than sending empties.
        if step_min is not None:
            params['stepMin'] = step_min
        if step_max is not None:
            params['stepMax'] = step_max

        if metric_names is not None and len(metric_names) > 1:
            # Endpoint only supports a single logName filter, so fetch
            # each metric individually and merge the results.
            raw: list = []
            for name in metric_names:
                p = dict(params)
                p['logName'] = name
                raw.extend(self._get('/api/runs/metrics', params=p)['metrics'])
        else:
            if metric_names is not None and len(metric_names) == 1:
                params['logName'] = metric_names[0]
            raw = self._get('/api/runs/metrics', params=params)['metrics']

        # Server returns 'logName' per point; downstream expects 'metric'.
        for row in raw:
            if 'logName' in row and 'metric' not in row:
                row['metric'] = row.pop('logName')

        return _to_dataframe(raw)

    # ------------------------------------------------------------------
    # Statistics / comparison
    # ------------------------------------------------------------------

    def get_statistics(
        self,
        project: str,
        run_id: Union[int, str],
        metric_names: Optional[List[str]] = None,
    ) -> Any:
        """Compute statistics for run metrics.

        Returns per-metric aggregations: ``count``, ``min``, ``max``,
        ``mean``, ``stddev``, as well as anomaly detection data.

        Args:
            project: Project name.
            run_id: Numeric server ID (``int``) or display ID string
                (e.g. ``"MMP-1"``).
            metric_names: Restrict to these metrics.

        Returns:
            Server response dict.
        """
        params: Dict[str, Any] = {
            'runId': self._resolve_run_id(project, run_id),
            'projectName': project,
        }
        if metric_names is not None and len(metric_names) == 1:
            params['logName'] = metric_names[0]
        return self._get('/api/runs/statistics', params=params)

    def compare_runs(
        self,
        project: str,
        run_ids: List[Union[int, str]],
        metric_name: str,
    ) -> Dict[str, Any]:
        """Compare a metric across multiple runs.

        Args:
            project: Project name.
            run_ids: List of run IDs (max 100). Each entry may be a numeric
                server ID or a display ID string (e.g. ``"MMP-1"``).
            metric_name: The metric to compare.

        Returns:
            Dict with per-run statistics and a ``bestRun`` recommendation.
        """
        cache: Dict[Union[int, str], int] = {}
        resolved: List[int] = []
        for r in run_ids[:100]:
            if r not in cache:
                cache[r] = self._resolve_run_id(project, r)
            resolved.append(cache[r])
        params: Dict[str, Any] = {
            'runIds': ','.join(str(r) for r in resolved),
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
        run_id: Union[int, str],
        file_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get file metadata with presigned download URLs.

        Args:
            project: Project name.
            run_id: Numeric server ID (``int``) or display ID string
                (e.g. ``"MMP-1"``).
            file_name: Filter by log name / file name.

        Returns:
            List of file dicts with keys: ``fileName``, ``fileType``,
            ``fileSize``, ``step``, ``time``, ``downloadUrl``.
        """
        params: Dict[str, Any] = {
            'runId': self._resolve_run_id(project, run_id),
            'projectName': project,
        }
        if file_name is not None:
            params['logName'] = file_name
        return self._get('/api/runs/files', params=params)['files']

    def download_file(
        self,
        project: str,
        run_id: Union[int, str],
        file_name: str,
        destination: Union[str, Path] = '.',
    ) -> Path:
        """Download a file artifact to local disk.

        Args:
            project: Project name.
            run_id: Numeric server ID (``int``) or display ID string
                (e.g. ``"MMP-1"``).
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
        run_id: Union[int, str],
        log_type: Optional[str] = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch console logs for a run.

        Args:
            project: Project name.
            run_id: Numeric server ID (``int``) or display ID string
                (e.g. ``"MMP-1"``).
            log_type: Filter by type: ``"info"``, ``"error"``, ``"warning"``,
                ``"debug"``, ``"print"``.
            limit: Max lines (max 10 000).
            offset: Pagination offset.

        Returns:
            List of log dicts with keys: ``message``, ``logType``, ``time``,
            ``lineNumber``, ``step``.
        """
        params: Dict[str, Any] = {
            'runId': self._resolve_run_id(project, run_id),
            'projectName': project,
            'limit': min(limit, 10000),
            'offset': offset,
        }
        if log_type is not None:
            params['logType'] = log_type
        return self._get('/api/runs/logs', params=params)['logs']

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_run_id(self, project: str, run_id: Union[int, str]) -> int:
        """Resolve a run ID argument to a numeric server ID.

        Accepts an ``int`` (returned as-is) or a display ID string like
        ``"MMP-1"`` (looked up via :meth:`get_run`). Numeric strings are
        treated as server IDs.
        """
        if isinstance(run_id, int):
            return run_id
        if isinstance(run_id, str) and run_id.isdigit():
            return int(run_id)
        return int(self.get_run(project, run_id)['id'])

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
    sort: Optional[str] = None,
    offset: int = 0,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """List runs in a project. See :meth:`Client.list_runs`."""
    return _get_client().list_runs(
        project,
        search=search,
        tags=tags,
        limit=limit,
        sort=sort,
        offset=offset,
        filters=filters,
    )


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
    run_id: Union[int, str],
    metric_names: Optional[List[str]] = None,
    limit: int = 10000,
    step_min: Optional[int] = None,
    step_max: Optional[int] = None,
) -> Any:
    """Fetch metric data. See :meth:`Client.get_metrics`.

    ``step_min`` / ``step_max`` are **inclusive** bounds that scope the query
    to a step window (e.g. ``step_min=4000, step_max=4500``) instead of
    pulling the full series.
    """
    return _get_client().get_metrics(
        project,
        run_id,
        metric_names=metric_names,
        limit=limit,
        step_min=step_min,
        step_max=step_max,
    )


def get_statistics(
    project: str,
    run_id: Union[int, str],
    metric_names: Optional[List[str]] = None,
) -> Any:
    """Compute metric statistics. See :meth:`Client.get_statistics`."""
    return _get_client().get_statistics(project, run_id, metric_names=metric_names)


def compare_runs(
    project: str,
    run_ids: List[Union[int, str]],
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
    run_id: Union[int, str],
    file_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get file metadata. See :meth:`Client.get_files`."""
    return _get_client().get_files(project, run_id, file_name=file_name)


def download_file(
    project: str,
    run_id: Union[int, str],
    file_name: str,
    destination: Union[str, Path] = '.',
) -> Path:
    """Download a file. See :meth:`Client.download_file`."""
    return _get_client().download_file(project, run_id, file_name, destination)


def get_logs(
    project: str,
    run_id: Union[int, str],
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
    token = os.environ.get('PLUTO_API_KEY')
    if token:
        return token

    # Deprecated env var
    token = os.environ.get('MLOP_API_TOKEN')
    if token:
        warnings.warn(
            'MLOP_API_TOKEN is deprecated. Use PLUTO_API_KEY instead.',
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
