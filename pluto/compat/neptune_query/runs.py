"""Neptune-query runs compatibility shim.

Drop-in replacements for ``neptune_query.runs`` functions that read from
the Pluto REST API instead of Neptune.

Usage::

    from pluto.compat.neptune_query import runs as nq_runs
    from pluto.compat.neptune_query.filters import AttributeFilter, Filter

    run_ids = nq_runs.list_runs(project="tfc/MQTransformer", runs=filters)
    df = nq_runs.fetch_runs_table(project="tfc/MQTransformer", runs=run_ids,
                                   attributes="a | b")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pluto.query import Client

if TYPE_CHECKING:
    import pandas as pd

from .filters import AttributeFilter, Filter, _resolve_attribute

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'NeptuneQueryCompat'

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_client: Optional[Client] = None
_run_cache: Dict[str, Dict[str, Any]] = {}


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client()
    return _client


def _normalize_project(project: str) -> str:
    """Strip Neptune workspace prefix (``workspace/project`` → ``project``)."""
    if '/' in project:
        return project.split('/', 1)[1]
    return project


def _get_run_cached(project: str, display_id: str) -> Dict[str, Any]:
    """Fetch a run, using an in-session cache to avoid redundant API calls."""
    key = f'{project}/{display_id}'
    if key not in _run_cache:
        _run_cache[key] = _get_client().get_run(project, display_id)
    return _run_cache[key]


def _parse_attributes(attributes: str) -> List[str]:
    """Split a pipe-separated attribute string into a list of trimmed names."""
    return [a.strip() for a in attributes.split('|') if a.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_runs(
    project: str,
    runs: Optional[Filter] = None,
) -> List[str]:
    """List run display IDs matching *runs* filter.

    Args:
        project: Neptune-style project (``workspace/project`` or just ``project``).
        runs: A :class:`Filter` instance. If ``None``, all runs are returned.

    Returns:
        List of display-ID strings (e.g. ``["MQT-42"]``).
    """
    global _last_project
    project = _normalize_project(project)
    _last_project = project
    client = _get_client()

    # Fast path: if filter has a sys/id match, look up directly
    if runs is not None:
        direct_id = runs.get_match_value('sys/id')
        if direct_id is not None:
            try:
                run = _get_run_cached(project, direct_id)
            except Exception:
                logger.debug('%s: direct lookup for %s failed', tag, direct_id)
                return []
            if runs.evaluate(run):
                return [run['displayId']]
            return []

    # General path: list + client-side filter
    all_runs = client.list_runs(project, limit=200)
    result: List[str] = []
    for r in all_runs:
        if runs is None or runs.evaluate(r):
            result.append(r['displayId'])
    return result


def fetch_runs_table(
    project: str,
    runs: Union[List[str], str],
    attributes: str,
    sort_by: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch run attributes as a DataFrame.

    Args:
        project: Neptune-style project (``workspace/project`` or just ``project``).
        runs: Single display-ID string or list of display-ID strings.
        attributes: Pipe-separated attribute names (e.g. ``"a | b | c"``).
        sort_by: Optional attribute name to sort rows by.

    Returns:
        DataFrame indexed by ``"run"`` with one column per attribute.
    """
    import pandas as pd

    global _last_project
    project = _normalize_project(project)
    _last_project = project
    if isinstance(runs, str):
        runs = [runs]
    attr_names = _parse_attributes(attributes)

    rows: List[Dict[str, Any]] = []
    for display_id in runs:
        run = _get_run_cached(project, display_id)
        row: Dict[str, Any] = {'run': display_id}
        for attr in attr_names:
            row[attr] = _resolve_attribute(run, attr)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        # Ensure columns exist even when no rows
        df = pd.DataFrame(columns=['run'] + attr_names)
    df = df.set_index('run')

    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by)

    return df


def fetch_metrics(
    project: str,
    runs: Union[List[str], str],
    attributes: Optional[AttributeFilter] = None,
) -> pd.DataFrame:
    """Fetch time-series metrics for a run, filtered by :class:`AttributeFilter`.

    Args:
        project: Neptune-style project (``workspace/project`` or just ``project``).
        runs: Single display-ID string or list (only first element used).
        attributes: An :class:`AttributeFilter` to select metric names.

    Returns:
        DataFrame with ``step`` as index and one column per matching metric.
    """
    import pandas as pd

    global _last_project
    project = _normalize_project(project)
    _last_project = project
    if isinstance(runs, list):
        display_id = runs[0]
    else:
        display_id = runs

    client = _get_client()
    run = _get_run_cached(project, display_id)
    numeric_id: int = run['id']

    # Get all metric names for this run, then filter
    all_names = client.get_metric_names(project, run_ids=[numeric_id])
    if attributes is not None:
        matching = [n for n in all_names if attributes.matches_name(n)]
    else:
        matching = all_names

    if not matching:
        return pd.DataFrame()

    # Fetch long-format metrics and pivot to wide
    long_df = client.get_metrics(project, numeric_id, metric_names=matching)
    if isinstance(long_df, list):
        long_df = pd.DataFrame(long_df)
    if long_df.empty:
        return pd.DataFrame()

    wide = long_df.pivot_table(index='step', columns='metric', values='value')
    wide.columns.name = None  # remove "metric" label from columns axis
    return wide


def download_files(
    files: pd.DataFrame,
    destination: Union[str, Path],
) -> pd.DataFrame:
    """Download file artifacts referenced in *files* DataFrame.

    Args:
        files: DataFrame from :func:`fetch_runs_table` where column values
            are file attribute names (the attribute must correspond to a
            file uploaded to the run).
        destination: Local directory to download files into.

    Returns:
        DataFrame with same shape, cell values replaced by local file paths.
    """
    destination = Path(destination)
    client = _get_client()
    result = files.astype(object).copy()

    for display_id in files.index:
        project = _normalize_project(_infer_project())
        run = _get_run_cached(project, display_id)
        numeric_id: int = run['id']

        for col in files.columns:
            file_attr = col  # column name is the attribute / file name
            run_dest = destination / display_id
            try:
                path = client.download_file(
                    project, numeric_id, file_attr, destination=run_dest
                )
                result.at[display_id, col] = str(path)
            except Exception:
                logger.debug(
                    '%s: failed to download %s for run %s',
                    tag,
                    file_attr,
                    display_id,
                )

    return result


# ---------------------------------------------------------------------------
# Project inference for download_files
# ---------------------------------------------------------------------------

_last_project: Optional[str] = None


def _infer_project() -> str:
    """Return the most recently used project name.

    :func:`download_files` doesn't receive a ``project`` argument in the
    Neptune API, so we remember the last project used by other functions.
    """
    if _last_project is not None:
        return _last_project
    raise ValueError(
        'Cannot infer project for download_files. '
        'Call list_runs or fetch_runs_table first.'
    )
