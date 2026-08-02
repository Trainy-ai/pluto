"""
Export wandb cloud runs to the on-disk pluto.migrate staging format.

Reads complete run data through the wandb public API (``wandb.Api()``)
and stages each run as ``run.json`` + parquet parts + downloaded files
under ``output_dir/{entity}/{project}/runs/{run_id}/``. Runs are staged
in a ``.tmp`` directory renamed into place only after the export
sentinel is written, so an interrupted export never leaves a directory
that looks complete; re-running skips finished runs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict, Union

import yaml

from pluto.migrate.schema import PartWriter
from pluto.migrate.state import is_run_exported, mark_run_exported, write_json_atomic

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'migrate'

MANIFEST_FILENAME = 'manifest.json'

# wandb encodes non-finite metric points as JSON strings; map them back to the
# real floats so they migrate instead of being dropped as "text".
_NONFINITE_STRINGS = {
    'NaN': float('nan'),
    'nan': float('nan'),
    'Infinity': float('inf'),
    'inf': float('inf'),
    'Inf': float('inf'),
    '-Infinity': float('-inf'),
    '-inf': float('-inf'),
    '-Inf': float('-inf'),
}

# Non-numeric string history points (status labels, phase names, etc.) migrate
# as a "string-series" (rendered as a categorical/state-timeline in Pluto).
# Guard pathological values: a single point longer than this is almost never a
# real categorical state (it's a stray log line / serialized blob), so drop it
# rather than pollute the timeline. The per-key cardinality guard lives in the
# loader (it needs the whole series to count distinct values).
_STRING_SERIES_MAX_LEN = 200


class _RowBase(TypedDict):
    """Identity columns shared by every staged row (see schema.write_row)."""

    project_id: str
    run_id: str


# wandb built-in custom-chart presets (panelDefId) -> a stable short name.
# Each corresponds to a Vega-Lite template the Pluto side renders; the backing
# table (tableKey) migrates as a Table and supplies the chart's data. Presets
# outside this set (user-authored Vega) are staged but flagged: the server has
# no template to rebuild them from.
_WANDB_CHART_PRESETS = {
    'wandb/bar/v0': 'bar',
    'wandb/line/v0': 'line',
    'wandb/lineseries/v0': 'lineseries',
    'wandb/scatter/v0': 'scatter',
    'wandb/pr_curve/v0': 'pr_curve',
    'wandb/roc_curve/v0': 'roc_curve',
    # area-under-curve is the SDK's shared preset behind pr_curve + roc_curve.
    'wandb/area-under-curve/v0': 'area-under-curve',
    'wandb/confusion_matrix/v0': 'confusion_matrix',
    'wandb/confusion_matrix/v1': 'confusion_matrix',
    'wandb/histogram/v0': 'histogram',
}

# scan_history dict values whose media file lives under the run's files/
_FILE_MEDIA_TYPES = {
    'image-file',
    'audio-file',
    'video-file',
    'table-file',
    'plotly-file',
    'object3D-file',
    'html-file',
}

# Leading "<iso timestamp> <message>" console lines (wandb writes these
# when x_show_timestamps is enabled).
_CONSOLE_TS_RE = re.compile(
    r'^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?'
    r'(?:Z|[+-]\d{2}:?\d{2})?)\s+(.*)$'
)


def parse_iso_ms(value: Optional[str]) -> Optional[int]:
    """Parse an ISO-8601 string to epoch milliseconds (UTC assumed if naive)."""
    if not value:
        return None
    try:
        normalized = value.replace(',', '.').replace('Z', '+00:00')
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        return None


class WandbExporter:
    """Stage one wandb project's runs on disk for later loading into Pluto."""

    def __init__(
        self,
        entity: str,
        project: str,
        output_dir: Union[str, Path],
        api: Optional[Any] = None,
        api_key: Optional[str] = None,
        run_ids: Optional[List[str]] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        include_artifacts: bool = True,
        artifact_max_bytes: Optional[int] = None,
        include_console: bool = True,
        include_system: bool = True,
        include_files: bool = True,
        history_page_size: int = 1000,
        system_samples: int = 100_000,
        download_workers: int = 16,
    ) -> None:
        self.entity = entity
        self.project = project
        self.output_dir = Path(output_dir)
        self._api = api
        self._api_key = api_key
        self.run_ids = set(run_ids) if run_ids else None
        # Validate up front: an unparseable --after/--before must error, not
        # silently drop the filter and export the entire project.
        self.after_ms = self._parse_filter_date('--after', after)
        self.before_ms = self._parse_filter_date('--before', before)
        self.include_artifacts = include_artifacts
        self.artifact_max_bytes = artifact_max_bytes
        self.include_console = include_console
        self.include_system = include_system
        self.include_files = include_files
        self.history_page_size = history_page_size
        self.system_samples = system_samples
        self.download_workers = max(1, download_workers)
        # Coverage: what got migrated vs. dropped, so nothing is lost silently.
        self._cov_migrated: Counter = Counter()  # per-run (reset in _export_run)
        self._cov_skipped: Counter = Counter()
        self._cov_migrated_total: Counter = Counter()
        self._cov_skipped_total: Counter = Counter()

    @staticmethod
    def _bins_from_packed(packed: Any) -> Optional[List[float]]:
        """Reconstruct histogram bin edges from wandb's compact packedBins
        ({min, size, count}); returns count+1 edges, or None if unusable."""
        if not isinstance(packed, dict):
            return None
        try:
            mn = float(packed['min'])
            sz = float(packed['size'])
            cnt = int(packed['count'])
        except (KeyError, TypeError, ValueError):
            return None
        if cnt <= 0:
            return None
        return [mn + i * sz for i in range(cnt + 1)]

    def _migrated(self, category: str, n: int = 1) -> None:
        self._cov_migrated[category] += n

    def _skipped(self, category: str, n: int = 1) -> None:
        self._cov_skipped[category] += n

    @staticmethod
    def _fmt_coverage(migrated: Counter, skipped: Counter) -> str:
        m = ', '.join(f'{v} {k}' for k, v in sorted(migrated.items())) or 'nothing'
        line = f'migrated: {m}'
        if skipped:
            s = ', '.join(f'{v} {k}' for k, v in sorted(skipped.items()))
            line += f'; NOT migrated: {s}'
        return line

    @staticmethod
    def _parse_filter_date(flag: str, value: Optional[str]) -> Optional[int]:
        """None when the flag was not supplied; raise when it was supplied but
        cannot be parsed (so a typo like ``2024/01/01`` fails loudly instead of
        being silently ignored)."""
        if value is None:
            return None
        ms = parse_iso_ms(value)
        if ms is None:
            raise ValueError(
                f'{flag}: could not parse date {value!r}; use ISO-8601, '
                'e.g. 2024-01-01 or 2024-01-01T00:00:00Z'
            )
        return ms

    @property
    def api(self) -> Any:
        if self._api is None:
            import wandb

            self._api = wandb.Api(api_key=self._api_key)
        return self._api

    @property
    def project_path(self) -> str:
        return f'{self.entity}/{self.project}'

    def _purge_empty_wandb_cache(self) -> int:
        """Delete 0-byte parquet files from wandb's local cache.

        A crash (or OOM) mid-download truncates a run-history parquet to 0
        bytes; wandb then reuses that empty file on every later read and the
        export fails with 'Parquet file too small. Size is 0' — permanently,
        until the file is removed. A valid parquet is never empty, so deleting
        these is always safe: wandb re-downloads the real data on next read.
        Returns the number removed."""
        env = os.environ.get('WANDB_CACHE_DIR')
        cache = Path(env) if env else Path.home() / '.cache' / 'wandb'
        removed: List[str] = []
        try:
            for p in cache.rglob('*.parquet'):
                try:
                    if p.stat().st_size == 0:
                        p.unlink()
                        removed.append(p.name)
                except OSError:
                    pass
        except Exception as e:  # cache dir absent/unreadable — nothing to do
            logger.debug(f'{tag}: wandb cache sweep skipped: {e}')
        if removed:
            # WARNING, not INFO: this is a non-routine recovery event (a prior
            # crash left junk in wandb's cache) the user should see. Name the
            # files so it's clear exactly what was cleared + re-downloaded.
            shown = ', '.join(removed[:10])
            if len(removed) > 10:
                shown += f' (+{len(removed) - 10} more)'
            logger.warning(
                f'{tag}: cleared {len(removed)} empty (crash-truncated) wandb '
                f'cache file(s); wandb will re-download them: {shown}'
            )
        return len(removed)

    def export(self) -> Dict[str, Any]:
        """Export all matching runs. Returns {'exported', 'skipped', 'failed'}."""
        exported, skipped = 0, 0
        failed: List[Dict[str, str]] = []

        # A prior crash can poison wandb's cache with 0-byte parquets that make
        # every history read fail; clear them up front so a crash never blocks
        # a later export.
        self._purge_empty_wandb_cache()
        runs_root = self.output_dir / self.entity / self.project / 'runs'
        for run in self.api.runs(self.project_path):
            if self.run_ids is not None and run.id not in self.run_ids:
                continue
            created_ms = parse_iso_ms(getattr(run, 'created_at', None))
            if created_ms is not None:
                if self.after_ms is not None and created_ms < self.after_ms:
                    continue
                if self.before_ms is not None and created_ms > self.before_ms:
                    continue

            run_dir = runs_root / run.id
            if is_run_exported(run_dir):
                logger.info(f'{tag}: {run.id} already exported, skipping')
                skipped += 1
                continue

            # Retry once: recover from a crash-truncated cache read (purge the
            # empty parquet so wandb re-downloads) or a transient network blip,
            # instead of losing the run. _export_run rebuilds its tmp dir each
            # attempt, so a retry is clean.
            last_err: Optional[Exception] = None
            for attempt in range(2):
                try:
                    self._export_run(run, run_dir)
                    exported += 1
                    logger.info(
                        f'{tag}: exported {run.id} ({run.name})'
                        + (' (on retry)' if attempt else '')
                    )
                    last_err = None
                    break
                except Exception as e:  # keep going: one bad run must not stop all
                    last_err = e
                    if attempt == 0:
                        self._purge_empty_wandb_cache()
                        logger.warning(
                            f'{tag}: export of {run.id} failed ({e}); retrying'
                        )
            if last_err is not None:
                logger.error(f'{tag}: export failed for {run.id}: {last_err}')
                failed.append(
                    {
                        'run_id': run.id,
                        'error': f'{type(last_err).__name__}: {last_err}',
                    }
                )

        coverage = {
            'migrated': dict(self._cov_migrated_total),
            'not_migrated': dict(self._cov_skipped_total),
        }
        total_cov = self._fmt_coverage(
            self._cov_migrated_total, self._cov_skipped_total
        )
        log = logger.warning if self._cov_skipped_total else logger.info
        log(f'{tag}: coverage across {exported} run(s) — {total_cov}')

        summary = {
            'exported': exported,
            'skipped': skipped,
            'failed': failed,
            'coverage': coverage,
        }
        # Per-project path: a shared output_dir/manifest.json would be clobbered
        # (and its .tmp raced) when multiple projects export concurrently.
        manifest_dir = self.output_dir / self.entity / self.project
        manifest_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            manifest_dir / MANIFEST_FILENAME,
            {
                'source': 'wandb',
                'project': self.project_path,
                **summary,
            },
        )
        return summary

    def _export_run(self, run: Any, run_dir: Path) -> None:
        tmp_dir = run_dir.with_name(run_dir.name + '.tmp')
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)  # leftovers from an interrupted export
        tmp_dir.mkdir(parents=True)

        self._cov_migrated = Counter()
        self._cov_skipped = Counter()
        created_ms = parse_iso_ms(getattr(run, 'created_at', None))
        with PartWriter(tmp_dir) as writer:
            self._write_run_json(run, tmp_dir, created_ms)
            self._export_history(run, writer)
            if self.include_system:
                self._export_system_metrics(run, writer)
            files_dir = tmp_dir / 'files'
            if self.include_files or self.include_console:
                self._download_files(run, files_dir)
            if self.include_console:
                self._export_console(run, writer, files_dir, created_ms)
            if self.include_artifacts:
                self._export_artifacts(run, writer, tmp_dir)

        # Custom-chart (wandb.plot.*) panel definitions live in the raw
        # config.yaml (downloaded above), not the staging rows — recover them
        # once the file is on disk.
        self._export_custom_charts(run, tmp_dir)

        # Run-level context that has no home in the staging schema (sweep
        # membership, input-artifact lineage) — flag it so it's not lost silently.
        self._flag_run_level_omissions(run)

        # Coverage: one clear line per run of what was migrated vs. dropped
        # (dropped items also surface at WARNING so they're never silent).
        cov = self._fmt_coverage(self._cov_migrated, self._cov_skipped)
        logger.info(f'{tag}: {run.id} coverage — {cov}')
        if self._cov_skipped:
            dropped = ', '.join(
                f'{v} {k}' for k, v in sorted(self._cov_skipped.items())
            )
            logger.warning(f'{tag}: {run.id} NOT migrated: {dropped}')
        self._cov_migrated_total.update(self._cov_migrated)
        self._cov_skipped_total.update(self._cov_skipped)

        mark_run_exported(tmp_dir, {'rows': writer.rows_written})
        if run_dir.exists():
            shutil.rmtree(run_dir)
        os.rename(tmp_dir, run_dir)

    def _write_run_json(
        self, run: Any, tmp_dir: Path, created_ms: Optional[int]
    ) -> None:
        summary_dict = getattr(getattr(run, 'summary', None), '_json_dict', None) or {}
        summary_dict = {k: v for k, v in summary_dict.items() if not k.startswith('_')}
        updated_ms = parse_iso_ms(getattr(run, 'heartbeat_at', None)) or created_ms
        write_json_atomic(
            tmp_dir / 'run.json',
            {
                'entity': self.entity,
                'project': self.project,
                'run_id': run.id,
                'name': run.name,
                'notes': getattr(run, 'notes', None),
                'tags': list(getattr(run, 'tags', []) or []),
                'state': getattr(run, 'state', None),
                'config': dict(getattr(run, 'config', {}) or {}),
                'summary': summary_dict,
                'createdAt': created_ms,
                'updatedAt': updated_ms,
                'url': getattr(run, 'url', None),
                'metadata': getattr(run, 'metadata', None),
                'sweep': self._sweep_block(run),
            },
        )

    @staticmethod
    def _sweep_block(run: Any) -> Optional[Dict[str, Any]]:
        """Capture the run's wandb sweep — id, name, and search-space config.

        Runs that belong to a sweep carry a Sweep object; migrating it lets Pluto
        group the runs (tag ``sweep:<id>``) and keep the search space, mirroring
        the native ``pluto.sweep`` data model. Best-effort: any API hiccup
        returns None rather than failing the export.
        """
        try:
            sweep = getattr(run, 'sweep', None)
            if sweep is None:
                return None
            sweep_id = getattr(sweep, 'id', None)
            if not sweep_id:
                return None
            block: Dict[str, Any] = {'id': sweep_id}
            name = getattr(sweep, 'name', None)
            if name:
                block['name'] = name
            config = getattr(sweep, 'config', None)
            if isinstance(config, dict) and config:
                block['config'] = config
            return block
        except Exception:
            return None

    @staticmethod
    def _chart_table_key(panel_config: Dict[str, Any]) -> Optional[str]:
        """Pull a custom chart's backing-table key from its userQuery.

        wandb encodes it as a ``summaryTable`` field carrying a ``tableKey``
        arg: ``userQuery.queryFields[].fields[].{name: summaryTable,
        args: [{name: tableKey, value: <key>}]}``. That value is the table's
        log name, which migrates as a Table and supplies the chart data.
        """
        user_query = panel_config.get('userQuery') or {}
        for query_field in user_query.get('queryFields') or []:
            for field in query_field.get('fields') or []:
                if field.get('name') != 'summaryTable':
                    continue
                for arg in field.get('args') or []:
                    if arg.get('name') == 'tableKey':
                        return arg.get('value')
        return None

    def _export_custom_charts(self, run: Any, tmp_dir: Path) -> None:
        """Recover wandb custom-chart (``wandb.plot.*``) panel definitions.

        The panels live in the run's raw config under ``_wandb.value.visualize``
        — the public API strips ``_wandb``, so we read the downloaded
        ``config.yaml`` instead. Each panel binds a built-in Vega preset
        (``panelDefId``) to a backing table via ``fieldSettings`` column
        mappings. We stage a normalized ``custom_charts.json``; the loader
        forwards it so the Pluto side can rebuild the panel from preset +
        migrated table. The raw Vega spec is server-side, not reconstructed here.
        """
        config_path = tmp_dir / 'files' / 'config.yaml'
        if not config_path.exists():
            return  # --no-files/--no-console, or a run without config.yaml
        try:
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            visualize = ((raw.get('_wandb') or {}).get('value') or {}).get(
                'visualize'
            ) or {}
        except Exception as e:
            logger.warning(
                f'{tag}: {run.id} could not parse config.yaml for custom '
                f'charts: {type(e).__name__}: {e}'
            )
            return
        if not isinstance(visualize, dict) or not visualize:
            return

        panels = []
        for key, spec in visualize.items():
            if not isinstance(spec, dict):
                continue
            panel_config = spec.get('panel_config') or {}
            panel_def = panel_config.get('panelDefId')
            preset = (
                _WANDB_CHART_PRESETS.get(panel_def)
                if isinstance(panel_def, str)
                else None
            )
            table_key = self._chart_table_key(panel_config)
            field_settings = panel_config.get('fieldSettings')
            fields = (
                {k: v for k, v in field_settings.items() if v is not None}
                if isinstance(field_settings, dict)
                else {}
            )
            string_settings = panel_config.get('stringSettings')
            title = (
                string_settings.get('title')
                if isinstance(string_settings, dict)
                else None
            )
            panels.append(
                {
                    'key': key,
                    'panelDefId': panel_def,
                    'preset': preset,  # None for user-authored (non-preset) Vega
                    'title': title,
                    # Whole stringSettings dict (title + axis titles, e.g.
                    # x-axis-title/y-axis-title) so the renderer can substitute
                    # ${string:...}; falls back to raw column names if absent.
                    'strings': string_settings
                    if isinstance(string_settings, dict)
                    else {},
                    'tableKey': table_key,  # backing table's log name
                    'fields': fields,
                    'specLang': 'vega-lite' if preset else 'vega',
                }
            )
            if preset and table_key:
                self._migrated('custom-chart')
            else:
                # Unknown preset or unresolved backing table: staged for
                # reference, but the server has no template to rebuild it from.
                self._skipped('custom-chart-unsupported')

        if panels:
            write_json_atomic(tmp_dir / 'custom_charts.json', {'panels': panels})

    def _row_base(self, run: Any) -> '_RowBase':
        return {'project_id': self.project_path, 'run_id': run.id}

    def _export_history(self, run: Any, writer: PartWriter) -> None:
        for row in run.scan_history(page_size=self.history_page_size):
            step = row.get('_step')
            ts = row.get('_timestamp')
            if step is None or ts is None:
                continue
            timestamp_ms = int(float(ts) * 1000)
            for key, value in row.items():
                if key.startswith('_'):
                    continue
                self._export_history_value(
                    run, writer, key, value, int(step), timestamp_ms
                )

    def _export_history_value(
        self,
        run: Any,
        writer: PartWriter,
        key: str,
        value: Any,
        step: int,
        timestamp_ms: int,
    ) -> None:
        base = self._row_base(run)
        if value is None:
            return
        # Booleans (bool is an int subclass) are real metric series in wandb
        # (e.g. is_best=True/False each epoch); record them as 1.0/0.0 rather
        # than silently dropping the whole series.
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)):
            writer.write_row(
                **base,
                attribute_path=key,
                attribute_type='metric',
                step=step,
                timestamp_ms=timestamp_ms,
                float_value=float(value),
            )
            self._migrated('metric')
            return
        if isinstance(value, str):
            # wandb serializes non-finite metric points as strings
            # ('NaN'/'Infinity'/'-Infinity'); coerce them back to real floats so
            # they migrate like any other metric (ClickHouse stores NaN/Inf
            # natively — a run that logged nan/inf numerically already works).
            nonfinite = _NONFINITE_STRINGS.get(value.strip())
            if nonfinite is not None:
                writer.write_row(
                    **base,
                    attribute_path=key,
                    attribute_type='metric',
                    step=step,
                    timestamp_ms=timestamp_ms,
                    float_value=nonfinite,
                )
                self._migrated('metric')
                return
            # Any other string (a status label, phase name, etc.) is a
            # categorical series: stage it as a string_series row so it can be
            # rendered as a state timeline. Over-long values are stray blobs,
            # not real states -> drop with a flag.
            if len(value) > _STRING_SERIES_MAX_LEN:
                self._skipped('string-series-too-long')
                return
            writer.write_row(
                **base,
                attribute_path=key,
                attribute_type='string_series',
                step=step,
                timestamp_ms=timestamp_ms,
                string_value=value,
            )
            self._migrated('string_series')
            return
        if isinstance(value, dict):
            media_type = value.get('_type')
            if media_type in _FILE_MEDIA_TYPES and value.get('path'):
                if not self.include_files:
                    # --no-files: the file won't be downloaded, so don't stage a
                    # row pointing at it (a dangling ref would fail the loader).
                    self._skipped('media-file(--no-files)')
                    return
                # Bounding boxes / segmentation masks ride on the image value as
                # references to sidecar files (….boxes2D.json / ….mask.png). Stage
                # those refs; the loader resolves the box JSON inline and
                # re-uploads the mask PNG (as fileType "mask") into the image's
                # annotations.
                annotation_value = None
                boxes, masks = value.get('boxes'), value.get('masks')
                if boxes or masks:
                    annotation_value = json.dumps(
                        {k: v for k, v in {'boxes': boxes, 'masks': masks}.items() if v}
                    )
                    if boxes:
                        self._migrated('image-boxes')
                    if masks:
                        self._migrated('image-masks')
                writer.write_row(
                    **base,
                    attribute_path=key,
                    attribute_type='media',
                    step=step,
                    timestamp_ms=timestamp_ms,
                    string_value=media_type,
                    file_value=f'files/{value["path"]}',
                    caption=value.get('caption'),
                    annotation_value=annotation_value,
                )
                self._migrated('media')
            elif media_type == 'histogram':
                bins = value.get('bins')
                if bins is None:
                    # wandb stores histogram edges compactly as packedBins
                    # ({min, size, count}), not a `bins` array. Reconstruct the
                    # real edges so the migrated histogram keeps its true value
                    # range instead of a generic 0..N axis. (Loader synthesizes
                    # integer edges only if this is also absent.)
                    bins = self._bins_from_packed(value.get('packedBins'))
                writer.write_row(
                    **base,
                    attribute_path=key,
                    attribute_type='media',
                    step=step,
                    timestamp_ms=timestamp_ms,
                    string_value=json.dumps(
                        {
                            '_type': 'histogram',
                            'values': value.get('values'),
                            'bins': bins,
                        }
                    ),
                )
                self._migrated('histogram')
            elif media_type == 'images/separated' and value.get('filenames'):
                if not self.include_files:
                    self._skipped('media-file(--no-files)', len(value['filenames']))
                    return
                captions = value.get('captions') or []
                for i, filename in enumerate(value['filenames']):
                    writer.write_row(
                        **base,
                        attribute_path=key,
                        attribute_type='media',
                        step=step,
                        timestamp_ms=timestamp_ms,
                        string_value='image-file',
                        file_value=f'files/{filename}',
                        caption=captions[i] if i < len(captions) else None,
                    )
                self._migrated('media', len(value['filenames']))
            elif isinstance(value.get(media_type), list) and all(
                isinstance(it, dict)
                and it.get('path')
                and it.get('_type') in _FILE_MEDIA_TYPES
                for it in value[media_type]
            ):
                # Media LISTS logged at one step: _type 'videos'/'audio' hold
                # their items under a key matching the _type, each a full
                # {path, caption, _type: <x>-file} dict (wandb.log({"k":[v0,v1]})).
                # Emit one media row per item (loader batches same name+step ->
                # sampleIndex order).
                if not self.include_files:
                    self._skipped('media-file(--no-files)', len(value[media_type]))
                    return
                items = value[media_type]
                for it in items:
                    writer.write_row(
                        **base,
                        attribute_path=key,
                        attribute_type='media',
                        step=step,
                        timestamp_ms=timestamp_ms,
                        string_value=it['_type'],
                        file_value=f'files/{it["path"]}',
                        caption=it.get('caption'),
                    )
                self._migrated('media', len(items))
            else:
                # Unknown wandb media type (custom chart, molecule, bokeh, a
                # partitioned/joined table, ...). Count by type so the coverage
                # report says exactly what was dropped.
                self._skipped(f'unsupported({media_type})')
        else:
            self._skipped(f'unsupported({type(value).__name__})')

    def _export_system_metrics(self, run: Any, writer: PartWriter) -> None:
        base = self._row_base(run)
        try:
            # run.history() defaults to samples=500, which silently downsamples
            # long runs (a 2h run at ~2s sampling has ~3600 points). Request a
            # high sample count and warn if we still hit the cap.
            events = run.history(
                stream='events', pandas=False, samples=self.system_samples
            )
        except Exception as e:
            logger.warning(f'{tag}: system metrics unavailable for {run.id}: {e}')
            return
        events = list(events)
        if len(events) >= self.system_samples:
            logger.warning(
                f'{tag}: system metrics for {run.id} hit the '
                f'{self.system_samples}-sample cap and may be downsampled; '
                'raise system_samples to capture full resolution'
            )
        for index, row in enumerate(events):
            ts = row.get('_timestamp')
            if ts is None:
                continue
            timestamp_ms = int(float(ts) * 1000)
            for key, value in row.items():
                if not key.startswith('system.'):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                # Source-native name; the loader translates to Pluto's sys/
                # namespace, so staged exports stay platform-agnostic.
                writer.write_row(
                    **base,
                    attribute_path=key,
                    attribute_type='system_metric',
                    step=index,
                    timestamp_ms=timestamp_ms,
                    float_value=float(value),
                )
                self._migrated('system-metric')

    def _download_files(self, run: Any, files_dir: Path) -> None:
        files_dir.mkdir(parents=True, exist_ok=True)
        targets = [
            f for f in run.files() if self.include_files or f.name == 'output.log'
        ]
        if not targets:
            return

        def _download_one(f: Any) -> Optional[str]:
            # Each wandb file is a separate HTTP request (file API -> storage
            # redirect), so downloads are latency-bound. A media-heavy run has
            # hundreds of tiny files; downloading them concurrently is a large
            # (~10-30x) speedup and is the dominant cost at scale. Distinct
            # filenames -> no write contention. Returns the name on failure.
            try:
                f.download(root=str(files_dir), exist_ok=True)
                return None
            except Exception as e:
                logger.warning(f'{tag}: failed to download {f.name}: {e}')
                return f.name

        workers = min(self.download_workers, len(targets))
        if workers <= 1:
            results = [_download_one(f) for f in targets]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # Results collected on this thread -> the coverage counter below
                # is only touched here, never concurrently.
                results = list(pool.map(_download_one, targets))
        failed = [name for name in results if name is not None]
        if failed:
            # Media rows were already staged (and counted migrated) pointing at
            # these files; a failed download leaves a dangling ref the loader
            # silently skips. Count it so coverage/--strict surface the loss
            # instead of a false "fully migrated".
            self._skipped('file-download-failed', len(failed))

    def _export_console(
        self,
        run: Any,
        writer: PartWriter,
        files_dir: Path,
        created_ms: Optional[int],
    ) -> None:
        output_log = files_dir / 'output.log'
        if not output_log.exists():
            return
        base = self._row_base(run)
        # Fall back to the run's creation time, then its heartbeat, for lines
        # without an embedded timestamp. Only if neither is parseable do we
        # stamp 0 (1970) — and we warn, rather than silently dating logs to the
        # epoch.
        fallback_ms = created_ms
        if fallback_ms is None:
            fallback_ms = parse_iso_ms(getattr(run, 'heartbeat_at', None))
        if fallback_ms is None:
            logger.warning(
                f'{tag}: {run.id} has no parseable creation/heartbeat time; '
                'console lines without embedded timestamps will be stamped '
                '0 (1970-01-01)'
            )
            fallback_ms = 0
        with open(output_log, errors='replace') as f:
            for line_number, raw_line in enumerate(f, start=1):
                message = raw_line.rstrip('\n')
                if not message.strip():
                    continue
                timestamp_ms = self._parse_console_line_time(message, fallback_ms)
                writer.write_row(
                    **base,
                    attribute_path='console',
                    attribute_type='console',
                    step=line_number,
                    timestamp_ms=timestamp_ms,
                    string_value=message,
                )
                self._migrated('console-line')

    @staticmethod
    def _parse_console_line_time(message: str, fallback_ms: int) -> int:
        """Best-effort per-line timestamp; the message itself is never altered
        (a leading ISO prefix may be the user's own logging format)."""
        match = _CONSOLE_TS_RE.match(message)
        if match:
            parsed = parse_iso_ms(match.group(1))
            if parsed is not None:
                return parsed
        return fallback_ms

    def _flag_run_level_omissions(self, run: Any) -> None:
        """Surface run-level context for the coverage report / --strict. Sweep
        membership now migrates (see _sweep_block -> manifest 'sweep'); input-
        artifact (used_artifacts) lineage still doesn't. Both are wrapped: these
        are extra API reads that must never fail an export."""
        try:
            if getattr(run, 'sweep', None) is not None:
                self._migrated('sweep')
        except Exception as e:
            logger.debug(f'{tag}: sweep check failed for {run.id}: {e}')
        # Input lineage only matters when artifacts are being migrated at all.
        if self.include_artifacts:
            try:
                if any(True for _ in run.used_artifacts()):
                    self._skipped('artifact-input-lineage')
            except Exception as e:
                logger.debug(f'{tag}: used_artifacts check failed for {run.id}: {e}')

    def _export_artifacts(self, run: Any, writer: PartWriter, tmp_dir: Path) -> None:
        base = self._row_base(run)
        try:
            artifacts: Iterable[Any] = run.logged_artifacts()
        except Exception as e:
            logger.warning(f'{tag}: artifacts unavailable for {run.id}: {e}')
            return
        versioning_lost = False
        for artifact in artifacts:
            # Only the artifact's *files* migrate — versions, non-'latest'
            # aliases, and the type/lineage graph don't. Skip wandb's internal
            # per-run history artifact (always present, always v0/latest) so it
            # doesn't false-trigger the flag.
            if getattr(artifact, 'type', None) != 'wandb-history':
                aliases = set(getattr(artifact, 'aliases', None) or [])
                if getattr(artifact, 'version', None) not in (None, 'v0') or (
                    aliases - {'latest'}
                ):
                    versioning_lost = True
            size = getattr(artifact, 'size', None)
            if (
                self.artifact_max_bytes is not None
                and size is not None
                and size > self.artifact_max_bytes
            ):
                logger.info(
                    f'{tag}: skipping artifact {artifact.name} '
                    f'({size} bytes > cap {self.artifact_max_bytes})'
                )
                self._skipped('artifact-over-size-cap')
                continue
            dest = tmp_dir / 'artifacts' / artifact.name
            try:
                artifact.download(root=str(dest))
            except Exception as e:
                logger.warning(f'{tag}: failed to download {artifact.name}: {e}')
                self._skipped('artifact-download-failed')
                continue
            timestamp_ms = parse_iso_ms(getattr(artifact, 'created_at', None)) or 0
            meta = json.dumps(
                {
                    'name': artifact.name,
                    'type': getattr(artifact, 'type', None),
                    'size': size,
                }
            )
            for path in sorted(p for p in dest.rglob('*') if p.is_file()):
                writer.write_row(
                    **base,
                    attribute_path=artifact.name,
                    attribute_type='artifact',
                    step=0,
                    timestamp_ms=timestamp_ms,
                    string_value=meta,
                    file_value=str(path.relative_to(tmp_dir)),
                )
                self._migrated('artifact-file')
        if versioning_lost:
            self._skipped('artifact-versioning')


def list_wandb_projects(entity: str, api_key: Optional[str] = None) -> List[str]:
    """Return the names of every wandb project under ``entity`` (for migrating a
    whole account at once). Imported lazily so the base package stays light."""
    import wandb

    api = wandb.Api(api_key=api_key)
    return [p.name for p in api.projects(entity)]
