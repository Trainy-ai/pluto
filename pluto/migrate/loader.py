"""
Load staged export directories into Pluto through the public client API.

Replays each exported run — run.json manifest plus parquet parts — as a
Pluto run with the ORIGINAL wall-clock timestamps (``op.log(timestamp=)``)
and creation time (``settings.compat`` createdAt/updatedAt). Idempotency
is run-level: a ``run_id`` external id (``wandb::{entity}/{project}/{id}``)
makes re-creation collide server-side. ``loaded_runs.json`` records
finished loads so re-runs skip them; a run that already exists on the
server but isn't in the local cache is skipped by default rather than
re-replayed, because re-replaying would duplicate media. Pass
``force_resume`` to intentionally resume and re-replay such a run
(metric points carry identical staged timestamps, so the backend's
replace-by-time dedup keeps metrics safe; media may duplicate).

Each run is loaded independently: a single run failing (unreadable
manifest, init error, replay error, dead sync process) is recorded in
``failed`` and never aborts the rest of the batch, so a re-run retries
only the runs that did not finish.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

import pluto
from pluto.migrate.schema import iter_part_tables, part_files
from pluto.migrate.state import (
    LOADED_CACHE_FILENAME,
    LoadedCache,
    is_run_exported,
    read_json,
)
from pluto.op import RunExistsError

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'migrate'

CONSOLE_BATCH_SIZE = 1000

# A categorical/string series with more distinct values than this is treated as
# free-form text, not a set of states — rendering it as a state timeline would
# be meaningless (one band per point), so we skip it rather than send it. The
# guard lives here (not the exporter) because it needs the whole run's series to
# count distinct values.
STRING_SERIES_MAX_CARDINALITY = 50

# All three take (data, caption=...); table-file and inline histograms are
# handled separately in _replay_media.
_MEDIA_LOADERS = {
    'image-file': pluto.Image,
    'audio-file': pluto.Audio,
    'video-file': pluto.Video,
}


def _resolve_within(run_dir: Path, rel: Optional[str]) -> Optional[Path]:
    """Resolve a staged file path, rejecting anything outside ``run_dir``.

    A malicious or corrupt part row could carry an absolute path or a
    ``..`` sequence in ``file_value``; without this guard the loader would
    happily read and upload arbitrary host files (e.g. ``/etc/passwd``).
    Symlinks are resolved too, so a symlink pointing outside is rejected.
    Returns None for empty/out-of-bounds paths (caller treats as missing).
    """
    if not rel:
        return None
    base = run_dir.resolve()
    candidate = (run_dir / rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        logger.warning(f'{tag}: refusing staged file outside run dir: {rel!r}')
        return None
    return candidate


class PlutoLoader:
    """Replay a pluto.migrate export directory into Pluto."""

    def __init__(
        self,
        input_dir: Union[str, Path],
        dest_project: Optional[str] = None,
        flush_every: int = 500,
        max_pending: int = 5000,
        dry_run: bool = False,
        run_ids: Optional[List[str]] = None,
        force_resume: bool = False,
        stall_timeout: float = 600.0,
        cleanup: bool = False,
        projects: Optional[List[str]] = None,
        exclude_projects: Optional[List[str]] = None,
        cache_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        # Resume ledger. Overridable so parallel per-project loaders each get
        # their own file (one shared ledger would race under concurrent writes).
        self.cache_path = (
            Path(cache_path)
            if cache_path is not None
            else self.input_dir / LOADED_CACHE_FILENAME
        )
        self.dest_project = dest_project
        self.flush_every = flush_every
        self.max_pending = max_pending
        self.dry_run = dry_run
        self.run_ids = set(run_ids) if run_ids else None
        # Which staged projects to load: include-list (None = all) minus excludes.
        self.projects = set(projects) if projects else None
        self.exclude_projects = set(exclude_projects) if exclude_projects else set()
        self.force_resume = force_resume
        self.stall_timeout = stall_timeout
        # Delete each run's staged files once it is confirmed loaded, so a large
        # migration doesn't keep a full duplicate copy on local disk. Peak disk
        # then bounds to the un-loaded backlog rather than the whole export.
        self.cleanup = cleanup

    def load(self) -> Dict[str, Any]:
        """Load all staged runs. Returns {'loaded', 'skipped', 'failed'}.

        Each run is isolated: any failure (unreadable manifest, init error,
        replay error) is recorded in ``failed`` and the batch continues.
        """
        loaded, skipped, would_load = 0, 0, 0
        failed: List[Dict[str, str]] = []
        cache = LoadedCache(self.cache_path)

        for run_dir in self._discover_runs():
            # A truncated/hand-edited run.json must not sink the whole batch.
            try:
                manifest = read_json(run_dir / 'run.json')
                run_id = manifest['run_id']
                external_id = (
                    f'wandb::{manifest["entity"]}/{manifest["project"]}/{run_id}'
                )
            except Exception as e:
                logger.error(f'{tag}: unreadable manifest in {run_dir}: {e}')
                failed.append(
                    {'run_id': run_dir.name, 'error': f'{type(e).__name__}: {e}'}
                )
                continue

            if self.run_ids is not None and run_id not in self.run_ids:
                continue
            # Key the load cache on (run, destination project) so loading the
            # same export into a *different* Pluto project isn't wrongly skipped.
            dest_project = self.dest_project or manifest['project']
            cache_key = f'{external_id}@@{dest_project}'
            if cache.is_loaded(cache_key) and not self.force_resume:
                logger.info(
                    f'{tag}: {external_id} already loaded into {dest_project!r}, '
                    'skipping'
                )
                skipped += 1
                continue
            if self.dry_run:
                self._print_dry_run(run_dir, manifest, external_id)
                would_load += 1
                continue

            op = None
            try:
                try:
                    op = self._init_run(manifest, external_id, run_dir)
                    # Remember, before replaying, that we created this run. If
                    # replay crashes now, a later re-run sees the in_progress
                    # marker and resumes to complete it (rather than skipping).
                    cache.mark_in_progress(cache_key)
                except RunExistsError:
                    # The run exists server-side but isn't marked done here.
                    if self.force_resume or cache.is_in_progress(cache_key):
                        # We started this run before and it didn't finish (crash
                        # mid-replay), or the user forced it: resume and
                        # re-replay to complete it. Metric points dedup by their
                        # identical staged timestamps; media sent before the
                        # crash may duplicate.
                        logger.warning(
                            f'{tag}: {external_id} exists on server but was not '
                            'finished; resuming to complete it (media may '
                            'duplicate)'
                        )
                        cache.mark_in_progress(cache_key)
                        op = self._init_run(manifest, external_id, run_dir, resume=True)
                    else:
                        # Exists but we never started it here (e.g. loaded on
                        # another machine): skip re-replay to avoid duplicating
                        # media. BUT the create-with-existing above already
                        # reopened it to RUNNING server-side (the DDP-style
                        # "create an existing run = resume it" path). Left as-is
                        # the run is stuck RUNNING with a now() finish time, so
                        # re-attach and finish() — no replay — to restore its
                        # terminal status + historical statusUpdated, then skip.
                        logger.info(
                            f'{tag}: {external_id} already exists on server; '
                            'restoring its finished state and skipping (pass '
                            '--force-resume to resume and re-replay)'
                        )
                        try:
                            restore = self._init_run(
                                manifest, external_id, run_dir, resume=True
                            )
                            restore.finish(
                                code=0 if manifest.get('state') == 'finished' else 1
                            )
                        except Exception as e:
                            logger.warning(
                                f'{tag}: could not restore terminal status for '
                                f'{external_id}: {e}'
                            )
                        cache.mark_loaded(cache_key, {'note': 'existed-on-server'})
                        skipped += 1
                        continue

                self._replay_run(run_dir, op)
                op.finish(code=0 if manifest.get('state') == 'finished' else 1)
                cache.mark_loaded(cache_key, {'pluto_run_id': op.settings._op_id})
                loaded += 1
                logger.info(f'{tag}: loaded {external_id}')
                if self.cleanup:
                    # Reclaim disk now that the run is safely loaded (and
                    # recorded in loaded_runs.json, so it's still skipped on a
                    # re-run even though its staged files are gone).
                    shutil.rmtree(run_dir, ignore_errors=True)
            except Exception as e:
                logger.error(f'{tag}: load failed for {external_id}: {e}')
                failed.append({'run_id': run_id, 'error': f'{type(e).__name__}: {e}'})
                if op is not None:
                    try:
                        op.finish(code=1)
                    except Exception:
                        pass

        if self.dry_run:
            print(
                f'[dry-run] would load {would_load} run(s); '
                f'{skipped} already loaded (would skip)'
            )
        return {'loaded': loaded, 'skipped': skipped, 'failed': failed}

    def _discover_runs(self) -> List[Path]:
        # Layout: input_dir/{entity}/{project}/runs/{run_id}. Filter by the
        # project component so a scoped load (--project / --exclude) only picks
        # up the requested projects.
        return sorted(
            d
            for d in self.input_dir.glob('*/*/runs/*')
            if d.is_dir()
            and is_run_exported(d)
            and (d / 'run.json').exists()
            and self._project_in_scope(d.parent.parent.name)
        )

    def _project_in_scope(self, project: str) -> bool:
        if project in self.exclude_projects:
            return False
        return self.projects is None or project in self.projects

    def _init_run(
        self,
        manifest: Dict[str, Any],
        external_id: str,
        run_dir: Path,
        resume: Optional[bool] = None,
    ) -> Any:
        tags = list(manifest.get('tags') or [])
        if 'import:wandb' not in tags:
            tags.append('import:wandb')
        compat: Dict[str, Any] = {
            'createdAt': manifest.get('createdAt'),
            'updatedAt': manifest.get('updatedAt'),
        }
        # Forward the original run's own metadata (git/OS/GPU/python/args) as
        # systemMetadata so repro context survives the migration. Only set when
        # present so normal runs (empty compat) are unaffected.
        if manifest.get('metadata') is not None:
            compat['systemMetadata'] = manifest['metadata']
        settings: Dict[str, Any] = {
            'compat': compat,
            # Never attribute the migration host's console/hardware to the
            # imported run.
            'disable_console': True,
            'disable_system_metrics': True,
            # The historical-timestamp path only exists in the sync store;
            # force it on even if the user's defaults disable it.
            'sync_process_enabled': True,
        }
        op = pluto.init(
            project=self.dest_project or manifest['project'],
            name=manifest.get('name'),
            config=manifest.get('config') or None,
            tags=tags,
            run_id=external_id,
            resume=self.force_resume if resume is None else resume,
            settings=settings,
        )
        wandb_block = {
            k: v
            for k, v in {
                'notes': manifest.get('notes'),
                'url': manifest.get('url'),
                'state': manifest.get('state'),
                'summary': manifest.get('summary'),
                # Custom-chart (wandb.plot.*) panel specs recovered by the
                # exporter: each binds a Vega preset to a migrated backing
                # table. Forwarded here so the Pluto side can rebuild the panels.
                'custom_charts': self._read_custom_charts(run_dir),
            }.items()
            if v
        }
        if wandb_block:
            op.update_config({'wandb': wandb_block})
        return op

    @staticmethod
    def _read_custom_charts(run_dir: Path) -> Optional[List[Dict[str, Any]]]:
        """Load the exporter's staged custom-chart panel specs, if any."""
        path = run_dir / 'custom_charts.json'
        if not path.exists():
            return None
        try:
            panels = read_json(path).get('panels') or None
        except Exception as e:
            logger.warning(
                f'{tag}: could not read {path.name}: {type(e).__name__}: {e}'
            )
            return None
        return panels

    @staticmethod
    def _sys_metric_name(name: str) -> str:
        """Map a source-native system metric name into Pluto's sys/ namespace."""
        if name.startswith('system.'):
            return 'sys/' + name[len('system.') :]
        if name.startswith('sys/'):
            return name
        return f'sys/{name}'

    def _replay_run(self, run_dir: Path, op: Any) -> None:
        # (attribute_type, step, timestamp_ms) of the group being buffered;
        # rows are staged in write order so same-step metrics are contiguous.
        group_key: Optional[Tuple[str, int, int]] = None
        group_metrics: Dict[str, float] = {}
        # Closed groups accumulate here and flush through one SQLite
        # transaction per flush_every groups (op._log_metrics_batch).
        pending_groups: List[Tuple[Dict[str, float], int, float]] = []
        console_lines: List[Tuple[str, str, float, int]] = []
        # Categorical/status series buffered per attribute_path for the whole
        # run: the cardinality guard (STRING_SERIES_MAX_CARDINALITY) needs every
        # point before it can decide whether the series is renderable, so these
        # are sent once at the end rather than streamed. Each entry is a list of
        # (step, timestamp_ms, value).
        string_series: Dict[str, List[Tuple[int, int, str]]] = {}

        def close_group() -> None:
            nonlocal group_key, group_metrics
            if group_key is not None and group_metrics:
                _, step, timestamp_ms = group_key
                pending_groups.append((group_metrics, step, timestamp_ms / 1000))
            group_key, group_metrics = None, {}

        def flush_pending(force: bool = False) -> None:
            if pending_groups and (force or len(pending_groups) >= self.flush_every):
                op._log_metrics_batch(list(pending_groups))
                pending_groups.clear()
                self._wait_for_backpressure(op)

        # Media/console/artifact rows enqueue outside the scalar-metric flush,
        # so they need their own backpressure cadence — otherwise a run that is
        # mostly images/logs (few scalars) never triggers _wait_for_backpressure
        # and can balloon the sync queue / staged files past max_pending.
        nonmetric_since_check = 0

        def note_nonmetric(n: int = 1) -> None:
            nonlocal nonmetric_since_check
            nonmetric_since_check += n
            if nonmetric_since_check >= self.flush_every:
                nonmetric_since_check = 0
                self._wait_for_backpressure(op)

        # Media grouping: consecutive media rows sharing (name, step, timestamp)
        # are one logged batch (e.g. wandb.log({"gallery": [img0, img1, ...]})).
        # Logging them as a single list makes op.log assign each item its
        # sampleIndex (0,1,2,...), which the server uses to preserve logged
        # order — one op.log per row would make every item sampleIndex 0.
        media_key: Optional[Tuple[str, int, int]] = None
        media_items: List[Any] = []

        def flush_media() -> None:
            nonlocal media_key, media_items
            if media_key is not None and media_items:
                name, step, ts_ms = media_key
                value = media_items[0] if len(media_items) == 1 else media_items
                op.log({name: value}, step=step, timestamp=ts_ms / 1000)
                note_nonmetric(len(media_items))
            media_key, media_items = None, []

        for table in iter_part_tables(run_dir):
            for row in table.to_pylist():
                attr_type = row['attribute_type']
                if attr_type in ('metric', 'system_metric'):
                    flush_media()
                    key = (attr_type, row['step'], row['timestamp_ms'])
                    if key != group_key:
                        close_group()
                        flush_pending()
                        group_key = key
                    name = row['attribute_path']
                    if attr_type == 'system_metric':
                        name = self._sys_metric_name(name)
                    group_metrics[name] = row['float_value']
                    continue
                close_group()
                if attr_type == 'media':
                    # One malformed media/histogram row must not abort the whole
                    # run's replay — skip it and keep the rest of the run's data.
                    try:
                        value = self._build_media_value(run_dir, row)
                    except Exception as e:
                        logger.warning(
                            f'{tag}: skipping bad media row '
                            f'{row.get("attribute_path")!r} @ step '
                            f'{row.get("step")}: {type(e).__name__}: {e}'
                        )
                        value = None
                    if value is not None:
                        mkey = (
                            row['attribute_path'],
                            row['step'],
                            row['timestamp_ms'],
                        )
                        if mkey != media_key:
                            flush_media()
                            media_key = mkey
                        media_items.append(value)
                    continue
                flush_media()  # any non-media row ends the current media batch
                if attr_type == 'console':
                    console_lines.append(
                        (
                            row['string_value'] or '',
                            'INFO',
                            row['timestamp_ms'] / 1000,
                            row['step'],
                        )
                    )
                    if len(console_lines) >= CONSOLE_BATCH_SIZE:
                        op._log_console(console_lines)
                        note_nonmetric(len(console_lines))
                        console_lines = []
                elif attr_type == 'string_series':
                    string_series.setdefault(row['attribute_path'], []).append(
                        (row['step'], row['timestamp_ms'], row['string_value'] or '')
                    )
                elif attr_type == 'artifact':
                    try:
                        self._replay_artifact(run_dir, op, row)
                    except Exception as e:
                        logger.warning(
                            f'{tag}: skipping bad artifact row '
                            f'{row.get("attribute_path")!r} @ step '
                            f'{row.get("step")}: {type(e).__name__}: {e}'
                        )
                    note_nonmetric()

        flush_media()
        close_group()
        flush_pending(force=True)
        if console_lines:
            op._log_console(console_lines)
            self._wait_for_backpressure(op)
        if string_series:
            self._send_string_series(op, string_series)

    def _send_string_series(
        self, op: Any, series: Dict[str, List[Tuple[int, int, str]]]
    ) -> None:
        """Send categorical/status series to the data-ingest endpoint.

        String history points have no numeric/media home; Pluto stores them as
        a ``string-series`` dataType in the data ingest (rendered as a state
        timeline). This posts NDJSON directly to ``url_data`` — mirroring the
        sync process's data upload — rather than routing through the scalar or
        media client paths.

        Applies the cardinality guard (:data:`STRING_SERIES_MAX_CARDINALITY`):
        a series with too many distinct values is free-form text, not states,
        so it is skipped. Failures are non-fatal — the run's scalars/media are
        already loaded and must not be lost to an ingest hiccup here.
        """
        s = op.settings
        if not s.url_data or not s._op_id:
            return
        renderable: Dict[str, List[Tuple[int, int, str]]] = {}
        for name, points in series.items():
            distinct = {v for _, _, v in points}
            if len(distinct) > STRING_SERIES_MAX_CARDINALITY:
                logger.warning(
                    f'{tag}: string-series {name!r} has {len(distinct)} distinct '
                    f'values (> {STRING_SERIES_MAX_CARDINALITY}); skipping as '
                    'free-form text, not a categorical timeline'
                )
                continue
            renderable[name] = points
        if not renderable:
            return

        common = {
            'Authorization': f'Bearer {s._auth}',
            'User-Agent': str(s.tag),
            'X-Run-Id': str(s._op_id),
            'X-Run-Name': str(s._op_name or ''),
            'X-Project-Name': str(s.project or ''),
        }
        # NDJSON: one {time, step, dataType, logName, data} object per point.
        lines = [
            json.dumps(
                {
                    'time': ts_ms,
                    'step': step,
                    'dataType': 'string-series',
                    'logName': name,
                    'data': value,
                }
            )
            for name, points in renderable.items()
            for step, ts_ms, value in points
        ]
        body = '\n'.join(lines) + '\n'
        try:
            # Register the log names (logType DATA) so the server indexes them,
            # then ingest the points.
            resp = httpx.post(
                s.url_meta,
                headers={**common, 'Content-Type': 'application/json'},
                content=json.dumps(
                    {
                        'runId': s._op_id,
                        'logType': 'DATA',
                        'logName': list(renderable),
                    }
                ),
                timeout=30.0,
            )
            resp.raise_for_status()
            resp = httpx.post(
                s.url_data,
                headers={**common, 'Content-Type': 'application/x-ndjson'},
                content=body,
                timeout=60.0,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning(
                f'{tag}: failed to send {len(renderable)} string-series to '
                f'ingest: {type(e).__name__}: {e}'
            )

    def _build_media_value(self, run_dir: Path, row: Dict[str, Any]) -> Optional[Any]:
        """Convert one staged media row into a pluto media value.

        Returns None to skip the row (missing file, empty histogram, unknown
        inline type). The caller batches consecutive same-key values so a
        multi-sample step (e.g. an image gallery) keeps its logged order.
        """
        name = row['attribute_path']
        step = row['step']
        media_type = row['string_value'] or ''

        if media_type.startswith('{'):  # inline JSON (histogram)
            payload = json.loads(media_type)
            if payload.get('_type') == 'histogram':
                values = payload.get('values')
                bins = payload.get('bins')
                if values is None:
                    logger.warning(
                        f'{tag}: histogram {name!r} @ step {step} has no '
                        'values, skipping'
                    )
                    return None
                if bins is None:
                    # wandb frequently stores histogram counts without bin
                    # edges. pluto.Histogram's pre-binned form needs
                    # len(edges) == len(counts) + 1, so synthesize integer
                    # edges — the counts are preserved, only the x-axis is
                    # generic.
                    bins = list(range(len(values) + 1))
                return pluto.Histogram([values, bins], bins=None)
            return None  # unknown inline media type

        path = _resolve_within(run_dir, row['file_value'])
        if path is None or not path.exists():
            logger.warning(
                f'{tag}: media file missing for {name!r} '
                f'({row["file_value"]}), skipping'
            )
            return None

        caption = row['caption']
        if media_type == 'table-file':
            table_json = read_json(path)
            return pluto.Table(
                data=table_json.get('data'),
                columns=table_json.get('columns', []),
            )
        make = _MEDIA_LOADERS.get(media_type)
        if make is not None:
            return make(str(path), caption=caption)
        # plotly/html/object3D/unknown -> raw artifact
        return pluto.Artifact(str(path), caption=caption)

    def _replay_artifact(self, run_dir: Path, op: Any, row: Dict[str, Any]) -> None:
        path = _resolve_within(run_dir, row['file_value'])
        if path is None or not path.exists():
            logger.warning(
                f'{tag}: artifact file missing ({row["file_value"]}), skipping'
            )
            return
        metadata = None
        if row['string_value']:
            try:
                metadata = json.loads(row['string_value'])
            except ValueError:
                metadata = None
        op.log(
            {
                row['attribute_path']: pluto.Artifact(
                    str(path), caption=path.name, metadata=metadata
                )
            },
            step=row['step'],
            timestamp=row['timestamp_ms'] / 1000,
        )

    def _wait_for_backpressure(self, op: Any) -> None:
        """Bound the sync queue so huge runs don't balloon SQLite/memory.

        Fails fast if the sync subprocess has died: without this the loop
        would sleep out the full ``stall_timeout`` on every flush (the
        pending count never drops with no uploader), turning a large replay
        into hours of pure sleeping. Raising here surfaces the dead process
        as a per-run failure so the batch continues with the next run.

        For an *alive but slow* uploader (unreachable server, throttling)
        it stays bounded by ``stall_timeout``: it logs and moves on rather
        than hanging — data stays in the sync store either way.
        """
        manager = getattr(op, '_sync_manager', None)
        if manager is None:
            return
        deadline = time.time() + self.stall_timeout
        sleep_s = 0.5
        throttled = False
        while True:
            proc = getattr(manager, '_process', None)
            if proc is not None:
                # poll() -> None while alive, an int exit code once dead.
                # (isinstance keeps a mocked manager from tripping this.)
                code = proc.poll()
                if isinstance(code, int):
                    raise RuntimeError(
                        f'sync process exited (code {code}) with data still '
                        'pending; aborting this run (data preserved in the '
                        'sync store for a later retry)'
                    )
            try:
                pending = manager.get_pending_count()
            except Exception:
                return
            if pending <= self.max_pending:
                return
            if time.time() >= deadline:
                logger.warning(
                    f'{tag}: sync queue still has {pending} pending records '
                    f'after {self.stall_timeout:.0f}s; continuing (uploads '
                    'proceed in the background)'
                )
                return
            if not throttled:
                logger.info(
                    f'{tag}: {pending} records pending upload, throttling loader'
                )
                throttled = True
            time.sleep(sleep_s)
            sleep_s = min(sleep_s * 2, 5.0)

    def _print_dry_run(
        self, run_dir: Path, manifest: Dict[str, Any], external_id: str
    ) -> None:
        parts = part_files(run_dir)
        size = sum(p.stat().st_size for p in parts)
        print(
            f'[dry-run] {external_id} -> project '
            f'{self.dest_project or manifest["project"]!r} '
            f'name={manifest.get("name")!r} '
            f'parts={len(parts)} ({size / 1e6:.1f} MB)'
        )
