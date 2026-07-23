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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    ) -> None:
        self.input_dir = Path(input_dir)
        self.dest_project = dest_project
        self.flush_every = flush_every
        self.max_pending = max_pending
        self.dry_run = dry_run
        self.run_ids = set(run_ids) if run_ids else None
        self.force_resume = force_resume
        self.stall_timeout = stall_timeout

    def load(self) -> Dict[str, Any]:
        """Load all staged runs. Returns {'loaded', 'skipped', 'failed'}.

        Each run is isolated: any failure (unreadable manifest, init error,
        replay error) is recorded in ``failed`` and the batch continues.
        """
        loaded, skipped, would_load = 0, 0, 0
        failed: List[Dict[str, str]] = []
        cache = LoadedCache(self.input_dir / LOADED_CACHE_FILENAME)

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
            if cache.is_loaded(external_id) and not self.force_resume:
                logger.info(f'{tag}: {external_id} already loaded, skipping')
                skipped += 1
                continue
            if self.dry_run:
                self._print_dry_run(run_dir, manifest, external_id)
                would_load += 1
                continue

            op = None
            try:
                try:
                    op = self._init_run(manifest, external_id)
                except RunExistsError:
                    # The run exists server-side but isn't in loaded_runs.json.
                    # Re-replaying would duplicate media, so by default treat it
                    # as already loaded and skip; record it so future re-runs
                    # skip via the cache. force_resume opts into resume +
                    # re-replay (e.g. to heal a genuinely interrupted load).
                    if not self.force_resume:
                        logger.info(
                            f'{tag}: {external_id} already exists on server; '
                            'skipping (pass --force-resume to resume and '
                            're-replay)'
                        )
                        cache.mark_loaded(external_id, {'note': 'existed-on-server'})
                        skipped += 1
                        continue
                    logger.warning(
                        f'{tag}: {external_id} already exists on server; '
                        'resuming to re-replay (--force-resume; media may '
                        'duplicate)'
                    )
                    op = self._init_run(manifest, external_id, resume=True)

                self._replay_run(run_dir, op)
                op.finish(code=0 if manifest.get('state') == 'finished' else 1)
                cache.mark_loaded(external_id, {'pluto_run_id': op.settings._op_id})
                loaded += 1
                logger.info(f'{tag}: loaded {external_id}')
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
        return sorted(
            d
            for d in self.input_dir.glob('*/*/runs/*')
            if d.is_dir() and is_run_exported(d) and (d / 'run.json').exists()
        )

    def _init_run(
        self,
        manifest: Dict[str, Any],
        external_id: str,
        resume: Optional[bool] = None,
    ) -> Any:
        tags = list(manifest.get('tags') or [])
        if 'import:wandb' not in tags:
            tags.append('import:wandb')
        settings: Dict[str, Any] = {
            'compat': {
                'createdAt': manifest.get('createdAt'),
                'updatedAt': manifest.get('updatedAt'),
            },
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
            }.items()
            if v
        }
        if wandb_block:
            op.update_config({'wandb': wandb_block})
        return op

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

        for table in iter_part_tables(run_dir):
            for row in table.to_pylist():
                attr_type = row['attribute_type']
                if attr_type in ('metric', 'system_metric'):
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
                    self._replay_media(run_dir, op, row)
                    note_nonmetric()
                elif attr_type == 'console':
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
                elif attr_type == 'artifact':
                    self._replay_artifact(run_dir, op, row)
                    note_nonmetric()

        close_group()
        flush_pending(force=True)
        if console_lines:
            op._log_console(console_lines)
            self._wait_for_backpressure(op)

    def _replay_media(self, run_dir: Path, op: Any, row: Dict[str, Any]) -> None:
        name = row['attribute_path']
        step = row['step']
        timestamp = row['timestamp_ms'] / 1000
        media_type = row['string_value'] or ''

        if media_type.startswith('{'):  # inline JSON (histogram)
            payload = json.loads(media_type)
            if payload.get('_type') == 'histogram':
                op.log(
                    {
                        name: pluto.Histogram(
                            [payload.get('values'), payload.get('bins')],
                            bins=None,
                        )
                    },
                    step=step,
                    timestamp=timestamp,
                )
            return

        path = _resolve_within(run_dir, row['file_value'])
        if path is None or not path.exists():
            logger.warning(
                f'{tag}: media file missing for {name!r} '
                f'({row["file_value"]}), skipping'
            )
            return

        caption = row['caption']
        if media_type == 'table-file':
            try:
                table_json = read_json(path)
                value: Any = pluto.Table(
                    data=table_json.get('data'),
                    columns=table_json.get('columns', []),
                )
            except Exception as e:
                logger.warning(f'{tag}: bad table file {path}: {e}')
                return
        else:
            make = _MEDIA_LOADERS.get(media_type)
            if make is not None:
                value = make(str(path), caption=caption)
            else:  # plotly/html/object3D/unknown -> raw artifact
                value = pluto.Artifact(str(path), caption=caption)
        op.log({name: value}, step=step, timestamp=timestamp)

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
