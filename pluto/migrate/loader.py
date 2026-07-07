"""
Load staged export directories into Pluto through the public client API.

Replays each exported run — run.json manifest plus parquet parts — as a
Pluto run with the ORIGINAL wall-clock timestamps (``op.log(timestamp=)``)
and creation time (``settings.compat`` createdAt/updatedAt). Idempotency
is run-level: a ``run_id`` external id (``wandb::{entity}/{project}/{id}``)
makes re-creation collide server-side, and ``loaded_runs.json`` records
finished loads so re-runs skip them. Metrics for a given (name, step) are
sent exactly once — the backend dedupes replayed points by highest
timestamp, so partial re-loads with identical staged timestamps are safe.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pluto
from pluto.migrate.state import LOADED_CACHE_FILENAME, LoadedCache, read_json

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'migrate'

CONSOLE_BATCH_SIZE = 1000

_MEDIA_LOADERS = {
    'image-file': lambda path, caption: pluto.Image(path, caption=caption),
    'audio-file': lambda path, caption: pluto.Audio(path, caption=caption),
    'video-file': lambda path, caption: pluto.Video(path, caption=caption),
}


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
    ) -> None:
        self.input_dir = Path(input_dir)
        self.dest_project = dest_project
        self.flush_every = flush_every
        self.max_pending = max_pending
        self.dry_run = dry_run
        self.run_ids = set(run_ids) if run_ids else None
        self.force_resume = force_resume

    def load(self) -> Dict[str, Any]:
        """Load all staged runs. Returns {'loaded', 'skipped', 'failed'}."""
        loaded, skipped = 0, 0
        failed: List[Dict[str, str]] = []
        cache = LoadedCache(self.input_dir / LOADED_CACHE_FILENAME)

        for run_dir in self._discover_runs():
            manifest = read_json(run_dir / 'run.json')
            run_id = manifest['run_id']
            if self.run_ids is not None and run_id not in self.run_ids:
                continue
            external_id = (
                f'wandb::{manifest["entity"]}/{manifest["project"]}/{run_id}'
            )
            if cache.is_loaded(external_id) and not self.force_resume:
                logger.info(f'{tag}: {external_id} already loaded, skipping')
                skipped += 1
                continue
            if self.dry_run:
                self._print_dry_run(run_dir, manifest, external_id)
                continue

            try:
                op = self._init_run(manifest, external_id)
            except RuntimeError as e:
                if 'already exists' in str(e):
                    # Another machine (or a crashed load after finish) already
                    # created this run — record and move on.
                    logger.warning(
                        f'{tag}: {external_id} exists on server, skipping'
                    )
                    cache.mark_loaded(external_id, {'pluto_run_id': None})
                    skipped += 1
                    continue
                raise

            try:
                self._replay_run(run_dir, manifest, op)
                op.finish(code=0 if manifest.get('state') == 'finished' else 1)
                cache.mark_loaded(
                    external_id, {'pluto_run_id': op.settings._op_id}
                )
                loaded += 1
                logger.info(f'{tag}: loaded {external_id}')
            except Exception as e:
                logger.error(f'{tag}: load failed for {external_id}: {e}')
                failed.append(
                    {'run_id': run_id, 'error': f'{type(e).__name__}: {e}'}
                )
                try:
                    op.finish(code=1)
                except Exception:
                    pass

        return {'loaded': loaded, 'skipped': skipped, 'failed': failed}

    def _discover_runs(self) -> List[Path]:
        from pluto.migrate.state import is_run_exported

        return sorted(
            d
            for d in self.input_dir.glob('*/*/runs/*')
            if d.is_dir() and is_run_exported(d) and (d / 'run.json').exists()
        )

    def _init_run(self, manifest: Dict[str, Any], external_id: str) -> Any:
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
        }
        op = pluto.init(
            project=self.dest_project or manifest['project'],
            name=manifest.get('name'),
            config=manifest.get('config') or None,
            tags=tags,
            run_id=external_id,
            resume=self.force_resume,
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

    def _replay_run(
        self, run_dir: Path, manifest: Dict[str, Any], op: Any
    ) -> None:
        from pluto.migrate.schema import iter_part_tables

        # (attribute_type, step, timestamp_ms) of the group being buffered;
        # rows are staged in write order so same-step metrics are contiguous.
        group_key: Optional[Tuple[str, int, int]] = None
        group_metrics: Dict[str, float] = {}
        console_lines: List[Tuple[str, str, float, int]] = []
        flushes = 0

        def flush_group() -> None:
            nonlocal group_key, group_metrics, flushes
            if group_key is None or not group_metrics:
                group_key, group_metrics = None, {}
                return
            _, step, timestamp_ms = group_key
            op.log(group_metrics, step=step, timestamp=timestamp_ms / 1000)
            group_key, group_metrics = None, {}
            flushes += 1
            if flushes % self.flush_every == 0:
                self._wait_for_backpressure(op)

        for table in iter_part_tables(run_dir):
            for row in table.to_pylist():
                attr_type = row['attribute_type']
                if attr_type in ('metric', 'system_metric'):
                    key = (attr_type, row['step'], row['timestamp_ms'])
                    if key != group_key:
                        flush_group()
                        group_key = key
                    group_metrics[row['attribute_path']] = row['float_value']
                    continue
                flush_group()
                if attr_type == 'media':
                    self._replay_media(run_dir, op, row)
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
                        console_lines = []
                elif attr_type == 'artifact':
                    self._replay_artifact(run_dir, op, row)

        flush_group()
        if console_lines:
            op._log_console(console_lines)

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

        path = run_dir / (row['file_value'] or '')
        if not row['file_value'] or not path.exists():
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
                value = make(str(path), caption)
            else:  # plotly/html/object3D/unknown -> raw artifact
                value = pluto.Artifact(str(path), caption=caption)
        op.log({name: value}, step=step, timestamp=timestamp)

    def _replay_artifact(self, run_dir: Path, op: Any, row: Dict[str, Any]) -> None:
        path = run_dir / (row['file_value'] or '')
        if not row['file_value'] or not path.exists():
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
        """Bound the sync queue so huge runs don't balloon SQLite/memory."""
        try:
            pending = op._sync_manager.get_pending_count()
        except Exception:
            return
        while pending > self.max_pending:
            logger.info(
                f'{tag}: {pending} records pending upload, throttling loader'
            )
            time.sleep(0.5)
            pending = op._sync_manager.get_pending_count()

    def _print_dry_run(
        self, run_dir: Path, manifest: Dict[str, Any], external_id: str
    ) -> None:
        from pluto.migrate.schema import part_files

        parts = part_files(run_dir)
        size = sum(p.stat().st_size for p in parts)
        print(
            f'[dry-run] {external_id} -> project '
            f'{self.dest_project or manifest["project"]!r} '
            f'name={manifest.get("name")!r} '
            f'parts={len(parts)} ({size / 1e6:.1f} MB)'
        )
