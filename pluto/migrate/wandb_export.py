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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from pluto.migrate.schema import PartWriter
from pluto.migrate.state import is_run_exported, mark_run_exported, write_json_atomic

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'migrate'

MANIFEST_FILENAME = 'manifest.json'

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
    ) -> None:
        self.entity = entity
        self.project = project
        self.output_dir = Path(output_dir)
        self._api = api
        self._api_key = api_key
        self.run_ids = set(run_ids) if run_ids else None
        self.after_ms = parse_iso_ms(after)
        self.before_ms = parse_iso_ms(before)
        self.include_artifacts = include_artifacts
        self.artifact_max_bytes = artifact_max_bytes
        self.include_console = include_console
        self.include_system = include_system
        self.include_files = include_files
        self.history_page_size = history_page_size

    @property
    def api(self) -> Any:
        if self._api is None:
            import wandb

            self._api = wandb.Api(api_key=self._api_key)
        return self._api

    @property
    def project_path(self) -> str:
        return f'{self.entity}/{self.project}'

    def export(self) -> Dict[str, Any]:
        """Export all matching runs. Returns {'exported', 'skipped', 'failed'}."""
        exported, skipped = 0, 0
        failed: List[Dict[str, str]] = []

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

            try:
                self._export_run(run, run_dir)
                exported += 1
                logger.info(f'{tag}: exported {run.id} ({run.name})')
            except Exception as e:  # keep going: one bad run must not stop all
                logger.error(f'{tag}: export failed for {run.id}: {e}')
                failed.append({'run_id': run.id, 'error': f'{type(e).__name__}: {e}'})

        summary = {'exported': exported, 'skipped': skipped, 'failed': failed}
        write_json_atomic(
            self.output_dir / MANIFEST_FILENAME,
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
            },
        )

    def _row_base(self, run: Any) -> Dict[str, str]:
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
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            writer.write_row(
                **base,
                attribute_path=key,
                attribute_type='metric',
                step=step,
                timestamp_ms=timestamp_ms,
                float_value=float(value),
            )
            return
        if isinstance(value, dict):
            media_type = value.get('_type')
            if media_type in _FILE_MEDIA_TYPES and value.get('path'):
                writer.write_row(
                    **base,
                    attribute_path=key,
                    attribute_type='media',
                    step=step,
                    timestamp_ms=timestamp_ms,
                    string_value=media_type,
                    file_value=f'files/{value["path"]}',
                    caption=value.get('caption'),
                )
            elif media_type == 'histogram':
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
                            'bins': value.get('bins'),
                        }
                    ),
                )
            elif media_type == 'images/separated' and value.get('filenames'):
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
            else:
                logger.debug(
                    f'{tag}: skipping unsupported history value '
                    f'{key!r} (_type={media_type!r})'
                )

    def _export_system_metrics(self, run: Any, writer: PartWriter) -> None:
        base = self._row_base(run)
        try:
            events = run.history(stream='events', pandas=False)
        except Exception as e:
            logger.warning(f'{tag}: system metrics unavailable for {run.id}: {e}')
            return
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

    def _download_files(self, run: Any, files_dir: Path) -> None:
        files_dir.mkdir(parents=True, exist_ok=True)
        for f in run.files():
            if not self.include_files and f.name != 'output.log':
                continue
            try:
                f.download(root=str(files_dir), exist_ok=True)
            except Exception as e:
                logger.warning(f'{tag}: failed to download {f.name}: {e}')

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
        fallback_ms = created_ms or 0
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

    def _export_artifacts(self, run: Any, writer: PartWriter, tmp_dir: Path) -> None:
        base = self._row_base(run)
        try:
            artifacts: Iterable[Any] = run.logged_artifacts()
        except Exception as e:
            logger.warning(f'{tag}: artifacts unavailable for {run.id}: {e}')
            return
        for artifact in artifacts:
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
                continue
            dest = tmp_dir / 'artifacts' / artifact.name
            try:
                artifact.download(root=str(dest))
            except Exception as e:
                logger.warning(f'{tag}: failed to download {artifact.name}: {e}')
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
