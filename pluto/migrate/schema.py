"""
Parquet staging format for pluto.migrate.

One long/tall schema holds every per-point record of a run (metrics,
system metrics, media references, console lines, artifact files);
``attribute_type`` discriminates. Run-scalar data (name, config, tags,
summary, timestamps) lives in the sibling ``run.json`` manifest instead.

Rows are written through :class:`PartWriter`, which rotates output files
(``part-00000.parquet``, ``part-00001.parquet``, ...) once the current
part exceeds ``max_part_bytes`` on disk, so arbitrarily long runs stage
in bounded memory and load back part by part.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq

ATTRIBUTE_TYPES = {
    'metric',  # scalar metric history point (float_value)
    'system_metric',  # host stats history point (float_value)
    'media',  # image/audio/video/table/histogram (string_value=_type,
    #           file_value=relative path or inline JSON in string_value)
    'console',  # one console line (string_value, step=line number)
    'artifact',  # one file inside a logged artifact (file_value)
}

SCHEMA = pa.schema(
    [
        pa.field('project_id', pa.string()),
        pa.field('run_id', pa.string()),
        pa.field('attribute_path', pa.string()),
        pa.field('attribute_type', pa.string()),
        pa.field('step', pa.int64()),
        pa.field('timestamp_ms', pa.int64()),
        pa.field('float_value', pa.float64()),
        pa.field('string_value', pa.string()),
        pa.field('file_value', pa.string()),
        pa.field('caption', pa.string()),
    ]
)

_COLUMNS = [f.name for f in SCHEMA]

DEFAULT_MAX_PART_BYTES = 50 * 1024 * 1024
DEFAULT_ROWS_PER_FLUSH = 20_000

PART_PREFIX = 'part-'
PART_SUFFIX = '.parquet'


class PartWriter:
    """Buffered writer producing rotated parquet parts for one run.

    Rows accumulate in memory and flush to the open part every
    ``rows_per_flush`` rows; after each flush the part rotates if it
    outgrew ``max_part_bytes``. Use as a context manager so the trailing
    buffer is flushed and the last part closed.
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
        rows_per_flush: int = DEFAULT_ROWS_PER_FLUSH,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._max_part_bytes = max_part_bytes
        self._rows_per_flush = rows_per_flush
        self._buffer: List[dict] = []
        self._part_index = 0
        self._writer: Optional[pq.ParquetWriter] = None
        self._part_path: Optional[Path] = None
        self.rows_written = 0

    def __enter__(self) -> 'PartWriter':
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def write_row(
        self,
        *,
        project_id: str,
        run_id: str,
        attribute_path: str,
        attribute_type: str,
        timestamp_ms: int,
        step: Optional[int] = None,
        float_value: Optional[float] = None,
        string_value: Optional[str] = None,
        file_value: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> None:
        if attribute_type not in ATTRIBUTE_TYPES:
            raise ValueError(
                f'unknown attribute_type {attribute_type!r}, '
                f'expected one of {sorted(ATTRIBUTE_TYPES)}'
            )
        self._buffer.append(
            {
                'project_id': project_id,
                'run_id': run_id,
                'attribute_path': attribute_path,
                'attribute_type': attribute_type,
                'step': step,
                'timestamp_ms': timestamp_ms,
                'float_value': float_value,
                'string_value': string_value,
                'file_value': file_value,
                'caption': caption,
            }
        )
        if len(self._buffer) >= self._rows_per_flush:
            self._flush()

    def close(self) -> None:
        self._flush()
        self._close_part()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=SCHEMA)
        self._buffer = []
        if self._writer is None:
            self._part_path = self._run_dir / (
                f'{PART_PREFIX}{self._part_index:05d}{PART_SUFFIX}'
            )
            self._writer = pq.ParquetWriter(
                self._part_path, SCHEMA, compression='snappy'
            )
            self._part_index += 1
        self._writer.write_table(table)
        self.rows_written += table.num_rows
        assert self._part_path is not None  # set alongside _writer above
        if os.path.getsize(self._part_path) >= self._max_part_bytes:
            self._close_part()

    def _close_part(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._part_path = None


def part_files(run_dir: Union[str, Path]) -> List[Path]:
    """Return a run's parquet parts in write order."""
    return sorted(Path(run_dir).glob(f'{PART_PREFIX}*{PART_SUFFIX}'))


def iter_part_tables(run_dir: Union[str, Path]) -> Iterator[pa.Table]:
    """Yield each parquet part as a table, in write order (bounded memory)."""
    for path in part_files(run_dir):
        yield pq.read_table(path, schema=SCHEMA)
