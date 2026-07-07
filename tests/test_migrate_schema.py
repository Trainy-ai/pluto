"""
Unit tests for pluto.migrate.schema (parquet part writing/reading) and
pluto.migrate.state (resume bookkeeping).

The migration pipeline stages source-platform data on disk as parquet
parts in a long/tall schema (one row per data point) plus JSON state
files that make both phases resumable. These tests pin the round-trip,
part rotation, and resume semantics.
"""

from __future__ import annotations

import pytest

pytest.importorskip('pyarrow')

from pluto.migrate.schema import (  # noqa: E402
    ATTRIBUTE_TYPES,
    PartWriter,
    iter_part_tables,
    part_files,
)
from pluto.migrate.state import (  # noqa: E402
    LoadedCache,
    is_run_exported,
    mark_run_exported,
    read_json,
    write_json_atomic,
)


def _metric_row(i: int) -> dict:
    return dict(
        project_id='acme/vision',
        run_id='abc123',
        attribute_path=f'loss/train_{i % 3}',
        attribute_type='metric',
        step=i,
        timestamp_ms=1600000000000 + i,
        float_value=float(i) / 7,
    )


class TestPartWriter:
    def test_round_trip_preserves_rows_and_order(self, tmp_path):
        with PartWriter(tmp_path) as w:
            for i in range(10):
                w.write_row(**_metric_row(i))
            w.write_row(
                project_id='acme/vision',
                run_id='abc123',
                attribute_path='sample',
                attribute_type='media',
                step=3,
                timestamp_ms=1600000000003,
                string_value='image-file',
                file_value='files/media/images/sample_3.png',
                caption='a caption',
            )

        rows = []
        for table in iter_part_tables(tmp_path):
            rows.extend(table.to_pylist())

        assert len(rows) == 11
        assert [r['step'] for r in rows[:10]] == list(range(10))
        assert rows[0]['float_value'] == pytest.approx(0.0)
        assert rows[0]['string_value'] is None
        media = rows[10]
        assert media['attribute_type'] == 'media'
        assert media['file_value'] == 'files/media/images/sample_3.png'
        assert media['caption'] == 'a caption'
        assert media['float_value'] is None

    def test_rotation_creates_multiple_ordered_parts(self, tmp_path):
        with PartWriter(tmp_path, max_part_bytes=1, rows_per_flush=10) as w:
            for i in range(35):
                w.write_row(**_metric_row(i))

        files = part_files(tmp_path)
        assert len(files) > 1
        assert files == sorted(files)

        steps = []
        for table in iter_part_tables(tmp_path):
            steps.extend(table.column('step').to_pylist())
        assert steps == list(range(35))

    def test_invalid_attribute_type_raises(self, tmp_path):
        with PartWriter(tmp_path) as w:
            with pytest.raises(ValueError, match='attribute_type'):
                w.write_row(**{**_metric_row(0), 'attribute_type': 'bogus'})

    def test_no_rows_writes_no_parts(self, tmp_path):
        with PartWriter(tmp_path):
            pass
        assert part_files(tmp_path) == []

    def test_attribute_types_cover_migration_scope(self):
        assert ATTRIBUTE_TYPES == {
            'metric',
            'system_metric',
            'media',
            'console',
            'artifact',
        }


class TestState:
    def test_write_json_atomic_round_trip_and_overwrite(self, tmp_path):
        path = tmp_path / 'x.json'
        write_json_atomic(path, {'a': 1})
        assert read_json(path) == {'a': 1}
        write_json_atomic(path, {'a': 2})
        assert read_json(path) == {'a': 2}

    def test_export_sentinel(self, tmp_path):
        run_dir = tmp_path / 'run1'
        run_dir.mkdir()
        assert not is_run_exported(run_dir)
        mark_run_exported(run_dir, {'rows': 42})
        assert is_run_exported(run_dir)

    def test_loaded_cache_persists_across_instances(self, tmp_path):
        path = tmp_path / 'loaded_runs.json'
        cache = LoadedCache(path)
        assert not cache.is_loaded('wandb::acme/vision/abc123')
        cache.mark_loaded('wandb::acme/vision/abc123', {'pluto_run_id': 7})

        reopened = LoadedCache(path)
        assert reopened.is_loaded('wandb::acme/vision/abc123')
        assert not reopened.is_loaded('wandb::acme/vision/other')
