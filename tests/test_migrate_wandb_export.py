"""
Unit tests for pluto.migrate.wandb_export.WandbExporter.

The exporter reads runs from the wandb cloud API and stages them on disk
(parquet parts + run.json + downloaded files). These tests drive it with
fake wandb API objects — no network, no real wandb — and pin the staged
layout: metric/media/system/console/artifact rows, original timestamps,
resume-by-sentinel, and the artifact size cap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip('pyarrow')

from pluto.migrate.schema import iter_part_tables  # noqa: E402
from pluto.migrate.state import is_run_exported, read_json  # noqa: E402
from pluto.migrate.wandb_export import WandbExporter  # noqa: E402

CREATED_AT = '2025-05-01T10:00:00Z'
CREATED_AT_MS = 1746093600000
T0 = 1746093601.0  # first history point, epoch seconds


class FakeSummary:
    def __init__(self, d):
        self._json_dict = d


class FakeFile:
    def __init__(self, name, size=10, content=b'x'):
        self.name = name
        self.size = size
        self._content = content

    def download(self, root, replace=False, exist_ok=False):
        path = Path(root) / self.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._content)
        return path


class FakeArtifact:
    def __init__(self, name='model-weights:v2', type='model', size=100):
        self.name = name
        self.type = type
        self.size = size
        self.created_at = CREATED_AT

    def download(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / 'model.pt').write_bytes(b'weights')
        return str(root)


class FakeRun:
    entity = 'acme'
    project = 'vision'

    def __init__(self, run_id='abc123', artifacts=None, output_log=None):
        self.id = run_id
        self.name = 'sunny-lion-1'
        self.notes = 'baseline run'
        self.tags = ['baseline']
        self.state = 'finished'
        self.config = {'lr': 0.1}
        self.summary = FakeSummary({'loss': 0.05, '_wandb': {'runtime': 12}})
        self.created_at = CREATED_AT
        self.heartbeat_at = '2025-05-01T12:00:00Z'
        self.url = f'https://wandb.ai/acme/vision/runs/{run_id}'
        self.metadata = {'gpu': 'NVIDIA H100'}
        self._artifacts = artifacts if artifacts is not None else [FakeArtifact()]
        self._files = [
            FakeFile(
                'output.log',
                content=output_log
                if output_log is not None
                else b'starting up\nepoch 0 done\n',
            ),
            FakeFile('media/images/sample_3_abc.png', content=b'PNG'),
            FakeFile('requirements.txt'),
        ]
        self.scan_history_calls = 0

    def scan_history(self, page_size=1000):
        self.scan_history_calls += 1
        return iter(
            [
                {'_step': 0, '_timestamp': T0, 'loss': 1.0, 'acc': 0.1},
                {'_step': 1, '_timestamp': T0 + 1, 'loss': 0.5},
                {
                    '_step': 3,
                    '_timestamp': T0 + 3,
                    'sample': {
                        '_type': 'image-file',
                        'path': 'media/images/sample_3_abc.png',
                        'caption': 'a dog',
                    },
                    'weights': {
                        '_type': 'histogram',
                        'values': [1, 2, 1],
                        'bins': [0, 1, 2, 3],
                    },
                },
            ]
        )

    def history(self, stream='default', pandas=True, samples=None):
        assert stream == 'events' and pandas is False
        return [
            {'_timestamp': T0, 'system.gpu.0.gpu': 55.0, 'system.cpu': 12.0},
            {'_timestamp': T0 + 2, 'system.gpu.0.gpu': 60.0},
        ]

    def files(self):
        return list(self._files)

    def logged_artifacts(self):
        return list(self._artifacts)


class FakeApi:
    def __init__(self, runs):
        self._runs = runs

    def runs(self, path, filters=None):
        assert path == 'acme/vision'
        return list(self._runs)


def _export(tmp_path, run=None, **kwargs):
    run = run or FakeRun()
    exporter = WandbExporter(
        entity='acme',
        project='vision',
        output_dir=tmp_path,
        api=FakeApi([run]),
        **kwargs,
    )
    summary = exporter.export()
    return run, tmp_path / 'acme' / 'vision' / 'runs' / run.id, summary


def _rows(run_dir, attribute_type=None):
    rows = []
    for table in iter_part_tables(run_dir):
        rows.extend(table.to_pylist())
    if attribute_type:
        rows = [r for r in rows if r['attribute_type'] == attribute_type]
    return rows


class TestWandbExporter:
    def test_run_json_manifest(self, tmp_path):
        run, run_dir, summary = _export(tmp_path)
        assert summary == {'exported': 1, 'skipped': 0, 'failed': []}
        manifest = read_json(run_dir / 'run.json')
        assert manifest['name'] == 'sunny-lion-1'
        assert manifest['notes'] == 'baseline run'
        assert manifest['tags'] == ['baseline']
        assert manifest['state'] == 'finished'
        assert manifest['config'] == {'lr': 0.1}
        assert manifest['summary'] == {'loss': 0.05}  # _wandb internals dropped
        assert manifest['createdAt'] == CREATED_AT_MS
        assert manifest['updatedAt'] == CREATED_AT_MS + 2 * 3600 * 1000
        assert manifest['url'] == run.url
        assert is_run_exported(run_dir)

    def test_metric_rows_preserve_step_and_timestamp(self, tmp_path):
        _, run_dir, _ = _export(tmp_path)
        metrics = _rows(run_dir, 'metric')
        assert {
            'attribute_path': 'loss',
            'step': 0,
            'timestamp_ms': int(T0 * 1000),
            'float_value': 1.0,
        }.items() <= metrics[0].items()
        assert [m['attribute_path'] for m in metrics] == ['loss', 'acc', 'loss']

    def test_media_and_histogram_rows(self, tmp_path):
        _, run_dir, _ = _export(tmp_path)
        media = {m['attribute_path']: m for m in _rows(run_dir, 'media')}
        img = media['sample']
        assert img['string_value'] == 'image-file'
        assert img['file_value'] == 'files/media/images/sample_3_abc.png'
        assert img['caption'] == 'a dog'
        assert img['step'] == 3
        assert (run_dir / 'files/media/images/sample_3_abc.png').exists()
        hist = media['weights']
        assert hist['file_value'] is None
        assert json.loads(hist['string_value']) == {
            '_type': 'histogram',
            'values': [1, 2, 1],
            'bins': [0, 1, 2, 3],
        }

    def test_system_metric_rows_keep_source_names(self, tmp_path):
        # Staging is source-faithful; the loader owns the sys/ translation.
        _, run_dir, _ = _export(tmp_path)
        sys_rows = _rows(run_dir, 'system_metric')
        assert {
            'attribute_path': 'system.gpu.0.gpu',
            'step': 0,
            'timestamp_ms': int(T0 * 1000),
            'float_value': 55.0,
        }.items() <= sys_rows[0].items()
        assert {r['attribute_path'] for r in sys_rows} == {
            'system.gpu.0.gpu',
            'system.cpu',
        }

    def test_console_rows_from_output_log(self, tmp_path):
        _, run_dir, _ = _export(tmp_path)
        console = _rows(run_dir, 'console')
        assert [(r['string_value'], r['step']) for r in console] == [
            ('starting up', 1),
            ('epoch 0 done', 2),
        ]
        # no per-line timestamps in the log -> stamped with run createdAt
        assert console[0]['timestamp_ms'] == CREATED_AT_MS

    def test_console_lines_with_timestamps_are_parsed(self, tmp_path):
        log = b'2025-05-01T10:00:05.500Z first line\nplain line\n'
        run = FakeRun(run_id='tsrun', output_log=log)
        _, run_dir, _ = _export(tmp_path, run=run)
        console = _rows(run_dir, 'console')
        # Timestamp parsed for the row time, but the message content is
        # preserved verbatim — user log lines must not be rewritten.
        assert console[0]['string_value'] == '2025-05-01T10:00:05.500Z first line'
        assert console[0]['timestamp_ms'] == CREATED_AT_MS + 5500
        assert console[1]['string_value'] == 'plain line'
        assert console[1]['timestamp_ms'] == CREATED_AT_MS

    def test_artifact_rows_and_download(self, tmp_path):
        _, run_dir, _ = _export(tmp_path)
        rows = _rows(run_dir, 'artifact')
        assert len(rows) == 1
        row = rows[0]
        assert row['attribute_path'] == 'model-weights:v2'
        assert row['file_value'] == 'artifacts/model-weights:v2/model.pt'
        assert row['timestamp_ms'] == CREATED_AT_MS
        meta = json.loads(row['string_value'])
        assert meta['type'] == 'model'
        assert (run_dir / 'artifacts/model-weights:v2/model.pt').exists()

    def test_artifact_size_cap_skips_download(self, tmp_path):
        big = FakeArtifact(name='huge:v0', size=10**12)
        run = FakeRun(run_id='bigrun', artifacts=[big])
        _, run_dir, _ = _export(tmp_path, run=run, artifact_max_bytes=10**6)
        assert _rows(run_dir, 'artifact') == []
        assert not (run_dir / 'artifacts').exists()

    def test_resume_skips_completed_runs(self, tmp_path):
        run, run_dir, _ = _export(tmp_path)
        assert run.scan_history_calls == 1
        exporter = WandbExporter(
            entity='acme', project='vision', output_dir=tmp_path, api=FakeApi([run])
        )
        summary = exporter.export()
        assert summary == {'exported': 0, 'skipped': 1, 'failed': []}
        assert run.scan_history_calls == 1  # untouched on resume

    def test_run_failure_is_recorded_not_raised(self, tmp_path):
        run = FakeRun(run_id='boom')
        run.scan_history = lambda page_size=1000: (_ for _ in ()).throw(
            RuntimeError('api exploded')
        )
        exporter = WandbExporter(
            entity='acme', project='vision', output_dir=tmp_path, api=FakeApi([run])
        )
        summary = exporter.export()
        assert summary['exported'] == 0
        assert summary['failed'] and summary['failed'][0]['run_id'] == 'boom'
        assert not is_run_exported(tmp_path / 'acme' / 'vision' / 'runs' / 'boom')
        manifest = read_json(tmp_path / 'manifest.json')
        assert manifest['failed'][0]['run_id'] == 'boom'

    def test_run_ids_filter(self, tmp_path):
        wanted, unwanted = FakeRun(run_id='keep'), FakeRun(run_id='drop')
        exporter = WandbExporter(
            entity='acme',
            project='vision',
            output_dir=tmp_path,
            api=FakeApi([wanted, unwanted]),
            run_ids=['keep'],
        )
        summary = exporter.export()
        assert summary['exported'] == 1
        assert (tmp_path / 'acme/vision/runs/keep').exists()
        assert not (tmp_path / 'acme/vision/runs/drop').exists()
