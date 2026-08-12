"""
Unit tests for pluto.migrate.loader.PlutoLoader.

The loader replays staged export dirs into Pluto through the public
client API. pluto.init is mocked; these tests pin the init kwargs
(external id, compat createdAt, import tag, host-pollution guards), the
per-step metric replay with original timestamps, media/histogram
conversion, console/artifact replay, finish-code mapping, dedup, and
resume caching.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

pytest.importorskip('pyarrow')

import pluto  # noqa: E402
from pluto.migrate.loader import PlutoLoader  # noqa: E402
from pluto.migrate.schema import PartWriter  # noqa: E402
from pluto.migrate.state import (  # noqa: E402
    LoadedCache,
    mark_run_exported,
    read_json,
    write_json_atomic,
)

CREATED_AT_MS = 1746093600000
UPDATED_AT_MS = CREATED_AT_MS + 7200000
T0_MS = 1746093601000
EXTERNAL_ID = 'wandb::acme/vision/abc123'
# The load cache is keyed by (external id, destination project); _stage_run
# uses project 'vision' and tests don't override dest_project.
CACHE_KEY = f'{EXTERNAL_ID}@@vision'


def _stage_run(tmp_path, run_id='abc123', state='finished'):
    run_dir = tmp_path / 'acme' / 'vision' / 'runs' / run_id
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / 'run.json',
        {
            'entity': 'acme',
            'project': 'vision',
            'run_id': run_id,
            'name': 'sunny-lion-1',
            'notes': 'baseline run',
            'tags': ['baseline'],
            'state': state,
            'config': {'lr': 0.1},
            'summary': {'loss': 0.05},
            'createdAt': CREATED_AT_MS,
            'updatedAt': UPDATED_AT_MS,
            'url': f'https://wandb.ai/acme/vision/runs/{run_id}',
        },
    )
    media_file = run_dir / 'files' / 'media' / 'images' / 'sample_3.png'
    media_file.parent.mkdir(parents=True)
    media_file.write_bytes(b'PNG')
    artifact_file = run_dir / 'artifacts' / 'model-weights:v2' / 'model.pt'
    artifact_file.parent.mkdir(parents=True)
    artifact_file.write_bytes(b'weights')

    base = dict(project_id='acme/vision', run_id=run_id)
    with PartWriter(run_dir) as w:
        w.write_row(
            **base,
            attribute_path='loss',
            attribute_type='metric',
            step=0,
            timestamp_ms=T0_MS,
            float_value=1.0,
        )
        w.write_row(
            **base,
            attribute_path='acc',
            attribute_type='metric',
            step=0,
            timestamp_ms=T0_MS,
            float_value=0.1,
        )
        w.write_row(
            **base,
            attribute_path='loss',
            attribute_type='metric',
            step=1,
            timestamp_ms=T0_MS + 1000,
            float_value=0.5,
        )
        w.write_row(
            **base,
            attribute_path='sample',
            attribute_type='media',
            step=3,
            timestamp_ms=T0_MS + 3000,
            string_value='image-file',
            file_value='files/media/images/sample_3.png',
            caption='a dog',
        )
        w.write_row(
            **base,
            attribute_path='weights',
            attribute_type='media',
            step=3,
            timestamp_ms=T0_MS + 3000,
            string_value=json.dumps(
                {'_type': 'histogram', 'values': [1, 2, 1], 'bins': [0, 1, 2, 3]}
            ),
        )
        w.write_row(
            **base,
            attribute_path='system.gpu.0.gpu',
            attribute_type='system_metric',
            step=0,
            timestamp_ms=T0_MS,
            float_value=55.0,
        )
        w.write_row(
            **base,
            attribute_path='console',
            attribute_type='console',
            step=1,
            timestamp_ms=T0_MS,
            string_value='starting up',
        )
        w.write_row(
            **base,
            attribute_path='model-weights:v2',
            attribute_type='artifact',
            step=0,
            timestamp_ms=CREATED_AT_MS,
            string_value=json.dumps({'name': 'model-weights:v2', 'type': 'model'}),
            file_value='artifacts/model-weights:v2/model.pt',
        )
    mark_run_exported(run_dir, {'rows': 8})
    return run_dir


def _stage_media_table_run(tmp_path, run_id='mtable'):
    """Stage a run with a wandb media table (an image column).

    Mirrors what the exporter actually stages today: the ``table-file`` row
    points at wandb's *run-files* copy of the table, where media cells are
    collapsed to the literal string ``"Image"`` (the full-fidelity
    ``{_type: image-file, path, sha256}`` refs live only in the separate
    artifact copy). The cell images are staged as their own artifact.
    """
    run_dir = tmp_path / 'acme' / 'vision' / 'runs' / run_id
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / 'run.json',
        {
            'entity': 'acme',
            'project': 'vision',
            'run_id': run_id,
            'name': 'rich-table-1',
            'notes': '',
            'tags': [],
            'state': 'finished',
            'config': {},
            'summary': {},
            'createdAt': CREATED_AT_MS,
            'updatedAt': UPDATED_AT_MS,
            'url': f'https://wandb.ai/acme/vision/runs/{run_id}',
        },
    )
    # wandb's lossy run-files table copy: image cells are the string "Image".
    table_rel = 'files/media/table/media_table.table.json'
    table_file = run_dir / table_rel
    table_file.parent.mkdir(parents=True)
    write_json_atomic(
        table_file,
        {
            'columns': ['idx', 'img', 'score'],
            'data': [[0, 'Image', 0.0], [1, 'Image', 0.25]],
        },
    )
    # the cell images arrive separately, as the auto-created run-table artifact.
    img_rel = 'artifacts/run-mtable-media_table:v0/media/images/a.png'
    img_file = run_dir / img_rel
    img_file.parent.mkdir(parents=True)
    img_file.write_bytes(b'PNG')

    base = dict(project_id='acme/vision', run_id=run_id)
    with PartWriter(run_dir) as w:
        w.write_row(
            **base,
            attribute_path='media_table',
            attribute_type='media',
            step=0,
            timestamp_ms=T0_MS,
            string_value='table-file',
            file_value=table_rel,
        )
        w.write_row(
            **base,
            attribute_path='run-mtable-media_table:v0',
            attribute_type='artifact',
            step=0,
            timestamp_ms=CREATED_AT_MS,
            string_value=json.dumps(
                {'name': 'run-mtable-media_table:v0', 'type': 'run_table'}
            ),
            file_value=img_rel,
        )
    mark_run_exported(run_dir, {'rows': 2})
    return run_dir


def _stage_string_series_run(tmp_path, values, run_id='ssrun', name='phase'):
    """Stage a minimal run whose only history is one string_series attribute."""
    run_dir = tmp_path / 'acme' / 'vision' / 'runs' / run_id
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / 'run.json',
        {
            'entity': 'acme',
            'project': 'vision',
            'run_id': run_id,
            'name': 'sunny-lion-1',
            'state': 'finished',
            'config': {},
            'summary': {name: values[-1]},
            'createdAt': CREATED_AT_MS,
            'updatedAt': UPDATED_AT_MS,
            'url': f'https://wandb.ai/acme/vision/runs/{run_id}',
        },
    )
    base = dict(project_id='acme/vision', run_id=run_id)
    with PartWriter(run_dir) as w:
        for i, v in enumerate(values):
            w.write_row(
                **base,
                attribute_path=name,
                attribute_type='string_series',
                step=i,
                timestamp_ms=T0_MS + i * 1000,
                string_value=v,
            )
    mark_run_exported(run_dir, {'rows': len(values)})
    return run_dir


@pytest.fixture
def mock_init():
    with mock.patch('pluto.init') as init:
        op = mock.MagicMock()
        op.settings._op_id = 42
        op._sync_manager.get_pending_count.return_value = 0
        # A real Op sets this to None on a confirmed finish; without it a bare
        # MagicMock auto-vivifies a truthy attribute and looks like a failed
        # status update to the loader.
        op._status_update_error = None
        init.return_value = op
        yield init, op


def _log_calls_with(op, predicate):
    return [c for c in op.log.call_args_list if predicate(c.args[0])]


class TestPlutoLoader:
    def test_init_kwargs(self, tmp_path, mock_init):
        init, op = mock_init
        _stage_run(tmp_path)
        summary = PlutoLoader(tmp_path).load()
        assert summary == {'loaded': 1, 'skipped': 0, 'failed': []}
        kwargs = init.call_args.kwargs
        assert kwargs['project'] == 'vision'
        assert kwargs['name'] == 'sunny-lion-1'
        assert kwargs['config'] == {'lr': 0.1}
        assert kwargs['tags'] == ['baseline', 'import:wandb']
        assert kwargs['run_id'] == EXTERNAL_ID
        settings = kwargs['settings']
        assert settings['compat'] == {
            'createdAt': CREATED_AT_MS,
            'updatedAt': UPDATED_AT_MS,
        }
        assert settings['disable_console'] is True
        assert settings['disable_system_metrics'] is True
        # The historical-timestamp path only exists in the sync store.
        assert settings['sync_process_enabled'] is True

    def test_wandb_scalars_pushed_via_update_config(self, tmp_path, mock_init):
        _, op = mock_init
        _stage_run(tmp_path)
        PlutoLoader(tmp_path).load()
        wandb_block = op.update_config.call_args.args[0]['wandb']
        assert wandb_block['notes'] == 'baseline run'
        assert wandb_block['state'] == 'finished'
        assert wandb_block['summary'] == {'loss': 0.05}

    def test_custom_charts_forwarded_to_wandb_config(self, tmp_path, mock_init):
        _, op = mock_init
        run_dir = _stage_run(tmp_path)
        panels = [
            {
                'key': 'bar',
                'preset': 'bar',
                'title': 'per-class',
                'tableKey': 'bar_table',
                'fields': {'label': 'label', 'value': 'value'},
                'specLang': 'vega-lite',
            }
        ]
        write_json_atomic(run_dir / 'custom_charts.json', {'panels': panels})
        PlutoLoader(tmp_path).load()
        wandb_block = op.update_config.call_args.args[0]['wandb']
        assert wandb_block['custom_charts'] == panels

    def test_image_annotations_resolves_boxes_and_masks(self, tmp_path):
        # Boxes: the .boxes2D.json is inlined into the annotations JSON. Masks:
        # the .mask.png is resolved to a path spec for pluto.Image to re-upload.
        run_dir = tmp_path / 'run'
        box_dir = run_dir / 'files' / 'media' / 'metadata' / 'boxes2D'
        box_dir.mkdir(parents=True)
        box_content = {'box_data': [{'class_id': 1}], 'class_labels': {'1': 'cat'}}
        (box_dir / 'a.boxes2D.json').write_text(json.dumps(box_content))
        mask_dir = run_dir / 'files' / 'media' / 'images' / 'mask'
        mask_dir.mkdir(parents=True)
        (mask_dir / 'a.mask.png').write_bytes(b'PNG')
        annotation_value = json.dumps(
            {
                'boxes': {'pred': {'path': 'media/metadata/boxes2D/a.boxes2D.json'}},
                'masks': {'pred': {'path': 'media/images/mask/a.mask.png'}},
            }
        )
        boxes_str, masks_spec = PlutoLoader(tmp_path)._image_annotations(
            run_dir, annotation_value
        )
        assert json.loads(boxes_str) == {'boxes': {'pred': box_content}}
        assert masks_spec['pred']['path'].endswith('media/images/mask/a.mask.png')

    def test_mask_class_labels_preserved(self, tmp_path):
        # A staged mask carrying class_labels (folded in by the exporter) keeps
        # them in the mask spec, so pluto.Image re-uploads a coloured mask
        # instead of a blank one.
        run_dir = tmp_path / 'run'
        mask_dir = run_dir / 'files' / 'media' / 'images' / 'mask'
        mask_dir.mkdir(parents=True)
        (mask_dir / 'a.mask.png').write_bytes(b'PNG')
        annotation_value = json.dumps(
            {
                'masks': {
                    'pred': {
                        'path': 'media/images/mask/a.mask.png',
                        'class_labels': {'0': 'bg', '1': 'cat'},
                    }
                }
            }
        )
        _, masks_spec = PlutoLoader(tmp_path)._image_annotations(
            run_dir, annotation_value
        )
        assert masks_spec['pred']['class_labels'] == {'0': 'bg', '1': 'cat'}
        assert masks_spec['pred']['path'].endswith('a.mask.png')

    def test_image_annotations_none_when_absent(self, tmp_path):
        assert PlutoLoader(tmp_path)._image_annotations(tmp_path, None) == (None, None)

    def test_sweep_membership_migrated(self, tmp_path, mock_init):
        init, op = mock_init
        run_dir = _stage_run(tmp_path)
        # inject a sweep block into the staged manifest (as the exporter would)
        manifest = read_json(run_dir / 'run.json')
        manifest['sweep'] = {
            'id': 'swp1',
            'name': 'my-sweep',
            'config': {'method': 'grid', 'parameters': {'lr': {'values': [0.1]}}},
        }
        write_json_atomic(run_dir / 'run.json', manifest)
        PlutoLoader(tmp_path).load()
        # run is tagged so it groups under its sweep (like native pluto.sweep)
        assert 'sweep:swp1' in init.call_args.kwargs['tags']
        # search space survives in the wandb config block
        wandb_block = op.update_config.call_args.args[0]['wandb']
        assert wandb_block['sweep']['id'] == 'swp1'
        assert wandb_block['sweep']['config']['method'] == 'grid'

    def test_no_custom_charts_key_when_absent(self, tmp_path, mock_init):
        _, op = mock_init
        _stage_run(tmp_path)
        PlutoLoader(tmp_path).load()
        wandb_block = op.update_config.call_args.args[0]['wandb']
        assert 'custom_charts' not in wandb_block

    def test_metrics_batched_per_step_with_timestamps(self, tmp_path, mock_init):
        _, op = mock_init
        _stage_run(tmp_path)
        PlutoLoader(tmp_path).load()
        op._log_metrics_batch.assert_called_once()
        groups = op._log_metrics_batch.call_args.args[0]
        assert groups[0] == ({'loss': 1.0, 'acc': 0.1}, 0, T0_MS / 1000)
        assert groups[1] == ({'loss': 0.5}, 1, (T0_MS + 1000) / 1000)

    def test_media_converted_to_pluto_types(self, tmp_path, mock_init):
        _, op = mock_init
        _stage_run(tmp_path)
        PlutoLoader(tmp_path).load()
        image_calls = _log_calls_with(
            op, lambda d: any(isinstance(v, pluto.Image) for v in d.values())
        )
        assert len(image_calls) == 1
        call = image_calls[0]
        assert list(call.args[0]) == ['sample']
        assert call.args[0]['sample']._caption == 'a dog'
        assert call.kwargs == {'step': 3, 'timestamp': (T0_MS + 3000) / 1000}

        hist_calls = _log_calls_with(
            op, lambda d: any(isinstance(v, pluto.Histogram) for v in d.values())
        )
        assert len(hist_calls) == 1
        hist = hist_calls[0].args[0]['weights']
        assert hist._freq == [1, 2, 1]
        assert hist._bins == [0, 1, 2, 3]

    def test_media_in_table_migrates_degraded_not_dropped(self, tmp_path, mock_init):
        # Characterization test for the known media-in-table gap. wandb keeps
        # two copies of a media table; the exporter stages the lossy run-files
        # copy, so image cells load as the literal text "Image". The table is
        # NOT dropped and does NOT crash, and the cell images arrive as a
        # separate, unlinked artifact. When we wire cells to real images this
        # test should flip to assert image cells.
        _, op = mock_init
        _stage_media_table_run(tmp_path)
        summary = PlutoLoader(tmp_path).load()
        assert summary == {'loaded': 1, 'skipped': 0, 'failed': []}

        table_calls = _log_calls_with(
            op, lambda d: any(isinstance(v, pluto.Table) for v in d.values())
        )
        assert len(table_calls) == 1
        table = table_calls[0].args[0]['media_table']
        assert table._col == ['idx', 'img', 'score']
        # image column (index 1) is the degraded literal string, not a picture
        assert [row[1] for row in table._table] == ['Image', 'Image']

        # the cell images still migrate, but as a disconnected artifact
        art_calls = _log_calls_with(
            op, lambda d: any(isinstance(v, pluto.Artifact) for v in d.values())
        )
        assert len(art_calls) == 1

    def test_system_metrics_translated_to_sys_names(self, tmp_path, mock_init):
        # Staged rows keep wandb's source-native 'system.*' names; the
        # loader owns the translation to Pluto's 'sys/' namespace.
        _, op = mock_init
        _stage_run(tmp_path)
        PlutoLoader(tmp_path).load()
        groups = op._log_metrics_batch.call_args.args[0]
        assert ({'sys/gpu.0.gpu': 55.0}, 0, T0_MS / 1000) in groups

    def test_console_replayed_with_timestamps(self, tmp_path, mock_init):
        _, op = mock_init
        _stage_run(tmp_path)
        PlutoLoader(tmp_path).load()
        op._log_console.assert_called_once_with(
            [('starting up', 'INFO', T0_MS / 1000, 1)]
        )

    def test_artifacts_replayed(self, tmp_path, mock_init):
        _, op = mock_init
        _stage_run(tmp_path)
        PlutoLoader(tmp_path).load()
        art_calls = _log_calls_with(
            op, lambda d: any(isinstance(v, pluto.Artifact) for v in d.values())
        )
        assert len(art_calls) == 1
        assert art_calls[0].kwargs['step'] == 0

    def test_string_series_sent_to_ingest_with_data_logtype(self, tmp_path, mock_init):
        _, op = mock_init
        op.settings.url_data = 'http://ingest/data'
        op.settings.url_meta = 'http://api/logName/add'
        _stage_string_series_run(tmp_path, values=['warmup', 'train', 'train', 'eval'])
        with mock.patch('pluto.migrate.loader.httpx') as httpx_mock:
            PlutoLoader(tmp_path).load()
        calls = httpx_mock.post.call_args_list
        assert len(calls) == 2
        # 1st POST registers the log name with logType DATA.
        meta = json.loads(calls[0].kwargs['content'])
        assert meta['logType'] == 'DATA'
        assert meta['logName'] == ['phase']
        # 2nd POST ingests the points as NDJSON string-series (raw data).
        lines = [json.loads(x) for x in calls[1].kwargs['content'].strip().split('\n')]
        assert [x['step'] for x in lines] == [0, 1, 2, 3]
        assert [x['data'] for x in lines] == ['warmup', 'train', 'train', 'eval']
        assert all(
            x['dataType'] == 'string-series' and x['logName'] == 'phase' for x in lines
        )

    def test_string_series_high_cardinality_still_sent(self, tmp_path, mock_init):
        _, op = mock_init
        op.settings.url_data = 'http://ingest/data'
        op.settings.url_meta = 'http://api/logName/add'
        # No cardinality guard: even an all-distinct series is sent in full
        # (no data loss).
        _stage_string_series_run(tmp_path, values=[f'v{i}' for i in range(60)])
        with mock.patch('pluto.migrate.loader.httpx') as httpx_mock:
            PlutoLoader(tmp_path).load()
        calls = httpx_mock.post.call_args_list
        assert len(calls) == 2  # logName/add + ingest
        lines = [json.loads(x) for x in calls[1].kwargs['content'].strip().split('\n')]
        assert len(lines) == 60  # every point sent

    def test_finish_code_mapping(self, tmp_path, mock_init):
        _, op = mock_init
        _stage_run(tmp_path, run_id='crashed1', state='crashed')
        PlutoLoader(tmp_path).load()
        op.finish.assert_called_once_with(code=1)

    def test_loaded_cache_written_and_resume_skips(self, tmp_path, mock_init):
        init, op = mock_init
        _stage_run(tmp_path)
        PlutoLoader(tmp_path).load()
        assert LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(CACHE_KEY)

        init.reset_mock()
        summary = PlutoLoader(tmp_path).load()
        assert summary == {'loaded': 0, 'skipped': 1, 'failed': []}
        init.assert_not_called()

    def _stage_min_run(self, tmp_path, project, run_id):
        d = tmp_path / 'acme' / project / 'runs' / run_id
        d.mkdir(parents=True)
        write_json_atomic(
            d / 'run.json',
            {
                'entity': 'acme',
                'project': project,
                'run_id': run_id,
                'name': run_id,
                'state': 'finished',
            },
        )
        with PartWriter(d) as w:
            w.write_row(
                project_id=f'acme/{project}',
                run_id=run_id,
                attribute_path='loss',
                attribute_type='metric',
                step=0,
                timestamp_ms=T0_MS,
                float_value=1.0,
            )
        mark_run_exported(d, {'rows': 1})

    def test_discover_respects_project_scope(self, tmp_path):
        self._stage_min_run(tmp_path, 'vision', 'v1')
        self._stage_min_run(tmp_path, 'audio', 'a1')
        self._stage_min_run(tmp_path, 'text', 't1')
        # include only
        got = {
            d.parent.parent.name
            for d in PlutoLoader(
                tmp_path, projects=['vision', 'audio']
            )._discover_runs()
        }
        assert got == {'vision', 'audio'}
        # exclude
        got2 = {
            d.parent.parent.name
            for d in PlutoLoader(tmp_path, exclude_projects=['vision'])._discover_runs()
        }
        assert got2 == {'audio', 'text'}
        # default = all
        got3 = {d.parent.parent.name for d in PlutoLoader(tmp_path)._discover_runs()}
        assert got3 == {'vision', 'audio', 'text'}

    def test_custom_cache_path(self, tmp_path, mock_init):
        _stage_run(tmp_path)
        custom = tmp_path / 'ledger-vision.json'
        PlutoLoader(tmp_path, cache_path=custom).load()
        assert custom.exists()  # resume ledger went to the custom path
        assert not (tmp_path / 'loaded_runs.json').exists()

    def test_external_id_collision_restores_and_skips_by_default(
        self, tmp_path, mock_init
    ):
        # A collision means the run already exists server-side but isn't in the
        # local cache. Re-replaying would duplicate media, so the loader skips
        # the replay. BUT the create-with-existing already reopened the run to
        # RUNNING (DDP-style resume), so it re-attaches (resume) and finish()es
        # to restore the terminal status + historical timestamp — without
        # replaying — then skips.
        from pluto.op import RunExistsError

        init, _ = mock_init
        restore_op = mock.MagicMock()
        restore_op._status_update_error = None  # finish() confirmed the status
        init.side_effect = [
            RunExistsError(
                "Run with externalId 'wandb::acme/vision/abc123' already exists."
            ),
            restore_op,  # the resume re-attach used to restore terminal status
        ]
        _stage_run(tmp_path)
        summary = PlutoLoader(tmp_path).load()
        assert summary == {'loaded': 0, 'skipped': 1, 'failed': []}
        assert init.call_count == 2  # plain init (raises) + resume to restore
        restore_op.finish.assert_called_once()  # restored to terminal status
        restore_op._log_metrics_batch.assert_not_called()  # NOT re-replayed
        assert LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(CACHE_KEY)

    def test_external_id_collision_replays_with_force_resume(self, tmp_path, mock_init):
        # force_resume opts into healing a genuinely interrupted load: resume
        # the existing run and re-replay (accepting possible media duplication).
        from pluto.op import RunExistsError

        init, op = mock_init
        init.side_effect = [
            RunExistsError(
                "Run with externalId 'wandb::acme/vision/abc123' already exists."
            ),
            op,
        ]
        _stage_run(tmp_path)
        summary = PlutoLoader(tmp_path, force_resume=True).load()
        assert summary == {'loaded': 1, 'skipped': 0, 'failed': []}
        assert init.call_count == 2
        assert init.call_args_list[1].kwargs['resume'] is True
        op._log_metrics_batch.assert_called_once()  # actually re-replayed
        assert LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(CACHE_KEY)

    def test_interrupted_run_resumes_and_completes_on_rerun(self, tmp_path, mock_init):
        # A run created server-side that then crashes mid-replay is marked
        # in_progress (not done). On a plain re-run the loader must recognize it
        # started that run and resume to COMPLETE it — not skip it as though it
        # were fully loaded elsewhere.
        from pluto.op import RunExistsError

        init, op = mock_init
        _stage_run(tmp_path)
        # 1st init succeeds (run created); 2nd re-run init collides; 3rd is the
        # resume that completes it.
        init.side_effect = [op, RunExistsError('exists'), op]
        with mock.patch.object(
            PlutoLoader,
            '_replay_run',
            side_effect=[RuntimeError('crash mid-replay'), None],
        ):
            s1 = PlutoLoader(tmp_path).load()  # crashes mid-replay
            assert s1['loaded'] == 0 and len(s1['failed']) == 1
            cache = LoadedCache(tmp_path / 'loaded_runs.json')
            assert cache.is_in_progress(CACHE_KEY)  # marker persisted
            assert not cache.is_loaded(CACHE_KEY)

            s2 = PlutoLoader(tmp_path).load()  # re-run resumes + completes
        assert s2 == {'loaded': 1, 'skipped': 0, 'failed': []}
        assert init.call_args_list[-1].kwargs['resume'] is True  # healed via resume
        assert LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(CACHE_KEY)

    def test_collision_restore_failure_reported_not_marked_loaded(
        self, tmp_path, mock_init
    ):
        # Collision restore path: if re-attaching to finish() the existing run
        # throws, the run is still RUNNING server-side with no terminal status.
        # It must NOT be marked loaded (that would strand it RUNNING and skip it
        # forever) — it's reported failed so a later run retries it.
        from pluto.op import RunExistsError

        init, _ = mock_init
        restore_op = mock.MagicMock()
        restore_op.finish.side_effect = RuntimeError('server rejected finish')
        init.side_effect = [
            RunExistsError(
                "Run with externalId 'wandb::acme/vision/abc123' already exists."
            ),
            restore_op,  # resume re-attach; its finish() blows up below
        ]
        _stage_run(tmp_path)
        summary = PlutoLoader(tmp_path).load()
        assert summary['loaded'] == 0 and summary['skipped'] == 0
        assert [f['run_id'] for f in summary['failed']] == ['abc123']
        assert 'restore failed' in summary['failed'][0]['error']
        # Crucially: NOT marked loaded, so a re-run gets another shot.
        assert not LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(CACHE_KEY)

    def test_unconfirmed_terminal_status_reported_not_marked_loaded(
        self, tmp_path, mock_init
    ):
        # finish() replayed everything but could NOT confirm the run's terminal
        # status on the server (a dropped connection that outlasted retries, now
        # surfaced via op._status_update_error). The run must be recorded as
        # failed and NOT cached as loaded — else a re-run skips it and it stays
        # stranded RUNNING/FAILED forever (the real-world bug this guards).
        _, op = mock_init
        op._status_update_error = ConnectionResetError('peer reset during finish')
        _stage_run(tmp_path)
        summary = PlutoLoader(tmp_path).load()
        assert summary['loaded'] == 0 and summary['skipped'] == 0
        assert [f['run_id'] for f in summary['failed']] == ['abc123']
        assert 'status unconfirmed' in summary['failed'][0]['error']
        assert not LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(CACHE_KEY)

    def test_restore_unconfirmed_terminal_status_reported_not_marked_loaded(
        self, tmp_path, mock_init
    ):
        # Same guarantee on the collision→restore path: if the restoring
        # finish() couldn't confirm the terminal status, don't mark it loaded.
        from pluto.op import RunExistsError

        init, _ = mock_init
        restore_op = mock.MagicMock()
        restore_op._status_update_error = ConnectionResetError('reset during restore')
        init.side_effect = [
            RunExistsError(
                "Run with externalId 'wandb::acme/vision/abc123' already exists."
            ),
            restore_op,
        ]
        _stage_run(tmp_path)
        summary = PlutoLoader(tmp_path).load()
        assert summary['loaded'] == 0 and summary['skipped'] == 0
        assert [f['run_id'] for f in summary['failed']] == ['abc123']
        assert not LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(CACHE_KEY)

    def test_stale_reap_healed_via_resume(self, tmp_path, mock_init):
        # A finished import comes back FAILED (a stale-run reap silently rejected
        # our COMPLETED). The loader must detect it via read-back and heal it with
        # a resume+finish, ending up loaded — not stranded.
        init, op = mock_init
        _stage_run(tmp_path)
        with mock.patch.object(
            PlutoLoader, '_read_run_status', side_effect=['FAILED', 'COMPLETED']
        ):
            summary = PlutoLoader(tmp_path).load()
        assert summary == {'loaded': 1, 'skipped': 0, 'failed': []}
        # the heal reopened the run: a second init with resume=True
        assert any(c.kwargs.get('resume') is True for c in init.call_args_list)
        assert LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(CACHE_KEY)

    def test_stale_reap_unhealable_reported_not_loaded(self, tmp_path, mock_init):
        # If the run stays FAILED after the bounded heal attempts, it is recorded
        # failed (a later pass retries) and NOT cached as loaded.
        _, op = mock_init
        _stage_run(tmp_path)
        with mock.patch.object(PlutoLoader, '_read_run_status', return_value='FAILED'):
            summary = PlutoLoader(tmp_path).load()
        assert summary['loaded'] == 0
        assert [f['run_id'] for f in summary['failed']] == ['abc123']
        assert 'not confirmed' in summary['failed'][0]['error']
        assert not LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(CACHE_KEY)

    def test_unverifiable_status_proceeds_best_effort(self, tmp_path, mock_init):
        # If we can't read the status back at all (endpoint down / network),
        # don't fail a run over our own inability to verify — proceed loaded.
        _, op = mock_init
        _stage_run(tmp_path)
        with mock.patch.object(PlutoLoader, '_read_run_status', return_value=None):
            summary = PlutoLoader(tmp_path).load()
        assert summary == {'loaded': 1, 'skipped': 0, 'failed': []}

    def test_wandb_failed_run_not_status_verified(self, tmp_path, mock_init):
        # A run that failed on wandb is intended FAILED; a FAILED status is
        # correct, so the loader must not try to "heal" it to COMPLETED — the
        # read-back is skipped entirely for code=1.
        _, op = mock_init
        _stage_run(tmp_path, run_id='crashed1', state='crashed')
        with mock.patch.object(
            PlutoLoader, '_read_run_status', return_value='FAILED'
        ) as read:
            summary = PlutoLoader(tmp_path).load()
        assert summary['loaded'] == 1
        op.finish.assert_called_once_with(code=1)
        read.assert_not_called()

    def test_read_run_status_parses_and_degrades(self, tmp_path):
        loader = PlutoLoader(tmp_path)
        op = mock.MagicMock()
        op.settings._op_id = 7
        op.settings.url_api = 'http://api'
        op.settings._auth = 'tok'
        ok = mock.MagicMock(status_code=200)
        ok.json.return_value = {'status': 'COMPLETED'}
        with mock.patch('pluto.migrate.loader.httpx.get', return_value=ok) as g:
            assert loader._read_run_status(op) == 'COMPLETED'
        assert 'api/runs/details/7' in g.call_args.args[0]
        # a non-200 and an exception both degrade to None (best-effort)
        with mock.patch(
            'pluto.migrate.loader.httpx.get',
            return_value=mock.MagicMock(status_code=404),
        ):
            assert loader._read_run_status(op) is None
        with mock.patch('pluto.migrate.loader.httpx.get', side_effect=Exception('x')):
            assert loader._read_run_status(op) is None

    def test_skip_run_ids_bypasses_run_without_attempting(self, tmp_path, mock_init):
        # A run whose id is in skip_run_ids (already attempted-and-failed in an
        # earlier `all` pass) is skipped outright — never re-initialized, never
        # counted as loaded/skipped/failed.
        init, _ = mock_init
        _stage_run(tmp_path)
        summary = PlutoLoader(tmp_path, skip_run_ids=['abc123']).load()
        assert summary == {'loaded': 0, 'skipped': 0, 'failed': []}
        init.assert_not_called()

    def test_backpressure_throttles_then_gives_up(self, tmp_path, mock_init):
        _, op = mock_init
        op._sync_manager.get_pending_count.side_effect = [10, 10, 3]
        _stage_run(tmp_path)
        with mock.patch('pluto.migrate.loader.time.sleep') as sleep:
            PlutoLoader(tmp_path, max_pending=5).load()
        assert sleep.called  # throttled while pending > max_pending

        op._sync_manager.get_pending_count.side_effect = None
        op._sync_manager.get_pending_count.return_value = 10
        (tmp_path / 'loaded_runs.json').unlink()
        with mock.patch('pluto.migrate.loader.time.sleep'):
            summary = PlutoLoader(tmp_path, max_pending=5, stall_timeout=0).load()
        assert summary['loaded'] == 1  # bounded: gives up waiting, keeps going

    def test_dry_run_makes_no_runs(self, tmp_path, mock_init):
        init, _ = mock_init
        _stage_run(tmp_path)
        summary = PlutoLoader(tmp_path, dry_run=True).load()
        assert summary['loaded'] == 0
        init.assert_not_called()
        assert not (tmp_path / 'loaded_runs.json').exists()

    def test_dest_project_override(self, tmp_path, mock_init):
        init, _ = mock_init
        _stage_run(tmp_path)
        PlutoLoader(tmp_path, dest_project='legacy-wandb').load()
        assert init.call_args.kwargs['project'] == 'legacy-wandb'

    def test_missing_media_file_skipped_not_fatal(self, tmp_path, mock_init):
        _, op = mock_init
        run_dir = _stage_run(tmp_path)
        (run_dir / 'files' / 'media' / 'images' / 'sample_3.png').unlink()
        summary = PlutoLoader(tmp_path).load()
        assert summary['loaded'] == 1
        assert not _log_calls_with(
            op, lambda d: any(isinstance(v, pluto.Image) for v in d.values())
        )

    def test_unreadable_manifest_recorded_and_batch_continues(
        self, tmp_path, mock_init
    ):
        # One run's run.json is corrupt; the other must still load and the bad
        # one is recorded in failed[] rather than aborting the whole batch.
        _stage_run(tmp_path, run_id='good1')
        bad = _stage_run(tmp_path, run_id='bad1')
        (bad / 'run.json').write_text('{ truncated')  # invalid JSON
        summary = PlutoLoader(tmp_path).load()
        assert summary['loaded'] == 1
        assert len(summary['failed']) == 1
        assert summary['failed'][0]['run_id'] == 'bad1'

    def test_init_failure_recorded_and_batch_continues(self, tmp_path, mock_init):
        # A non-RunExistsError from pluto.init on one run must not abort the
        # batch; it is recorded as failed and the next run still loads.
        init, op = mock_init
        _stage_run(tmp_path, run_id='aaa1')
        _stage_run(tmp_path, run_id='bbb2')
        init.side_effect = [ConnectionError('server down'), op]
        summary = PlutoLoader(tmp_path).load()
        assert summary['loaded'] == 1
        assert len(summary['failed']) == 1
        assert 'ConnectionError' in summary['failed'][0]['error']

    def test_dead_sync_process_fails_fast(self, tmp_path, mock_init):
        # If the sync subprocess has exited, backpressure must raise (recorded
        # as a failed run) instead of sleeping out the full stall_timeout.
        _, op = mock_init
        op._sync_manager.get_pending_count.return_value = 999
        op._sync_manager._process.poll.return_value = 1  # exited, code 1
        _stage_run(tmp_path)
        summary = PlutoLoader(tmp_path, max_pending=5, stall_timeout=600).load()
        assert summary['loaded'] == 0
        assert len(summary['failed']) == 1
        assert 'sync process exited' in summary['failed'][0]['error']

    def test_path_traversal_media_refused(self, tmp_path, mock_init):
        # A media row whose file_value escapes the run dir must be refused, not
        # read+uploaded off the host.
        _, op = mock_init
        run_dir = tmp_path / 'acme' / 'vision' / 'runs' / 'evil1'
        run_dir.mkdir(parents=True)
        write_json_atomic(
            run_dir / 'run.json',
            {
                'entity': 'acme',
                'project': 'vision',
                'run_id': 'evil1',
                'name': 'evil',
                'state': 'finished',
            },
        )
        outside = tmp_path / 'secret.txt'
        outside.write_bytes(b'top secret')
        with PartWriter(run_dir) as w:
            w.write_row(
                project_id='acme/vision',
                run_id='evil1',
                attribute_path='sneaky',
                attribute_type='media',
                step=0,
                timestamp_ms=T0_MS,
                string_value='image-file',
                file_value='../../../../secret.txt',
            )
        mark_run_exported(run_dir, {'rows': 1})
        summary = PlutoLoader(tmp_path).load()
        assert summary['loaded'] == 1  # run itself loads
        # ...but the out-of-bounds file was never turned into an upload.
        assert not _log_calls_with(
            op, lambda d: any(isinstance(v, pluto.Image) for v in d.values())
        )

    def test_dry_run_reports_would_load(self, tmp_path, mock_init, capsys):
        _stage_run(tmp_path)
        PlutoLoader(tmp_path, dry_run=True).load()
        out = capsys.readouterr().out
        assert 'would load 1' in out

    def test_histogram_with_null_bins_loads(self, tmp_path, mock_init):
        # wandb often stores histogram counts with bins=None; this must not
        # crash the run (real data hit `len(None)` in pluto.Histogram).
        _, op = mock_init
        run_dir = tmp_path / 'acme' / 'vision' / 'runs' / 'histrun'
        run_dir.mkdir(parents=True)
        write_json_atomic(
            run_dir / 'run.json',
            {
                'entity': 'acme',
                'project': 'vision',
                'run_id': 'histrun',
                'name': 'h',
                'state': 'finished',
            },
        )
        with PartWriter(run_dir) as w:
            w.write_row(
                project_id='acme/vision',
                run_id='histrun',
                attribute_path='weights',
                attribute_type='media',
                step=0,
                timestamp_ms=T0_MS,
                string_value=json.dumps(
                    {'_type': 'histogram', 'values': [3, 1, 4, 1], 'bins': None}
                ),
            )
        mark_run_exported(run_dir, {'rows': 1})
        summary = PlutoLoader(tmp_path).load()
        assert summary == {'loaded': 1, 'skipped': 0, 'failed': []}
        # the histogram was logged (synthesized bins), not dropped
        assert _log_calls_with(
            op, lambda d: any(isinstance(v, pluto.Histogram) for v in d.values())
        )

    def test_multi_image_step_logged_as_ordered_list(self, tmp_path, mock_init):
        # An image gallery (many media rows sharing name+step+timestamp) must be
        # logged in ONE op.log call as a list, so each image gets its sampleIndex
        # (0,1,2) and the server preserves logged order — not 3 separate calls
        # that all collapse to sampleIndex 0.
        _, op = mock_init
        run_dir = tmp_path / 'acme' / 'vision' / 'runs' / 'galrun'
        run_dir.mkdir(parents=True)
        write_json_atomic(
            run_dir / 'run.json',
            {
                'entity': 'acme',
                'project': 'vision',
                'run_id': 'galrun',
                'name': 'g',
                'state': 'finished',
            },
        )
        for i in range(3):
            f = run_dir / 'files' / f'idx{i}.png'
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b'PNG')
        with PartWriter(run_dir) as w:
            for i in range(3):
                w.write_row(
                    project_id='acme/vision',
                    run_id='galrun',
                    attribute_path='gallery',
                    attribute_type='media',
                    step=5,
                    timestamp_ms=T0_MS,
                    string_value='image-file',
                    file_value=f'files/idx{i}.png',
                    caption=f'c{i}',
                )
        mark_run_exported(run_dir, {'rows': 3})
        summary = PlutoLoader(tmp_path).load()
        assert summary == {'loaded': 1, 'skipped': 0, 'failed': []}
        gallery_calls = _log_calls_with(op, lambda d: 'gallery' in d)
        assert len(gallery_calls) == 1  # ONE batched call, not three
        value = gallery_calls[0].args[0]['gallery']
        assert isinstance(value, list) and len(value) == 3
        # order preserved -> sampleIndex 0,1,2 assigned by op.log's enumerate
        assert [img._caption for img in value] == ['c0', 'c1', 'c2']

    def test_cleanup_removes_staged_files_after_load(self, tmp_path, mock_init):
        # --cleanup frees each run's staged files once it's loaded, but the run
        # stays recorded as loaded (so a re-run still skips it).
        run_dir = _stage_run(tmp_path)
        assert run_dir.exists()
        summary = PlutoLoader(tmp_path, cleanup=True).load()
        assert summary['loaded'] == 1
        assert not run_dir.exists()  # staged files reclaimed
        assert LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(CACHE_KEY)

    def test_no_cleanup_keeps_staged_files(self, tmp_path, mock_init):
        run_dir = _stage_run(tmp_path)
        PlutoLoader(tmp_path).load()  # cleanup defaults off
        assert run_dir.exists()

    def test_run_metadata_forwarded_as_system_metadata(self, tmp_path, mock_init):
        # run.metadata (git/OS/GPU) is staged; the loader forwards it via
        # compat['systemMetadata'] so repro context survives the migration.
        init, _ = mock_init
        run_dir = _stage_run(tmp_path)
        manifest = json.loads((run_dir / 'run.json').read_text())
        manifest['metadata'] = {'gpu': 'H100', 'python': '3.12'}
        write_json_atomic(run_dir / 'run.json', manifest)
        PlutoLoader(tmp_path).load()
        compat = init.call_args.kwargs['settings']['compat']
        assert compat['systemMetadata'] == {'gpu': 'H100', 'python': '3.12'}

    def test_no_metadata_keeps_compat_clean(self, tmp_path, mock_init):
        # Runs without metadata must not get a systemMetadata key (normal runs
        # send empty compat; only migration populates it).
        init, _ = mock_init
        _stage_run(tmp_path)  # no 'metadata' in the staged run.json
        PlutoLoader(tmp_path).load()
        assert 'systemMetadata' not in init.call_args.kwargs['settings']['compat']

    def test_cache_key_includes_dest_project(self, tmp_path, mock_init):
        # Loading the same export into a different dest project must NOT be
        # skipped just because it was loaded into another project.
        _stage_run(tmp_path)
        PlutoLoader(tmp_path, dest_project='proj-a').load()
        cache = LoadedCache(tmp_path / 'loaded_runs.json')
        assert cache.is_loaded(f'{EXTERNAL_ID}@@proj-a')
        assert not cache.is_loaded(f'{EXTERNAL_ID}@@proj-b')  # different dest
        # a second load into proj-b actually runs (not skipped)
        s = PlutoLoader(tmp_path, dest_project='proj-b').load()
        assert s['loaded'] == 1 and s['skipped'] == 0

    def test_bad_media_row_does_not_fail_whole_run(self, tmp_path, mock_init):
        # A single unparseable inline-media payload is skipped; the run's other
        # data still loads and the run is not marked failed.
        _, op = mock_init
        run_dir = tmp_path / 'acme' / 'vision' / 'runs' / 'mixrun'
        run_dir.mkdir(parents=True)
        write_json_atomic(
            run_dir / 'run.json',
            {
                'entity': 'acme',
                'project': 'vision',
                'run_id': 'mixrun',
                'name': 'm',
                'state': 'finished',
            },
        )
        with PartWriter(run_dir) as w:
            w.write_row(
                project_id='acme/vision',
                run_id='mixrun',
                attribute_path='broken',
                attribute_type='media',
                step=0,
                timestamp_ms=T0_MS,
                string_value='{not valid json',  # json.loads raises
            )
            w.write_row(
                project_id='acme/vision',
                run_id='mixrun',
                attribute_path='loss',
                attribute_type='metric',
                step=1,
                timestamp_ms=T0_MS + 1000,
                float_value=0.5,
            )
        mark_run_exported(run_dir, {'rows': 2})
        summary = PlutoLoader(tmp_path).load()
        assert summary == {'loaded': 1, 'skipped': 0, 'failed': []}
        op._log_metrics_batch.assert_called()  # the good metric still replayed
