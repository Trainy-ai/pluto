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
    write_json_atomic,
)

CREATED_AT_MS = 1746093600000
UPDATED_AT_MS = CREATED_AT_MS + 7200000
T0_MS = 1746093601000
EXTERNAL_ID = 'wandb::acme/vision/abc123'


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


@pytest.fixture
def mock_init():
    with mock.patch('pluto.init') as init:
        op = mock.MagicMock()
        op.settings._op_id = 42
        op._sync_manager.get_pending_count.return_value = 0
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

    def test_finish_code_mapping(self, tmp_path, mock_init):
        _, op = mock_init
        _stage_run(tmp_path, run_id='crashed1', state='crashed')
        PlutoLoader(tmp_path).load()
        op.finish.assert_called_once_with(code=1)

    def test_loaded_cache_written_and_resume_skips(self, tmp_path, mock_init):
        init, op = mock_init
        _stage_run(tmp_path)
        PlutoLoader(tmp_path).load()
        assert LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(EXTERNAL_ID)

        init.reset_mock()
        summary = PlutoLoader(tmp_path).load()
        assert summary == {'loaded': 0, 'skipped': 1, 'failed': []}
        init.assert_not_called()

    def test_external_id_collision_skips_by_default(self, tmp_path, mock_init):
        # A collision means the run already exists server-side but isn't in the
        # local cache. Re-replaying would duplicate media, so by default the
        # loader skips it (and records it so future re-runs skip via cache),
        # rather than resuming and re-uploading.
        from pluto.op import RunExistsError

        init, op = mock_init
        init.side_effect = [
            RunExistsError(
                "Run with externalId 'wandb::acme/vision/abc123' already exists."
            ),
        ]
        _stage_run(tmp_path)
        summary = PlutoLoader(tmp_path).load()
        assert summary == {'loaded': 0, 'skipped': 1, 'failed': []}
        assert init.call_count == 1  # no second (resume) init
        op._log_metrics_batch.assert_not_called()  # nothing re-replayed
        assert LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(EXTERNAL_ID)

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
        assert LoadedCache(tmp_path / 'loaded_runs.json').is_loaded(EXTERNAL_ID)

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
