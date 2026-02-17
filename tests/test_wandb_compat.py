"""
Tests for the wandb-to-pluto compatibility layer.

These tests validate that:
1. wandb API calls are correctly routed to pluto equivalents
2. Data types are converted properly
3. Config and Summary behave like wandb's dict-like objects
4. Module-level state (run, config, summary) works correctly
5. Unsupported features degrade gracefully (no-ops, not errors)
"""

import os
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestConfig:
    """Tests for the wandb.config-compatible Config class."""

    def test_attribute_set_and_get(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        c.lr = 0.001
        assert c.lr == 0.001

    def test_dict_set_and_get(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        c['batch_size'] = 32
        assert c['batch_size'] == 32

    def test_update_dict(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        c.update({'a': 1, 'b': 2})
        assert c.a == 1
        assert c['b'] == 2

    def test_update_with_namespace(self):
        """Test update() with argparse-like namespace."""
        import argparse

        from pluto.compat.wandb.config import Config

        ns = argparse.Namespace(lr=0.01, epochs=10)
        c = Config()
        c.update(ns)
        assert c.lr == 0.01
        assert c.epochs == 10

    def test_setdefaults(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        c.lr = 0.001
        c.setdefaults({'lr': 0.01, 'batch_size': 64})
        assert c.lr == 0.001  # not overwritten
        assert c.batch_size == 64  # newly set

    def test_keys_values_items(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        c.update({'a': 1, 'b': 2})
        assert set(c.keys()) == {'a', 'b'}
        assert set(c.values()) == {1, 2}
        assert set(c.items()) == {('a', 1), ('b', 2)}

    def test_get_with_default(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        assert c.get('missing', 42) == 42
        c.x = 10
        assert c.get('x', 42) == 10

    def test_contains(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        c.lr = 0.01
        assert 'lr' in c
        assert 'missing' not in c

    def test_len(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        assert len(c) == 0
        c.update({'a': 1, 'b': 2})
        assert len(c) == 2

    def test_iter(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        c.update({'a': 1, 'b': 2})
        assert set(c) == {'a', 'b'}

    def test_load(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        c._load({'x': 1, 'y': 2})
        assert c.x == 1
        assert c['y'] == 2

    def test_as_dict(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        c.update({'a': 1})
        assert c.as_dict() == {'a': 1}

    def test_attribute_error_on_missing(self):
        from pluto.compat.wandb.config import Config

        c = Config()
        with pytest.raises(AttributeError):
            _ = c.missing_key

    def test_sync_calls_update_config(self):
        from pluto.compat.wandb.config import Config

        mock_op = MagicMock()
        c = Config(op=mock_op)
        c.lr = 0.01
        mock_op.update_config.assert_called_with({'lr': 0.01})

    def test_sync_error_does_not_raise(self):
        from pluto.compat.wandb.config import Config

        mock_op = MagicMock()
        mock_op.update_config.side_effect = RuntimeError('connection failed')
        c = Config(op=mock_op)
        c.lr = 0.01  # should not raise
        assert c.lr == 0.01


class TestSummary:
    """Tests for the wandb.summary-compatible Summary class."""

    def test_dict_set_and_get(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s['best_acc'] = 0.95
        assert s['best_acc'] == 0.95

    def test_attribute_set_and_get(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s.final_loss = 0.1
        assert s.final_loss == 0.1

    def test_update_from_log(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s._update_from_log({'loss': 0.5, 'acc': 0.9})
        assert s['loss'] == 0.5
        assert s['acc'] == 0.9

    def test_update_from_log_ignores_non_scalars(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s._update_from_log({'loss': 0.5, 'image': 'not_a_scalar', 'flag': True})
        assert s['loss'] == 0.5
        assert 'image' not in s
        assert s['flag'] is True  # bools are scalar subclass of int

    def test_manual_override(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s._update_from_log({'loss': 0.5})
        s['loss'] = 0.1  # manual override
        assert s['loss'] == 0.1

    def test_keys_values_items(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s.update({'a': 1, 'b': 2})
        assert set(s.keys()) == {'a', 'b'}

    def test_get_default(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        assert s.get('missing', 99) == 99

    def test_contains(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s.x = 1
        assert 'x' in s
        assert 'y' not in s

    def test_as_dict(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s.update({'a': 1})
        assert s.as_dict() == {'a': 1}

    def test_summary_aggregation_min(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s._set_metric_definition('loss', {'name': 'loss', 'summary': 'min'})
        s._update_from_log({'loss': 0.5})
        s._update_from_log({'loss': 0.3})
        s._update_from_log({'loss': 0.7})
        assert s['loss'] == pytest.approx(0.3)

    def test_summary_aggregation_max(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s._set_metric_definition('acc', {'name': 'acc', 'summary': 'max'})
        s._update_from_log({'acc': 0.8})
        s._update_from_log({'acc': 0.95})
        s._update_from_log({'acc': 0.9})
        assert s['acc'] == pytest.approx(0.95)

    def test_summary_aggregation_mean(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s._set_metric_definition('loss', {'name': 'loss', 'summary': 'mean'})
        s._update_from_log({'loss': 1.0})
        s._update_from_log({'loss': 2.0})
        s._update_from_log({'loss': 3.0})
        assert s['loss'] == pytest.approx(2.0)

    def test_summary_aggregation_first(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s._set_metric_definition('lr', {'name': 'lr', 'summary': 'first'})
        s._update_from_log({'lr': 0.01})
        s._update_from_log({'lr': 0.001})
        s._update_from_log({'lr': 0.0001})
        assert s['lr'] == pytest.approx(0.01)

    def test_summary_aggregation_last(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s._set_metric_definition('step', {'name': 'step', 'summary': 'last'})
        s._update_from_log({'step': 1})
        s._update_from_log({'step': 2})
        s._update_from_log({'step': 3})
        assert s['step'] == 3

    def test_summary_aggregation_glob(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        s._set_metric_definition('val/*', {'name': 'val/*', 'summary': 'min'})
        s._update_from_log({'val/loss': 0.5, 'val/acc': 0.8})
        s._update_from_log({'val/loss': 0.3, 'val/acc': 0.6})
        s._update_from_log({'val/loss': 0.7, 'val/acc': 0.9})
        assert s['val/loss'] == pytest.approx(0.3)
        assert s['val/acc'] == pytest.approx(0.6)

    def test_summary_no_definition_keeps_last(self):
        from pluto.compat.wandb.summary import Summary

        s = Summary()
        # No definition set — default behavior
        s._update_from_log({'loss': 0.5})
        s._update_from_log({'loss': 0.3})
        s._update_from_log({'loss': 0.7})
        assert s['loss'] == pytest.approx(0.7)


class TestDataTypes:
    """Tests for wandb data type wrappers."""

    def test_image_from_numpy(self):
        from pluto.compat.wandb.data_types import Image

        data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image(data, caption='test')
        assert img.caption == 'test'
        pluto_img = img._to_pluto()
        assert pluto_img.__class__.__name__ == 'Image'

    def test_image_from_path(self, tmp_path):
        from pluto.compat.wandb.data_types import Image

        p = tmp_path / 'test.png'
        # Write a minimal PNG
        from PIL import Image as PILImage

        pil = PILImage.new('RGB', (10, 10))
        pil.save(str(p))

        img = Image(str(p))
        pluto_img = img._to_pluto()
        assert pluto_img.__class__.__name__ == 'Image'

    def test_audio_wrapper(self):
        from pluto.compat.wandb.data_types import Audio

        data = np.random.randn(16000).astype(np.float32)
        a = Audio(data, sample_rate=16000, caption='test_audio')
        pluto_audio = a._to_pluto()
        assert pluto_audio.__class__.__name__ == 'Audio'

    def test_video_wrapper(self, tmp_path):
        from pluto.compat.wandb.data_types import Video

        p = tmp_path / 'test.mp4'
        p.write_bytes(b'\x00' * 100)
        v = Video(str(p), fps=30, caption='test_video')
        pluto_video = v._to_pluto()
        assert pluto_video.__class__.__name__ == 'Video'

    def test_table_from_data(self):
        from pluto.compat.wandb.data_types import Table

        t = Table(columns=['a', 'b'], data=[[1, 2], [3, 4]])
        assert t.columns == ['a', 'b']
        pluto_table = t._to_pluto()
        assert pluto_table.__class__.__name__ == 'Table'

    def test_table_add_data(self):
        from pluto.compat.wandb.data_types import Table

        t = Table(columns=['x', 'y'])
        t.add_data(1, 2)
        t.add_data(3, 4)
        assert len(t._data) == 2

    def test_table_add_column(self):
        from pluto.compat.wandb.data_types import Table

        t = Table(columns=['x'], data=[[1], [2]])
        t.add_column('y', [10, 20])
        assert 'y' in t.columns
        assert t._data[0] == [1, 10]

    def test_table_get_column(self):
        from pluto.compat.wandb.data_types import Table

        t = Table(columns=['a', 'b'], data=[[1, 2], [3, 4]])
        assert t.get_column('a') == [1, 3]
        assert t.get_column('b') == [2, 4]
        assert t.get_column('missing') == []

    def test_table_from_dataframe(self):
        pd = pytest.importorskip('pandas')

        from pluto.compat.wandb.data_types import Table

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        t = Table(dataframe=df)
        pluto_table = t._to_pluto()
        assert pluto_table.__class__.__name__ == 'Table'

    def test_histogram_from_sequence(self):
        from pluto.compat.wandb.data_types import Histogram

        h = Histogram(sequence=[1, 2, 3, 4, 5], num_bins=10)
        pluto_hist = h._to_pluto()
        assert pluto_hist.__class__.__name__ == 'Histogram'

    def test_histogram_from_np_histogram(self):
        from pluto.compat.wandb.data_types import Histogram

        counts, bins = np.histogram([1, 2, 3, 4, 5], bins=5)
        h = Histogram(np_histogram=(counts, bins))
        pluto_hist = h._to_pluto()
        assert pluto_hist.__class__.__name__ == 'Histogram'

    def test_html_from_string(self):
        from pluto.compat.wandb.data_types import Html

        h = Html('<h1>Hello</h1>')
        pluto_text = h._to_pluto()
        assert pluto_text.__class__.__name__ == 'Text'

    def test_html_from_file(self, tmp_path):
        from pluto.compat.wandb.data_types import Html

        p = tmp_path / 'test.html'
        p.write_text('<h1>Hello</h1>')
        h = Html(str(p))
        assert h._html == '<h1>Hello</h1>'

    def test_alert_level(self):
        from pluto.compat.wandb.data_types import AlertLevel

        assert AlertLevel.INFO == 'INFO'
        assert AlertLevel.WARN == 'WARN'
        assert AlertLevel.ERROR == 'ERROR'

    def test_artifact_add_file(self, tmp_path):
        from pluto.compat.wandb.data_types import Artifact

        f1 = tmp_path / 'model.pt'
        f1.write_bytes(b'\x00' * 100)
        art = Artifact('my-model', type='model')
        art.add_file(str(f1), name='model.pt')
        assert len(art._files) == 1
        assert art._files[0]['name'] == 'model.pt'

    def test_artifact_add_dir(self, tmp_path):
        from pluto.compat.wandb.data_types import Artifact

        d = tmp_path / 'data'
        d.mkdir()
        (d / 'a.txt').write_text('hello')
        (d / 'b.txt').write_text('world')
        art = Artifact('my-data', type='dataset')
        art.add_dir(str(d))
        assert len(art._files) == 2

    def test_artifact_to_pluto_files(self, tmp_path):
        from pluto.compat.wandb.data_types import Artifact

        f1 = tmp_path / 'file.bin'
        f1.write_bytes(b'\x00' * 50)
        art = Artifact('test', type='model')
        art.add_file(str(f1))
        pluto_files = art._to_pluto_files()
        assert len(pluto_files) == 1
        assert pluto_files[0].__class__.__name__ == 'Artifact'

    def test_graph_wrapper(self):
        from pluto.compat.wandb.data_types import Graph

        g = Graph()
        pluto_graph = g._to_pluto()
        assert pluto_graph.__class__.__name__ == 'Graph'


class TestRun:
    """Tests for the wandb.Run-compatible Run class."""

    def _make_run(self, **kwargs):
        """Create a Run with a mocked Op."""
        from pluto.compat.wandb.run import Run

        op = MagicMock()
        op.id = 123
        op.run_id = None
        op.tags = ['tag1']
        op.resumed = False
        op.settings = MagicMock()
        op.settings._op_name = 'test-run'
        op.settings.project = 'test-project'
        op.settings.url_view = 'https://pluto.trainy.ai/run/123'
        op.settings.get_dir.return_value = '/tmp/test-run'
        op.config = {}
        return Run(op=op, **kwargs), op

    def test_properties(self):
        run, op = self._make_run(name='my-run', notes='some notes', group='grp')
        assert run.id == '123'
        assert run.name == 'my-run'
        assert run.project == 'test-project'
        assert run.notes == 'some notes'
        assert run.group == 'grp'
        assert run.entity == ''
        assert run.url == 'https://pluto.trainy.ai/run/123'

    def test_tags_get_and_set(self):
        run, op = self._make_run()
        assert run.tags == ('tag1',)

        # Setting tags
        run.tags = ('tag1', 'tag2')
        op.add_tags.assert_called_with(['tag2'])

    def test_name_setter(self):
        run, op = self._make_run(name='original')
        run.name = 'new-name'
        assert run.name == 'new-name'

    def test_log_scalars(self):
        run, op = self._make_run()
        run.log({'loss': 0.5, 'acc': 0.9})
        op.log.assert_called_once()
        logged_data = op.log.call_args[0][0]
        assert logged_data['loss'] == 0.5
        assert logged_data['acc'] == 0.9

    def test_log_nested_dict(self):
        run, op = self._make_run()
        run.log({'train': {'loss': 0.5}})
        op.log.assert_called_once()
        logged_data = op.log.call_args[0][0]
        assert 'train/loss' in logged_data

    def test_log_with_step(self):
        run, op = self._make_run()
        run.log({'loss': 0.5}, step=10)
        op.log.assert_called_once()
        assert op.log.call_args[1]['step'] == 10

    def test_log_commit_false(self):
        run, op = self._make_run()
        run.log({'loss': 0.5}, commit=False)
        op.log.assert_not_called()

        run.log({'acc': 0.9})  # default commit=True
        op.log.assert_called_once()
        logged_data = op.log.call_args[0][0]
        assert 'loss' in logged_data
        assert 'acc' in logged_data

    def test_log_updates_summary(self):
        run, op = self._make_run()
        run.log({'loss': 0.5})
        assert run.summary['loss'] == 0.5

    def test_log_converts_data_types(self):
        from pluto.compat.wandb.data_types import Image

        run, op = self._make_run()
        data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        run.log({'image': Image(data)})
        op.log.assert_called_once()
        logged_data = op.log.call_args[0][0]
        assert logged_data['image'].__class__.__name__ == 'Image'

    def test_finish(self):
        run, op = self._make_run()
        run.finish()
        op.finish.assert_called_once()

    def test_finish_with_exit_code(self):
        run, op = self._make_run()
        run.finish(exit_code=1)
        op.finish.assert_called_once_with(code=1)

    def test_context_manager(self):
        run, op = self._make_run()
        with run:
            run.log({'x': 1})
        op.finish.assert_called_once_with(code=0)

    def test_context_manager_with_exception(self):
        run, op = self._make_run()
        with pytest.raises(ValueError):
            with run:
                raise ValueError('test')
        op.finish.assert_called_once_with(code=1)

    def test_watch(self):
        run, op = self._make_run()
        model = MagicMock()
        run.watch(model, log_freq=500)
        op.watch.assert_called_once_with(model, log_freq=500)

    def test_alert(self):
        run, op = self._make_run()
        run.alert(title='Loss spike', text='Loss exceeded 10.0', level='WARN')
        op.alert.assert_called_once()

    def test_define_metric_returns_stub(self):
        run, op = self._make_run()
        m = run.define_metric('loss', step_metric='epoch')
        assert m.name == 'loss'
        # Verify Op.define_metric was called
        op.define_metric.assert_called_once_with(
            'loss', step_metric='epoch', summary=None, goal=None, hidden=None
        )

    def test_define_metric_with_step_metric(self):
        run, op = self._make_run()
        run.define_metric('val/loss', step_metric='epoch')
        op.define_metric.assert_called_once_with(
            'val/loss', step_metric='epoch', summary=None, goal=None, hidden=None
        )

    def test_unsupported_methods_no_error(self):
        run, op = self._make_run()
        run.save()
        run.restore('test')
        run.log_code()
        run.mark_preempting()
        run.use_artifact('test')

    def test_log_artifact_with_artifact_object(self, tmp_path):
        from pluto.compat.wandb.data_types import Artifact

        run, op = self._make_run()
        f = tmp_path / 'model.pt'
        f.write_bytes(b'\x00' * 100)
        art = Artifact('model', type='model')
        art.add_file(str(f))
        run.log_artifact(art)
        op.log.assert_called()

    def test_repr(self):
        run, op = self._make_run(name='exp-1')
        r = repr(run)
        assert 'test-project' in r
        assert 'exp-1' in r

    def test_step_tracking(self):
        run, op = self._make_run()
        assert run.step == 0
        run.log({'x': 1})
        assert run.step == 1
        run.log({'x': 2})
        assert run.step == 2
        run.log({'x': 3}, step=10)
        assert run.step == 10

    def test_offline_disabled(self):
        run_online, _ = self._make_run(mode='online')
        assert not run_online.offline
        assert not run_online.disabled

        run_off, _ = self._make_run(mode='offline')
        assert run_off.offline

        run_dis, _ = self._make_run(mode='disabled')
        assert run_dis.disabled

    def test_path(self):
        run, _ = self._make_run()
        assert run.path == '/test-project/123'


class TestModuleAPI:
    """Tests for the module-level wandb API (init, log, finish, etc.)."""

    def test_log_before_init_raises(self):
        import pluto.compat.wandb as wandb

        # Ensure no active run
        wandb.run = None
        with pytest.raises(RuntimeError, match='wandb.log.*called before wandb.init'):
            wandb.log({'x': 1})

    def test_watch_before_init_raises(self):
        import pluto.compat.wandb as wandb

        wandb.run = None
        with pytest.raises(RuntimeError, match='wandb.watch.*called before wandb.init'):
            wandb.watch(MagicMock())

    def test_finish_without_init_is_noop(self):
        import pluto.compat.wandb as wandb

        wandb.run = None
        wandb.finish()  # should not raise

    def test_define_metric_without_init(self):
        import pluto.compat.wandb as wandb

        wandb.run = None
        m = wandb.define_metric('loss')
        assert m.name == 'loss'

    def test_unsupported_module_functions_noop(self):
        import pluto.compat.wandb as wandb

        # These should all be no-ops, never raise
        wandb.save('*.pt')
        wandb.restore('model.pt')
        wandb.log_code()
        wandb.mark_preempting()

    def test_login_returns_true(self):
        import pluto.compat.wandb as wandb

        assert wandb.login() is True

    def test_settings_class(self):
        import pluto.compat.wandb as wandb

        s = wandb.Settings(mode='offline', console='auto')
        assert s.mode == 'offline'
        assert s.console == 'auto'

    def test_data_types_importable(self):
        """Test that all wandb data types are importable from the module."""
        from pluto.compat.wandb import (
            AlertLevel,
            Artifact,
            Audio,
            Graph,
            Histogram,
            Html,
            Image,
            Table,
            Video,
        )

        assert Image is not None
        assert Audio is not None
        assert Video is not None
        assert Table is not None
        assert Histogram is not None
        assert Html is not None
        assert Graph is not None
        assert Artifact is not None
        assert AlertLevel is not None

    def test_import_as_wandb(self):
        """Test the canonical import pattern."""
        import pluto.compat.wandb as wandb

        assert hasattr(wandb, 'init')
        assert hasattr(wandb, 'log')
        assert hasattr(wandb, 'finish')
        assert hasattr(wandb, 'watch')
        assert hasattr(wandb, 'unwatch')
        assert hasattr(wandb, 'alert')
        assert hasattr(wandb, 'config')
        assert hasattr(wandb, 'summary')
        assert hasattr(wandb, 'run')
        assert hasattr(wandb, 'Image')
        assert hasattr(wandb, 'Table')
        assert hasattr(wandb, 'Run')

    @mock.patch.dict(
        os.environ,
        {
            'WANDB_PROJECT': 'env-project',
        },
        clear=False,
    )
    def test_init_picks_up_wandb_project_env(self):
        """Test that WANDB_PROJECT env var is used as fallback."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = MagicMock()
            mock_op.id = 1
            mock_op.run_id = None
            mock_op.tags = []
            mock_op.resumed = False
            mock_op.settings = MagicMock()
            mock_op.settings._op_name = 'test'
            mock_op.settings.project = 'env-project'
            mock_op.settings.url_view = None
            mock_op.settings.get_dir.return_value = '/tmp'
            mock_op.config = {}
            mock_init.return_value = mock_op

            wandb.init()
            mock_init.assert_called_once()
            assert mock_init.call_args[1]['project'] == 'env-project'
            wandb.finish()

    @mock.patch.dict(
        os.environ,
        {
            'WANDB_MODE': 'disabled',
        },
        clear=False,
    )
    def test_init_disabled_mode_from_env(self):
        """Test that WANDB_MODE=disabled creates noop run."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = MagicMock()
            mock_op.id = 1
            mock_op.run_id = None
            mock_op.tags = []
            mock_op.resumed = False
            mock_op.settings = MagicMock()
            mock_op.settings._op_name = 'test'
            mock_op.settings.project = 'test'
            mock_op.settings.url_view = None
            mock_op.settings.get_dir.return_value = '/tmp'
            mock_op.config = {}
            mock_init.return_value = mock_op

            wandb.init(project='test')
            # Should pass mode='noop' in settings
            call_kwargs = mock_init.call_args[1]
            settings = call_kwargs.get('settings', {})
            assert settings.get('mode') == 'noop'
            wandb.finish()

    def test_init_creates_disabled_run_on_failure(self):
        """Test that init creates a disabled run if pluto.init fails."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            # First call fails, second call (disabled) succeeds
            mock_op = MagicMock()
            mock_op.id = 0
            mock_op.run_id = None
            mock_op.tags = []
            mock_op.resumed = False
            mock_op.settings = MagicMock()
            mock_op.settings._op_name = 'disabled'
            mock_op.settings.project = 'disabled'
            mock_op.settings.url_view = None
            mock_op.settings.get_dir.return_value = '/tmp'
            mock_op.config = {}
            mock_init.side_effect = [RuntimeError('auth failed'), mock_op]

            run = wandb.init(project='test')
            assert run.disabled
            wandb.finish()

    @mock.patch.dict(
        os.environ,
        {
            'WANDB_TAGS': 'tag1,tag2, tag3',
        },
        clear=False,
    )
    def test_init_tags_from_env(self):
        """Test that WANDB_TAGS env var is parsed."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = MagicMock()
            mock_op.id = 1
            mock_op.run_id = None
            mock_op.tags = ['tag1', 'tag2', 'tag3']
            mock_op.resumed = False
            mock_op.settings = MagicMock()
            mock_op.settings._op_name = 'test'
            mock_op.settings.project = 'test'
            mock_op.settings.url_view = None
            mock_op.settings.get_dir.return_value = '/tmp'
            mock_op.config = {}
            mock_init.return_value = mock_op

            wandb.init(project='test')
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['tags'] == ['tag1', 'tag2', 'tag3']
            wandb.finish()


class TestFlattenDict:
    """Tests for nested dict flattening (wandb convention)."""

    def test_flat_dict_unchanged(self):
        from pluto.compat.wandb.run import _flatten_dict

        assert _flatten_dict({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}

    def test_nested_dict(self):
        from pluto.compat.wandb.run import _flatten_dict

        result = _flatten_dict({'train': {'loss': 0.5, 'acc': 0.9}})
        assert result == {'train/loss': 0.5, 'train/acc': 0.9}

    def test_deeply_nested(self):
        from pluto.compat.wandb.run import _flatten_dict

        result = _flatten_dict({'a': {'b': {'c': 1}}})
        assert result == {'a/b/c': 1}

    def test_mixed_nesting(self):
        from pluto.compat.wandb.run import _flatten_dict

        result = _flatten_dict({'loss': 0.5, 'train': {'acc': 0.9}})
        assert result == {'loss': 0.5, 'train/acc': 0.9}


class TestValueConversion:
    """Tests for wandb data type → pluto type conversion in log()."""

    def test_scalar_passthrough(self):
        from pluto.compat.wandb.run import _convert_value

        assert _convert_value(42) == 42
        assert _convert_value(3.14) == 3.14
        assert _convert_value('hello') == 'hello'

    def test_image_converted(self):
        from pluto.compat.wandb.data_types import Image
        from pluto.compat.wandb.run import _convert_value

        data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        img = Image(data)
        result = _convert_value(img)
        assert result.__class__.__name__ == 'Image'
        # Should be pluto Image, not wandb Image
        assert result.__class__.__module__.startswith('pluto.file')

    def test_table_converted(self):
        from pluto.compat.wandb.data_types import Table
        from pluto.compat.wandb.run import _convert_value

        t = Table(columns=['a'], data=[[1], [2]])
        result = _convert_value(t)
        assert result.__class__.__name__ == 'Table'
        assert result.__class__.__module__.startswith('pluto.data')

    def test_histogram_converted(self):
        from pluto.compat.wandb.data_types import Histogram
        from pluto.compat.wandb.run import _convert_value

        h = Histogram(sequence=[1, 2, 3, 4, 5])
        result = _convert_value(h)
        assert result.__class__.__name__ == 'Histogram'
        assert result.__class__.__module__.startswith('pluto.data')


# ---------------------------------------------------------------------------
# Helpers for parity contract tests
# ---------------------------------------------------------------------------


def _make_mock_op(**overrides):
    """Create a consistently-configured mock Op for contract tests."""
    op = MagicMock()
    op.id = overrides.get('id', 1)
    op.run_id = overrides.get('run_id', None)
    op.tags = list(overrides.get('tags', []))
    op.resumed = overrides.get('resumed', False)
    op.settings = MagicMock()
    op.settings._op_name = overrides.get('name', 'test-run')
    op.settings.project = overrides.get('project', 'test-project')
    op.settings.url_view = overrides.get('url_view', None)
    op.settings.get_dir.return_value = '/tmp/pluto'
    op.config = {}
    return op


class TestParityContract:
    """Snapshot/contract tests that pin the exact pluto calls the compat layer
    produces for common wandb usage patterns.

    Each test simulates a realistic wandb workflow end-to-end via the
    module-level API and asserts the full call sequence that reaches
    the underlying pluto.Op mock.
    """

    def test_standard_training_loop(self):
        """Standard loop: init → config → log N steps → finish."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            run = wandb.init(project='my-project', config={'lr': 0.01, 'epochs': 5})

            # --- init assertions ---
            mock_init.assert_called_once()
            init_kw = mock_init.call_args[1]
            assert init_kw['project'] == 'my-project'
            # config should be forwarded
            assert init_kw['config']['lr'] == 0.01
            assert init_kw['config']['epochs'] == 5

            # --- log loop ---
            for i in range(5):
                wandb.log({'loss': 1.0 / (i + 1), 'acc': i * 0.2})

            assert mock_op.log.call_count == 5
            # First call
            first_data = mock_op.log.call_args_list[0][0][0]
            assert first_data == {'loss': 1.0, 'acc': 0.0}
            # Last call
            last_data = mock_op.log.call_args_list[4][0][0]
            assert last_data['loss'] == pytest.approx(0.2)
            assert last_data['acc'] == pytest.approx(0.8)

            # --- summary tracks last values ---
            assert run.summary['loss'] == pytest.approx(0.2)
            assert run.summary['acc'] == pytest.approx(0.8)

            # --- finish ---
            wandb.finish()
            mock_op.finish.assert_called_once()

    def test_nested_metric_namespaces(self):
        """Nested dicts are flattened with / separators before reaching pluto."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            wandb.init(project='test')
            wandb.log(
                {
                    'train': {'loss': 0.5, 'acc': 0.9},
                    'val': {'loss': 0.6, 'acc': 0.85},
                    'lr': 0.001,
                }
            )

            logged = mock_op.log.call_args[0][0]
            assert set(logged.keys()) == {
                'train/loss',
                'train/acc',
                'val/loss',
                'val/acc',
                'lr',
            }
            assert logged['train/loss'] == 0.5
            assert logged['val/acc'] == 0.85
            assert logged['lr'] == 0.001

            wandb.finish()

    def test_commit_false_buffering(self):
        """commit=False accumulates data; next commit=True flushes all."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            wandb.init(project='test')

            # These should NOT call op.log
            wandb.log({'loss': 0.5}, commit=False)
            wandb.log({'acc': 0.9}, commit=False)
            assert mock_op.log.call_count == 0

            # This flushes everything
            wandb.log({'lr': 0.01})
            assert mock_op.log.call_count == 1

            flushed = mock_op.log.call_args[0][0]
            assert flushed == {'loss': 0.5, 'acc': 0.9, 'lr': 0.01}

            wandb.finish()

    def test_commit_false_later_values_win(self):
        """When same key is buffered then committed, last value wins."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            wandb.init(project='test')

            wandb.log({'loss': 0.5}, commit=False)
            wandb.log({'loss': 0.3})  # overrides buffered value

            flushed = mock_op.log.call_args[0][0]
            assert flushed['loss'] == 0.3

            wandb.finish()

    def test_explicit_step_forwarded(self):
        """step= kwarg is passed through to pluto op.log."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            wandb.init(project='test')

            wandb.log({'x': 1}, step=0)
            wandb.log({'x': 2}, step=5)
            wandb.log({'x': 3}, step=100)

            steps = [c[1]['step'] for c in mock_op.log.call_args_list]
            assert steps == [0, 5, 100]

            wandb.finish()

    def test_config_mutations_sync(self):
        """Config changes after init reach pluto via update_config."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            run = wandb.init(project='test', config={'lr': 0.01})

            # Attribute assignment
            wandb.config.batch_size = 32
            mock_op.update_config.assert_called_with({'batch_size': 32})

            # Dict assignment
            wandb.config['optimizer'] = 'adam'
            mock_op.update_config.assert_called_with({'optimizer': 'adam'})

            # Bulk update
            wandb.config.update({'dropout': 0.1, 'weight_decay': 1e-4})
            mock_op.update_config.assert_called_with(
                {'dropout': 0.1, 'weight_decay': 1e-4}
            )

            # All values accessible
            assert run.config.as_dict() == {
                'lr': 0.01,
                'batch_size': 32,
                'optimizer': 'adam',
                'dropout': 0.1,
                'weight_decay': 1e-4,
            }

            wandb.finish()

    def test_config_from_argparse(self):
        """argparse.Namespace passed to init is forwarded as config dict."""
        import argparse

        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            args = argparse.Namespace(lr=0.001, epochs=10, model='resnet50')
            wandb.init(project='test', config=args)

            init_kw = mock_init.call_args[1]
            assert init_kw['config']['lr'] == 0.001
            assert init_kw['config']['epochs'] == 10
            assert init_kw['config']['model'] == 'resnet50'

            wandb.finish()

    def test_config_include_exclude_keys(self):
        """config_include_keys and config_exclude_keys filter correctly."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            # include_keys: only lr and epochs pass through
            wandb.init(
                project='test',
                config={'lr': 0.01, 'epochs': 5, 'secret': 'xxx'},
                config_include_keys=['lr', 'epochs'],
            )
            init_kw = mock_init.call_args[1]
            assert 'lr' in init_kw['config']
            assert 'epochs' in init_kw['config']
            assert 'secret' not in init_kw['config']
            wandb.finish()

            mock_init.reset_mock()
            mock_init.return_value = mock_op

            # exclude_keys: secret filtered out
            wandb.init(
                project='test',
                config={'lr': 0.01, 'epochs': 5, 'secret': 'xxx'},
                config_exclude_keys=['secret'],
            )
            init_kw = mock_init.call_args[1]
            assert 'lr' in init_kw['config']
            assert 'secret' not in init_kw['config']
            wandb.finish()

    def test_tags_lifecycle(self):
        """Tags from init() and runtime mutations reach pluto correctly."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op(tags=['baseline', 'v1'])
            mock_init.return_value = mock_op

            run = wandb.init(project='test', tags=['baseline', 'v1'])

            # init forwards tags
            assert mock_init.call_args[1]['tags'] == ['baseline', 'v1']

            # Runtime tag mutation via property setter
            run.tags = ['baseline', 'v1', 'promoted']
            mock_op.add_tags.assert_called_once_with(['promoted'])

            wandb.finish()

    def test_data_type_conversion_in_log(self):
        """wandb data types logged via log() arrive as pluto types at op.log."""
        import pluto.compat.wandb as wandb
        from pluto.compat.wandb.data_types import Histogram, Image, Table

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            wandb.init(project='test')

            img_data = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
            wandb.log(
                {
                    'loss': 0.5,
                    'image': Image(img_data, caption='sample'),
                    'table': Table(columns=['a', 'b'], data=[[1, 2]]),
                    'dist': Histogram(sequence=[1, 2, 3, 4, 5]),
                }
            )

            logged = mock_op.log.call_args[0][0]

            # Scalars pass through
            assert logged['loss'] == 0.5

            # Data types are converted to pluto equivalents
            assert logged['image'].__class__.__module__.startswith('pluto.file')
            assert logged['image'].__class__.__name__ == 'Image'

            assert logged['table'].__class__.__module__.startswith('pluto.data')
            assert logged['table'].__class__.__name__ == 'Table'

            assert logged['dist'].__class__.__module__.startswith('pluto.data')
            assert logged['dist'].__class__.__name__ == 'Histogram'

            wandb.finish()

    def test_summary_tracks_last_scalar_per_key(self):
        """summary auto-updates to last-logged scalar; non-scalars ignored."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            run = wandb.init(project='test')

            wandb.log({'loss': 1.0, 'name': 'first'})
            wandb.log({'loss': 0.5, 'name': 'second'})
            wandb.log({'loss': 0.1, 'name': 'third'})

            assert run.summary['loss'] == pytest.approx(0.1)
            # Strings are not tracked in summary
            assert 'name' not in run.summary

            # Manual override still works
            run.summary['best_loss'] = 0.05
            assert run.summary['best_loss'] == 0.05

            wandb.finish()

    def test_context_manager_lifecycle(self):
        """with wandb.init() as run: produces correct init/finish sequence."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            with wandb.init(project='test') as run:
                run.log({'step': 1})
                run.log({'step': 2})

            # init called exactly once
            mock_init.assert_called_once()
            # Two log calls
            assert mock_op.log.call_count == 2
            # finish called exactly once with success code
            mock_op.finish.assert_called_once_with(code=0)

    def test_context_manager_exception_exit_code(self):
        """Exception inside context manager produces exit_code=1."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            with pytest.raises(RuntimeError):
                with wandb.init(project='test') as run:
                    run.log({'step': 1})
                    raise RuntimeError('training crashed')

            mock_op.finish.assert_called_once_with(code=1)

    def test_reinit_finishes_previous_run(self):
        """Calling init() twice finishes the first run before creating second."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            op1 = _make_mock_op(id=1)
            op2 = _make_mock_op(id=2)
            mock_init.side_effect = [op1, op2]

            wandb.init(project='test')
            run2 = wandb.init(project='test', reinit=True)

            # First run was finished before second started
            op1.finish.assert_called_once()
            assert run2.id == '2'

            wandb.finish()

    def test_watch_forwards_to_op(self):
        """wandb.watch() forwards model and log_freq to op.watch."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            wandb.init(project='test')

            model = MagicMock()
            wandb.watch(model, log_freq=100)

            mock_op.watch.assert_called_once_with(model, log_freq=100)

            wandb.finish()

    def test_alert_forwards_to_op(self):
        """wandb.alert() forwards title, text, level to op.alert."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            wandb.init(project='test')
            wandb.alert(title='Divergence', text='Loss is NaN', level='ERROR')

            mock_op.alert.assert_called_once_with(
                title='Divergence',
                message='Loss is NaN',
                level='ERROR',
            )

            wandb.finish()

    def test_full_realistic_workflow(self):
        """End-to-end: init with config+tags, log mixed data, mutate config,
        update summary, finish. Asserts complete call sequence."""
        import argparse

        import pluto.compat.wandb as wandb
        from pluto.compat.wandb.data_types import Image

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op(tags=['experiment'])
            mock_init.return_value = mock_op

            # 1. Init with argparse config and tags
            args = argparse.Namespace(lr=0.001, batch_size=64)
            run = wandb.init(
                project='cifar10',
                name='resnet-run-1',
                config=args,
                tags=['experiment'],
            )

            # 2. Update config post-init
            wandb.config.update({'scheduler': 'cosine'})

            # 3. Training loop with mixed types
            for epoch in range(3):
                wandb.log(
                    {
                        'train': {'loss': 1.0 / (epoch + 1)},
                        'epoch': epoch,
                    }
                )

            # 4. Log an image
            img = Image(np.zeros((4, 4, 3), dtype=np.uint8))
            wandb.log({'sample': img})

            # 5. Manual summary
            run.summary['best_epoch'] = 2

            # 6. Finish
            wandb.finish()

            # --- Assertions ---

            # init forwarded correctly
            init_kw = mock_init.call_args[1]
            assert init_kw['project'] == 'cifar10'
            assert init_kw['name'] == 'resnet-run-1'
            assert init_kw['tags'] == ['experiment']
            assert init_kw['config']['lr'] == 0.001
            assert init_kw['config']['batch_size'] == 64

            # config mutation synced
            mock_op.update_config.assert_any_call({'scheduler': 'cosine'})

            # 3 training steps + 1 image = 4 log calls
            assert mock_op.log.call_count == 4

            # Training steps flattened correctly
            for i in range(3):
                call_data = mock_op.log.call_args_list[i][0][0]
                assert 'train/loss' in call_data
                assert 'epoch' in call_data

            # Image converted to pluto type
            img_call = mock_op.log.call_args_list[3][0][0]
            assert img_call['sample'].__class__.__module__.startswith('pluto.file')

            # Summary state
            assert run.summary['best_epoch'] == 2
            assert run.summary['epoch'] == 2
            assert run.summary['train/loss'] == pytest.approx(1.0 / 3)

            # finish called
            mock_op.finish.assert_called_once()

    def test_module_state_reset_after_finish(self):
        """After finish(), module-level config/summary/run are reset."""
        import pluto.compat.wandb as wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            wandb.init(project='test', config={'lr': 0.01})
            assert wandb.run is not None
            assert 'lr' in wandb.config

            wandb.finish()

            assert wandb.run is None
            assert len(wandb.config) == 0
            assert len(wandb.summary) == 0

    def test_log_artifact_call_sequence(self, tmp_path):
        """log_artifact with Artifact object produces one op.log per file."""
        import pluto.compat.wandb as wandb
        from pluto.compat.wandb.data_types import Artifact

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            wandb.init(project='test')

            f1 = tmp_path / 'weights.pt'
            f2 = tmp_path / 'config.json'
            f1.write_bytes(b'\x00' * 50)
            f2.write_text('{}')

            art = Artifact('checkpoint', type='model')
            art.add_file(str(f1), name='weights.pt')
            art.add_file(str(f2), name='config.json')
            wandb.log_artifact(art)

            # Each file in the artifact produces one op.log call
            assert mock_op.log.call_count == 2

            wandb.finish()


class TestTopLevelWandbPackage:
    """Tests that ``import wandb`` resolves to the pluto shim and that
    all common import patterns used in real wandb code work."""

    def test_import_wandb(self):
        """Plain ``import wandb`` works and exposes the core API."""
        import wandb

        assert hasattr(wandb, 'init')
        assert hasattr(wandb, 'log')
        assert hasattr(wandb, 'finish')
        assert hasattr(wandb, 'watch')
        assert hasattr(wandb, 'config')
        assert hasattr(wandb, 'summary')
        assert hasattr(wandb, 'run')
        assert hasattr(wandb, 'Image')
        assert hasattr(wandb, 'Table')
        assert hasattr(wandb, 'Histogram')
        assert hasattr(wandb, 'Audio')
        assert hasattr(wandb, 'Video')
        assert hasattr(wandb, 'Html')
        assert hasattr(wandb, 'Artifact')
        assert hasattr(wandb, 'AlertLevel')
        assert hasattr(wandb, 'Api')

    def test_from_wandb_import_init(self):
        """``from wandb import init, log, finish`` works."""
        from wandb import finish, init, log  # noqa: F401

    def test_from_wandb_import_data_types(self):
        """``from wandb import Image, Table, ...`` works."""
        from wandb import (  # noqa: F401
            AlertLevel,
            Artifact,
            Audio,
            Histogram,
            Html,
            Image,
            Table,
            Video,
        )

    def test_from_wandb_import_api(self):
        """``from wandb import Api`` works."""
        from wandb import Api  # noqa: F401

        api = Api()
        with pytest.raises(NotImplementedError):
            api.runs()

    def test_wandb_sdk_import(self):
        """``import wandb.sdk`` works."""
        import wandb.sdk  # noqa: F401

        assert hasattr(wandb.sdk, 'init')

    def test_wandb_sdk_data_types_import(self):
        """``from wandb.sdk.data_types import Image`` works."""
        from wandb.sdk.data_types import Image  # noqa: F401

    def test_wandb_data_types_import(self):
        """``from wandb.data_types import Table`` works."""
        from wandb.data_types import Table  # noqa: F401

    def test_wandb_plot_import(self):
        """``from wandb import plot; wandb.plot.line_series(...)`` works."""
        import wandb.plot

        # Should be no-ops, not errors
        result = wandb.plot.line_series([1, 2], [[1, 2]], title='test')
        assert result is None
        assert wandb.plot.confusion_matrix() is None
        assert wandb.plot.roc_curve() is None
        assert wandb.plot.pr_curve() is None

    def test_wandb_apis_import(self):
        """``from wandb.apis import Api`` works."""
        from wandb.apis import Api  # noqa: F401

    def test_wandb_util_import(self):
        """``from wandb.util import generate_id`` works."""
        from wandb.util import generate_id

        rid = generate_id()
        assert isinstance(rid, str)
        assert len(rid) == 8

    def test_wandb_login(self):
        """``wandb.login()`` returns True (no-op)."""
        import wandb

        assert wandb.login() is True

    def test_wandb_init_log_finish_e2e(self):
        """Full workflow through top-level ``import wandb``."""
        import wandb

        with mock.patch('pluto.init') as mock_init:
            mock_op = _make_mock_op()
            mock_init.return_value = mock_op

            run = wandb.init(project='test-shim')
            wandb.config.lr = 0.01
            wandb.log({'loss': 0.5})
            wandb.log({'loss': 0.3})
            assert run.summary['loss'] == pytest.approx(0.3)
            wandb.finish()

            mock_init.assert_called_once()
            assert mock_init.call_args[1]['project'] == 'test-shim'
            assert mock_op.log.call_count == 2
            mock_op.finish.assert_called_once()

    def test_wandb_settings_class(self):
        """``wandb.Settings(...)`` works."""
        import wandb

        s = wandb.Settings(mode='offline')
        assert s.mode == 'offline'

    def test_wandb_integration_lightning_import(self):
        """``from wandb.integration.lightning import WandbLogger`` works."""
        try:
            from wandb.integration.lightning import WandbLogger  # noqa: F401
        except ImportError:
            # Lightning not installed — that's fine, the import path itself resolved
            pytest.skip('lightning not installed')
