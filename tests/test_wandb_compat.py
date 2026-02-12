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
        assert 'flag' not in s  # bools excluded

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
        import pandas as pd

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
        op.add_tags.assert_called()

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
