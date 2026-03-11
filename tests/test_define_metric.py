"""Tests for Op.define_metric() and Op.get_metric_definition()."""

from unittest.mock import MagicMock


class TestDefineMetric:
    """Tests for pluto core define_metric functionality."""

    def _make_op(self):
        """Create a minimal Op with mocked internals for testing."""
        from pluto.op import Op

        op = object.__new__(Op)
        op._metric_definitions = {}
        op._sync_manager = None
        op._iface = None
        return op

    def test_define_metric_stores_definition(self):
        op = self._make_op()
        result = op.define_metric('loss', summary='min', goal='minimize')
        assert result == {'name': 'loss', 'summary': 'min', 'goal': 'minimize'}
        assert op._metric_definitions['loss'] == result

    def test_define_metric_only_includes_non_none(self):
        op = self._make_op()
        result = op.define_metric('acc')
        assert result == {'name': 'acc'}
        assert 'step_metric' not in result
        assert 'summary' not in result

    def test_define_metric_with_step_metric(self):
        op = self._make_op()
        result = op.define_metric('val/loss', step_metric='epoch')
        assert result['step_metric'] == 'epoch'

    def test_define_metric_with_hidden(self):
        op = self._make_op()
        result = op.define_metric('internal', hidden=True)
        assert result['hidden'] is True

    def test_define_metric_glob_match(self):
        op = self._make_op()
        op.define_metric('val/*', summary='min')
        defn = op.get_metric_definition('val/loss')
        assert defn is not None
        assert defn['summary'] == 'min'

    def test_define_metric_exact_over_glob(self):
        op = self._make_op()
        op.define_metric('val/*', summary='min')
        op.define_metric('val/loss', summary='max')
        defn = op.get_metric_definition('val/loss')
        assert defn['summary'] == 'max'

    def test_get_metric_definition_returns_none_for_unknown(self):
        op = self._make_op()
        assert op.get_metric_definition('unknown') is None

    def test_define_metric_enqueues_to_sync(self):
        op = self._make_op()
        op._sync_manager = MagicMock()
        op.define_metric('loss', summary='min')
        op._sync_manager.enqueue_metric_definition.assert_called_once()
        call_args = op._sync_manager.enqueue_metric_definition.call_args
        assert call_args[0][0] == {'name': 'loss', 'summary': 'min'}

    def test_define_metric_falls_back_to_iface(self):
        op = self._make_op()
        op._iface = MagicMock()
        op.define_metric('acc', summary='max')
        op._iface.update_metric_definitions.assert_called_once_with(
            [{'name': 'acc', 'summary': 'max'}]
        )

    def test_define_metric_iface_error_does_not_raise(self):
        op = self._make_op()
        op._iface = MagicMock()
        op._iface.update_metric_definitions.side_effect = RuntimeError('server down')
        # Should not raise
        result = op.define_metric('loss', summary='min')
        assert result == {'name': 'loss', 'summary': 'min'}
