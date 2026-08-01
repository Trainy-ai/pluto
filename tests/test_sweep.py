"""Unit tests for pluto.sweep (native hyperparameter sweeps).

Covers the pure combo generators, config validation, sweep() storage, and the
agent() loop's context handling — without touching the network (pluto.init is
not called; the agent's function only reads the active sweep context).
"""

from __future__ import annotations

import sys

import pytest

import pluto

# The pluto.sweep *function* shadows the pluto.sweep *module* attribute, so
# `import pluto.sweep as sw` would bind the function. Grab the real module.
sw = sys.modules['pluto.sweep']


class TestComboGeneration:
    def test_grid_is_cartesian_product(self):
        combos = list(
            sw._grid_combos({'lr': {'values': [0.1, 0.01]}, 'bs': {'values': [16, 32]}})
        )
        assert combos == [
            {'lr': 0.1, 'bs': 16},
            {'lr': 0.1, 'bs': 32},
            {'lr': 0.01, 'bs': 16},
            {'lr': 0.01, 'bs': 32},
        ]

    def test_grid_constant_value(self):
        combos = list(
            sw._grid_combos({'opt': {'value': 'adam'}, 'lr': {'values': [1, 2]}})
        )
        assert combos == [{'opt': 'adam', 'lr': 1}, {'opt': 'adam', 'lr': 2}]

    def test_grid_rejects_range_param(self):
        with pytest.raises(ValueError, match="use method='random'"):
            list(sw._grid_combos({'lr': {'min': 0.0, 'max': 1.0}}))

    def test_random_respects_each_spec(self):
        params = {
            'const': {'value': 'x'},
            'choice': {'values': ['a', 'b']},
            'frange': {'min': 0.0, 'max': 1.0},
            'irange': {'min': 1, 'max': 3},  # both ints -> int sample
        }
        for _ in range(50):
            c = sw._random_combo(params)
            assert c['const'] == 'x'
            assert c['choice'] in ('a', 'b')
            assert 0.0 <= c['frange'] <= 1.0
            assert isinstance(c['irange'], int) and 1 <= c['irange'] <= 3

    def test_random_log_uniform(self):
        for _ in range(50):
            v = sw._sample_param(
                'lr', {'min': 1e-4, 'max': 1e-1, 'distribution': 'log_uniform_values'}
            )
            assert 1e-4 <= v <= 1e-1


class TestValidation:
    def test_bad_method_rejected(self):
        with pytest.raises(ValueError, match='method must be one of'):
            sw._validate_config(
                {'method': 'nope', 'parameters': {'a': {'values': [1]}}}
            )

    def test_missing_parameters_rejected(self):
        with pytest.raises(ValueError, match="non-empty 'parameters'"):
            sw._validate_config({'method': 'grid'})

    def test_bayes_not_implemented(self):
        with pytest.raises(NotImplementedError, match='bayes'):
            pluto.sweep({'method': 'bayes', 'parameters': {'a': {'values': [1]}}})


class TestSweepStorage:
    def test_sweep_returns_id_and_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.setenv('PLUTO_DIR', str(tmp_path))
        sid = pluto.sweep(
            {'method': 'grid', 'parameters': {'a': {'values': [1, 2]}}},
            project='demo',
        )
        assert isinstance(sid, str) and len(sid) == 8
        loaded = sw._load_sweep(sid)
        assert loaded['method'] == 'grid'
        assert loaded['_project'] == 'demo'
        # survives losing the in-memory registry (reads the on-disk copy)
        sw._SWEEP_REGISTRY.pop(sid)
        assert sw._load_sweep(sid)['parameters'] == {'a': {'values': [1, 2]}}

    def test_load_unknown_sweep_raises(self):
        with pytest.raises(ValueError, match='unknown sweep id'):
            sw._load_sweep('doesnotexist')


class TestAgent:
    def test_grid_agent_sets_context_per_combo(self, monkeypatch):
        monkeypatch.setattr(pluto, 'ops', [])
        seen = []

        def fake_fn():
            # the agent must have installed the sampled combo before calling us
            assert sw._active_sweep is not None
            seen.append(dict(sw._active_sweep['config']))

        sid = pluto.sweep(
            {
                'method': 'grid',
                'parameters': {'a': {'values': [1, 2]}, 'b': {'values': [9]}},
            }
        )
        pluto.agent(sid, fake_fn)
        assert seen == [{'a': 1, 'b': 9}, {'a': 2, 'b': 9}]
        assert sw._active_sweep is None  # cleared afterwards

    def test_grid_agent_count_caps_runs(self, monkeypatch):
        monkeypatch.setattr(pluto, 'ops', [])
        n = []
        sid = pluto.sweep(
            {'method': 'grid', 'parameters': {'a': {'values': [1, 2, 3, 4]}}}
        )
        pluto.agent(sid, lambda: n.append(1), count=2)
        assert len(n) == 2

    def test_random_agent_requires_count(self, monkeypatch):
        monkeypatch.setattr(pluto, 'ops', [])
        sid = pluto.sweep(
            {'method': 'random', 'parameters': {'a': {'min': 0.0, 'max': 1.0}}}
        )
        with pytest.raises(ValueError, match='needs count'):
            pluto.agent(sid, lambda: None)

    def test_agent_clears_context_even_if_function_raises(self, monkeypatch):
        monkeypatch.setattr(pluto, 'ops', [])

        def boom():
            raise RuntimeError('training blew up')

        sid = pluto.sweep({'method': 'grid', 'parameters': {'a': {'values': [1]}}})
        pluto.agent(sid, boom)  # must not propagate; context cleared
        assert sw._active_sweep is None
