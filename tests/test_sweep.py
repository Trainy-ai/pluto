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

    def test_bayes_requires_metric(self):
        with pytest.raises(ValueError, match='bayes sweep needs metric'):
            pluto.sweep({'method': 'bayes', 'parameters': {'a': {'values': [1]}}})

    def test_bayes_with_metric_is_accepted(self):
        sid = pluto.sweep(
            {
                'method': 'bayes',
                'metric': {'name': 'loss', 'goal': 'minimize'},
                'parameters': {'a': {'min': 0.0, 'max': 1.0}},
            }
        )
        assert isinstance(sid, str)


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

    def test_agent_exposes_declared_spec(self, monkeypatch):
        # The declared spec (method/metric/search-space) is available during the
        # run so init() can stamp it onto config.sweep; cleared afterwards.
        monkeypatch.setattr(pluto, 'ops', [])
        captured = {}

        def fn():
            captured['declared'] = dict(sw._active_declared)

        sid = pluto.sweep(
            {
                'method': 'grid',
                'metric': {'name': 'loss', 'goal': 'minimize'},
                'parameters': {'a': {'values': [1]}},
            }
        )
        pluto.agent(sid, fn)
        d = captured['declared']
        assert d['id'] == sid
        assert d['method'] == 'grid'
        assert d['metric'] == {'name': 'loss', 'goal': 'minimize'}
        assert d['parameters'] == {'a': {'values': [1]}}
        assert sw._active_declared is None  # cleared after the agent finishes


class TestResume:
    def test_grid_resume_skips_completed_combos(self, monkeypatch):
        monkeypatch.setattr(pluto, 'ops', [])
        import pluto.query as pq

        # two combos (lr=0.1) already COMPLETED in the sweep
        monkeypatch.setattr(
            pq,
            'list_runs',
            lambda project, tags=None, limit=200, offset=0: (
                [
                    {'displayId': 'A', 'status': 'COMPLETED'},
                    {'displayId': 'B', 'status': 'COMPLETED'},
                ]
                if offset == 0
                else []
            ),
        )
        configs = {'A': {'lr': 0.1, 'bs': 16}, 'B': {'lr': 0.1, 'bs': 32}}
        monkeypatch.setattr(
            pq, 'get_run', lambda project, did: {'config': configs[did]}
        )

        seen = []
        sid = pluto.sweep(
            {
                'method': 'grid',
                'parameters': {
                    'lr': {'values': [0.1, 0.01]},
                    'bs': {'values': [16, 32]},
                },
            },
            project='p',
        )
        pluto.agent(
            sid,
            lambda: seen.append(
                (sw._active_sweep['config']['lr'], sw._active_sweep['config']['bs'])
            ),
        )
        assert seen == [(0.01, 16), (0.01, 32)]  # only the not-done combos

    def test_random_resume_runs_remaining_count(self, monkeypatch):
        monkeypatch.setattr(pluto, 'ops', [])
        import pluto.query as pq

        # 3 of a target 5 already done -> run 2 more
        monkeypatch.setattr(
            pq,
            'list_runs',
            lambda project, tags=None, limit=200, offset=0: (
                [{'displayId': str(i), 'status': 'COMPLETED'} for i in range(3)]
                if offset == 0
                else []
            ),
        )
        n = []
        sid = pluto.sweep(
            {'method': 'random', 'parameters': {'a': {'min': 0.0, 'max': 1.0}}},
            project='p',
        )
        pluto.agent(sid, lambda: n.append(1), count=5)
        assert len(n) == 2

    def test_resume_query_failure_runs_everything(self, monkeypatch):
        monkeypatch.setattr(pluto, 'ops', [])
        import pluto.query as pq

        def boom(*a, **k):
            raise RuntimeError('backend down')

        monkeypatch.setattr(pq, 'list_runs', boom)
        n = []
        sid = pluto.sweep(
            {'method': 'grid', 'parameters': {'a': {'values': [1, 2, 3]}}}, project='p'
        )
        pluto.agent(sid, lambda: n.append(1))  # query fails -> run all 3
        assert len(n) == 3


class TestBayes:
    def test_optuna_suggest_honors_spec(self):
        import optuna

        trial = optuna.create_study().ask()
        assert sw._optuna_suggest(trial, 'c', {'value': 5}) == 5
        assert sw._optuna_suggest(trial, 'ch', {'values': ['a', 'b']}) in ('a', 'b')
        iv = sw._optuna_suggest(trial, 'i', {'min': 1, 'max': 3})
        assert isinstance(iv, int) and 1 <= iv <= 3
        fv = sw._optuna_suggest(trial, 'f', {'min': 0.0, 'max': 1.0})
        assert isinstance(fv, float) and 0.0 <= fv <= 1.0

    def test_bayes_runs_count_and_optimizes(self, monkeypatch):
        # Drive _run_bayes without the network by faking the run: objective is
        # (x-0.7)^2, so optuna should home in near x=0.7.
        monkeypatch.setattr(pluto, 'ops', [])
        tried = []

        def fake_run_combo(sweep_id, combo, project, function, metric_name, i, total):
            x = combo['x']
            tried.append(x)
            return (x - 0.7) ** 2

        monkeypatch.setattr(sw, '_run_combo', fake_run_combo)
        sid = pluto.sweep(
            {
                'method': 'bayes',
                'metric': {'name': 'loss', 'goal': 'minimize'},
                'parameters': {'x': {'min': 0.0, 'max': 1.0}},
            }
        )
        pluto.agent(sid, lambda: None, count=25)
        assert len(tried) == 25
        best = min(tried, key=lambda x: (x - 0.7) ** 2)
        assert abs(best - 0.7) < 0.15  # optuna found the neighborhood

    def test_bayes_requires_count(self, monkeypatch):
        monkeypatch.setattr(pluto, 'ops', [])
        sid = pluto.sweep(
            {
                'method': 'bayes',
                'metric': {'name': 'loss', 'goal': 'minimize'},
                'parameters': {'x': {'min': 0.0, 'max': 1.0}},
            }
        )
        with pytest.raises(ValueError, match='needs count'):
            pluto.agent(sid, lambda: None)
