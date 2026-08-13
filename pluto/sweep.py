"""Hyperparameter sweeps — mirrors the wandb sweep SDK.

A *sweep* runs a training function many times, each with a different combination
of hyperparameters drawn from a search space, to find the best-performing set::

    sweep_id = pluto.sweep(
        {
            "method": "grid",  # grid | random  (bayes: planned, via optuna)
            "metric": {"name": "val_loss", "goal": "minimize"},
            "parameters": {
                "lr": {"values": [0.1, 0.01]},
                "batch_size": {"values": [16, 32]},
            },
        },
        project="demo",
    )

    def train():
        run = pluto.init()          # sampled config injected: run.config["lr"], ...
        ...                         # train with those settings
        run.log({"val_loss": ...})
        run.finish()

    pluto.agent(sweep_id, train, count=4)   # runs the 4 grid combinations

The config schema mirrors wandb's (``method`` / ``metric`` / ``parameters``) so
existing wandb sweep configs work verbatim and wandb sweeps migrate cleanly.

Single-machine model: the "brain" that picks each next combination runs
client-side in :func:`agent` (grid enumerates the search space, random samples
it). Every run started under an agent is tagged ``sweep:<id>`` and carries its
sampled hyperparameters in ``config`` — the same data model the wandb migration
produces — so a sweep dashboard can group and compare runs off of that.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import random
import string
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'sweep'

_VALID_METHODS = ('grid', 'random', 'bayes')

# Set by agent() around each function() call so pluto.init() picks up the sampled
# hyperparameters + the sweep tag. Mirrors how wandb.agent feeds wandb.config.
_active_sweep: Optional[Dict[str, Any]] = None

# The Op that pluto.init() created for the current sweep run — init() hands it
# back here so the agent can read the objective metric (bayes) and finish the
# run, without relying on pluto.ops (which finish() mutates + ids get reused).
_last_run_op: Optional[Any] = None

# The active sweep's *declared* spec (id + method + metric + search space),
# constant for the whole agent() run. init() stamps it onto each run's config
# (as ``config.sweep``) so the server has the real declaration for a native
# sweep — mirroring ``config.wandb.sweep`` for migrated sweeps — instead of
# having to infer method/objective/search-space from the runs.
_active_declared: Optional[Dict[str, Any]] = None

# id -> config, backed by an on-disk copy so an agent in a separate process can
# still load the sweep. (When the backend gains a sweep entity, _store_sweep /
# _load_sweep become the only spots that need to talk to it.)
_SWEEP_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _gen_sweep_id(n: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choice(alphabet) for _ in range(n))


def _sweeps_dir() -> str:
    base = os.environ.get('PLUTO_DIR') or os.path.join(
        os.path.expanduser('~'), '.pluto'
    )
    d = os.path.join(base, 'sweeps')
    os.makedirs(d, exist_ok=True)
    return d


def _store_sweep(sweep_id: str, config: Dict[str, Any]) -> None:
    _SWEEP_REGISTRY[sweep_id] = config
    try:
        with open(os.path.join(_sweeps_dir(), f'{sweep_id}.json'), 'w') as f:
            json.dump(config, f)
    except Exception as e:  # persistence is best-effort; in-process still works
        logger.debug(f'{tag}: could not persist sweep {sweep_id}: {e}')


def _load_sweep(sweep_id: str) -> Dict[str, Any]:
    if sweep_id in _SWEEP_REGISTRY:
        return _SWEEP_REGISTRY[sweep_id]
    path = os.path.join(_sweeps_dir(), f'{sweep_id}.json')
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        _SWEEP_REGISTRY[sweep_id] = cfg
        return cfg
    raise ValueError(
        f'unknown sweep id {sweep_id!r}; call pluto.sweep(...) to create it first'
    )


def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError(f'sweep config must be a dict, got {type(config).__name__}')
    method = config.get('method', 'grid')
    if method not in _VALID_METHODS:
        raise ValueError(
            f'sweep method must be one of {_VALID_METHODS}, got {method!r}'
        )
    if method == 'bayes':
        metric = config.get('metric') or {}
        if not isinstance(metric, dict) or not metric.get('name'):
            raise ValueError(
                "a bayes sweep needs metric={'name': ..., 'goal': ...} to optimize"
            )
    parameters = config.get('parameters')
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("sweep config needs a non-empty 'parameters' dict")
    for name, spec in parameters.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f'parameter {name!r} must map to a dict '
                "(e.g. {'values': [...]} or {'min': .., 'max': ..}), "
                f'got {type(spec).__name__}'
            )
    return config


def _grid_values(name: str, spec: Dict[str, Any]) -> List[Any]:
    if 'value' in spec:
        return [spec['value']]
    if 'values' in spec:
        return list(spec['values'])
    raise ValueError(
        f"grid search needs discrete 'values'/'value' for parameter {name!r}, "
        "but it has a range/distribution — use method='random' instead"
    )


def _grid_combos(parameters: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    names = list(parameters)
    value_lists = [_grid_values(n, parameters[n]) for n in names]
    for combo in itertools.product(*value_lists):
        yield dict(zip(names, combo))


def _sample_param(name: str, spec: Dict[str, Any]) -> Any:
    if 'value' in spec:
        return spec['value']
    if 'values' in spec:
        return random.choice(list(spec['values']))
    dist = spec.get('distribution')
    lo, hi = spec.get('min'), spec.get('max')
    if lo is not None and hi is not None:
        if dist == 'int_uniform' or (
            dist is None and isinstance(lo, int) and isinstance(hi, int)
        ):
            return random.randint(int(lo), int(hi))
        if dist in ('log_uniform_values', 'log_uniform'):
            return math.exp(random.uniform(math.log(lo), math.log(hi)))
        return random.uniform(lo, hi)
    if dist == 'normal':
        return random.gauss(spec.get('mu', 0.0), spec.get('sigma', 1.0))
    raise ValueError(
        f"cannot sample parameter {name!r}: give 'values', 'value', or "
        "'min'/'max' (optionally with 'distribution')"
    )


def _random_combo(parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {name: _sample_param(name, spec) for name, spec in parameters.items()}


def _combo_key(config: Dict[str, Any], swept_names: List[str]) -> tuple:
    """A hashable identity for a combination, over just the swept parameters."""
    return tuple((name, config.get(name)) for name in swept_names)


def _declared_meta(sweep_id: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The declared sweep spec to stamp on each run (id + method/metric/params)."""
    meta: Dict[str, Any] = {'id': sweep_id}
    for key in ('method', 'metric', 'parameters'):
        if key in cfg:
            meta[key] = cfg[key]
    return meta


def _completed_sweep_runs(
    project: Optional[str], sweep_id: str
) -> List[Dict[str, Any]]:
    """List already-COMPLETED runs of this sweep (for resume). Best-effort: any
    query failure (no backend, project not created yet) yields [] — the agent
    then simply runs everything, so resume never breaks a fresh sweep."""
    if not project:
        return []
    import pluto.query as pq

    done: List[Dict[str, Any]] = []
    try:
        offset = 0
        while True:
            page = pq.list_runs(
                project, tags=[f'sweep:{sweep_id}'], limit=200, offset=offset
            )
            if not page:
                break
            done.extend(r for r in page if r.get('status') == 'COMPLETED')
            if len(page) < 200:
                break
            offset += 200
    except Exception as e:
        logger.debug(f'{tag}: could not list sweep {sweep_id} runs (no resume): {e}')
        return []
    return done


def _fetch_run_config(project: Optional[str], run: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch a completed run's config (list_runs omits it) to match grid combos."""
    if not project:
        return {}
    import pluto.query as pq

    rid = run.get('displayId') or run.get('id')
    if rid is None:
        return {}
    try:
        full = pq.get_run(project, rid)
        cfg = full.get('config')
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _fetch_run_metric(
    project: Optional[str], run: Dict[str, Any], metric_name: str
) -> Optional[float]:
    """Fetch a completed run's final value for ``metric_name`` (bayes seeding)."""
    if not project:
        return None
    import pluto.query as pq

    rid = run.get('displayId') or run.get('id')
    if rid is None:
        return None
    try:
        data = pq.get_metrics(project, rid, metric_names=[metric_name])
        vals = [
            p['value']
            for p in (data or [])
            if isinstance(p, dict)
            and p.get('metric') == metric_name
            and isinstance(p.get('value'), (int, float))
        ]
        return vals[-1] if vals else None  # last logged value == the objective
    except Exception:
        return None


def _optuna_distribution(spec: Dict[str, Any]) -> Any:
    """Translate one parameter spec into an optuna distribution (for seeding)."""
    import optuna.distributions as od

    if 'values' in spec:
        return od.CategoricalDistribution(list(spec['values']))
    lo, hi = spec['min'], spec['max']
    dist = spec.get('distribution')
    if dist == 'int_uniform' or (
        dist is None and isinstance(lo, int) and isinstance(hi, int)
    ):
        return od.IntDistribution(int(lo), int(hi))
    log = dist in ('log_uniform_values', 'log_uniform')
    return od.FloatDistribution(float(lo), float(hi), log=log)


def _optuna_suggest(trial: Any, name: str, spec: Dict[str, Any]) -> Any:
    """Ask optuna for the next value of one parameter, honoring its spec."""
    if 'value' in spec:  # a constant — not optimized, just passed through
        return spec['value']
    if 'values' in spec:
        return trial.suggest_categorical(name, list(spec['values']))
    lo, hi = spec.get('min'), spec.get('max')
    if lo is None or hi is None:
        raise ValueError(
            f"cannot optimize parameter {name!r}: give 'values', 'value', or "
            "'min'/'max'"
        )
    dist = spec.get('distribution')
    if dist == 'int_uniform' or (
        dist is None and isinstance(lo, int) and isinstance(hi, int)
    ):
        return trial.suggest_int(name, int(lo), int(hi))
    log = dist in ('log_uniform_values', 'log_uniform')
    return trial.suggest_float(name, float(lo), float(hi), log=log)


def _seed_study(
    study: Any,
    project: Optional[str],
    completed: List[Dict[str, Any]],
    parameters: Dict[str, Any],
    metric_name: str,
) -> int:
    """Replay completed runs' (params -> objective) into the study so a resumed
    bayes search learns from prior results. Best-effort; skips any run whose
    config/objective can't be fetched."""
    import optuna

    # Only params optuna actually optimizes (constants have no distribution).
    opt_params = {n: s for n, s in parameters.items() if 'value' not in s}
    if not project or not opt_params:
        return 0
    seeded = 0
    for run in completed:
        cfg = _fetch_run_config(project, run)
        value = _fetch_run_metric(project, run, metric_name)
        if not cfg or value is None:
            continue
        try:
            params = {n: cfg[n] for n in opt_params if n in cfg}
            if len(params) != len(opt_params):
                continue  # missing a swept param; can't reconstruct the trial
            distributions = {n: _optuna_distribution(opt_params[n]) for n in params}
            study.add_trial(
                optuna.trial.create_trial(
                    params=params, distributions=distributions, value=value
                )
            )
            seeded += 1
        except Exception:
            continue
    return seeded


def _run_combo(
    sweep_id: str,
    combo: Dict[str, Any],
    project: Optional[str],
    function: Callable[[], Any],
    metric_name: Optional[str],
    index: int,
    total: int,
) -> Optional[float]:
    """Run ``function`` once with ``combo`` injected; return the run's objective
    (its final ``metric_name`` value) if asked. Sets/clears the sweep context and
    finishes any run the function left open."""
    global _active_sweep, _last_run_op
    _active_sweep = {'id': sweep_id, 'config': combo, 'project': project}
    _last_run_op = None
    objective: Optional[float] = None
    try:
        function()
    except Exception as e:
        logger.error(
            f'{tag}: sweep {sweep_id} run {index + 1}/{total} raised '
            f'{type(e).__name__}: {e}'
        )
    finally:
        _active_sweep = None
        op = _last_run_op
        _last_run_op = None
        if op is not None:
            if metric_name is not None:
                objective = getattr(op, '_latest_metrics', {}).get(metric_name)
            # Close the run if the function didn't, so the next combo starts clean.
            if not getattr(op, '_finished', True):
                try:
                    op.finish()
                except Exception as e:
                    logger.debug(f'{tag}: error finishing sweep run: {e}')
    return objective


def sweep(
    config: Dict[str, Any],
    project: Optional[str] = None,
    entity: Optional[str] = None,
) -> str:
    """Create a sweep from a search-space config and return its id.

    ``config`` mirrors wandb's schema — ``method`` (``grid``/``random``/
    ``bayes``), ``metric`` (``{"name", "goal"}``; required for ``bayes``), and
    ``parameters`` (each a ``{"values": [...]}`` / ``{"value": x}`` /
    ``{"min", "max"}`` spec). Pass the returned id to :func:`agent`. ``entity``
    is accepted for wandb parity and currently unused.
    """
    cfg = dict(_validate_config(config))
    sweep_id = _gen_sweep_id()
    cfg['_project'] = project  # remember the target project for the agent
    _store_sweep(sweep_id, cfg)
    logger.info(
        f'{tag}: created sweep {sweep_id} (method={cfg.get("method", "grid")}, '
        f'{len(cfg["parameters"])} parameters)'
    )
    return sweep_id


def agent(
    sweep_id: str,
    function: Callable[[], Any],
    count: Optional[int] = None,
    project: Optional[str] = None,
) -> None:
    """Run ``function`` once per hyperparameter combination in the sweep.

    ``function`` takes no arguments and should call :func:`pluto.init` inside —
    the sampled hyperparameters are injected into that run's ``config`` and the
    run is tagged ``sweep:<id>`` (mirrors ``wandb.agent``). ``grid`` runs every
    combination (``count`` caps it); ``random`` and ``bayes`` run ``count``
    combinations (``count`` required). ``bayes`` uses optuna to pick each next
    combination from the results so far. Already-COMPLETED runs are skipped
    (resume); any run the function leaves open is finished automatically.
    """
    cfg = _load_sweep(sweep_id)
    method = cfg.get('method', 'grid')
    parameters = cfg['parameters']
    proj = project or cfg.get('_project')
    metric = cfg.get('metric') or {}
    metric_name = metric.get('name') if isinstance(metric, dict) else None

    # Resume: skip combinations whose run already COMPLETED. Best-effort — needs
    # a project to query; a fresh sweep just sees zero done and runs everything.
    completed = _completed_sweep_runs(proj, sweep_id) if proj else []
    n_done = len(completed)

    global _active_declared
    _active_declared = _declared_meta(sweep_id, cfg)
    try:
        if method == 'bayes':
            _run_bayes(sweep_id, cfg, proj, function, count, completed, metric_name)
            return

        if method == 'grid':
            swept = list(parameters)
            done_keys = {
                _combo_key(rc, swept)
                for r in completed
                if (rc := _fetch_run_config(proj, r))
            }
            combos = [
                c
                for c in _grid_combos(parameters)
                if _combo_key(c, swept) not in done_keys
            ]
            if count is not None:
                combos = combos[:count]  # cap new runs this invocation
        else:  # random
            if count is None:
                raise ValueError(
                    "method='random' needs count=<n> in pluto.agent(sweep_id, fn, "
                    'count=n) — a random search has no natural end'
                )
            remaining = max(0, count - n_done)  # count is the sweep's total target
            combos = [_random_combo(parameters) for _ in range(remaining)]

        if n_done:
            logger.info(
                f'{tag}: resuming sweep {sweep_id}: {n_done} run(s) already done, '
                f'{len(combos)} to go'
            )
        logger.info(f'{tag}: agent starting {len(combos)} runs for sweep {sweep_id}')
        for i, combo in enumerate(combos):
            _run_combo(sweep_id, combo, proj, function, None, i, len(combos))
        logger.info(f'{tag}: agent finished {len(combos)} runs for sweep {sweep_id}')
    finally:
        _active_declared = None


def _run_bayes(
    sweep_id: str,
    cfg: Dict[str, Any],
    project: Optional[str],
    function: Callable[[], Any],
    count: Optional[int],
    completed: List[Dict[str, Any]],
    metric_name: Optional[str],
) -> None:
    """Bayesian search via optuna: ask for a combination, run it, tell optuna the
    objective, repeat — learning as it goes. Resumes by (a) not exceeding the
    total ``count`` and (b) seeding the study with completed runs' results."""
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "sweep method='bayes' needs optuna — install it with "
            '`pip install optuna` (or `pip install "pluto-ml[sweep]"`).'
        )
    if count is None:
        raise ValueError("method='bayes' needs count=<n> in pluto.agent(...)")
    if not metric_name:
        raise ValueError(
            "a bayes sweep needs metric={'name': ..., 'goal': ...} to optimize"
        )
    parameters = cfg['parameters']
    goal = str((cfg.get('metric') or {}).get('goal') or 'minimize').lower()
    direction = 'maximize' if goal.startswith('max') else 'minimize'

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)
    seeded = _seed_study(study, project, completed, parameters, metric_name)
    remaining = max(0, count - len(completed))
    if seeded or completed:
        logger.info(
            f'{tag}: resuming bayes sweep {sweep_id}: {len(completed)} done '
            f'({seeded} seeded into the optimizer), {remaining} to go'
        )
    logger.info(f'{tag}: agent starting {remaining} bayes runs for sweep {sweep_id}')
    for i in range(remaining):
        trial = study.ask()
        combo = {
            name: _optuna_suggest(trial, name, spec)
            for name, spec in parameters.items()
        }
        objective = _run_combo(
            sweep_id, combo, project, function, metric_name, i, remaining
        )
        if objective is None:  # the run didn't log the metric — skip this trial
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            logger.warning(
                f'{tag}: bayes run {i + 1}/{remaining} logged no {metric_name!r}; '
                'optuna cannot learn from it'
            )
        else:
            study.tell(trial, objective)
    logger.info(f'{tag}: agent finished {remaining} bayes runs for sweep {sweep_id}')
