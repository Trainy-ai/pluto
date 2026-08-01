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


def sweep(
    config: Dict[str, Any],
    project: Optional[str] = None,
    entity: Optional[str] = None,
) -> str:
    """Create a sweep from a search-space config and return its id.

    ``config`` mirrors wandb's schema — ``method`` (``grid``/``random``),
    ``metric`` (``{"name", "goal"}``), and ``parameters`` (each a
    ``{"values": [...]}`` / ``{"value": x}`` / ``{"min", "max"}`` spec). Pass the
    returned id to :func:`agent`. ``entity`` is accepted for wandb parity and
    currently unused.
    """
    cfg = dict(_validate_config(config))
    if cfg.get('method', 'grid') == 'bayes':
        raise NotImplementedError(
            "sweep method='bayes' isn't supported yet — use 'grid' or 'random'. "
            '(Bayesian search is planned, wrapping optuna.)'
        )
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
    combination (``count`` caps it); ``random`` runs ``count`` sampled
    combinations (``count`` required). Any run the function leaves open is
    finished automatically.
    """
    import pluto

    cfg = _load_sweep(sweep_id)
    method = cfg.get('method', 'grid')
    parameters = cfg['parameters']
    proj = project or cfg.get('_project')

    if method == 'grid':
        combos = list(_grid_combos(parameters))
        if count is not None:
            combos = combos[:count]
    elif method == 'random':
        if count is None:
            raise ValueError(
                "method='random' needs count=<n> in pluto.agent(sweep_id, fn, "
                'count=n) — a random search has no natural end'
            )
        combos = [_random_combo(parameters) for _ in range(count)]
    else:  # bayes — rejected in sweep(), guard here too
        raise NotImplementedError(f'sweep method {method!r} is not supported')

    global _active_sweep
    logger.info(f'{tag}: agent starting {len(combos)} runs for sweep {sweep_id}')
    for i, combo in enumerate(combos):
        _active_sweep = {'id': sweep_id, 'config': combo, 'project': proj}
        before = {id(o) for o in (pluto.ops or [])}
        try:
            function()
        except Exception as e:
            logger.error(
                f'{tag}: sweep {sweep_id} run {i + 1}/{len(combos)} raised '
                f'{type(e).__name__}: {e}'
            )
        finally:
            _active_sweep = None
            # Close any run the function created but didn't finish, so the next
            # combination starts clean.
            for op in list(pluto.ops or []):
                if id(op) not in before and not getattr(op, '_finished', True):
                    try:
                        op.finish()
                    except Exception as e:
                        logger.debug(f'{tag}: error finishing sweep run: {e}')
    logger.info(f'{tag}: agent finished {len(combos)} runs for sweep {sweep_id}')
