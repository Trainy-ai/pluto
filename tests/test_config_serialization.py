"""Unit tests for serializing run configs that contain non-JSON-native types.

The motivating case: a user loads a Hydra/OmegaConf config and passes it to
``wandb.init(config=...)`` (intercepted by the pluto wandb shim) or to
``pluto.init(config=...)``. OmegaConf's ``DictConfig`` is not a ``dict``
subclass and is not JSON serializable, so ``json.dumps(config)`` used to raise
``TypeError: Object of type DictConfig is not JSON serializable`` and silently
disable all Pluto logging.

No server or network required.
"""

import json

import pytest

from pluto.sets import Settings

OmegaConf = pytest.importorskip('omegaconf').OmegaConf


def _dict_of_dictconfig():
    """A plain dict whose nested values are still OmegaConf nodes.

    This is what ``dict(cfg)`` produces for a Hydra ``DictConfig``: ``dict()``
    is a *shallow* conversion, so the top-level object becomes a plain ``dict``
    but the nested values remain ``DictConfig`` / ``ListConfig`` nodes. The
    nested ``full_name`` carries an unresolved interpolation so we can assert
    ``resolve=True`` behavior.
    """
    cfg = OmegaConf.create(
        {
            'lr': 0.01,
            'model': {'name': 'resnet', 'full_name': '${model.name}-v2'},
            'wandb': {'project': 'demo', 'enabled': True, 'layers': [1, 2, 3]},
        }
    )
    return dict(cfg)  # top-level dict, nested DictConfig/ListConfig values


def test_start_payload_serializes_nested_dictconfig():
    from pluto.api import make_compat_start_v1

    payload = make_compat_start_v1(_dict_of_dictconfig(), Settings(), info=None)

    decoded = json.loads(payload.decode())
    inner = json.loads(decoded['config'])
    assert inner['model']['name'] == 'resnet'
    assert inner['wandb']['project'] == 'demo'
    assert inner['wandb']['layers'] == [1, 2, 3]
    # Interpolation resolved through the nested DictConfig node.
    assert inner['model']['full_name'] == 'resnet-v2'


def test_update_config_payload_serializes_dictconfig():
    from pluto.api import make_compat_update_config_v1

    cfg = OmegaConf.create({'epochs': 100, 'opt': {'name': 'adamw'}})
    payload = make_compat_update_config_v1(Settings(), cfg)

    decoded = json.loads(payload.decode())
    inner = json.loads(decoded['config'])
    assert inner['epochs'] == 100
    assert inner['opt']['name'] == 'adamw'


def test_to_native_config_deep_converts_and_resolves():
    from pluto.util import to_native_config

    cfg = OmegaConf.create({'a': {'b': 1}, 'ref': '${a.b}', 'items': [1, 2]})
    native = to_native_config(cfg)

    # DictConfig/ListConfig are not dict/list subclasses, so isinstance here
    # confirms the OmegaConf nodes were actually converted to native types.
    assert isinstance(native, dict)
    assert isinstance(native['a'], dict)
    assert isinstance(native['items'], list)
    assert native['ref'] == 1  # interpolation resolved


def test_to_native_config_handles_dict_of_dictconfig():
    from pluto.util import to_native_config

    native = to_native_config(_dict_of_dictconfig())

    assert isinstance(native, dict)
    # No OmegaConf nodes left anywhere; round-trips through json.
    assert json.loads(json.dumps(native))['model']['full_name'] == 'resnet-v2'


def test_to_native_config_passes_through_plain_data():
    from pluto.util import to_native_config

    plain = {'a': 1, 'b': [1, 2, {'c': 3}], 'd': 'x'}
    assert to_native_config(plain) == plain
    assert to_native_config(None) is None


def test_start_payload_does_not_crash_on_unknown_type():
    """A non-serializable, non-OmegaConf value must not disable logging.

    The str() safety net should stringify it instead of raising.
    """
    from pluto.api import make_compat_start_v1

    class Weird:
        def __repr__(self):
            return '<weird>'

    payload = make_compat_start_v1({'obj': Weird()}, Settings(), info=None)
    inner = json.loads(json.loads(payload.decode())['config'])
    assert inner['obj'] == '<weird>'


def test_to_native_config_does_not_raise_on_unresolvable_interpolation():
    """An interpolation that can't be resolved must degrade, not crash.

    Hydra configs routinely carry references that aren't resolvable at the
    moment pluto.init() runs (e.g. unset ${oc.env:VAR}). resolve=True raises
    InterpolationKeyError on these; we must fall back to the unresolved form
    rather than abort init().
    """
    from pluto.util import to_native_config

    cfg = OmegaConf.create({'x': '${nope}'})
    native = to_native_config(cfg)  # must not raise

    assert isinstance(native, dict)
    assert native['x'] == '${nope}'  # preserved literally, not resolved


def test_to_native_config_does_not_raise_on_interpolation_cycle():
    from pluto.util import to_native_config

    cfg = OmegaConf.create({'a': '${b}', 'b': '${a}'})
    native = to_native_config(cfg)  # must not raise

    assert isinstance(native, dict)
    assert set(native) == {'a', 'b'}


def test_start_payload_does_not_crash_on_bad_interpolation():
    """The json.dumps safety net must hold even when to_container() raises."""
    from pluto.api import make_compat_start_v1

    cfg = OmegaConf.create({'x': '${nope}'})
    # Pass the raw DictConfig as a nested value (the api.py default= path).
    payload = make_compat_start_v1({'cfg': cfg}, Settings(), info=None)

    inner = json.loads(json.loads(payload.decode())['config'])
    assert inner['cfg']['x'] == '${nope}'
