"""
Unit tests for the wandb-to-pluto compatibility shim.

These tests focus on argument-resolution logic in `_make_patched_init`
without requiring a real wandb install or a real Pluto backend. wandb
and pluto are both swapped for `MagicMock`s so we can assert on the
arguments forwarded to `pluto.init`.
"""

from unittest import mock
from unittest.mock import MagicMock

import pytest

from pluto.compat import wandb as wandb_compat


@pytest.fixture
def clean_env(monkeypatch):
    """Strip any project / api-key env vars that could mask kwarg resolution."""
    for var in (
        'PLUTO_PROJECT',
        'MLOP_PROJECT',
        'WANDB_PROJECT',
        'PLUTO_API_KEY',
        'MLOP_API_KEY',
        'WANDB_NAME',
        'WANDB_TAGS',
        'WANDB_NOTES',
        'WANDB_RUN_GROUP',
        'WANDB_JOB_TYPE',
        'PLUTO_RUN_ID',
        'DISABLE_WANDB_LOGGING',
    ):
        monkeypatch.delenv(var, raising=False)


def _make_fake_wandb_run(run_id='wid-1', name='run-1', project=None):
    """Build a MagicMock that looks like a wandb.Run."""
    run = MagicMock()
    run.id = run_id
    run.name = name
    run.tags = None
    run.notes = None
    run.config = {}
    run.project = project
    return run


def _invoke_patched_init(wandb_run, fake_pluto, init_kwargs):
    """
    Run `_make_patched_init` against mock wandb / pluto modules.

    Returns the kwargs that the shim forwarded to `pluto.init`, or
    None if pluto.init was never called.
    """
    fake_wandb_module = MagicMock()
    original_init = MagicMock(return_value=wandb_run)
    patched = wandb_compat._make_patched_init(original_init, fake_wandb_module)

    with mock.patch.object(wandb_compat, '_safe_import_pluto', return_value=fake_pluto):
        result = patched(**init_kwargs)

    if fake_pluto.init.called:
        return result, fake_pluto.init.call_args.kwargs
    return result, None


def test_init_failure_logs_traceback(clean_env, monkeypatch, caplog):
    """When pluto.init fails, the log record must carry the traceback.

    Previously the shim logged only ``type(e).__name__: e``, so users saw
    *what* broke but never *where*. The error must be logged with exc_info so
    the traceback (and the raise-site line number) is surfaced.
    """
    import logging

    monkeypatch.setenv('PLUTO_PROJECT', 'p')

    fake_pluto = MagicMock()
    fake_pluto.init.side_effect = TypeError('boom')
    wandb_run = _make_fake_wandb_run()

    with caplog.at_level(logging.ERROR, logger='pluto.compat.wandb'):
        result, _ = _invoke_patched_init(wandb_run, fake_pluto, {'project': 'p'})

    # Dual-logging disabled => the real wandb run is returned unmodified.
    assert result is wandb_run
    records = [r for r in caplog.records if 'DUAL-LOGGING DISABLED' in r.getMessage()]
    assert records, 'expected a DUAL-LOGGING DISABLED error log'
    assert (
        records[0].exc_info is not None
    ), 'error must be logged with exc_info so the traceback is surfaced'


def test_project_kwarg_is_used_when_no_env_vars_set(clean_env):
    """
    Lightning's `WandbLogger(project="my-project", ...)` forwards
    `project=` to wandb.init. With neither PLUTO_PROJECT nor
    WANDB_PROJECT set, the shim must still pick up the kwarg.
    """
    fake_pluto = MagicMock()
    fake_pluto.init.return_value = MagicMock(id=42)
    wandb_run = _make_fake_wandb_run()

    _, forwarded = _invoke_patched_init(
        wandb_run,
        fake_pluto,
        {'project': 'my-project', 'entity': 'my-team'},
    )

    assert forwarded is not None, 'pluto.init should have been called'
    assert forwarded['project'] == 'my-project'


def test_pluto_project_env_takes_precedence_over_kwarg(clean_env, monkeypatch):
    """PLUTO_PROJECT env var wins over the wandb.init kwarg."""
    monkeypatch.setenv('PLUTO_PROJECT', 'env-project')

    fake_pluto = MagicMock()
    fake_pluto.init.return_value = MagicMock(id=42)
    wandb_run = _make_fake_wandb_run()

    _, forwarded = _invoke_patched_init(
        wandb_run,
        fake_pluto,
        {'project': 'kwarg-project'},
    )

    assert forwarded['project'] == 'env-project'


def test_kwarg_takes_precedence_over_wandb_project_env(clean_env, monkeypatch):
    """
    With PLUTO_PROJECT unset, the explicit kwarg should win over
    WANDB_PROJECT — kwargs are more specific than env-level defaults.
    """
    monkeypatch.setenv('WANDB_PROJECT', 'env-wandb-project')

    fake_pluto = MagicMock()
    fake_pluto.init.return_value = MagicMock(id=42)
    wandb_run = _make_fake_wandb_run()

    _, forwarded = _invoke_patched_init(
        wandb_run,
        fake_pluto,
        {'project': 'kwarg-project'},
    )

    assert forwarded['project'] == 'kwarg-project'


def test_wandb_project_env_used_when_no_kwarg(clean_env, monkeypatch):
    """If no project kwarg is given, fall back to WANDB_PROJECT."""
    monkeypatch.setenv('WANDB_PROJECT', 'env-wandb-project')

    fake_pluto = MagicMock()
    fake_pluto.init.return_value = MagicMock(id=42)
    wandb_run = _make_fake_wandb_run()

    _, forwarded = _invoke_patched_init(wandb_run, fake_pluto, {})

    assert forwarded['project'] == 'env-wandb-project'


def test_falls_back_to_wandb_run_project_attribute(clean_env):
    """
    Final fallback: read `project` from the resolved wandb run object.
    Some wandb integrations don't pass project explicitly but the
    underlying Run still ends up with one set.
    """
    fake_pluto = MagicMock()
    fake_pluto.init.return_value = MagicMock(id=42)
    wandb_run = _make_fake_wandb_run(project='run-attr-project')

    _, forwarded = _invoke_patched_init(wandb_run, fake_pluto, {})

    assert forwarded['project'] == 'run-attr-project'


def test_no_project_anywhere_skips_pluto_init(clean_env):
    """
    With nothing set — no env var, no kwarg, no run attribute — the
    shim must not call pluto.init and must return the original wandb
    run unwrapped (wandb-only mode).
    """
    fake_pluto = MagicMock()
    wandb_run = _make_fake_wandb_run(project=None)

    result, forwarded = _invoke_patched_init(wandb_run, fake_pluto, {})

    assert forwarded is None
    assert result is wandb_run


def test_omegaconf_config_flows_through_shim_and_serializes(clean_env, monkeypatch):
    """End-to-end shape of the real report: a Hydra user calls
    ``wandb.init(config=dict(OmegaConf.load(...)))``.

    ``dict(cfg)`` yields a plain top-level dict with nested ``DictConfig``
    values. Drive the real shim with that exact object and confirm (a) it
    forwards the config to ``pluto.init`` and (b) the forwarded object
    serializes through Pluto's real start-payload pipeline
    (``to_native_config`` -> ``make_compat_start_v1``) — the path that used to
    raise ``TypeError: Object of type DictConfig is not JSON serializable`` and
    disable dual-logging.
    """
    import json

    OmegaConf = pytest.importorskip('omegaconf').OmegaConf
    monkeypatch.setenv('PLUTO_PROJECT', 'p')

    cfg = OmegaConf.create(
        {
            'lr': 0.01,
            'model': {'name': 'resnet', 'full_name': '${model.name}-v2'},
        }
    )
    wandb_config = dict(cfg)  # top-level dict, nested DictConfig values

    fake_pluto = MagicMock()
    fake_pluto.init.return_value = MagicMock(id=1)
    wandb_run = _make_fake_wandb_run()

    _, forwarded = _invoke_patched_init(wandb_run, fake_pluto, {'config': wandb_config})

    assert forwarded is not None, 'pluto.init should have been called'
    assert 'model' in forwarded['config']

    # Serialize exactly as pluto.init() would: normalize, then build payload.
    from pluto.api import make_compat_start_v1
    from pluto.sets import Settings
    from pluto.util import to_native_config

    native = to_native_config(forwarded['config'])
    payload = make_compat_start_v1(native, Settings(), info=None)
    inner = json.loads(json.loads(payload.decode())['config'])
    assert inner['model']['full_name'] == 'resnet-v2'  # interpolation resolved


# ---------------------------------------------------------------------------
# WandbRunWrapper.log value routing
#
# The shim pre-filters each logged value before forwarding to Pluto. These
# tests pin the routing that backs the /resume-crashed-run use case: string
# paths (e.g. checkpoint/r2_path) must reach Pluto as config (latest-wins,
# queryable via get_run().config), and numpy scalars must not be silently
# dropped the way plain str/np values were before.
# ---------------------------------------------------------------------------


def _make_wrapper():
    """Build a WandbRunWrapper with mock wandb/pluto runs (no atexit)."""
    wandb_run = MagicMock()
    wandb_run._step = 7
    pluto_run = MagicMock()
    pluto_module = MagicMock()
    # Avoid registering a real atexit handler during the test.
    with mock.patch.object(wandb_compat.atexit, 'register'):
        wrapper = wandb_compat.WandbRunWrapper(
            wandb_run, pluto_run, pluto_module, wandb_disabled=False
        )
    return wrapper, pluto_run


def test_log_routes_strings_to_config_not_metrics():
    """checkpoint/r2_path (a str) must land in Pluto config, not log()."""
    wrapper, pluto_run = _make_wrapper()

    wrapper.log(
        {
            'checkpoint/step': 100,
            'checkpoint/r2_path': 's3://bucket/run/ckpt-100.pt',
            'checkpoint/local_path': '/nfs/run/ckpt-100.pt',
        }
    )

    # Strings forwarded to config (latest-wins, readable via get_run().config).
    assert pluto_run.update_config.call_count == 1
    cfg = pluto_run.update_config.call_args.args[0]
    assert cfg['checkpoint/r2_path'] == 's3://bucket/run/ckpt-100.pt'
    assert cfg['checkpoint/local_path'] == '/nfs/run/ckpt-100.pt'

    # Numeric value still goes to metrics; strings must NOT be in log().
    logged = pluto_run.log.call_args.args[0]
    assert logged == {'checkpoint/step': 100}
    assert 'checkpoint/r2_path' not in logged


def test_log_failure_unexpected_error_is_loud_then_deduped(caplog):
    """A non-network failure (e.g. OSError) must surface at ERROR, not vanish
    at debug -- that silent debug-swallow is what hid the long-filename bug.
    A recurring identical failure is shouted once, then drops to debug."""
    import logging

    wrapper, pluto_run = _make_wrapper()
    pluto_run.log.side_effect = OSError(36, 'File name too long')

    with caplog.at_level(logging.DEBUG, logger='pluto.compat.wandb'):
        wrapper.log({'loss': 0.1})  # first occurrence
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert 'OSError' in errors[0].message
        # wandb itself is never impacted: the wrapped run.log was still called.
        wrapper._wandb_run.log.assert_called()

        caplog.clear()
        wrapper.log({'loss': 0.2})  # same (context, type) -> deduped to debug
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            r.levelno == logging.DEBUG and 'OSError' in r.message
            for r in caplog.records
        )


def test_log_failure_httpx_error_stays_at_debug(caplog):
    """Transient network errors are noisy and already retried by the sync
    layer, so they must stay at debug even on the first occurrence."""
    import logging

    import httpx

    wrapper, pluto_run = _make_wrapper()
    pluto_run.log.side_effect = httpx.ConnectError('connection refused')

    with caplog.at_level(logging.DEBUG, logger='pluto.compat.wandb'):
        wrapper.log({'loss': 0.1})

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        'Failed to log media/metrics' in r.message
        for r in caplog.records
        if r.levelno == logging.DEBUG
    )


def test_log_forwards_numpy_scalars_as_metrics():
    """np.int64/np.float32 must reach Pluto metrics, not be dropped."""
    np = pytest.importorskip('numpy')
    wrapper, pluto_run = _make_wrapper()

    wrapper.log(
        {
            'checkpoint/step': np.int64(100),
            'loss': np.float32(0.5),
        }
    )

    logged = pluto_run.log.call_args.args[0]
    assert logged['checkpoint/step'] == 100
    assert isinstance(logged['checkpoint/step'], int)  # .item() -> python int
    assert abs(logged['loss'] - 0.5) < 1e-6
    assert isinstance(logged['loss'], float)


def test_log_forwards_any_item_scalar_like_pluto_core():
    """Any scalar exposing .item() is forwarded, matching Pluto's own log().

    Guards against the shim being stricter than op._process_log_item_sync:
    e.g. an ``epoch`` that arrives as a 0-d-tensor-like wrapper rather than a
    plain int must still reach Pluto instead of being silently dropped.
    """
    wrapper, pluto_run = _make_wrapper()

    class _ScalarLike:
        def __init__(self, v):
            self._v = v

        def item(self):
            return self._v

    wrapper.log({'checkpoint/epoch': _ScalarLike(12)})

    logged = pluto_run.log.call_args.args[0]
    assert logged == {'checkpoint/epoch': 12}


def test_log_does_not_treat_failing_item_as_scalar():
    """A non-scalar whose .item() raises must not crash or produce a metric."""
    wrapper, pluto_run = _make_wrapper()

    class _MultiElement:
        def item(self):
            raise ValueError('can only convert an array of size 1')

    wrapper.log({'weird': _MultiElement()})

    assert not pluto_run.log.called
    assert not pluto_run.update_config.called


def test_unforwardable_value_alerts_sentry_once_not_user():
    """An unmappable value alerts Sentry (maintainers) once — not the user."""
    wrapper, pluto_run = _make_wrapper()

    class _Opaque:
        """Not numeric, not media, not JSON-serializable."""

        def item(self):
            raise ValueError('not a scalar')

    with mock.patch('pluto.sentry.capture_message') as cap:
        wrapper.log({'mystery': _Opaque()})
        wrapper.log({'mystery': _Opaque()})  # second time: no duplicate alert

    # Exactly one maintainer-facing Sentry alert, grouped by type name.
    assert cap.call_count == 1
    assert '_Opaque' in cap.call_args.args[0]
    assert cap.call_args.kwargs.get('level') == 'warning'
    # Nothing forwarded to the user's run, and no user-facing exception.
    assert not pluto_run.log.called
    assert not pluto_run.update_config.called


def test_json_serializable_unmapped_value_falls_back_to_config():
    """A dict/None with no metric mapping is preserved as config, not dropped."""
    wrapper, pluto_run = _make_wrapper()

    wrapper.log({'meta/info': {'kind': 'resume', 'attempt': 3}, 'note': None})

    cfg = pluto_run.update_config.call_args.args[0]
    assert cfg['meta/info'] == {'kind': 'resume', 'attempt': 3}
    assert cfg['note'] is None
    assert not pluto_run.log.called  # no numeric metrics in this call


def test_log_skips_redundant_config_updates():
    """An unchanged str/bool config value must not re-trigger update_config."""
    wrapper, pluto_run = _make_wrapper()

    # First log: config is synced.
    wrapper.log({'phase': 'train', 'loss': 0.5})
    assert pluto_run.update_config.call_count == 1
    assert pluto_run.update_config.call_args.args[0] == {'phase': 'train'}

    # Same config value again: update_config must NOT be called.
    pluto_run.update_config.reset_mock()
    wrapper.log({'phase': 'train', 'loss': 0.4})
    assert pluto_run.update_config.call_count == 0

    # Changed config value: update_config is called again, with only the change.
    wrapper.log({'phase': 'val', 'loss': 0.3})
    assert pluto_run.update_config.call_count == 1
    assert pluto_run.update_config.call_args.args[0] == {'phase': 'val'}


def test_omegaconf_value_falls_back_to_config_not_dropped():
    """A logged OmegaConf node is storable as config (not Sentry-dropped)."""
    OmegaConf = pytest.importorskip('omegaconf').OmegaConf
    wrapper, pluto_run = _make_wrapper()

    cfg_node = OmegaConf.create({'lr': 0.01, 'sched': {'name': 'cosine'}})

    with mock.patch('pluto.sentry.capture_message') as cap:
        wrapper.log({'hparams': cfg_node})

    # Stored as config, deep-converted to native containers; not dropped.
    cfg = pluto_run.update_config.call_args.args[0]
    assert cfg['hparams'] == {'lr': 0.01, 'sched': {'name': 'cosine'}}
    assert not cap.called  # OmegaConf is storable -> no maintainer alert


# ---------------------------------------------------------------------------
# Media caption forwarding: wandb.Image/Audio/Video(caption=...) must reach
# pluto.Image/Audio/Video(caption=...). Regression for the shim dropping it.
# ---------------------------------------------------------------------------
def _fake_wandb_media(type_name, path, caption=None):
    """An object whose type().__name__ matches what the shim dispatches on."""
    cls = type(type_name, (), {})
    obj = cls()
    obj._path = path
    obj._caption = caption
    return obj


@pytest.mark.parametrize('type_name', ['Image', 'Audio', 'Video'])
def test_wandb_media_caption_forwarded(type_name):
    """caption= on a wandb media object is forwarded to the pluto equivalent."""
    pluto_module = MagicMock()
    media = _fake_wandb_media(type_name, '/tmp/x.png', caption='a fluffy cat')

    wandb_compat._convert_wandb_to_pluto('eval/images', media, pluto_module)

    factory = getattr(pluto_module, type_name)
    factory.assert_called_once()
    assert factory.call_args.kwargs.get('caption') == 'a fluffy cat'


@pytest.mark.parametrize('type_name', ['Image', 'Audio', 'Video'])
def test_wandb_media_without_caption(type_name):
    """No caption on the wandb object forwards caption=None (not a crash)."""
    pluto_module = MagicMock()
    media = _fake_wandb_media(type_name, '/tmp/x.png', caption=None)

    wandb_compat._convert_wandb_to_pluto('eval/images', media, pluto_module)

    factory = getattr(pluto_module, type_name)
    factory.assert_called_once()
    assert factory.call_args.kwargs.get('caption') is None
