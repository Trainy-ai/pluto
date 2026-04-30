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
