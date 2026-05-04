"""
Integration tests for the .pth-based wandb auto-activation hook.

Each test runs in a fresh subprocess so we exercise the real Python startup
path: site.py → zzzz_pluto_wandb_hook.pth → pluto._wandb_hook.install() →
finder on sys.meta_path → import wandb → apply_wandb_patches.

These tests caught nothing in CI before they existed: the unit tests in
test_wandb_compat.py call patched_init directly with mocks, so the .pth and
import-hook layers were entirely uncovered. PR #85 (the original wandb
shim) shipped without any test that would notice if the .pth stopped firing.

Skipped if zzzz_pluto_wandb_hook.pth isn't deployed in site-packages — for
editable installs that means running `bash dev-install.sh` first.
"""

import os
import site
import subprocess
import sys
import sysconfig
import textwrap

import pytest

_CHECK_PATCHED = (
    "print('PATCHED' if 'patched_init' in wandb.init.__qualname__ "
    "else 'NOT_PATCHED')"
)


def _pth_deployed() -> bool:
    # Check both sysconfig.purelib (where dev-install.sh deploys it) and
    # site.getsitepackages() (defensively, in case of unusual layouts).
    candidates = [sysconfig.get_path('purelib'), *site.getsitepackages()]
    return any(
        os.path.exists(os.path.join(d, 'zzzz_pluto_wandb_hook.pth')) for d in candidates
    )


pytestmark = pytest.mark.skipif(
    not _pth_deployed(),
    reason='zzzz_pluto_wandb_hook.pth not in site-packages '
    '(run `bash dev-install.sh` for editable installs)',
)


@pytest.fixture
def fake_wandb(tmp_path):
    """A minimal wandb stand-in that satisfies apply_wandb_patches."""
    pkg_dir = tmp_path / 'fake_pkgs'
    pkg_dir.mkdir()
    (pkg_dir / 'wandb.py').write_text(
        textwrap.dedent("""
            def init(*args, **kwargs):
                return None
            def log(*args, **kwargs):
                pass
            def finish(*args, **kwargs):
                pass
        """)
    )
    return str(pkg_dir)


@pytest.fixture
def empty_home(tmp_path):
    """A clean HOME with no login marker and no keyring file."""
    h = tmp_path / 'home'
    h.mkdir()
    return str(h)


def _run_subprocess(code: str, env_overrides: dict) -> subprocess.CompletedProcess:
    """Run `python -c code` in a clean env with overrides applied.

    Strips PLUTO_*/WANDB_* (test isolation) and COVERAGE_*/PYTEST_* (so the
    subprocess isn't a pytest-cov-instrumented child — that injection has
    been observed to swallow logger output to the captured stderr pipe on
    Python 3.12 in CI).
    """
    skip_prefixes = ('PLUTO_', 'WANDB_', 'COVERAGE_', 'PYTEST_')
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(skip_prefixes) and k != 'DISABLE_WANDB_LOGGING'
    }
    # Preserve any inherited PYTHONPATH so pluto/site-packages still resolves
    if 'PYTHONPATH' in env_overrides and 'PYTHONPATH' in os.environ:
        env_overrides['PYTHONPATH'] = (
            env_overrides['PYTHONPATH'] + os.pathsep + os.environ['PYTHONPATH']
        )
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, '-c', code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_pth_registers_finder_at_startup(empty_home):
    """The .pth runs install() unconditionally; the finder must be on meta_path."""
    code = (
        'import sys; '
        'names = [type(f).__name__ for f in sys.meta_path]; '
        "print('FOUND' if '_PlutoWandbFinder' in names "
        "else 'MISSING:' + ','.join(names))"
    )
    result = _run_subprocess(code, {'HOME': empty_home})
    assert (
        'FOUND' in result.stdout
    ), f'finder not registered. stdout={result.stdout!r} stderr={result.stderr!r}'


def test_import_wandb_patches_when_credentials_present(fake_wandb, empty_home):
    """PLUTO_API_KEY in env satisfies _has_pluto_credentials → patches apply."""
    code = 'import wandb; ' + _CHECK_PATCHED
    result = _run_subprocess(
        code,
        {
            'HOME': empty_home,
            'PYTHONPATH': fake_wandb,
            'PLUTO_API_KEY': 'fake-test-key',
        },
    )
    assert (
        'PATCHED' in result.stdout
    ), f'wandb not patched. stdout={result.stdout!r} stderr={result.stderr!r}'


def test_import_wandb_emits_discoverability_hint_with_no_credentials(
    fake_wandb, empty_home
):
    """No credentials at all → WARNING hint, wandb left unpatched."""
    # Diagnostic pings around `import wandb` so a CI failure tells us whether
    # stderr capture itself is broken or whether the logger output is dropped.
    code = (
        'import logging, sys; '
        'logging.basicConfig(level=logging.WARNING); '
        'sys.stderr.write("DIAG-PRE\\n"); sys.stderr.flush(); '
        'import wandb; '
        'sys.stderr.write("DIAG-POST\\n"); sys.stderr.flush(); '
        + _CHECK_PATCHED
    )
    result = _run_subprocess(
        code,
        {'HOME': empty_home, 'PYTHONPATH': fake_wandb},
    )
    assert 'NOT_PATCHED' in result.stdout, (
        f'wandb should not be patched. stdout={result.stdout!r} '
        f'stderr={result.stderr!r}'
    )
    assert 'no Pluto credentials found' in result.stderr, (
        f'discoverability hint missing. stderr={result.stderr!r}'
    )
    assert 'pluto login' in result.stderr


def test_import_wandb_emits_partial_config_warning(fake_wandb, empty_home):
    """PLUTO_PROJECT without auth → 'config detected but no API key' WARNING."""
    code = (
        'import logging; '
        'logging.basicConfig(level=logging.WARNING); '
        'import wandb; ' + _CHECK_PATCHED
    )
    result = _run_subprocess(
        code,
        {
            'HOME': empty_home,
            'PYTHONPATH': fake_wandb,
            'PLUTO_PROJECT': 'partial-project',
        },
    )
    assert 'NOT_PATCHED' in result.stdout
    assert (
        'Pluto config detected but no API key' in result.stderr
    ), f'partial-config warning missing. stderr={result.stderr!r}'


def test_login_marker_satisfies_credential_check(fake_wandb, empty_home):
    """A `pluto login` marker file alone (no env vars) should activate patches."""
    marker_dir = os.path.join(empty_home, '.pluto')
    os.makedirs(marker_dir)
    open(os.path.join(marker_dir, '.login_ok'), 'w').close()
    code = 'import wandb; ' + _CHECK_PATCHED
    result = _run_subprocess(code, {'HOME': empty_home, 'PYTHONPATH': fake_wandb})
    assert 'PATCHED' in result.stdout, (
        f'login marker should activate patches. stdout={result.stdout!r} '
        f'stderr={result.stderr!r}'
    )
