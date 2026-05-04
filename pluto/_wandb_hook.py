"""
Import hook that intercepts `import wandb` to enable dual-logging to Pluto.

Loaded via a .pth file at Python startup. Registers a sys.meta_path finder
that, when `import wandb` is executed, loads the real wandb package and then
monkey-patches it to dual-log to Pluto.

Activation:
    The hook itself installs unconditionally when pluto-ml is on the path —
    installing the package is the user's opt-in signal. Whether the
    patches actually fire is decided later, when `import wandb` runs:

      - Credentials available (any of: PLUTO_API_KEY env var, WANDB_API_KEY
        when DISABLE_WANDB_LOGGING=true, the marker written by `pluto login`,
        or the keyring file written by `pluto login`) → patches applied,
        wandb dual-logs to Pluto.
      - No credentials → a one-time discoverability hint is logged
        (pointing at `pluto login` / PLUTO_API_KEY) and wandb runs unpatched.

Project name is no longer required at install time; the runtime falls back
to the `project=` kwarg on wandb.init, then WANDB_PROJECT, then the resolved
wandb run's project attribute.
"""

import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)

_hook_installed = False
_hint_emitted = False

# Mirrors pluto.auth.LOGIN_MARKER_PATH. Duplicated as a literal here so this
# module stays import-free of the rest of pluto at .pth load time.
_LOGIN_MARKER_PATH = os.path.expanduser('~/.pluto/.login_ok')


def _keyring_cfg_path() -> str:
    """
    keyrings.alt.file.PlaintextKeyring storage location, mirrored from
    keyring.util.platform_ so we don't have to import keyring at .pth load
    time. macOS uses Keychain by default (the marker above covers Mac); this
    only matters for Linux/Windows users on the file-based fallback.
    """
    if sys.platform == 'win32':
        root = os.environ.get('LOCALAPPDATA') or os.environ.get('ProgramData') or '.'
        return os.path.join(root, 'Python Keyring', 'keyring_pass.cfg')
    base = os.environ.get('XDG_DATA_HOME') or os.path.expanduser('~/.local/share')
    return os.path.join(base, 'python_keyring', 'keyring_pass.cfg')


def _keyring_cfg_has_pluto() -> bool:
    """Backward compat: detect a `pluto login` done before the marker existed."""
    path = _keyring_cfg_path()
    if not os.path.exists(path):
        return False
    try:
        import configparser

        cp = configparser.RawConfigParser()
        cp.read(path, encoding='utf-8')
        return cp.has_section('pluto')
    except Exception:
        return False


def _has_pluto_credentials() -> bool:
    """True if some Pluto auth source is available without prompting."""
    if os.environ.get('PLUTO_API_KEY'):
        return True
    wandb_disabled = os.environ.get('DISABLE_WANDB_LOGGING', '').lower() in (
        'true',
        '1',
        'yes',
    )
    if wandb_disabled and os.environ.get('WANDB_API_KEY'):
        return True
    if os.path.exists(_LOGIN_MARKER_PATH):
        return True
    if _keyring_cfg_has_pluto():
        return True
    return False


def _has_partial_pluto_signal() -> bool:
    """True if the user set a Pluto env var but has no auth — partial config."""
    return any(
        os.environ.get(v)
        for v in (
            'PLUTO_PROJECT',
            'PLUTO_URL_APP',
            'PLUTO_URL_API',
            'PLUTO_URL_INGEST',
        )
    )


def _emit_discoverability_hint() -> None:
    """Log a one-time hint when wandb is imported but Pluto isn't activated."""
    global _hint_emitted
    if _hint_emitted:
        return
    _hint_emitted = True
    if _has_partial_pluto_signal():
        msg = (
            'pluto.compat.wandb: Pluto config detected but no API key found. '
            'Run `pluto login` (or set PLUTO_API_KEY) to enable dual-logging '
            'to Pluto. Continuing with wandb-only logging.'
        )
    else:
        msg = (
            'pluto.compat.wandb: pluto-ml is installed but no Pluto credentials '
            'found. Run `pluto login` (or set PLUTO_API_KEY) to enable '
            'dual-logging to Pluto. Continuing with wandb-only logging.'
        )
    # logger.warning lets structured-logging consumers (Sentry etc.) capture it,
    # but the hint also goes through plain stderr so it's never lost to a
    # misconfigured logging setup or a test framework that intercepts the
    # logging stream (Python 3.12 was observed to drop the logger output in
    # subprocess-pytest contexts on CI).
    logger.warning(msg)
    print(msg, file=sys.stderr, flush=True)


class _PlutoWandbFinder:
    """
    Meta path finder that intercepts `import wandb` to apply dual-logging patches.

    Uses find_module/load_module (not the newer find_spec/exec_module from PEP 451)
    because the spec-based API doesn't cleanly support "load the real package, then
    patch it" — exec_module runs on a partially-initialized module object, causing
    circular import issues with wandb's internal imports.

    On first `import wandb`, this finder:
    1. Temporarily removes itself from sys.meta_path (to avoid recursion)
    2. Loads the real wandb package via normal import machinery
    3. Decides whether to patch based on credential availability:
       - Credentials present → applies dual-logging monkey-patches
       - No credentials → emits a one-time discoverability hint
    4. Re-inserts itself (for future imports, though wandb is now cached)
    """

    _patching = False

    def find_module(self, fullname, path=None):
        # Only intercept top-level `import wandb`, and only once
        if fullname == 'wandb' and not self._patching:
            return self
        return None

    def load_module(self, fullname):
        # If wandb is already in sys.modules, it's been loaded
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Prevent re-entrant calls
        self._patching = True
        try:
            # Remove ourselves so the real import machinery finds the real wandb
            sys.meta_path.remove(self)
            try:
                real_wandb = importlib.import_module('wandb')
            finally:
                # Always re-insert ourselves
                sys.meta_path.insert(0, self)

            if _has_pluto_credentials():
                try:
                    from pluto.compat.wandb import apply_wandb_patches

                    apply_wandb_patches(real_wandb)
                    logger.info(
                        'pluto._wandb_hook: Successfully patched wandb for '
                        'dual-logging'
                    )
                except Exception as e:
                    logger.warning(
                        f'pluto._wandb_hook: Failed to apply wandb patches: {e}. '
                        f'wandb will work normally without Pluto dual-logging.'
                    )
            else:
                _emit_discoverability_hint()

            return real_wandb
        finally:
            self._patching = False


def install():
    """
    Register the wandb import hook on sys.meta_path.

    Always installs the finder when called — credential resolution is
    deferred until `import wandb` actually runs (see _PlutoWandbFinder).
    This ensures users who run `pluto login` after Python starts (or who
    pass `project=` only as a kwarg) still get dual-logging, and that
    users with no Pluto config see a discoverability hint instead of
    silent inactivity.

    Safe to call multiple times.
    """
    global _hook_installed

    if _hook_installed:
        return

    # If wandb is already imported, the finder is too late. Try to patch in
    # place if credentials are available; otherwise log the hint.
    if 'wandb' in sys.modules:
        if _has_pluto_credentials():
            logger.warning(
                'pluto._wandb_hook: wandb already imported before hook '
                'installation. Attempting to patch existing wandb module.'
            )
            try:
                from pluto.compat.wandb import apply_wandb_patches

                apply_wandb_patches(sys.modules['wandb'])
            except Exception as e:
                logger.warning(
                    f'pluto._wandb_hook: Failed to patch already-imported '
                    f'wandb: {e}'
                )
        else:
            _emit_discoverability_hint()
        _hook_installed = True
        return

    finder = _PlutoWandbFinder()
    sys.meta_path.insert(0, finder)
    _hook_installed = True


def uninstall():
    """Remove the wandb import hook (for testing)."""
    global _hook_installed, _hint_emitted
    sys.meta_path[:] = [
        f for f in sys.meta_path if not isinstance(f, _PlutoWandbFinder)
    ]
    _hook_installed = False
    _hint_emitted = False
