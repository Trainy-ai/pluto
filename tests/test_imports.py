"""Smoke tests to verify all pluto submodules import without errors.

These tests catch cases where an optional dependency (e.g. matplotlib, torch)
is accidentally imported at module level instead of lazily. A top-level import
of an optional dep will crash any downstream package (like pluto-mcp-server)
that is installed in an environment without that dep.
"""

import importlib
import pkgutil

import pluto


def _all_submodules(package):
    """Yield dotted names for every submodule/subpackage under *package*."""
    prefix = package.__name__ + '.'
    for importer, modname, ispkg in pkgutil.walk_packages(
        package.__path__, prefix=prefix
    ):
        yield modname


def test_import_pluto_top_level():
    """Top-level 'import pluto' must succeed."""
    mod = importlib.import_module('pluto')
    assert hasattr(mod, '__version__')


def test_import_all_pluto_submodules():
    """Every submodule under pluto/ must be importable.

    This catches accidental hard imports of optional dependencies
    (matplotlib, torch, etc.) at module scope.
    """
    failures = []
    for modname in _all_submodules(pluto):
        try:
            importlib.import_module(modname)
        except Exception as exc:
            failures.append(f'{modname}: {exc}')

    assert not failures, 'Failed to import submodules:\n' + '\n'.join(failures)


def test_import_mlop_compat():
    """The deprecated mlop compat layer must still import."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        mod = importlib.import_module('mlop')
    assert hasattr(mod, '__version__')
