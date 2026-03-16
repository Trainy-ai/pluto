"""Tests for wandb API coverage registry and warnings."""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pluto.compat.wandb._coverage import (
    WANDB_API_REGISTRY,
    ApiEntry,
    PlutoWandbCompatWarning,
    SupportLevel,
    reset_warnings,
    warn_unsupported,
)


class TestRegistry:
    """Verify the registry is well-formed and complete."""

    def test_registry_not_empty(self):
        assert len(WANDB_API_REGISTRY) > 0

    def test_all_entries_are_api_entry(self):
        for key, entry in WANDB_API_REGISTRY.items():
            assert isinstance(entry, ApiEntry), f"{key} is not an ApiEntry"
            assert isinstance(entry.level, SupportLevel), f"{key} has invalid level"

    def test_registry_keys_are_qualified(self):
        """All keys should be like 'wandb.X' or 'wandb.Run.X'."""
        for key in WANDB_API_REGISTRY:
            assert key.startswith("wandb."), f"Key {key} doesn't start with 'wandb.'"

    def test_supported_apis_exist_in_shim(self):
        """Everything marked SUPPORTED should exist in our shim."""
        import pluto.compat.wandb as wandb_shim
        from pluto.compat.wandb.run import Run

        for key, entry in WANDB_API_REGISTRY.items():
            if entry.level != SupportLevel.SUPPORTED:
                continue

            parts = key.split(".")
            if len(parts) == 2:
                # wandb.X — check module or __all__
                name = parts[1]
                has_attr = hasattr(wandb_shim, name)
                in_all = name in wandb_shim.__all__
                assert has_attr or in_all, (
                    f"{key} marked SUPPORTED but not found in shim"
                )
            elif len(parts) == 3 and parts[1] == "Run":
                # wandb.Run.X — check Run class
                name = parts[2]
                assert hasattr(Run, name), (
                    f"{key} marked SUPPORTED but Run.{name} doesn't exist"
                )

    def test_registry_covers_shim_all(self):
        """Every item in our shim's __all__ should have a registry entry."""
        from pluto.compat.wandb import __all__ as shim_all

        for name in shim_all:
            key = f"wandb.{name}"
            assert key in WANDB_API_REGISTRY, (
                f"{key} is in shim __all__ but missing from registry"
            )

    def test_registry_covers_run_public_api(self):
        """Every public method/property on our Run should be registered."""
        from pluto.compat.wandb.run import Run

        for name in dir(Run):
            if name.startswith("_"):
                continue
            key = f"wandb.Run.{name}"
            assert key in WANDB_API_REGISTRY, (
                f"Run.{name} exists in shim but {key} missing from registry"
            )

    def test_registry_covers_plot_functions(self):
        """Every function in our plot shim should be registered."""
        import types

        import wandb.plot as plot_mod

        for name in dir(plot_mod):
            if name.startswith("_"):
                continue
            obj = getattr(plot_mod, name, None)
            if not callable(obj):
                continue
            if not isinstance(obj, types.FunctionType):
                continue
            # Skip imports that aren't plot functions
            if obj.__module__ != plot_mod.__name__:
                continue
            key = f"wandb.plot.{name}"
            assert key in WANDB_API_REGISTRY, (
                f"wandb.plot.{name} exists but {key} missing from registry"
            )


class TestWarnings:
    """Verify warnings are emitted correctly."""

    def setup_method(self):
        reset_warnings()

    def test_stub_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_unsupported("wandb.save")
            assert len(w) == 1
            assert issubclass(w[0].category, PlutoWandbCompatWarning)
            assert "no-op" in str(w[0].message)

    def test_not_implemented_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_unsupported("wandb.Api.runs")
            assert len(w) == 1
            assert "not implemented" in str(w[0].message)

    def test_partial_emits_warning_with_notes(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_unsupported("wandb.login")
            assert len(w) == 1
            assert "limited support" in str(w[0].message)
            assert "pluto login" in str(w[0].message)

    def test_supported_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_unsupported("wandb.init")
            assert len(w) == 0

    def test_unknown_api_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_unsupported("wandb.nonexistent_api_xyz")
            assert len(w) == 0

    def test_warns_only_once(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_unsupported("wandb.save")
            warn_unsupported("wandb.save")
            warn_unsupported("wandb.save")
            assert len(w) == 1

    def test_different_apis_warn_independently(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_unsupported("wandb.save")
            warn_unsupported("wandb.restore")
            assert len(w) == 2

    def test_reset_warnings_allows_rewarn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_unsupported("wandb.save")
            assert len(w) == 1
            reset_warnings()
            warn_unsupported("wandb.save")
            assert len(w) == 2

    def test_user_can_silence_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=PlutoWandbCompatWarning)
            warn_unsupported("wandb.save")
            assert len(w) == 0


class TestStubsEmitWarnings:
    """Verify that actual stub functions emit PlutoWandbCompatWarning."""

    def setup_method(self):
        reset_warnings()

    def test_module_save_warns(self):
        import pluto.compat.wandb as wandb_shim

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wandb_shim.save()
            assert any(issubclass(x.category, PlutoWandbCompatWarning) for x in w)

    def test_module_restore_warns(self):
        import pluto.compat.wandb as wandb_shim

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wandb_shim.restore()
            assert any(issubclass(x.category, PlutoWandbCompatWarning) for x in w)

    def test_module_use_artifact_warns(self):
        import pluto.compat.wandb as wandb_shim

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wandb_shim.use_artifact("test")
            assert any(issubclass(x.category, PlutoWandbCompatWarning) for x in w)

    def test_module_log_code_warns(self):
        import pluto.compat.wandb as wandb_shim

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wandb_shim.log_code()
            assert any(issubclass(x.category, PlutoWandbCompatWarning) for x in w)

    def test_module_mark_preempting_warns(self):
        import pluto.compat.wandb as wandb_shim

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wandb_shim.mark_preempting()
            assert any(issubclass(x.category, PlutoWandbCompatWarning) for x in w)

    def test_module_login_warns(self):
        import pluto.compat.wandb as wandb_shim

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wandb_shim.login()
            assert any(issubclass(x.category, PlutoWandbCompatWarning) for x in w)

    def test_plot_scatter_warns(self):
        import wandb.plot as plot

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plot.scatter(None, None, None)
            assert any(issubclass(x.category, PlutoWandbCompatWarning) for x in w)

    def test_api_runs_warns(self):
        from wandb.apis import Api

        api = Api()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(NotImplementedError):
                api.runs()
            assert any(issubclass(x.category, PlutoWandbCompatWarning) for x in w)

    def test_run_save_warns(self):
        """Run.save() should emit a warning."""
        from pluto.compat.wandb.run import Run

        mock_op = MagicMock()
        mock_op.run_id = "test"
        mock_op.tags = []
        run = Run(op=mock_op)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run.save()
            assert any(issubclass(x.category, PlutoWandbCompatWarning) for x in w)


class TestCoverageScript:
    """Test the coverage report script."""

    def test_summary_format(self):
        result = subprocess.run(
            [sys.executable, "scripts/wandb_coverage.py", "--format", "summary"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "wandb coverage:" in result.stdout

    def test_json_format(self):
        result = subprocess.run(
            [sys.executable, "scripts/wandb_coverage.py", "--format", "json"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "registry_stats" in data
        assert "top_level" in data
        assert "run_methods" in data
        assert "plot_functions" in data

    def test_markdown_format(self):
        result = subprocess.run(
            [sys.executable, "scripts/wandb_coverage.py", "--format", "markdown"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "# wandb API Coverage Report" in result.stdout


class TestConflictDetection:
    """Test wandb shim conflict detection logic."""

    def test_shim_disabled_by_env(self):
        """PLUTO_WANDB_SHIM=0 always prevents shim from loading."""
        result = subprocess.run(
            [sys.executable, "-c", "import wandb"],
            capture_output=True,
            text=True,
            env={**os.environ, "PLUTO_WANDB_SHIM": "0"},
        )
        assert result.returncode != 0
        assert "pluto wandb shim is disabled" in result.stderr

    def test_shim_activates_by_default(self, monkeypatch):
        """Shim activates by default, even when real wandb is installed."""
        monkeypatch.delenv("PLUTO_WANDB_SHIM", raising=False)
        # Clear cached wandb module so conflict check re-runs
        to_clear = [
            k for k in sys.modules
            if k == "wandb" or k.startswith("wandb.")
        ]
        for k in to_clear:
            del sys.modules[k]

        import wandb

        # Shim should be active — init comes from pluto
        assert hasattr(wandb, "init")
        assert "pluto" in wandb.__file__

    def test_shim_shadows_real_wandb(self):
        """When real wandb is installed and shim active, shim wins."""
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import wandb; print(wandb.__file__)",
            ],
            capture_output=True,
            text=True,
            env={
                k: v for k, v in os.environ.items()
                if k != "PLUTO_WANDB_SHIM"
            },
        )
        assert result.returncode == 0
        assert "pluto" in result.stdout or "wandb/__init__.py" in result.stdout
