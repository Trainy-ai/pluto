#!/usr/bin/env python3
"""wandb API coverage report.

Compares real wandb (from site-packages) against the pluto shim + registry
to produce a coverage report.

Usage:
    python scripts/wandb_coverage.py --format summary
    python scripts/wandb_coverage.py --format markdown
    python scripts/wandb_coverage.py --format json
"""

import argparse
import importlib
import inspect
import json
import os
import sys
from pathlib import Path


def _import_real_wandb():
    """Import real wandb from site-packages, bypassing local shim."""
    repo_root = str(Path(__file__).resolve().parent.parent)

    # Remove repo root and wandb shim from sys.path
    clean_path = [
        p for p in sys.path
        if os.path.realpath(p) != os.path.realpath(repo_root)
    ]

    # Clear any cached wandb modules
    to_clear = [
        k for k in sys.modules
        if k == "wandb" or k.startswith("wandb.")
    ]
    for k in to_clear:
        del sys.modules[k]

    old_path = sys.path[:]
    sys.path = clean_path
    try:
        real_wandb = importlib.import_module("wandb")
        return real_wandb
    except ImportError:
        return None
    finally:
        sys.path = old_path
        # Clean up so subsequent imports use local shim again
        to_clear = [
            k for k in sys.modules
            if k == "wandb" or k.startswith("wandb.")
        ]
        for k in to_clear:
            del sys.modules[k]


def _get_real_wandb_all(real_wandb):
    """Get wandb.__all__ from real wandb."""
    if hasattr(real_wandb, "__all__"):
        return list(real_wandb.__all__)
    return []


def _get_real_run_public(real_wandb):
    """Get public methods/properties of wandb.Run."""
    run_cls = getattr(real_wandb, "Run", None)
    if run_cls is None:
        # Try wandb.sdk.wandb_run.Run
        try:
            from importlib import import_module

            sdk_run = import_module("wandb.sdk.wandb_run")
            run_cls = sdk_run.Run
        except Exception:
            return []

    members = []
    for name in dir(run_cls):
        if name.startswith("_"):
            continue
        members.append(name)
    return sorted(members)


def _get_real_plot_functions(real_wandb):
    """Get public functions from wandb.plot."""
    try:
        plot_mod = importlib.import_module("wandb.plot")
    except ImportError:
        return []

    funcs = []
    for name in dir(plot_mod):
        if name.startswith("_"):
            continue
        obj = getattr(plot_mod, name, None)
        if callable(obj) and not inspect.isclass(obj):
            funcs.append(name)
    return sorted(funcs)


def _get_shim_exports():
    """Get what our shim exports."""
    # Import from pluto.compat.wandb (not the top-level wandb shim)
    from pluto.compat.wandb import __all__ as shim_all

    return list(shim_all)


def _get_shim_run_members():
    """Get public members of our Run class."""
    from pluto.compat.wandb.run import Run

    members = []
    for name in dir(Run):
        if name.startswith("_"):
            continue
        members.append(name)
    return sorted(members)


def _get_shim_plot_functions():
    """Get functions from our plot shim."""
    import wandb.plot as plot_mod

    funcs = []
    for name in dir(plot_mod):
        if name.startswith("_"):
            continue
        obj = getattr(plot_mod, name, None)
        if callable(obj) and not inspect.isclass(obj):
            funcs.append(name)
    return sorted(funcs)


def build_report(real_wandb):
    """Build the coverage comparison report."""
    from pluto.compat.wandb._coverage import WANDB_API_REGISTRY, SupportLevel

    report = {
        "top_level": [],
        "run_methods": [],
        "plot_functions": [],
        "registry_stats": {},
    }

    # --- Top-level ---
    real_all = _get_real_wandb_all(real_wandb) if real_wandb else []
    shim_all = _get_shim_exports()

    all_top = sorted(set(real_all) | set(shim_all))
    for name in all_top:
        key = f"wandb.{name}"
        entry = WANDB_API_REGISTRY.get(key)
        report["top_level"].append({
            "name": name,
            "in_real_wandb": name in real_all,
            "in_shim": name in shim_all,
            "registry_level": entry.level.value if entry else "unregistered",
            "notes": entry.notes if entry else "",
        })

    # --- Run methods ---
    real_run = _get_real_run_public(real_wandb) if real_wandb else []
    shim_run = _get_shim_run_members()

    all_run = sorted(set(real_run) | set(shim_run))
    for name in all_run:
        key = f"wandb.Run.{name}"
        entry = WANDB_API_REGISTRY.get(key)
        report["run_methods"].append({
            "name": name,
            "in_real_wandb": name in real_run,
            "in_shim": name in shim_run,
            "registry_level": entry.level.value if entry else "unregistered",
            "notes": entry.notes if entry else "",
        })

    # --- Plot functions ---
    real_plot = _get_real_plot_functions(real_wandb) if real_wandb else []
    shim_plot = _get_shim_plot_functions()

    all_plot = sorted(set(real_plot) | set(shim_plot))
    for name in all_plot:
        key = f"wandb.plot.{name}"
        entry = WANDB_API_REGISTRY.get(key)
        report["plot_functions"].append({
            "name": name,
            "in_real_wandb": name in real_plot,
            "in_shim": name in shim_plot,
            "registry_level": entry.level.value if entry else "unregistered",
            "notes": entry.notes if entry else "",
        })

    # --- Stats ---
    stats = {level.value: 0 for level in SupportLevel}
    for entry in WANDB_API_REGISTRY.values():
        stats[entry.level.value] += 1
    stats["total"] = len(WANDB_API_REGISTRY)
    report["registry_stats"] = stats

    return report


def format_markdown(report):
    """Format report as markdown."""
    lines = ["# wandb API Coverage Report\n"]

    stats = report["registry_stats"]
    lines.append("## Summary\n")
    lines.append("| Level | Count |")
    lines.append("|-------|-------|")
    for level in ["supported", "partial", "stub", "not_implemented", "missing"]:
        lines.append(f"| {level} | {stats.get(level, 0)} |")
    lines.append(f"| **total** | **{stats['total']}** |")
    lines.append("")

    for section, title in [
        ("top_level", "Top-level API (`wandb.*`)"),
        ("run_methods", "Run API (`wandb.Run.*`)"),
        ("plot_functions", "Plot API (`wandb.plot.*`)"),
    ]:
        items = report[section]
        if not items:
            continue
        lines.append(f"## {title}\n")
        lines.append(
            "| Name | Real wandb | Shim | Status | Notes |"
        )
        lines.append("|------|-----------|------|--------|-------|")
        for item in items:
            real = "yes" if item["in_real_wandb"] else "no"
            shim = "yes" if item["in_shim"] else "no"
            name = item["name"]
            level = item["registry_level"]
            notes = item["notes"]
            lines.append(
                f"| `{name}` | {real} | {shim}"
                f" | {level} | {notes} |"
            )
        lines.append("")

    return "\n".join(lines)


def format_summary(report):
    """One-line summary for CI logs."""
    stats = report["registry_stats"]
    total = stats["total"]
    supported = stats.get("supported", 0)
    partial = stats.get("partial", 0)
    stub = stats.get("stub", 0)
    not_impl = stats.get("not_implemented", 0)
    missing = stats.get("missing", 0)
    return (
        f"wandb coverage: {supported}/{total} supported, "
        f"{partial} partial, {stub} stub, "
        f"{not_impl} not implemented, {missing} missing"
    )


def main():
    parser = argparse.ArgumentParser(description="wandb API coverage report")
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "summary"],
        default="summary",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    real_wandb = _import_real_wandb()
    if real_wandb is None and args.format != "summary":
        print(
            "WARNING: real wandb not installed. "
            "Report will only show shim/registry data.",
            file=sys.stderr,
        )

    report = build_report(real_wandb)

    if args.format == "json":
        output = json.dumps(report, indent=2)
    elif args.format == "markdown":
        output = format_markdown(report)
    else:
        output = format_summary(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + "\n")
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
