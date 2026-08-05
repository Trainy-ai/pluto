"""
CLI for pluto.migrate: `pluto migrate wandb export|load|all`.

Three orthogonal knobs:
  * phase  — the subcommand: export / load / all (both)
  * scope  — which projects: default all, --project (repeatable), --exclude
  * flags  — --workers (parallelism), --cleanup, ...

This module keeps its imports light — wandb/pyarrow (the 'migrate' extra)
load inside the handlers, so the base `pluto` CLI works without them and a
missing dep produces an install hint instead of an ImportError traceback.
"""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pluto.migrate import _INSTALL_HINT

# Seconds a load pass waits for the export to finish before rescanning for
# newly-completed runs (the `all` pipeline).
_ALL_POLL_SECONDS = 10.0


# --------------------------------------------------------------------------- #
# Argument flags
# --------------------------------------------------------------------------- #
def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Scope + parallelism — shared by export/load/all."""
    parser.add_argument(
        '--project',
        action='append',
        dest='projects',
        default=None,
        help='wandb project to migrate (repeatable). Omit to migrate ALL '
        'projects under the entity.',
    )
    parser.add_argument(
        '--exclude',
        action='append',
        default=None,
        help='project to skip when migrating all projects (repeatable)',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='parallelism (default: 4): projects migrated concurrently, and the '
        'per-run file-download budget on export. Each concurrent project holds '
        'its run history + upload queue in memory, so raise this only if the host '
        'has the RAM (~2-4 GB/worker); lower it on small machines.',
    )


def _add_export_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--entity', required=True, help='wandb entity (team/user)')
    parser.add_argument(
        '--output', required=True, help='directory to stage exported data in'
    )
    parser.add_argument(
        '--wandb-api-key', help='wandb API key (default: WANDB_API_KEY / wandb login)'
    )
    parser.add_argument(
        '--run-id',
        action='append',
        dest='run_ids',
        help='only these wandb run ids (repeatable)',
    )
    parser.add_argument('--after', help='only runs created after this ISO date')
    parser.add_argument('--before', help='only runs created before this ISO date')
    parser.add_argument(
        '--no-artifacts', action='store_true', help='skip logged artifacts'
    )
    parser.add_argument(
        '--artifact-max-size-mb', type=int, help='skip artifacts larger than N MB'
    )
    parser.add_argument(
        '--no-console', action='store_true', help='skip console output.log'
    )
    parser.add_argument(
        '--no-system-metrics', action='store_true', help='skip GPU/CPU system metrics'
    )
    parser.add_argument(
        '--no-files', action='store_true', help='skip media/file downloads'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='exit non-zero if any run had data that could not be migrated',
    )


def _add_load_flags(parser: argparse.ArgumentParser, with_input: bool = True) -> None:
    if with_input:
        parser.add_argument(
            '--input', required=True, help='export directory to load from'
        )
        parser.add_argument(
            '--run-id',
            action='append',
            dest='run_ids',
            help='only these wandb run ids (repeatable)',
        )
    parser.add_argument(
        '--dest-project',
        help='Pluto project to load into (default: the wandb project name; '
        'only valid with a single --project)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print what would be loaded without creating runs',
    )
    parser.add_argument(
        '--force-resume',
        action='store_true',
        help='re-load runs already marked loaded (may duplicate media files)',
    )
    parser.add_argument(
        '--flush-every',
        type=int,
        default=500,
        help='steps between sync-queue backpressure checks (default: 500)',
    )
    parser.add_argument(
        '--max-pending',
        type=int,
        default=5000,
        help='max queued records before the loader throttles (default: 5000)',
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help="delete each run's staged files once it's confirmed loaded",
    )


def add_migrate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach the `migrate` subcommand to the top-level pluto CLI."""
    p_migrate = subparsers.add_parser(
        'migrate', help='import historical data from another platform'
    )
    sources = p_migrate.add_subparsers(dest='source', required=True)

    p_wandb = sources.add_parser('wandb', help='migrate from Weights & Biases')
    actions = p_wandb.add_subparsers(dest='action', required=True)

    p_export = actions.add_parser(
        'export', help='download wandb runs to a local staging directory'
    )
    _add_export_flags(p_export)
    _add_common_flags(p_export)

    p_load = actions.add_parser(
        'load', help='load a staged export directory into Pluto'
    )
    _add_load_flags(p_load)
    _add_common_flags(p_load)

    p_all = actions.add_parser('all', help='export + load (one project or all)')
    _add_export_flags(p_all)
    _add_load_flags(p_all, with_input=False)
    _add_common_flags(p_all)


# --------------------------------------------------------------------------- #
# Per-project workers (module-level so they're picklable for ProcessPoolExecutor)
# --------------------------------------------------------------------------- #
def _export_one_project(
    args: argparse.Namespace, project: str, cache_path: Optional[str] = None
) -> int:
    from pluto.migrate.wandb_export import WandbExporter

    try:
        exporter = WandbExporter(
            entity=args.entity,
            project=project,
            output_dir=args.output,
            api_key=args.wandb_api_key,
            run_ids=getattr(args, 'run_ids', None),
            after=args.after,
            before=args.before,
            include_artifacts=not args.no_artifacts,
            artifact_max_bytes=(
                args.artifact_max_size_mb * 1024 * 1024
                if args.artifact_max_size_mb is not None
                else None
            ),
            include_console=not args.no_console,
            include_system=not args.no_system_metrics,
            include_files=not args.no_files,
            download_workers=args.workers,
        )
    except ValueError as e:  # e.g. an unparseable --after/--before
        print(f'[{project}] error: {e}')
        return 2

    summary = exporter.export()
    print(
        f'[{project}] export: {summary["exported"]} exported, '
        f'{summary["skipped"]} skipped, {len(summary["failed"])} failed'
    )
    for failure in summary['failed']:
        print(f'  [{project}] failed {failure["run_id"]}: {failure["error"]}')
    not_migrated = summary.get('coverage', {}).get('not_migrated', {})
    if not_migrated:
        dropped = ', '.join(f'{v} {k}' for k, v in sorted(not_migrated.items()))
        print(f'  [{project}] NOT migrated: {dropped}')
        if getattr(args, 'strict', False):
            print(f'[{project}] --strict: some data could not be migrated')
            return 2
    return 1 if summary['failed'] else 0


def _load_one_project(
    args: argparse.Namespace, project: str, cache_path: Optional[str] = None
) -> int:
    from pluto.migrate.loader import PlutoLoader

    input_dir = getattr(args, 'input', None) or args.output
    summary = PlutoLoader(
        input_dir=input_dir,
        dest_project=args.dest_project,
        flush_every=args.flush_every,
        max_pending=args.max_pending,
        dry_run=args.dry_run,
        run_ids=getattr(args, 'run_ids', None),
        force_resume=args.force_resume,
        cleanup=getattr(args, 'cleanup', False),
        projects=[project],
        cache_path=cache_path,
    ).load()
    if not args.dry_run:
        print(
            f'[{project}] load: {summary["loaded"]} loaded, '
            f'{summary["skipped"]} skipped, {len(summary["failed"])} failed'
        )
    for failure in summary['failed']:
        print(f'  [{project}] failed {failure["run_id"]}: {failure["error"]}')
    return 1 if summary['failed'] else 0


def _all_one_project(
    args: argparse.Namespace, project: str, cache_path: Optional[str] = None
) -> int:
    """Export + load one project concurrently (upload while downloading)."""
    from pluto.migrate.loader import PlutoLoader

    result: Dict[str, int] = {}

    def _export_worker() -> None:
        try:
            result['code'] = _export_one_project(args, project)
        except Exception as e:  # a crashed export must not read as success (code 0)
            print(f'[{project}] export crashed: {type(e).__name__}: {e}')
            result['code'] = 2

    export_thread = threading.Thread(target=_export_worker, daemon=True)
    export_thread.start()

    loaded_total = 0
    failed: List[Dict[str, str]] = []
    # A run that failed once is attempted at most once: skip it in later passes
    # (skip_run_ids below) rather than re-running the identical staged data every
    # poll. That both stops the every-poll re-attempt and keeps `failed` accurate
    # — a skipped run can't later succeed, so it never lingers as a false failure.
    failed_ids: set = set()

    def _load_pass() -> None:
        nonlocal loaded_total
        summary = PlutoLoader(
            input_dir=args.output,
            dest_project=args.dest_project,
            flush_every=args.flush_every,
            max_pending=args.max_pending,
            dry_run=False,
            run_ids=getattr(args, 'run_ids', None),
            skip_run_ids=list(failed_ids),
            force_resume=args.force_resume,
            cleanup=getattr(args, 'cleanup', False),
            projects=[project],
            cache_path=cache_path,
        ).load()
        loaded_total += summary['loaded']
        for f in summary['failed']:
            if f['run_id'] not in failed_ids:  # count each failing run once
                failed_ids.add(f['run_id'])
                failed.append(f)

    while export_thread.is_alive():
        _load_pass()
        export_thread.join(timeout=_ALL_POLL_SECONDS)
    _load_pass()

    export_code = result.get('code', 0)
    if export_code == 2:
        return 2
    print(f'[{project}] all: {loaded_total} loaded, {len(failed)} failed')
    for failure in failed:
        print(f'  [{project}] failed {failure["run_id"]}: {failure["error"]}')
    return max(export_code, 1 if failed else 0)


# --------------------------------------------------------------------------- #
# Project resolution + orchestration
# --------------------------------------------------------------------------- #
def _apply_exclude(projects: List[str], exclude: Optional[List[str]]) -> List[str]:
    excl = set(exclude or [])
    return [p for p in projects if p not in excl]


def _resolve_export_projects(args: argparse.Namespace) -> List[str]:
    """Which wandb projects to export: explicit --project list, else every
    project under the entity. Minus --exclude."""
    if args.projects:
        projects = list(args.projects)
    else:
        from pluto.migrate.wandb_export import list_wandb_projects

        projects = list_wandb_projects(args.entity, args.wandb_api_key)
    return _apply_exclude(projects, args.exclude)


def _resolve_load_projects(args: argparse.Namespace) -> List[str]:
    """Which staged projects to load, from the staging layout
    input_dir/{entity}/{project}/runs/. Minus --exclude, filtered by --project."""
    input_dir = Path(getattr(args, 'input', None) or args.output)
    staged = sorted({d.name for d in input_dir.glob('*/*') if (d / 'runs').is_dir()})
    if args.projects:
        want = set(args.projects)
        staged = [p for p in staged if p in want]
    return _apply_exclude(staged, args.exclude)


def _cache_path_for(args: argparse.Namespace, project: str) -> str:
    """Per-project resume ledger so parallel loaders don't clobber one file."""
    base = Path(getattr(args, 'input', None) or args.output)
    return str(base / f'loaded_runs.{project}.json')


def _run_over_projects(
    worker: Callable[..., int],
    args: argparse.Namespace,
    projects: List[str],
    per_project_cache: bool,
) -> int:
    """Run ``worker`` for each project — in-process for one, else serial (workers<=1)
    or across a process pool (workers>1). ProcessPoolExecutor because pluto.init
    uses global state and can't be run concurrently in one process."""
    if not projects:
        print('migrate: no matching projects')
        return 0

    def _safe(p: str, c: Optional[str]) -> int:
        try:
            return worker(args, p, c)
        except Exception as e:  # clean message, not a raw traceback
            print(f'[{p}] worker crashed: {type(e).__name__}: {e}')
            return 2

    if len(projects) == 1:
        return _safe(projects[0], None)

    workers = max(1, getattr(args, 'workers', 1))
    # One process per project; more workers than projects can't fan out further.
    effective = min(workers, len(projects))
    note = (
        f'workers={effective}'
        if effective == workers
        else f'workers={effective} (requested {workers}, capped to {len(projects)} '
        'projects — each project runs in one process)'
    )
    print(f'migrate: {len(projects)} projects [{", ".join(projects)}]; {note}')

    def cache(p: str) -> Optional[str]:
        return _cache_path_for(args, p) if per_project_cache else None

    codes: List[int] = []
    if effective <= 1:
        for p in projects:
            codes.append(_safe(p, cache(p)))
        return max(codes)

    # max_tasks_per_child (a fresh process per project, so no pluto global state
    # lingers between projects) is Python 3.11+. On 3.10 the pool reuses workers,
    # which is fine — the loader already does many init/finish cycles per process.
    pool_kwargs: Dict[str, Any] = {'max_workers': effective}
    if sys.version_info >= (3, 11):
        pool_kwargs['max_tasks_per_child'] = 1
    with ProcessPoolExecutor(**pool_kwargs) as pool:
        futures = {pool.submit(worker, args, p, cache(p)): p for p in projects}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                codes.append(fut.result())
            except Exception as e:  # a crashed worker must not sink the rest
                print(f'[{p}] worker crashed: {type(e).__name__}: {e}')
                codes.append(2)
    return max(codes) if codes else 0


def _reject_dest_project_for_many(projects: List[str], args) -> bool:
    if len(projects) > 1 and getattr(args, 'dest_project', None):
        print(
            'error: --dest-project cannot be used when migrating multiple projects '
            '(each project keeps its own name). Use a single --project to rename.'
        )
        return True
    return False


def cmd_migrate(args: argparse.Namespace) -> int:
    if args.action == 'export':
        try:
            projects = _resolve_export_projects(args)
        except ImportError as e:
            print(f'{_INSTALL_HINT} ({e})')
            return 2
        return _run_over_projects(
            _export_one_project, args, projects, per_project_cache=False
        )

    if args.action == 'load':
        try:
            from pluto.migrate.loader import PlutoLoader  # noqa: F401
        except ImportError as e:
            print(f'{_INSTALL_HINT} ({e})')
            return 2
        projects = _resolve_load_projects(args)
        if _reject_dest_project_for_many(projects, args):
            return 2
        # Common case: one loader over the whole (optionally filtered) input —
        # handles all staged projects serially. Only split into parallel
        # per-project loaders when there's real parallelism to exploit.
        if getattr(args, 'workers', 1) > 1 and len(projects) > 1:
            return _run_over_projects(
                _load_one_project, args, projects, per_project_cache=True
            )
        return _run_load(args)

    if args.action == 'all':
        if args.dry_run:
            print(
                'error: --dry-run is not supported with `all` (it would still '
                'download everything). Run `export` first, then `load --dry-run`.'
            )
            return 2
        try:
            projects = _resolve_export_projects(args)
        except ImportError as e:
            print(f'{_INSTALL_HINT} ({e})')
            return 2
        if _reject_dest_project_for_many(projects, args):
            return 2
        return _run_over_projects(
            _all_one_project, args, projects, per_project_cache=True
        )

    raise AssertionError(f'unknown action {args.action!r}')


def _run_load(args: argparse.Namespace) -> int:
    """Single loader over the whole input (all staged projects, optionally
    filtered by --project/--exclude). Backward-compatible default load path."""
    try:
        from pluto.migrate.loader import PlutoLoader
    except ImportError as e:
        print(f'{_INSTALL_HINT} ({e})')
        return 2

    summary = PlutoLoader(
        input_dir=args.input,
        dest_project=args.dest_project,
        flush_every=args.flush_every,
        max_pending=args.max_pending,
        dry_run=args.dry_run,
        run_ids=getattr(args, 'run_ids', None),
        force_resume=args.force_resume,
        cleanup=getattr(args, 'cleanup', False),
        projects=getattr(args, 'projects', None),
        exclude_projects=getattr(args, 'exclude', None),
    ).load()
    if not args.dry_run:
        print(
            f'load: {summary["loaded"]} loaded, {summary["skipped"]} skipped, '
            f'{len(summary["failed"])} failed'
        )
    for failure in summary['failed']:
        print(f'  failed {failure["run_id"]}: {failure["error"]}')
    return 1 if summary['failed'] else 0


def run_migrate(argv: List[str]) -> int:
    """Standalone entry (also used by tests): argv excludes 'migrate'."""
    parser = argparse.ArgumentParser(prog='pluto migrate')
    subparsers = parser.add_subparsers(dest='command', required=True)
    add_migrate_parser(subparsers)
    args = parser.parse_args(['migrate', *argv])
    return cmd_migrate(args)
