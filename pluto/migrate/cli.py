"""
CLI for pluto.migrate: `pluto migrate wandb export|load|all`.

This module keeps its imports light — wandb/pyarrow (the 'migrate'
extra) load inside the command handlers, so the top-level `pluto` CLI
works without them and missing deps produce an install hint instead of
an ImportError traceback.
"""

from __future__ import annotations

import argparse
import threading
from typing import Any, Dict, List, Optional

from pluto.migrate import _INSTALL_HINT


def _add_export_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--entity', required=True, help='wandb entity (team/user)')
    parser.add_argument('--project', required=True, help='wandb project name')
    parser.add_argument(
        '--output', required=True, help='directory to stage exported data in'
    )
    parser.add_argument(
        '--wandb-api-key',
        help='wandb API key (default: WANDB_API_KEY / wandb login)',
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
        '--artifact-max-size-mb',
        type=int,
        help='skip artifacts larger than this many MB',
    )
    parser.add_argument(
        '--no-console', action='store_true', help='skip console output.log'
    )
    parser.add_argument(
        '--no-system-metrics',
        action='store_true',
        help='skip GPU/CPU system metrics',
    )
    parser.add_argument(
        '--no-files', action='store_true', help='skip media/file downloads'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='exit non-zero if any run had data that could not be migrated '
        '(unsupported media, string metrics, dropped annotations)',
    )
    parser.add_argument(
        '--download-workers',
        type=int,
        default=16,
        help='concurrent wandb file downloads per run (default: 16). Media-heavy '
        'runs are latency-bound on per-file downloads; raise for more parallelism',
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
        help='Pluto project to load into (default: the wandb project name)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print what would be loaded without creating runs',
    )
    parser.add_argument(
        '--force-resume',
        action='store_true',
        help='re-load runs already marked loaded (resumes via external id; '
        'may duplicate media files)',
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
        help="delete each run's staged files once it's confirmed loaded, so a "
        'large migration does not keep a full duplicate copy on local disk',
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

    p_load = actions.add_parser(
        'load', help='load a staged export directory into Pluto'
    )
    _add_load_flags(p_load)

    p_all = actions.add_parser('all', help='export then load in one go')
    _add_export_flags(p_all)
    _add_load_flags(p_all, with_input=False)


def _run_export(args: argparse.Namespace) -> int:
    try:
        from pluto.migrate.wandb_export import WandbExporter
    except ImportError as e:
        print(f'{_INSTALL_HINT} ({e})')
        return 2

    try:
        exporter = WandbExporter(
            entity=args.entity,
            project=args.project,
            output_dir=args.output,
            api_key=args.wandb_api_key,
            run_ids=args.run_ids,
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
            download_workers=args.download_workers,
        )
    except ValueError as e:
        # e.g. an unparseable --after/--before value.
        print(f'error: {e}')
        return 2
    summary = exporter.export()
    print(
        f'export: {summary["exported"]} exported, {summary["skipped"]} skipped, '
        f'{len(summary["failed"])} failed'
    )
    for failure in summary['failed']:
        print(f'  failed {failure["run_id"]}: {failure["error"]}')

    not_migrated = summary.get('coverage', {}).get('not_migrated', {})
    if not_migrated:
        dropped = ', '.join(f'{v} {k}' for k, v in sorted(not_migrated.items()))
        print(f'  NOT migrated: {dropped}')
        if getattr(args, 'strict', False):
            print('error: --strict set and some data could not be migrated')
            return 2

    return 1 if summary['failed'] else 0


def _make_loader(args: argparse.Namespace, input_dir: Optional[str] = None) -> Any:
    """Build a PlutoLoader from CLI args (raises ImportError if extras missing)."""
    from pluto.migrate.loader import PlutoLoader

    return PlutoLoader(
        input_dir=input_dir if input_dir is not None else args.input,
        dest_project=args.dest_project,
        flush_every=args.flush_every,
        max_pending=args.max_pending,
        dry_run=args.dry_run,
        run_ids=getattr(args, 'run_ids', None),
        force_resume=args.force_resume,
        cleanup=getattr(args, 'cleanup', False),
    )


def _run_load(args: argparse.Namespace, input_dir: Optional[str] = None) -> int:
    try:
        loader = _make_loader(args, input_dir)
    except ImportError as e:
        print(f'{_INSTALL_HINT} ({e})')
        return 2

    summary = loader.load()
    # In dry-run the loader already printed an accurate "would load N" summary;
    # printing "0 loaded" here would misleadingly imply nothing would happen.
    if not args.dry_run:
        print(
            f'load: {summary["loaded"]} loaded, {summary["skipped"]} skipped, '
            f'{len(summary["failed"])} failed'
        )
    for failure in summary['failed']:
        print(f'  failed {failure["run_id"]}: {failure["error"]}')
    return 1 if summary['failed'] else 0


def cmd_migrate(args: argparse.Namespace) -> int:
    if args.action == 'export':
        return _run_export(args)
    if args.action == 'load':
        return _run_load(args)
    if args.action == 'all':
        if args.dry_run:
            print(
                'error: --dry-run is not supported with `all` (it would still '
                'download everything). Run `export` first, then preview with '
                '`load --dry-run`.'
            )
            return 2
        return _run_all(args)
    raise AssertionError(f'unknown action {args.action!r}')


# Seconds a load pass waits for the export to finish before scanning for
# newly-completed runs again. Small enough to keep the pipeline flowing.
_ALL_POLL_SECONDS = 10.0


def _run_all(args: argparse.Namespace) -> int:
    """Export and load concurrently: export stages runs in a background thread
    while the main thread repeatedly loads whichever runs have finished staging.

    Staging is atomic per run (a sentinel is written last) and the loader only
    picks up completed runs, so the two phases pipeline safely — a run is
    uploaded as soon as it's downloaded, instead of waiting for the whole
    export. With --cleanup, each loaded run's staged files are freed, bounding
    peak disk to the un-loaded backlog.
    """
    # Fail fast if the migrate extras are missing (before spawning the export).
    try:
        _make_loader(args, input_dir=args.output)
    except ImportError as e:
        print(f'{_INSTALL_HINT} ({e})')
        return 2

    result: Dict[str, int] = {}

    def _export_worker() -> None:
        result['code'] = _run_export(args)

    export_thread = threading.Thread(target=_export_worker, daemon=True)
    export_thread.start()
    print('migrate: exporting and loading concurrently...')

    loaded_total = 0
    failed: List[Dict[str, str]] = []

    def _load_pass() -> None:
        nonlocal loaded_total
        # Fresh loader each pass so it re-reads loaded_runs.json and skips runs
        # already loaded in an earlier pass (only newly-staged runs load).
        summary = _make_loader(args, input_dir=args.output).load()
        loaded_total += summary['loaded']
        failed.extend(summary['failed'])

    # Keep loading newly-completed runs until the export finishes.
    while export_thread.is_alive():
        _load_pass()
        export_thread.join(timeout=_ALL_POLL_SECONDS)
    # Final pass for runs that finished staging after the last scan.
    _load_pass()

    export_code = result.get('code', 0)
    if export_code == 2:  # missing deps / bad args — nothing to report on load
        return 2
    print(f'migrate all: {loaded_total} loaded, {len(failed)} failed')
    for failure in failed:
        print(f'  failed {failure["run_id"]}: {failure["error"]}')
    return max(export_code, 1 if failed else 0)


def run_migrate(argv: List[str]) -> int:
    """Standalone entry (also used by tests): argv excludes 'migrate'."""
    parser = argparse.ArgumentParser(prog='pluto migrate')
    subparsers = parser.add_subparsers(dest='command', required=True)
    # Reuse the same tree shape as the top-level CLI: migrate -> wandb -> action
    add_migrate_parser(subparsers)
    args = parser.parse_args(['migrate', *argv])
    return cmd_migrate(args)
