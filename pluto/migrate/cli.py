"""
CLI for pluto.migrate: `pluto migrate wandb export|load|all`.

This module keeps its imports light — wandb/pyarrow (the 'migrate'
extra) load inside the command handlers, so the top-level `pluto` CLI
works without them and missing deps produce an install hint instead of
an ImportError traceback.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

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
    )
    summary = exporter.export()
    print(
        f'export: {summary["exported"]} exported, {summary["skipped"]} skipped, '
        f'{len(summary["failed"])} failed'
    )
    for failure in summary['failed']:
        print(f'  failed {failure["run_id"]}: {failure["error"]}')
    return 1 if summary['failed'] else 0


def _run_load(args: argparse.Namespace, input_dir: Optional[str] = None) -> int:
    try:
        from pluto.migrate.loader import PlutoLoader
    except ImportError as e:
        print(f'{_INSTALL_HINT} ({e})')
        return 2

    loader = PlutoLoader(
        input_dir=input_dir if input_dir is not None else args.input,
        dest_project=args.dest_project,
        flush_every=args.flush_every,
        max_pending=args.max_pending,
        dry_run=args.dry_run,
        run_ids=getattr(args, 'run_ids', None),
        force_resume=args.force_resume,
    )
    summary = loader.load()
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
        export_code = _run_export(args)
        if export_code == 2:  # missing deps — nothing was staged
            return export_code
        # Per-run export failures must not block loading the runs that DID
        # stage successfully; both phases are independently resumable.
        load_code = _run_load(args, input_dir=args.output)
        return max(export_code, load_code)
    raise AssertionError(f'unknown action {args.action!r}')


def run_migrate(argv: List[str]) -> int:
    """Standalone entry (also used by tests): argv excludes 'migrate'."""
    parser = argparse.ArgumentParser(prog='pluto migrate')
    subparsers = parser.add_subparsers(dest='command', required=True)
    # Reuse the same tree shape as the top-level CLI: migrate -> wandb -> action
    add_migrate_parser(subparsers)
    args = parser.parse_args(['migrate', *argv])
    return cmd_migrate(args)
