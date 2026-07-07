"""
Unit tests for the `pluto migrate wandb` CLI (pluto.migrate.cli).

The CLI is thin arg-parsing over WandbExporter/PlutoLoader; both are
mocked here. Heavy deps (wandb/pyarrow) must only be imported inside
command handlers so `pluto --help` stays light — pinned by the
subprocess help test.
"""

from __future__ import annotations

import subprocess
import sys
from unittest import mock

from pluto.migrate.cli import run_migrate


def _mock_exporter(summary=None):
    exporter = mock.MagicMock()
    exporter.export.return_value = summary or {
        'exported': 1,
        'skipped': 0,
        'failed': [],
    }
    return exporter


def _mock_loader(summary=None):
    loader = mock.MagicMock()
    loader.load.return_value = summary or {'loaded': 1, 'skipped': 0, 'failed': []}
    return loader


class TestMigrateCli:
    def test_export_wires_flags_to_exporter(self, tmp_path):
        exporter = _mock_exporter()
        with mock.patch(
            'pluto.migrate.wandb_export.WandbExporter', return_value=exporter
        ) as cls:
            code = run_migrate(
                [
                    'wandb',
                    'export',
                    '--entity',
                    'acme',
                    '--project',
                    'vision',
                    '--output',
                    str(tmp_path),
                    '--run-id',
                    'r1',
                    '--run-id',
                    'r2',
                    '--after',
                    '2025-01-01',
                    '--no-artifacts',
                    '--artifact-max-size-mb',
                    '512',
                ]
            )
        assert code == 0
        kwargs = cls.call_args.kwargs
        assert kwargs['entity'] == 'acme'
        assert kwargs['project'] == 'vision'
        assert kwargs['run_ids'] == ['r1', 'r2']
        assert kwargs['after'] == '2025-01-01'
        assert kwargs['include_artifacts'] is False
        assert kwargs['artifact_max_bytes'] == 512 * 1024 * 1024
        exporter.export.assert_called_once()

    def test_load_wires_flags_to_loader(self, tmp_path):
        loader = _mock_loader()
        with mock.patch(
            'pluto.migrate.loader.PlutoLoader', return_value=loader
        ) as cls:
            code = run_migrate(
                [
                    'wandb',
                    'load',
                    '--input',
                    str(tmp_path),
                    '--dest-project',
                    'legacy',
                    '--dry-run',
                ]
            )
        assert code == 0
        kwargs = cls.call_args.kwargs
        assert kwargs['dest_project'] == 'legacy'
        assert kwargs['dry_run'] is True
        loader.load.assert_called_once()

    def test_all_exports_then_loads(self, tmp_path):
        exporter, loader = _mock_exporter(), _mock_loader()
        with (
            mock.patch(
                'pluto.migrate.wandb_export.WandbExporter', return_value=exporter
            ),
            mock.patch('pluto.migrate.loader.PlutoLoader', return_value=loader),
        ):
            code = run_migrate(
                [
                    'wandb',
                    'all',
                    '--entity',
                    'acme',
                    '--project',
                    'vision',
                    '--output',
                    str(tmp_path),
                ]
            )
        assert code == 0
        exporter.export.assert_called_once()
        loader.load.assert_called_once()

    def test_failures_produce_nonzero_exit(self, tmp_path):
        exporter = _mock_exporter(
            {'exported': 0, 'skipped': 0, 'failed': [{'run_id': 'x', 'error': 'e'}]}
        )
        with mock.patch(
            'pluto.migrate.wandb_export.WandbExporter', return_value=exporter
        ):
            code = run_migrate(
                [
                    'wandb',
                    'export',
                    '--entity',
                    'acme',
                    '--project',
                    'vision',
                    '--output',
                    str(tmp_path),
                ]
            )
        assert code == 1

    def test_top_level_cli_help_does_not_need_migrate_extras(self):
        result = subprocess.run(
            [sys.executable, '-m', 'pluto', 'migrate', '--help'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert 'wandb' in result.stdout
