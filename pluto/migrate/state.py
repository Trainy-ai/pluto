"""
Resume bookkeeping for pluto.migrate.

Export marks each fully-staged run with a sentinel file written last, so
an interrupted export re-does at most one run. Load records finished
runs in a single ``loaded_runs.json`` next to the export, written only
after ``finish()`` drains — so a re-run skips completed runs and retries
partial ones. All writes are atomic (tmp file + fsync + rename).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'migrate'

EXPORT_SENTINEL = '_export_complete.json'
LOADED_CACHE_FILENAME = 'loaded_runs.json'


def write_json_atomic(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Union[str, Path]) -> Any:
    with open(path) as f:
        return json.load(f)


def is_run_exported(run_dir: Union[str, Path]) -> bool:
    return (Path(run_dir) / EXPORT_SENTINEL).exists()


def mark_run_exported(
    run_dir: Union[str, Path], summary: Optional[Dict[str, Any]] = None
) -> None:
    write_json_atomic(Path(run_dir) / EXPORT_SENTINEL, summary or {})


class LoadedCache:
    """Load-phase resume cache tracking each external id's load state.

    Entries are ``{status: 'in_progress' | 'done', ...}``. ``in_progress`` is
    written right after a run is created server-side but before its replay
    finishes; ``done`` after ``finish()`` drains. That lets a re-run tell a
    run *we* started but didn't finish (resume + complete it) from one that
    merely exists server-side (loaded elsewhere → skip, no media dup).
    Legacy entries without a status are treated as ``done``.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._entries: Dict[str, Any] = {}
        if self._path.exists():
            try:
                data = read_json(self._path)
                if isinstance(data, dict):
                    self._entries = data
                else:
                    raise ValueError('cache is not a JSON object')
            except (json.JSONDecodeError, ValueError, OSError) as e:
                # A corrupt/empty cache must not abort the whole load. Move it
                # aside and start fresh; already-loaded runs are re-detected via
                # the server-side external-id collision (RunExistsError).
                logger.warning(
                    f'{tag}: {self._path.name} unreadable ({e}); starting with '
                    'an empty load cache (backed up as .corrupt)'
                )
                try:
                    self._path.replace(self._path.with_suffix('.corrupt'))
                except OSError:
                    pass

    def _status(self, external_id: str) -> Optional[str]:
        entry = self._entries.get(external_id)
        if entry is None:
            return None
        if isinstance(entry, dict):
            return entry.get('status', 'done')  # legacy entries -> done
        return 'done'

    def is_loaded(self, external_id: str) -> bool:
        return self._status(external_id) == 'done'

    def is_in_progress(self, external_id: str) -> bool:
        return self._status(external_id) == 'in_progress'

    def mark_in_progress(self, external_id: str) -> None:
        if self.is_loaded(external_id):
            return  # never downgrade a completed run
        self._entries[external_id] = {'status': 'in_progress'}
        write_json_atomic(self._path, self._entries)

    def mark_loaded(self, external_id: str, info: Dict[str, Any]) -> None:
        self._entries[external_id] = {**info, 'status': 'done'}
        write_json_atomic(self._path, self._entries)
