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
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

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
    """Load-phase resume cache: which external ids finished loading."""

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._entries: Dict[str, Any] = {}
        if self._path.exists():
            self._entries = read_json(self._path)

    def is_loaded(self, external_id: str) -> bool:
        return external_id in self._entries

    def mark_loaded(self, external_id: str, info: Dict[str, Any]) -> None:
        self._entries[external_id] = info
        write_json_atomic(self._path, self._entries)
