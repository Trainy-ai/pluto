import inspect
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path

test_id = str(uuid.uuid4())[-2:]


def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def get_task_name() -> str:
    """Returns a user-unique task name for each test

    Must be called from each test_<name> function.
    """
    caller_func_name = inspect.stack()[1][3]
    test_name = caller_func_name.replace('_', '-').replace('test-', 't-')
    return f'{test_name}-{test_id}'


def capture_sync_db_path(run):
    """Grab the run's sync DB path while the sync manager is still attached.

    Must be called BEFORE run.finish() — finish() detaches the manager.
    Returns None when the run isn't using the sync process.
    """
    manager = getattr(run, '_sync_manager', None)
    return manager.db_path if manager is not None else None


def assert_sync_files_completed(db_path):
    """Assert every file enqueued for upload actually reached the server.

    finish() waits for the sync process to drain, so by the time it returns
    every file_uploads row must be COMPLETED (status 2). Anything else —
    PENDING, IN_PROGRESS, FAILED, ABANDONED — means a payload was silently
    lost (the silent-loss signature: the log name is registered server-side
    but the bytes never arrived). This check is local and cheap, and turns
    upload smoke tests into loss-detecting tests.
    """
    if not db_path or not os.path.exists(db_path):
        return  # sync process not in use for this run

    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        rows = conn.execute(
            'SELECT log_name, status, retry_count, error_message '
            'FROM file_uploads WHERE status != 2'
        ).fetchall()
    finally:
        conn.close()

    status_names = {0: 'PENDING', 1: 'IN_PROGRESS', 3: 'FAILED', 4: 'ABANDONED'}
    assert not rows, (
        'file payload(s) were never uploaded (silent-loss signature): '
        + '; '.join(
            f'"{log_name}" status={status_names.get(status, status)} '
            f'retries={retries} error={error!r}'
            for log_name, status, retries, error in rows
        )
        + f' — see {db_path}'
    )


def download_file_with_poll(
    project,
    run_id,
    log_name,
    destination,
    timeout=60,
    interval=2,
) -> 'Path':
    """Download a just-uploaded file, tolerating eventual consistency.

    Retries QueryError ("no file found") until *timeout*, then FAILS the
    test — after a successful finish(), a file that never becomes
    downloadable is a lost payload, not eventual consistency. (This used
    to be a pytest.skip, which masked exactly the silent-loss bug it was
    meant to catch.)
    """
    import pytest

    import pluto.query as pq

    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            return pq.download_file(project, run_id, log_name, destination)
        except pq.QueryError as exc:
            last_error = exc
            time.sleep(interval)
    pytest.fail(
        f'file "{log_name}" for run {run_id} never became downloadable '
        f'within {timeout}s of finish() — the payload was lost, not '
        f'eventually consistent. Last error: {last_error}'
    )
