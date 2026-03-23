"""Tests for `pluto sync` CLI command and related retry functionality.

Covers:
- SyncStore.reset_failed_to_pending()
- retry_sync() function
- _get_auth_token() helper
- _cmd_sync() CLI handler (discovery, blocking sync, background sync)
- pluto.sync.retry CLI entry point
"""

import argparse
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from pluto.sync.store import RecordType, SyncStatus, SyncStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """Create a temporary SyncStore for testing."""
    db_path = tmp_path / 'sync.db'
    s = SyncStore(str(db_path))
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """SyncStore with a run and mixed-status records."""
    store.register_run('run-1', 'my-project', op_id=42)

    ts = int(time.time() * 1000)

    # 3 PENDING records
    for i in range(3):
        store.enqueue('run-1', RecordType.METRIC, {'loss': 0.5 + i}, ts, step=i)

    # 2 FAILED records (enqueue then mark failed)
    failed_ids = []
    for i in range(2):
        rid = store.enqueue('run-1', RecordType.METRIC, {'loss': 0.9}, ts, step=10 + i)
        failed_ids.append(rid)
    store.mark_failed(failed_ids, 'network error')

    # 1 IN_PROGRESS record
    ip_id = store.enqueue('run-1', RecordType.CONFIG, {'lr': 0.01}, ts)
    store.mark_in_progress([ip_id])

    # 1 COMPLETED record (should NOT be reset)
    done_id = store.enqueue('run-1', RecordType.METRIC, {'acc': 0.99}, ts, step=99)
    store.mark_in_progress([done_id])
    store.mark_completed([done_id])

    # 1 FAILED file upload
    fid = store.enqueue_file(
        run_id='run-1',
        local_path='/tmp/img.png',
        file_name='img',
        file_ext='.png',
        file_type='image/png',
        file_size=1024,
        log_name='images/img',
        timestamp_ms=ts,
        step=0,
    )
    store.mark_files_failed([fid], 'upload timeout')

    return store


def _make_sync_db(tmp_path, run_id='run-1', project='test-proj', op_id=42, n_pending=3):
    """Helper: create a .pluto/.../sync.db with pending records."""
    pluto_dir = tmp_path / '.pluto' / run_id
    pluto_dir.mkdir(parents=True)
    db_path = pluto_dir / 'sync.db'
    s = SyncStore(str(db_path))
    s.register_run(run_id, project, op_id=op_id)
    ts = int(time.time() * 1000)
    for i in range(n_pending):
        s.enqueue(run_id, RecordType.METRIC, {'loss': 0.5}, ts, step=i)
    s.close()
    return str(db_path)


# ---------------------------------------------------------------------------
# Tests: SyncStore.reset_failed_to_pending
# ---------------------------------------------------------------------------


class TestResetFailedToPending:
    def test_resets_failed_and_in_progress(self, populated_store):
        """FAILED and IN_PROGRESS records are reset; PENDING/COMPLETED untouched."""
        store = populated_store
        reset_count = store.reset_failed_to_pending()

        # 2 FAILED records + 1 IN_PROGRESS record + 1 FAILED file = 4
        assert reset_count == 4

        # All non-completed records should now be PENDING
        row = store.conn.execute(
            'SELECT COUNT(*) FROM sync_queue WHERE status = ?',
            (int(SyncStatus.PENDING),),
        ).fetchone()
        assert row[0] == 6  # 3 original PENDING + 2 reset FAILED + 1 reset IN_PROGRESS

        # COMPLETED record untouched
        row = store.conn.execute(
            'SELECT COUNT(*) FROM sync_queue WHERE status = ?',
            (int(SyncStatus.COMPLETED),),
        ).fetchone()
        assert row[0] == 1

        # File upload reset to PENDING
        row = store.conn.execute(
            'SELECT status, retry_count, error_message FROM file_uploads WHERE id = 1'
        ).fetchone()
        assert row[0] == int(SyncStatus.PENDING)
        assert row[1] == 0
        assert row[2] is None

    def test_noop_when_nothing_to_reset(self, store):
        """Returns 0 when no FAILED/IN_PROGRESS records exist."""
        store.register_run('run-1', 'proj')
        ts = int(time.time() * 1000)
        store.enqueue('run-1', RecordType.METRIC, {'x': 1}, ts)
        assert store.reset_failed_to_pending() == 0

    def test_clears_retry_count_and_error(self, store):
        """Retry count and error message are zeroed on reset."""
        store.register_run('run-1', 'proj')
        ts = int(time.time() * 1000)
        rid = store.enqueue('run-1', RecordType.METRIC, {'x': 1}, ts)
        store.mark_failed([rid], 'err1')
        store.mark_failed([rid], 'err2')

        # retry_count should be 2
        row = store.conn.execute(
            'SELECT retry_count FROM sync_queue WHERE id = ?', (rid,)
        ).fetchone()
        assert row[0] == 2

        store.reset_failed_to_pending()

        row = store.conn.execute(
            'SELECT status, retry_count, error_message FROM sync_queue WHERE id = ?',
            (rid,),
        ).fetchone()
        assert row[0] == int(SyncStatus.PENDING)
        assert row[1] == 0
        assert row[2] is None


# ---------------------------------------------------------------------------
# Tests: retry_sync
# ---------------------------------------------------------------------------


class TestRetrySync:
    def test_returns_true_when_nothing_pending(self, tmp_path):
        """retry_sync returns True immediately if no pending records."""
        db_path = str(tmp_path / 'sync.db')
        s = SyncStore(db_path)
        s.register_run('run-1', 'proj', op_id=1)
        s.close()

        from pluto.sync.process import retry_sync

        result = retry_sync(
            db_path=db_path,
            settings_dict={'_auth': 'fake', 'project': 'proj'},
            timeout=5.0,
        )
        assert result is True

    @patch('pluto.sync.process._sync_batch')
    def test_returns_true_when_all_synced(self, mock_batch, tmp_path):
        """retry_sync returns True when _sync_batch drains everything."""
        db_path = str(tmp_path / 'sync.db')
        s = SyncStore(db_path)
        s.register_run('run-1', 'proj', op_id=1)
        ts = int(time.time() * 1000)
        for i in range(3):
            s.enqueue('run-1', RecordType.METRIC, {'x': i}, ts, step=i)
        s.close()

        call_count = 0

        def fake_sync_batch(
            store, uploader, log, max_retries, batch_size, file_batch_size
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Mark all records as completed
                records = store.get_pending_records(limit=100)
                if records:
                    store.mark_in_progress([r.id for r in records])
                    store.mark_completed([r.id for r in records])
                return 3
            return 0

        mock_batch.side_effect = fake_sync_batch

        from pluto.sync.process import retry_sync

        result = retry_sync(
            db_path=db_path,
            settings_dict={'_auth': 'fake', 'project': 'proj'},
            timeout=10.0,
        )
        assert result is True
        assert call_count == 2

    @patch('pluto.sync.process._sync_batch')
    def test_returns_false_when_records_remain(self, mock_batch, tmp_path):
        """retry_sync returns False when some records can't be synced."""
        db_path = str(tmp_path / 'sync.db')
        s = SyncStore(db_path)
        s.register_run('run-1', 'proj', op_id=1)
        ts = int(time.time() * 1000)
        s.enqueue('run-1', RecordType.METRIC, {'x': 1}, ts)
        s.close()

        mock_batch.return_value = 0

        from pluto.sync.process import retry_sync

        result = retry_sync(
            db_path=db_path,
            settings_dict={'_auth': 'fake', 'project': 'proj'},
            timeout=10.0,
        )
        assert result is False

    @patch('pluto.sync.process._sync_batch')
    @patch('pluto.sync.process.time')
    def test_timeout_returns_false(self, mock_time, mock_batch, tmp_path):
        """retry_sync returns False on timeout with pending records."""
        db_path = str(tmp_path / 'sync.db')
        s = SyncStore(db_path)
        s.register_run('run-1', 'proj', op_id=1)
        ts = int(time.time() * 1000)
        s.enqueue('run-1', RecordType.METRIC, {'x': 1}, ts)
        s.close()

        # Simulate time passing beyond timeout
        mock_time.time.side_effect = [0.0, 0.0, 100.0]
        mock_time.perf_counter = time.perf_counter
        mock_batch.return_value = 1

        from pluto.sync.process import retry_sync

        result = retry_sync(
            db_path=db_path,
            settings_dict={'_auth': 'fake', 'project': 'proj'},
            timeout=5.0,
        )
        assert result is False

    @patch('pluto.sync.process._sync_batch')
    def test_verbose_prints_progress(self, mock_batch, tmp_path, capsys):
        """Verbose mode prints reset count and progress."""
        db_path = str(tmp_path / 'sync.db')
        s = SyncStore(db_path)
        s.register_run('run-1', 'proj', op_id=1)
        ts = int(time.time() * 1000)
        rid = s.enqueue('run-1', RecordType.METRIC, {'x': 1}, ts)
        s.mark_failed([rid], 'err')
        s.close()

        call_count = 0

        def fake_sync_batch(
            store, uploader, log, max_retries, batch_size, file_batch_size
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                records = store.get_pending_records(limit=100)
                if records:
                    store.mark_in_progress([r.id for r in records])
                    store.mark_completed([r.id for r in records])
                return 1
            return 0

        mock_batch.side_effect = fake_sync_batch

        from pluto.sync.process import retry_sync

        result = retry_sync(
            db_path=db_path,
            settings_dict={'_auth': 'fake', 'project': 'proj'},
            timeout=10.0,
            verbose=True,
        )
        assert result is True

        captured = capsys.readouterr()
        assert 'Reset 1 failed/stale records' in captured.out
        assert 'synced' in captured.out.lower()


# ---------------------------------------------------------------------------
# Tests: _get_auth_token
# ---------------------------------------------------------------------------


class TestGetAuthToken:
    def test_env_var_pluto_api_key(self, monkeypatch):
        monkeypatch.setenv('PLUTO_API_KEY', 'tok123')
        monkeypatch.delenv('MLOP_API_TOKEN', raising=False)
        from pluto.__main__ import _get_auth_token

        assert _get_auth_token() == 'tok123'

    def test_env_var_mlop_fallback(self, monkeypatch):
        monkeypatch.delenv('PLUTO_API_KEY', raising=False)
        monkeypatch.setenv('MLOP_API_TOKEN', 'legacy-tok')
        from pluto.__main__ import _get_auth_token

        assert _get_auth_token() == 'legacy-tok'

    def test_pluto_takes_precedence(self, monkeypatch):
        monkeypatch.setenv('PLUTO_API_KEY', 'new')
        monkeypatch.setenv('MLOP_API_TOKEN', 'old')
        from pluto.__main__ import _get_auth_token

        assert _get_auth_token() == 'new'

    def test_returns_none_when_no_env_and_no_keyring(self, monkeypatch):
        monkeypatch.delenv('PLUTO_API_KEY', raising=False)
        monkeypatch.delenv('MLOP_API_TOKEN', raising=False)
        # Patch keyring import to raise ImportError
        import builtins

        from pluto.__main__ import _get_auth_token

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'keyring':
                raise ImportError('no keyring')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_import)
        assert _get_auth_token() is None


# ---------------------------------------------------------------------------
# Tests: _cmd_sync (CLI handler)
# ---------------------------------------------------------------------------


class TestCmdSync:
    def _make_args(self, **kwargs):
        defaults = {
            'path': None,
            'dir': None,
            'background': False,
            'timeout': 60.0,
            'verbose': False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_specific_path_not_found(self, tmp_path):
        """--path to nonexistent file exits with code 1."""
        from pluto.__main__ import _cmd_sync

        args = self._make_args(path=str(tmp_path / 'sync.db'))
        with pytest.raises(SystemExit, match='1'):
            _cmd_sync(args)

    def test_path_must_be_sync_db(self, tmp_path):
        """--path to non-sync.db file exits with code 1."""
        from pluto.__main__ import _cmd_sync

        args = self._make_args(path=str(tmp_path / 'other.db'))
        with pytest.raises(SystemExit, match='1'):
            _cmd_sync(args)

    def test_no_databases_found(self, tmp_path, capsys):
        """Prints message when no sync.db files found under --dir."""
        from pluto.__main__ import _cmd_sync

        args = self._make_args(dir=str(tmp_path))
        _cmd_sync(args)

        captured = capsys.readouterr()
        assert 'No sync databases found' in captured.out

    def test_no_auth_exits(self, tmp_path):
        """Exits with code 1 when no auth token is available."""
        from pluto.__main__ import _cmd_sync

        db_path = _make_sync_db(tmp_path)
        args = self._make_args(path=db_path)

        with patch('pluto.__main__._get_auth_token', return_value=None):
            with pytest.raises(SystemExit, match='1'):
                _cmd_sync(args)

    def test_no_pending_records(self, tmp_path, capsys):
        """Prints message when databases exist but have no pending records."""
        from pluto.__main__ import _cmd_sync

        # Create a db with all-completed records
        pluto_dir = tmp_path / '.pluto' / 'run-done'
        pluto_dir.mkdir(parents=True)
        db_path = str(pluto_dir / 'sync.db')
        s = SyncStore(db_path)
        s.register_run('run-done', 'proj', op_id=1)
        ts = int(time.time() * 1000)
        rid = s.enqueue('run-done', RecordType.METRIC, {'x': 1}, ts)
        s.mark_in_progress([rid])
        s.mark_completed([rid])
        s.close()

        args = self._make_args(dir=str(tmp_path))
        with patch('pluto.__main__._get_auth_token', return_value='tok'):
            _cmd_sync(args)

        captured = capsys.readouterr()
        assert 'No pending records' in captured.out

    def test_discovers_databases_recursively(self, tmp_path, capsys):
        """Finds sync.db files under .pluto/ recursively."""
        from pluto.__main__ import _cmd_sync

        _make_sync_db(tmp_path, run_id='run-a', project='proj-a', n_pending=2)
        _make_sync_db(tmp_path, run_id='run-b', project='proj-b', n_pending=3)

        args = self._make_args(dir=str(tmp_path))
        with (
            patch('pluto.__main__._get_auth_token', return_value='tok'),
            patch('pluto.sync.process.retry_sync', return_value=True) as mock_retry,
        ):
            _cmd_sync(args)

        assert mock_retry.call_count == 2
        captured = capsys.readouterr()
        assert 'proj-a/run-a' in captured.out
        assert 'proj-b/run-b' in captured.out

    def test_blocking_sync_calls_retry_sync(self, tmp_path, capsys):
        """Blocking mode calls retry_sync directly."""
        from pluto.__main__ import _cmd_sync

        db_path = _make_sync_db(tmp_path, n_pending=5)
        args = self._make_args(path=db_path, timeout=30.0)

        with (
            patch('pluto.__main__._get_auth_token', return_value='tok'),
            patch('pluto.sync.process.retry_sync', return_value=True) as mock_retry,
        ):
            _cmd_sync(args)

        mock_retry.assert_called_once()
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs['db_path'] == db_path
        assert call_kwargs.kwargs['timeout'] == 30.0

    def test_blocking_sync_reports_failure(self, tmp_path, capsys):
        """Prints failure message when retry_sync returns False."""
        from pluto.__main__ import _cmd_sync

        db_path = _make_sync_db(tmp_path)
        args = self._make_args(path=db_path)

        with (
            patch('pluto.__main__._get_auth_token', return_value='tok'),
            patch('pluto.sync.process.retry_sync', return_value=False),
        ):
            _cmd_sync(args)

        captured = capsys.readouterr()
        assert 'failed to sync' in captured.out.lower()

    def _mock_settings(self):
        """Create a mock Settings whose to_dict() returns JSON-serializable data."""
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {
            'tag': 'pluto',
            'project': 'pluto',
            'host': 'https://pluto.trainy.ai',
        }
        return mock_settings

    def test_background_mode_spawns_subprocess(self, tmp_path, capsys):
        """--background spawns subprocess.Popen for each database."""
        from pluto.__main__ import _cmd_sync

        _make_sync_db(tmp_path, run_id='run-bg', n_pending=2)
        args = self._make_args(dir=str(tmp_path), background=True, timeout=45.0)

        with (
            patch('pluto.__main__._get_auth_token', return_value='tok'),
            patch('pluto.sets.Settings', return_value=self._mock_settings()),
            patch('subprocess.Popen') as mock_popen,
        ):
            _cmd_sync(args)

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert '-m' in cmd
        assert 'pluto.sync.retry' in cmd
        assert '--timeout' in cmd
        assert '45.0' in cmd

        captured = capsys.readouterr()
        assert 'background sync process' in captured.out.lower()

    def test_background_verbose_passes_flag(self, tmp_path):
        """--background --verbose passes --verbose to subprocess."""
        from pluto.__main__ import _cmd_sync

        _make_sync_db(tmp_path, run_id='run-v', n_pending=1)
        args = self._make_args(dir=str(tmp_path), background=True, verbose=True)

        with (
            patch('pluto.__main__._get_auth_token', return_value='tok'),
            patch('pluto.sets.Settings', return_value=self._mock_settings()),
            patch('subprocess.Popen') as mock_popen,
        ):
            _cmd_sync(args)

        cmd = mock_popen.call_args[0][0]
        assert '--verbose' in cmd

    def test_settings_populated_with_run_info(self, tmp_path):
        """Settings dict passed to retry_sync includes project/op_id/op_name from DB."""
        from pluto.__main__ import _cmd_sync

        db_path = _make_sync_db(tmp_path, run_id='myrun', project='myproj', op_id=99)
        args = self._make_args(path=db_path)

        with (
            patch('pluto.__main__._get_auth_token', return_value='tok'),
            patch('pluto.sync.process.retry_sync', return_value=True) as mock_retry,
        ):
            _cmd_sync(args)

        settings = mock_retry.call_args.kwargs['settings_dict']
        assert settings['project'] == 'myproj'
        assert settings['_op_id'] == 99
        assert settings['_op_name'] == 'myrun'
        assert settings['_auth'] == 'tok'

    def test_sqlite_error_handled_gracefully(self, tmp_path, capsys):
        """Corrupt database is skipped with a warning."""
        from pluto.__main__ import _cmd_sync

        pluto_dir = tmp_path / '.pluto' / 'bad-run'
        pluto_dir.mkdir(parents=True)
        corrupt_db = pluto_dir / 'sync.db'
        corrupt_db.write_text('this is not a sqlite database')

        args = self._make_args(dir=str(tmp_path))
        with patch('pluto.__main__._get_auth_token', return_value='tok'):
            _cmd_sync(args)

        captured = capsys.readouterr()
        assert 'Warning' in captured.err or 'No pending records' in captured.out


# ---------------------------------------------------------------------------
# Tests: pluto.sync.retry CLI entry point
# ---------------------------------------------------------------------------


class TestSyncRetryCli:
    def test_valid_args(self, tmp_path):
        """retry.main() parses args and calls retry_sync."""
        db_path = str(tmp_path / 'sync.db')
        settings = json.dumps({'_auth': 'tok', 'project': 'p'})

        with (
            patch(
                'sys.argv',
                [
                    'retry',
                    '--db-path',
                    db_path,
                    '--settings',
                    settings,
                    '--timeout',
                    '30',
                ],
            ),
            patch('pluto.sync.process.retry_sync', return_value=True) as mock_retry,
        ):
            from pluto.sync.retry import main

            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0

        mock_retry.assert_called_once_with(
            db_path=db_path,
            settings_dict={'_auth': 'tok', 'project': 'p'},
            timeout=30.0,
            verbose=False,
        )

    def test_failure_exits_1(self, tmp_path):
        """Exit code 1 when retry_sync returns False."""
        db_path = str(tmp_path / 'sync.db')
        settings = json.dumps({'_auth': 'tok'})

        with (
            patch('sys.argv', ['retry', '--db-path', db_path, '--settings', settings]),
            patch('pluto.sync.process.retry_sync', return_value=False),
        ):
            from pluto.sync.retry import main

            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_invalid_json_exits_1(self, tmp_path):
        """Exit code 1 when --settings is not valid JSON."""
        db_path = str(tmp_path / 'sync.db')

        with patch(
            'sys.argv',
            ['retry', '--db-path', db_path, '--settings', '{bad json'],
        ):
            from pluto.sync.retry import main

            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
