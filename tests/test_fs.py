"""Tests for network-filesystem detection (pluto/_fs.py)."""

from unittest.mock import mock_open, patch

from pluto._fs import get_fs_type, is_network_fs

# A representative /proc/self/mountinfo with a local root and an NFS mount.
_MOUNTINFO = (
    '21 1 0:20 / / rw,relatime shared:1 - ext4 /dev/root rw\n'
    '30 21 0:26 / /home rw,relatime shared:2 - ext4 /dev/sda1 rw\n'
    '47 21 0:42 / /mnt/nfs rw,relatime shared:3 - nfs4 server:/export rw\n'
    '55 47 0:42 / /mnt/nfs/sub rw,relatime shared:4 - nfs4 server:/export/sub rw\n'
)


def _patched(path, mountinfo=_MOUNTINFO):
    """Run get_fs_type(path) against a mocked mountinfo, with realpath identity."""
    with patch('pluto._fs.open', mock_open(read_data=mountinfo)):
        with patch('os.path.realpath', side_effect=lambda p: p):
            return get_fs_type(path)


class TestGetFsType:
    def test_local_path(self):
        assert _patched('/home/user/run') == 'ext4'

    def test_root_path(self):
        assert _patched('/var/tmp/x') == 'ext4'

    def test_nfs_path(self):
        assert _patched('/mnt/nfs/project/run') == 'nfs4'

    def test_longest_mount_wins(self):
        # /mnt/nfs/sub is more specific than /mnt/nfs; both are nfs4 here, but
        # the match must resolve to the nested mount, not the parent.
        assert _patched('/mnt/nfs/sub/run') == 'nfs4'

    def test_nonexistent_mountinfo_returns_none(self):
        with patch('pluto._fs.open', side_effect=OSError):
            assert get_fs_type('/whatever') is None

    def test_octal_escaped_mount_point(self):
        # mountinfo escapes spaces as \040; a path with a space must still match.
        mountinfo = (
            '21 1 0:20 / / rw - ext4 /dev/root rw\n'
            '47 21 0:42 / /mnt/my\\040share rw - nfs4 server:/export rw\n'
        )
        assert _patched('/mnt/my share/run', mountinfo) == 'nfs4'


class TestIsNetworkFs:
    def test_local_is_not_network(self):
        with patch('pluto._fs.get_fs_type', return_value='ext4'):
            assert is_network_fs('/home/user') is False

    def test_nfs_is_network(self):
        with patch('pluto._fs.get_fs_type', return_value='nfs4'):
            assert is_network_fs('/mnt/nfs') is True

    def test_lustre_is_network(self):
        with patch('pluto._fs.get_fs_type', return_value='lustre'):
            assert is_network_fs('/scratch') is True

    def test_fuse_sshfs_is_network(self):
        with patch('pluto._fs.get_fs_type', return_value='fuse.sshfs'):
            assert is_network_fs('/remote') is True

    def test_unknown_fs_is_not_network(self):
        # Undeterminable (e.g. non-Linux) must not warn.
        with patch('pluto._fs.get_fs_type', return_value=None):
            assert is_network_fs('/anything') is False
