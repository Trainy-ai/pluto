"""Filesystem detection helpers.

Pluto stages local state — most importantly the WAL-mode SQLite database used
to hand data from the training process to the sync process — under the run
directory. SQLite's WAL locking relies on POSIX byte-range locks plus a shared
-shm mmap, neither of which behaves reliably on network filesystems (NFS,
Lustre, SMB/CIFS, ...). On those mounts the lock handoff degrades into
SQLITE_PROTOCOL ("locking protocol") races that show up as repeated lock
retries and can badly throttle logging.

These helpers let `init()` detect that situation up front and tell the user to
point the staging dir (``pluto.init(dir=...)`` / ``PLUTO_DIR``) at node-local
storage. Detection is best-effort and Linux-only — on any other platform, or if
the mount table can't be read, we return ``None``/``False`` and stay silent
rather than risk a false alarm.
"""

import os
import re
from typing import Optional

# /proc/self/mountinfo escapes space, tab, newline and backslash in paths and
# fstypes as octal sequences (\040, \011, \012, \134). Decode them so prefix
# matching works for mount points that contain such characters.
_OCTAL_ESCAPE = re.compile(r'\\([0-7]{3})')


def _unescape_mountinfo(field: str) -> str:
    return _OCTAL_ESCAPE.sub(lambda m: chr(int(m.group(1), 8)), field)


# Filesystem type prefixes (as reported in /proc/self/mountinfo) whose locking
# semantics are unreliable for WAL-mode SQLite. Matched as prefixes so that
# e.g. both "nfs" and "nfs4", or "fuse.sshfs", are covered.
_NETWORK_FS_PREFIXES = (
    'nfs',
    'cifs',
    'smb',
    'lustre',
    'gpfs',
    'ceph',
    'glusterfs',
    'afs',
    'ncpfs',
    'fuse.sshfs',
    'fuse.glusterfs',
    'beegfs',
)


def get_fs_type(path: str) -> Optional[str]:
    """Return the filesystem type backing ``path``, or ``None`` if unknown.

    Linux-only: reads ``/proc/self/mountinfo`` and returns the type of the
    most specific (longest) mount point that is a prefix of ``path``. The path
    need not exist yet — matching is done on the resolved path string, so a
    not-yet-created run directory still resolves to its parent mount. Returns
    ``None`` on non-Linux platforms or if the mount table can't be parsed.
    """
    try:
        target = os.path.realpath(path)
        best_mount = ''
        best_type: Optional[str] = None
        with open('/proc/self/mountinfo', 'r') as f:
            for line in f:
                # Format: "<id> <pid> <maj:min> <root> <mountpoint> <opts> \
                #          [optional fields] - <fstype> <source> <superopts>"
                left, sep, right = line.partition(' - ')
                if not sep:
                    continue
                left_fields = left.split()
                right_fields = right.split()
                if len(left_fields) < 5 or not right_fields:
                    continue
                mount_point = _unescape_mountinfo(left_fields[4])
                fstype = _unescape_mountinfo(right_fields[0])
                # Longest mount point that is a path-prefix of target wins.
                if target == mount_point or target.startswith(
                    mount_point.rstrip('/') + '/'
                ):
                    if len(mount_point) >= len(best_mount):
                        best_mount = mount_point
                        best_type = fstype
        return best_type
    except (OSError, ValueError, IndexError):
        return None


def is_network_fs(path: str) -> bool:
    """True if ``path`` appears to live on a network filesystem.

    Best-effort and conservative: returns ``False`` when the filesystem type
    can't be determined (e.g. non-Linux), so callers never warn spuriously.
    """
    fstype = get_fs_type(path)
    return fstype is not None and fstype.lower().startswith(_NETWORK_FS_PREFIXES)
