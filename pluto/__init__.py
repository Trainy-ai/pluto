import os
import subprocess
from typing import Any, Callable, List, Optional

from .auth import login, logout
from .data import Data, Graph, Histogram, Table
from .file import Artifact, Audio, File, Image, Text, Video
from .init import finish, init
from .sets import Settings, setup
from .sys import System

_hooks: List[Any] = []
ops: Optional[List[Any]] = None
log: Optional[Callable[..., Any]] = None
watch: Optional[Callable[..., Any]] = None
alert: Optional[Callable[..., Any]] = None

__all__ = (
    'Data',
    'Graph',
    'Histogram',
    'Table',
    'File',
    'Artifact',
    'Text',
    'Image',
    'Audio',
    'Video',
    'System',
    'Settings',
    'alert',
    'init',
    'login',
    'logout',
    'watch',
    'finish',
    'setup',
)

__version__ = '0.0.2'


# Replaced with the current commit when building the wheels.
_PLUTO_COMMIT_SHA = '{{PLUTO_COMMIT_SHA}}'


def _get_git_commit():
    if 'PLUTO_COMMIT_SHA' not in _PLUTO_COMMIT_SHA:
        # This is a release build, so we don't need to get the commit hash from
        # git, as it's already been set.
        return _PLUTO_COMMIT_SHA

    # This is a development build (pip install -e .), so we need to get the
    # commit hash from git.
    try:
        cwd = os.path.dirname(__file__)
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=cwd,
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        changes = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=cwd,
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if changes:
            commit_hash += '-dirty'
        return commit_hash
    except Exception:  # pylint: disable=broad-except
        return _MLOP_COMMIT_SHA
