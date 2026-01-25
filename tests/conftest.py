"""
Global pytest fixtures for pluto tests.

This module provides fixtures that ensure proper cleanup of pluto resources
between tests, which is especially important for:
- Neptune compatibility tests with dual-logging
- Tests that spawn sync processes
- Tests running on Python 3.12+ with stricter daemon thread handling
"""

import logging
import os
import signal
import subprocess
import time

import pytest

# Logger for test cleanup debugging
_cleanup_logger = logging.getLogger('pluto.test.cleanup')
_cleanup_logger.setLevel(logging.DEBUG)

# Suppress pluto logging during tests to reduce noise
logging.getLogger('pluto').setLevel(logging.WARNING)
logging.getLogger('pluto-sync').setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def cleanup_pluto_resources():
    """
    Fixture that runs after each test to ensure pluto resources are cleaned up.

    This is critical for:
    1. Neptune compat tests that spawn sync processes
    2. Tests on Python 3.12+ where daemon thread handling is stricter
    3. Preventing resource leaks between tests

    The fixture:
    - Waits for any running pluto operations to finish
    - Terminates orphaned sync processes
    - Clears the global pluto.ops list
    """
    # Run the test first
    yield

    # Cleanup after test completes
    _cleanup_pluto_state()


def _cleanup_pluto_state():
    """
    Clean up all pluto state after a test.

    This handles:
    1. Finishing any active pluto runs
    2. Terminating sync processes
    3. Clearing global state
    """
    try:
        import pluto

        # Finish any active runs
        if hasattr(pluto, 'ops') and pluto.ops:
            for op in list(pluto.ops):  # Copy list to avoid mutation during iteration
                try:
                    # Use SIGTERM code to trigger graceful shutdown with wait=False
                    # This leverages Op.finish()'s built-in preemption handling
                    op.finish(code=signal.SIGTERM)
                except Exception as e:
                    _cleanup_logger.debug(f'Error during pluto op cleanup in test: {e}')

        # Clear the ops list
        if hasattr(pluto, 'ops'):
            pluto.ops = []

    except ImportError:
        pass  # pluto not installed, nothing to clean up

    # Also clean up any orphaned sync processes
    _terminate_orphan_sync_processes()


def _terminate_orphan_sync_processes():
    """
    Find and terminate any orphaned pluto sync processes.

    This is a safety net for tests that don't properly clean up.
    Only terminates processes that match the pluto.sync pattern.
    """
    try:
        # Find pluto sync processes
        result = subprocess.run(
            ['pgrep', '-f', 'python.*-m.*pluto.sync'],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid_str in pids:
                try:
                    pid = int(pid_str.strip())
                    # Send SIGTERM for graceful shutdown
                    os.kill(pid, signal.SIGTERM)
                except (ValueError, OSError):
                    pass  # Process already gone or invalid PID

            # Brief wait for processes to exit
            time.sleep(0.5)

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError, OSError):
        pass  # pgrep not available or subprocess error, skip cleanup


@pytest.fixture
def neptune_compat_env(tmp_path):
    """
    Fixture that sets up a clean environment for Neptune compat tests.

    This fixture:
    1. Sets required environment variables
    2. Creates a temp directory for test artifacts
    3. Cleans up after the test

    Usage:
        def test_neptune_compat(neptune_compat_env):
            tmp_dir = neptune_compat_env
            # Test code here
    """
    # Store original env vars
    original_env = {}
    env_vars = ['PLUTO_PROJECT', 'PLUTO_API_KEY', 'DISABLE_NEPTUNE_LOGGING']
    for var in env_vars:
        original_env[var] = os.environ.get(var)

    # Set up test environment
    os.environ['PLUTO_PROJECT'] = os.environ.get('PLUTO_PROJECT', 'test-project')

    yield tmp_path

    # Restore original env vars
    for var, value in original_env.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
    # Note: pluto cleanup is handled by the autouse cleanup_pluto_resources fixture
