import inspect
import os
import re
import uuid

import mlop

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


def init_run(project: str, name: str, config):
    """Initialize a noop run for tests so no network calls are made."""
    settings = mlop.Settings()
    settings.mode = 'noop'
    settings.disable_store = True
    settings.disable_iface = True
    run = mlop.init(project=project, name=name, config=config, settings=settings)
    os.makedirs(f'{run.settings.get_dir()}/files', exist_ok=True)
    return run
