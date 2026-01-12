import importlib
import logging
import os
import queue
import sys
import warnings
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = 'Settings'


def _get_env_with_deprecation(new_key: str, old_key: str) -> Optional[str]:
    """Get env var with fallback to deprecated MLOP_* name."""
    value = os.environ.get(new_key)
    if value is None:
        old_value = os.environ.get(old_key)
        if old_value is not None:
            warnings.warn(
                f'Environment variable {old_key} is deprecated. '
                f'Use {new_key} instead.',
                DeprecationWarning,
                stacklevel=3,
            )
            return old_value
    return value


class Settings:
    tag: str = f"{__name__.split('.')[0]}"
    dir: str = str(os.path.abspath(os.getcwd()))

    _auth: Optional[str] = None
    _sys: Any = {}
    compat: Dict[str, Any] = {}
    project: str = tag
    mode: str = 'perf'  # noop | debug | perf
    meta: List[str] = []
    message: queue.Queue[Any] = queue.Queue()
    disable_store: bool = True  # TODO: make false
    disable_iface: bool = False
    disable_progress: bool = True
    disable_console: bool = False  # disable file-based logging

    _op_name: Optional[str] = None
    _op_id: Optional[int] = None
    _op_status: int = -1

    store_db: str = 'store.db'
    store_table_num: str = 'num'
    store_table_file: str = 'file'
    store_max_size: int = 2**14
    store_aggregate_interval: float = 2 ** (-1)

    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    insecure_disable_ssl: bool = False

    x_log_level: int = 2**4  # logging.NOTSET
    x_internal_check_process: int = 1  # TODO: make configurable
    x_file_stream_retry_max: int = 2**2
    x_file_stream_retry_wait_min_seconds: float = 2 ** (-1)
    x_file_stream_retry_wait_max_seconds: float = 2
    x_file_stream_timeout_seconds: int = 2**5  # 2**2
    x_file_stream_max_conn: int = 2**5
    x_file_stream_max_size: int = 2**18
    x_file_stream_transmit_interval: int = 2**3
    x_sys_sampling_interval: int = 2**2
    x_sys_label: str = 'sys'
    x_grad_label: str = 'grad'
    x_param_label: str = 'param'

    host: Optional[str] = None
    url_view: Optional[str] = None
    url_webhook: Optional[str] = None

    def update(self, settings: Union['Settings', Dict[str, Any]]) -> None:
        if isinstance(settings, Settings):
            settings = settings.to_dict()
        for key, value in settings.items():
            setattr(self, key, value)
        self.update_host()

    def update_host(self) -> None:
        if self.host is not None:
            self.url_app = f'http://{self.host}:3000'
            self.url_api = f'http://{self.host}:3001'
            self.url_ingest = f'http://{self.host}:3003'
            self.url_py = f'http://{self.host}:3004'
        elif not (  # backwards compatibility
            hasattr(self, 'url_app')
            and hasattr(self, 'url_api')
            and hasattr(self, 'url_ingest')
            and hasattr(self, 'url_py')
        ):
            self.url_app = 'https://pluto.trainy.ai'
            self.url_api = 'https://pluto-api.trainy.ai'
            self.url_ingest = 'https://pluto-ingest.trainy.ai'
            self.url_py = 'https://pluto-py.trainy.ai'
        self.update_url()

    def update_url(self) -> None:
        self.url_token = f'{self.url_app}/api-keys'
        self.url_login = f'{self.url_api}/api/slug'
        self.url_start = f'{self.url_api}/api/runs/create'
        self.url_stop = f'{self.url_api}/api/runs/status/update'
        self.url_meta = f'{self.url_api}/api/runs/logName/add'
        self.url_graph = f'{self.url_api}/api/runs/modelGraph/create'
        self.url_update_tags = f'{self.url_api}/api/runs/tags/update'
        self.url_update_config = f'{self.url_api}/api/runs/config/update'
        self.url_num = f'{self.url_ingest}/ingest/metrics'
        self.url_data = f'{self.url_ingest}/ingest/data'
        self.url_file = f'{self.url_ingest}/files'
        self.url_message = f'{self.url_ingest}/ingest/logs'
        self.url_alert = f'{self.url_py}/api/runs/alert'
        self.url_trigger = f'{self.url_py}/api/runs/trigger'

    def to_dict(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self.__annotations__.keys()}

    def get_dir(self) -> str:
        op_segment = self._op_name or str(self._op_id or 'run')
        return os.path.join(
            self.dir,
            '.' + self.tag,
            self.project,
            op_segment,
        )

    def _nb(self) -> bool:
        return (
            get_console() in ['ipython', 'jupyter']
            or self._nb_colab()
            or self._nb_kaggle()
        )

    def _nb_colab(self) -> bool:
        return 'google.colab' in sys.modules

    def _nb_kaggle(self) -> bool:
        return (
            os.getenv('KAGGLE_KERNEL_RUN_TYPE') is not None
            or 'kaggle_environments' in sys.modules
            or 'kaggle' in sys.modules
        )


def get_console() -> str:
    try:
        ipython_module = importlib.import_module('IPython')
    except ImportError:
        return 'python'

    get_ipython = getattr(ipython_module, 'get_ipython', None)
    ipython = get_ipython() if callable(get_ipython) else None
    if ipython is None:
        return 'python'

    if 'spyder' in sys.modules or 'terminal' in ipython.__module__:
        return 'ipython'

    connection_file = (
        ipython.config.get('IPKernelApp', {}).get('connection_file', '')
        or ipython.config.get('ColabKernelApp', {}).get('connection_file', '')
    ).lower()
    if 'jupyter' not in connection_file:
        return 'ipython'
    else:
        return 'jupyter'


def setup(settings: Union[Settings, Dict[str, Any], None] = None) -> Settings:
    if isinstance(settings, Settings):
        settings.update(settings)
        return settings

    new_settings = Settings()

    # Read PLUTO_DEBUG_LEVEL environment variable (with MLOP_DEBUG_LEVEL fallback)
    env_log_level = _get_env_with_deprecation('PLUTO_DEBUG_LEVEL', 'MLOP_DEBUG_LEVEL')
    if env_log_level is not None:
        level_map = {
            'DEBUG': 10,
            'INFO': 20,
            'WARNING': 30,
            'ERROR': 40,
            'CRITICAL': 50,
        }
        if env_log_level.upper() in level_map:
            new_settings.x_log_level = level_map[env_log_level.upper()]
        elif env_log_level.isdigit():
            new_settings.x_log_level = int(env_log_level)
        else:
            logger.warning(
                f'{tag}: invalid PLUTO_DEBUG_LEVEL "{env_log_level}", using default. '
                f'Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL'
            )

    # Prepare settings dict and check for URL overrides
    settings_dict = settings if isinstance(settings, dict) else {}

    # Read URL environment variables (with MLOP_* fallback)
    env_url_app = _get_env_with_deprecation('PLUTO_URL_APP', 'MLOP_URL_APP')
    env_url_api = _get_env_with_deprecation('PLUTO_URL_API', 'MLOP_URL_API')
    env_url_ingest = _get_env_with_deprecation('PLUTO_URL_INGEST', 'MLOP_URL_INGEST')
    env_url_py = _get_env_with_deprecation('PLUTO_URL_PY', 'MLOP_URL_PY')

    # If any URL env var is set, ensure all four URLs are in settings_dict
    # This prevents update_host() from resetting to defaults
    if any([env_url_app, env_url_api, env_url_ingest, env_url_py]):
        # Set defaults for any URLs not provided
        default_urls = {
            'url_app': 'https://pluto.trainy.ai',
            'url_api': 'https://pluto-api.trainy.ai',
            'url_ingest': 'https://pluto-ingest.trainy.ai',
            'url_py': 'https://pluto-py.trainy.ai',
        }

        # Merge: user params > env vars > defaults
        env_url_map = {
            'url_app': env_url_app,
            'url_api': env_url_api,
            'url_ingest': env_url_ingest,
            'url_py': env_url_py,
        }
        for url_key, default_value in default_urls.items():
            if url_key not in settings_dict:
                env_value = env_url_map.get(url_key)
                settings_dict[url_key] = (
                    env_value if env_value is not None else default_value
                )

    # Read PLUTO_API_TOKEN environment variable (with MLOP_API_TOKEN fallback)
    # Only apply if not already set via function parameters
    env_api_token = _get_env_with_deprecation('PLUTO_API_TOKEN', 'MLOP_API_TOKEN')
    if env_api_token is not None and '_auth' not in settings_dict:
        new_settings._auth = env_api_token

    # Read PLUTO_PROJECT environment variable (with MLOP_PROJECT fallback)
    # Only apply if not already set via function parameters
    env_project = _get_env_with_deprecation('PLUTO_PROJECT', 'MLOP_PROJECT')
    if env_project is not None and 'project' not in settings_dict:
        new_settings.project = env_project

    # Apply all settings (user params override env vars)
    new_settings.update(settings_dict)

    return new_settings
