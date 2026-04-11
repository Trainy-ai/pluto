"""
Shared helpers for the pluto compat layers (wandb, neptune, etc.).

These used to be duplicated verbatim across compat/wandb.py and
compat/neptune.py. Any future compat shim should import from here
instead of copy-pasting.
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_env_with_deprecation(new_key: str, old_key: str) -> Optional[str]:
    """Get env var, falling back to a deprecated MLOP_* name with a warning."""
    import warnings

    value = os.environ.get(new_key)
    if value is None:
        old_value = os.environ.get(old_key)
        if old_value is not None:
            warnings.warn(
                f'Environment variable {old_key} is deprecated. Use {new_key} instead.',
                DeprecationWarning,
                stacklevel=3,
            )
            return old_value
    return value


def get_pluto_config_from_env() -> Optional[Dict[str, Any]]:
    """
    Extract Pluto configuration from environment variables.

    Returns:
        Config dict if PLUTO_PROJECT is set, None otherwise.
    """
    project = get_env_with_deprecation('PLUTO_PROJECT', 'MLOP_PROJECT')
    if not project:
        return None

    config: Dict[str, Any] = {'project': project}

    if api_key := get_env_with_deprecation('PLUTO_API_KEY', 'MLOP_API_KEY'):
        config['api_key'] = api_key
    if url_app := get_env_with_deprecation('PLUTO_URL_APP', 'MLOP_URL_APP'):
        config['url_app'] = url_app
    if url_api := get_env_with_deprecation('PLUTO_URL_API', 'MLOP_URL_API'):
        config['url_api'] = url_api
    if url_ingest := get_env_with_deprecation('PLUTO_URL_INGEST', 'MLOP_URL_INGEST'):
        config['url_ingest'] = url_ingest

    return config


def safe_import_pluto():
    """Safely import pluto, returning None if unavailable."""
    try:
        import pluto

        return pluto
    except ImportError:
        logger.warning('pluto.compat: pluto not installed, compat layer inactive')
        return None
