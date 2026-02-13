"""wandb.config-compatible dict-like configuration object."""

import logging
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'WandbCompat.Config'


class Config:
    """A dict-like configuration object compatible with wandb.config.

    Supports attribute access, dict access, and syncs mutations to pluto
    via op.update_config().
    """

    def __init__(self, op: Optional[Any] = None) -> None:
        # Use object.__setattr__ to avoid triggering our __setattr__
        object.__setattr__(self, '_op', op)
        object.__setattr__(self, '_data', {})

    def _load(self, data: Optional[Dict[str, Any]]) -> None:
        if data:
            object.__getattribute__(self, '_data').update(data)

    def _sync(self, updates: Dict[str, Any]) -> None:
        op = object.__getattribute__(self, '_op')
        if op is not None:
            try:
                op.update_config(updates)
            except Exception as e:
                logger.debug('%s: failed to sync config: %s', tag, e)

    # -- Attribute access --

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith('_'):
            object.__setattr__(self, key, value)
        else:
            self[key] = value

    def __getattr__(self, key: str) -> Any:
        data = object.__getattribute__(self, '_data')
        try:
            return data[key]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __delattr__(self, key: str) -> None:
        data = object.__getattribute__(self, '_data')
        if key in data:
            del data[key]
        else:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    # -- Dict access --

    def __setitem__(self, key: str, value: Any) -> None:
        data = object.__getattribute__(self, '_data')
        data[key] = value
        self._sync({key: value})

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, '_data')[key]

    def __delitem__(self, key: str) -> None:
        del object.__getattribute__(self, '_data')[key]

    def __contains__(self, key: object) -> bool:
        return key in object.__getattribute__(self, '_data')

    def __iter__(self) -> Iterator[str]:
        return iter(object.__getattribute__(self, '_data'))

    def __len__(self) -> int:
        return len(object.__getattribute__(self, '_data'))

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, '_data'))

    def __bool__(self) -> bool:
        return bool(object.__getattribute__(self, '_data'))

    # -- Dict-like methods --

    def keys(self):
        return object.__getattribute__(self, '_data').keys()

    def values(self):
        return object.__getattribute__(self, '_data').values()

    def items(self):
        return object.__getattribute__(self, '_data').items()

    def get(self, key: str, default: Any = None) -> Any:
        return object.__getattribute__(self, '_data').get(key, default)

    def update(
        self,
        d: Any = None,
        allow_val_change: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Update config with a dict, namespace, or keyword arguments."""
        if d is not None:
            # Support argparse namespaces
            if hasattr(d, '__dict__') and not isinstance(d, dict):
                d = vars(d)
            if isinstance(d, dict):
                object.__getattribute__(self, '_data').update(d)
                self._sync(d)
        if kwargs:
            object.__getattribute__(self, '_data').update(kwargs)
            self._sync(kwargs)

    def setdefaults(self, d: Dict[str, Any]) -> None:
        """Set defaults — only sets keys that are not already present."""
        data = object.__getattribute__(self, '_data')
        updates = {}
        for k, v in d.items():
            if k not in data:
                data[k] = v
                updates[k] = v
        if updates:
            self._sync(updates)

    def as_dict(self) -> Dict[str, Any]:
        return dict(object.__getattribute__(self, '_data'))
