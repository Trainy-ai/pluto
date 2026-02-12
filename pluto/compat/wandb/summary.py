"""wandb.summary-compatible dict-like summary metrics object."""

from typing import Any, Dict, Iterator, Optional


class Summary:
    """A dict-like object tracking summary metrics, compatible with wandb.summary.

    Auto-populated from log() calls (last value per key for scalars).
    Supports manual overrides via dict/attribute access.
    """

    def __init__(self) -> None:
        object.__setattr__(self, '_data', {})

    def _update_from_log(self, data: Dict[str, Any]) -> None:
        """Called internally after each log() call to update last values."""
        store = object.__getattribute__(self, '_data')
        for k, v in data.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                store[k] = v
            elif hasattr(v, 'item') and callable(v.item):
                store[k] = v.item()

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
            raise AttributeError(f"'Summary' object has no attribute '{key}'")

    # -- Dict access --

    def __setitem__(self, key: str, value: Any) -> None:
        object.__getattribute__(self, '_data')[key] = value

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

    def update(self, d: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if d:
            object.__getattribute__(self, '_data').update(d)
        if kwargs:
            object.__getattribute__(self, '_data').update(kwargs)

    def as_dict(self) -> Dict[str, Any]:
        return dict(object.__getattribute__(self, '_data'))
