"""wandb.summary-compatible dict-like summary metrics object."""

from fnmatch import fnmatch
from typing import Any, Dict, Iterator, Optional


class Summary:
    """A dict-like object tracking summary metrics, compatible with wandb.summary.

    Auto-populated from log() calls (last value per key for scalars).
    Supports manual overrides via dict/attribute access.
    When metric definitions specify aggregation (min/max/mean/first/last),
    summary values are computed accordingly instead of always keeping last.
    """

    def __init__(self) -> None:
        object.__setattr__(self, '_data', {})
        object.__setattr__(self, '_metric_definitions', {})
        # Tracks running state for mean aggregation: {key: {'sum': float, 'count': int}}
        object.__setattr__(self, '_agg_state', {})

    def _set_metric_definition(self, name: str, definition: Dict[str, Any]) -> None:
        """Register a metric definition for summary aggregation."""
        defs = object.__getattribute__(self, '_metric_definitions')
        defs[name] = definition

    def _find_definition(self, key: str) -> Optional[Dict[str, Any]]:
        """Find a metric definition by exact match first, then glob pattern."""
        defs = object.__getattribute__(self, '_metric_definitions')
        if key in defs:
            return defs[key]
        for pattern, defn in defs.items():
            if fnmatch(key, pattern):
                return defn
        return None

    def _update_from_log(self, data: Dict[str, Any]) -> None:
        """Called internally after each log() call to update summary values."""
        store = object.__getattribute__(self, '_data')
        agg_state = object.__getattribute__(self, '_agg_state')
        for k, v in data.items():
            if isinstance(v, (int, float)):
                val = v
            elif hasattr(v, 'item') and callable(v.item):
                val = v.item()
            else:
                continue

            defn = self._find_definition(k)
            if defn is None:
                # Default: keep last value
                store[k] = val
                continue

            summary_mode = defn.get('summary')
            if summary_mode == 'min':
                if k in store:
                    store[k] = min(store[k], val)
                else:
                    store[k] = val
            elif summary_mode == 'max':
                if k in store:
                    store[k] = max(store[k], val)
                else:
                    store[k] = val
            elif summary_mode == 'mean':
                if k not in agg_state:
                    agg_state[k] = {'sum': 0.0, 'count': 0}
                agg_state[k]['sum'] += val
                agg_state[k]['count'] += 1
                store[k] = agg_state[k]['sum'] / agg_state[k]['count']
            elif summary_mode == 'first':
                if k not in store:
                    store[k] = val
            else:
                # "last" or unrecognized — keep last value
                store[k] = val

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
