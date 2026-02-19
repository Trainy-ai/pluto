"""Neptune-query filter compatibility classes.

Provides :class:`Filter` and :class:`AttributeFilter` with the same API
as ``neptune_query.filters`` so existing code works without changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Mapping from Neptune ``sys/*`` attribute paths to Pluto run-dict keys.
_SYS_ATTR_MAP: Dict[str, str] = {
    'sys/id': 'displayId',
    'sys/name': 'name',
    'sys/creation_time': 'createdAt',
}


def _resolve_attribute(run: Dict[str, Any], attr: str) -> Any:
    """Resolve a Neptune-style attribute path against a Pluto run dict.

    Resolution order:
    1. ``sys/*`` attributes via ``_SYS_ATTR_MAP``
    2. Special ``id`` attribute → ``displayId``
    3. Flat key in ``config`` dict
    4. Nested ``/``-separated lookup in ``config``
    5. Presence check in ``logNames`` list (returns ``True`` / ``None``)
    """
    # sys/* attributes
    if attr in _SYS_ATTR_MAP:
        return run.get(_SYS_ATTR_MAP[attr])

    # Bare "id" → displayId
    if attr == 'id':
        return run.get('displayId')

    config = run.get('config') or {}

    # Flat key lookup
    if attr in config:
        return config[attr]

    # Nested slash-separated lookup
    parts = attr.split('/')
    node: Any = config
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            node = None
            break
    if node is not None:
        return node

    # Check logNames for file/metric existence
    log_names = run.get('logNames') or []
    if attr in log_names:
        return True

    return None


@dataclass
class _Predicate:
    """A single filter predicate."""

    attr: str
    op: str  # "exists" or "matches"
    value: Any = None


@dataclass
class Filter:
    """Client-side filter for Pluto runs, matching the Neptune ``Filter`` API.

    Supports ``exists`` and ``matches`` predicates combined with ``&``.

    Examples::

        f = Filter.exists("model/best_model_path")
        f = Filter.matches("sys/id", "MQT-42")
        f = Filter.exists("model/best_model_path") & Filter.matches("sys/id", "MQT-42")
    """

    _predicates: List[_Predicate] = field(default_factory=list)

    # ----- constructors -----

    @classmethod
    def exists(cls, attribute_name: str) -> Filter:
        return cls(_predicates=[_Predicate(attr=attribute_name, op='exists')])

    @classmethod
    def matches(cls, attribute_name: str, value: Any) -> Filter:
        return cls(
            _predicates=[_Predicate(attr=attribute_name, op='matches', value=value)]
        )

    # ----- combinators -----

    def __and__(self, other: Filter) -> Filter:
        return Filter(_predicates=self._predicates + other._predicates)

    def __iand__(self, other: Filter) -> Filter:
        self._predicates.extend(other._predicates)
        return self

    # ----- evaluation -----

    def evaluate(self, run: Dict[str, Any]) -> bool:
        """Test whether *run* satisfies all predicates (AND logic)."""
        for pred in self._predicates:
            val = _resolve_attribute(run, pred.attr)
            if pred.op == 'exists':
                if val is None:
                    return False
            elif pred.op == 'matches':
                if val is None:
                    return False
                if str(val) != str(pred.value):
                    return False
        return True

    # ----- introspection helpers -----

    def get_match_value(self, attribute_name: str) -> Optional[str]:
        """Return the value for a ``matches`` predicate on *attribute_name*, if any."""
        for pred in self._predicates:
            if pred.op == 'matches' and pred.attr == attribute_name:
                return str(pred.value)
        return None


@dataclass
class AttributeFilter:
    """Filter for metric/attribute names, matching the Neptune ``AttributeFilter`` API.

    Args:
        name: A regex string **or** a list of exact metric names.
        type: Ignored (Pluto metrics are all numeric series).

    Examples::

        AttributeFilter(name=r"test/raw/", type="float_series")
        AttributeFilter(name=["loss", "accuracy"])
    """

    name: Union[str, List[str]]
    type: Optional[str] = None

    def matches_name(self, candidate: str) -> bool:
        """Return ``True`` if *candidate* matches this filter."""
        if isinstance(self.name, list):
            return candidate in self.name
        return bool(re.search(self.name, candidate))
