"""Standalone replacements for ``neptune_scale.types``.

Provides :class:`File` and :class:`Histogram` with the same constructor
signatures as the real ``neptune_scale.types`` so that existing code like::

    from neptune_scale.types import File, Histogram

works unchanged after switching to::

    from pluto.compat.neptune_scale.types import File, Histogram
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union


class File:
    """Drop-in replacement for ``neptune_scale.types.File``.

    Parameters
    ----------
    source:
        A file path (str / PathLike) **or** raw ``bytes``.
    mime_type:
        Optional MIME type hint (e.g. ``"image/png"``).
    """

    def __init__(
        self,
        source: Any,
        mime_type: Optional[str] = None,
    ) -> None:
        self.source = source
        self.mime_type = mime_type

    def __repr__(self) -> str:
        if isinstance(self.source, str):
            src = self.source
        else:
            src = type(self.source).__name__
        return f'File(source={src!r}, mime_type={self.mime_type!r})'


class Histogram:
    """Drop-in replacement for ``neptune_scale.types.Histogram``.

    Parameters
    ----------
    bin_edges:
        Sequence of N+1 bin edge values.
    counts:
        Sequence of N count values (mutually exclusive with *densities*).
    densities:
        Sequence of N density values (mutually exclusive with *counts*).
    """

    def __init__(
        self,
        bin_edges: Union[Sequence[float], Any],
        counts: Optional[Sequence[float]] = None,
        densities: Optional[Sequence[float]] = None,
    ) -> None:
        self.bin_edges = bin_edges
        self.counts = counts
        self.densities = densities

    def bin_edges_as_list(self) -> List[float]:
        return list(self.bin_edges)

    def counts_as_list(self) -> List[float]:
        if self.counts is None:
            return []
        return list(self.counts)

    def densities_as_list(self) -> List[float]:
        if self.densities is None:
            return []
        return list(self.densities)

    def __repr__(self) -> str:
        return (
            f'Histogram(bin_edges=<{len(self.bin_edges)} edges>, '
            f'counts={self.counts is not None}, '
            f'densities={self.densities is not None})'
        )
