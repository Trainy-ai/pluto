"""Neptune-query compatibility shim.

Drop-in replacement for ``neptune_query`` that reads from Pluto instead.

Usage::

    # Before (Neptune):
    from neptune_query import runs as nq_runs
    from neptune_query.filters import AttributeFilter, Filter

    # After (Pluto):
    from pluto.compat.neptune_query import runs as nq_runs
    from pluto.compat.neptune_query.filters import AttributeFilter, Filter
"""

from pluto.compat.neptune_query import filters, runs

__all__ = ['filters', 'runs']
