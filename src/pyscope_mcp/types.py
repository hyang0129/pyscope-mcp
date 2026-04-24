"""Typed return shapes for all structured MCP tool responses.

All TypedDicts here are used as return-type annotations in graph.py and as the
canonical shapes that server.py serialises to JSON. Introducing this module when
the neighborhood tool lands (issue #46) is intentional — a types.py earns its
existence when it houses multiple TypedDicts together rather than one in isolation.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class SearchResult(TypedDict):
    """Return shape for :meth:`CallGraphIndex.search`."""

    results: list[str]
    truncated: bool
    total_matched: int


class StatsResult(TypedDict):
    """Return shape for :meth:`CallGraphIndex.stats`."""

    functions: int
    function_edges: int
    modules: int
    module_edges: int


class CallersResult(TypedDict):
    """Return shape for :meth:`CallGraphIndex.callers_of`."""

    results: list[str]
    truncated: bool


class CalleesResult(TypedDict):
    """Return shape for :meth:`CallGraphIndex.callees_of`."""

    results: list[str]
    truncated: bool


class NeighborhoodResult(TypedDict, total=False):
    """Return shape for :meth:`CallGraphIndex.neighborhood`.

    All fields except ``depth_truncated`` are **always present**.
    ``depth_truncated`` is present **only** when ``truncated=True``; it is
    absent from the dict when ``truncated=False``.

    ``total=False`` is used (rather than a base-class split) because
    ``depth_truncated`` must be omitted entirely (not set to ``None``) in
    the non-truncated path — ``NotRequired`` marks it as the sole optional
    field while the remaining fields are always constructed by
    :meth:`CallGraphIndex.neighborhood`.

    ``token_budget_used`` reflects the serialised character count / 4
    (4 chars/token estimate).
    """

    symbol: str
    depth_full: int
    depth_truncated: NotRequired[int]  # present only when truncated=True
    edges: list[list[str]]  # list of [caller, callee] pairs
    truncated: bool
    token_budget_used: int
