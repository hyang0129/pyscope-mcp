"""Typed return shapes for all structured MCP tool responses.

All TypedDicts here are used as return-type annotations in graph.py and as the
canonical shapes that server.py serialises to JSON. Introducing this module when
the neighborhood tool lands (issue #46) is intentional — a types.py earns its
existence when it houses multiple TypedDicts together rather than one in isolation.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

# Two-state completeness signal for every edge-traversing tool response.
# "complete"  — none of the traversed FQNs appear in missed_callers (directly
#               or via class-prefix match).  The result is probably the full picture.
# "partial"   — at least one traversed FQN is directly in missed_callers, OR is
#               a method (≥3 dotted segments) whose class prefix (e.g. a.b.C for
#               a.b.C.method) matches a missed_callers key.  The result is
#               definitely or likely incomplete — widen with grep.
Completeness = Literal["complete", "partial"]


class SearchResult(TypedDict):
    """Return shape for :meth:`CallGraphIndex.search`."""

    results: list[str]
    truncated: bool
    total_matched: int
    stale: bool
    stale_files: list[str]
    stale_action: NotRequired[str]
    index_stale_reason: NotRequired[str]
    commit_stale: NotRequired[bool | None]
    index_git_sha: NotRequired[str | None]
    head_git_sha: NotRequired[str | None]


class StatsResult(TypedDict):
    """Return shape for :meth:`CallGraphIndex.stats`."""

    functions: int
    function_edges: int
    modules: int
    module_edges: int
    commit_stale: NotRequired[bool | None]
    index_git_sha: NotRequired[str | None]
    head_git_sha: NotRequired[str | None]


class CallersResult(TypedDict):
    """Return shape for :meth:`CallGraphIndex.callers_of`."""

    results: list[str]
    truncated: bool
    dropped: int  # number of results cut by the cap; 0 when cap does not fire
    completeness: Completeness
    stale: bool
    stale_files: list[str]
    stale_action: NotRequired[str]
    index_stale_reason: NotRequired[str]
    commit_stale: NotRequired[bool | None]
    index_git_sha: NotRequired[str | None]
    head_git_sha: NotRequired[str | None]


class CalleesResult(TypedDict):
    """Return shape for :meth:`CallGraphIndex.callees_of`."""

    results: list[str]
    truncated: bool
    dropped: int  # number of results cut by the cap; 0 when cap does not fire
    completeness: Completeness
    stale: bool
    stale_files: list[str]
    stale_action: NotRequired[str]
    index_stale_reason: NotRequired[str]
    commit_stale: NotRequired[bool | None]
    index_git_sha: NotRequired[str | None]
    head_git_sha: NotRequired[str | None]


class ModuleResult(TypedDict):
    """Return shape for :meth:`CallGraphIndex.module_callers` and
    :meth:`CallGraphIndex.module_callees`."""

    results: list[str]
    truncated: bool
    dropped: int  # number of results cut by the cap; 0 when cap does not fire
    completeness: Completeness
    stale: bool
    stale_files: list[str]
    stale_action: NotRequired[str]
    index_stale_reason: NotRequired[str]
    commit_stale: NotRequired[bool | None]
    index_git_sha: NotRequired[str | None]
    head_git_sha: NotRequired[str | None]


class NeighborhoodResult(TypedDict, total=False):
    """Return shape for :meth:`CallGraphIndex.neighborhood`.

    All fields except ``depth_truncated``, ``stale_action``, and
    ``index_stale_reason`` are **always present**.
    ``depth_truncated`` is present **only** when ``truncated=True``; it is
    absent from the dict when ``truncated=False``.
    ``stale_action`` is present only when ``stale=True``.
    ``index_stale_reason`` is present only on the pre-v3 (pre-v4) index path.

    ``total=False`` is used (rather than a base-class split) because multiple
    fields must be omitted entirely when not applicable — ``NotRequired`` marks
    them as optional while the remaining fields are always constructed by
    :meth:`CallGraphIndex.neighborhood`.

    ``token_budget_used`` reflects the serialised character count / 4
    (4 chars/token estimate).

    Hub suppression fields (always present — never ``NotRequired``):
    ``hub_suppressed`` is a list of FQNs whose caller-side expansion was
    skipped because their in-degree exceeded the threshold. Always present;
    empty list when no suppression occurred. ``hub_threshold`` is the
    in-degree threshold that was applied for this query (the cached default
    or the per-call override). Consumers can reproduce or override the
    decision using these fields.
    """

    symbol: str
    depth_full: int
    depth_truncated: NotRequired[int]  # present only when truncated=True
    edges: list[list[str]]  # list of [caller, callee] pairs
    truncated: bool
    token_budget_used: int
    completeness: Completeness
    hub_suppressed: list[str]  # always present; empty when no suppression
    hub_threshold: int  # in-degree threshold applied for this query
    stale: bool
    stale_files: list[str]
    stale_action: NotRequired[str]
    index_stale_reason: NotRequired[str]
    commit_stale: NotRequired[bool | None]
    index_git_sha: NotRequired[str | None]
    head_git_sha: NotRequired[str | None]
