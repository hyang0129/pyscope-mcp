"""Shared pytest fixtures for the pyscope-mcp test suite.

The single helper exported here, :func:`make_nodes`, converts a compact
``{caller: [callees]}`` dict (the legacy ``raw`` shape) into the site-keyed
``nodes`` shape introduced by epic #76 child #1.

Centralising the conversion keeps every fixture site in the test corpus
compact: each test still expresses its graph as ``{caller: [callees]}`` —
the most natural representation for call-only tests — and lets the helper
expand to the site-keyed record shape ``CallGraphIndex.from_nodes``
expects.

The helper also supports building call edges of an arbitrary kind via the
``kind`` keyword and unioning multiple kind buckets through repeated calls
in the same test (``make_nodes(raw1, kind="call")`` then merged with
``make_nodes(raw2, kind="import")``).  This keeps the next-edge-kind PR
cost bounded to one helper call per kind, not one fixture migration per
site (Success Metric 3 of epic #76).
"""

from __future__ import annotations


def make_nodes(
    raw: dict[str, list[str]],
    *,
    kind: str = "call",
) -> dict[str, dict]:
    """Convert a ``{caller: [callees]}`` dict to the site-keyed nodes shape.

    Each key in *raw* becomes (or extends) a node with
    ``calls[kind] = sorted(callees)``.  Each callee gets a corresponding
    ``called_by[kind]`` entry naming its caller.  Empty callee lists are
    preserved as nodes with no edges so callers like ``CallGraphIndex
    .callers_of`` see the FQN as present-but-isolated rather than missing.

    Determinism: callee/caller lists are sorted within each kind bucket so
    ``make_nodes(raw)`` and ``make_nodes(reversed_raw)`` for the same
    logical edges produce dict equal output (and, when fed through
    ``CallGraphIndex.save``, byte-identical JSON).

    Parameters
    ----------
    raw:
        Compact ``{caller_fqn: [callee_fqn, ...]}`` mapping.  Same shape as
        the pre-migration ``raw`` field; preserved here as the ergonomic
        author-facing form.
    kind:
        Edge kind to file the callees under.  Defaults to ``"call"`` so
        existing tests get call edges for free; pass ``"import"``,
        ``"except"``, ``"annotation"``, or ``"isinstance"`` to build
        fixtures for the other kinds defined by epic #76.

    Returns
    -------
    dict[str, dict]
        Site-keyed nodes mapping suitable for ``CallGraphIndex.from_nodes``.
        Every symbol that appears as a caller or callee is present as a key,
        even when its edge buckets are empty for the requested kind.
    """
    forward: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {}
    for caller, callees in raw.items():
        forward.setdefault(caller, set()).update(callees)
        for callee in callees:
            reverse.setdefault(callee, set()).add(caller)
            forward.setdefault(callee, set())  # ensure node exists

    all_symbols = set(forward) | set(reverse)
    nodes: dict[str, dict] = {}
    for sym in sorted(all_symbols):
        record: dict[str, dict[str, list[str]]] = {"calls": {}, "called_by": {}}
        callees = forward.get(sym, set())
        if callees:
            record["calls"][kind] = sorted(callees)
        callers = reverse.get(sym, set())
        if callers:
            record["called_by"][kind] = sorted(callers)
        nodes[sym] = record
    return nodes
