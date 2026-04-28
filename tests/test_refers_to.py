"""Tests for the refers_to() method (issue #71).

Covers all reference kinds (call, import, except, annotation, isinstance),
both granularities (function/module), both kind filters (all/callers),
depth enforcement, depth_exceeds_max guard, fqn_not_in_graph error,
context priority (call beats others), deduplication, and BFS ordering.

These tests use synthetic node dicts so no source files or real analyzer
runs are needed.  The conftest.make_nodes helper builds call edges; for
non-call edge kinds we build the nodes dict directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyscope_mcp.graph import CallGraphIndex
from conftest import make_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nodes_with_kind(fqn_from: str, fqn_to: str, kind: str) -> dict[str, dict]:
    """Build a minimal nodes dict with one edge of the given kind."""
    return {
        fqn_from: {
            "calls": {kind: [fqn_to]},
            "called_by": {},
        },
        fqn_to: {
            "calls": {},
            "called_by": {kind: [fqn_from]},
        },
    }


def _merge_nodes(*node_dicts: dict[str, dict]) -> dict[str, dict]:
    """Merge multiple node dicts, union-ing their calls/called_by buckets."""
    merged: dict[str, dict] = {}
    for nd in node_dicts:
        for fqn, node in nd.items():
            if fqn not in merged:
                merged[fqn] = {"calls": {}, "called_by": {}}
            for kind, targets in node.get("calls", {}).items():
                merged[fqn]["calls"].setdefault(kind, [])
                merged[fqn]["calls"][kind] = sorted(
                    set(merged[fqn]["calls"][kind]) | set(targets)
                )
            for kind, sources in node.get("called_by", {}).items():
                merged[fqn]["called_by"].setdefault(kind, [])
                merged[fqn]["called_by"][kind] = sorted(
                    set(merged[fqn]["called_by"][kind]) | set(sources)
                )
    return merged


def _idx(tmp_path: Path, nodes: dict[str, dict]) -> CallGraphIndex:
    return CallGraphIndex.from_nodes(str(tmp_path), nodes)


# ---------------------------------------------------------------------------
# 1. All five reference kinds are found by kind="all"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind_name", ["call", "import", "except", "annotation", "isinstance"])
def test_refers_to_all_finds_each_kind(tmp_path: Path, kind_name: str) -> None:
    """kind='all' must find referencing functions for every supported edge kind."""
    target = "pkg.target.Symbol"
    referrer = "pkg.user.consumer"
    nodes = _nodes_with_kind(referrer, target, kind_name)
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to(target, kind="all", depth=1)

    assert result.get("isError") is not True, f"unexpected error: {result}"
    fqns = [e["fqn"] for e in result["results"]]
    assert referrer in fqns, (
        f"kind='{kind_name}' referrer not found; results={fqns}"
    )
    # context field must match the edge kind
    entry = next(e for e in result["results"] if e["fqn"] == referrer)
    assert entry["context"] == kind_name, (
        f"expected context='{kind_name}', got '{entry['context']}'"
    )


# ---------------------------------------------------------------------------
# 2. kind="callers" finds only call edges
# ---------------------------------------------------------------------------

def test_refers_to_callers_excludes_non_call_edges(tmp_path: Path) -> None:
    """kind='callers' must only traverse call edges, not import/except/etc."""
    target = "pkg.target.Symbol"
    call_referrer = "pkg.a.caller"
    import_referrer = "pkg.b.importer"

    nodes = _merge_nodes(
        _nodes_with_kind(call_referrer, target, "call"),
        _nodes_with_kind(import_referrer, target, "import"),
    )
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to(target, kind="callers", depth=1)
    fqns = [e["fqn"] for e in result["results"]]

    assert call_referrer in fqns, "call referrer must appear in kind='callers' results"
    assert import_referrer not in fqns, "import referrer must NOT appear in kind='callers' results"


# ---------------------------------------------------------------------------
# 3. context priority: "call" beats other kinds when a function does both
# ---------------------------------------------------------------------------

def test_refers_to_call_context_priority(tmp_path: Path) -> None:
    """When a function has both call and import edges to target, context must be 'call'."""
    target = "pkg.target.Symbol"
    referrer = "pkg.a.multi"

    # referrer both imports and calls target
    nodes = _merge_nodes(
        _nodes_with_kind(referrer, target, "call"),
        _nodes_with_kind(referrer, target, "import"),
    )
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to(target, kind="all", depth=1)
    assert result.get("isError") is not True
    entries = [e for e in result["results"] if e["fqn"] == referrer]
    assert len(entries) == 1, f"referrer must appear exactly once, got {entries}"
    assert entries[0]["context"] == "call", (
        f"'call' must win context priority, got '{entries[0]['context']}'"
    )


# ---------------------------------------------------------------------------
# 4. depth=1 vs depth=2 BFS
# ---------------------------------------------------------------------------

def test_refers_to_depth1_does_not_include_depth2(tmp_path: Path) -> None:
    """depth=1 must not include depth-2 referrers."""
    target = "pkg.target.fn"
    depth1 = "pkg.a.direct"
    depth2 = "pkg.b.indirect"

    nodes = _merge_nodes(
        _nodes_with_kind(depth1, target, "call"),
        _nodes_with_kind(depth2, depth1, "call"),
    )
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to(target, kind="all", depth=1)
    fqns = [e["fqn"] for e in result["results"]]
    assert depth1 in fqns
    assert depth2 not in fqns, "depth-2 referrer must not appear with depth=1"


def test_refers_to_depth2_includes_both_depths(tmp_path: Path) -> None:
    """depth=2 must include both direct (depth-1) and transitive (depth-2) referrers."""
    target = "pkg.target.fn"
    depth1 = "pkg.a.direct"
    depth2 = "pkg.b.indirect"

    nodes = _merge_nodes(
        _nodes_with_kind(depth1, target, "call"),
        _nodes_with_kind(depth2, depth1, "call"),
    )
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to(target, kind="all", depth=2)
    fqns = [e["fqn"] for e in result["results"]]
    assert depth1 in fqns
    assert depth2 in fqns


def test_refers_to_depth_exceeds_max(tmp_path: Path) -> None:
    """depth > 2 must return isError:True with error_reason='depth_exceeds_max'."""
    nodes = make_nodes({"pkg.a.fn": []})
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to("pkg.a.fn", kind="all", depth=3)
    assert result.get("isError") is True
    assert result.get("error_reason") == "depth_exceeds_max"


# ---------------------------------------------------------------------------
# 5. fqn_not_in_graph error
# ---------------------------------------------------------------------------

def test_refers_to_fqn_not_in_graph(tmp_path: Path) -> None:
    """Absent FQN must return isError:True with error_reason='fqn_not_in_graph'."""
    nodes = make_nodes({"pkg.a.fn": []})
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to("does.not.exist", kind="all", depth=1)
    assert result.get("isError") is True
    assert result.get("error_reason") == "fqn_not_in_graph"


# ---------------------------------------------------------------------------
# 6. Function granularity result shape
# ---------------------------------------------------------------------------

def test_refers_to_function_granularity_result_shape(tmp_path: Path) -> None:
    """Function granularity results must be dicts with fqn, context, depth fields."""
    target = "pkg.target.fn"
    referrer = "pkg.a.caller"
    nodes = _nodes_with_kind(referrer, target, "call")
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to(target, kind="all", granularity="function", depth=1)
    assert result.get("isError") is not True
    assert len(result["results"]) == 1

    entry = result["results"][0]
    assert "fqn" in entry
    assert "context" in entry
    assert "depth" in entry
    assert entry["fqn"] == referrer
    assert entry["context"] == "call"
    assert entry["depth"] == 1


# ---------------------------------------------------------------------------
# 7. Module granularity result shape
# ---------------------------------------------------------------------------

def test_refers_to_module_granularity_result_shape(tmp_path: Path) -> None:
    """Module granularity results must be a flat list of module FQN strings."""
    target = "pkg.target.fn"
    referrer = "pkg.caller_mod.some_func"
    nodes = _nodes_with_kind(referrer, target, "call")
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to(target, kind="callers", granularity="module", depth=1)
    assert result.get("isError") is not True
    # results should be strings (module FQNs)
    assert "pkg.caller_mod" in result["results"]
    for item in result["results"]:
        assert isinstance(item, str), f"module result item must be str, got {type(item)}: {item!r}"


# ---------------------------------------------------------------------------
# 8. Deduplication: same FQN via multiple edge kinds appears once
# ---------------------------------------------------------------------------

def test_refers_to_deduplication_across_kinds(tmp_path: Path) -> None:
    """A function that appears via multiple edge kinds must appear only once in results."""
    target = "pkg.target.Symbol"
    referrer = "pkg.a.multi"

    # referrer has import, call, and annotation edges to target
    nodes = _merge_nodes(
        _nodes_with_kind(referrer, target, "call"),
        _nodes_with_kind(referrer, target, "import"),
        _nodes_with_kind(referrer, target, "annotation"),
    )
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to(target, kind="all", depth=1)
    assert result.get("isError") is not True
    fqns = [e["fqn"] for e in result["results"]]
    assert fqns.count(referrer) == 1, (
        f"referrer must appear exactly once despite multiple edge kinds; got count={fqns.count(referrer)}"
    )


# ---------------------------------------------------------------------------
# 9. BFS depth ordering: depth-1 before depth-2
# ---------------------------------------------------------------------------

def test_refers_to_depth1_sorted_before_depth2(tmp_path: Path) -> None:
    """In depth=2 results, depth-1 entries must appear before depth-2 entries."""
    target = "pkg.target.fn"
    depth1 = "pkg.z.direct"   # z > a alphabetically — would sort last without BFS ranking
    depth2 = "pkg.a.indirect"  # a < z alphabetically — would sort first without BFS ranking

    nodes = _merge_nodes(
        _nodes_with_kind(depth1, target, "call"),
        _nodes_with_kind(depth2, depth1, "call"),
    )
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to(target, kind="all", depth=2)
    fqns = [e["fqn"] for e in result["results"]]

    assert depth1 in fqns
    assert depth2 in fqns
    assert fqns.index(depth1) < fqns.index(depth2), (
        "depth-1 referrer must appear before depth-2 referrer regardless of alphabetical order"
    )


# ---------------------------------------------------------------------------
# 10. Zero referrers — empty results is not an error
# ---------------------------------------------------------------------------

def test_refers_to_zero_referrers_is_not_error(tmp_path: Path) -> None:
    """A present FQN with zero referrers must return results=[] not isError."""
    nodes = make_nodes({"pkg.mod.isolated": []})
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to("pkg.mod.isolated", kind="all", depth=1)
    assert result.get("isError") is not True
    assert result["results"] == []
    assert result["truncated"] is False
    assert result["dropped"] == 0


# ---------------------------------------------------------------------------
# 11. Standard response fields present
# ---------------------------------------------------------------------------

def test_refers_to_standard_fields_present(tmp_path: Path) -> None:
    """Result dict must always include truncated, dropped, completeness, stale, stale_files."""
    nodes = make_nodes({"pkg.mod.fn": []})
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to("pkg.mod.fn", kind="all", depth=1)
    for field in ("results", "truncated", "dropped", "completeness", "stale", "stale_files"):
        assert field in result, f"field '{field}' must be present in refers_to result"


# ---------------------------------------------------------------------------
# 12. Module granularity deduplication across multiple symbols
# ---------------------------------------------------------------------------

def test_refers_to_module_granularity_deduplicates_modules(tmp_path: Path) -> None:
    """Two callers from the same module must result in that module appearing once."""
    target = "pkg.target.fn"
    caller1 = "pkg.caller_mod.func_a"
    caller2 = "pkg.caller_mod.func_b"

    nodes = _merge_nodes(
        _nodes_with_kind(caller1, target, "call"),
        _nodes_with_kind(caller2, target, "call"),
    )
    idx = _idx(tmp_path, nodes)

    result = idx.refers_to(target, kind="callers", granularity="module", depth=1)
    assert result.get("isError") is not True
    assert result["results"].count("pkg.caller_mod") == 1, (
        "same caller module must appear only once in module granularity results"
    )
