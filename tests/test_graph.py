from __future__ import annotations

import json
from pathlib import Path

from pyscope_mcp.graph import CallGraphIndex


def _sample_raw() -> dict[str, list[str]]:
    return {
        "sample.a.top": ["sample.b.helper"],
        "sample.b.helper": ["sample.b.inner"],
        "sample.b.inner": [],
    }


def test_from_raw_builds_graphs() -> None:
    idx = CallGraphIndex.from_raw("/tmp/sample", _sample_raw())
    stats = idx.stats()
    assert stats["functions"] >= 3
    assert stats["function_edges"] == 2
    assert stats["modules"] >= 2


def test_queries() -> None:
    idx = CallGraphIndex.from_raw("/tmp/sample", _sample_raw())
    # callers_of / callees_of now return dict shapes
    callees = idx.callees_of("sample.a.top", depth=1)
    assert isinstance(callees, dict)
    assert "results" in callees
    assert "truncated" in callees
    assert "sample.b.helper" in callees["results"]

    callees_deep = idx.callees_of("sample.a.top", depth=2)
    assert "sample.b.inner" in callees_deep["results"]

    callers = idx.callers_of("sample.b.helper", depth=1)
    assert isinstance(callers, dict)
    assert "results" in callers
    assert "truncated" in callers
    assert "sample.a.top" in callers["results"]

    result = idx.search("helper")
    assert result["results"] == ["sample.b.helper"]
    assert result["truncated"] is False
    assert result["total_matched"] == 1


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    idx = CallGraphIndex.from_raw("/tmp/sample", _sample_raw())
    out = tmp_path / "index.json"
    idx.save(out)

    loaded = CallGraphIndex.load(out)
    assert loaded.stats() == idx.stats()
    # Compare only the query results (not staleness — the original idx has file_shas=None
    # while the saved+loaded v4 index has file_shas={} with no stored hashes and the
    # symbol files aren't on disk, so staleness differs by design).
    assert loaded.search("helper")["results"] == idx.search("helper")["results"]
    assert loaded.search("helper")["truncated"] == idx.search("helper")["truncated"]
    # v4 schema: missed_callers round-trips correctly (default is empty)
    assert loaded.missed_callers == {}
    assert loaded.completeness_for(["sample.a.top"]) == "complete"


def _prefix_raw() -> dict[str, list[str]]:
    """Raw data for prefix-match tests.

    Module graph derived from this:
      pkg.agents.writer  →  pkg.io.disk
      pkg.agents.reader  →  pkg.io.network
      pkg.core           →  pkg.agents.writer, pkg.agents.reader
    """
    return {
        "pkg.agents.writer.write": ["pkg.io.disk.save"],
        "pkg.agents.reader.read": ["pkg.io.network.fetch"],
        "pkg.core.run": ["pkg.agents.writer.write", "pkg.agents.reader.read"],
        "pkg.io.disk.save": [],
        "pkg.io.network.fetch": [],
    }


def test_module_callers_prefix_match() -> None:
    """module_callers with a prefix returns callers of all matched modules."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callers("pkg.agents")
    assert result["truncated"] is False
    # pkg.core calls into pkg.agents.writer and pkg.agents.reader
    assert "pkg.core" in result["results"]
    # The matched seeds themselves must NOT appear in results
    assert "pkg.agents.writer" not in result["results"]
    assert "pkg.agents.reader" not in result["results"]


def test_module_callees_prefix_match() -> None:
    """module_callees with a prefix returns callees of all matched modules."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callees("pkg.agents")
    assert result["truncated"] is False
    # pkg.agents.writer calls pkg.io.disk; pkg.agents.reader calls pkg.io.network
    assert "pkg.io.disk" in result["results"]
    assert "pkg.io.network" in result["results"]


def test_module_callers_exact_match_backward_compat() -> None:
    """An exact FQN returns the same callers as before, now in structured form."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callers("pkg.agents.writer")
    assert isinstance(result, dict)
    assert "results" in result
    assert "truncated" in result
    assert result["truncated"] is False
    assert "pkg.core" in result["results"]


def test_module_callers_no_match() -> None:
    """A prefix matching no module nodes returns empty results, no error."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callers("nonexistent.pkg")
    assert result["results"] == []
    assert result["truncated"] is False
    assert "stale" in result  # staleness fields always present


def test_module_callees_no_match() -> None:
    """A prefix matching no module nodes returns empty results, no error."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callees("nonexistent.pkg")
    assert result["results"] == []
    assert result["truncated"] is False
    assert "stale" in result


def test_module_callers_empty_prefix() -> None:
    """Empty-string prefix matches all modules; returns structured dict with no error."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callers("")
    # All modules are seeds — callers within the same set are excluded.
    # The important invariants are shape and no error.
    assert isinstance(result, dict)
    assert "results" in result
    assert "truncated" in result
    assert isinstance(result["results"], list)
    assert isinstance(result["truncated"], bool)


def test_module_callees_empty_prefix() -> None:
    """Empty-string prefix matches all modules; returns structured dict with no error."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callees("")
    assert isinstance(result, dict)
    assert "results" in result
    assert "truncated" in result
    assert isinstance(result["results"], list)
    assert isinstance(result["truncated"], bool)


def test_module_callers_truncation() -> None:
    """When more than 50 distinct callers exist, truncated=True and len==50."""
    # Build a graph where one "target" module has 60 distinct callers
    raw: dict[str, list[str]] = {}
    for i in range(60):
        raw[f"callers.mod{i}.fn"] = [f"target.core.fn"]
    raw["target.core.fn"] = []
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)
    result = idx.module_callers("target")
    assert result["truncated"] is True
    assert len(result["results"]) == 50


def test_module_callees_truncation() -> None:
    """When more than 50 distinct callees exist, truncated=True and len==50."""
    raw: dict[str, list[str]] = {
        "source.mod.fn": [f"target.mod{i}.fn" for i in range(60)]
    }
    for i in range(60):
        raw[f"target.mod{i}.fn"] = []
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)
    result = idx.module_callees("source")
    assert result["truncated"] is True
    assert len(result["results"]) == 50


def test_search_truncation() -> None:
    """When matches exceed the cap, truncated=True and total_matched reflects the full count."""
    # Build a raw graph with 5 symbols all containing "fn"
    raw = {f"pkg.mod.fn{i}": [] for i in range(5)}
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)

    # Cap at 3 — should truncate
    result = idx.search("fn", limit=3)
    assert len(result["results"]) == 3
    assert result["truncated"] is True
    assert result["total_matched"] == 5

    # Cap at 10 — should not truncate
    result_all = idx.search("fn", limit=10)
    assert len(result_all["results"]) == 5
    assert result_all["truncated"] is False
    assert result_all["total_matched"] == 5

    # Cap exactly equal to match count — truncated must be False (condition is total > limit, not >=)
    result_exact = idx.search("fn", limit=5)
    assert len(result_exact["results"]) == 5
    assert result_exact["truncated"] is False
    assert result_exact["total_matched"] == 5


# ---------------------------------------------------------------------------
# callers_of / callees_of — dict shape
# ---------------------------------------------------------------------------

def test_callers_of_returns_dict() -> None:
    """callers_of returns {results: list[str], truncated: bool}, not a bare list."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _sample_raw())
    result = idx.callers_of("sample.b.helper", depth=1)
    assert isinstance(result, dict)
    assert "results" in result
    assert "truncated" in result
    assert isinstance(result["results"], list)
    assert isinstance(result["truncated"], bool)
    assert "sample.a.top" in result["results"]


def test_callees_of_returns_dict() -> None:
    """callees_of returns {results: list[str], truncated: bool}, not a bare list."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _sample_raw())
    result = idx.callees_of("sample.a.top", depth=1)
    assert isinstance(result, dict)
    assert "results" in result
    assert "truncated" in result
    assert isinstance(result["results"], list)
    assert isinstance(result["truncated"], bool)
    assert "sample.b.helper" in result["results"]


def test_callers_of_truncation() -> None:
    """callers_of caps at 50 and sets truncated=True when exceeded."""
    # Build a raw graph where one function has 60 distinct callers
    raw: dict[str, list[str]] = {}
    for i in range(60):
        raw[f"callers.mod{i}.fn"] = ["target.core.fn"]
    raw["target.core.fn"] = []
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)
    result = idx.callers_of("target.core.fn", depth=1)
    assert result["truncated"] is True
    assert len(result["results"]) == 50


def test_callees_of_truncation() -> None:
    """callees_of caps at 50 and sets truncated=True when exceeded."""
    raw: dict[str, list[str]] = {
        "source.mod.fn": [f"target.mod{i}.fn" for i in range(60)]
    }
    for i in range(60):
        raw[f"target.mod{i}.fn"] = []
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)
    result = idx.callees_of("source.mod.fn", depth=1)
    assert result["truncated"] is True
    assert len(result["results"]) == 50


# ---------------------------------------------------------------------------
# neighborhood
# ---------------------------------------------------------------------------

def _neighborhood_raw() -> dict[str, list[str]]:
    """
    Simple chain: a.top → b.middle → c.leaf
    Also: d.caller → b.middle
    """
    return {
        "a.top": ["b.middle"],
        "b.middle": ["c.leaf"],
        "c.leaf": [],
        "d.caller": ["b.middle"],
    }


def test_neighborhood_non_truncated() -> None:
    """neighborhood with generous budget returns all edges, truncated=False."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _neighborhood_raw())
    result = idx.neighborhood("b.middle", depth=2, token_budget=10000)
    assert result["symbol"] == "b.middle"
    assert result["truncated"] is False
    assert "depth_truncated" not in result
    # Should have edges in both directions
    edges = [tuple(e) for e in result["edges"]]
    assert ("b.middle", "c.leaf") in edges  # callee direction
    assert ("a.top", "b.middle") in edges   # caller direction
    assert ("d.caller", "b.middle") in edges  # second caller
    assert isinstance(result["token_budget_used"], int)
    assert result["token_budget_used"] >= 0


def test_neighborhood_truncated() -> None:
    """neighborhood with tiny budget truncates and sets depth_truncated."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _neighborhood_raw())
    # Budget of 1 token = 4 chars, which cannot fit any edge
    result = idx.neighborhood("b.middle", depth=2, token_budget=1)
    assert result["truncated"] is True
    assert "depth_truncated" in result
    # Must satisfy budget constraint
    result_json = json.dumps(result["edges"])
    # The edges list itself should be small
    assert len(result["edges"]) == 0 or len(result_json) <= 1 * 4 + 50  # loose check


def test_neighborhood_token_budget_enforced() -> None:
    """neighborhood response content fits within token_budget * 4 chars."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _neighborhood_raw())
    token_budget = 5  # very tight
    result = idx.neighborhood("b.middle", depth=2, token_budget=token_budget)
    # The edges JSON representation must fit strictly within the budget.
    # +2 accounts for the outer '[]' brackets in JSON serialisation of the list.
    edges_json = json.dumps(result["edges"])
    assert len(edges_json) <= token_budget * 4 + 2  # +2 for '[]' list brackets


def test_neighborhood_unknown_symbol() -> None:
    """neighborhood on a symbol not in the graph returns empty edges, truncated=False."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _neighborhood_raw())
    result = idx.neighborhood("nonexistent.symbol", depth=2, token_budget=1000)
    assert result["symbol"] == "nonexistent.symbol"
    assert result["edges"] == []
    assert result["truncated"] is False


def test_neighborhood_depth_full_reflects_graph() -> None:
    """depth_full reflects actual graph depth, not declared depth parameter."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _neighborhood_raw())
    # Graph has max depth 2 from b.middle; requesting depth=5 should give depth_full <= 2
    result = idx.neighborhood("b.middle", depth=5, token_budget=10000)
    assert result["depth_full"] <= 2


def test_neighborhood_edges_deduplicated() -> None:
    """Each edge appears at most once in the result."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _neighborhood_raw())
    result = idx.neighborhood("b.middle", depth=3, token_budget=10000)
    edge_tuples = [tuple(e) for e in result["edges"]]
    assert len(edge_tuples) == len(set(edge_tuples)), "Edges must be deduplicated"


def test_neighborhood_deterministic() -> None:
    """Same input always produces the same output (Law 3)."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _neighborhood_raw())
    r1 = idx.neighborhood("b.middle", depth=2, token_budget=10000)
    r2 = idx.neighborhood("b.middle", depth=2, token_budget=10000)
    assert r1["edges"] == r2["edges"]
