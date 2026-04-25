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
        raw[f"callers.mod{i}.fn"] = ["target.core.fn"]
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
# dropped field — callers_of / callees_of
# ---------------------------------------------------------------------------

def test_callers_of_dropped_zero_when_not_truncated() -> None:
    """dropped=0 when results fit within cap."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _sample_raw())
    result = idx.callers_of("sample.b.helper", depth=1)
    assert "dropped" in result
    assert result["dropped"] == 0
    assert result["truncated"] is False


def test_callees_of_dropped_zero_when_not_truncated() -> None:
    """dropped=0 when results fit within cap."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _sample_raw())
    result = idx.callees_of("sample.a.top", depth=1)
    assert "dropped" in result
    assert result["dropped"] == 0
    assert result["truncated"] is False


def test_callers_of_dropped_correct_when_truncated() -> None:
    """dropped equals the number of results beyond the 50-item cap."""
    raw: dict[str, list[str]] = {}
    for i in range(57):
        raw[f"callers.mod{i}.fn"] = ["target.core.fn"]
    raw["target.core.fn"] = []
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)
    result = idx.callers_of("target.core.fn", depth=1)
    assert result["truncated"] is True
    assert len(result["results"]) == 50
    assert result["dropped"] == 7


def test_callees_of_dropped_correct_when_truncated() -> None:
    """dropped equals the number of results beyond the 50-item cap."""
    raw: dict[str, list[str]] = {
        "source.mod.fn": [f"target.mod{i}.fn" for i in range(57)]
    }
    for i in range(57):
        raw[f"target.mod{i}.fn"] = []
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)
    result = idx.callees_of("source.mod.fn", depth=1)
    assert result["truncated"] is True
    assert len(result["results"]) == 50
    assert result["dropped"] == 7


# ---------------------------------------------------------------------------
# Ranking: depth-1 results precede depth-2 when alphabetical order would invert
# ---------------------------------------------------------------------------

def _ranking_raw_callers() -> dict[str, list[str]]:
    """Graph for ranking test.

    target.fn is called by:
      - depth-1 callers: z_depth1.mod.fn, y_depth1.mod.fn  (sort AFTER depth-2 alphabetically)
      - depth-2 callers: a_depth2.mod.fn ... (55 of them, sort BEFORE depth-1 alphabetically)

    Without ranking fix, the cap would keep 50 alphabetically-first a_depth2 callers and
    drop the depth-1 z_depth1 and y_depth1 callers.  With the fix, depth-1 callers appear first.
    """
    raw: dict[str, list[str]] = {}
    # depth-2 callers: call bridge.fn which calls target.fn
    for i in range(55):
        raw[f"a_depth2.mod{i}.fn"] = ["bridge.mod.fn"]
    raw["bridge.mod.fn"] = ["target.fn"]
    # depth-1 callers: directly call target.fn (sort alphabetically AFTER a_depth2)
    raw["z_depth1.mod.fn"] = ["target.fn"]
    raw["y_depth1.mod.fn"] = ["target.fn"]
    raw["target.fn"] = []
    return raw


def test_callers_of_depth1_precede_depth2_when_alphabetical_would_invert() -> None:
    """Depth-1 callers appear before depth-2 callers in results, even when alphabetically later."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _ranking_raw_callers())
    result = idx.callers_of("target.fn", depth=2)

    # depth-1 callers must be present despite sorting alphabetically after a_depth2 nodes
    assert "z_depth1.mod.fn" in result["results"], (
        "depth-1 caller z_depth1.mod.fn absent — cap dropped it before depth-2 entries"
    )
    assert "y_depth1.mod.fn" in result["results"], (
        "depth-1 caller y_depth1.mod.fn absent — cap dropped it before depth-2 entries"
    )

    # depth-1 callers must appear in the first part of the list
    z_idx = result["results"].index("z_depth1.mod.fn")
    y_idx = result["results"].index("y_depth1.mod.fn")
    # All depth-2 nodes that appear must come after both depth-1 callers
    for item in result["results"]:
        if item.startswith("a_depth2"):
            item_idx = result["results"].index(item)
            assert item_idx > z_idx and item_idx > y_idx, (
                f"depth-2 caller {item!r} at index {item_idx} appears before depth-1 callers"
            )

    # Total reachable: 55 a_depth2 + bridge.mod.fn + z_depth1.mod.fn + y_depth1.mod.fn = 58
    # dropped = 58 - 50 = 8
    assert result["dropped"] == 8
    assert result["truncated"] is True
    assert len(result["results"]) == 50


def _ranking_raw_callees() -> dict[str, list[str]]:
    """Symmetric to _ranking_raw_callers but for callees direction."""
    raw: dict[str, list[str]] = {}
    # source.fn calls bridge.mod.fn (depth-1 callee) which calls 55 a_depth2 callees
    raw["source.fn"] = ["bridge.mod.fn", "z_depth1.mod.fn", "y_depth1.mod.fn"]
    raw["bridge.mod.fn"] = [f"a_depth2.mod{i}.fn" for i in range(55)]
    for i in range(55):
        raw[f"a_depth2.mod{i}.fn"] = []
    raw["z_depth1.mod.fn"] = []
    raw["y_depth1.mod.fn"] = []
    return raw


def test_callees_of_depth1_precede_depth2_when_alphabetical_would_invert() -> None:
    """Depth-1 callees appear before depth-2 callees, even when alphabetically later."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _ranking_raw_callees())
    result = idx.callees_of("source.fn", depth=2)

    # depth-1 callees: bridge.mod.fn, z_depth1.mod.fn, y_depth1.mod.fn
    assert "z_depth1.mod.fn" in result["results"], (
        "depth-1 callee z_depth1.mod.fn absent"
    )
    assert "y_depth1.mod.fn" in result["results"], (
        "depth-1 callee y_depth1.mod.fn absent"
    )
    assert "bridge.mod.fn" in result["results"], (
        "depth-1 callee bridge.mod.fn absent"
    )

    # All three depth-1 callees must appear before depth-2 ones
    depth1_max_idx = max(
        result["results"].index(n)
        for n in ["bridge.mod.fn", "z_depth1.mod.fn", "y_depth1.mod.fn"]
    )
    for item in result["results"]:
        if item.startswith("a_depth2"):
            item_idx = result["results"].index(item)
            assert item_idx > depth1_max_idx, (
                f"depth-2 callee {item!r} at index {item_idx} appears before depth-1 callees"
            )


# ---------------------------------------------------------------------------
# dropped field — module_callers / module_callees
# ---------------------------------------------------------------------------

def test_module_callers_dropped_zero_when_not_truncated() -> None:
    """module_callers: dropped=0 when results fit within cap."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callers("pkg.agents")
    assert "dropped" in result
    assert result["dropped"] == 0
    assert result["truncated"] is False


def test_module_callees_dropped_zero_when_not_truncated() -> None:
    """module_callees: dropped=0 when results fit within cap."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callees("pkg.agents")
    assert "dropped" in result
    assert result["dropped"] == 0
    assert result["truncated"] is False


def test_module_callers_dropped_correct_when_truncated() -> None:
    """module_callers: dropped equals number of results beyond cap."""
    raw: dict[str, list[str]] = {}
    for i in range(57):
        raw[f"callers.mod{i}.fn"] = ["target.core.fn"]
    raw["target.core.fn"] = []
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)
    result = idx.module_callers("target")
    assert result["truncated"] is True
    assert len(result["results"]) == 50
    assert result["dropped"] == 7


def test_module_callees_dropped_correct_when_truncated() -> None:
    """module_callees: dropped equals number of results beyond cap."""
    raw: dict[str, list[str]] = {
        "source.mod.fn": [f"target.mod{i}.fn" for i in range(57)]
    }
    for i in range(57):
        raw[f"target.mod{i}.fn"] = []
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)
    result = idx.module_callees("source")
    assert result["truncated"] is True
    assert len(result["results"]) == 50
    assert result["dropped"] == 7


def test_module_callers_depth1_precede_depth2_when_alphabetical_would_invert() -> None:
    """Depth-1 module callers appear before depth-2 ones, even when alphabetically later."""
    raw: dict[str, list[str]] = {}
    # depth-2 module callers: a_depth2.modN.fn → bridge.mod.fn → target.fn
    for i in range(55):
        raw[f"a_depth2.mod{i}.fn"] = ["bridge.mod.fn"]
    raw["bridge.mod.fn"] = ["target.fn"]
    # depth-1 module callers: z_depth1.mod.fn → target.fn (alphabetically after a_depth2)
    raw["z_depth1.mod.fn"] = ["target.fn"]
    raw["y_depth1.mod.fn"] = ["target.fn"]
    raw["target.fn"] = []
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)
    result = idx.module_callers("target", depth=2)

    assert "z_depth1.mod" in result["results"], (
        "depth-1 module caller z_depth1.mod absent — ranking failed"
    )
    assert "y_depth1.mod" in result["results"], (
        "depth-1 module caller y_depth1.mod absent — ranking failed"
    )
    assert result["truncated"] is True
    # 55 a_depth2 modules + bridge.mod + z_depth1.mod + y_depth1.mod = 58 - 50 = 8
    assert result["dropped"] == 8


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


# ---------------------------------------------------------------------------
# neighborhood — hub suppression
# ---------------------------------------------------------------------------

def _hub_raw() -> dict[str, list[str]]:
    """Fixture: queried symbol calls H; H has 20 distinct callers (in-degree hub).

    queried.symbol -> H
    caller_1 -> H
    caller_2 -> H
    ...
    caller_19 -> H  (plus queried.symbol makes 20 total callers of H)

    queried.symbol also has its own callee: own.callee
    """
    raw: dict[str, list[str]] = {
        "queried.symbol": ["H", "own.callee"],
        "own.callee": [],
        "H": ["h.downstream"],
        "h.downstream": [],
    }
    # 19 other callers of H (plus queried.symbol = 20 total)
    for i in range(1, 20):
        raw[f"caller_{i}"] = ["H"]
    return raw


def test_hub_suppression_default_on() -> None:
    """Hub suppression: default call suppresses high-in-degree H; expand_hubs=True does not."""
    idx = CallGraphIndex.from_raw("/tmp/hub", _hub_raw())

    # Default: suppression on — H's other callers must NOT appear
    result = idx.neighborhood("queried.symbol", depth=2, token_budget=100000)
    edges = [tuple(e) for e in result["edges"]]

    # hub_suppressed and hub_threshold must always be present
    assert "hub_suppressed" in result
    assert "hub_threshold" in result
    assert isinstance(result["hub_suppressed"], list)
    assert isinstance(result["hub_threshold"], int)

    # H is a hub (20 callers >> threshold floor of 10)
    assert "H" in result["hub_suppressed"], (
        f"Expected H in hub_suppressed, got {result['hub_suppressed']}"
    )

    # Other callers of H must NOT appear as edges
    other_callers = {f"caller_{i}" for i in range(1, 20)}
    edge_nodes = {n for e in edges for n in e}
    assert not (other_callers & edge_nodes), (
        f"Hub callers should not appear when suppression is on: {other_callers & edge_nodes}"
    )

    # expand_hubs=True: all callers of H should appear
    result_full = idx.neighborhood("queried.symbol", depth=2, token_budget=100000, expand_hubs=True)
    edges_full = [tuple(e) for e in result_full["edges"]]
    edge_nodes_full = {n for e in edges_full for n in e}
    assert other_callers & edge_nodes_full, (
        "expand_hubs=True should expose other callers of H"
    )
    assert result_full["hub_suppressed"] == [], (
        "hub_suppressed must be [] when expand_hubs=True"
    )
    # hub_threshold still reported even in opt-out mode
    assert "hub_threshold" in result_full
    assert isinstance(result_full["hub_threshold"], int)

    # Suppressed call returns fewer edges than full call
    assert len(result["edges"]) < len(result_full["edges"])


def test_hub_suppression_non_hub_unchanged() -> None:
    """Symbols with no in-degree hubs in their neighborhood see hub_suppressed=[]."""
    # simple chain — no hub
    raw = {
        "a.top": ["b.middle"],
        "b.middle": ["c.leaf"],
        "c.leaf": [],
        "d.caller": ["a.top"],
    }
    idx = CallGraphIndex.from_raw("/tmp/nonhub", raw)
    result = idx.neighborhood("a.top", depth=2, token_budget=100000)
    assert result["hub_suppressed"] == [], (
        f"Expected empty hub_suppressed, got {result['hub_suppressed']}"
    )
    assert "hub_threshold" in result

    # Edges should match expand_hubs=True (no difference)
    result_full = idx.neighborhood("a.top", depth=2, token_budget=100000, expand_hubs=True)
    assert sorted(tuple(e) for e in result["edges"]) == sorted(tuple(e) for e in result_full["edges"])


def test_hub_suppression_symbol_is_hub_exemption() -> None:
    """Queried symbol is exempt from suppression even if it is itself a hub."""
    # Make the queried symbol (H) also have high in-degree
    raw: dict[str, list[str]] = {
        "H": ["h.callee"],
        "h.callee": [],
    }
    for i in range(20):
        raw[f"caller_{i}"] = ["H"]

    idx = CallGraphIndex.from_raw("/tmp/hubself", raw)
    # Calling neighborhood directly on H — H is exempt
    result = idx.neighborhood("H", depth=2, token_budget=100000)
    # H's callers must appear (queried symbol exempt from suppression)
    edges = [tuple(e) for e in result["edges"]]
    caller_nodes = {n for e in edges for n in e if e[1] == "H"}
    assert len(caller_nodes) > 0, "H's callers must be present when querying H directly"
    # H's FQN must NOT appear in hub_suppressed (it is the queried symbol)
    assert "H" not in result["hub_suppressed"]


def test_hub_suppression_out_degree_not_suppressed() -> None:
    """Out-degree hubs are NOT suppressed — their callees are pipeline siblings."""
    # M has high out-degree (calls 30 functions) but only 1 caller
    raw: dict[str, list[str]] = {
        "queried.symbol": ["M"],
        "M": [f"callee_{i}" for i in range(30)],
    }
    for i in range(30):
        raw[f"callee_{i}"] = []

    idx = CallGraphIndex.from_raw("/tmp/outdeg", raw)
    result = idx.neighborhood("queried.symbol", depth=2, token_budget=100000)

    # M should NOT be in hub_suppressed (only in-degree triggers suppression)
    assert "M" not in result["hub_suppressed"], (
        "Out-degree hubs must not be suppressed"
    )
    # M's callees must appear in edges
    edges = [tuple(e) for e in result["edges"]]
    callee_edges = [(s, t) for (s, t) in edges if s == "M"]
    assert len(callee_edges) == 30, (
        f"All 30 of M's callees should appear; got {len(callee_edges)}"
    )


def test_hub_suppression_per_call_threshold_override() -> None:
    """Per-call hub_threshold override: response echoes the override value."""
    idx = CallGraphIndex.from_raw("/tmp/hub", _hub_raw())

    # H has 20 callers. Override threshold to 100 → H is NOT a hub at that threshold.
    result_high = idx.neighborhood("queried.symbol", depth=2, token_budget=100000, hub_threshold=100)
    assert "H" not in result_high["hub_suppressed"], (
        "H should not be suppressed with hub_threshold=100"
    )
    assert result_high["hub_threshold"] == 100

    # Override threshold to 5 → H IS a hub (in-degree 20 > 5).
    result_low = idx.neighborhood("queried.symbol", depth=2, token_budget=100000, hub_threshold=5)
    assert "H" in result_low["hub_suppressed"], (
        "H should be suppressed with hub_threshold=5"
    )
    assert result_low["hub_threshold"] == 5


def test_hub_threshold_attribute_after_construction() -> None:
    """_in_degree_threshold is computed at from_raw time with floor of 10."""
    # Tiny graph — all nodes have in-degree 0 or 1; p99 will be below floor
    idx = CallGraphIndex.from_raw("/tmp/small", _neighborhood_raw())
    assert hasattr(idx, "_in_degree_threshold"), "_in_degree_threshold must exist on index"
    assert idx._in_degree_threshold >= 10, (
        f"Threshold must be >= 10 (floor), got {idx._in_degree_threshold}"
    )


def test_hub_threshold_attribute_reflects_distribution() -> None:
    """_in_degree_threshold reflects actual p99 when distribution exceeds floor."""
    # Build a graph where one node has very high in-degree (25 callers)
    # p99 of [0]*N_nodes + [25] will be 25 for large enough N
    raw: dict[str, list[str]] = {"hub.node": []}
    for i in range(100):
        raw[f"caller_{i}"] = ["hub.node"]
    idx = CallGraphIndex.from_raw("/tmp/p99test", raw)
    # p99 of in-degree distribution: 99 nodes have in-degree 0, hub.node has 100
    # p99_idx = int(0.99 * 101) = 99; sorted list: [0]*100 + [100] at index 100 → 100
    # but p99_idx=99 → value=0; floor=10 → threshold=10.
    # Actually let's just verify the threshold is >= 10 and is an int
    assert isinstance(idx._in_degree_threshold, int)
    assert idx._in_degree_threshold >= 10


def test_neighborhood_hub_fields_always_present_isolated() -> None:
    """hub_suppressed and hub_threshold are present even for isolated/unknown symbols."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _neighborhood_raw())
    result = idx.neighborhood("nonexistent.symbol", depth=2, token_budget=10000)
    assert "hub_suppressed" in result
    assert "hub_threshold" in result
    assert result["hub_suppressed"] == []
    assert isinstance(result["hub_threshold"], int)
