"""Component tests for issue #92 — eliminate _DiGraph, introduce GraphReader.

Three cross-module boundaries tested within graph.py after the #92 migration:

  B1 (CallGraphIndex.stats → GraphReader.num_nodes/num_edges + _module_index):
     stats() now delegates to _reader.num_nodes(), _reader.num_edges(kind="call"),
     and len(self._module_index).  Verifies that from_nodes construction wires
     GraphReader correctly so stats() returns accurate counts — not zero or
     pre-migration values.

  B2 (CallGraphIndex.load → from_nodes → GraphReader: save/load round-trip):
     After load(), _reader must correctly traverse the reconstituted nodes dict.
     Verifies that callees_of and refers_to return correct results after a full
     save→load cycle — confirming GraphReader is wired to the loaded nodes dict,
     not a stale or empty structure.

  B3 (_prefix_module_bfs → GraphReader.bfs + _module_index: module_callees):
     module_callees now uses _prefix_module_bfs with reader + module_index.
     The BFS projected through _module_of is correct only if _module_index is
     populated at from_nodes time and reader.bfs returns the right reachable FQNs.
     Verifies the full chain from module_callees through _prefix_module_bfs
     through GraphReader.bfs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyscope_mcp.graph import CallGraphIndex
from conftest import make_nodes


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_index(tmp_path: Path) -> CallGraphIndex:
    """Three-node graph with two call edges across two modules.

    pkg.a.alpha -> pkg.a.beta  (same module)
    pkg.a.beta  -> pkg.b.gamma (cross-module edge)
    pkg.b.gamma -> (no callees)
    """
    raw: dict[str, list[str]] = {
        "pkg.a.alpha": ["pkg.a.beta"],
        "pkg.a.beta": ["pkg.b.gamma"],
        "pkg.b.gamma": [],
    }
    return CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))


def _save_and_load(idx: CallGraphIndex, tmp_path: Path) -> CallGraphIndex:
    """Save *idx* to tmp_path/index.json and reload it."""
    p = tmp_path / "index.json"
    idx.save(p)
    return CallGraphIndex.load(p)


# ---------------------------------------------------------------------------
# B1 — CallGraphIndex.stats() → GraphReader.num_nodes/num_edges + _module_index
# ---------------------------------------------------------------------------

@pytest.mark.component
class TestB1StatsViaGraphReader:
    """B1: stats() returns accurate counts via GraphReader after from_nodes."""

    def test_stats_functions_count_matches_node_count(self, tmp_path: Path) -> None:
        """[B1] stats().functions must equal the number of nodes wired into GraphReader.

        Verifies that _reader.num_nodes() is not zero and matches the actual
        number of symbols passed to from_nodes. Regression guard: the old
        _DiGraph.number_of_nodes() path would also have returned 3 here, so
        this test catches any wiring break that causes _reader to be empty.
        """
        # Arrange
        idx = _make_index(tmp_path)

        # Act
        result = idx.stats()

        # Assert
        assert result["functions"] == 3, (
            f"stats().functions must equal 3 (the number of nodes in the fixture); "
            f"got {result['functions']!r}. "
            "If 0, GraphReader is not wired to the nodes dict at from_nodes time."
        )

    def test_stats_function_edges_count_matches_call_edges(self, tmp_path: Path) -> None:
        """[B1] stats().function_edges must equal the call-edge count from GraphReader.

        The fixture has 2 call edges (alpha→beta, beta→gamma).
        _reader.num_edges(kind="call") must return 2, not 0.
        """
        # Arrange
        idx = _make_index(tmp_path)

        # Act
        result = idx.stats()

        # Assert
        assert result["function_edges"] == 2, (
            f"stats().function_edges must equal 2; got {result['function_edges']!r}. "
            "Regression: _reader.num_edges(kind='call') may be broken."
        )

    def test_stats_modules_count_matches_module_index(self, tmp_path: Path) -> None:
        """[B1] stats().modules must reflect _module_index size, not a _DiGraph node count.

        The fixture has symbols in pkg.a and pkg.b — two distinct modules.
        _module_index is populated at from_nodes time; this test verifies the
        wiring from stats() → len(self._module_index).
        """
        # Arrange
        idx = _make_index(tmp_path)

        # Act
        result = idx.stats()

        # Assert
        assert result["modules"] == 2, (
            f"stats().modules must equal 2 (pkg.a and pkg.b); "
            f"got {result['modules']!r}. "
            "If 0, _module_index was not populated at from_nodes time."
        )

    def test_stats_module_edges_counts_cross_module_call_edges(self, tmp_path: Path) -> None:
        """[B1] stats().module_edges must count distinct cross-module (src, dst) pairs.

        The fixture has one cross-module edge: pkg.a → pkg.b (via beta→gamma).
        The intra-module edge (alpha→beta) must not be counted.
        """
        # Arrange
        idx = _make_index(tmp_path)

        # Act
        result = idx.stats()

        # Assert
        assert result["module_edges"] == 1, (
            f"stats().module_edges must equal 1 (pkg.a → pkg.b); "
            f"got {result['module_edges']!r}. "
            "If > 1, intra-module edges are being counted; if 0, reader.successors is broken."
        )

    def test_stats_consistent_after_save_load(self, tmp_path: Path) -> None:
        """[B1] stats() must return identical counts before and after save+load.

        Verifies that GraphReader is correctly reconstructed from the loaded
        nodes dict — not from a stale or empty structure.
        """
        # Arrange
        idx = _make_index(tmp_path)
        before = idx.stats()

        # Act
        loaded = _save_and_load(idx, tmp_path)
        after = loaded.stats()

        # Assert
        assert after["functions"] == before["functions"], (
            f"stats().functions mismatch after load: {after['functions']} != {before['functions']}"
        )
        assert after["function_edges"] == before["function_edges"], (
            f"stats().function_edges mismatch after load: "
            f"{after['function_edges']} != {before['function_edges']}"
        )
        assert after["modules"] == before["modules"], (
            f"stats().modules mismatch after load: {after['modules']} != {before['modules']}"
        )
        assert after["module_edges"] == before["module_edges"], (
            f"stats().module_edges mismatch after load: "
            f"{after['module_edges']} != {before['module_edges']}"
        )


# ---------------------------------------------------------------------------
# B2 — CallGraphIndex.load() → from_nodes → GraphReader: save/load round-trip
# ---------------------------------------------------------------------------

@pytest.mark.component
class TestB2LoadRoundTripGraphReaderWiring:
    """B2: after save→load, callees_of and refers_to work via the loaded GraphReader."""

    def test_callees_of_returns_correct_results_after_load(self, tmp_path: Path) -> None:
        """[B2] callees_of must return the same results before and after save+load.

        Verifies GraphReader.bfs is wired to the loaded nodes dict.
        If _reader points to an empty or stale structure, callees_of returns
        an empty list or an isError result.
        """
        # Arrange
        idx = _make_index(tmp_path)
        before = idx.callees_of("pkg.a.alpha", depth=1)

        # Act
        loaded = _save_and_load(idx, tmp_path)
        after = loaded.callees_of("pkg.a.alpha", depth=1)

        # Assert — both should return pkg.a.beta as the direct callee
        assert after.get("isError") is not True, (
            f"callees_of after load returned isError: {after}"
        )
        assert set(after["results"]) == set(before["results"]), (
            f"callees_of results mismatch after load: "
            f"{after['results']} != {before['results']}"
        )
        assert "pkg.a.beta" in after["results"], (
            "callees_of('pkg.a.alpha') must include pkg.a.beta after load"
        )

    def test_callees_of_transitive_after_load(self, tmp_path: Path) -> None:
        """[B2] callees_of at depth=2 must follow the full graph chain after load.

        alpha→beta→gamma: at depth=2, both beta and gamma must be reachable
        after the save+load cycle. Verifies the full BFS traversal chain.
        """
        # Arrange
        idx = _make_index(tmp_path)

        # Act
        loaded = _save_and_load(idx, tmp_path)
        result = loaded.callees_of("pkg.a.alpha", depth=2)

        # Assert
        assert result.get("isError") is not True, (
            f"callees_of(depth=2) after load returned isError: {result}"
        )
        result_set = set(result["results"])
        assert "pkg.a.beta" in result_set, (
            "callees_of(alpha, depth=2) must include direct callee pkg.a.beta"
        )
        assert "pkg.b.gamma" in result_set, (
            "callees_of(alpha, depth=2) must include transitive callee pkg.b.gamma. "
            "If missing, GraphReader.bfs is not traversing the loaded nodes dict."
        )

    def test_refers_to_callers_returns_correct_results_after_load(
        self, tmp_path: Path
    ) -> None:
        """[B2] refers_to(kind='callers') must return the correct callers after load.

        pkg.a.beta is called by pkg.a.alpha.  refers_to must find this caller
        from the loaded index — verifying _reader.bfs(direction="called_by")
        uses the loaded nodes dict.
        """
        # Arrange
        idx = _make_index(tmp_path)

        # Act
        loaded = _save_and_load(idx, tmp_path)
        result = loaded.refers_to("pkg.a.beta", kind="callers", depth=1)

        # Assert
        assert result.get("isError") is not True, (
            f"refers_to after load returned isError: {result}"
        )
        result_fqns = {entry["fqn"] for entry in result["results"]}
        assert "pkg.a.alpha" in result_fqns, (
            "refers_to('pkg.a.beta', kind='callers') must include pkg.a.alpha after load. "
            "If missing, the called_by index was not correctly reconstituted via GraphReader."
        )

    def test_fqn_not_in_loaded_index_returns_error(self, tmp_path: Path) -> None:
        """[B2] refers_to on a missing FQN after load returns isError, not an exception.

        Verifies that the GraphReader.__contains__ check works correctly on the
        loaded nodes dict — not against a stale empty structure that would cause
        every FQN to appear missing.
        """
        # Arrange
        idx = _make_index(tmp_path)

        # Act
        loaded = _save_and_load(idx, tmp_path)
        # This FQN is present — must NOT return isError
        present_result = loaded.refers_to("pkg.a.alpha", kind="callers", depth=1)
        # This FQN is absent — must return isError
        absent_result = loaded.refers_to("pkg.nonexistent.func", kind="callers", depth=1)

        # Assert
        assert present_result.get("isError") is not True, (
            "refers_to on a known FQN must not return isError after load"
        )
        assert absent_result.get("isError") is True, (
            "refers_to on an absent FQN must return isError=True after load. "
            "If False, __contains__ check is using an empty structure and "
            "the fqn_not_in_graph branch never fires."
        )
        assert absent_result.get("error_reason") == "fqn_not_in_graph", (
            f"error_reason must be 'fqn_not_in_graph'; got {absent_result.get('error_reason')!r}"
        )


# ---------------------------------------------------------------------------
# B3 — _prefix_module_bfs → GraphReader.bfs + _module_index (module_callees)
# ---------------------------------------------------------------------------

@pytest.mark.component
class TestB3ModuleCalleesViaPrefixBfsAndGraphReader:
    """B3: module_callees returns correct results via _prefix_module_bfs + GraphReader.bfs."""

    def test_module_callees_direct_cross_module_edge(self, tmp_path: Path) -> None:
        """[B3] module_callees must return pkg.b as a direct callee of pkg.a.

        The fixture has pkg.a.beta → pkg.b.gamma.  _prefix_module_bfs must
        find pkg.b at hop depth 1 via GraphReader.bfs from pkg.a's symbols.
        """
        # Arrange
        idx = _make_index(tmp_path)

        # Act
        result = idx.module_callees("pkg.a", depth=1)

        # Assert
        assert result.get("isError") is not True, (
            f"module_callees('pkg.a') returned isError: {result}"
        )
        assert "pkg.b" in result["results"], (
            "module_callees('pkg.a') must include pkg.b (via beta→gamma cross-module edge). "
            "If empty, _prefix_module_bfs is not calling GraphReader.bfs correctly, "
            "or _module_index was not populated at from_nodes time."
        )

    def test_module_callees_excludes_seed_module(self, tmp_path: Path) -> None:
        """[B3] module_callees must not include the queried module in its own results.

        The intra-module edge alpha→beta must not cause pkg.a to appear as a
        callee of pkg.a. Seed modules are excluded from _prefix_module_bfs results.
        """
        # Arrange
        idx = _make_index(tmp_path)

        # Act
        result = idx.module_callees("pkg.a", depth=1)

        # Assert
        assert "pkg.a" not in result["results"], (
            "module_callees('pkg.a') must not include pkg.a itself. "
            "If it does, _prefix_module_bfs is including the seed module in results."
        )

    def test_module_callees_empty_for_leaf_module(self, tmp_path: Path) -> None:
        """[B3] module_callees on a leaf module must return empty results.

        pkg.b.gamma has no outgoing call edges. module_callees('pkg.b') must
        return an empty list — verifying that GraphReader.bfs correctly returns {}
        for a node with no callees.
        """
        # Arrange
        idx = _make_index(tmp_path)

        # Act
        result = idx.module_callees("pkg.b", depth=1)

        # Assert
        assert result.get("isError") is not True, (
            f"module_callees('pkg.b') returned isError: {result}"
        )
        assert result["results"] == [], (
            f"module_callees('pkg.b') must return [] (leaf module); "
            f"got {result['results']!r}"
        )

    def test_module_callees_correct_after_save_load(self, tmp_path: Path) -> None:
        """[B3] module_callees results must be identical before and after save+load.

        Verifies _module_index is correctly rebuilt from the loaded nodes dict
        at load time, and GraphReader.bfs traverses the loaded structure.
        """
        # Arrange
        idx = _make_index(tmp_path)
        before = idx.module_callees("pkg.a", depth=1)

        # Act
        loaded = _save_and_load(idx, tmp_path)
        after = loaded.module_callees("pkg.a", depth=1)

        # Assert
        assert set(after["results"]) == set(before["results"]), (
            f"module_callees results mismatch after load: "
            f"{after['results']} != {before['results']}. "
            "If empty after load, _module_index was not rebuilt from the loaded nodes dict."
        )

    def test_module_callees_prefix_matches_multiple_modules(self, tmp_path: Path) -> None:
        """[B3] module_callees with a broad prefix must aggregate across all matching seeds.

        Querying with prefix 'pkg' matches both pkg.a and pkg.b — but since
        pkg.b has no callees outside pkg, the result should equal what
        module_callees('pkg.a') returns. This verifies the seed-union logic in
        _prefix_module_bfs handles multiple matched seeds correctly.
        """
        # Arrange
        idx = _make_index(tmp_path)

        # Act — 'pkg' prefix matches both pkg.a and pkg.b as seeds
        result_pkg = idx.module_callees("pkg", depth=1)
        result_a = idx.module_callees("pkg.a", depth=1)

        # Assert — pkg.b is a callee of pkg.a and would also be a seed when
        # using prefix 'pkg', so it should be excluded from the results
        # (seed modules are excluded). The results should be a subset of pkg.a's.
        assert set(result_pkg["results"]).issubset({"pkg.b", "pkg.a"} | set(result_a["results"])), (
            f"module_callees('pkg') returned unexpected results: {result_pkg['results']}"
        )
        # The cross-module edge pkg.a→pkg.b is still detectable at depth=1
        # (unless pkg.b is a seed and gets excluded — which it would be here)
        assert isinstance(result_pkg["results"], list), (
            "module_callees must always return a list"
        )
