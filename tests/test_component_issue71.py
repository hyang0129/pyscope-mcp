"""Component tests for issue #71 — refers_to typed symbol lookup.

Three cross-module boundaries tested:

  B1 (graph.py → server.py): refers_to with kind='all' uses the new
     _bfs_nodes_called_by path and context-priority logic.  The server's
     _dispatch_tool must pass kind/granularity/depth arguments through
     correctly and must NOT wrap a successful multi-kind result in
     isError:True.  The result dict content text must deserialize to a
     payload with a 'results' list of function-granularity entries, each
     containing 'fqn', 'context', and 'depth' fields.

  B2 (graph.py → server.py): refers_to with granularity='module' returns
     a flat list of module FQN strings, not a list of dicts.  The server's
     _text() serialization path must not trigger the isError branch.  The
     content text must deserialize to a payload where 'results' is a list
     of strings (module FQNs), not a list of dicts.

  B3 (graph.py → server.py): refers_to with depth=2 traverses hop-2 BFS
     in _bfs_nodes_called_by.  Results must include both depth-1 and
     depth-2 referrers, and every entry in the function-granularity result
     must carry a 'depth' field that correctly distinguishes the two hops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pyscope_mcp.graph import CallGraphIndex
from conftest import make_nodes
import pyscope_mcp.server as _srv


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _git_fail() -> MagicMock:
    """Fake subprocess.run result: git unavailable."""
    m = MagicMock()
    m.returncode = 128
    m.stdout = ""
    return m


def _nodes_with_kind(fqn_from: str, fqn_to: str, kind: str) -> dict[str, dict]:
    """Build a minimal nodes dict with one directional edge of the given kind."""
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


def _make_multi_kind_nodes() -> dict[str, dict]:
    """Graph with import and annotation (non-call) edges to target.

    pkg.target.Symbol  ← import     — pkg.a.importer
    pkg.target.Symbol  ← annotation — pkg.b.annotator
    """
    target = "pkg.target.Symbol"
    return _merge_nodes(
        _nodes_with_kind("pkg.a.importer", target, "import"),
        _nodes_with_kind("pkg.b.annotator", target, "annotation"),
    )


# ---------------------------------------------------------------------------
# B1 — graph.py → server.py: kind='all' multi-kind edge propagation
# ---------------------------------------------------------------------------

class TestB1KindAllPath:
    """B1: server dispatches refers_to(kind='all') — _bfs_nodes_called_by path."""

    def test_graph_refers_to_kind_all_finds_non_call_referrer(
        self, tmp_path: Path
    ) -> None:
        """[B1-graph] refers_to(kind='all') must find an import-edge referrer.

        Verifies that _bfs_nodes_called_by is reached and returns the
        import-edge referrer when kind='all' is specified.
        """
        # Arrange
        nodes = _make_multi_kind_nodes()
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.refers_to("pkg.target.Symbol", kind="all", depth=1)

        # Assert
        assert result.get("isError") is not True, (
            f"refers_to(kind='all') must not set isError for a present FQN; got: {result}"
        )
        fqns = [e["fqn"] for e in result["results"]]
        assert "pkg.a.importer" in fqns, (
            f"import-edge referrer must appear in kind='all' results; got fqns={fqns}"
        )
        assert "pkg.b.annotator" in fqns, (
            f"annotation-edge referrer must appear in kind='all' results; got fqns={fqns}"
        )

    def test_graph_refers_to_kind_all_context_field_reflects_edge_kind(
        self, tmp_path: Path
    ) -> None:
        """[B1-graph] context field must reflect the actual edge kind in kind='all' mode.

        An import-edge referrer must have context='import'; an annotation-edge
        referrer must have context='annotation'.
        """
        # Arrange
        nodes = _make_multi_kind_nodes()
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.refers_to("pkg.target.Symbol", kind="all", depth=1)

        # Assert
        assert result.get("isError") is not True
        by_fqn = {e["fqn"]: e for e in result["results"]}

        importer_entry = by_fqn.get("pkg.a.importer")
        assert importer_entry is not None, "import-edge referrer must be in results"
        assert importer_entry["context"] == "import", (
            f"import-edge referrer must have context='import', "
            f"got context={importer_entry['context']!r}"
        )

        annotator_entry = by_fqn.get("pkg.b.annotator")
        assert annotator_entry is not None, "annotation-edge referrer must be in results"
        assert annotator_entry["context"] == "annotation", (
            f"annotation-edge referrer must have context='annotation', "
            f"got context={annotator_entry['context']!r}"
        )

    def test_graph_refers_to_kind_all_call_wins_context_priority(
        self, tmp_path: Path
    ) -> None:
        """[B1-graph] When a function has both import and call edges, context must be 'call'.

        Context priority: call > import > except > annotation > isinstance.
        """
        # Arrange — one referrer with both call and import edges to target
        target = "pkg.target.Symbol"
        referrer = "pkg.dual.consumer"
        nodes = _merge_nodes(
            _nodes_with_kind(referrer, target, "call"),
            _nodes_with_kind(referrer, target, "import"),
        )
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.refers_to(target, kind="all", depth=1)

        # Assert
        assert result.get("isError") is not True
        entries = [e for e in result["results"] if e["fqn"] == referrer]
        assert len(entries) == 1, (
            f"referrer must appear exactly once; got {len(entries)} entries"
        )
        assert entries[0]["context"] == "call", (
            f"'call' must win context priority over 'import'; "
            f"got context={entries[0]['context']!r}"
        )

    @pytest.mark.asyncio
    async def test_server_dispatch_refers_to_kind_all_is_not_error(
        self, tmp_path: Path
    ) -> None:
        """[B1-server] _dispatch_tool('refers_to', kind='all') must return isError:False.

        Verifies the wiring: server.py passes kind='all' to graph.py, receives
        a non-error result, and wraps it with _text() rather than the isError
        branch.
        """
        # Arrange
        nodes = _make_multi_kind_nodes()
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)
        _srv._INDEX = idx
        _srv._INDEX_PATH = tmp_path / "index.json"

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            response = await _srv._dispatch_tool(
                "refers_to",
                {"fqn": "pkg.target.Symbol", "kind": "all", "granularity": "function"},
            )

        # Assert
        assert response.get("isError") is False, (
            "server _dispatch_tool must return isError:False for a successful "
            f"kind='all' lookup; got response={response}"
        )
        content_text = response["content"][0]["text"]
        payload = json.loads(content_text)
        assert "results" in payload, (
            f"response content must contain 'results' key; keys={list(payload)}"
        )
        fqns = [e["fqn"] for e in payload["results"]]
        assert "pkg.a.importer" in fqns, (
            f"import-edge referrer must appear in server response; fqns={fqns}"
        )

    @pytest.mark.asyncio
    async def test_server_dispatch_refers_to_kind_all_function_entry_shape(
        self, tmp_path: Path
    ) -> None:
        """[B1-server] Function-granularity entries in kind='all' response must have fqn/context/depth.

        Verifies that the context field (reflecting edge kind) survives
        serialisation through _text() and JSON deserialization.
        """
        # Arrange — single referrer via import edge
        target = "pkg.target.Symbol"
        nodes = _nodes_with_kind("pkg.a.importer", target, "import")
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)
        _srv._INDEX = idx
        _srv._INDEX_PATH = tmp_path / "index.json"

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            response = await _srv._dispatch_tool(
                "refers_to",
                {"fqn": target, "kind": "all", "granularity": "function"},
            )

        # Assert
        assert response.get("isError") is False
        payload = json.loads(response["content"][0]["text"])
        results = payload["results"]
        assert len(results) == 1, f"expected 1 result, got {len(results)}"
        entry = results[0]
        for field in ("fqn", "context", "depth"):
            assert field in entry, (
                f"function-granularity entry must have '{field}' field; keys={list(entry)}"
            )
        assert entry["fqn"] == "pkg.a.importer"
        assert entry["context"] == "import", (
            f"context must be 'import' for an import-edge referrer; got {entry['context']!r}"
        )
        assert entry["depth"] == 1


# ---------------------------------------------------------------------------
# B2 — graph.py → server.py: granularity='module' result shape
# ---------------------------------------------------------------------------

class TestB2ModuleGranularity:
    """B2: server dispatches refers_to(granularity='module') — flat string list path."""

    def test_graph_refers_to_module_granularity_returns_string_list(
        self, tmp_path: Path
    ) -> None:
        """[B2-graph] refers_to(granularity='module') must return a flat list of module FQN strings."""
        # Arrange — two referrers from different modules
        target = "pkg.target.fn"
        nodes = _merge_nodes(
            _nodes_with_kind("pkg.mod_a.caller", target, "call"),
            _nodes_with_kind("pkg.mod_b.importer", target, "import"),
        )
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.refers_to(target, kind="all", granularity="module", depth=1)

        # Assert
        assert result.get("isError") is not True
        assert "results" in result
        for item in result["results"]:
            assert isinstance(item, str), (
                f"module-granularity results must be strings, got {type(item)}: {item!r}"
            )
        assert "pkg.mod_a" in result["results"], (
            f"pkg.mod_a must appear; results={result['results']}"
        )
        assert "pkg.mod_b" in result["results"], (
            f"pkg.mod_b must appear; results={result['results']}"
        )

    def test_graph_refers_to_module_granularity_deduplicates(
        self, tmp_path: Path
    ) -> None:
        """[B2-graph] Two referrers from the same module must produce that module only once."""
        # Arrange — both callers are in pkg.shared_mod
        target = "pkg.target.fn"
        nodes = _merge_nodes(
            _nodes_with_kind("pkg.shared_mod.func_a", target, "call"),
            _nodes_with_kind("pkg.shared_mod.func_b", target, "import"),
        )
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.refers_to(target, kind="all", granularity="module", depth=1)

        # Assert
        assert result.get("isError") is not True
        count = result["results"].count("pkg.shared_mod")
        assert count == 1, (
            f"pkg.shared_mod must appear exactly once; results={result['results']}"
        )

    @pytest.mark.asyncio
    async def test_server_dispatch_refers_to_module_granularity_is_not_error(
        self, tmp_path: Path
    ) -> None:
        """[B2-server] _dispatch_tool('refers_to', granularity='module') must return isError:False.

        The server _text() serialization path must accept a flat list of strings
        without triggering the isError branch.
        """
        # Arrange
        target = "pkg.target.fn"
        nodes = _merge_nodes(
            _nodes_with_kind("pkg.mod_a.caller", target, "call"),
            _nodes_with_kind("pkg.mod_b.importer", target, "import"),
        )
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)
        _srv._INDEX = idx
        _srv._INDEX_PATH = tmp_path / "index.json"

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            response = await _srv._dispatch_tool(
                "refers_to",
                {"fqn": target, "kind": "all", "granularity": "module"},
            )

        # Assert
        assert response.get("isError") is False, (
            "server _dispatch_tool must return isError:False for module-granularity "
            f"refers_to; got response={response}"
        )
        content_text = response["content"][0]["text"]
        payload = json.loads(content_text)
        assert "results" in payload
        for item in payload["results"]:
            assert isinstance(item, str), (
                f"module-granularity results in server response must be strings; "
                f"got {type(item)}: {item!r}"
            )

    @pytest.mark.asyncio
    async def test_server_dispatch_refers_to_module_granularity_modules_present(
        self, tmp_path: Path
    ) -> None:
        """[B2-server] Module FQNs must survive serialization through the server layer."""
        # Arrange
        target = "pkg.target.fn"
        nodes = _merge_nodes(
            _nodes_with_kind("pkg.mod_a.caller", target, "call"),
            _nodes_with_kind("pkg.mod_b.importer", target, "import"),
        )
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)
        _srv._INDEX = idx
        _srv._INDEX_PATH = tmp_path / "index.json"

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            response = await _srv._dispatch_tool(
                "refers_to",
                {"fqn": target, "kind": "all", "granularity": "module"},
            )

        # Assert
        payload = json.loads(response["content"][0]["text"])
        assert "pkg.mod_a" in payload["results"], (
            f"pkg.mod_a must survive server serialization; results={payload['results']}"
        )
        assert "pkg.mod_b" in payload["results"], (
            f"pkg.mod_b must survive server serialization; results={payload['results']}"
        )


# ---------------------------------------------------------------------------
# B3 — graph.py → server.py: depth=2 multi-hop BFS propagation
# ---------------------------------------------------------------------------

class TestB3Depth2Propagation:
    """B3: server dispatches refers_to(depth=2) — hop-2 BFS in _bfs_nodes_called_by."""

    def test_graph_refers_to_depth2_includes_hop2_referrer(
        self, tmp_path: Path
    ) -> None:
        """[B3-graph] refers_to(depth=2) must include depth-2 referrers via call edges.

        Hop-2 traversal: start ← (import) hop1_ref ← (call) hop2_caller.
        The hop-2 caller reaches start through hop1_ref via a call edge.
        """
        # Arrange
        target = "pkg.target.fn"
        hop1 = "pkg.hop1.importer"   # imports target
        hop2 = "pkg.hop2.caller"     # calls hop1

        nodes = _merge_nodes(
            _nodes_with_kind(hop1, target, "import"),  # hop1 → target (import)
            _nodes_with_kind(hop2, hop1, "call"),      # hop2 → hop1 (call)
        )
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.refers_to(target, kind="all", depth=2)

        # Assert
        assert result.get("isError") is not True, (
            f"refers_to(depth=2) must not be an error; got: {result}"
        )
        fqns = [e["fqn"] for e in result["results"]]
        assert hop1 in fqns, f"hop-1 referrer {hop1!r} must be in depth=2 results; fqns={fqns}"
        assert hop2 in fqns, f"hop-2 caller {hop2!r} must be in depth=2 results; fqns={fqns}"

    def test_graph_refers_to_depth2_depth_field_distinguishes_hops(
        self, tmp_path: Path
    ) -> None:
        """[B3-graph] depth field in results must correctly distinguish hop-1 vs hop-2."""
        # Arrange
        target = "pkg.target.fn"
        hop1 = "pkg.hop1.importer"
        hop2 = "pkg.hop2.caller"

        nodes = _merge_nodes(
            _nodes_with_kind(hop1, target, "import"),
            _nodes_with_kind(hop2, hop1, "call"),
        )
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.refers_to(target, kind="all", depth=2)

        # Assert
        assert result.get("isError") is not True
        by_fqn = {e["fqn"]: e for e in result["results"]}

        hop1_entry = by_fqn.get(hop1)
        assert hop1_entry is not None, f"{hop1!r} must be in results"
        assert hop1_entry["depth"] == 1, (
            f"hop-1 referrer must have depth=1; got depth={hop1_entry['depth']}"
        )

        hop2_entry = by_fqn.get(hop2)
        assert hop2_entry is not None, f"{hop2!r} must be in results"
        assert hop2_entry["depth"] == 2, (
            f"hop-2 caller must have depth=2; got depth={hop2_entry['depth']}"
        )

    def test_graph_refers_to_depth1_excludes_hop2(
        self, tmp_path: Path
    ) -> None:
        """[B3-graph] depth=1 must NOT include hop-2 callers.

        Negative control for the depth parameter: the same graph with depth=1
        must exclude the hop-2 caller that depth=2 includes.
        """
        # Arrange
        target = "pkg.target.fn"
        hop1 = "pkg.hop1.importer"
        hop2 = "pkg.hop2.caller"

        nodes = _merge_nodes(
            _nodes_with_kind(hop1, target, "import"),
            _nodes_with_kind(hop2, hop1, "call"),
        )
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.refers_to(target, kind="all", depth=1)

        # Assert
        assert result.get("isError") is not True
        fqns = [e["fqn"] for e in result["results"]]
        assert hop1 in fqns, "hop-1 referrer must still appear with depth=1"
        assert hop2 not in fqns, (
            f"hop-2 caller must NOT appear with depth=1; got fqns={fqns}"
        )

    @pytest.mark.asyncio
    async def test_server_dispatch_refers_to_depth2_is_not_error(
        self, tmp_path: Path
    ) -> None:
        """[B3-server] _dispatch_tool('refers_to', depth=2) must return isError:False.

        Verifies that the server correctly passes depth=2 to graph.py and
        serializes the multi-hop result without triggering the isError branch.
        """
        # Arrange
        target = "pkg.target.fn"
        hop1 = "pkg.hop1.importer"
        hop2 = "pkg.hop2.caller"
        nodes = _merge_nodes(
            _nodes_with_kind(hop1, target, "import"),
            _nodes_with_kind(hop2, hop1, "call"),
        )
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)
        _srv._INDEX = idx
        _srv._INDEX_PATH = tmp_path / "index.json"

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            response = await _srv._dispatch_tool(
                "refers_to",
                {"fqn": target, "kind": "all", "depth": 2},
            )

        # Assert
        assert response.get("isError") is False, (
            f"server must return isError:False for depth=2 refers_to; got: {response}"
        )
        payload = json.loads(response["content"][0]["text"])
        assert "results" in payload
        fqns = [e["fqn"] for e in payload["results"]]
        assert hop1 in fqns, f"hop-1 referrer must be in server response; fqns={fqns}"
        assert hop2 in fqns, f"hop-2 caller must be in server response; fqns={fqns}"

    @pytest.mark.asyncio
    async def test_server_dispatch_refers_to_depth2_depth_field_survives_serialization(
        self, tmp_path: Path
    ) -> None:
        """[B3-server] depth field per entry must survive _text() serialization.

        The depth field distinguishes hop-1 from hop-2 referrers.  It must be
        present in the deserialized server response.
        """
        # Arrange
        target = "pkg.target.fn"
        hop1 = "pkg.hop1.importer"
        hop2 = "pkg.hop2.caller"
        nodes = _merge_nodes(
            _nodes_with_kind(hop1, target, "import"),
            _nodes_with_kind(hop2, hop1, "call"),
        )
        idx = CallGraphIndex.from_nodes(str(tmp_path), nodes)
        _srv._INDEX = idx
        _srv._INDEX_PATH = tmp_path / "index.json"

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            response = await _srv._dispatch_tool(
                "refers_to",
                {"fqn": target, "kind": "all", "depth": 2},
            )

        # Assert
        assert response.get("isError") is False
        payload = json.loads(response["content"][0]["text"])
        by_fqn = {e["fqn"]: e for e in payload["results"]}

        hop1_entry = by_fqn.get(hop1)
        assert hop1_entry is not None
        assert "depth" in hop1_entry, "depth field must survive serialization"
        assert hop1_entry["depth"] == 1, (
            f"hop-1 referrer depth must be 1; got {hop1_entry['depth']}"
        )

        hop2_entry = by_fqn.get(hop2)
        assert hop2_entry is not None
        assert "depth" in hop2_entry, "depth field must survive serialization"
        assert hop2_entry["depth"] == 2, (
            f"hop-2 caller depth must be 2; got {hop2_entry['depth']}"
        )
