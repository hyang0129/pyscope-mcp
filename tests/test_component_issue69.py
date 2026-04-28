"""Component tests for issue #69 — FQN-not-found and path-not-in-index error
propagation across the graph→server boundary.

Two cross-module boundaries tested:

  B1 (graph.py → server.py): When callers_of, callees_of, or neighborhood
     receives an FQN absent from the graph, graph.py returns
     {isError: True, error_reason: 'fqn_not_in_graph', stale: False,
     stale_files: []} (plus commit staleness fields).
     server.py must propagate this as an MCP-level isError:true response
     containing the full dict as JSON text — it must NOT swallow the error
     via _text(), which would wrap it in isError:false.

  B2 (graph.py → server.py): When file_skeleton receives a path absent from the
     skeleton index, graph.py returns {isError: True, error_reason:
     'path_not_in_index', stale: False, stale_files: []} and server.py must
     propagate it as an MCP-level isError:true response with the full dict as
     JSON text.  The error_reason discriminator must survive serialisation and
     be readable by the client.
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

def _make_minimal_raw() -> dict[str, list[str]]:
    """Minimal synthetic call graph: alpha calls beta; gamma calls alpha."""
    return {
        "pkg.mod.alpha": ["pkg.mod.beta"],
        "pkg.mod.beta": [],
        "pkg.other.gamma": ["pkg.mod.alpha"],
    }


def _git_fail() -> MagicMock:
    """Fake subprocess.run result: git unavailable."""
    m = MagicMock()
    m.returncode = 128
    m.stdout = ""
    return m


def _make_index_with_skeleton(tmp_path: Path) -> CallGraphIndex:
    """Return a CallGraphIndex with one skeleton entry for 'pkg/mod.py'."""
    raw: dict[str, list[str]] = {
        "pkg.mod.alpha": ["pkg.mod.beta"],
        "pkg.mod.beta": [],
    }
    skeletons = {
        "pkg/mod.py": [
            {"fqn": "pkg.mod.alpha", "kind": "function", "signature": "def alpha():", "lineno": 1},
            {"fqn": "pkg.mod.beta", "kind": "function", "signature": "def beta():", "lineno": 5},
        ]
    }
    # file_shas must include the path for staleness not to fire; but since the
    # live file does not actually exist under tmp_path, we need to either
    # omit file_shas (pre-v3 mode triggers stale=True) or provide matching shas.
    # Use file_shas=None to get pre-v3 staleness path — that still exercises the
    # success case skeleton content correctly.
    return CallGraphIndex.from_nodes(
        str(tmp_path),
        make_nodes(raw),
        skeletons=skeletons,
        file_shas=None,  # pre-v3: stale=True path, but results still returned
    )


# ---------------------------------------------------------------------------
# RPC harness (mirrors the style used in test_component_issue66.py)
# ---------------------------------------------------------------------------

class _FakeReader:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)
        self._pos = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._pos >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._pos]
        self._pos += 1
        return line + b"\n"


class _FakeWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        pass

    def responses(self) -> list[dict]:
        result = []
        for chunk in self.chunks:
            for line in chunk.split(b"\n"):
                line = line.strip()
                if line:
                    result.append(json.loads(line))
        return result


def _rpc_req(method: str, params: Any = None, req_id: int = 1) -> bytes:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg).encode()


async def _run_rpc(server, lines: list[bytes]) -> list[dict]:
    reader = _FakeReader(lines)
    writer = _FakeWriter()
    await server._loop(reader, writer)
    return writer.responses()


# ---------------------------------------------------------------------------
# B1: graph.py → server.py — FQN-not-in-graph error propagation
#
# Contract:
#   1. CallGraphIndex.callers_of/callees_of/neighborhood with an absent FQN
#      returns {isError: True, error_reason: "fqn_not_in_graph",
#              stale: False, stale_files: [], ...commit fields}.
#   2. server.py._dispatch_tool checks result.get("isError") and, when True,
#      returns {"content": [{"type": "text", "text": <json>}], "isError": True}
#      — it must NOT route through _text() which would set isError:False.
#   3. An FQN that is present but has zero callers/callees returns
#      results: [] (not isError).
# ---------------------------------------------------------------------------

class TestB1FqnNotFoundErrorPropagation:
    """graph.py fqn_not_in_graph error → server.py MCP isError:true propagation."""

    @pytest.fixture()
    def _wired_server(self, tmp_path: Path):
        """Server fixture with a real in-memory index, wired into the RPC loop."""
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw), git_sha=None)
        idx_path = tmp_path / "index.json"
        idx.save(idx_path)

        _srv._INDEX_PATH = idx_path
        _srv._INDEX = idx
        _srv._BUILD_LOCK = None
        _srv._SERVER._shutdown_requested = False

        yield _srv._SERVER, _srv, tmp_path

        # Teardown — reset module-level state
        _srv._INDEX = None
        _srv._BUILD_LOCK = None

    # ------------------------------------------------------------------
    # Graph-layer unit tests (no server involvement)
    # ------------------------------------------------------------------

    def test_graph_callers_of_bad_fqn_returns_is_error(self, tmp_path: Path) -> None:
        """[B1-graph] refers_to with absent FQN returns isError:True dict.

        Verifies the graph-layer contract in isolation: the dict must have
        isError=True and error_reason="fqn_not_in_graph".  Stale must be
        False and stale_files must be empty — a missing FQN is not a
        staleness issue.
        """
        # Arrange
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.refers_to("does.not.exist", kind="callers", depth=1)

        # Assert
        assert result.get("isError") is True, (
            "refers_to must return isError:True for an absent FQN"
        )
        assert result.get("error_reason") == "fqn_not_in_graph", (
            f"error_reason must be 'fqn_not_in_graph', got {result.get('error_reason')!r}"
        )
        assert result.get("stale") is False, "stale must be False for a not-found error"
        assert result.get("stale_files") == [], "stale_files must be [] for a not-found error"

    def test_graph_callees_of_bad_fqn_returns_is_error(self, tmp_path: Path) -> None:
        """[B1-graph] callees_of with absent FQN returns isError:True dict."""
        # Arrange
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.callees_of("totally.absent.func", depth=1)

        # Assert
        assert result.get("isError") is True
        assert result.get("error_reason") == "fqn_not_in_graph"
        assert result.get("stale") is False
        assert result.get("stale_files") == []

    def test_graph_neighborhood_bad_fqn_returns_is_error(self, tmp_path: Path) -> None:
        """[B1-graph] neighborhood with absent FQN returns isError:True dict."""
        # Arrange
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.neighborhood("ghost.symbol", depth=2)

        # Assert
        assert result.get("isError") is True
        assert result.get("error_reason") == "fqn_not_in_graph"
        assert result.get("stale") is False
        assert result.get("stale_files") == []

    def test_graph_callers_of_known_fqn_zero_callers_returns_empty_results(
        self, tmp_path: Path
    ) -> None:
        """[B1-graph] refers_to a known FQN with zero callers returns results:[] not isError.

        Negative case: the isError path must NOT trigger when the FQN is present
        but simply has no callers.
        """
        # Arrange — pkg.other.gamma has no callers in the raw graph
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.refers_to("pkg.other.gamma", kind="callers", depth=1)

        # Assert
        assert result.get("isError") is not True, (
            "refers_to must NOT set isError for a known FQN with zero callers"
        )
        assert "results" in result, "refers_to must include 'results' key"
        assert result["results"] == [], (
            "refers_to for a node with zero callers must return results:[]"
        )

    def test_graph_callees_of_known_fqn_zero_callees_returns_empty_results(
        self, tmp_path: Path
    ) -> None:
        """[B1-graph] callees_of a known FQN with zero callees returns results:[] not isError."""
        # Arrange — pkg.mod.beta has no callees
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.callees_of("pkg.mod.beta", depth=1)

        # Assert
        assert result.get("isError") is not True
        assert "results" in result
        assert result["results"] == []

    def test_graph_neighborhood_known_fqn_isolated_returns_empty_edges(
        self, tmp_path: Path
    ) -> None:
        """[B1-graph] neighborhood for an isolated node returns edges:[] not isError."""
        # Arrange — add a node with no edges
        raw = _make_minimal_raw()
        raw["pkg.mod.isolated"] = []
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            result = idx.neighborhood("pkg.mod.isolated", depth=2)

        # Assert
        assert result.get("isError") is not True, (
            "neighborhood must NOT set isError for a known (but isolated) FQN"
        )
        assert "edges" in result
        assert result["edges"] == [], (
            "neighborhood for an isolated node must return edges:[]"
        )

    # ------------------------------------------------------------------
    # Server-layer tests via _dispatch_tool (unit, no RPC loop)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_server_dispatch_callers_of_bad_fqn_is_error_true(
        self, tmp_path: Path
    ) -> None:
        """[B1-server] _dispatch_tool('refers_to', bad FQN) surfaces isError:true.

        Verifies the server.py isError-check path for refers_to:
          result.get("isError") → True  ⟹  MCP response isError:True
        The content text must be valid JSON and must contain error_reason.
        """
        # Arrange
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
        _srv._INDEX = idx
        _srv._INDEX_PATH = tmp_path / "index.json"

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            response = await _srv._dispatch_tool(
                "refers_to", {"fqn": "no.such.function", "kind": "callers"}
            )

        # Assert
        assert response.get("isError") is True, (
            "server _dispatch_tool must return isError:True when graph returns "
            "isError:True for an absent FQN"
        )
        assert "content" in response
        content_text = response["content"][0]["text"]
        payload = json.loads(content_text)
        assert payload.get("error_reason") == "fqn_not_in_graph", (
            f"error_reason must be 'fqn_not_in_graph' in the content JSON, "
            f"got {payload.get('error_reason')!r}"
        )

    @pytest.mark.asyncio
    async def test_server_dispatch_callees_of_bad_fqn_is_error_true(
        self, tmp_path: Path
    ) -> None:
        """[B1-server] _dispatch_tool('callees_of', bad FQN) surfaces isError:true."""
        # Arrange
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
        _srv._INDEX = idx
        _srv._INDEX_PATH = tmp_path / "index.json"

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            response = await _srv._dispatch_tool(
                "callees_of", {"fqn": "missing.module.func"}
            )

        # Assert
        assert response.get("isError") is True
        content_text = response["content"][0]["text"]
        payload = json.loads(content_text)
        assert payload.get("error_reason") == "fqn_not_in_graph"
        assert payload.get("stale") is False

    @pytest.mark.asyncio
    async def test_server_dispatch_neighborhood_bad_fqn_is_error_true(
        self, tmp_path: Path
    ) -> None:
        """[B1-server] _dispatch_tool('neighborhood', bad FQN) surfaces isError:true."""
        # Arrange
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
        _srv._INDEX = idx
        _srv._INDEX_PATH = tmp_path / "index.json"

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            response = await _srv._dispatch_tool(
                "neighborhood", {"symbol": "phantom.func"}
            )

        # Assert
        assert response.get("isError") is True
        content_text = response["content"][0]["text"]
        payload = json.loads(content_text)
        assert payload.get("error_reason") == "fqn_not_in_graph"

    @pytest.mark.asyncio
    async def test_server_dispatch_callers_of_known_fqn_is_error_false(
        self, tmp_path: Path
    ) -> None:
        """[B1-server] _dispatch_tool('refers_to', known FQN) returns isError:false.

        Negative case: server must NOT set isError:true for a valid FQN that
        simply has zero callers.  Verifies the _text() path is taken instead.
        """
        # Arrange
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
        _srv._INDEX = idx
        _srv._INDEX_PATH = tmp_path / "index.json"

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            response = await _srv._dispatch_tool(
                "refers_to", {"fqn": "pkg.other.gamma", "kind": "callers"}
            )

        # Assert
        assert response.get("isError") is False, (
            "server must return isError:False for a known FQN with zero callers"
        )
        content_text = response["content"][0]["text"]
        payload = json.loads(content_text)
        assert "results" in payload
        assert payload["results"] == []

    # ------------------------------------------------------------------
    # Server-layer tests via full RPC loop
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rpc_callers_of_bad_fqn_is_error_true(
        self, _wired_server
    ) -> None:
        """[B1-rpc] refers_to with absent FQN: full RPC loop propagates isError:true.

        Exercises: JSON-RPC request → _SERVER._loop → _tools_call →
        _dispatch_tool("refers_to") → isError check → MCP response with
        isError:true and error_reason in content JSON.
        """
        server, srv, _ = _wired_server

        # Arrange
        lines = [
            _rpc_req(
                "tools/call",
                {"name": "refers_to", "arguments": {"fqn": "absolute.garbage.fqn", "kind": "callers"}},
                req_id=1,
            )
        ]

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            responses = await _run_rpc(server, lines)

        # Assert
        assert len(responses) == 1
        rpc_result = responses[0]["result"]
        assert rpc_result["isError"] is True, (
            "MCP result isError must be True when FQN is absent from the graph"
        )
        content_text = rpc_result["content"][0]["text"]
        payload = json.loads(content_text)
        assert payload.get("error_reason") == "fqn_not_in_graph", (
            f"error_reason must be 'fqn_not_in_graph' in the MCP content JSON, "
            f"got {payload.get('error_reason')!r}"
        )
        assert payload.get("stale") is False
        assert payload.get("stale_files") == []

    @pytest.mark.asyncio
    async def test_rpc_callees_of_bad_fqn_is_error_true(
        self, _wired_server
    ) -> None:
        """[B1-rpc] callees_of with absent FQN: full RPC loop propagates isError:true."""
        server, srv, _ = _wired_server

        # Arrange
        lines = [
            _rpc_req(
                "tools/call",
                {"name": "callees_of", "arguments": {"fqn": "not.in.graph.at.all"}},
                req_id=2,
            )
        ]

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            responses = await _run_rpc(server, lines)

        # Assert
        assert len(responses) == 1
        rpc_result = responses[0]["result"]
        assert rpc_result["isError"] is True
        payload = json.loads(rpc_result["content"][0]["text"])
        assert payload.get("error_reason") == "fqn_not_in_graph"

    @pytest.mark.asyncio
    async def test_rpc_neighborhood_bad_fqn_is_error_true(
        self, _wired_server
    ) -> None:
        """[B1-rpc] neighborhood with absent FQN: full RPC loop propagates isError:true."""
        server, srv, _ = _wired_server

        # Arrange
        lines = [
            _rpc_req(
                "tools/call",
                {"name": "neighborhood", "arguments": {"symbol": "xyz.unknown.sym"}},
                req_id=3,
            )
        ]

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            responses = await _run_rpc(server, lines)

        # Assert
        assert len(responses) == 1
        rpc_result = responses[0]["result"]
        assert rpc_result["isError"] is True
        payload = json.loads(rpc_result["content"][0]["text"])
        assert payload.get("error_reason") == "fqn_not_in_graph"

    @pytest.mark.asyncio
    async def test_rpc_callers_of_valid_fqn_is_error_false(
        self, _wired_server
    ) -> None:
        """[B1-rpc] refers_to with a valid FQN: full RPC loop does NOT set isError.

        Negative-path test: ensures the isError propagation is gated correctly
        and a well-formed FQN with zero callers does not accidentally surface as
        an error.
        """
        server, srv, _ = _wired_server

        # Arrange — pkg.other.gamma exists but has no callers in the raw graph
        lines = [
            _rpc_req(
                "tools/call",
                {"name": "refers_to", "arguments": {"fqn": "pkg.other.gamma", "kind": "callers"}},
                req_id=4,
            )
        ]

        # Act
        with patch("subprocess.run", return_value=_git_fail()):
            responses = await _run_rpc(server, lines)

        # Assert
        assert len(responses) == 1
        rpc_result = responses[0]["result"]
        assert rpc_result["isError"] is False, (
            "MCP result isError must be False for a valid FQN with zero callers"
        )
        payload = json.loads(rpc_result["content"][0]["text"])
        assert "results" in payload
        assert payload["results"] == [], "results must be [] for zero-caller FQN"


# ---------------------------------------------------------------------------
# B2: graph.py → server.py
#
# Contract: When CallGraphIndex.file_skeleton("bad/path.py") is called and the
# path is not in the index, it returns
#   {isError: True, error_reason: "path_not_in_index", stale: False, stale_files: []}
# (plus commit fields, no stale_action).
# When server.py's dispatch handler for file_skeleton receives this, it
# propagates isError: True at the MCP result level and the content text
# contains the full JSON dict including error_reason.
# The error_reason discriminator must survive serialisation and be readable
# by the client.
# ---------------------------------------------------------------------------

class TestB2PathNotInIndexErrorPropagation:
    """graph.py.file_skeleton path_not_in_index → server.py MCP isError propagation."""

    # ------------------------------------------------------------------
    # graph.py unit-side contract
    # ------------------------------------------------------------------

    def test_graph_returns_error_dict_for_missing_path(self, tmp_path: Path) -> None:
        """[B2-graph] file_skeleton on absent path returns isError dict with error_reason.

        Verifies Scenario D in graph.py: path not in self.skeletons.
        The returned dict must have isError=True, error_reason='path_not_in_index',
        stale=False, stale_files=[] — and no stale_action key.
        """
        idx = _make_index_with_skeleton(tmp_path)

        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            result = idx.file_skeleton("does/not/exist.py")

        assert result.get("isError") is True, (
            f"file_skeleton must return isError:True for absent path, got: {result}"
        )
        assert result.get("error_reason") == "path_not_in_index", (
            f"error_reason must be 'path_not_in_index', got: {result.get('error_reason')!r}"
        )
        assert result.get("stale") is False, (
            f"stale must be False in path_not_in_index error, got: {result.get('stale')!r}"
        )
        assert result.get("stale_files") == [], (
            f"stale_files must be [] in path_not_in_index error, got: {result.get('stale_files')!r}"
        )
        # No stale_action — the path is wrong, not stale.
        assert "stale_action" not in result, (
            "stale_action must NOT be present in path_not_in_index error (path is wrong, not stale)"
        )

    def test_graph_returns_commit_fields_in_error_dict(self, tmp_path: Path) -> None:
        """[B2-graph] path_not_in_index error dict includes commit-level staleness fields.

        The three commit fields (commit_stale, index_git_sha, head_git_sha) must
        always be present in the error dict so callers can distinguish 'path wrong'
        from 'index stale' without relying on stale_action absence alone.
        """
        idx = _make_index_with_skeleton(tmp_path)

        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            result = idx.file_skeleton("nonexistent/file.py")

        for field in ("commit_stale", "index_git_sha", "head_git_sha"):
            assert field in result, (
                f"commit field '{field}' must be present in path_not_in_index error dict"
            )

    # ------------------------------------------------------------------
    # server.py dispatch contract (direct _dispatch_tool path, no RPC loop)
    # ------------------------------------------------------------------

    @pytest.fixture()
    def _wired_index(self, tmp_path: Path):
        """Inject a real CallGraphIndex into server._INDEX."""
        idx = _make_index_with_skeleton(tmp_path)

        import pyscope_mcp.server as srv
        srv._INDEX = idx
        srv._INDEX_PATH = tmp_path / "index.json"

        yield srv

        srv._INDEX = None
        srv._INDEX_PATH = None

    @pytest.mark.asyncio
    async def test_server_dispatch_propagates_iserror_true_for_missing_path(
        self, _wired_index
    ) -> None:
        """[B2-server] _dispatch_tool('file_skeleton', bad path) → MCP isError:true.

        When graph.py returns isError:True, server.py must NOT wrap the result
        in a success _text() payload — it must propagate isError:True at the
        MCP result level.
        """
        import pyscope_mcp.server as srv

        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            result = await srv._dispatch_tool(
                "file_skeleton", {"path": "bad/path.py"}
            )

        assert result.get("isError") is True, (
            f"MCP result must have isError:True for absent path, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_server_dispatch_error_content_contains_error_reason(
        self, _wired_index
    ) -> None:
        """[B2-server] error content text contains 'error_reason': 'path_not_in_index'.

        The content[0].text must be JSON that includes error_reason so the MCP
        client can branch on it programmatically — not just read a human-readable
        message string.
        """
        import pyscope_mcp.server as srv

        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            result = await srv._dispatch_tool(
                "file_skeleton", {"path": "no/such/file.py"}
            )

        assert "content" in result, "MCP result must have 'content' key"
        assert len(result["content"]) >= 1, "MCP result content must be non-empty"
        text = result["content"][0]["text"]

        # The text must be valid JSON.
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"content[0].text must be valid JSON, got: {text!r}\nError: {exc}"
            )

        assert payload.get("error_reason") == "path_not_in_index", (
            f"Deserialised content must have error_reason='path_not_in_index', "
            f"got: {payload.get('error_reason')!r}"
        )
        assert payload.get("isError") is True, (
            f"Deserialised content must have isError:True, got: {payload.get('isError')!r}"
        )
        assert payload.get("stale") is False, (
            f"Deserialised content stale must be False, got: {payload.get('stale')!r}"
        )
        assert payload.get("stale_files") == [], (
            f"Deserialised content stale_files must be [], got: {payload.get('stale_files')!r}"
        )

    @pytest.mark.asyncio
    async def test_server_dispatch_error_reason_survives_serialisation(
        self, _wired_index
    ) -> None:
        """[B2-server] error_reason discriminator survives JSON serialisation round-trip.

        This is the core end-to-end contract for this boundary: the string
        'path_not_in_index' must be identical after json.loads(json.dumps(...)).
        A regression that loses or transforms the discriminator would silently
        break all client branching logic.
        """
        import pyscope_mcp.server as srv

        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            result = await srv._dispatch_tool(
                "file_skeleton", {"path": "definitely/not/there.py"}
            )

        raw_text = result["content"][0]["text"]
        # Double round-trip: server serialises → client deserialises
        client_view = json.loads(raw_text)
        # Re-serialise (client might store/log) → re-deserialise
        client_view_2 = json.loads(json.dumps(client_view))

        assert client_view_2["error_reason"] == "path_not_in_index", (
            "error_reason must survive a double JSON round-trip without mutation"
        )

    # ------------------------------------------------------------------
    # Positive case: valid path returns skeleton content (not isError)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_server_dispatch_valid_path_returns_success(
        self, _wired_index
    ) -> None:
        """[B2-server] file_skeleton with a valid indexed path returns isError:false.

        Verifies the positive / non-error branch so we know the error path is
        not silently always-on. The response must have isError:False and the
        content payload must include 'results'.
        """
        import pyscope_mcp.server as srv

        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            result = await srv._dispatch_tool(
                "file_skeleton", {"path": "pkg/mod.py"}
            )

        assert result.get("isError") is False, (
            f"file_skeleton on a valid path must return isError:False, got: {result}"
        )
        payload = json.loads(result["content"][0]["text"])
        assert "results" in payload, (
            f"Success payload must include 'results', got keys: {list(payload.keys())}"
        )
        assert isinstance(payload["results"], list), (
            "results must be a list"
        )
        # At least one symbol was indexed for pkg/mod.py.
        assert len(payload["results"]) >= 1, (
            f"Expected ≥1 symbol in results for pkg/mod.py, got: {payload['results']}"
        )

    def test_graph_valid_path_does_not_return_iserror(self, tmp_path: Path) -> None:
        """[B2-graph] file_skeleton on a known path does NOT return isError.

        Complement to the error test: verifies the positive branch at the graph
        layer to confirm error_reason is only present on the absent-path path.
        """
        idx = _make_index_with_skeleton(tmp_path)

        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            result = idx.file_skeleton("pkg/mod.py")

        assert not result.get("isError"), (
            f"file_skeleton on a known path must NOT return isError, got: {result}"
        )
        assert "error_reason" not in result, (
            "error_reason must not be present in success response"
        )
        assert "results" in result, (
            "success response must include 'results'"
        )

    # ------------------------------------------------------------------
    # Full RPC loop path
    # ------------------------------------------------------------------

    @pytest.fixture()
    def _wired_server(self, tmp_path: Path):
        """Server fixture with real index, hooked into the RPC loop."""
        idx = _make_index_with_skeleton(tmp_path)

        import pyscope_mcp.server as srv
        srv._INDEX = idx
        srv._INDEX_PATH = tmp_path / "index.json"
        srv._SERVER._shutdown_requested = False

        yield srv._SERVER, srv

        srv._INDEX = None
        srv._INDEX_PATH = None

    @pytest.mark.asyncio
    async def test_rpc_loop_file_skeleton_missing_path_iserror(
        self, _wired_server
    ) -> None:
        """[B2-rpc] file_skeleton via full RPC loop: absent path yields isError:true.

        Exercises the full JSON-RPC 2.0 wire path: request bytes → _loop →
        _tools_call → _dispatch_tool → graph.file_skeleton → response bytes.
        """
        server, srv = _wired_server
        lines = [
            _rpc_req(
                "tools/call",
                {"name": "file_skeleton", "arguments": {"path": "ghost/file.py"}},
                req_id=42,
            )
        ]

        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            responses = await _run_rpc(server, lines)

        assert len(responses) == 1
        rpc_result = responses[0]["result"]
        assert rpc_result.get("isError") is True, (
            f"RPC result must have isError:True for absent path, got: {rpc_result}"
        )
        payload = json.loads(rpc_result["content"][0]["text"])
        assert payload.get("error_reason") == "path_not_in_index", (
            f"error_reason must be 'path_not_in_index' in RPC response, "
            f"got: {payload.get('error_reason')!r}"
        )

    @pytest.mark.asyncio
    async def test_rpc_loop_file_skeleton_valid_path_success(
        self, _wired_server
    ) -> None:
        """[B2-rpc] file_skeleton via full RPC loop: valid path yields isError:false.

        Positive case through the RPC loop to confirm the loop wiring works for
        both the error and success branches.
        """
        server, srv = _wired_server
        lines = [
            _rpc_req(
                "tools/call",
                {"name": "file_skeleton", "arguments": {"path": "pkg/mod.py"}},
                req_id=43,
            )
        ]

        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            responses = await _run_rpc(server, lines)

        assert len(responses) == 1
        rpc_result = responses[0]["result"]
        assert rpc_result.get("isError") is False, (
            f"RPC result must have isError:False for valid path, got: {rpc_result}"
        )
        payload = json.loads(rpc_result["content"][0]["text"])
        assert "results" in payload, (
            f"Success payload must include 'results', got keys: {list(payload.keys())}"
        )
