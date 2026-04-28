"""Component tests for issue #66 — commit-SHA staleness + build MCP tool.

Three cross-module boundaries tested:

  B1 (graph.py → types.py): _commit_staleness() output merges into all query
     TypedDicts via **commit unpacking.  Contract: the three commit fields
     (commit_stale, index_git_sha, head_git_sha) are present with the correct
     Python types in every query response.

  B2 (server.py → graph.py via RPC loop): the 'build' tool dispatched through
     the full _SERVER._loop path returns a stats() JSON payload that includes
     the three new commit fields.  This covers the wiring between server.py's
     lock/subprocess path and the graph.py reload that follows.

  B3 (cli.py → graph.py): cmd_build() captures the git SHA at build time and
     passes it to CallGraphIndex.from_raw().  The saved index contains that SHA
     and load() restores it — the CLI→graph wiring for the git_sha parameter
     introduced in issue #66.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pyscope_mcp.graph import CallGraphIndex
from conftest import make_nodes

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_MOCK_HEAD_SHA = "aabbccdd00112233aabbccdd00112233aabbccdd"
_MOCK_INDEX_SHA = "1111111111111111111111111111111111111111"


def _make_minimal_raw() -> dict[str, list[str]]:
    return {
        "pkg.mod.alpha": ["pkg.mod.beta"],
        "pkg.mod.beta": [],
        "pkg.other.gamma": ["pkg.mod.alpha"],
    }


def _git_ok(sha: str = _MOCK_HEAD_SHA) -> MagicMock:
    """Fake subprocess.run result: git success."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = sha + "\n"
    return m


def _git_fail() -> MagicMock:
    """Fake subprocess.run result: git failure."""
    m = MagicMock()
    m.returncode = 128
    m.stdout = ""
    return m


# ---------------------------------------------------------------------------
# RPC harness (mirrors the style used in test_rpc_server.py / test_component_issue62.py)
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
# B1: graph.py → types.py
#
# Contract: _commit_staleness() output merges into TypedDicts.
# Tested at two representative surfaces (callers_of and stats) to confirm the
# **commit unpacking is wired — not every tool, since that would duplicate
# test_commit_staleness.py.
# ---------------------------------------------------------------------------

class TestB1GraphToTypes:
    """graph.py._commit_staleness() → types.py TypedDict merge contract."""

    def test_callers_of_commit_fields_have_correct_types(self, tmp_path: Path) -> None:
        """[B1] callers_of result: commit fields present with correct Python types.

        Checks the type contract, not just key presence: commit_stale is bool,
        index_git_sha and head_git_sha are str.  A regression that changes the
        merge from **commit to a manual copy could omit or coerce fields.
        """
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw), git_sha=_MOCK_INDEX_SHA)

        with patch("subprocess.run", return_value=_git_ok(_MOCK_HEAD_SHA)):
            result = idx.refers_to("pkg.mod.beta", kind="callers", depth=1)

        # [Assert] All three commit fields are present.
        assert "commit_stale" in result, "commit_stale must be in refers_to result"
        assert "index_git_sha" in result, "index_git_sha must be in refers_to result"
        assert "head_git_sha" in result, "head_git_sha must be in refers_to result"

        # [Assert] Types match the TypedDict contract (NotRequired[bool|None] / NotRequired[str|None]).
        assert isinstance(result["commit_stale"], bool), (
            f"commit_stale must be bool, got {type(result['commit_stale'])}"
        )
        assert isinstance(result["index_git_sha"], str), (
            f"index_git_sha must be str, got {type(result['index_git_sha'])}"
        )
        assert isinstance(result["head_git_sha"], str), (
            f"head_git_sha must be str, got {type(result['head_git_sha'])}"
        )

    def test_stats_commit_fields_none_when_git_unavailable(self, tmp_path: Path) -> None:
        """[B1] stats() commit fields are None when git binary is absent.

        The _commit_staleness() 'all None' path must merge into StatsResult
        correctly — if **commit unpacking is broken for the None path, the fields
        would be absent from the dict entirely rather than present with None.
        """
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw), git_sha=_MOCK_INDEX_SHA)

        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            result = idx.stats()

        # [Assert] All three commit fields are present even when None.
        assert "commit_stale" in result, "commit_stale key must be present even when None"
        assert "index_git_sha" in result, "index_git_sha key must be present even when None"
        assert "head_git_sha" in result, "head_git_sha key must be present even when None"

        # [Assert] All None — that is the contract when git is unavailable.
        assert result["commit_stale"] is None
        assert result["index_git_sha"] is None
        assert result["head_git_sha"] is None

    def test_search_commit_stale_true_when_sha_differs(self, tmp_path: Path) -> None:
        """[B1] search() commit_stale is True when HEAD diverged from index SHA.

        Verifies the **commit merge carries the actual commit_stale value through
        to SearchResult; a merge that drops the value would silently be False/None.
        """
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw), git_sha=_MOCK_INDEX_SHA)

        with patch("subprocess.run", return_value=_git_ok(_MOCK_HEAD_SHA)):
            result = idx.search("alpha")

        assert result["commit_stale"] is True, (
            "commit_stale must be True when HEAD SHA differs from index git_sha"
        )
        assert result["index_git_sha"] == _MOCK_INDEX_SHA
        assert result["head_git_sha"] == _MOCK_HEAD_SHA


# ---------------------------------------------------------------------------
# B2: server.py → graph.py (build tool via full RPC loop)
#
# Contract: the 'build' tool dispatched through the RPC loop returns a
# stats() payload (as JSON text) that includes the three new commit fields.
# test_commit_staleness.py tests _dispatch_tool() directly; this test exercises
# the wiring through _SERVER._loop → _tools_call → _dispatch_tool.
# ---------------------------------------------------------------------------

class TestB2ServerGraphBuildRpcLoop:
    """server.py 'build' tool → graph.py.stats() commit fields via RPC loop."""

    @pytest.fixture()
    def _wired_server(self, tmp_path: Path):
        """Server fixture with a real saved index, hooked into the RPC loop."""
        raw = _make_minimal_raw()
        idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw), git_sha=_MOCK_INDEX_SHA)
        idx_path = tmp_path / "index.json"
        idx.save(idx_path)

        import pyscope_mcp.server as srv
        srv._INDEX_PATH = idx_path
        srv._INDEX = idx
        srv._BUILD_LOCK = None
        srv._SERVER._shutdown_requested = False

        yield srv._SERVER, srv, tmp_path

        # Teardown — reset module-level state
        srv._INDEX = None
        srv._BUILD_LOCK = None

    @pytest.mark.asyncio
    async def test_build_rpc_stats_payload_has_commit_fields(
        self, _wired_server, tmp_path: Path
    ) -> None:
        """[B2] build tool via RPC loop: stats payload includes commit_stale, index_git_sha, head_git_sha.

        Verifies the full path: JSON-RPC request → _SERVER._loop → _tools_call
        → _dispatch_tool("build") → subprocess mock → CallGraphIndex.load →
        stats() → JSON serialisation.  The commit fields must survive the full
        round-trip.
        """
        server, srv, td = _wired_server

        # Mock the build subprocess (returns success).
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""

        lines = [
            _rpc_req("tools/call", {"name": "build", "arguments": {}}, req_id=10)
        ]

        with patch("pyscope_mcp.server.subprocess.run", return_value=mock_proc):
            with patch("subprocess.run", return_value=_git_ok(_MOCK_HEAD_SHA)):
                responses = await _run_rpc(server, lines)

        assert len(responses) == 1
        rpc_result = responses[0]["result"]
        assert rpc_result["isError"] is False, (
            f"build tool returned isError:true — {rpc_result.get('content')}"
        )

        payload = json.loads(rpc_result["content"][0]["text"])

        # [Assert] Normal stats fields survive the round-trip.
        assert "functions" in payload, "stats payload must include 'functions'"
        assert "function_edges" in payload, "stats payload must include 'function_edges'"

        # [Assert] New commit-staleness fields are present in the JSON payload.
        assert "commit_stale" in payload, (
            "build tool stats payload must include 'commit_stale' — "
            "_commit_staleness() merge may be missing from stats()"
        )
        assert "index_git_sha" in payload, (
            "build tool stats payload must include 'index_git_sha'"
        )
        assert "head_git_sha" in payload, (
            "build tool stats payload must include 'head_git_sha'"
        )

    @pytest.mark.asyncio
    async def test_build_rpc_concurrent_rejection_through_loop(
        self, _wired_server
    ) -> None:
        """[B2] build tool RPC: concurrent call rejected through full _loop path.

        Ensures the lock-held guard in _dispatch_tool surfaces as isError:true
        when routed through the RPC loop rather than via direct _dispatch_tool
        invocation.
        """
        server, srv, _ = _wired_server
        lock = srv._get_build_lock()
        await lock.acquire()

        lines = [
            _rpc_req("tools/call", {"name": "build", "arguments": {}}, req_id=20)
        ]
        try:
            responses = await _run_rpc(server, lines)
        finally:
            lock.release()

        assert len(responses) == 1
        rpc_result = responses[0]["result"]
        assert rpc_result["isError"] is True
        assert "already in progress" in rpc_result["content"][0]["text"]


# ---------------------------------------------------------------------------
# B3: cli.py → graph.py
#
# Contract: cmd_build() captures git SHA via subprocess and forwards it through
# CallGraphIndex.from_raw() into the saved index.  load() must restore it.
# This is the CLI→graph wiring for the git_sha parameter introduced in #66.
# ---------------------------------------------------------------------------

class TestB3CliToGraph:
    """cli.cmd_build() → graph.py git_sha capture and persistence contract."""

    def test_cmd_build_git_sha_persisted_in_saved_index(self, tmp_path: Path) -> None:
        """[B3] cmd_build captures git SHA from subprocess and saves it in the index.

        When git rev-parse HEAD succeeds, cmd_build() calls from_raw(..., git_sha=sha)
        and the resulting index must contain that SHA after save+load.
        """
        from unittest.mock import call

        # Create a minimal package structure so the analyzer can run.
        pkg_dir = tmp_path / "mypkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "mod.py").write_text("def foo(): pass\n")

        import pyscope_mcp.cli as cli_mod

        # Build subprocess mock: first call is the analyzer's git rev-parse (in cli.cmd_build),
        # analyzer internally may call subprocess too — we only care that the final index
        # captures the SHA our mock returns.
        git_mock = MagicMock()
        git_mock.returncode = 0
        git_mock.stdout = _MOCK_INDEX_SHA + "\n"

        import argparse
        args = argparse.Namespace(
            root=str(tmp_path),
            package="mypkg",
            output=str(tmp_path / "out" / "index.json"),
        )

        with patch("pyscope_mcp.cli.subprocess.run", return_value=git_mock):
            ret = cli_mod.cmd_build(args)

        assert ret == 0, "cmd_build must return 0 on success"

        # [Assert] The saved index contains the git SHA that subprocess reported.
        saved_path = tmp_path / "out" / "index.json"
        assert saved_path.exists(), "cmd_build must write the index file"
        payload = json.loads(saved_path.read_text())
        assert "git_sha" in payload, "saved index must contain 'git_sha' key"
        assert payload["git_sha"] == _MOCK_INDEX_SHA, (
            f"saved index git_sha={payload['git_sha']!r}, "
            f"expected {_MOCK_INDEX_SHA!r} — "
            "cli.cmd_build() may not be forwarding the captured SHA to from_raw()"
        )

    def test_cmd_build_git_sha_none_when_git_absent(self, tmp_path: Path) -> None:
        """[B3] cmd_build sets git_sha=None when git binary is absent.

        FileNotFoundError from subprocess.run must be caught and git_sha must
        remain None in the saved index — not propagate as an uncaught exception.
        """
        pkg_dir = tmp_path / "mypkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "mod.py").write_text("def foo(): pass\n")

        import pyscope_mcp.cli as cli_mod
        import argparse
        args = argparse.Namespace(
            root=str(tmp_path),
            package="mypkg",
            output=str(tmp_path / "out2" / "index.json"),
        )

        with patch("pyscope_mcp.cli.subprocess.run", side_effect=FileNotFoundError("no git")):
            ret = cli_mod.cmd_build(args)

        assert ret == 0, "cmd_build must return 0 even when git is absent"

        saved_path = tmp_path / "out2" / "index.json"
        payload = json.loads(saved_path.read_text())

        # [Assert] git_sha is null in the saved index (not omitted, not an error value).
        assert "git_sha" in payload, "git_sha key must still be present when git is absent"
        assert payload["git_sha"] is None, (
            f"git_sha must be null when git is unavailable, got {payload['git_sha']!r}"
        )

    def test_cmd_build_git_sha_roundtrips_through_load(self, tmp_path: Path) -> None:
        """[B3] git_sha captured by cmd_build survives save→load round-trip.

        Verifies the full CLI→graph contract: git SHA captured → from_raw →
        save → load → git_sha attribute accessible on the loaded index.
        """
        pkg_dir = tmp_path / "mypkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "mod.py").write_text("def bar(): pass\n")

        import pyscope_mcp.cli as cli_mod
        import argparse
        args = argparse.Namespace(
            root=str(tmp_path),
            package="mypkg",
            output=str(tmp_path / "out3" / "index.json"),
        )

        git_mock = MagicMock()
        git_mock.returncode = 0
        git_mock.stdout = _MOCK_INDEX_SHA + "\n"

        with patch("pyscope_mcp.cli.subprocess.run", return_value=git_mock):
            cli_mod.cmd_build(args)

        # [Act] Load the saved index.
        loaded = CallGraphIndex.load(tmp_path / "out3" / "index.json")

        # [Assert] git_sha is accessible and matches what git reported.
        assert loaded.git_sha == _MOCK_INDEX_SHA, (
            f"loaded index git_sha={loaded.git_sha!r}, expected {_MOCK_INDEX_SHA!r} — "
            "save/load round-trip for git_sha is broken"
        )
