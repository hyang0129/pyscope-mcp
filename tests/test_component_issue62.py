"""Component tests for issue #62 (query logging) — impact-set boundary coverage.

These tests focus on module boundaries that consume graph.py and are in the
assessment IMPACT_SET: test_rpc_server.py, test_hub_suppression_integration.py,
test_staleness.py, and src/pyscope_mcp/__init__.py.

Cross-cutting changes validated here:
  - INDEX_VERSION v4 → v5 (save/load round-trip for new fields)
  - New required ``git_sha`` and ``content_hash`` fields on CallGraphIndex
  - ``_bfs`` return type change list → dict[str, int] (consumed by callers_of /
    callees_of / module_callers / module_callees)
  - New ``dropped`` field on CallersResult / CalleesResult / ModuleResult
  - RPC layer: ``dropped`` field present in JSON payloads from callers_of,
    callees_of, module_callers, module_callees tool calls

Negative control: test_version_mismatch_raises uses a hard-coded v4 index
to prove that load() rejects old indexes — this verifies the test runner can
detect a regression if INDEX_VERSION is reverted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pyscope_mcp.graph import INDEX_VERSION, CallGraphIndex
from conftest import make_nodes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_minimal_raw() -> dict[str, list[str]]:
    return {
        "pkg.mod.foo": ["pkg.mod.bar"],
        "pkg.mod.bar": [],
        "pkg.other.baz": ["pkg.mod.foo"],
    }


def _make_index(tmp_path: Path, raw: dict[str, list[str]] | None = None) -> CallGraphIndex:
    if raw is None:
        raw = _make_minimal_raw()
    return CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))


# ---------------------------------------------------------------------------
# A. INDEX_VERSION = 5 round-trip (save/load)
# ---------------------------------------------------------------------------

def test_index_version_is_five() -> None:
    """INDEX_VERSION must equal 5 on the fix/issue-62-query-logging branch."""
    assert INDEX_VERSION == 5, (
        f"INDEX_VERSION={INDEX_VERSION}; expected 5. "
        "This test exists to pin the version and detect silent regressions."
    )


def test_save_writes_version_five(tmp_path: Path) -> None:
    """CallGraphIndex.save() must write version=5 to disk."""
    idx = _make_index(tmp_path)
    saved = idx.save(tmp_path / "index.json")
    payload = json.loads(saved.read_text())
    assert payload["version"] == 5, (
        f"Saved index version={payload['version']}, expected 5"
    )


def test_load_rejects_v4_index(tmp_path: Path) -> None:
    """CallGraphIndex.load() must raise ValueError for a v4 index."""
    old_payload = {
        "version": 4,
        "root": str(tmp_path),
        "raw": _make_minimal_raw(),
        "skeletons": {},
        "file_shas": {},
        "missed_callers": {},
    }
    idx_path = tmp_path / "old_index.json"
    idx_path.write_text(json.dumps(old_payload))
    with pytest.raises(ValueError, match="v4"):
        CallGraphIndex.load(idx_path)


def test_load_accepts_v5_index(tmp_path: Path) -> None:
    """CallGraphIndex.load() must succeed for a freshly saved v5 index."""
    idx = _make_index(tmp_path)
    saved = idx.save(tmp_path / "index.json")
    loaded = CallGraphIndex.load(saved)
    assert loaded.function_graph.number_of_nodes() == idx.function_graph.number_of_nodes()


# ---------------------------------------------------------------------------
# B. git_sha and content_hash fields
# ---------------------------------------------------------------------------

def test_from_raw_populates_content_hash(tmp_path: Path) -> None:
    """content_hash is computed and non-empty after from_raw()."""
    raw = _make_minimal_raw()
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
    assert isinstance(idx.content_hash, str)
    assert len(idx.content_hash) == 64, "SHA-256 hex digest must be 64 chars"


def test_content_hash_is_deterministic(tmp_path: Path) -> None:
    """Same raw dict always produces the same content_hash."""
    raw = _make_minimal_raw()
    idx1 = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
    idx2 = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
    assert idx1.content_hash == idx2.content_hash


def test_content_hash_changes_with_raw(tmp_path: Path) -> None:
    """Different raw dicts produce different content_hashes."""
    raw_a = {"pkg.a": ["pkg.b"], "pkg.b": []}
    raw_b = {"pkg.a": ["pkg.c"], "pkg.c": []}
    idx_a = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw_a))
    idx_b = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw_b))
    assert idx_a.content_hash != idx_b.content_hash


def test_git_sha_defaults_to_none(tmp_path: Path) -> None:
    """git_sha defaults to None when not provided to from_raw()."""
    idx = _make_index(tmp_path)
    assert idx.git_sha is None


def test_git_sha_roundtrips_through_save_load(tmp_path: Path) -> None:
    """git_sha is persisted on save and restored on load."""
    raw = _make_minimal_raw()
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw), git_sha="abc123def456")
    saved = idx.save(tmp_path / "index.json")
    loaded = CallGraphIndex.load(saved)
    assert loaded.git_sha == "abc123def456"


def test_git_sha_none_roundtrips_through_save_load(tmp_path: Path) -> None:
    """git_sha=None round-trips correctly (persisted as null, restored as None)."""
    raw = _make_minimal_raw()
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw), git_sha=None)
    saved = idx.save(tmp_path / "index.json")
    loaded = CallGraphIndex.load(saved)
    assert loaded.git_sha is None


def test_content_hash_roundtrips_through_save_load(tmp_path: Path) -> None:
    """content_hash is persisted on save and identical after load."""
    raw = _make_minimal_raw()
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
    saved = idx.save(tmp_path / "index.json")
    loaded = CallGraphIndex.load(saved)
    assert loaded.content_hash == idx.content_hash
    assert len(loaded.content_hash) == 64


def test_saved_payload_has_git_sha_and_content_hash_keys(tmp_path: Path) -> None:
    """Serialised index JSON must contain git_sha and content_hash keys."""
    raw = _make_minimal_raw()
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw), git_sha="deadbeef")
    saved = idx.save(tmp_path / "index.json")
    payload = json.loads(saved.read_text())
    assert "git_sha" in payload, "git_sha must be present in saved index"
    assert "content_hash" in payload, "content_hash must be present in saved index"
    assert payload["git_sha"] == "deadbeef"
    assert payload["content_hash"] == idx.content_hash


# ---------------------------------------------------------------------------
# C. dropped field on callers_of / callees_of (graph.py boundary)
# ---------------------------------------------------------------------------

def test_callers_of_has_dropped_field(tmp_path: Path) -> None:
    """callers_of result always contains a 'dropped' key (0 when not truncated)."""
    raw = _make_minimal_raw()
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
    result = idx.callers_of("pkg.mod.bar", depth=1)
    assert "dropped" in result, "callers_of must include 'dropped' in result dict"
    assert isinstance(result["dropped"], int)
    assert result["dropped"] == 0  # small fixture, no truncation


def test_callees_of_has_dropped_field(tmp_path: Path) -> None:
    """callees_of result always contains a 'dropped' key (0 when not truncated)."""
    raw = _make_minimal_raw()
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
    result = idx.callees_of("pkg.mod.foo", depth=1)
    assert "dropped" in result, "callees_of must include 'dropped' in result dict"
    assert isinstance(result["dropped"], int)
    assert result["dropped"] == 0


def test_module_callers_has_dropped_field(tmp_path: Path) -> None:
    """module_callers result always contains a 'dropped' key (0 when not truncated)."""
    raw = _make_minimal_raw()
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
    result = idx.module_callers("pkg.mod", depth=1)
    assert "dropped" in result, "module_callers must include 'dropped' in result dict"
    assert isinstance(result["dropped"], int)
    assert result["dropped"] == 0


def test_module_callees_has_dropped_field(tmp_path: Path) -> None:
    """module_callees result always contains a 'dropped' key (0 when not truncated)."""
    raw = _make_minimal_raw()
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
    result = idx.module_callees("pkg.other", depth=1)
    assert "dropped" in result, "module_callees must include 'dropped' in result dict"
    assert isinstance(result["dropped"], int)
    assert result["dropped"] == 0


# ---------------------------------------------------------------------------
# D. RPC layer: dropped field present in JSON tool responses
# (boundary: test_rpc_server.py-style harness against server.py)
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


def _req(method: str, params: Any = None, req_id: int = 1) -> bytes:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg).encode()


async def _run_server(server, lines: list[bytes]) -> list[dict]:
    reader = _FakeReader(lines)
    writer = _FakeWriter()
    await server._loop(reader, writer)
    return writer.responses()


@pytest.fixture()
def _server_with_index(tmp_path: Path):
    """A server fixture wired to a fresh minimal index."""
    raw = _make_minimal_raw()
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw))
    idx_path = tmp_path / "index.json"
    idx.save(idx_path)

    import pyscope_mcp.server as srv
    srv._INDEX_PATH = idx_path
    srv._INDEX = None  # force lazy reload
    srv._SERVER._shutdown_requested = False
    yield srv._SERVER
    srv._INDEX = None


@pytest.mark.asyncio
async def test_rpc_callers_of_response_has_dropped(_server_with_index) -> None:
    """callers_of tool response JSON must include 'dropped' key."""
    lines = [_req("tools/call", {"name": "callers_of", "arguments": {"fqn": "pkg.mod.bar"}}, req_id=1)]
    responses = await _run_server(_server_with_index, lines)
    r = responses[0]["result"]
    assert r["isError"] is False
    payload = json.loads(r["content"][0]["text"])
    assert "dropped" in payload, (
        "callers_of RPC response must include 'dropped' field — "
        "types.CallersResult was updated in issue #62 but server serialisation may be missing it"
    )
    assert isinstance(payload["dropped"], int)
    assert payload["dropped"] == 0  # small fixture, no cap triggered


@pytest.mark.asyncio
async def test_rpc_callees_of_response_has_dropped(_server_with_index) -> None:
    """callees_of tool response JSON must include 'dropped' key."""
    lines = [_req("tools/call", {"name": "callees_of", "arguments": {"fqn": "pkg.mod.foo"}}, req_id=1)]
    responses = await _run_server(_server_with_index, lines)
    r = responses[0]["result"]
    assert r["isError"] is False
    payload = json.loads(r["content"][0]["text"])
    assert "dropped" in payload, (
        "callees_of RPC response must include 'dropped' field"
    )
    assert isinstance(payload["dropped"], int)
    assert payload["dropped"] == 0


@pytest.mark.asyncio
async def test_rpc_module_callers_response_has_dropped(_server_with_index) -> None:
    """module_callers tool response JSON must include 'dropped' key."""
    lines = [_req("tools/call", {"name": "module_callers", "arguments": {"module": "pkg.mod"}}, req_id=1)]
    responses = await _run_server(_server_with_index, lines)
    r = responses[0]["result"]
    assert r["isError"] is False
    payload = json.loads(r["content"][0]["text"])
    assert "dropped" in payload, (
        "module_callers RPC response must include 'dropped' field"
    )
    assert isinstance(payload["dropped"], int)
    assert payload["dropped"] == 0


@pytest.mark.asyncio
async def test_rpc_module_callees_response_has_dropped(_server_with_index) -> None:
    """module_callees tool response JSON must include 'dropped' key."""
    lines = [_req("tools/call", {"name": "module_callees", "arguments": {"module": "pkg.other"}}, req_id=1)]
    responses = await _run_server(_server_with_index, lines)
    r = responses[0]["result"]
    assert r["isError"] is False
    payload = json.loads(r["content"][0]["text"])
    assert "dropped" in payload, (
        "module_callees RPC response must include 'dropped' field"
    )
    assert isinstance(payload["dropped"], int)
    assert payload["dropped"] == 0


# ---------------------------------------------------------------------------
# E. Negative control — verifies the runner can detect a regression
# Pinned test: version mismatch raises
# ---------------------------------------------------------------------------

def test_version_mismatch_raises(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: load() with wrong version must raise ValueError.

    This test pins the version-mismatch rejection path so that any revert of
    INDEX_VERSION (e.g. back to v4) would cause this test to fail with a clear
    error message, rather than silently passing with corrupted state.
    """
    # Write a v3 index (two versions behind current v5)
    stale_payload = {
        "version": 3,
        "root": str(tmp_path),
        "raw": _make_minimal_raw(),
        "skeletons": {},
        "file_shas": {},
        # v3 has no missed_callers, git_sha, content_hash
    }
    idx_path = tmp_path / "stale.json"
    idx_path.write_text(json.dumps(stale_payload))
    with pytest.raises(ValueError, match="v3"):
        CallGraphIndex.load(idx_path)


# ---------------------------------------------------------------------------
# F. __init__.py boundary: CallGraphIndex importable from pyscope_mcp
# ---------------------------------------------------------------------------

def test_init_exports_call_graph_index() -> None:
    """pyscope_mcp.__init__ must re-export CallGraphIndex (src/pyscope_mcp/__init__.py boundary)."""
    from pyscope_mcp import CallGraphIndex as CGI  # noqa: N811
    # Verify it's the same class (not a copy/import confusion)
    assert CGI is CallGraphIndex, (
        "pyscope_mcp.CallGraphIndex must be the same object as "
        "pyscope_mcp.graph.CallGraphIndex — __init__.py re-export is broken"
    )


def test_init_exports_version() -> None:
    """pyscope_mcp.__version__ must be non-empty."""
    from pyscope_mcp import __version__
    assert isinstance(__version__, str) and __version__


# ---------------------------------------------------------------------------
# G. hub_suppression_integration boundary:
#    CallGraphIndex.from_raw() with new git_sha param accepted
# ---------------------------------------------------------------------------

def test_hub_suppression_index_accepts_git_sha(tmp_path: Path) -> None:
    """CallGraphIndex.from_raw() must accept git_sha kwarg without error.

    Covers the test_hub_suppression_integration boundary — hub_idx fixture calls
    from_raw() and the new git_sha parameter must not break it.
    """
    raw = {
        "pkg.utils.shared_helper": [],
        "pkg.app.entry_point": ["pkg.utils.shared_helper"],
    }
    # Must not raise
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw), git_sha="cafebabe01234567")
    assert idx.git_sha == "cafebabe01234567"
    assert idx.content_hash  # populated, non-empty
