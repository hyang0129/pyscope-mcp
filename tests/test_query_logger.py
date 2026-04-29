"""Tests for the QueryLogger (src/pyscope_mcp/_log.py).

Uses the same FakeReader/FakeWriter in-process harness from test_rpc_server.py
to exercise the full dispatch path, including the logger intercept in server.py.

Test scenarios per the refined spec:
  S1 — happy-path tool call
  S2 — error-path tool call (unknown tool name)
  S3 — neighborhood with truncation / hub-suppression
  S4 — opt-out path (PYSCOPE_MCP_LOG=0)
  S5 — log rotation at 10 MB
  S6 — reload path (index identity updates after reload)
  S7 — logger-failure path (internal I/O error does not affect tool response)
  S8 — index schema v5 round-trip (git_sha + content_hash)
  S9 — default-on behaviour (PYSCOPE_MCP_LOG unset → logger enabled,
       activation emits a WARNING with the active log path)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pyscope_mcp._rpc import RpcServer
from pyscope_mcp.graph import INDEX_VERSION, CallGraphIndex
from conftest import make_nodes


# ---------------------------------------------------------------------------
# Re-use the pipe harness from test_rpc_server
# ---------------------------------------------------------------------------

class FakeReader:
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


class FakeWriter:
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


def _line(obj: Any) -> bytes:
    return json.dumps(obj).encode()


def _req(method: str, params: Any = None, req_id: int = 1) -> bytes:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return _line(msg)


async def _run(server: RpcServer, lines: list[bytes]) -> list[dict]:
    reader = FakeReader(lines)
    writer = FakeWriter()
    await server._loop(reader, writer)
    return writer.responses()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_graph() -> dict[str, list[str]]:
    return {
        "pkg.mod.foo": ["pkg.mod.bar"],
        "pkg.mod.bar": [],
        "pkg.other.baz": ["pkg.mod.foo"],
    }


@pytest.fixture()
def tmp_index(tmp_path: Path, raw_graph: dict) -> Path:
    """Minimal v5 index on disk."""
    idx = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw_graph))
    idx_path = tmp_path / ".pyscope-mcp" / "index.json"
    idx.save(idx_path)
    return idx_path


@pytest.fixture(autouse=True)
def reset_server_state(tmp_index: Path):
    """Reset server module state + logger between every test."""
    import pyscope_mcp.server as srv
    import pyscope_mcp._log as _log

    srv._INDEX_PATH = tmp_index
    srv._INDEX = None
    srv._SERVER._shutdown_requested = False
    _log._LOGGER = None
    yield
    srv._INDEX = None
    _log._LOGGER = None


@pytest.fixture()
def server(tmp_index: Path) -> RpcServer:
    import pyscope_mcp.server as srv
    srv._INDEX_PATH = tmp_index
    srv._INDEX = None
    return srv._SERVER


@pytest.fixture()
def log_path(tmp_index: Path) -> Path:
    return tmp_index.parent / "query.jsonl"


def _init_logger(log_path: Path) -> None:
    """Init the logger singleton pointing at log_path."""
    from pyscope_mcp import _log
    with patch.dict(os.environ, {"PYSCOPE_MCP_LOG": "1"}):
        _log.init(log_path)


def _read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# S1 — happy-path tool call (refers_to)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s1_happy_path_callers_of(server: RpcServer, tmp_index: Path, log_path: Path):
    """S1: refers_to call appends one valid JSONL entry."""
    import pyscope_mcp.server as srv
    srv._INDEX = CallGraphIndex.load(tmp_index)

    _init_logger(log_path)

    lines = [
        _req("tools/call", {"name": "refers_to", "arguments": {"fqn": "pkg.mod.bar", "kind": "callers"}}, req_id=7),
    ]
    responses = await _run(server, lines)

    # Tool must return a non-error result.
    assert responses[0]["result"]["isError"] is False

    entries = _read_log(log_path)
    assert len(entries) == 1, f"Expected 1 log entry, got {len(entries)}"
    e = entries[0]

    # Required fields.
    assert e["v"] == 1
    assert e["ts"]  # non-empty ISO string
    assert e["server_id"]  # non-empty UUID string
    assert e["rpc_id"] == 7
    assert e["tool"] == "refers_to"
    assert "fqn" in e["args"]
    assert isinstance(e["duration_ms"], int) and e["duration_ms"] >= 0
    assert e["is_error"] is False
    assert isinstance(e["truncated"], bool)
    assert isinstance(e["result_count"], int) and e["result_count"] >= 0
    assert e["edge_count"] is None
    assert e["hub_suppressed_count"] is None
    assert e["depth_full"] is None
    assert e["token_budget_used"] is None
    assert e["index_version"] == INDEX_VERSION
    # No full response payload in the entry.
    assert "results" not in e


# ---------------------------------------------------------------------------
# S2 — error-path tool call (unknown tool name)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s2_error_path_unknown_tool(server: RpcServer, tmp_index: Path, log_path: Path):
    """S2: unknown tool name logs is_error=true entry."""
    import pyscope_mcp.server as srv
    srv._INDEX = CallGraphIndex.load(tmp_index)

    _init_logger(log_path)

    lines = [
        _req("tools/call", {"name": "does_not_exist", "arguments": {}}, req_id=3),
    ]
    responses = await _run(server, lines)

    assert responses[0]["result"]["isError"] is True

    entries = _read_log(log_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["is_error"] is True
    assert e["error_msg"]  # non-empty
    assert e["tool"] == "does_not_exist"


# ---------------------------------------------------------------------------
# S3 — neighborhood with truncation and hub suppression
# ---------------------------------------------------------------------------

@pytest.fixture()
def hub_index(tmp_path: Path) -> Path:
    """Synthetic graph with a hub node (many in-edges) to trigger suppression."""
    # hub node: "pkg.util.common" called by 20 distinct callers → high in-degree
    raw: dict[str, list[str]] = {"pkg.util.common": []}
    for i in range(20):
        raw[f"pkg.agent.worker_{i}.run"] = ["pkg.util.common"]
    # Add the target symbol
    raw["pkg.mod.target"] = ["pkg.util.common", "pkg.agent.worker_0.run"]
    idx = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw))
    idx_path = tmp_path / ".pyscope-mcp" / "index.json"
    idx.save(idx_path)
    return idx_path


@pytest.mark.asyncio
async def test_s3_neighborhood_truncation_hub_suppression(tmp_path: Path, hub_index: Path):
    """S3: neighborhood with truncation/hub-suppression logs correct stats."""
    import pyscope_mcp.server as srv
    import pyscope_mcp._log as _log

    srv._INDEX_PATH = hub_index
    srv._INDEX = CallGraphIndex.load(hub_index)
    srv._SERVER._shutdown_requested = False

    log_path = hub_index.parent / "query.jsonl"
    with patch.dict(os.environ, {"PYSCOPE_MCP_LOG": "1"}):
        _log.init(log_path)

    lines = [
        # Very small token_budget to force truncation
        _req("tools/call", {
            "name": "neighborhood",
            "arguments": {"symbol": "pkg.mod.target", "depth": 2, "token_budget": 10},
        }, req_id=5),
    ]

    reader = FakeReader(lines)
    writer = FakeWriter()
    await srv._SERVER._loop(reader, writer)
    responses = writer.responses()

    assert responses[0]["result"]["isError"] is False
    payload = json.loads(responses[0]["result"]["content"][0]["text"])

    entries = _read_log(log_path)
    assert len(entries) == 1
    e = entries[0]

    assert e["tool"] == "neighborhood"
    # edge_count should match actual edges in payload
    assert e["edge_count"] == len(payload["edges"])
    # hub_suppressed_count should match
    assert e["hub_suppressed_count"] == len(payload["hub_suppressed"])
    # depth_full and token_budget_used present
    assert e["depth_full"] is not None
    assert e["token_budget_used"] is not None
    assert e["result_count"] is None  # neighborhood has no "results" key


# ---------------------------------------------------------------------------
# S4 — opt-out path (PYSCOPE_MCP_LOG=0)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s4_opt_out_no_file_created(server: RpcServer, tmp_index: Path, log_path: Path):
    """S4: PYSCOPE_MCP_LOG=0 — no log file written, tool response unchanged."""
    import pyscope_mcp.server as srv
    import pyscope_mcp._log as _log

    srv._INDEX = CallGraphIndex.load(tmp_index)

    with patch.dict(os.environ, {"PYSCOPE_MCP_LOG": "0"}):
        _log.init(log_path)

    assert _log._LOGGER is None

    lines = [
        _req("tools/call", {"name": "stats", "arguments": {}}, req_id=1),
    ]
    responses = await _run(server, lines)
    assert responses[0]["result"]["isError"] is False
    assert not log_path.exists()


# ---------------------------------------------------------------------------
# S5 — log rotation at 10 MB
# ---------------------------------------------------------------------------

def test_s5_log_rotation(tmp_path: Path):
    """S5: rotation at LOG_MAX_BYTES, max LOG_BACKUP_COUNT historical files."""
    from pyscope_mcp._log import LOG_MAX_BYTES, QueryLogger

    log_path = tmp_path / "query.jsonl"
    ql = QueryLogger(log_path)

    # Write a big chunk to simulate a file at the rotation threshold.
    big_line = "x" * (LOG_MAX_BYTES + 100)
    log_path.write_text(big_line, encoding="utf-8")

    # One more write triggers rotation.
    ql._append("new_entry")

    assert log_path.exists()
    assert (log_path.parent / "query.jsonl.1").exists()
    content = (log_path.parent / "query.jsonl.1").read_text(encoding="utf-8")
    assert big_line in content
    assert log_path.read_text(encoding="utf-8").strip() == "new_entry"


def test_s5_rotation_caps_at_backup_count(tmp_path: Path):
    """S5: oldest backup is deleted when exceeding LOG_BACKUP_COUNT rotations."""
    from pyscope_mcp._log import LOG_BACKUP_COUNT, LOG_MAX_BYTES, QueryLogger

    log_path = tmp_path / "query.jsonl"
    ql = QueryLogger(log_path)

    # Pre-create backup files .1 through .LOG_BACKUP_COUNT (already at limit).
    for i in range(1, LOG_BACKUP_COUNT + 1):
        backup = log_path.with_suffix(log_path.suffix + f".{i}")
        backup.write_text(f"backup_{i}", encoding="utf-8")

    # Fill the main log to trigger rotation.
    log_path.write_text("x" * (LOG_MAX_BYTES + 100), encoding="utf-8")
    ql._append("latest")

    # .6 should NOT exist
    too_old = log_path.with_suffix(log_path.suffix + f".{LOG_BACKUP_COUNT + 1}")
    assert not too_old.exists()
    # .1 should be the renamed current log
    assert (log_path.with_suffix(log_path.suffix + ".1")).exists()
    # Only LOG_BACKUP_COUNT backups remain (no +1)
    backups = list(tmp_path.glob("query.jsonl.*"))
    assert len(backups) <= LOG_BACKUP_COUNT


# ---------------------------------------------------------------------------
# S6 — reload path: index identity updates after reload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s6_reload_updates_index_identity(tmp_path: Path):
    """S6: entries after reload carry new index identity; prior entries unchanged."""
    import pyscope_mcp.server as srv
    import pyscope_mcp._log as _log

    # Build two different indexes.
    raw_a = {"pkg.a.foo": ["pkg.a.bar"], "pkg.a.bar": []}
    raw_b = {"pkg.b.hello": ["pkg.b.world"], "pkg.b.world": []}
    idx_a = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw_a))
    idx_b = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw_b))
    idx_path = tmp_path / ".pyscope-mcp" / "index.json"
    idx_a.save(idx_path)

    srv._INDEX_PATH = idx_path
    srv._INDEX = CallGraphIndex.load(idx_path)
    hash_a = srv._INDEX.content_hash

    log_path = idx_path.parent / "query.jsonl"
    with patch.dict(os.environ, {"PYSCOPE_MCP_LOG": "1"}):
        _log.init(log_path)

    # First call (index A).
    lines_pre = [
        _req("tools/call", {"name": "stats", "arguments": {}}, req_id=1),
    ]
    reader = FakeReader(lines_pre)
    writer = FakeWriter()
    await srv._SERVER._loop(reader, writer)

    # Switch to index B on disk and reload.
    idx_b.save(idx_path)
    hash_b = idx_b.content_hash

    lines_reload = [
        _req("tools/call", {"name": "reload", "arguments": {}}, req_id=2),
        _req("tools/call", {"name": "stats", "arguments": {}}, req_id=3),
    ]
    reader = FakeReader(lines_reload)
    writer = FakeWriter()
    await srv._SERVER._loop(reader, writer)

    entries = _read_log(log_path)
    assert len(entries) == 3

    # First entry uses hash_a.
    assert entries[0]["index_content_hash"] == hash_a
    # reload entry uses hash_b (new index already loaded by _dispatch_tool).
    assert entries[1]["tool"] == "reload"
    assert entries[1]["index_content_hash"] == hash_b
    # Third entry also uses hash_b.
    assert entries[2]["index_content_hash"] == hash_b


# ---------------------------------------------------------------------------
# S7 — logger-failure path (internal error does not affect tool response)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s7_logger_failure_does_not_affect_tool_response(
    server: RpcServer, tmp_index: Path, log_path: Path, caplog
):
    """S7: when the logger's internal _append raises, tool returns normally and warning emitted."""
    import pyscope_mcp.server as srv
    import pyscope_mcp._log as _log

    srv._INDEX = CallGraphIndex.load(tmp_index)

    _init_logger(log_path)

    # Patch _append (the I/O method) to raise, while leaving write()'s exception handler intact.
    def _boom_append(self, line: str):
        raise OSError("simulated disk full")

    import logging
    with patch.object(_log.QueryLogger, "_append", _boom_append):
        with caplog.at_level(logging.WARNING, logger="pyscope_mcp._log"):
            lines = [
                _req("tools/call", {"name": "stats", "arguments": {}}, req_id=1),
            ]
            responses = await _run(server, lines)

    # Tool must still return a valid result — logger failure must not propagate.
    assert responses[0]["result"]["isError"] is False
    # At least one warning was emitted via the logging infrastructure.
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_records, "Expected at least one warning from QueryLogger"


# ---------------------------------------------------------------------------
# S8 — index schema v5 round-trip
# ---------------------------------------------------------------------------

def test_s8_v5_roundtrip_git_sha_content_hash(tmp_path: Path):
    """S8: save/load round-trips git_sha and content_hash; v4 load fails."""
    raw = {"pkg.mod.foo": ["pkg.mod.bar"], "pkg.mod.bar": []}
    fake_sha = "a1b2c3d4e5f6789012345678901234567890abcd"

    idx = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw), git_sha=fake_sha)
    assert idx.git_sha == fake_sha
    assert len(idx.content_hash) == 64  # SHA-256 hex

    idx_path = tmp_path / "index.json"
    idx.save(idx_path)

    payload = json.loads(idx_path.read_text())
    assert payload["version"] == INDEX_VERSION
    assert payload["git_sha"] == fake_sha
    assert payload["content_hash"] == idx.content_hash

    loaded = CallGraphIndex.load(idx_path)
    assert loaded.git_sha == fake_sha
    assert loaded.content_hash == idx.content_hash


def test_s8_content_hash_is_deterministic(tmp_path: Path):
    """S8: two builds against the same raw dict produce the same content_hash."""
    raw = {"pkg.mod.foo": ["pkg.mod.bar", "pkg.mod.baz"], "pkg.mod.bar": [], "pkg.mod.baz": []}
    idx1 = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw))
    idx2 = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw))
    assert idx1.content_hash == idx2.content_hash
    assert len(idx1.content_hash) == 64


def test_s8_non_git_build_produces_none_git_sha(tmp_path: Path):
    """S8: building without a git_sha produces git_sha=None; load/serve still work."""
    raw = {"pkg.mod.foo": []}
    idx = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw), git_sha=None)
    assert idx.git_sha is None
    idx_path = tmp_path / "index.json"
    idx.save(idx_path)

    loaded = CallGraphIndex.load(idx_path)
    assert loaded.git_sha is None
    assert loaded.stats()["functions"] >= 1


def test_s8_v4_index_fails_to_load(tmp_path: Path):
    """S8: loading a v4 index raises ValueError with clear message."""
    # Write a manually crafted v4 index blob.
    raw = {"pkg.mod.foo": []}
    v4_payload = {
        "version": 4,
        "root": str(tmp_path),
        "raw": raw,
        "skeletons": {},
        "file_shas": {},
        "missed_callers": {},
    }
    idx_path = tmp_path / "index.json"
    idx_path.write_text(json.dumps(v4_payload))

    with pytest.raises(ValueError, match="v4"):
        CallGraphIndex.load(idx_path)


def test_s8_different_raw_produces_different_hash(tmp_path: Path):
    """S8: different raw dicts produce different content hashes."""
    raw_a = {"pkg.a.foo": ["pkg.a.bar"], "pkg.a.bar": []}
    raw_b = {"pkg.b.hello": ["pkg.b.world"], "pkg.b.world": []}
    idx_a = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw_a))
    idx_b = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw_b))
    assert idx_a.content_hash != idx_b.content_hash


# ---------------------------------------------------------------------------
# S9 — default-on behaviour (PYSCOPE_MCP_LOG unset → logger enabled)
# ---------------------------------------------------------------------------

def test_s9_default_on_when_env_unset(monkeypatch, tmp_path: Path):
    """S9: PYSCOPE_MCP_LOG unset — logger is enabled (default-on)."""
    from pyscope_mcp import _log

    # Remove PYSCOPE_MCP_LOG from the environment (if inherited from the shell).
    monkeypatch.delenv("PYSCOPE_MCP_LOG", raising=False)

    log_path = tmp_path / "query.jsonl"
    _log.init(log_path)

    assert _log._LOGGER is not None, (
        "default-on broken: PYSCOPE_MCP_LOG unset should activate the logger"
    )


def test_s9_init_emits_warning_with_log_path(monkeypatch, caplog, tmp_path: Path):
    """S9: on activation, init emits a WARNING-level message with the log path.

    The WARNING level matters: Python's ``logging.lastResort`` handler routes
    WARNING+ to stderr by default, so users see the announcement even before
    ``_rpc.RpcServer.run()`` configures its own handlers.
    """
    import logging

    from pyscope_mcp import _log

    monkeypatch.delenv("PYSCOPE_MCP_LOG", raising=False)

    log_path = tmp_path / "query.jsonl"
    with caplog.at_level(logging.WARNING, logger="pyscope_mcp._log"):
        _log.init(log_path)

    assert _log._LOGGER is not None
    matching = [
        rec for rec in caplog.records
        if rec.levelno == logging.WARNING and "Query logging enabled" in rec.getMessage()
    ]
    assert matching, (
        f"expected one WARNING-level 'Query logging enabled' record, "
        f"got: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    # The active log path must be in the announcement so users can find the file.
    assert any(str(log_path) in rec.getMessage() for rec in matching), (
        f"WARNING did not include the log path {log_path}: "
        f"{[r.getMessage() for r in matching]}"
    )


def test_s9_explicit_zero_overrides_default(monkeypatch, tmp_path: Path):
    """S9: PYSCOPE_MCP_LOG=0 still disables, even with the default flipped to on."""
    from pyscope_mcp import _log

    monkeypatch.setenv("PYSCOPE_MCP_LOG", "0")

    log_path = tmp_path / "query.jsonl"
    _log.init(log_path)

    assert _log._LOGGER is None
    assert not log_path.exists()
