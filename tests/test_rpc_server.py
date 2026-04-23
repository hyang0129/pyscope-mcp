"""Tests for the hand-rolled JSON-RPC 2.0 stdio transport (_rpc.py + server.py).

Test strategy: feed JSON lines directly into the RPC loop via asyncio.Queue-backed
streams to avoid subprocess overhead while still exercising the full dispatch path.

The harness wires:
  input_queue  →  FakeReader  →  RpcServer._loop  →  FakeWriter  →  output_list

Each test sends one or more JSON-RPC messages and inspects the responses written
to the output list.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pyscope_mcp._rpc import (
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    RpcServer,
)
from pyscope_mcp.graph import CallGraphIndex


# ---------------------------------------------------------------------------
# In-process pipe harness
# ---------------------------------------------------------------------------

class FakeReader:
    """Feeds pre-encoded lines to the RPC loop as if they came from stdin."""

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
    """Captures bytes written by the RPC loop."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        pass

    def responses(self) -> list[dict]:
        """Parse all written chunks as JSON objects."""
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


def _notif(method: str, params: Any = None) -> bytes:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return _line(msg)


async def _run(server: RpcServer, lines: list[bytes]) -> list[dict]:
    reader = FakeReader(lines)
    writer = FakeWriter()
    await server._loop(reader, writer)
    return writer.responses()


# ---------------------------------------------------------------------------
# Minimal in-memory index fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_index(tmp_path: Path) -> Path:
    """Create a minimal CallGraphIndex on disk for testing."""
    raw: dict[str, list[str]] = {
        "pkg.mod.foo": ["pkg.mod.bar"],
        "pkg.mod.bar": [],
        "pkg.other.baz": ["pkg.mod.foo"],
    }
    idx = CallGraphIndex.from_raw(tmp_path, raw)
    idx_path = tmp_path / "index.json"
    idx.save(idx_path)
    return idx_path


@pytest.fixture()
def server(tmp_index: Path) -> RpcServer:
    """A server pre-wired with the in-memory index."""
    # Import server module to register all tool handlers
    import pyscope_mcp.server as srv
    srv._INDEX_PATH = tmp_index
    srv._INDEX = None  # force lazy reload
    return srv._SERVER


# ---------------------------------------------------------------------------
# Helpers to reset module-level state between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_server_state(tmp_index: Path):
    import pyscope_mcp.server as srv
    srv._INDEX_PATH = tmp_index
    srv._INDEX = None
    srv._SERVER._initialized = False
    srv._SERVER._shutdown_requested = False
    yield
    srv._INDEX = None


# ---------------------------------------------------------------------------
# Golden handshake
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_golden_handshake(server: RpcServer, tmp_index: Path):
    """initialize → notifications/initialized → tools/list returns all 7 tools."""
    lines = [
        _req("initialize", {"protocolVersion": "2025-03-26", "clientInfo": {"name": "test"}}, req_id=1),
        _notif("notifications/initialized"),
        _req("tools/list", req_id=2),
    ]
    responses = await _run(server, lines)

    # initialize response
    init_resp = responses[0]
    assert init_resp["id"] == 1
    assert init_resp["result"]["protocolVersion"] == "2025-03-26"
    assert init_resp["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert init_resp["result"]["serverInfo"]["name"] == "pyscope-mcp"

    # tools/list response (no response for notification)
    list_resp = responses[1]
    assert list_resp["id"] == 2
    tools = list_resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert tool_names == {"stats", "reload", "callers_of", "callees_of", "module_callers", "module_callees", "search"}
    assert len(tools) == 7


@pytest.mark.asyncio
async def test_tool_annotations(server: RpcServer):
    """All tools include annotations; readOnlyHint correct per tool."""
    lines = [_req("tools/list", req_id=1)]
    responses = await _run(server, lines)
    tools = {t["name"]: t for t in responses[0]["result"]["tools"]}

    for name in ("stats", "callers_of", "callees_of", "module_callers", "module_callees", "search"):
        ann = tools[name]["annotations"]
        assert ann["readOnlyHint"] is True, f"{name} should be readOnly"
        assert ann["idempotentHint"] is True

    reload_ann = tools["reload"]["annotations"]
    assert reload_ann["readOnlyHint"] is False
    assert reload_ann["idempotentHint"] is True


# ---------------------------------------------------------------------------
# Individual tool happy-path calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_stats(server: RpcServer):
    lines = [_req("tools/call", {"name": "stats", "arguments": {}}, req_id=1)]
    responses = await _run(server, lines)
    r = responses[0]["result"]
    assert r["isError"] is False
    payload = json.loads(r["content"][0]["text"])
    assert "functions" in payload
    assert "modules" in payload


@pytest.mark.asyncio
async def test_tool_callers_of(server: RpcServer):
    lines = [_req("tools/call", {"name": "callers_of", "arguments": {"fqn": "pkg.mod.bar"}}, req_id=1)]
    responses = await _run(server, lines)
    r = responses[0]["result"]
    assert r["isError"] is False


@pytest.mark.asyncio
async def test_tool_callees_of(server: RpcServer):
    lines = [_req("tools/call", {"name": "callees_of", "arguments": {"fqn": "pkg.mod.foo"}}, req_id=1)]
    responses = await _run(server, lines)
    r = responses[0]["result"]
    assert r["isError"] is False


@pytest.mark.asyncio
async def test_tool_module_callers(server: RpcServer):
    lines = [_req("tools/call", {"name": "module_callers", "arguments": {"module": "pkg.mod"}}, req_id=1)]
    responses = await _run(server, lines)
    r = responses[0]["result"]
    assert r["isError"] is False


@pytest.mark.asyncio
async def test_tool_module_callees(server: RpcServer):
    lines = [_req("tools/call", {"name": "module_callees", "arguments": {"module": "pkg.mod"}}, req_id=1)]
    responses = await _run(server, lines)
    r = responses[0]["result"]
    assert r["isError"] is False


@pytest.mark.asyncio
async def test_tool_search(server: RpcServer):
    lines = [_req("tools/call", {"name": "search", "arguments": {"query": "foo"}}, req_id=1)]
    responses = await _run(server, lines)
    r = responses[0]["result"]
    assert r["isError"] is False


@pytest.mark.asyncio
async def test_tool_reload(server: RpcServer):
    lines = [_req("tools/call", {"name": "reload", "arguments": {}}, req_id=1)]
    responses = await _run(server, lines)
    r = responses[0]["result"]
    assert r["isError"] is False
    payload = json.loads(r["content"][0]["text"])
    assert "functions" in payload


# ---------------------------------------------------------------------------
# Invalid params — missing required args
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_callers_of_missing_fqn(server: RpcServer):
    """Missing required 'fqn' returns isError:true, not a JSON-RPC error."""
    lines = [_req("tools/call", {"name": "callers_of", "arguments": {}}, req_id=1)]
    responses = await _run(server, lines)
    r = responses[0]
    # Should be a successful RPC response with isError=true
    assert "result" in r
    assert r["result"]["isError"] is True


@pytest.mark.asyncio
async def test_callees_of_missing_fqn(server: RpcServer):
    lines = [_req("tools/call", {"name": "callees_of", "arguments": {}}, req_id=1)]
    responses = await _run(server, lines)
    assert responses[0]["result"]["isError"] is True


@pytest.mark.asyncio
async def test_search_missing_query(server: RpcServer):
    lines = [_req("tools/call", {"name": "search", "arguments": {}}, req_id=1)]
    responses = await _run(server, lines)
    assert responses[0]["result"]["isError"] is True


# ---------------------------------------------------------------------------
# Unknown tool name → isError:true (NOT a JSON-RPC error)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_tool_name(server: RpcServer):
    lines = [_req("tools/call", {"name": "does_not_exist", "arguments": {}}, req_id=1)]
    responses = await _run(server, lines)
    r = responses[0]
    assert "result" in r, "Should return a result, not an error"
    assert r["result"]["isError"] is True
    assert "unknown tool" in r["result"]["content"][0]["text"]


# ---------------------------------------------------------------------------
# Handler raises → isError:true
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handler_raises(server: RpcServer):
    """A tool handler that raises an exception surfaces as isError:true."""
    import pyscope_mcp.server as srv

    async def _exploding(name, arguments):
        raise RuntimeError("boom from handler")

    with patch.object(srv, "_dispatch_tool", _exploding):
        lines = [_req("tools/call", {"name": "stats", "arguments": {}}, req_id=1)]
        responses = await _run(server, lines)
    r = responses[0]
    assert "result" in r
    assert r["result"]["isError"] is True
    assert "boom from handler" in r["result"]["content"][0]["text"]


# ---------------------------------------------------------------------------
# JSON-RPC error cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_malformed_json_parse_error(server: RpcServer):
    """A malformed JSON line returns -32700 and the loop continues."""
    lines = [
        b"this is not json",
        _req("ping", req_id=99),
    ]
    responses = await _run(server, lines)
    assert responses[0]["error"]["code"] == PARSE_ERROR
    # loop continues — ping responds
    assert responses[1]["id"] == 99
    assert responses[1]["result"] == {}


@pytest.mark.asyncio
async def test_unknown_method(server: RpcServer):
    """An unknown method returns -32601."""
    lines = [_req("no_such_method", req_id=1)]
    responses = await _run(server, lines)
    assert responses[0]["error"]["code"] == METHOD_NOT_FOUND


@pytest.mark.asyncio
async def test_batch_request_rejected(server: RpcServer):
    """A JSON array (batch) is rejected with -32600."""
    lines = [
        _line([
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        ])
    ]
    responses = await _run(server, lines)
    assert responses[0]["error"]["code"] == INVALID_REQUEST


@pytest.mark.asyncio
async def test_missing_jsonrpc_field(server: RpcServer):
    """Request missing 'jsonrpc' field returns -32600."""
    lines = [_line({"id": 1, "method": "ping"})]
    responses = await _run(server, lines)
    assert responses[0]["error"]["code"] == INVALID_REQUEST


# ---------------------------------------------------------------------------
# EOF — clean exit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_eof_clean_exit(server: RpcServer):
    """Empty input (EOF) exits without error."""
    responses = await _run(server, [])
    assert responses == []


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ping(server: RpcServer):
    lines = [_req("ping", req_id=42)]
    responses = await _run(server, lines)
    assert responses[0]["result"] == {}
    assert responses[0]["id"] == 42


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shutdown(server: RpcServer):
    lines = [_req("shutdown", req_id=1)]
    responses = await _run(server, lines)
    assert responses[0]["result"] == {}


# ---------------------------------------------------------------------------
# initialize — protocol version negotiation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_initialize_known_version(server: RpcServer):
    """Known versions are echoed back."""
    for version in ("2024-11-05", "2025-03-26", "2025-06-18"):
        server._initialized = False
        lines = [_req("initialize", {"protocolVersion": version}, req_id=1)]
        responses = await _run(server, lines)
        assert responses[0]["result"]["protocolVersion"] == version


@pytest.mark.asyncio
async def test_initialize_unknown_version_fallback(server: RpcServer):
    """Unknown version falls back to newest."""
    server._initialized = False
    lines = [_req("initialize", {"protocolVersion": "2099-01-01"}, req_id=1)]
    responses = await _run(server, lines)
    assert responses[0]["result"]["protocolVersion"] == "2025-06-18"


# ---------------------------------------------------------------------------
# Notifications produce no response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notification_no_response(server: RpcServer):
    """Notification messages (no id) must not produce any response."""
    lines = [
        _notif("notifications/initialized"),
        _notif("notifications/cancelled"),
    ]
    responses = await _run(server, lines)
    assert responses == []


# ---------------------------------------------------------------------------
# stdout hygiene — no print() in server module
# ---------------------------------------------------------------------------

def _code_lines_with_print(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, line) for any non-comment, non-docstring line containing print(."""
    source = path.read_text()
    result = []
    in_docstring = False
    docstring_char = None
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        # Toggle docstring tracking
        for marker in ('"""', "'''"):
            count = line.count(marker)
            if not in_docstring and count >= 1:
                in_docstring = True
                docstring_char = marker
                if count >= 2:  # opened and closed on same line
                    in_docstring = False
                    docstring_char = None
                break
            elif in_docstring and docstring_char == marker and count >= 1:
                in_docstring = False
                docstring_char = None
                break
        if in_docstring:
            continue
        # Skip comment-only lines
        if stripped.startswith("#"):
            continue
        # Skip lines where print( only appears inside a string literal or comment
        # Simple heuristic: skip if the word before print( is inside quotes
        if "print(" in line:
            # Check if it appears as an actual call (not in a string or comment)
            # Strip inline comments first
            code_part = line.split("#")[0]
            if "print(" in code_part:
                result.append((i, line))
    return result


def test_no_print_in_server_module():
    """server.py must not contain any print() calls outside comments/strings."""
    server_path = Path(__file__).parent.parent / "src" / "pyscope_mcp" / "server.py"
    bad = _code_lines_with_print(server_path)
    assert bad == [], (
        "server.py contains print() calls (stdout hygiene violation):\n"
        + "\n".join(f"  line {ln}: {txt}" for ln, txt in bad)
    )


def test_no_print_in_rpc_module():
    """_rpc.py must not contain any print() calls outside comments/strings."""
    rpc_path = Path(__file__).parent.parent / "src" / "pyscope_mcp" / "_rpc.py"
    bad = _code_lines_with_print(rpc_path)
    assert bad == [], (
        "_rpc.py contains print() calls (stdout hygiene violation):\n"
        + "\n".join(f"  line {ln}: {txt}" for ln, txt in bad)
    )


# ---------------------------------------------------------------------------
# stdout hygiene — subprocess check
# ---------------------------------------------------------------------------

def test_stdout_hygiene_subprocess(tmp_index: Path, tmp_path: Path):
    """Server stdout must contain only valid JSON-RPC frames even when handler logs."""
    import subprocess

    # Write a tiny driver script that feeds one message and captures stdout/stderr
    driver = tmp_path / "driver.py"
    driver.write_text(
        f"""\
import sys, json, os

# Patch _INDEX_PATH before importing server
import pyscope_mcp.server as srv
from pathlib import Path
srv._INDEX_PATH = Path({str(tmp_index)!r})
srv._INDEX = None

import asyncio
from pyscope_mcp._rpc import RpcServer

async def main():
    import logging
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    # Send a tools/call for stats
    lines_in = [
        json.dumps({{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                     "params": {{"name": "stats", "arguments": {{}}}}}}).encode(),
    ]

    class FakeReader:
        def __init__(self, lines):
            self._lines = lines
            self._pos = 0
        def __aiter__(self): return self
        async def __anext__(self):
            if self._pos >= len(self._lines):
                raise StopAsyncIteration
            line = self._lines[self._pos]
            self._pos += 1
            return line + b"\\n"

    class FakeWriter:
        def write(self, data): sys.stdout.buffer.write(data); sys.stdout.buffer.flush()
        async def drain(self): pass

    await srv._SERVER._loop(FakeReader(lines_in), FakeWriter())

asyncio.run(main())
"""
    )
    result = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,
        timeout=10,
    )
    stdout = result.stdout.strip()
    assert stdout, "Server produced no stdout"
    # Every line of stdout must be valid JSON
    for line in stdout.splitlines():
        obj = json.loads(line)
        assert "jsonrpc" in obj or "result" in obj or "error" in obj


# ---------------------------------------------------------------------------
# Cold import budget
# ---------------------------------------------------------------------------

def test_cold_import_budget():
    """'import pyscope_mcp.server' must complete in under 500 ms."""
    import subprocess

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", "import pyscope_mcp.server"],
        capture_output=True,
        timeout=10,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result.returncode == 0, f"import failed: {result.stderr.decode()}"
    assert elapsed_ms < 500, (
        f"Cold import took {elapsed_ms:.0f} ms — exceeds 500 ms budget. "
        "Check that no heavy dependencies are imported at module level."
    )
