"""End-to-end test: spawn the CLI `serve` subprocess and exchange multiple
JSON-RPC requests over real OS pipes.

Unit tests in `test_rpc_server.py` use in-memory StreamReader/StreamWriter
mocks and never exercise `loop.connect_write_pipe` / StreamWriter.drain()
against the real stdio fds. A bug in the pipe/protocol wiring (e.g. passing
asyncio.BaseProtocol instead of FlowControlMixin) is invisible to those
tests but breaks any real MCP client on the second request. This test
exchanges several requests to catch that class of regression.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _send(proc: subprocess.Popen, msg: dict) -> None:
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(msg) + "\n").encode())
    proc.stdin.flush()


def _recv(proc: subprocess.Popen) -> dict:
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read().decode() if proc.stderr else ""
        raise AssertionError(f"server produced no stdout; stderr={err!r}")
    return json.loads(line.decode())


@pytest.fixture(scope="session")
def index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny synthetic index on disk.

    These tests exercise the stdio wire contract (framing, drain, stdout
    hygiene, lifecycle) — not the analyzer or query correctness. A minimal
    in-memory index is sufficient and keeps e2e wall-clock in the single
    seconds rather than the ~100 s a real analyzer build would take.
    """
    from pyscope_mcp.graph import CallGraphIndex

    raw: dict[str, list[str]] = {
        "pkg.mod.foo": ["pkg.mod.bar"],
        "pkg.mod.bar": [],
    }
    out_dir = tmp_path_factory.mktemp("pyscope_idx")
    idx = CallGraphIndex.from_raw(out_dir, raw)
    out = out_dir / "index.json"
    idx.save(out)
    return out


def test_stdio_multiple_requests(index_path: Path) -> None:
    """Regression: exchange ≥ 2 requests over real pipes.

    If StreamWriter.drain() fails after the first response (e.g. because the
    write protocol lacks _drain_helper), the server will hang or crash on the
    second request.
    """
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "pyscope_mcp.cli", "serve",
            "--root", str(repo_root),
            "--index", str(index_path),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )
    try:
        _send(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e", "version": "0"},
            },
        })
        r1 = _recv(proc)
        assert r1["id"] == 1
        assert r1["result"]["protocolVersion"] == "2024-11-05"
        assert r1["result"]["serverInfo"]["name"] == "pyscope-mcp"

        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        # Second request — this is the one that failed when drain() broke.
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        r2 = _recv(proc)
        assert r2["id"] == 2
        tools = r2["result"]["tools"]
        assert len(tools) == 8
        assert all("name" in t and "inputSchema" in t for t in tools)

        # Third request — ping.
        _send(proc, {"jsonrpc": "2.0", "id": 3, "method": "ping"})
        r3 = _recv(proc)
        assert r3 == {"jsonrpc": "2.0", "id": 3, "result": {}}

        # Unknown method → JSON-RPC error, server stays up.
        _send(proc, {"jsonrpc": "2.0", "id": 4, "method": "does/not/exist"})
        r4 = _recv(proc)
        assert r4["error"]["code"] == -32601

        # Tool failure → isError, not JSON-RPC error.
        _send(proc, {
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "no_such_tool", "arguments": {}},
        })
        r5 = _recv(proc)
        assert "error" not in r5
        assert r5["result"]["isError"] is True

        # Shutdown and clean exit on stdin EOF.
        _send(proc, {"jsonrpc": "2.0", "id": 99, "method": "shutdown"})
        r6 = _recv(proc)
        assert r6 == {"jsonrpc": "2.0", "id": 99, "result": {}}

        assert proc.stdin is not None
        proc.stdin.close()
        assert proc.wait(timeout=5) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)


# ---------------------------------------------------------------------------
# PR #41 review follow-ups — stdout hygiene under real stdio + clean shutdown
# ---------------------------------------------------------------------------

def _driver_source(index_path: Path, trigger: str) -> str:
    """Build a tiny subprocess driver that registers a trigger tool.

    `trigger` is Python code executed inside the tool handler body. It can
    call logging.warning(), print(), etc. — the point is to verify that any
    such writes do NOT corrupt stdout.

    The driver reads one JSON-RPC line from stdin and runs the RPC loop to
    EOF, writing responses to real stdout (via connect_write_pipe in
    RpcServer.run()).
    """
    return f"""\
import asyncio, sys, logging
from pathlib import Path

import pyscope_mcp.server as srv

srv._INDEX_PATH = Path({str(index_path)!r})
srv._INDEX = None

# Register a test-only tool that invokes the caller-provided trigger.
_original_tool_names = set(srv._TOOL_NAMES)
srv._TOOL_NAMES.add("trigger")
srv._TOOL_LIST.append({{
    "name": "trigger",
    "description": "Test-only handler.",
    "inputSchema": {{"type": "object", "properties": {{}}}},
    "annotations": {{}},
}})

_original_dispatch = srv._dispatch_tool
async def _patched(name, arguments):
    if name == "trigger":
{trigger}
        return {{"content": [{{"type": "text", "text": "ok"}}], "isError": False}}
    return await _original_dispatch(name, arguments)
srv._dispatch_tool = _patched

asyncio.run(srv._SERVER.run())
"""


def test_stdout_hygiene_with_handler_log(index_path: Path, tmp_path: Path) -> None:
    """A handler that logs must not corrupt stdout; log line must appear on stderr."""
    driver = tmp_path / "driver_log.py"
    driver.write_text(_driver_source(
        index_path,
        trigger='        logging.warning("stderr please %s", "yes")',
    ))
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))

    proc = subprocess.Popen(
        [sys.executable, str(driver)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )
    try:
        # Handshake then trigger the logging handler, then shut down.
        _send(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "e2e", "version": "0"}},
        })
        r1 = _recv(proc)
        assert r1["id"] == 1
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send(proc, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "trigger", "arguments": {}},
        })
        r2 = _recv(proc)
        assert r2["id"] == 2
        assert r2["result"]["isError"] is False

        assert proc.stdin is not None
        proc.stdin.close()
        stdout_rest = proc.stdout.read() if proc.stdout else b""
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)

    # Every non-empty stdout line must parse as JSON (we already consumed two
    # responses above via _recv; any remainder must still be frame-clean).
    for raw in stdout_rest.splitlines():
        if raw.strip():
            json.loads(raw)  # raises if the stream got corrupted

    # And the log line must be on stderr, not stdout.
    assert "stderr please yes" in stderr, f"log not on stderr; stderr={stderr!r}"


def test_print_in_handler_goes_to_stderr(index_path: Path, tmp_path: Path) -> None:
    """print() in a handler must land on stderr (stdout replacement in run())."""
    driver = tmp_path / "driver_print.py"
    driver.write_text(_driver_source(
        index_path,
        trigger='        print("oops from handler")',
    ))
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))

    proc = subprocess.Popen(
        [sys.executable, str(driver)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )
    try:
        _send(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "e2e", "version": "0"}},
        })
        _recv(proc)
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send(proc, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "trigger", "arguments": {}},
        })
        r2 = _recv(proc)
        assert r2["id"] == 2

        assert proc.stdin is not None
        proc.stdin.close()
        stdout_rest = proc.stdout.read() if proc.stdout else b""
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)

    for raw in stdout_rest.splitlines():
        if raw.strip():
            json.loads(raw)
    assert "oops from handler" in stderr, f"print did not reach stderr; stderr={stderr!r}"
    # And critically: stdout must not contain the raw print text.
    assert b"oops from handler" not in stdout_rest


def test_shutdown_then_eof_exits_zero(index_path: Path) -> None:
    """Full lifecycle: shutdown → stdin EOF → exit 0."""
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "pyscope_mcp.cli", "serve",
            "--root", str(repo_root),
            "--index", str(index_path),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "shutdown"})
        r = _recv(proc)
        assert r == {"jsonrpc": "2.0", "id": 1, "result": {}}
        assert proc.stdin is not None
        proc.stdin.close()
        assert proc.wait(timeout=5) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
