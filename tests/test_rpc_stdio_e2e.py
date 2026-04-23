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


@pytest.fixture
def index_path(tmp_path: Path) -> Path:
    """Build a real index of this repo and return its path."""
    repo_root = Path(__file__).resolve().parents[1]
    out = tmp_path / "index.json"
    env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))
    subprocess.run(
        [
            sys.executable, "-m", "pyscope_mcp.cli", "build",
            "--root", str(repo_root),
            "--output", str(out),
        ],
        env=env, check=True, capture_output=True,
    )
    assert out.exists()
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
        assert len(tools) == 7
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
