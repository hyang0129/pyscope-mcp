"""Integration tests for issue #96 — deferred index initialisation.

Two scenarios at integration tier (real subprocess, real OS pipes):

  slice-deferred-init-stale-schema-e2e (wiring):
    Verifies that `pyscope-mcp serve` with a stale-schema (v0) index does NOT
    crash before registering tools. The server starts, responds to the MCP
    initialize handshake, returns all 9 tools in tools/list, and returns
    isError:true with a "pyscope-mcp build" message for any index-dependent
    tool call. The server remains alive for subsequent requests.

  slice-deferred-init-missing-index-e2e (wiring):
    Verifies that `pyscope-mcp serve` with a missing index path does NOT exit
    with code 2 (the pre-#96 behaviour). The server starts, registers all 9
    tools, and returns isError:true with "pyscope-mcp build" for index-dependent
    tools. Clean shutdown is possible.

Pattern: real subprocess, real OS pipes, stdio JSON-RPC 2.0.
Mirrors test_integration_issue71.py conventions (same _send/_recv helpers,
same _handshake/_shutdown/_spawn_server pattern, same marker discipline).

No artifact tier: the deferred-error output is structural (isError flag, text
message substring), not an exact-value golden fixture — classifying this scenario
as volatile_output=true.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers (mirrors test_integration_issue71.py)
# ---------------------------------------------------------------------------


def _send(proc: subprocess.Popen, msg: dict) -> None:
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(msg) + "\n").encode())
    proc.stdin.flush()


def _recv(proc: subprocess.Popen, timeout: float = 10.0) -> dict:
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read().decode() if proc.stderr else ""
        raise AssertionError(f"server produced no stdout; stderr={err!r}")
    return json.loads(line.decode())


def _handshake(proc: subprocess.Popen, client_name: str = "it96-test") -> None:
    """Send MCP initialize + initialized notification; assert initialize OK."""
    _send(proc, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": client_name, "version": "0"},
        },
    })
    r = _recv(proc)
    assert r["id"] == 1
    assert r["result"]["protocolVersion"] == "2024-11-05"
    _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})


def _shutdown(proc: subprocess.Popen, req_id: int = 99) -> None:
    """Send shutdown + close stdin; assert server exits 0."""
    _send(proc, {"jsonrpc": "2.0", "id": req_id, "method": "shutdown"})
    r = _recv(proc)
    assert r == {"jsonrpc": "2.0", "id": req_id, "result": {}}
    assert proc.stdin is not None
    proc.stdin.close()
    assert proc.wait(timeout=5) == 0


def _spawn_server(index_path: Path, root: Path | None = None) -> subprocess.Popen:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))
    args = [
        sys.executable, "-m", "pyscope_mcp.cli", "serve",
        "--root", str(root or repo_root),
        "--index", str(index_path),
    ]
    return subprocess.Popen(
        args,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )


# ---------------------------------------------------------------------------
# Scenario: slice-deferred-init-stale-schema-e2e
# SCENARIO: slice-deferred-init-stale-schema-e2e
# layers_involved: src/pyscope_mcp/server.py, src/pyscope_mcp/cli.py
# ---------------------------------------------------------------------------


class TestDeferredInitStaleSchemaWiring:
    """Wiring tier — slice-deferred-init-stale-schema-e2e.

    Verifies:
    - server starts (does not crash) when given a v0 (stale-schema) index
    - MCP initialize handshake succeeds; serverInfo.name == 'pyscope-mcp'
    - tools/list returns all 9 tools in deferred-error state
    - stats returns isError:true with 'pyscope-mcp build' in the message
    - server remains alive after a deferred-error tool call (ping works)
    - clean shutdown returns {} and server exits 0
    """

    @pytest.fixture(autouse=True)
    def stale_index(self, tmp_path: Path) -> Path:
        """Write a v0 (stale-schema) index to a temp directory."""
        stale_dir = tmp_path / ".pyscope-mcp"
        stale_dir.mkdir(parents=True)
        idx_path = stale_dir / "index.json"
        idx_path.write_text('{"version": 0, "root": "/", "raw": {}}')
        self._stale_index_path = idx_path
        self._tmp_path = tmp_path
        return idx_path

    @pytest.mark.integration_wiring
    def test_server_starts_with_stale_schema_index(self) -> None:
        """Server must not crash before the MCP handshake when index schema is stale."""
        proc = _spawn_server(self._stale_index_path, root=self._tmp_path)
        try:
            _handshake(proc, client_name="it96-stale-startup")
            # Success: handshake completed — server started and is responsive
            assert proc.poll() is None, "Server process must still be running after handshake"
            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_tools_list_returns_all_9_tools_in_deferred_error_state(self) -> None:
        """tools/list must return all 9 tools even when the index is stale."""
        proc = _spawn_server(self._stale_index_path, root=self._tmp_path)
        try:
            _handshake(proc, client_name="it96-stale-tools-list")

            _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            r = _recv(proc)
            assert r["id"] == 2
            tools = r["result"]["tools"]
            tool_names = {t["name"] for t in tools}

            # Schema assertions (wiring tier)
            assert len(tools) == 9, (
                f"Expected 9 tools in deferred-error state; got {len(tools)}: "
                f"{sorted(tool_names)}"
            )
            assert all("name" in t and "inputSchema" in t for t in tools), (
                "Each tool entry must have 'name' and 'inputSchema'"
            )
            assert "stats" in tool_names, "stats must be registered"
            assert "reload" in tool_names, "reload must be registered"
            assert "build" in tool_names, "build must be registered"
            assert "refers_to" in tool_names, "refers_to must be registered"
            assert "callees_of" in tool_names, "callees_of must be registered"
            assert "module_callees" in tool_names, "module_callees must be registered"
            assert "search" in tool_names, "search must be registered"
            assert "file_skeleton" in tool_names, "file_skeleton must be registered"
            assert "neighborhood" in tool_names, "neighborhood must be registered"

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_stats_returns_is_error_with_build_message(self) -> None:
        """stats in deferred-error state returns isError:true with 'pyscope-mcp build'."""
        proc = _spawn_server(self._stale_index_path, root=self._tmp_path)
        try:
            _handshake(proc, client_name="it96-stale-stats")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "stats", "arguments": {}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"

            # Wiring tier: status code / isError flag assertion
            assert r["result"]["isError"] is True, (
                "stats must return isError:true when the index is in deferred-error state"
            )

            # Schema assertion: content array present and non-empty
            content = r["result"]["content"]
            assert isinstance(content, list) and len(content) > 0, (
                "isError result must have a non-empty 'content' list"
            )
            assert content[0].get("type") == "text", (
                "content[0].type must be 'text'"
            )

            # Message substring assertion (wiring: actionable message present)
            text = content[0]["text"]
            assert "pyscope-mcp build" in text, (
                f"Error message must contain 'pyscope-mcp build' so the agent "
                f"knows the remedy; got: {text!r}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_server_stays_alive_after_deferred_error_tool_call(self) -> None:
        """Server must remain responsive after a deferred-error tool call (ping test)."""
        proc = _spawn_server(self._stale_index_path, root=self._tmp_path)
        try:
            _handshake(proc, client_name="it96-stale-alive")

            # Trigger deferred-error path
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "stats", "arguments": {}},
            })
            r = _recv(proc)
            assert r["result"]["isError"] is True

            # Server must still respond to ping
            _send(proc, {"jsonrpc": "2.0", "id": 3, "method": "ping"})
            r_ping = _recv(proc)
            assert r_ping == {"jsonrpc": "2.0", "id": 3, "result": {}}, (
                "Server must remain alive and respond to ping after a deferred-error tool call"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    @pytest.mark.parametrize("tool_name,arguments", [
        ("refers_to",      {"fqn": "pkg.mod.foo"}),
        ("callees_of",     {"fqn": "pkg.mod.foo"}),
        ("module_callees", {"module": "pkg.mod"}),
        ("search",         {"query": "foo"}),
        ("file_skeleton",  {"path": "pkg/mod.py"}),
        ("neighborhood",   {"symbol": "pkg.mod.foo"}),
    ])
    def test_all_index_dependent_tools_return_is_error(
        self, tool_name: str, arguments: dict
    ) -> None:
        """Each index-dependent tool must return isError:true in deferred-error state."""
        proc = _spawn_server(self._stale_index_path, root=self._tmp_path)
        try:
            _handshake(proc, client_name=f"it96-stale-{tool_name}")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error for {tool_name}: {r}"
            assert r["result"]["isError"] is True, (
                f"{tool_name} must return isError:true in deferred-error state; "
                f"got isError={r['result'].get('isError')}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)


# ---------------------------------------------------------------------------
# Scenario: slice-deferred-init-missing-index-e2e
# SCENARIO: slice-deferred-init-missing-index-e2e
# layers_involved: src/pyscope_mcp/server.py, src/pyscope_mcp/cli.py
# ---------------------------------------------------------------------------


class TestDeferredInitMissingIndexWiring:
    """Wiring tier — slice-deferred-init-missing-index-e2e.

    Verifies:
    - server starts (does not exit with code 2) when index path does not exist
    - MCP initialize handshake succeeds
    - tools/list returns all 9 tools
    - stats returns isError:true with 'pyscope-mcp build' in the message
    - clean shutdown returns {} and server exits 0
    """

    @pytest.fixture(autouse=True)
    def missing_index(self, tmp_path: Path) -> Path:
        """Provide a path to an index that deliberately does not exist."""
        idx_path = tmp_path / ".pyscope-mcp" / "index.json"
        # Intentionally do NOT create the file or directory
        self._missing_index_path = idx_path
        self._tmp_path = tmp_path
        return idx_path

    @pytest.mark.integration_wiring
    def test_server_starts_with_missing_index(self) -> None:
        """Server must not exit with code 2 when the index file does not exist.

        Before issue #96, cli.cmd_serve had an early-exit guard that returned 2
        when index.exists() was False. This test verifies the guard is removed:
        the server must start and respond to the MCP initialize handshake.
        """
        proc = _spawn_server(self._missing_index_path, root=self._tmp_path)
        try:
            # If the server exited early (old behavior), _recv would raise AssertionError
            # because stdout would be empty. The handshake completing proves the server started.
            _handshake(proc, client_name="it96-missing-startup")
            assert proc.poll() is None, "Server must still be running after handshake"
            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_tools_list_returns_all_9_tools_for_missing_index(self) -> None:
        """tools/list must return all 9 tools even when the index file is missing."""
        proc = _spawn_server(self._missing_index_path, root=self._tmp_path)
        try:
            _handshake(proc, client_name="it96-missing-tools-list")

            _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            r = _recv(proc)
            assert r["id"] == 2
            tools = r["result"]["tools"]

            # Schema assertions (wiring tier)
            assert len(tools) == 9, (
                f"Expected 9 tools when index is missing; got {len(tools)}: "
                f"{[t['name'] for t in tools]}"
            )
            assert all("name" in t and "inputSchema" in t for t in tools), (
                "Each tool entry must have 'name' and 'inputSchema'"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_stats_returns_is_error_with_build_message_for_missing_index(self) -> None:
        """stats must return isError:true with 'pyscope-mcp build' when index is missing."""
        proc = _spawn_server(self._missing_index_path, root=self._tmp_path)
        try:
            _handshake(proc, client_name="it96-missing-stats")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "stats", "arguments": {}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"

            # Wiring tier: isError flag
            assert r["result"]["isError"] is True, (
                "stats must return isError:true when the index file is missing"
            )

            # Schema assertions
            content = r["result"]["content"]
            assert isinstance(content, list) and len(content) > 0
            assert content[0].get("type") == "text"

            # Actionable message check
            text = content[0]["text"]
            assert "pyscope-mcp build" in text, (
                f"Error message must reference 'pyscope-mcp build'; got: {text!r}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_reload_with_missing_index_returns_is_error(self) -> None:
        """reload while index is still missing must return isError:true; server stays up."""
        proc = _spawn_server(self._missing_index_path, root=self._tmp_path)
        try:
            _handshake(proc, client_name="it96-missing-reload")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "reload", "arguments": {}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"

            # reload on a missing index must return isError:true
            assert r["result"]["isError"] is True, (
                "reload with a missing index must return isError:true"
            )
            content = r["result"]["content"]
            assert isinstance(content, list) and len(content) > 0
            text = content[0]["text"]
            assert "pyscope-mcp build" in text, (
                f"reload error message must reference 'pyscope-mcp build'; got: {text!r}"
            )

            # Server must still be alive
            _send(proc, {"jsonrpc": "2.0", "id": 3, "method": "ping"})
            r_ping = _recv(proc)
            assert r_ping == {"jsonrpc": "2.0", "id": 3, "result": {}}

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)
