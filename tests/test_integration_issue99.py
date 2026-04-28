"""Integration tests for issue #99 — PyPI publication readiness: logging default-off.

Key change: ``_log._is_enabled()`` default flipped from ``"1"`` to ``"0"``.
Logging is now **off by default**; set ``PYSCOPE_MCP_LOG=1`` to enable.

Two wiring-tier scenarios exercised through a real spawned subprocess (real OS
pipes, real ``run_stdio`` call path) — the same harness as
``test_integration_issue96.py``.  Component-level unit tests for the same
behaviour exist in ``test_query_logger.py`` (S4); these tests verify the
*end-to-end* propagation through ``run_stdio`` → ``_log.init`` → ``_log.log_call``.

Scenarios:

  slice-log-default-off-e2e (wiring)
    Verifies that when ``PYSCOPE_MCP_LOG`` is absent from the environment,
    spawning the server, calling a tool, and shutting down does NOT produce a
    ``query.jsonl`` file.  Tool response must still be non-error.

  slice-log-opt-in-e2e (wiring)
    Verifies that when ``PYSCOPE_MCP_LOG=1``, spawning the server, calling a
    tool, and shutting down DOES produce a non-empty ``query.jsonl`` file in
    the same directory as the index.

No artifact tier: file presence / absence and JSONL structural shape are
structural assertions, not exact-value golden fixtures (volatile_output=true).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import make_nodes


# ---------------------------------------------------------------------------
# Helpers (mirror test_integration_issue96.py conventions)
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


def _handshake(proc: subprocess.Popen, client_name: str = "it99-test") -> None:
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


def _spawn_server(index_path: Path, root: Path, extra_env: dict | None = None) -> subprocess.Popen:
    """Spawn pyscope-mcp serve as a subprocess, with optional extra env vars."""
    repo_root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items()}
    env["PYTHONPATH"] = str(repo_root / "src")
    # Remove PYSCOPE_MCP_LOG from the inherited environment so tests have
    # explicit control over whether it is set.
    env.pop("PYSCOPE_MCP_LOG", None)
    if extra_env:
        env.update(extra_env)
    args = [
        sys.executable, "-m", "pyscope_mcp.cli", "serve",
        "--root", str(root),
        "--index", str(index_path),
    ]
    return subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


# ---------------------------------------------------------------------------
# Shared fixture: minimal valid index on disk
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_index(tmp_path: Path) -> tuple[Path, Path]:
    """Write a minimal valid index to a temp directory.

    Returns (index_path, index_dir).
    """
    from pyscope_mcp.graph import CallGraphIndex

    raw: dict[str, list[str]] = {
        "pkg.mod.foo": ["pkg.mod.bar"],
        "pkg.mod.bar": [],
    }
    idx = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw))
    idx_dir = tmp_path / ".pyscope-mcp"
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx_path = idx_dir / "index.json"
    idx.save(idx_path)
    return idx_path, tmp_path


# ---------------------------------------------------------------------------
# Scenario: slice-log-default-off-e2e
# SCENARIO: slice-log-default-off-e2e
# layers_involved: src/pyscope_mcp/_log.py, src/pyscope_mcp/server.py
# ---------------------------------------------------------------------------


class TestLogDefaultOffE2E:
    """Wiring tier — slice-log-default-off-e2e.

    Verifies that when ``PYSCOPE_MCP_LOG`` is unset (the new default after
    issue #99), the server does not create a ``query.jsonl`` file even after a
    successful tool call.

    Structural assertions only (no exact-value golden comparison):
    - MCP handshake succeeds (server starts)
    - tools/call stats returns isError=false (tool works normally)
    - query.jsonl does NOT exist after shutdown
    """

    @pytest.mark.integration_wiring
    def test_no_log_file_when_env_var_absent(self, minimal_index: tuple[Path, Path]) -> None:
        """No query.jsonl must be created when PYSCOPE_MCP_LOG is unset."""
        idx_path, root = minimal_index
        log_path = idx_path.parent / "query.jsonl"

        # Precondition: no stale log from a prior run.
        assert not log_path.exists(), "Precondition: query.jsonl must not exist before test"

        proc = _spawn_server(idx_path, root=root)
        try:
            _handshake(proc, client_name="it99-default-off")

            # Call a tool to trigger the log_call path.
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "stats", "arguments": {}},
            })
            r = _recv(proc)

            # Wiring assertion: tool response must have correct JSON-RPC shape.
            assert "result" in r, f"Expected result key, got: {r}"
            assert r["result"]["isError"] is False, (
                f"stats tool must succeed: {r['result']}"
            )
            # Wiring assertion: content array present and non-empty.
            content = r["result"].get("content", [])
            assert len(content) > 0, "result.content must be non-empty"
            assert content[0].get("type") == "text", "content[0].type must be 'text'"

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()

        # Core assertion: logging must be off by default after issue #99.
        assert not log_path.exists(), (
            "query.jsonl must NOT be created when PYSCOPE_MCP_LOG is unset "
            "(default-off behaviour, issue #99)"
        )


# ---------------------------------------------------------------------------
# Scenario: slice-log-opt-in-e2e
# SCENARIO: slice-log-opt-in-e2e
# layers_involved: src/pyscope_mcp/_log.py, src/pyscope_mcp/server.py
# ---------------------------------------------------------------------------


class TestLogOptInE2E:
    """Wiring tier — slice-log-opt-in-e2e.

    Verifies that when ``PYSCOPE_MCP_LOG=1`` is present in the environment,
    the server creates a non-empty ``query.jsonl`` file after a tool call.

    Structural assertions (wiring tier, no golden fixture):
    - MCP handshake succeeds
    - tools/call stats returns isError=false
    - query.jsonl exists and is non-empty after shutdown
    - Each line in query.jsonl is valid JSON with required top-level keys
    """

    @pytest.mark.integration_wiring
    def test_log_file_created_when_opt_in(self, minimal_index: tuple[Path, Path]) -> None:
        """query.jsonl must be created and contain valid JSONL when PYSCOPE_MCP_LOG=1."""
        idx_path, root = minimal_index
        log_path = idx_path.parent / "query.jsonl"

        proc = _spawn_server(idx_path, root=root, extra_env={"PYSCOPE_MCP_LOG": "1"})
        try:
            _handshake(proc, client_name="it99-opt-in")

            # Call a tool to trigger the log_call path.
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "stats", "arguments": {}},
            })
            r = _recv(proc)

            # Wiring assertion: tool response must have correct JSON-RPC shape.
            assert "result" in r, f"Expected result key, got: {r}"
            assert r["result"]["isError"] is False, (
                f"stats tool must succeed: {r['result']}"
            )
            content = r["result"].get("content", [])
            assert len(content) > 0, "result.content must be non-empty"
            assert content[0].get("type") == "text", "content[0].type must be 'text'"

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()

        # Core assertion: log file must exist when opt-in.
        assert log_path.exists(), (
            "query.jsonl must be created when PYSCOPE_MCP_LOG=1 (opt-in logging)"
        )
        assert log_path.stat().st_size > 0, "query.jsonl must be non-empty"

        # Wiring assertion: each log line must be valid JSON with required structure.
        lines = [ln.strip() for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) >= 1, "At least one log entry expected after a tool call"
        for line in lines:
            entry = json.loads(line)  # must parse without error
            # Schema assertions: required top-level keys.
            assert "v" in entry, f"Log entry missing 'v': {entry}"
            assert "ts" in entry, f"Log entry missing 'ts': {entry}"
            assert "tool" in entry, f"Log entry missing 'tool': {entry}"
            assert "is_error" in entry, f"Log entry missing 'is_error': {entry}"
            assert isinstance(entry["is_error"], bool), "'is_error' must be bool"
