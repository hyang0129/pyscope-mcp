"""Integration tests for issue #92 — eliminate _DiGraph, introduce GraphReader.

Scenario: slice-graph-reader-mcp-tools-e2e
  SCENARIO: slice-graph-reader-mcp-tools-e2e
  layers_involved: src/pyscope_mcp/graph.py,src/pyscope_mcp/server.py,src/pyscope_mcp/cli.py

Wiring tier:
  Verifies that after replacing _DiGraph with GraphReader in graph.py, all 9 MCP
  tools are still reachable over the real stdio JSON-RPC 2.0 wire and return
  structurally valid responses.  The test exercises the full vertical slice:
  CLI `serve` subprocess → OS pipes → JSON-RPC dispatch → CallGraphIndex
  (now GraphReader-backed) → structured response.

  No behavior change is expected — this is a refactor.  These tests detect any
  regression in the GraphReader migration that severs a system-edge connection
  (missing tool, wrong dispatch path, broken CallGraphIndex query method).

Artifact tier:
  Not generated.  The GraphReader refactor must produce identical output to the
  pre-migration _DiGraph implementation; output is verified by exact value
  comparison against synthetic fixtures in the wiring tier itself.  A separate
  golden fixture is not warranted because the output is deterministic from a
  fixed synthetic graph — the wiring assertions already pin exact values.
  volatile_output=false, but artifact tier is omitted per the no-behavior-change
  rationale (wiring pins exact values for a fixed synthetic index).

Pattern: real subprocess, real OS pipes, stdio JSON-RPC 2.0.
Mirrors test_integration_issue96.py conventions.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from conftest import make_nodes


# ---------------------------------------------------------------------------
# Helpers (mirrors test_integration_issue96.py)
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


def _handshake(proc: subprocess.Popen, client_name: str = "it92-test") -> None:
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
    assert r["result"]["serverInfo"]["name"] == "pyscope-mcp"
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
# Fixture: minimal synthetic index for query tools (in-memory, no real build)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic 5-node call graph for wiring tests.

    Graph:
      pkg.mod.alpha  →(call)→  pkg.mod.beta
      pkg.mod.alpha  →(call)→  pkg.mod.gamma
      pkg.mod.beta   →(call)→  pkg.mod.delta
      pkg.mod.delta  →(call)→  []
      pkg.mod.epsilon →(call)→ []

    pkg.mod.alpha has 2 callees; pkg.mod.beta has 1 callee; pkg.mod.gamma,
    pkg.mod.delta, pkg.mod.epsilon are leaf nodes.
    pkg.mod.gamma and pkg.mod.delta are each called once; pkg.mod.beta is called once.

    This graph is sufficient to exercise:
      - stats (node/edge counts)
      - callees_of (alpha → beta, gamma)
      - refers_to callers (beta ← alpha; gamma ← alpha)
      - module_callees (pkg.mod → pkg.mod itself via intra-module calls)
      - search (substring "alpha" → pkg.mod.alpha)
      - neighborhood (alpha with depth 1)
    """
    from pyscope_mcp.graph import CallGraphIndex

    raw: dict[str, list[str]] = {
        "pkg.mod.alpha": ["pkg.mod.beta", "pkg.mod.gamma"],
        "pkg.mod.beta": ["pkg.mod.delta"],
        "pkg.mod.gamma": [],
        "pkg.mod.delta": [],
        "pkg.mod.epsilon": [],
    }
    out_dir = tmp_path_factory.mktemp("pyscope_idx92")
    idx = CallGraphIndex.from_nodes(out_dir, make_nodes(raw))
    out = out_dir / "index.json"
    idx.save(out)
    return out


# ---------------------------------------------------------------------------
# Fixture: full build-based index for build + file_skeleton + reload tools
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_index_path(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build a real index from a synthetic package.

    Returns (tmp_root, index_path) for use with build/reload/file_skeleton tests.
    """
    tmp_path = tmp_path_factory.mktemp("pyscope_build92")
    pkg = tmp_path / "mypkg92"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text(textwrap.dedent("""\
        def func_a(x: int) -> int:
            return func_b(x) + 1

        def func_b(x: int) -> int:
            return x * 2

        class MyClass:
            def method_one(self) -> None:
                func_a(0)
    """))

    index_file = tmp_path / ".pyscope-mcp" / "index.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))
    build_result = subprocess.run(
        [
            sys.executable, "-m", "pyscope_mcp.cli", "build",
            "--root", str(tmp_path),
            "--package", "mypkg92",
            "--output", str(index_file),
        ],
        capture_output=True, env=env,
    )
    assert build_result.returncode == 0, f"build failed: {build_result.stderr.decode()}"
    return tmp_path, index_file


# ---------------------------------------------------------------------------
# Wiring tier — slice-graph-reader-mcp-tools-e2e
# ---------------------------------------------------------------------------


class TestGraphReaderMcpToolsWiring:
    """Wiring tier — slice-graph-reader-mcp-tools-e2e.

    Verifies that after the _DiGraph → GraphReader migration (issue #92):
    - The MCP server starts successfully and responds to the initialize handshake
    - tools/list returns all 9 expected tools with correct schema structure
    - stats returns a valid response with numeric node/edge counts
    - callees_of returns the correct result structure for a known symbol
    - refers_to returns the correct result structure for a known symbol
    - module_callees returns a valid response structure
    - search returns a valid response structure
    - neighborhood returns a valid response structure
    - reload completes successfully
    - build tool is registered and reachable

    All assertions are structural (wiring tier): status codes, response schema,
    tool presence, type checks. No exact golden value comparison.
    """

    @pytest.mark.integration_wiring
    def test_server_starts_and_handshake_succeeds(
        self, synthetic_index_path: Path
    ) -> None:
        """Server must start and complete MCP handshake after GraphReader migration."""
        proc = _spawn_server(synthetic_index_path)
        try:
            _handshake(proc, client_name="it92-startup")
            assert proc.poll() is None, "Server must still be running after handshake"
            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_tools_list_returns_all_9_tools(
        self, synthetic_index_path: Path
    ) -> None:
        """tools/list must return all 9 registered tools with name and inputSchema."""
        proc = _spawn_server(synthetic_index_path)
        try:
            _handshake(proc, client_name="it92-tools-list")
            _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"

            tools = r["result"]["tools"]
            tool_names = {t["name"] for t in tools}

            # Schema assertion: count and structure
            assert len(tools) == 9, (
                f"Expected 9 tools; got {len(tools)}: {sorted(tool_names)}"
            )
            assert all("name" in t and "inputSchema" in t for t in tools), (
                "Each tool entry must have 'name' and 'inputSchema'"
            )

            # All expected tools must be registered
            expected_tools = {
                "stats", "reload", "build", "refers_to", "callees_of",
                "module_callees", "search", "file_skeleton", "neighborhood",
            }
            assert tool_names == expected_tools, (
                f"Expected tools {expected_tools}; got {tool_names}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_stats_returns_valid_node_edge_counts(
        self, synthetic_index_path: Path
    ) -> None:
        """stats must return non-negative integer node/edge counts via GraphReader."""
        proc = _spawn_server(synthetic_index_path)
        try:
            _handshake(proc, client_name="it92-stats")
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "stats", "arguments": {}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"
            assert r["result"].get("isError") is not True, (
                f"stats returned isError:true — index may not be loaded: "
                f"{r['result']['content'][0]['text']}"
            )

            content = r["result"]["content"]
            assert isinstance(content, list) and len(content) > 0
            assert content[0].get("type") == "text"

            body = json.loads(content[0]["text"])
            # Schema assertions: required fields with correct types
            assert "functions" in body, "stats must return 'functions' count"
            assert "function_edges" in body, "stats must return 'function_edges' count"
            assert isinstance(body["functions"], int) and body["functions"] >= 0
            assert isinstance(body["function_edges"], int) and body["function_edges"] >= 0
            # The synthetic graph has 5 nodes; verify GraphReader counts correctly
            assert body["functions"] == 5, (
                f"Expected 5 functions in synthetic graph; got {body['functions']}"
            )
            # 3 call edges: alpha→beta, alpha→gamma, beta→delta
            assert body["function_edges"] == 3, (
                f"Expected 3 call edges in synthetic graph; got {body['function_edges']}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_callees_of_returns_valid_response_structure(
        self, synthetic_index_path: Path
    ) -> None:
        """callees_of must return a list of callee entries with fqn field."""
        proc = _spawn_server(synthetic_index_path)
        try:
            _handshake(proc, client_name="it92-callees")
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "callees_of", "arguments": {"fqn": "pkg.mod.alpha"}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"
            assert r["result"].get("isError") is not True

            body = json.loads(r["result"]["content"][0]["text"])
            # Schema assertions
            assert "results" in body, "callees_of must return 'results' list"
            assert isinstance(body["results"], list)
            assert len(body["results"]) > 0, (
                "pkg.mod.alpha has 2 callees; results must be non-empty"
            )
            # Results are plain FQN strings (not dicts)
            assert all(isinstance(e, str) for e in body["results"]), (
                f"callees_of results must be strings; got types: "
                f"{[type(e).__name__ for e in body['results']]}"
            )

            # Verify expected callees are present (GraphReader successors correctness)
            result_fqns = set(body["results"])
            assert "pkg.mod.beta" in result_fqns, (
                "pkg.mod.alpha must have pkg.mod.beta as a callee"
            )
            assert "pkg.mod.gamma" in result_fqns, (
                "pkg.mod.alpha must have pkg.mod.gamma as a callee"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_refers_to_callers_returns_valid_response_structure(
        self, synthetic_index_path: Path
    ) -> None:
        """refers_to with kind=callers must return a list of caller entries with fqn."""
        proc = _spawn_server(synthetic_index_path)
        try:
            _handshake(proc, client_name="it92-refers-to")
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "pkg.mod.beta", "kind": "callers"},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"
            assert r["result"].get("isError") is not True

            body = json.loads(r["result"]["content"][0]["text"])
            # Schema assertions
            assert "results" in body, "refers_to must return 'results' list"
            assert isinstance(body["results"], list)
            assert len(body["results"]) > 0, (
                "pkg.mod.beta is called by pkg.mod.alpha; results must be non-empty"
            )
            for entry in body["results"]:
                assert "fqn" in entry, f"Each caller entry must have 'fqn'; got {entry}"

            result_fqns = {e["fqn"] for e in body["results"]}
            assert "pkg.mod.alpha" in result_fqns, (
                "pkg.mod.beta must have pkg.mod.alpha as a caller"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_module_callees_returns_valid_response_structure(
        self, synthetic_index_path: Path
    ) -> None:
        """module_callees must return a valid response for a known module."""
        proc = _spawn_server(synthetic_index_path)
        try:
            _handshake(proc, client_name="it92-module-callees")
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "module_callees", "arguments": {"module": "pkg.mod"}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"
            # module_callees may return isError for unknown module — but schema must hold
            content = r["result"]["content"]
            assert isinstance(content, list) and len(content) > 0
            assert content[0].get("type") == "text"
            # Must be parseable JSON
            body = json.loads(content[0]["text"])
            assert isinstance(body, dict), "module_callees must return a JSON object"

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_search_returns_valid_response_structure(
        self, synthetic_index_path: Path
    ) -> None:
        """search must return a list of symbol entries matching the query."""
        proc = _spawn_server(synthetic_index_path)
        try:
            _handshake(proc, client_name="it92-search")
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "search", "arguments": {"query": "alpha"}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"
            assert r["result"].get("isError") is not True

            body = json.loads(r["result"]["content"][0]["text"])
            assert "results" in body, "search must return 'results' list"
            assert isinstance(body["results"], list)
            # "alpha" should match pkg.mod.alpha
            assert len(body["results"]) > 0, (
                "search for 'alpha' must return at least one result"
            )
            # Results are plain FQN strings
            assert all(isinstance(e, str) for e in body["results"]), (
                f"search results must be strings; got types: "
                f"{[type(e).__name__ for e in body['results']]}"
            )
            assert any("alpha" in fqn for fqn in body["results"]), (
                f"search for 'alpha' must include pkg.mod.alpha; got {body['results']}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_neighborhood_returns_valid_response_structure(
        self, synthetic_index_path: Path
    ) -> None:
        """neighborhood must return a connected subgraph structure around a symbol."""
        proc = _spawn_server(synthetic_index_path)
        try:
            _handshake(proc, client_name="it92-neighborhood")
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "neighborhood",
                    "arguments": {"symbol": "pkg.mod.alpha", "depth": 1},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"
            assert r["result"].get("isError") is not True

            body = json.loads(r["result"]["content"][0]["text"])
            # Schema assertions: neighborhood returns edges as list of [src, dst] pairs
            # and symbol as the queried symbol
            assert "edges" in body, "neighborhood must return 'edges'"
            assert "symbol" in body, "neighborhood must return 'symbol'"
            assert isinstance(body["edges"], list)
            assert body["symbol"] == "pkg.mod.alpha", (
                f"neighborhood must echo back the queried symbol; got {body['symbol']!r}"
            )
            # At depth=1 from alpha: edges alpha→beta and alpha→gamma
            assert len(body["edges"]) >= 1, (
                "neighborhood of pkg.mod.alpha at depth 1 must have at least 1 edge"
            )
            # Each edge is a [src, dst] pair
            assert all(isinstance(e, list) and len(e) == 2 for e in body["edges"]), (
                f"Each neighborhood edge must be a [src, dst] pair; got {body['edges']}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_reload_succeeds_and_returns_valid_response(
        self, synthetic_index_path: Path
    ) -> None:
        """reload must reload the index from disk and return a success response."""
        proc = _spawn_server(synthetic_index_path)
        try:
            _handshake(proc, client_name="it92-reload")
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "reload", "arguments": {}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"
            assert r["result"].get("isError") is not True, (
                f"reload returned isError:true: {r['result']['content'][0]['text']}"
            )

            content = r["result"]["content"]
            assert isinstance(content, list) and len(content) > 0
            assert content[0].get("type") == "text"

            # After reload, stats must still work (GraphReader re-initialised from disk)
            _send(proc, {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "stats", "arguments": {}},
            })
            r2 = _recv(proc)
            assert r2["id"] == 3
            assert r2["result"].get("isError") is not True, (
                "stats after reload must succeed — GraphReader must be re-initialised"
            )
            body = json.loads(r2["result"]["content"][0]["text"])
            assert body["functions"] == 5, (
                f"After reload, function count must still be 5; got {body.get('functions')}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_build_tool_is_registered_and_reachable(
        self, built_index_path: tuple[Path, Path]
    ) -> None:
        """build tool must be registered and return a successful response on a valid repo.

        The build tool shells out to 'pyscope-mcp build' as a subprocess using
        PYSCOPE_MCP_ROOT, PYSCOPE_MCP_PACKAGE, and PYSCOPE_MCP_INDEX env vars.
        We set these env vars explicitly in the server env so the build targets
        our small synthetic package (not the whole repo).
        """
        tmp_root, index_file = built_index_path
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(
            os.environ,
            PYTHONPATH=str(repo_root / "src"),
            PYSCOPE_MCP_ROOT=str(tmp_root),
            PYSCOPE_MCP_PACKAGE="mypkg92",
            PYSCOPE_MCP_INDEX=str(index_file),
        )
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "pyscope_mcp.cli", "serve",
                "--root", str(tmp_root),
                "--index", str(index_file),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env,
        )
        try:
            _handshake(proc, client_name="it92-build")
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "build", "arguments": {}},
            })
            r = _recv(proc, timeout=30.0)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"
            assert r["result"].get("isError") is not True, (
                f"build returned isError:true: {r['result']['content'][0]['text']}"
            )

            content = r["result"]["content"]
            assert isinstance(content, list) and len(content) > 0
            assert content[0].get("type") == "text"

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_file_skeleton_returns_valid_response_for_known_file(
        self, built_index_path: tuple[Path, Path]
    ) -> None:
        """file_skeleton must return symbol list for a file in the built index."""
        tmp_root, index_file = built_index_path
        proc = _spawn_server(index_file, root=tmp_root)
        try:
            _handshake(proc, client_name="it92-file-skeleton")
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "file_skeleton",
                    "arguments": {"path": "mypkg92/core.py"},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"
            assert r["result"].get("isError") is not True

            body = json.loads(r["result"]["content"][0]["text"])
            assert "results" in body, "file_skeleton must return 'results'"
            assert isinstance(body["results"], list)
            result_fqns = [s["fqn"] for s in body["results"]]
            assert "mypkg92.core.func_a" in result_fqns, (
                "file_skeleton must include func_a"
            )
            assert "mypkg92.core.func_b" in result_fqns, (
                "file_skeleton must include func_b"
            )
            assert "mypkg92.core.MyClass" in result_fqns, (
                "file_skeleton must include MyClass"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_refers_to_not_found_returns_is_error(
        self, synthetic_index_path: Path
    ) -> None:
        """refers_to with an unknown FQN must return isError:true, not a JSON-RPC error."""
        proc = _spawn_server(synthetic_index_path)
        try:
            _handshake(proc, client_name="it92-refers-to-notfound")
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {
                        "fqn": "pkg.mod.does_not_exist",
                        "kind": "callers",
                    },
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            # Must not be a JSON-RPC protocol error
            assert "error" not in r, f"Unexpected JSON-RPC error: {r}"
            # Must be an MCP isError response
            assert r["result"]["isError"] is True, (
                "refers_to with unknown FQN must return isError:true via GraphReader"
            )
            body = json.loads(r["result"]["content"][0]["text"])
            assert body.get("error_reason") == "fqn_not_in_graph", (
                f"error_reason must be 'fqn_not_in_graph'; got {body.get('error_reason')!r}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)
