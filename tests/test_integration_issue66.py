"""Integration tests for issue #66 — build MCP tool + commit-SHA staleness.

Two scenarios, two tiers each:

  slice-build-tool-e2e (wiring):
    Verifies the `build` MCP tool is reachable over the real stdio-RPC wire,
    returns the correct response structure (isError:false, stats payload present),
    and the concurrent-rejection guard surfaces correctly as isError:true when a
    second build call arrives while the lock is held.

  slice-commit-staleness-query-e2e (wiring):
    Verifies that query tools (callers_of, stats) return the three commit-level
    staleness fields (commit_stale, index_git_sha, head_git_sha) over the real
    stdio-RPC wire after a real `pyscope-mcp build` has written the index.
    The git subprocess is exercised live — no mocking — because _commit_staleness()
    calling git rev-parse on every query is the system-edge behaviour the tests
    must exercise (per ADR §3).

Pattern: real subprocess, real OS pipes, stdio JSON-RPC 2.0.
Mirrors test_rpc_stdio_e2e.py conventions exactly (same _send/_recv helpers,
same index_path fixture shape, same env dict).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers (mirrors test_rpc_stdio_e2e.py)
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


def _handshake(proc: subprocess.Popen, client_name: str = "it-test") -> None:
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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built_index(tmp_path_factory: pytest.TempPathFactory):
    """Build a real index for a synthetic package via `pyscope-mcp build`.

    Returns (index_path, pkg_root, repo_root, env).
    This fixture runs the CLI build subprocess once for the module; tests share
    the built artifact.  The package structure is deterministic so callers_of
    and stats produce stable results.
    """
    tmp = tmp_path_factory.mktemp("it66")
    pkg = tmp / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text(
        "def alpha(): pass\n\n"
        "def beta(): return alpha()\n\n"
        "def gamma(): return beta()\n"
    )

    index_file = tmp / ".pyscope-mcp" / "index.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))

    result = subprocess.run(
        [
            sys.executable, "-m", "pyscope_mcp.cli", "build",
            "--root", str(tmp),
            "--package", "mypkg",
            "--output", str(index_file),
        ],
        capture_output=True, env=env,
    )
    assert result.returncode == 0, (
        f"pyscope-mcp build failed in fixture:\n{result.stderr.decode()}"
    )
    return index_file, tmp, repo_root, env


# ===========================================================================
# Scenario 1: slice-build-tool-e2e
# ===========================================================================
# SCENARIO: slice-build-tool-e2e
# layers_involved: src/pyscope_mcp/server.py, src/pyscope_mcp/cli.py, src/pyscope_mcp/graph.py


class TestBuildToolWiring:
    """Wiring tier — slice-build-tool-e2e.

    Verifies:
    - build tool is in tools/list with correct name
    - build tool responds over stdio-RPC (isError:false, stats fields present)
    - response content type is text with JSON body containing 'functions' key
    - the concurrent-rejection path returns isError:true with expected message
    """

    @pytest.mark.integration_wiring
    def test_build_tool_rpc_response_structure(self, built_index, tmp_path: Path) -> None:
        """[wiring] build tool over real stdio-RPC: response has isError:false + stats fields.

        Builds a fresh index for an isolated package so the `build` subprocess
        writes to a temp location the server can reload.  The subprocess call
        is real (not mocked) — exercising the full system-edge path.
        """
        index_file, pkg_root, repo_root, env = built_index

        # We need a server whose env vars point at the same index so the
        # build tool subprocess can find and overwrite it.
        srv_env = dict(env,
                       PYSCOPE_MCP_ROOT=str(pkg_root),
                       PYSCOPE_MCP_PACKAGE="mypkg",
                       PYSCOPE_MCP_INDEX=str(index_file))
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "pyscope_mcp.cli", "serve",
                "--root", str(pkg_root),
                "--index", str(index_file),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=srv_env,
        )
        try:
            _handshake(proc, "it66-wiring-build")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "build", "arguments": {}},
            })
            r = _recv(proc, timeout=30)
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            result = r["result"]

            # [Assert] Tool-level isError:false
            assert result.get("isError") is not True, (
                f"build returned isError:true — "
                f"content: {result.get('content')}"
            )

            # [Assert] content structure — type=text, body is JSON
            content = result["content"]
            assert len(content) >= 1, "build result must have at least one content item"
            assert content[0]["type"] == "text", "build content type must be 'text'"

            # [Assert] body is valid JSON with stats shape fields present
            body = json.loads(content[0]["text"])
            assert "functions" in body, (
                "build stats payload must include 'functions' — "
                "server may have failed to reload the index"
            )
            assert "function_edges" in body, (
                "build stats payload must include 'function_edges'"
            )
            assert "modules" in body, "build stats payload must include 'modules'"

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_build_tool_missing_env_returns_error(self, built_index) -> None:
        """[wiring] build tool fails gracefully when pyscope-mcp binary not on PATH.

        Serves with a PATH that lacks the pyscope-mcp entry point; the tool
        must return isError:true (not a JSON-RPC error, not a server crash).
        """
        index_file, pkg_root, repo_root, env = built_index

        # Strip pyscope-mcp from PATH by removing the venv bin directory
        venv_bin = str(Path(sys.executable).parent)
        broken_path = ":".join(
            p for p in env.get("PATH", "").split(":")
            if p != venv_bin and "pyscope" not in p.lower()
        )
        broken_env = dict(env, PATH=broken_path)

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "pyscope_mcp.cli", "serve",
                "--root", str(pkg_root),
                "--index", str(index_file),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=broken_env,
        )
        try:
            _handshake(proc, "it66-wiring-nobin")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "build", "arguments": {}},
            })
            r = _recv(proc)
            assert r["id"] == 2

            # [Assert] response is a tool result, not a JSON-RPC error
            assert "error" not in r, (
                f"got JSON-RPC error instead of isError:true — {r}"
            )

            # [Assert] isError:true when subprocess unavailable
            assert r["result"]["isError"] is True, (
                "build with missing binary must return isError:true"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)


# ===========================================================================
# Scenario 2: slice-commit-staleness-query-e2e
# ===========================================================================
# SCENARIO: slice-commit-staleness-query-e2e
# layers_involved: src/pyscope_mcp/graph.py, src/pyscope_mcp/server.py


class TestCommitStalenessQueryWiring:
    """Wiring tier — slice-commit-staleness-query-e2e.

    Verifies that the three commit-staleness fields introduced in #66
    (commit_stale, index_git_sha, head_git_sha) are present and structurally
    correct in the JSON responses of query tools over the real stdio-RPC wire.

    The git subprocess (_commit_staleness calling git rev-parse HEAD) is
    exercised live — no mocking — because this IS the system-edge behaviour
    the tests must cover per the ADR.  The fields may be None when the server
    runs outside a git checkout; structural correctness (key presence) is the
    wiring-tier assertion.  Exact value correctness is the artifact tier's job.
    """

    @pytest.mark.integration_wiring
    def test_stats_response_has_commit_staleness_fields(self, built_index) -> None:
        """[wiring] stats tool response includes commit_stale/index_git_sha/head_git_sha keys."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it66-wiring-stats")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "stats", "arguments": {}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"
            assert r["result"].get("isError") is not True

            body = json.loads(r["result"]["content"][0]["text"])

            # [Assert] Structural presence — wiring tier checks keys, not values
            assert "commit_stale" in body, (
                "stats response must include 'commit_stale' — "
                "_commit_staleness() may not be merging into StatsResult"
            )
            assert "index_git_sha" in body, (
                "stats response must include 'index_git_sha'"
            )
            assert "head_git_sha" in body, (
                "stats response must include 'head_git_sha'"
            )

            # [Assert] Normal stats fields also present (regression guard)
            assert "functions" in body
            assert "function_edges" in body

            # [Assert] commit_stale is bool or None — not an unexpected type
            assert body["commit_stale"] is None or isinstance(body["commit_stale"], bool), (
                f"commit_stale must be bool or None, got {type(body['commit_stale'])}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_callers_of_response_has_commit_staleness_fields(self, built_index) -> None:
        """[wiring] refers_to response includes commit staleness keys over real stdio-RPC."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it66-wiring-callers")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "refers_to", "arguments": {"fqn": "mypkg.core.alpha", "kind": "callers"}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r
            assert r["result"].get("isError") is not True

            body = json.loads(r["result"]["content"][0]["text"])

            # [Assert] Commit staleness keys present
            assert "commit_stale" in body, (
                "refers_to response must include 'commit_stale' over the wire"
            )
            assert "index_git_sha" in body
            assert "head_git_sha" in body

            # [Assert] Query result fields also present (regression guard)
            assert "results" in body
            assert "stale" in body

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_search_response_has_commit_staleness_fields(self, built_index) -> None:
        """[wiring] search response includes commit staleness keys over real stdio-RPC."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it66-wiring-search")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "search", "arguments": {"query": "alpha"}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r
            assert r["result"].get("isError") is not True

            body = json.loads(r["result"]["content"][0]["text"])

            # [Assert] Commit staleness keys present
            assert "commit_stale" in body
            assert "index_git_sha" in body
            assert "head_git_sha" in body

            # [Assert] Result fields present
            assert "results" in body
            assert "truncated" in body

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_commit_staleness_reflects_actual_build_sha(self, tmp_path: Path) -> None:
        """[wiring] index_git_sha and head_git_sha are non-None when server root is a git checkout.

        This test directly exercises the two git subprocess paths introduced in #66:
        1. cli.cmd_build() calling git rev-parse HEAD to capture the build SHA.
        2. _commit_staleness() calling git rev-parse HEAD on every stats query.

        Strategy: build the tiny 3-function package (from built_index fixture approach)
        rooted in tmp_path, then manually inject the real HEAD SHA from the pyscope-mcp
        git checkout into the saved index, and serve with root=repo_root so
        _commit_staleness() will resolve git rev-parse HEAD successfully.

        This avoids running the slow full-repo analyzer build while still exercising
        the live git subprocess path on every query call.
        """
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))

        # Resolve the real HEAD SHA from the repo checkout.
        git_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True, text=True,
        )
        if git_result.returncode != 0:
            pytest.skip("git not available or repo has no HEAD; skipping SHA test")
        real_head_sha = git_result.stdout.strip()
        assert len(real_head_sha) == 40, f"unexpected HEAD SHA: {real_head_sha!r}"

        # Build a tiny package in tmp_path (fast analyzer run).
        pkg = tmp_path / "tpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text("def check(): pass\n")

        index_file = tmp_path / "idx_sha.json"
        build_result = subprocess.run(
            [
                sys.executable, "-m", "pyscope_mcp.cli", "build",
                "--root", str(tmp_path),
                "--package", "tpkg",
                "--output", str(index_file),
            ],
            capture_output=True, env=env,
        )
        assert build_result.returncode == 0, (
            f"build failed: {build_result.stderr.decode()}"
        )

        # Patch the saved index: update root to repo_root and git_sha to the real HEAD.
        # This simulates what happens when cmd_build() runs inside a git checkout.
        payload = json.loads(index_file.read_text())
        payload["root"] = str(repo_root)
        payload["git_sha"] = real_head_sha
        index_file.write_text(json.dumps(payload))

        # Serve with root=repo_root; _commit_staleness will call git rev-parse there
        # on every query — this is the live system-edge path.
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "pyscope_mcp.cli", "serve",
                "--root", str(repo_root),
                "--index", str(index_file),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env,
        )
        try:
            _handshake(proc, "it66-wiring-sha-check")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "stats", "arguments": {}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            body = json.loads(r["result"]["content"][0]["text"])

            # [Assert] index_git_sha is non-None — the patched SHA flows through load()
            assert body["index_git_sha"] is not None, (
                "index_git_sha must be non-None — save/load of git_sha may be broken"
            )

            # [Assert] head_git_sha is non-None — live git rev-parse in a real checkout
            assert body["head_git_sha"] is not None, (
                "head_git_sha must be non-None when server root is a git checkout — "
                "_commit_staleness() git subprocess path may be broken"
            )

            # [Assert] Both are 40-char hex SHA1 strings
            for field in ("index_git_sha", "head_git_sha"):
                sha = body[field]
                assert isinstance(sha, str) and len(sha) == 40, (
                    f"{field} must be a 40-char hex string, got {sha!r}"
                )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_commit_stale_true_after_new_commit_simulation(self, tmp_path: Path) -> None:
        """[wiring] commit_stale is True when index_git_sha != head_git_sha over the wire.

        Builds a tiny index then patches git_sha to all-zeros (simulating an
        index built at a past commit).  Serves with root=repo_root so
        _commit_staleness() resolves HEAD successfully.  Verifies commit_stale=True
        flows through the stats RPC response.
        """
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))

        # Check git is available; skip if not.
        git_check = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True,
        )
        if git_check.returncode != 0:
            pytest.skip("git not available; skipping commit_stale test")

        # Build a tiny synthetic package (fast).
        pkg = tmp_path / "tpkg2"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text("def fn(): pass\n")

        index_file = tmp_path / "idx_stale.json"
        build_result = subprocess.run(
            [
                sys.executable, "-m", "pyscope_mcp.cli", "build",
                "--root", str(tmp_path),
                "--package", "tpkg2",
                "--output", str(index_file),
            ],
            capture_output=True, env=env,
        )
        assert build_result.returncode == 0, (
            f"build failed: {build_result.stderr.decode()}"
        )

        # Patch: set root=repo_root and git_sha=all-zeros to force staleness.
        payload = json.loads(index_file.read_text())
        payload["root"] = str(repo_root)
        payload["git_sha"] = "0000000000000000000000000000000000000000"
        index_file.write_text(json.dumps(payload))

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "pyscope_mcp.cli", "serve",
                "--root", str(repo_root),
                "--index", str(index_file),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env,
        )
        try:
            _handshake(proc, "it66-wiring-stale-commit")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "stats", "arguments": {}},
            })
            r = _recv(proc)
            assert r["id"] == 2
            body = json.loads(r["result"]["content"][0]["text"])

            # [Assert] commit_stale is True (all-zeros SHA != real HEAD SHA)
            assert body["commit_stale"] is True, (
                f"commit_stale must be True when index SHA is all-zeros and "
                f"HEAD is real; got commit_stale={body.get('commit_stale')!r}, "
                f"index_git_sha={body.get('index_git_sha')!r}, "
                f"head_git_sha={body.get('head_git_sha')!r}"
            )

            # [Assert] index_git_sha echoes the patched all-zeros value
            assert body["index_git_sha"] == "0000000000000000000000000000000000000000"

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)


# ===========================================================================
# Artifact tier — slice-build-tool-e2e
# ===========================================================================


class TestBuildToolArtifact:
    """Artifact tier — slice-build-tool-e2e.

    Verifies the exact output of the build tool: the stats payload returned
    after a real `pyscope-mcp build` completes must enumerate the correct
    function/module counts for the synthetic package.

    Golden: the synthetic package has 3 functions (alpha, beta, gamma) in
    mypkg.core, forming 2 directed call edges (beta→alpha, gamma→beta).
    Module graph: 1 module node (mypkg.core), 0 module edges (all within one module).
    """

    @pytest.mark.integration_artifact
    def test_build_tool_exact_stats_payload(self, built_index) -> None:
        """[artifact] build tool returns exact stats for synthetic 3-function package.

        Golden fixture (inline): functions=3, function_edges=2, modules=1, module_edges=0.
        The built_index fixture constructs:
            def alpha(): pass
            def beta(): return alpha()   # 1 edge: beta → alpha
            def gamma(): return beta()   # 1 edge: gamma → beta
        """
        index_file, pkg_root, repo_root, env = built_index

        srv_env = dict(env,
                       PYSCOPE_MCP_ROOT=str(pkg_root),
                       PYSCOPE_MCP_PACKAGE="mypkg",
                       PYSCOPE_MCP_INDEX=str(index_file))

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "pyscope_mcp.cli", "serve",
                "--root", str(pkg_root),
                "--index", str(index_file),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=srv_env,
        )
        try:
            _handshake(proc, "it66-artifact-build")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "build", "arguments": {}},
            })
            r = _recv(proc, timeout=30)
            assert r["id"] == 2
            result = r["result"]
            assert result.get("isError") is not True, (
                f"build returned isError — content: {result.get('content')}"
            )

            body = json.loads(result["content"][0]["text"])

            # [Assert] Exact function count — golden: 3 (alpha, beta, gamma)
            assert body["functions"] == 3, (
                f"expected 3 functions (alpha, beta, gamma); got {body['functions']}"
            )

            # [Assert] Exact edge count — golden: 2 (beta→alpha, gamma→beta)
            assert body["function_edges"] == 2, (
                f"expected 2 function edges (beta→alpha, gamma→beta); "
                f"got {body['function_edges']}"
            )

            # [Assert] Module count — golden: 1 (mypkg.core)
            assert body["modules"] == 1, (
                f"expected 1 module (mypkg.core); got {body['modules']}"
            )

            # [Assert] Module edges — golden: 0 (all calls within mypkg.core)
            assert body["module_edges"] == 0, (
                f"expected 0 module edges; got {body['module_edges']}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)
