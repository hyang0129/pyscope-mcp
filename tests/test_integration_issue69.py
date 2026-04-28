"""Integration tests for issue #69 — not-found error_reason on all four query tools.

Two tiers:

  slice-not-found-error-reason-all-tools-e2e (wiring):
    Verifies that file_skeleton, callers_of, callees_of, and neighborhood all
    return isError:true at the MCP result level when given nonexistent paths/FQNs,
    that the error_reason key is present with the expected slug, and that stale is
    False with no stale_action key — over the real stdio-RPC wire.

    Also verifies the AC5 regression guard: callers_of with a present-but-zero-callers
    FQN returns isError NOT true with results=[].

  slice-not-found-error-reason-all-tools-e2e (artifact):
    Verifies the exact values of error_reason slugs, stale=False (not None, not 0),
    stale_files=[], no stale_action key, and the AC5 empty-results path.

Pattern: real subprocess, real OS pipes, stdio JSON-RPC 2.0.
Mirrors test_integration_issue66.py conventions exactly (same _send/_recv helpers,
same index fixture shape, same env dict).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers (mirrors test_integration_issue66.py)
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
    """Build a real index for a synthetic 3-function package.

    Package structure:
        mypkg/
            __init__.py
            core.py   # alpha, beta (calls alpha), gamma (calls beta)

    Real FQNs in index:
        mypkg.core.alpha   — called by beta; zero callers (AC5 target)
        mypkg.core.beta    — calls alpha, called by gamma
        mypkg.core.gamma   — calls beta; zero callers from graph perspective

    Wait — gamma has zero callers (nothing calls gamma), so callers_of(gamma)=[]
    and alpha is called by beta so callers_of(alpha)=[beta]. For AC5 we need a
    present-but-zero-callers FQN: use mypkg.core.gamma (nothing calls it).

    Bad inputs for error testing:
        bad path:  mypkg/does_not_exist.py
        bad FQN:   mypkg.does_not_exist.whatever

    Returns (index_file, pkg_root, repo_root, env).
    """
    tmp = tmp_path_factory.mktemp("it69")
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
# SCENARIO: slice-not-found-error-reason-all-tools-e2e
# layers_involved: src/pyscope_mcp/graph.py, src/pyscope_mcp/server.py
# ===========================================================================


class TestSliceNotFoundErrorReasonWiring:
    """Wiring tier — slice-not-found-error-reason-all-tools-e2e.

    Verifies the structural contract for not-found error responses:
    - isError:true at the MCP result level
    - content is present and parseable as JSON
    - error_reason key exists with the correct slug type (str)
    - stale is False, stale_action key is absent
    - AC5 regression guard: present-but-zero-callers FQN returns isError=False
    """

    @pytest.mark.integration_wiring
    def test_file_skeleton_bad_path_is_error_with_error_reason(
        self, built_index
    ) -> None:
        """[wiring] file_skeleton(bad_path) → isError:true, error_reason present, stale=false, no stale_action.

        AC1: file_skeleton with a path not in the index must return the not-found
        error shape, not a server crash or a JSON-RPC error.
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            # [Arrange]
            _handshake(proc, "it69-wiring-skeleton")

            # [Act]
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "file_skeleton",
                    "arguments": {"path": "mypkg/does_not_exist.py"},
                },
            })
            r = _recv(proc)

            # [Assert] JSON-RPC level: no transport error
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            result = r["result"]

            # [Assert] MCP tool level: isError:true
            assert result.get("isError") is True, (
                f"file_skeleton with bad path must return isError:true; got {result}"
            )

            # [Assert] content is parseable JSON
            content = result["content"]
            assert len(content) >= 1, "result must have at least one content item"
            assert content[0]["type"] == "text"
            body = json.loads(content[0]["text"])

            # [Assert] error_reason key is present and is a string
            assert "error_reason" in body, (
                "not-found response must include 'error_reason' key"
            )
            assert isinstance(body["error_reason"], str), (
                f"error_reason must be a string, got {type(body['error_reason'])}"
            )

            # [Assert] stale is False (not missing, not truthy)
            assert "stale" in body, "not-found response must include 'stale' key"
            assert body["stale"] is False, (
                f"stale must be False on not-found error, got {body['stale']!r}"
            )

            # [Assert] stale_action key must NOT be present
            assert "stale_action" not in body, (
                "not-found response must NOT include 'stale_action' — "
                "stale_action is only for stale=true results, not missing-path errors"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_callers_of_bad_fqn_is_error_with_error_reason(
        self, built_index
    ) -> None:
        """[wiring] refers_to(bad_fqn) → isError:true, error_reason present, stale=false, no stale_action.

        AC2: refers_to with an FQN not in the graph must return the not-found
        error shape — not an empty results list and not a server crash.
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            # [Arrange]
            _handshake(proc, "it69-wiring-callers")

            # [Act]
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.does_not_exist.whatever", "kind": "callers"},
                },
            })
            r = _recv(proc)

            # [Assert] JSON-RPC level: no transport error
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            result = r["result"]

            # [Assert] MCP tool level: isError:true
            assert result.get("isError") is True, (
                f"refers_to with bad FQN must return isError:true; got {result}"
            )

            # [Assert] content is parseable JSON with error_reason
            body = json.loads(result["content"][0]["text"])
            assert "error_reason" in body, (
                "not-found response must include 'error_reason' key"
            )
            assert isinstance(body["error_reason"], str)

            # [Assert] stale=False, no stale_action
            assert body.get("stale") is False, (
                f"stale must be False on not-found error, got {body.get('stale')!r}"
            )
            assert "stale_action" not in body, (
                "not-found response must NOT include 'stale_action'"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_callees_of_bad_fqn_is_error_with_error_reason(
        self, built_index
    ) -> None:
        """[wiring] callees_of(bad_fqn) → isError:true, error_reason present, stale=false, no stale_action.

        AC3: callees_of with an FQN not in the graph must return the same
        not-found error shape as callers_of.
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            # [Arrange]
            _handshake(proc, "it69-wiring-callees")

            # [Act]
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "callees_of",
                    "arguments": {"fqn": "mypkg.does_not_exist.whatever"},
                },
            })
            r = _recv(proc)

            # [Assert] JSON-RPC level: no transport error
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            result = r["result"]

            # [Assert] MCP tool level: isError:true
            assert result.get("isError") is True, (
                f"callees_of with bad FQN must return isError:true; got {result}"
            )

            # [Assert] content is parseable JSON with error_reason
            body = json.loads(result["content"][0]["text"])
            assert "error_reason" in body, (
                "not-found response must include 'error_reason' key"
            )
            assert isinstance(body["error_reason"], str)

            # [Assert] stale=False, no stale_action
            assert body.get("stale") is False, (
                f"stale must be False on not-found error, got {body.get('stale')!r}"
            )
            assert "stale_action" not in body, (
                "not-found response must NOT include 'stale_action'"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_neighborhood_bad_fqn_is_error_with_error_reason(
        self, built_index
    ) -> None:
        """[wiring] neighborhood(bad_symbol) → isError:true, error_reason present, stale=false, no stale_action.

        AC4: neighborhood with an FQN not in the graph must return the not-found
        error shape — not empty edges and not a server crash.
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            # [Arrange]
            _handshake(proc, "it69-wiring-neighborhood")

            # [Act]
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "neighborhood",
                    "arguments": {"symbol": "mypkg.does_not_exist.whatever"},
                },
            })
            r = _recv(proc)

            # [Assert] JSON-RPC level: no transport error
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            result = r["result"]

            # [Assert] MCP tool level: isError:true
            assert result.get("isError") is True, (
                f"neighborhood with bad FQN must return isError:true; got {result}"
            )

            # [Assert] content is parseable JSON with error_reason
            body = json.loads(result["content"][0]["text"])
            assert "error_reason" in body, (
                "not-found response must include 'error_reason' key"
            )
            assert isinstance(body["error_reason"], str)

            # [Assert] stale=False, no stale_action
            assert body.get("stale") is False, (
                f"stale must be False on not-found error, got {body.get('stale')!r}"
            )
            assert "stale_action" not in body, (
                "not-found response must NOT include 'stale_action'"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_callers_of_present_fqn_with_zero_callers_is_not_error(
        self, built_index
    ) -> None:
        """[wiring] AC5 regression guard: refers_to(present-but-no-callers FQN) → isError NOT true, results list present.

        mypkg.core.gamma is in the index (so isError must NOT be true) but
        nothing calls gamma, so results must be a list (possibly empty).
        This guards against the not-found path being incorrectly triggered for
        FQNs that are present but have zero incoming edges.
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            # [Arrange]
            _handshake(proc, "it69-wiring-ac5")

            # [Act]
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.core.gamma", "kind": "callers"},
                },
            })
            r = _recv(proc)

            # [Assert] JSON-RPC level: no transport error
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            result = r["result"]

            # [Assert] isError must NOT be true for a present FQN
            assert result.get("isError") is not True, (
                "refers_to with a present FQN (gamma) must NOT return isError:true — "
                "zero callers is not an error; got isError=True"
            )

            # [Assert] content is parseable JSON and includes 'results' key
            body = json.loads(result["content"][0]["text"])
            assert "results" in body, (
                "refers_to(gamma) must include 'results' key in response body"
            )
            assert isinstance(body["results"], list), (
                f"results must be a list, got {type(body['results'])}"
            )

            # [Assert] stale is present (and False for fresh index)
            assert "stale" in body, (
                "refers_to response must include 'stale' key"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)


# ===========================================================================
# Artifact tier — slice-not-found-error-reason-all-tools-e2e
# ===========================================================================


class TestSliceNotFoundErrorReasonArtifact:
    """Artifact tier — slice-not-found-error-reason-all-tools-e2e.

    Verifies exact output values against hard-coded golden expectations:
    - exact error_reason slug matches the spec ('path_not_in_index' or 'fqn_not_in_graph')
    - stale is exactly False (not None, not 0, not "false")
    - stale_files is exactly []
    - stale_action key is completely absent from the response body
    - AC5: callers_of(gamma) returns results==[] AND stale==False
    """

    @pytest.mark.integration_artifact
    def test_file_skeleton_bad_path_exact_error_reason(self, built_index) -> None:
        """[artifact] file_skeleton(bad_path) → error_reason=='path_not_in_index', stale==False, stale_files==[], no stale_action.

        Golden: error_reason='path_not_in_index' (defined in graph.py file_skeleton).
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            # [Arrange]
            _handshake(proc, "it69-artifact-skeleton")

            # [Act]
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "file_skeleton",
                    "arguments": {"path": "mypkg/does_not_exist.py"},
                },
            })
            r = _recv(proc)
            result = r["result"]
            body = json.loads(result["content"][0]["text"])

            # [Assert] Exact error_reason slug — golden: 'path_not_in_index'
            assert body["error_reason"] == "path_not_in_index", (
                f"expected error_reason='path_not_in_index', got {body['error_reason']!r}"
            )

            # [Assert] stale is exactly False (not None, not 0)
            assert body["stale"] is False, (
                f"stale must be exactly False (bool), got {body['stale']!r} "
                f"(type: {type(body['stale']).__name__})"
            )

            # [Assert] stale_files is exactly []
            assert body["stale_files"] == [], (
                f"stale_files must be exactly [], got {body['stale_files']!r}"
            )

            # [Assert] stale_action key is completely absent
            assert "stale_action" not in body, (
                f"stale_action must be absent from not-found response; "
                f"found stale_action={body['stale_action']!r}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_artifact
    def test_callers_of_bad_fqn_exact_error_reason(self, built_index) -> None:
        """[artifact] refers_to(bad_fqn) → error_reason=='fqn_not_in_graph', stale==False, stale_files==[], no stale_action.

        Golden: error_reason='fqn_not_in_graph' (defined in graph.py refers_to).
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            # [Arrange]
            _handshake(proc, "it69-artifact-callers")

            # [Act]
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.does_not_exist.whatever", "kind": "callers"},
                },
            })
            r = _recv(proc)
            result = r["result"]
            body = json.loads(result["content"][0]["text"])

            # [Assert] Exact error_reason slug — golden: 'fqn_not_in_graph'
            assert body["error_reason"] == "fqn_not_in_graph", (
                f"expected error_reason='fqn_not_in_graph', got {body['error_reason']!r}"
            )

            # [Assert] stale is exactly False (bool)
            assert body["stale"] is False, (
                f"stale must be exactly False (bool), got {body['stale']!r} "
                f"(type: {type(body['stale']).__name__})"
            )

            # [Assert] stale_files is exactly []
            assert body["stale_files"] == [], (
                f"stale_files must be exactly [], got {body['stale_files']!r}"
            )

            # [Assert] stale_action key is completely absent
            assert "stale_action" not in body, (
                f"stale_action must be absent from not-found response; "
                f"found stale_action={body['stale_action']!r}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_artifact
    def test_callees_of_bad_fqn_exact_error_reason(self, built_index) -> None:
        """[artifact] callees_of(bad_fqn) → error_reason=='fqn_not_in_graph', stale==False, stale_files==[], no stale_action.

        Golden: error_reason='fqn_not_in_graph' (defined in graph.py callees_of).
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            # [Arrange]
            _handshake(proc, "it69-artifact-callees")

            # [Act]
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "callees_of",
                    "arguments": {"fqn": "mypkg.does_not_exist.whatever"},
                },
            })
            r = _recv(proc)
            result = r["result"]
            body = json.loads(result["content"][0]["text"])

            # [Assert] Exact error_reason slug — golden: 'fqn_not_in_graph'
            assert body["error_reason"] == "fqn_not_in_graph", (
                f"expected error_reason='fqn_not_in_graph', got {body['error_reason']!r}"
            )

            # [Assert] stale is exactly False (bool)
            assert body["stale"] is False, (
                f"stale must be exactly False (bool), got {body['stale']!r} "
                f"(type: {type(body['stale']).__name__})"
            )

            # [Assert] stale_files is exactly []
            assert body["stale_files"] == [], (
                f"stale_files must be exactly [], got {body['stale_files']!r}"
            )

            # [Assert] stale_action key is completely absent
            assert "stale_action" not in body, (
                f"stale_action must be absent from not-found response; "
                f"found stale_action={body['stale_action']!r}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_artifact
    def test_neighborhood_bad_fqn_exact_error_reason(self, built_index) -> None:
        """[artifact] neighborhood(bad_symbol) → error_reason=='fqn_not_in_graph', stale==False, stale_files==[], no stale_action.

        Golden: error_reason='fqn_not_in_graph' (defined in graph.py neighborhood).
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            # [Arrange]
            _handshake(proc, "it69-artifact-neighborhood")

            # [Act]
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "neighborhood",
                    "arguments": {"symbol": "mypkg.does_not_exist.whatever"},
                },
            })
            r = _recv(proc)
            result = r["result"]
            body = json.loads(result["content"][0]["text"])

            # [Assert] Exact error_reason slug — golden: 'fqn_not_in_graph'
            assert body["error_reason"] == "fqn_not_in_graph", (
                f"expected error_reason='fqn_not_in_graph', got {body['error_reason']!r}"
            )

            # [Assert] stale is exactly False (bool)
            assert body["stale"] is False, (
                f"stale must be exactly False (bool), got {body['stale']!r} "
                f"(type: {type(body['stale']).__name__})"
            )

            # [Assert] stale_files is exactly []
            assert body["stale_files"] == [], (
                f"stale_files must be exactly [], got {body['stale_files']!r}"
            )

            # [Assert] stale_action key is completely absent
            assert "stale_action" not in body, (
                f"stale_action must be absent from not-found response; "
                f"found stale_action={body['stale_action']!r}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_artifact
    def test_callers_of_present_fqn_with_zero_callers_exact_values(
        self, built_index
    ) -> None:
        """[artifact] AC5: refers_to(gamma) → results==[], stale==False exactly.

        mypkg.core.gamma is present in the index (it calls beta) but nothing
        calls gamma itself.  The response must be a success (isError NOT true)
        with results exactly equal to [] and stale exactly equal to False.
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            # [Arrange]
            _handshake(proc, "it69-artifact-ac5")

            # [Act]
            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.core.gamma", "kind": "callers"},
                },
            })
            r = _recv(proc)
            result = r["result"]

            # [Assert] isError must NOT be true
            assert result.get("isError") is not True, (
                "refers_to(gamma) must NOT be isError:true — "
                "gamma is present in the index; zero callers is not an error"
            )

            body = json.loads(result["content"][0]["text"])

            # [Assert] results is exactly [] (golden: gamma has no callers)
            assert body["results"] == [], (
                f"refers_to(gamma) must return results=[] since nothing calls gamma; "
                f"got {body['results']!r}"
            )

            # [Assert] stale is exactly False (bool, not None, not 0)
            assert body["stale"] is False, (
                f"stale must be exactly False (bool), got {body['stale']!r} "
                f"(type: {type(body['stale']).__name__})"
            )

            # [Assert] stale_files is exactly []
            assert body["stale_files"] == [], (
                f"stale_files must be exactly [], got {body['stale_files']!r}"
            )

            # [Assert] No stale_action on a fresh (non-stale) result
            assert "stale_action" not in body, (
                "stale_action must be absent when stale=False"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)
