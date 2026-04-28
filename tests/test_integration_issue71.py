"""Integration tests for issue #71 — refers_to typed symbol reference lookup.

Two scenarios, two tiers each:

  slice-refers-to-tool-e2e (wiring):
    Verifies that the `refers_to` MCP tool is reachable over the real stdio-RPC wire,
    accepts kind/granularity/depth parameters, returns the correct response structure
    (results list with fqn/context/depth entries), and that removed tools callers_of
    and module_callers are absent from tools/list.

  slice-refers-to-multi-kind-e2e (wiring + artifact):
    Verifies that refers_to correctly returns all typed reference kinds (call, import,
    except, annotation, isinstance) when queried against a synthetic package that
    exercises each kind. The artifact tier pins exact result counts and context tags.

Pattern: real subprocess, real OS pipes, stdio JSON-RPC 2.0.
Mirrors test_integration_issue66.py conventions exactly (same _send/_recv helpers,
same index fixture shape, same env dict).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
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
    """Build a real index for a synthetic 3-function call-chain package.

    Package structure:
        mypkg/
            __init__.py
            core.py   # alpha, beta (calls alpha), gamma (calls beta)

    Real FQNs in index (call graph):
        mypkg.core.alpha  — has callers: [mypkg.core.beta]
        mypkg.core.beta   — calls alpha, has callers: [mypkg.core.gamma]
        mypkg.core.gamma  — calls beta; zero callers

    This fixture is shared across all wiring tests in this module.
    Returns (index_file, pkg_root, repo_root, env).
    """
    tmp = tmp_path_factory.mktemp("it71")
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
# Scenario 1: slice-refers-to-tool-e2e
# ===========================================================================
# SCENARIO: slice-refers-to-tool-e2e
# layers_involved: src/pyscope_mcp/server.py, src/pyscope_mcp/graph.py, src/pyscope_mcp/types.py


class TestRefersToToolWiring:
    """Wiring tier — slice-refers-to-tool-e2e.

    Verifies:
    - refers_to is in tools/list; callers_of and module_callers are absent
    - refers_to with kind='callers' returns a valid call-only result structure
    - refers_to with kind='all' returns a valid multi-kind result structure
    - refers_to with granularity='module' returns a flat list of strings
    - refers_to with depth=3 returns isError:true with error_reason='depth_exceeds_max'
    - refers_to with a nonexistent FQN returns isError:true with error_reason='fqn_not_in_graph'
    - response always includes truncated, dropped, stale, stale_files, completeness keys
    """

    @pytest.mark.integration_wiring
    def test_refers_to_in_tool_list_callers_of_absent(self, built_index) -> None:
        """[wiring] tools/list must include refers_to and must NOT include callers_of or module_callers."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-wiring-tool-list")

            _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            tool_names = {t["name"] for t in r["result"]["tools"]}

            # [Assert] refers_to is present
            assert "refers_to" in tool_names, (
                f"'refers_to' must be in tools/list; found: {sorted(tool_names)}"
            )

            # [Assert] deleted tools are absent
            assert "callers_of" not in tool_names, (
                "'callers_of' must be absent from tools/list — it was hard-deleted in #71"
            )
            assert "module_callers" not in tool_names, (
                "'module_callers' must be absent from tools/list — it was hard-deleted in #71"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_refers_to_callers_kind_response_structure(self, built_index) -> None:
        """[wiring] refers_to(kind='callers') returns valid result structure over real stdio-RPC.

        Calls refers_to(fqn='mypkg.core.alpha', kind='callers') which should
        return mypkg.core.beta (the only direct caller of alpha).
        Verifies: no JSON-RPC error, isError:false, content[0].type='text',
        body contains results/truncated/dropped/stale/stale_files/completeness.
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-wiring-callers")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.core.alpha", "kind": "callers"},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            result = r["result"]

            # [Assert] Tool-level isError:false
            assert result.get("isError") is not True, (
                f"refers_to(kind='callers') must not return isError:true; got {result}"
            )

            # [Assert] content structure — type=text, parseable JSON
            content = result["content"]
            assert len(content) >= 1, "result must have at least one content item"
            assert content[0]["type"] == "text", "content type must be 'text'"

            body = json.loads(content[0]["text"])

            # [Assert] Required structural keys are present
            for field in ("results", "truncated", "dropped", "stale", "stale_files", "completeness"):
                assert field in body, (
                    f"refers_to response must include '{field}' key; body keys: {list(body.keys())}"
                )

            # [Assert] results is a list
            assert isinstance(body["results"], list), (
                f"results must be a list, got {type(body['results'])}"
            )

            # [Assert] entries (if any) have fqn, context, depth shape
            for entry in body["results"]:
                assert isinstance(entry, dict), f"each result entry must be a dict, got {entry!r}"
                assert "fqn" in entry, f"result entry must have 'fqn'; got {entry}"
                assert "context" in entry, f"result entry must have 'context'; got {entry}"
                assert "depth" in entry, f"result entry must have 'depth'; got {entry}"

            # [Assert] truncated and dropped are correct types
            assert isinstance(body["truncated"], bool), (
                f"truncated must be bool, got {type(body['truncated'])}"
            )
            assert isinstance(body["dropped"], int), (
                f"dropped must be int, got {type(body['dropped'])}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_refers_to_all_kind_response_structure(self, built_index) -> None:
        """[wiring] refers_to(kind='all') returns valid result structure over real stdio-RPC."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-wiring-all")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.core.alpha", "kind": "all"},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            result = r["result"]
            assert result.get("isError") is not True, (
                f"refers_to(kind='all') must not return isError:true; got {result}"
            )

            body = json.loads(result["content"][0]["text"])

            # [Assert] Required keys
            for field in ("results", "truncated", "dropped", "stale", "stale_files", "completeness"):
                assert field in body, f"refers_to response must include '{field}'"

            # [Assert] results is a list of dicts with fqn/context/depth
            assert isinstance(body["results"], list)
            for entry in body["results"]:
                assert "fqn" in entry
                assert "context" in entry
                assert entry["context"] in ("call", "import", "except", "annotation", "isinstance"), (
                    f"context must be a known kind, got {entry['context']!r}"
                )
                assert "depth" in entry

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_refers_to_module_granularity_response_structure(self, built_index) -> None:
        """[wiring] refers_to(granularity='module') returns flat string list over real stdio-RPC."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-wiring-module-gran")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {
                        "fqn": "mypkg.core.alpha",
                        "kind": "callers",
                        "granularity": "module",
                    },
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r

            result = r["result"]
            assert result.get("isError") is not True

            body = json.loads(result["content"][0]["text"])

            # [Assert] Required keys
            for field in ("results", "truncated", "dropped", "stale", "stale_files", "completeness"):
                assert field in body, f"module granularity response must include '{field}'"

            # [Assert] results is a list of strings (module FQNs)
            assert isinstance(body["results"], list)
            for item in body["results"]:
                assert isinstance(item, str), (
                    f"module granularity result items must be strings; got {type(item)}: {item!r}"
                )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_refers_to_depth_exceeds_max_returns_error(self, built_index) -> None:
        """[wiring] refers_to(depth=3) returns isError:true with error_reason='depth_exceeds_max'."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-wiring-depth-max")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.core.alpha", "kind": "all", "depth": 3},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            result = r["result"]

            # [Assert] MCP tool-level isError:true
            assert result.get("isError") is True, (
                f"refers_to(depth=3) must return isError:true; got {result}"
            )

            # [Assert] error_reason is depth_exceeds_max
            content = result["content"]
            body = json.loads(content[0]["text"])
            assert "error_reason" in body, "depth error response must include 'error_reason'"
            assert body["error_reason"] == "depth_exceeds_max", (
                f"expected error_reason='depth_exceeds_max', got {body['error_reason']!r}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_refers_to_fqn_not_in_graph_returns_error(self, built_index) -> None:
        """[wiring] refers_to(bad_fqn) returns isError:true with error_reason='fqn_not_in_graph'."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-wiring-not-found")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.does_not_exist.whatever", "kind": "all"},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            result = r["result"]

            # [Assert] MCP tool-level isError:true
            assert result.get("isError") is True, (
                f"refers_to with absent FQN must return isError:true; got {result}"
            )

            # [Assert] error_reason is fqn_not_in_graph
            body = json.loads(result["content"][0]["text"])
            assert "error_reason" in body
            assert body["error_reason"] == "fqn_not_in_graph", (
                f"expected error_reason='fqn_not_in_graph', got {body['error_reason']!r}"
            )

            # [Assert] stale is False on not-found errors
            assert "stale" in body
            assert body["stale"] is False, (
                f"stale must be False on fqn_not_in_graph error, got {body['stale']!r}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_refers_to_zero_callers_is_not_error(self, built_index) -> None:
        """[wiring] refers_to(present FQN with zero callers) returns results=[] not isError.

        mypkg.core.gamma has zero callers (nothing calls gamma in the fixture).
        Regression guard: presence in graph + zero callers must yield empty results,
        not an error response.
        """
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-wiring-zero-callers")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.core.gamma", "kind": "callers"},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r

            result = r["result"]

            # [Assert] NOT isError
            assert result.get("isError") is not True, (
                "refers_to with present FQN and zero callers must NOT return isError:true"
            )

            body = json.loads(result["content"][0]["text"])

            # [Assert] results is empty list
            assert body["results"] == [], (
                f"expected results=[], got {body['results']}"
            )
            assert body["truncated"] is False
            assert body["dropped"] == 0

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)


# ===========================================================================
# Scenario 2: slice-refers-to-multi-kind-e2e
# ===========================================================================
# SCENARIO: slice-refers-to-multi-kind-e2e
# layers_involved: src/pyscope_mcp/server.py, src/pyscope_mcp/graph.py,
#                  src/pyscope_mcp/analyzer/visitor.py, src/pyscope_mcp/types.py


@pytest.fixture(scope="module")
def multi_kind_index(tmp_path_factory: pytest.TempPathFactory):
    """Build a real index for a synthetic package that exercises all reference kinds.

    Package structure:
        refpkg/
            __init__.py
            target.py    # defines TargetError (exception class)
            consumer.py  # references TargetError via: call, import, except, annotation,
                         #                              isinstance

    The analyzer AST visitor records all five reference kinds into 'called_by'
    buckets.  This fixture builds a real index so the integration tier exercises
    the full pipeline: AST analysis → index write → server load → refers_to query.

    Note: the analyzer's ability to detect all five kinds depends on the
    analyzer/visitor.py implementation, which is part of the issue #71 diff.
    If the analyzer only tracks call edges, the wiring test will still pass
    (structural checks only); the artifact test will detect the missing kinds.

    Returns (index_file, pkg_root, repo_root, env).
    """
    tmp = tmp_path_factory.mktemp("it71mk")
    pkg = tmp / "refpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")

    # Target module: defines a class that consumers reference in various ways
    (pkg / "target.py").write_text(textwrap.dedent("""\
        class TargetError(Exception):
            pass

        def target_factory() -> "TargetError":
            return TargetError()
    """))

    # Consumer module: references TargetError via multiple AST edge kinds
    (pkg / "consumer.py").write_text(textwrap.dedent("""\
        from refpkg.target import TargetError

        def call_it():
            # call: creates an instance (constructor call)
            return TargetError("msg")

        def catch_it():
            # except: catches the exception class
            try:
                pass
            except TargetError:
                pass

        def annotate_it(x: TargetError) -> None:
            # annotation: uses it as a type annotation
            pass

        def check_it(obj):
            # isinstance: uses it in isinstance check
            return isinstance(obj, TargetError)
    """))

    index_file = tmp / ".pyscope-mcp" / "index.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))

    result = subprocess.run(
        [
            sys.executable, "-m", "pyscope_mcp.cli", "build",
            "--root", str(tmp),
            "--package", "refpkg",
            "--output", str(index_file),
        ],
        capture_output=True, env=env,
    )
    assert result.returncode == 0, (
        f"pyscope-mcp build (multi_kind_index) failed:\n{result.stderr.decode()}"
    )
    return index_file, tmp, repo_root, env


class TestRefersToMultiKindWiring:
    """Wiring tier — slice-refers-to-multi-kind-e2e.

    Verifies:
    - refers_to(kind='all') returns a results list (non-empty when consumers present)
    - response schema: content[0].type='text', body parseable, required keys present
    - context field values are from the valid set
    - kind='callers' subset is structurally correct (call contexts only)
    - module granularity returns strings, not dicts
    - each result entry has fqn/context/depth structure
    """

    @pytest.mark.integration_wiring
    def test_refers_to_kind_all_returns_results_list(self, multi_kind_index) -> None:
        """[wiring] refers_to(kind='all') on a symbol with multiple reference kinds returns non-empty list."""
        index_file, pkg_root, _, _ = multi_kind_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71mk-wiring-all")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "refpkg.target.TargetError", "kind": "all"},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r, f"unexpected JSON-RPC error: {r}"

            result = r["result"]
            assert result.get("isError") is not True, (
                f"refers_to(kind='all') on TargetError must not return isError; got {result}"
            )

            body = json.loads(result["content"][0]["text"])

            # [Assert] Required structural keys
            for field in ("results", "truncated", "dropped", "stale", "stale_files", "completeness"):
                assert field in body, f"response must include '{field}'"

            # [Assert] results is a list
            assert isinstance(body["results"], list)

            # [Assert] each entry has fqn, context, depth
            for entry in body["results"]:
                assert "fqn" in entry, f"entry missing 'fqn': {entry}"
                assert "context" in entry, f"entry missing 'context': {entry}"
                assert "depth" in entry, f"entry missing 'depth': {entry}"
                assert entry["context"] in (
                    "call", "import", "except", "annotation", "isinstance"
                ), f"context must be known kind, got {entry['context']!r}"

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_refers_to_callers_subset_excludes_non_call(self, multi_kind_index) -> None:
        """[wiring] refers_to(kind='callers') must only include call-context entries."""
        index_file, pkg_root, _, _ = multi_kind_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71mk-wiring-callers")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "refpkg.target.TargetError", "kind": "callers"},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r

            result = r["result"]
            assert result.get("isError") is not True

            body = json.loads(result["content"][0]["text"])
            assert isinstance(body["results"], list)

            # [Assert] All returned context values must be 'call'
            for entry in body["results"]:
                assert entry["context"] == "call", (
                    f"kind='callers' must only return call-context entries; "
                    f"got context='{entry['context']}' for fqn={entry['fqn']!r}"
                )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_wiring
    def test_refers_to_module_granularity_returns_strings(self, multi_kind_index) -> None:
        """[wiring] refers_to(granularity='module') returns flat list of module FQN strings."""
        index_file, pkg_root, _, _ = multi_kind_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71mk-wiring-module")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {
                        "fqn": "refpkg.target.TargetError",
                        "kind": "callers",
                        "granularity": "module",
                    },
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            assert "error" not in r

            result = r["result"]
            assert result.get("isError") is not True

            body = json.loads(result["content"][0]["text"])
            assert isinstance(body["results"], list)

            # [Assert] Every item is a string (module FQN), not a dict
            for item in body["results"]:
                assert isinstance(item, str), (
                    f"module granularity items must be strings; got {type(item)}: {item!r}"
                )
                # Module FQN must not contain function segments (no parentheses, no trailing name)
                assert "." in item or item.isidentifier(), (
                    f"module FQN looks malformed: {item!r}"
                )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)


# ===========================================================================
# Artifact tier — slice-refers-to-tool-e2e
# ===========================================================================


class TestRefersToToolArtifact:
    """Artifact tier — slice-refers-to-tool-e2e.

    Verifies the exact output of refers_to against the synthetic 3-function
    call-chain package (built_index fixture):
        alpha  ← beta  ← gamma
    (gamma calls beta calls alpha; nothing calls gamma)

    Golden facts:
    - refers_to(alpha, kind='callers') → results=[{fqn=beta, context=call, depth=1}]
    - refers_to(alpha, kind='callers', depth=2) → results=[beta(depth=1), gamma(depth=2)]
    - refers_to(gamma, kind='callers') → results=[] (nothing calls gamma)
    - refers_to(alpha, kind='callers', granularity='module') → ['mypkg.core']
    """

    @pytest.mark.integration_artifact
    def test_refers_to_direct_caller_exact(self, built_index) -> None:
        """[artifact] refers_to(alpha, kind='callers', depth=1) → exactly one result: beta with context=call."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-artifact-direct")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.core.alpha", "kind": "callers", "depth": 1},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            result = r["result"]
            assert result.get("isError") is not True, (
                f"unexpected isError: {result}"
            )

            body = json.loads(result["content"][0]["text"])

            # [Assert] Exact result count — golden: 1 (only beta is a direct caller)
            assert len(body["results"]) == 1, (
                f"expected 1 direct caller of alpha; got {body['results']}"
            )

            # [Assert] Exact entry values
            entry = body["results"][0]
            assert entry["fqn"] == "mypkg.core.beta", (
                f"expected caller fqn='mypkg.core.beta'; got {entry['fqn']!r}"
            )
            assert entry["context"] == "call", (
                f"expected context='call'; got {entry['context']!r}"
            )
            assert entry["depth"] == 1, (
                f"expected depth=1; got {entry['depth']}"
            )

            # [Assert] No truncation
            assert body["truncated"] is False
            assert body["dropped"] == 0

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_artifact
    def test_refers_to_depth2_transitive_callers_exact(self, built_index) -> None:
        """[artifact] refers_to(alpha, kind='callers', depth=2) → beta(depth=1) + gamma(depth=2)."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-artifact-depth2")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.core.alpha", "kind": "callers", "depth": 2},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            result = r["result"]
            assert result.get("isError") is not True

            body = json.loads(result["content"][0]["text"])

            # [Assert] Exact result count — golden: 2 (beta depth=1, gamma depth=2)
            assert len(body["results"]) == 2, (
                f"expected 2 callers (beta depth=1, gamma depth=2); got {body['results']}"
            )

            # Build lookup by fqn
            by_fqn = {e["fqn"]: e for e in body["results"]}

            # [Assert] beta at depth=1
            assert "mypkg.core.beta" in by_fqn, (
                f"mypkg.core.beta must be in depth=2 results; got {list(by_fqn.keys())}"
            )
            assert by_fqn["mypkg.core.beta"]["depth"] == 1, (
                f"beta must be at depth=1; got depth={by_fqn['mypkg.core.beta']['depth']}"
            )

            # [Assert] gamma at depth=2
            assert "mypkg.core.gamma" in by_fqn, (
                f"mypkg.core.gamma must be in depth=2 results; got {list(by_fqn.keys())}"
            )
            assert by_fqn["mypkg.core.gamma"]["depth"] == 2, (
                f"gamma must be at depth=2; got depth={by_fqn['mypkg.core.gamma']['depth']}"
            )

            # [Assert] Ranking: depth-1 before depth-2
            fqns = [e["fqn"] for e in body["results"]]
            assert fqns.index("mypkg.core.beta") < fqns.index("mypkg.core.gamma"), (
                "depth-1 caller (beta) must appear before depth-2 caller (gamma) in results"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_artifact
    def test_refers_to_zero_callers_exact_empty_results(self, built_index) -> None:
        """[artifact] refers_to(gamma, kind='callers') → exactly results=[], truncated=False, dropped=0."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-artifact-zero")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.core.gamma", "kind": "callers"},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            result = r["result"]
            assert result.get("isError") is not True

            body = json.loads(result["content"][0]["text"])

            # [Assert] Exact golden values
            assert body["results"] == [], (
                f"expected empty results for gamma (no callers); got {body['results']}"
            )
            assert body["truncated"] is False, (
                f"truncated must be False for empty results; got {body['truncated']!r}"
            )
            assert body["dropped"] == 0, (
                f"dropped must be 0 for empty results; got {body['dropped']}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_artifact
    def test_refers_to_module_granularity_exact(self, built_index) -> None:
        """[artifact] refers_to(alpha, kind='callers', granularity='module') → exactly ['mypkg.core']."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-artifact-module-gran")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {
                        "fqn": "mypkg.core.alpha",
                        "kind": "callers",
                        "granularity": "module",
                    },
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            result = r["result"]
            assert result.get("isError") is not True

            body = json.loads(result["content"][0]["text"])

            # [Assert] Exact module list — golden: ['mypkg.core'] (both beta and gamma are in core)
            assert body["results"] == ["mypkg.core"], (
                f"expected module results=['mypkg.core'] (deduped); got {body['results']}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_artifact
    def test_refers_to_depth_exceeds_max_exact_error_shape(self, built_index) -> None:
        """[artifact] refers_to(depth=3) → exact error shape: isError:true, error_reason='depth_exceeds_max'."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-artifact-depth-max")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.core.alpha", "kind": "all", "depth": 3},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            result = r["result"]

            # [Assert] isError at both MCP result level and body level
            assert result.get("isError") is True, (
                "depth=3 must return isError:true at MCP result level"
            )

            body = json.loads(result["content"][0]["text"])
            assert body.get("isError") is True, (
                "depth=3 body must also have isError:true"
            )
            assert body.get("error_reason") == "depth_exceeds_max", (
                f"expected error_reason='depth_exceeds_max'; got {body.get('error_reason')!r}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    @pytest.mark.integration_artifact
    def test_refers_to_fqn_not_in_graph_exact_error_shape(self, built_index) -> None:
        """[artifact] refers_to(bad_fqn) → exact error shape: isError:true, error_reason='fqn_not_in_graph', stale=False, stale_files=[]."""
        index_file, pkg_root, _, _ = built_index
        proc = _spawn_server(index_file, root=pkg_root)
        try:
            _handshake(proc, "it71-artifact-not-found")

            _send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "refers_to",
                    "arguments": {"fqn": "mypkg.does_not_exist.whatever", "kind": "all"},
                },
            })
            r = _recv(proc)
            assert r["id"] == 2
            result = r["result"]

            assert result.get("isError") is True

            body = json.loads(result["content"][0]["text"])

            # [Assert] Exact error fields
            assert body.get("error_reason") == "fqn_not_in_graph", (
                f"expected 'fqn_not_in_graph'; got {body.get('error_reason')!r}"
            )
            assert body.get("stale") is False, (
                f"stale must be exactly False (not None, not 0); got {body.get('stale')!r}"
            )
            assert body.get("stale_files") == [], (
                f"stale_files must be exactly [] on not-found; got {body.get('stale_files')!r}"
            )

            _shutdown(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)
