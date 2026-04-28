"""Unit tests for commit-SHA staleness detection and `build` MCP tool.

Covers:
  - _commit_staleness(): clean (HEAD == index SHA), stale (HEAD != index SHA)
  - _commit_staleness(): git unavailable (FileNotFoundError) → all None
  - _commit_staleness(): index built outside git (git_sha=None) → all None
  - _commit_staleness(): git returns non-zero → head_sha = None, all None
  - All query tool responses include commit_stale / index_git_sha / head_git_sha
  - build tool: success path (subprocess succeeds, index reloaded)
  - build tool: concurrent rejection (lock held → isError: true)
  - build tool: subprocess failure (non-zero exit → isError: true)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyscope_mcp.graph import CallGraphIndex, SymbolSummary
from conftest import make_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym(fqn: str, kind: str = "function", lineno: int = 1) -> SymbolSummary:
    return SymbolSummary(
        fqn=fqn,
        kind=kind,
        signature=f"def {fqn.split('.')[-1]}(): ...",
        lineno=lineno,
    )


def _make_idx(
    tmp_path: Path,
    raw: dict[str, list[str]] | None = None,
    git_sha: str | None = None,
) -> CallGraphIndex:
    return CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw or {}), git_sha=git_sha)


_MOCK_HEAD = "abcdef1234567890abcdef1234567890abcdef12"
_MOCK_INDEX_SHA = "0000000000000000000000000000000000000000"


def _mock_git_success(sha: str = _MOCK_HEAD):
    """Return a mock for subprocess.run that reports git SHA successfully."""
    result = MagicMock()
    result.returncode = 0
    result.stdout = sha + "\n"
    return result


def _mock_git_failure(returncode: int = 1):
    """Return a mock for subprocess.run that reports git failure."""
    result = MagicMock()
    result.returncode = returncode
    result.stdout = ""
    return result


# ---------------------------------------------------------------------------
# 1. _commit_staleness() unit tests
# ---------------------------------------------------------------------------

def test_commit_staleness_clean(tmp_path: Path) -> None:
    """index_git_sha == HEAD → commit_stale: False."""
    idx = _make_idx(tmp_path, git_sha=_MOCK_HEAD)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx._commit_staleness()
    assert result["commit_stale"] is False
    assert result["index_git_sha"] == _MOCK_HEAD
    assert result["head_git_sha"] == _MOCK_HEAD


def test_commit_staleness_stale(tmp_path: Path) -> None:
    """index_git_sha != HEAD → commit_stale: True."""
    idx = _make_idx(tmp_path, git_sha=_MOCK_INDEX_SHA)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx._commit_staleness()
    assert result["commit_stale"] is True
    assert result["index_git_sha"] == _MOCK_INDEX_SHA
    assert result["head_git_sha"] == _MOCK_HEAD


def test_commit_staleness_git_unavailable(tmp_path: Path) -> None:
    """git binary absent → all fields None."""
    idx = _make_idx(tmp_path, git_sha=_MOCK_INDEX_SHA)
    with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
        result = idx._commit_staleness()
    assert result["commit_stale"] is None
    assert result["index_git_sha"] is None
    assert result["head_git_sha"] is None


def test_commit_staleness_git_nonzero(tmp_path: Path) -> None:
    """git rev-parse returns non-zero → all fields None."""
    idx = _make_idx(tmp_path, git_sha=_MOCK_INDEX_SHA)
    with patch("subprocess.run", return_value=_mock_git_failure(returncode=128)):
        result = idx._commit_staleness()
    assert result["commit_stale"] is None
    assert result["index_git_sha"] is None
    assert result["head_git_sha"] is None


def test_commit_staleness_no_index_sha(tmp_path: Path) -> None:
    """Index built outside git checkout (git_sha=None) → all fields None."""
    idx = _make_idx(tmp_path, git_sha=None)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx._commit_staleness()
    assert result["commit_stale"] is None
    assert result["index_git_sha"] is None
    assert result["head_git_sha"] is None


# ---------------------------------------------------------------------------
# 2. Query tools include commit staleness fields
# ---------------------------------------------------------------------------

def _make_idx_with_raw(tmp_path: Path, git_sha: str = _MOCK_INDEX_SHA) -> CallGraphIndex:
    raw = {
        "pkg.mod.fn_a": ["pkg.mod.fn_b"],
        "pkg.mod.fn_b": [],
    }
    return CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw), file_shas={}, git_sha=git_sha)


def test_callers_of_includes_commit_staleness(tmp_path: Path) -> None:
    idx = _make_idx_with_raw(tmp_path, git_sha=_MOCK_INDEX_SHA)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx.refers_to("pkg.mod.fn_b", kind="callers", depth=1)
    assert "commit_stale" in result
    assert "index_git_sha" in result
    assert "head_git_sha" in result
    assert result["commit_stale"] is True
    assert result["index_git_sha"] == _MOCK_INDEX_SHA
    assert result["head_git_sha"] == _MOCK_HEAD


def test_callees_of_includes_commit_staleness(tmp_path: Path) -> None:
    idx = _make_idx_with_raw(tmp_path, git_sha=_MOCK_INDEX_SHA)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx.callees_of("pkg.mod.fn_a", depth=1)
    assert "commit_stale" in result
    assert result["commit_stale"] is True


def test_search_includes_commit_staleness(tmp_path: Path) -> None:
    idx = _make_idx_with_raw(tmp_path, git_sha=_MOCK_INDEX_SHA)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx.search("fn_a")
    assert "commit_stale" in result
    assert result["commit_stale"] is True


def test_stats_includes_commit_staleness(tmp_path: Path) -> None:
    idx = _make_idx_with_raw(tmp_path, git_sha=_MOCK_INDEX_SHA)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx.stats()
    assert "commit_stale" in result
    assert result["commit_stale"] is True
    assert result["functions"] >= 0  # still has normal stats fields


def test_stats_clean_commit(tmp_path: Path) -> None:
    """stats() with HEAD == index SHA → commit_stale: False."""
    idx = _make_idx_with_raw(tmp_path, git_sha=_MOCK_HEAD)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx.stats()
    assert result["commit_stale"] is False


def test_neighborhood_includes_commit_staleness(tmp_path: Path) -> None:
    idx = _make_idx_with_raw(tmp_path, git_sha=_MOCK_INDEX_SHA)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx.neighborhood("pkg.mod.fn_a", depth=1)
    assert "commit_stale" in result
    assert result["commit_stale"] is True


def test_module_callers_includes_commit_staleness(tmp_path: Path) -> None:
    idx = _make_idx_with_raw(tmp_path, git_sha=_MOCK_INDEX_SHA)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx.refers_to("pkg.mod.fn_b", kind="callers", granularity="module", depth=1)
    assert "commit_stale" in result


def test_module_callees_includes_commit_staleness(tmp_path: Path) -> None:
    idx = _make_idx_with_raw(tmp_path, git_sha=_MOCK_INDEX_SHA)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx.module_callees("pkg.mod", depth=1)
    assert "commit_stale" in result


def test_file_skeleton_includes_commit_staleness(tmp_path: Path) -> None:
    """file_skeleton returns commit staleness fields."""
    skeletons = {"mod.py": [_sym("pkg.mod.fn_a")]}
    (tmp_path / "mod.py").write_text("def fn_a(): pass\n")
    import hashlib
    shas = {"mod.py": hashlib.sha256(b"def fn_a(): pass\n").hexdigest()}
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes({}), skeletons=skeletons, file_shas=shas, git_sha=_MOCK_INDEX_SHA)
    with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
        result = idx.file_skeleton("mod.py")
    assert "commit_stale" in result
    assert result["commit_stale"] is True


# ---------------------------------------------------------------------------
# 3. build MCP tool (server-level tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def server_with_index(tmp_path: Path):
    """Spin up a server module state with a minimal saved index."""
    from pyscope_mcp import server as srv

    raw = {"pkg.mod.fn_a": ["pkg.mod.fn_b"], "pkg.mod.fn_b": []}
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes(raw), git_sha=_MOCK_INDEX_SHA)
    index_path = tmp_path / "index.json"
    idx.save(index_path)

    # Patch module-level state
    srv._INDEX_PATH = index_path
    srv._INDEX = idx
    srv._BUILD_LOCK = None  # reset lock between tests

    yield srv, tmp_path

    # Clean up module state
    srv._INDEX_PATH = None
    srv._INDEX = None
    srv._BUILD_LOCK = None


@pytest.mark.asyncio
async def test_build_tool_success(server_with_index, tmp_path: Path) -> None:
    """build tool: subprocess succeeds → reloads index, returns stats."""
    srv, _ = server_with_index

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stderr = ""

    with patch("pyscope_mcp.server.subprocess.run", return_value=mock_proc):
        with patch("subprocess.run", return_value=_mock_git_success(_MOCK_HEAD)):
            result = await srv._dispatch_tool("build", {})

    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert "functions" in payload
    assert "function_edges" in payload


@pytest.mark.asyncio
async def test_build_tool_concurrent_rejection(server_with_index) -> None:
    """Concurrent build calls: second call returns isError:true immediately."""
    import asyncio
    srv, _ = server_with_index
    srv._BUILD_LOCK = None  # ensure fresh lock

    # Manually acquire the lock to simulate a build in progress
    lock = srv._get_build_lock()
    await lock.acquire()
    try:
        result = await srv._dispatch_tool("build", {})
    finally:
        lock.release()

    assert result["isError"] is True
    assert "already in progress" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_build_tool_subprocess_failure(server_with_index) -> None:
    """build tool: subprocess fails → isError:true with exit code and stderr."""
    srv, _ = server_with_index

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "analyzer error: syntax error in foo.py"

    with patch("pyscope_mcp.server.subprocess.run", return_value=mock_proc):
        result = await srv._dispatch_tool("build", {})

    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "build failed" in text
    assert "1" in text  # exit code


@pytest.mark.asyncio
async def test_build_tool_no_index_path(tmp_path: Path) -> None:
    """build tool with no index path raises RuntimeError (→ caught as tool error)."""
    from pyscope_mcp import server as srv

    original_path = srv._INDEX_PATH
    srv._INDEX_PATH = None
    srv._BUILD_LOCK = None
    try:
        with pytest.raises(RuntimeError, match="without an index path"):
            await srv._dispatch_tool("build", {})
    finally:
        srv._INDEX_PATH = original_path
