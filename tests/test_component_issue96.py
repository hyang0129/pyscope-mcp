"""Component tests for issue #96 — deferred index initialisation.

Three cross-module boundaries tested:

  B1 (server.py → graph.py): server.run_stdio() calls CallGraphIndex.load().
     When load raises FileNotFoundError (missing index), run_stdio must set
     _DEFERRED_ERROR to a non-None string containing "pyscope-mcp build" and
     leave _INDEX as None — without raising or crashing.
     When load raises ValueError (stale schema), the same deferred-error state
     applies.

  B2 (server.py → graph.py): server._dispatch_tool('reload') calls
     CallGraphIndex.load(). When the index is still broken, load raises again;
     _dispatch_tool must update _DEFERRED_ERROR, leave _INDEX as None, and
     return isError:true. When a valid index is placed on disk, _dispatch_tool
     must load it, clear _DEFERRED_ERROR, set _INDEX, and return isError:false.

  B3 (cli.py → server.py): cli.cmd_serve() passes the index path to
     server.run_stdio() unconditionally — no early-exit guard. If the index
     is missing, run_stdio enters deferred-error mode (B1 contract); cmd_serve
     must not raise or sys.exit before run_stdio is entered. The wiring is
     tested by patching asyncio.run to prevent the event loop from starting
     (component scope, no subprocess).
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import pyscope_mcp.server as _srv
from pyscope_mcp.graph import CallGraphIndex
from conftest import make_nodes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_valid_index(tmp_path: Path) -> Path:
    """Write a minimal valid index to disk and return its path."""
    raw: dict[str, list[str]] = {
        "pkg.mod.alpha": ["pkg.mod.beta"],
        "pkg.mod.beta": [],
    }
    idx = CallGraphIndex.from_nodes(tmp_path, make_nodes(raw))
    idx_path = tmp_path / "index.json"
    idx.save(idx_path)
    return idx_path


def _write_stale_schema_index(path: Path) -> None:
    """Write a v0 index (schema mismatch) to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 0, "root": "/", "raw": {}}))


def _reset_server_state() -> None:
    """Reset module-level server state between tests."""
    _srv._INDEX = None
    _srv._INDEX_PATH = None
    _srv._DEFERRED_ERROR = None
    _srv._SERVER._shutdown_requested = False


# ---------------------------------------------------------------------------
# B1 — server.run_stdio() → CallGraphIndex.load(): deferred-error state
# ---------------------------------------------------------------------------

class TestB1RunStdioDefersErrorOnMissingIndex:
    """B1: run_stdio sets _DEFERRED_ERROR when load raises FileNotFoundError."""

    def setup_method(self):
        _reset_server_state()

    def teardown_method(self):
        _reset_server_state()

    def test_run_stdio_missing_index_sets_deferred_error(self, tmp_path: Path) -> None:
        """[B1] run_stdio with missing index must set _DEFERRED_ERROR (not raise).

        Verifies: FileNotFoundError from load() is caught; _DEFERRED_ERROR is
        non-None and contains 'pyscope-mcp build' so the agent knows the remedy.
        """
        # Arrange — index file does not exist
        missing_index = tmp_path / ".pyscope-mcp" / "index.json"

        # Act — patch asyncio.run so the event loop never starts (component scope)
        with patch("asyncio.run"), patch("pyscope_mcp._log.init"):
            _srv.run_stdio(missing_index)

        # Assert
        assert _srv._DEFERRED_ERROR is not None, (
            "run_stdio must set _DEFERRED_ERROR when the index is missing, "
            "not crash or leave it None"
        )
        assert "pyscope-mcp build" in _srv._DEFERRED_ERROR, (
            f"_DEFERRED_ERROR must contain 'pyscope-mcp build' so the agent "
            f"knows the remedy; got: {_srv._DEFERRED_ERROR!r}"
        )

    def test_run_stdio_missing_index_leaves_index_none(self, tmp_path: Path) -> None:
        """[B1] run_stdio with missing index must leave _INDEX as None.

        Verifies the server does not enter a partial-load state where _INDEX
        points to a stale or empty object.
        """
        # Arrange
        missing_index = tmp_path / "no" / "such" / "index.json"

        # Act
        with patch("asyncio.run"), patch("pyscope_mcp._log.init"):
            _srv.run_stdio(missing_index)

        # Assert
        assert _srv._INDEX is None, (
            "run_stdio must leave _INDEX=None when the index file is missing; "
            f"got _INDEX={_srv._INDEX!r}"
        )

    def test_run_stdio_stale_schema_sets_deferred_error(self, tmp_path: Path) -> None:
        """[B1] run_stdio with stale-schema index (v0) must set _DEFERRED_ERROR.

        Verifies the ValueError path from CallGraphIndex.load() is caught and
        results in the same deferred-error state as the FileNotFoundError path.
        """
        # Arrange — v0 schema index on disk
        stale_index = tmp_path / "index.json"
        _write_stale_schema_index(stale_index)

        # Act
        with patch("asyncio.run"), patch("pyscope_mcp._log.init"):
            _srv.run_stdio(stale_index)

        # Assert
        assert _srv._DEFERRED_ERROR is not None, (
            "run_stdio must set _DEFERRED_ERROR when the index has a stale schema, "
            "not crash; the ValueError from load() must be caught"
        )
        assert "pyscope-mcp build" in _srv._DEFERRED_ERROR, (
            f"_DEFERRED_ERROR must contain 'pyscope-mcp build'; "
            f"got: {_srv._DEFERRED_ERROR!r}"
        )

    def test_run_stdio_stale_schema_leaves_index_none(self, tmp_path: Path) -> None:
        """[B1] run_stdio with stale-schema index must leave _INDEX=None."""
        # Arrange
        stale_index = tmp_path / "index.json"
        _write_stale_schema_index(stale_index)

        # Act
        with patch("asyncio.run"), patch("pyscope_mcp._log.init"):
            _srv.run_stdio(stale_index)

        # Assert
        assert _srv._INDEX is None, (
            "run_stdio must leave _INDEX=None for a stale-schema index"
        )

    def test_run_stdio_valid_index_clears_deferred_error(self, tmp_path: Path) -> None:
        """[B1] run_stdio with a valid index must set _INDEX and leave _DEFERRED_ERROR=None.

        Negative-case: the deferred-error path must NOT fire for a good index.
        """
        # Arrange — pre-arm a stale error to confirm run_stdio clears it
        _srv._DEFERRED_ERROR = "stale sentinel from previous run"
        idx_path = _make_valid_index(tmp_path)

        # Act
        with patch("asyncio.run"), patch("pyscope_mcp._log.init"):
            _srv.run_stdio(idx_path)

        # Assert
        assert _srv._DEFERRED_ERROR is None, (
            "run_stdio with a valid index must clear any pre-existing "
            f"_DEFERRED_ERROR; got: {_srv._DEFERRED_ERROR!r}"
        )
        assert _srv._INDEX is not None, (
            "run_stdio with a valid index must load and set _INDEX"
        )


# ---------------------------------------------------------------------------
# B2 — _dispatch_tool('reload') → CallGraphIndex.load(): state machine
# ---------------------------------------------------------------------------

class TestB2ReloadDispatchStateTransitions:
    """B2: _dispatch_tool('reload') correctly transitions _DEFERRED_ERROR state."""

    def setup_method(self):
        _reset_server_state()

    def teardown_method(self):
        _reset_server_state()

    @pytest.mark.asyncio
    async def test_reload_with_missing_index_returns_is_error(
        self, tmp_path: Path
    ) -> None:
        """[B2] reload on a still-missing index returns isError:true.

        The reload branch must call load(), receive FileNotFoundError,
        update _DEFERRED_ERROR, and return _error_result — not crash the server.
        """
        # Arrange — index path points to non-existent file
        missing_index = tmp_path / ".pyscope-mcp" / "index.json"
        _srv._INDEX_PATH = missing_index
        _srv._INDEX = None
        _srv._DEFERRED_ERROR = f"Index not found at {missing_index}. {_srv._DEFERRED_ERROR_ACTION}"

        # Act
        result = await _srv._dispatch_tool("reload", {})

        # Assert
        assert result.get("isError") is True, (
            "reload on a missing index must return isError:true; "
            f"got: {result}"
        )
        assert "content" in result, "isError result must have 'content' key"
        content_text = result["content"][0]["text"]
        assert "pyscope-mcp build" in content_text, (
            f"reload error message must contain 'pyscope-mcp build'; "
            f"got: {content_text!r}"
        )

    @pytest.mark.asyncio
    async def test_reload_with_missing_index_leaves_deferred_error_set(
        self, tmp_path: Path
    ) -> None:
        """[B2] reload on a still-missing index must NOT clear _DEFERRED_ERROR.

        After a failed reload, subsequent tool calls must still return
        isError:true — _DEFERRED_ERROR must be updated (not cleared).
        """
        # Arrange
        missing_index = tmp_path / ".pyscope-mcp" / "index.json"
        _srv._INDEX_PATH = missing_index
        _srv._INDEX = None
        _srv._DEFERRED_ERROR = None  # start clean; reload is the trigger

        # Act
        await _srv._dispatch_tool("reload", {})

        # Assert
        assert _srv._DEFERRED_ERROR is not None, (
            "_DEFERRED_ERROR must be set after reload fails on a missing index"
        )
        assert _srv._INDEX is None, (
            "_INDEX must remain None after a failed reload"
        )

    @pytest.mark.asyncio
    async def test_reload_with_valid_index_clears_deferred_error(
        self, tmp_path: Path
    ) -> None:
        """[B2] reload with a valid index on disk clears _DEFERRED_ERROR.

        After a successful reload, _DEFERRED_ERROR must be None and _INDEX
        must point to the loaded graph.
        """
        # Arrange — write a valid index; pre-arm the deferred-error state
        idx_path = _make_valid_index(tmp_path)
        _srv._INDEX_PATH = idx_path
        _srv._INDEX = None
        _srv._DEFERRED_ERROR = (
            f"Index not found at {idx_path}. {_srv._DEFERRED_ERROR_ACTION}"
        )

        # Act
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=128, stdout="")
            result = await _srv._dispatch_tool("reload", {})

        # Assert
        assert result.get("isError") is False, (
            "reload on a valid index must return isError:false; "
            f"got: {result}"
        )
        assert _srv._DEFERRED_ERROR is None, (
            "reload success must clear _DEFERRED_ERROR; "
            f"got: {_srv._DEFERRED_ERROR!r}"
        )
        assert _srv._INDEX is not None, (
            "reload success must set _INDEX to the loaded graph"
        )

    @pytest.mark.asyncio
    async def test_reload_success_enables_subsequent_tool_calls(
        self, tmp_path: Path
    ) -> None:
        """[B2] After a successful reload, index-dependent tools return isError:false.

        Verifies the full state-machine transition: deferred-error → reload →
        cleared → stats returns live data. This is the contract that agents
        depend on to recover from a missing-index startup.
        """
        # Arrange — write a valid index; simulate deferred-error startup state
        idx_path = _make_valid_index(tmp_path)
        _srv._INDEX_PATH = idx_path
        _srv._INDEX = None
        _srv._DEFERRED_ERROR = (
            f"Index not found at {idx_path}. {_srv._DEFERRED_ERROR_ACTION}"
        )

        # Confirm pre-condition: stats returns isError:true in deferred-error state
        pre_result = await _srv._dispatch_tool("stats", {})
        assert pre_result.get("isError") is True, (
            "pre-condition: stats must return isError:true in deferred-error state"
        )

        # Act — reload the now-valid index
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=128, stdout="")
            reload_result = await _srv._dispatch_tool("reload", {})
        assert reload_result.get("isError") is False, (
            f"reload must succeed now index is on disk; got: {reload_result}"
        )

        # Assert — subsequent stats call must return live data
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=128, stdout="")
            post_result = await _srv._dispatch_tool("stats", {})

        assert post_result.get("isError") is False, (
            "stats after successful reload must return isError:false; "
            f"got: {post_result}"
        )
        payload = json.loads(post_result["content"][0]["text"])
        assert "functions" in payload, (
            f"stats payload must include 'functions' after reload; "
            f"got keys: {list(payload.keys())}"
        )


# ---------------------------------------------------------------------------
# B3 — cli.cmd_serve() → server.run_stdio(): unconditional wiring contract
# ---------------------------------------------------------------------------

class TestB3CmdServeWiresRunStdioUnconditionally:
    """B3: cli.cmd_serve() passes index path to server.run_stdio() without early exit.

    The previous code had an early-exit guard:
        if not index.exists():
            print(...)
            return 2
    Issue #96 removes this guard so run_stdio() is always called — entering
    deferred-error mode if the index is missing.

    These tests verify the wiring at component scope (no subprocess). They
    patch asyncio.run to prevent the event loop from starting and inspect
    what state was established before the loop would start.
    """

    def setup_method(self):
        _reset_server_state()

    def teardown_method(self):
        _reset_server_state()

    def test_cmd_serve_calls_run_stdio_for_missing_index(self, tmp_path: Path) -> None:
        """[B3] cmd_serve with a missing index path calls run_stdio (not sys.exit/return 2).

        Verifies the early-exit guard has been removed. If run_stdio is called,
        _DEFERRED_ERROR will be set (B1 contract). If the guard is still present,
        run_stdio is never entered and _DEFERRED_ERROR stays None.
        """
        # Arrange
        missing_index = tmp_path / ".pyscope-mcp" / "index.json"

        import pyscope_mcp.cli as cli_mod

        # Build an argparse.Namespace that cmd_serve expects
        args = types.SimpleNamespace(
            root=str(tmp_path),
            index=str(missing_index),
        )

        # Act — patch asyncio.run so no event loop starts; patch _log.init
        with patch("asyncio.run"), patch("pyscope_mcp._log.init"):
            cli_mod.cmd_serve(args)

        # Assert — B1 contract: run_stdio entered deferred-error mode
        assert _srv._DEFERRED_ERROR is not None, (
            "cmd_serve with a missing index must call run_stdio, which sets "
            "_DEFERRED_ERROR. If _DEFERRED_ERROR is None, the early-exit guard "
            "was not removed and run_stdio was never entered."
        )
        assert "pyscope-mcp build" in _srv._DEFERRED_ERROR, (
            f"Deferred error message must contain 'pyscope-mcp build'; "
            f"got: {_srv._DEFERRED_ERROR!r}"
        )

    def test_cmd_serve_missing_index_does_not_raise(self, tmp_path: Path) -> None:
        """[B3] cmd_serve with missing index must not raise any exception.

        Before issue #96, the early-exit guard only returned 2 (no exception).
        After the fix, run_stdio catches the error. Neither path must raise.
        """
        # Arrange
        missing_index = tmp_path / "no" / "index.json"

        import pyscope_mcp.cli as cli_mod

        args = types.SimpleNamespace(
            root=str(tmp_path),
            index=str(missing_index),
        )

        # Act + Assert — no exception raised
        with patch("asyncio.run"), patch("pyscope_mcp._log.init"):
            try:
                cli_mod.cmd_serve(args)
            except Exception as exc:
                pytest.fail(
                    f"cmd_serve raised {type(exc).__name__}: {exc} — "
                    "it must not raise when the index is missing"
                )

    def test_cmd_serve_valid_index_reaches_run_stdio_with_live_index(
        self, tmp_path: Path
    ) -> None:
        """[B3] cmd_serve with a valid index passes it to run_stdio, which loads it.

        Negative-case: confirms the unconditional wiring works for the happy path
        too — run_stdio loads the index and _DEFERRED_ERROR stays None.
        """
        # Arrange — write a valid index
        idx_path = _make_valid_index(tmp_path)

        import pyscope_mcp.cli as cli_mod

        args = types.SimpleNamespace(
            root=str(tmp_path),
            index=str(idx_path),
        )

        # Act
        with patch("asyncio.run"), patch("pyscope_mcp._log.init"):
            cli_mod.cmd_serve(args)

        # Assert
        assert _srv._DEFERRED_ERROR is None, (
            "cmd_serve with a valid index must NOT set _DEFERRED_ERROR; "
            f"got: {_srv._DEFERRED_ERROR!r}"
        )
        assert _srv._INDEX is not None, (
            "cmd_serve with a valid index must load _INDEX via run_stdio"
        )

    def test_cmd_serve_index_path_is_passed_to_run_stdio(self, tmp_path: Path) -> None:
        """[B3] cmd_serve passes the resolved index path to run_stdio.

        Verifies that _INDEX_PATH set by run_stdio matches the path cmd_serve
        computed — the absolute resolved path must be recorded in server state
        so reload/build tools can locate the index later.
        """
        # Arrange — valid index at a known path
        idx_path = _make_valid_index(tmp_path)

        import pyscope_mcp.cli as cli_mod

        args = types.SimpleNamespace(
            root=str(tmp_path),
            index=str(idx_path),
        )

        # Act
        with patch("asyncio.run"), patch("pyscope_mcp._log.init"):
            cli_mod.cmd_serve(args)

        # Assert — _INDEX_PATH must be set and resolve to the same absolute path
        assert _srv._INDEX_PATH is not None, (
            "cmd_serve must set _INDEX_PATH via run_stdio"
        )
        assert Path(_srv._INDEX_PATH).resolve() == idx_path.resolve(), (
            f"_INDEX_PATH must match the path cmd_serve passed to run_stdio; "
            f"got {_srv._INDEX_PATH!r}, expected {idx_path!r}"
        )
