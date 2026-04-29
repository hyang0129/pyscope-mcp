"""Lightweight JSONL query logger for the pyscope-mcp serve path.

Each tools/call dispatch appends one structured entry describing inputs,
response shape stats, latency, error state, and the loaded index identity.

The log is write-only from the server's perspective and never feeds back into
tool responses.  It is synchronous (one write+flush per call); async/buffered
writes are deferred to a future follow-up.

Configuration (env vars, read at init time):
  PYSCOPE_MCP_LOG      "1" to enable, "0" to disable.
                       Defaults to "1" (on). Set to "0" to disable.
                       On activation, ``init`` emits a one-time WARNING
                       through the ``pyscope_mcp._log`` logger so users
                       see the active log path on stderr at startup.
  PYSCOPE_MCP_LOG_PATH Path to the log file.
                       Default: <index_dir>/.pyscope-mcp/query.jsonl
                       (The .pyscope-mcp/ dir is already .gitignored.)

Rotation: when the log file exceeds LOG_MAX_BYTES (10 MB), it is renamed to
``query.jsonl.1`` … ``query.jsonl.5`` (max 5 historical files); the oldest is
deleted when a 6th rotation would occur.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyscope_mcp.graph import CallGraphIndex

logger = logging.getLogger(__name__)

LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: int = 5
LOG_SCHEMA_VERSION: int = 1


class QueryLogger:
    """Appends JSONL entries to a rotating log file.

    Thread-safety: not required — the serve path is single-threaded (serial
    asyncio dispatch).

    Instantiate via :func:`init`.  Check :data:`_LOGGER` before calling
    :func:`log_call`; when logging is disabled the module-level singleton is
    ``None`` and the check is a single ``is None`` guard.
    """

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._warned = False  # emit at most one warning per session on I/O failure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        *,
        server_id: str,
        rpc_id: Any,
        tool: str,
        args: dict,
        duration_ms: int,
        result: dict,
        index: "CallGraphIndex | None",
    ) -> None:
        """Build and append one JSONL entry.  Swallows all exceptions internally."""
        try:
            entry = self._build_entry(
                server_id=server_id,
                rpc_id=rpc_id,
                tool=tool,
                args=args,
                duration_ms=duration_ms,
                result=result,
                index=index,
            )
            self._append(json.dumps(entry, separators=(",", ":")))
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                logger.warning("QueryLogger: failed to write log entry: %s", exc)
                self._warned = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_entry(
        self,
        *,
        server_id: str,
        rpc_id: Any,
        tool: str,
        args: dict,
        duration_ms: int,
        result: dict,
        index: "CallGraphIndex | None",
    ) -> dict:
        is_error: bool = bool(result.get("isError", False))

        # Extract response payload from MCP tool-result envelope.
        # _text() wraps the real payload in {"content": [{"type": "text", "text": "..."}], "isError": False}.
        payload: dict = {}
        if not is_error:
            try:
                import json as _json
                content = result.get("content", [])
                if content and content[0].get("type") == "text":
                    payload = _json.loads(content[0]["text"])
            except Exception:  # noqa: BLE001
                payload = {}

        # Shape stats — derived from payload fields by name.
        truncated: bool | None = payload.get("truncated") if "truncated" in payload else None
        result_count: int | None = None
        if "results" in payload and isinstance(payload["results"], list):
            result_count = len(payload["results"])
        edge_count: int | None = None
        if "edges" in payload and isinstance(payload["edges"], list):
            edge_count = len(payload["edges"])
        hub_suppressed_count: int | None = None
        if "hub_suppressed" in payload and isinstance(payload["hub_suppressed"], list):
            hub_suppressed_count = len(payload["hub_suppressed"])
        depth_full: int | None = payload.get("depth_full")
        token_budget_used: int | None = payload.get("token_budget_used")

        entry: dict = {
            "v": LOG_SCHEMA_VERSION,
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
            "server_id": server_id,
            "rpc_id": rpc_id,
            "tool": tool,
            "args": args,
            "duration_ms": duration_ms,
            "is_error": is_error,
            "truncated": truncated,
            "result_count": result_count,
            "edge_count": edge_count,
            "hub_suppressed_count": hub_suppressed_count,
            "depth_full": depth_full,
            "token_budget_used": token_budget_used,
            "index_version": _index_version(index),
            "index_git_sha": index.git_sha if index is not None else None,
            "index_content_hash": index.content_hash if index is not None else None,
        }
        if is_error:
            # Extract error message text.
            try:
                content = result.get("content", [])
                if content and content[0].get("type") == "text":
                    entry["error_msg"] = content[0]["text"]
            except Exception:  # noqa: BLE001
                entry["error_msg"] = str(result)

        return entry

    def _append(self, line: str) -> None:
        """Write one line to the log file, rotating if necessary."""
        self._maybe_rotate()
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def _maybe_rotate(self) -> None:
        """Rotate the log file if it has reached LOG_MAX_BYTES."""
        try:
            size = self._log_path.stat().st_size if self._log_path.exists() else 0
        except OSError:
            size = 0
        if size < LOG_MAX_BYTES:
            return
        # Delete the oldest backup if at the limit.
        oldest = self._log_path.with_suffix(self._log_path.suffix + f".{LOG_BACKUP_COUNT}")
        if oldest.exists():
            oldest.unlink()
        # Shift existing backups up by one.
        for i in range(LOG_BACKUP_COUNT - 1, 0, -1):
            src = self._log_path.with_suffix(self._log_path.suffix + f".{i}")
            dst = self._log_path.with_suffix(self._log_path.suffix + f".{i + 1}")
            if src.exists():
                src.rename(dst)
        # Rename current log to .1
        if self._log_path.exists():
            self._log_path.rename(self._log_path.with_suffix(self._log_path.suffix + ".1"))


# ---------------------------------------------------------------------------
# Module-level singleton and public interface
# ---------------------------------------------------------------------------

def _index_version(index: "CallGraphIndex | None") -> int | None:
    """Return INDEX_VERSION from the graph module (avoids circular import at module level)."""
    if index is None:
        return None
    try:
        from pyscope_mcp.graph import INDEX_VERSION
        return INDEX_VERSION
    except Exception:  # noqa: BLE001
        return None


_LOGGER: QueryLogger | None = None


def init(log_path: Path) -> None:
    """Initialise the module-level logger singleton.

    Safe to call multiple times; subsequent calls replace the previous instance.
    No-op when ``PYSCOPE_MCP_LOG`` is ``"0"``.

    On activation, emits a one-time WARNING-level log message announcing
    the active log path. Python's ``logging.lastResort`` handler routes
    WARNING+ to stderr by default, which makes the announcement visible
    to a human starting the server even before ``_rpc.RpcServer.run()``
    has set up its own logging handlers.
    """
    global _LOGGER
    enabled = _is_enabled()
    if not enabled:
        _LOGGER = None
        return
    _LOGGER = QueryLogger(log_path)
    logger.warning(
        "Query logging enabled (PYSCOPE_MCP_LOG=1 default). "
        "Logs: %s (rotates at %d MB \u00d7 %d backups). "
        "Set PYSCOPE_MCP_LOG=0 to disable.",
        log_path,
        LOG_MAX_BYTES // (1024 * 1024),
        LOG_BACKUP_COUNT,
    )


def _is_enabled() -> bool:
    """Return True when logging is enabled.

    Logging is on by default.  Set ``PYSCOPE_MCP_LOG=0`` to disable.
    """
    return os.environ.get("PYSCOPE_MCP_LOG", "1") not in ("0", "false", "False", "no", "No")


def log_call(
    *,
    server_id: str,
    rpc_id: Any,
    tool: str,
    args: dict,
    duration_ms: int,
    result: dict,
    index: "CallGraphIndex | None",
) -> None:
    """Append a log entry if the logger is initialised.  Never raises."""
    if _LOGGER is None:
        return
    try:
        _LOGGER.write(
            server_id=server_id,
            rpc_id=rpc_id,
            tool=tool,
            args=args,
            duration_ms=duration_ms,
            result=result,
            index=index,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("QueryLogger.write failed (log_call guard): %s", exc)
