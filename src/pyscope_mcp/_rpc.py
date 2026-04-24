"""Hand-rolled JSON-RPC 2.0 stdio transport for pyscope-mcp.

Design notes
------------
* stdio only — no SSE/HTTP. Accepted trade-off; see issue #40.
* Serial request processing in v1. All handlers are async def but do sync
  work today. A future long-running tool (e.g. reindex) must revisit
  concurrency explicitly rather than silently blocking the loop.
* Batch requests (JSON arrays) are rejected with -32600. MCP does not use
  them and rejecting them early prevents accidentally processing partial
  batches.
* stdout hygiene: the RPC writer holds a direct reference to the original
  stdout buffer captured at import time. Any code that later replaces
  sys.stdout will not corrupt the protocol stream.
* Unknown top-level fields (_meta, progressToken, future additions) are
  silently ignored — permissive parsing by design.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
from collections.abc import Awaitable, Callable
from typing import Any

__all__ = [
    "RpcServer",
    "RpcError",
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 standard error codes
# ---------------------------------------------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

_ERROR_MESSAGES: dict[int, str] = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid Request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
}

# Capture original stdout buffer *before* any sys.stdout replacement.
# This reference is the only path that writes to the protocol stream.
_RAW_STDOUT = sys.stdout.buffer


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------
class RpcError(Exception):
    """Raised inside method handlers to return a JSON-RPC error response."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------
def _error_obj(code: int, message: str | None = None, data: Any = None) -> dict:
    return {
        "code": code,
        "message": message or _ERROR_MESSAGES.get(code, "Error"),
        **({"data": data} if data is not None else {}),
    }


def _response(id: Any, result: Any) -> bytes:
    return (json.dumps({"jsonrpc": "2.0", "id": id, "result": result}) + "\n").encode()


def _error_response(id: Any, code: int, message: str | None = None, data: Any = None) -> bytes:
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": id,
                "error": _error_obj(code, message, data),
            }
        )
        + "\n"
    ).encode()


# ---------------------------------------------------------------------------
# Method handler type
# ---------------------------------------------------------------------------
Handler = Callable[[Any, dict | None], Awaitable[Any]]


# ---------------------------------------------------------------------------
# RpcServer
# ---------------------------------------------------------------------------
class RpcServer:
    """Minimal JSON-RPC 2.0 server that dispatches over stdio.

    Usage::

        server = RpcServer(name="pyscope-mcp", version="0.1.0")

        @server.method("ping")
        async def handle_ping(id, params):
            return {}

        asyncio.run(server.run())
    """

    # Known protocol versions. Echo back if the client sends one we know;
    # fall back to newest if unknown (but still in the same major family).
    _KNOWN_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
    _NEWEST_VERSION = "2025-06-18"

    def __init__(self, name: str, version: str, instructions: str = "") -> None:
        self._name = name
        self._version = version
        self._instructions = instructions
        self._handlers: dict[str, Handler] = {}
        self._shutdown_requested = False

        # Register lifecycle methods
        self.method("initialize")(self._handle_initialize)
        self.method("notifications/initialized")(self._handle_noop)
        self.method("notifications/cancelled")(self._handle_noop)  # Ignored: v1 is serial-dispatch with no long-running tools to cancel. Future concurrency MUST revisit.
        self.method("ping")(self._handle_ping)
        self.method("shutdown")(self._handle_shutdown)

    # ------------------------------------------------------------------
    # Decorator for registering method handlers
    # ------------------------------------------------------------------
    def method(self, name: str) -> Callable[[Handler], Handler]:
        def decorator(fn: Handler) -> Handler:
            self._handlers[name] = fn
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Built-in lifecycle handlers
    # ------------------------------------------------------------------
    async def _handle_initialize(self, id: Any, params: dict | None) -> dict:  # noqa: A002
        p = params or {}
        client_version = p.get("protocolVersion", self._NEWEST_VERSION)

        if client_version in self._KNOWN_VERSIONS:
            negotiated = client_version
        else:
            # Unknown version — check if it looks like a future minor bump we
            # can tolerate.  Simple heuristic: accept if the year part is
            # recognisable; otherwise return newest.
            negotiated = self._NEWEST_VERSION

        result_dict = {
            "protocolVersion": negotiated,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self._name, "version": self._version},
            **({"instructions": self._instructions} if self._instructions else {}),
        }
        return result_dict

    async def _handle_noop(self, id: Any, params: dict | None) -> None:  # noqa: A002
        return None

    async def _handle_ping(self, id: Any, params: dict | None) -> dict:  # noqa: A002
        return {}

    async def _handle_shutdown(self, id: Any, params: dict | None) -> dict:  # noqa: A002
        # MCP shutdown is two-phase: this method prepares (sets flag for introspection);
        # the client closes stdin, which causes _loop to exit on EOF.
        self._shutdown_requested = True
        return {}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Read JSON-RPC messages from stdin, dispatch, write responses to stdout."""

        # Install stderr logging before anything else so no stray writes hit stdout.
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
        for handler in logging.root.handlers:
            handler.stream = sys.stderr  # type: ignore[attr-defined]

        # Replace sys.stdout so accidental print() calls go to stderr.
        # The RPC writer uses _RAW_STDOUT directly.
        _orig_stdout = sys.stdout
        _safe = io.TextIOWrapper(
            io.FileIO(sys.stderr.fileno(), mode="w", closefd=False),
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
        sys.stdout = _safe

        loop = asyncio.get_running_loop()

        # Async stdin reader
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        # Async stdout writer (using the raw buffer captured at module import).
        # Must use FlowControlMixin (not BaseProtocol) — StreamWriter.drain()
        # calls protocol._drain_helper(), which only exists on FlowControlMixin.
        from asyncio.streams import FlowControlMixin  # type: ignore[attr-defined]  # Private CPython symbol — no compat guarantee; tested in e2e against target Python version.

        write_transport, write_protocol = await loop.connect_write_pipe(
            lambda: FlowControlMixin(loop=loop), _RAW_STDOUT
        )
        writer = asyncio.StreamWriter(  # type: ignore[call-arg]
            write_transport, write_protocol, reader, loop
        )

        try:
            await self._loop(reader, writer)
        finally:
            try:
                await writer.drain()
            except Exception:
                pass
            sys.stdout = _orig_stdout

    async def _loop(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        async for line in reader:
            line = line.rstrip(b"\r\n")
            if not line:
                continue

            # --- Parse ---
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                writer.write(
                    _error_response(None, PARSE_ERROR, f"Parse error: {exc}")
                )
                await writer.drain()
                continue

            # --- Reject batch ---
            if isinstance(msg, list):
                writer.write(
                    _error_response(
                        None,
                        INVALID_REQUEST,
                        "Batch requests are not supported",
                    )
                )
                await writer.drain()
                continue

            # --- Validate structure ---
            if not isinstance(msg, dict):
                writer.write(
                    _error_response(None, INVALID_REQUEST, "Request must be a JSON object")
                )
                await writer.drain()
                continue

            req_id = msg.get("id")  # may be absent for notifications
            method = msg.get("method")
            params = msg.get("params")  # may be absent

            if msg.get("jsonrpc") != "2.0" or not isinstance(method, str):
                writer.write(
                    _error_response(
                        req_id,
                        INVALID_REQUEST,
                        "Missing or invalid 'jsonrpc'/'method' field",
                    )
                )
                await writer.drain()
                continue

            # --- Dispatch ---
            handler = self._handlers.get(method)
            if handler is None:
                # Notifications with no handler are silently dropped per spec
                if req_id is None:
                    continue
                writer.write(_error_response(req_id, METHOD_NOT_FOUND))
                await writer.drain()
                continue

            # Notifications: no response
            is_notification = req_id is None
            try:
                result = await handler(req_id, params)
            except RpcError as exc:
                if not is_notification:
                    writer.write(
                        _error_response(req_id, exc.code, exc.message, exc.data)
                    )
                    await writer.drain()
                continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unhandled exception in handler for %r", method)
                if not is_notification:
                    writer.write(
                        _error_response(req_id, INTERNAL_ERROR, str(exc))
                    )
                    await writer.drain()
                continue

            if not is_notification:
                writer.write(_response(req_id, result))
                await writer.drain()

        # EOF — clean exit.
        # Termination contract: the loop relies on the client closing stdin (EOF)
        # as the actual termination signal. Well-behaved MCP clients close the pipe
        # after sending the `exit` notification that follows `shutdown`.
        logger.debug("stdin EOF — RPC loop exiting cleanly")
