"""pyscope-mcp MCP server — hand-rolled JSON-RPC 2.0 stdio transport.

This module registers all tool handlers against the lightweight _rpc.RpcServer
instead of the mcp SDK. The mcp SDK (and its transitive deps httpx, pydantic,
anyio, jsonschema, pydantic_settings) are no longer required at serve time.

Accepted trade-offs (one-way doors, see issue #40):
* stdio transport only — no SSE / streamable HTTP.
* No resources, prompts, sampling, elicitation, or roots MCP surface.
* No auto-tracking of future MCP spec revisions.
* No SDK in-process test client — we use a pipe harness in tests.

Do NOT add print() calls to this module. stdout belongs to the RPC stream.
All diagnostic output must go through logging (which is routed to stderr by
_rpc.RpcServer.run() before any request is processed).
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from pathlib import Path

from pyscope_mcp import __version__
from pyscope_mcp._rpc import RpcError, RpcServer
from pyscope_mcp.graph import CallGraphIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (process-wide singleton)
# ---------------------------------------------------------------------------
_INDEX: CallGraphIndex | None = None
_INDEX_PATH: Path | None = None

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
_SERVER = RpcServer(
    name="pyscope-mcp",
    version=__version__,
    instructions=(
        "Query the prebuilt Python call-graph index. "
        "Use stats to check graph size, callers_of/callees_of for function-level "
        "traversal, module_callers/module_callees for module-level traversal, "
        "search for symbol lookup, and reload after rebuilding the index."
    ),
)

# ---------------------------------------------------------------------------
# Tool descriptor helpers
# ---------------------------------------------------------------------------
_TOOL_LIST = [
    {
        "name": "stats",
        "description": "Return function/module node + edge counts for the loaded index.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "reload",
        "description": "Re-read the index file from disk (use after running 'pyscope-mcp build').",
        "inputSchema": {"type": "object", "properties": {}},
        # reload reads disk; no side effects outside process state
        "annotations": {"readOnlyHint": False, "idempotentHint": True},
    },
    {
        "name": "callers_of",
        "description": (
            "List functions that (transitively, up to depth) call the given function. "
            "Results are capped at 50 and ranked by (hop_depth ASC, -total_degree DESC, fqn ASC) "
            "so depth-1 callers always appear before depth-2 ones. "
            "When the cap triggers, `truncated` is true and `dropped` is the number of results cut. "
            "`dropped` is always present (0 when cap does not fire) — use it to detect partial views "
            "and narrow your query or escalate accordingly. "
            "Response includes result-scoped staleness: `stale: bool`, `stale_files: list[str]`, "
            "and `stale_action: str` when stale is true. "
            "Response includes `completeness: 'complete' | 'partial'`. "
            "'partial' means at least one result FQN has unresolved static-dispatch calls "
            "(e.g. getattr, duck typing, decorator registries) — verify with grep before "
            "treating the result as exhaustive. "
            "'complete' means no result FQN (or its class siblings) appears in the miss log. "
            "Returns {results: [...], truncated: bool, dropped: int, completeness: str, stale: bool, stale_files: [...]}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "Fully-qualified name, e.g. pkg.mod.func"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 1},
            },
            "required": ["fqn"],
        },
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "callees_of",
        "description": (
            "List functions (transitively, up to depth) called by the given function. "
            "Results are capped at 50 and ranked by (hop_depth ASC, -total_degree DESC, fqn ASC) "
            "so depth-1 callees always appear before depth-2 ones. "
            "When the cap triggers, `truncated` is true and `dropped` is the number of results cut. "
            "`dropped` is always present (0 when cap does not fire) — use it to detect partial views "
            "and narrow your query or escalate accordingly. "
            "Response includes result-scoped staleness: `stale: bool`, `stale_files: list[str]`, "
            "and `stale_action: str` when stale is true. "
            "Response includes `completeness: 'complete' | 'partial'`. "
            "'partial' means at least one result FQN has unresolved static-dispatch calls "
            "(e.g. getattr, duck typing, decorator registries) — verify with grep before "
            "treating the result as exhaustive. "
            "'complete' means no result FQN (or its class siblings) appears in the miss log. "
            "Returns {results: [...], truncated: bool, dropped: int, completeness: str, stale: bool, stale_files: [...]}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "Fully-qualified name, e.g. pkg.mod.func"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 1},
            },
            "required": ["fqn"],
        },
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "module_callers",
        "description": (
            "List modules that import/call into the given module or package prefix. "
            "The `module` argument is treated as a dotted-prefix query: supply a full "
            "FQN (e.g. `pkg.mod`) for an exact match, or a package prefix "
            "(e.g. `pkg.agents`) to aggregate callers across all matching modules. "
            "An empty string matches all modules. "
            "Results are capped at 50, deduplicated, and ranked by "
            "(hop_depth ASC, -total_degree DESC, fqn ASC) so depth-1 module callers "
            "always appear before depth-2 ones. "
            "When the cap triggers, `truncated` is true and `dropped` is the number of results cut. "
            "`dropped` is always present (0 when cap does not fire) — use it to detect partial views "
            "and narrow the prefix or escalate accordingly. "
            "Response includes result-scoped staleness: `stale: bool`, `stale_files: list[str]`, "
            "and `stale_action: str` when stale is true. "
            "Response includes `completeness: 'complete' | 'partial'`. "
            "'partial' means at least one symbol in the result modules has unresolved "
            "static-dispatch calls — verify with grep before treating as exhaustive. "
            "'complete' means no symbol in the result modules appears in the miss log. "
            "Returns {results: [...], truncated: bool, dropped: int, completeness: str, stale: bool, stale_files: [...]}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "Dotted module prefix or exact FQN, e.g. 'pkg.agents' or 'pkg.agents.writer'",
                },
                "depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 1},
            },
            "required": ["module"],
        },
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "module_callees",
        "description": (
            "List modules that the given module or package prefix imports/calls into. "
            "The `module` argument is treated as a dotted-prefix query: supply a full "
            "FQN (e.g. `pkg.mod`) for an exact match, or a package prefix "
            "(e.g. `pkg.agents`) to aggregate callees across all matching modules. "
            "An empty string matches all modules. "
            "Results are capped at 50, deduplicated, and ranked by "
            "(hop_depth ASC, -total_degree DESC, fqn ASC) so depth-1 module callees "
            "always appear before depth-2 ones. "
            "When the cap triggers, `truncated` is true and `dropped` is the number of results cut. "
            "`dropped` is always present (0 when cap does not fire) — use it to detect partial views "
            "and narrow the prefix or escalate accordingly. "
            "Response includes result-scoped staleness: `stale: bool`, `stale_files: list[str]`, "
            "and `stale_action: str` when stale is true. "
            "Response includes `completeness: 'complete' | 'partial'`. "
            "'partial' means at least one symbol in the result modules has unresolved "
            "static-dispatch calls — verify with grep before treating as exhaustive. "
            "'complete' means no symbol in the result modules appears in the miss log. "
            "Returns {results: [...], truncated: bool, dropped: int, completeness: str, stale: bool, stale_files: [...]}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "Dotted module prefix or exact FQN, e.g. 'pkg.agents' or 'pkg.agents.writer'",
                },
                "depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 1},
            },
            "required": ["module"],
        },
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "search",
        "description": (
            "Substring search over known fully-qualified function names. "
            "Results are capped at `limit` (default 50); use a more specific query if `truncated` is true. "
            "Response includes result-scoped staleness: `stale: bool`, `stale_files: list[str]`, "
            "and `stale_action: str` when stale is true. "
            "Returns {results: [...], truncated: bool, total_matched: int, stale: bool, stale_files: [...]}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["query"],
        },
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "file_skeleton",
        "description": (
            "Return all top-level functions, classes, and methods defined in the given file. "
            "Input path must be relative to the repo root (e.g. 'agents/avatar_agent.py'). "
            "Each symbol includes: fqn (fully-qualified name), kind (function|class|method), "
            "signature (first def-line only, no body), and lineno. "
            "Results are sorted by lineno and capped at 50; when the cap triggers, "
            "`truncated` is true and `total` holds the full count. "
            "Response always includes `stale: bool` and `stale_files: list[str]`. "
            "When stale is true, `stale_files` lists the relative paths of changed files "
            "and `stale_action` provides a remediation string. "
            "If the file was added after the last build, returns isError:true alongside stale fields. "
            "For pre-v3 indexes, `index_stale_reason: 'index_format_incompatible'` is set. "
            "Returns {results: [...], truncated: bool, total: int, stale: bool, stale_files: [...], "
            "stale_action?: str} or {isError: true, stale: true, stale_files: [], stale_action: str}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repo root, e.g. 'agents/avatar_agent.py'",
                },
            },
            "required": ["path"],
        },
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "neighborhood",
        "description": (
            "Return a bounded bidirectional subgraph around a symbol — callers and callees "
            "together, ranked by proximity (depth) and degree, truncated to a declared token "
            "budget. "
            "Edges are ranked depth-first with degree tiebreak (deterministic). "
            "The result always fits within token_budget * 4 characters. "
            "When the budget is hit, `truncated` is true, `depth_truncated` indicates the first "
            "level where dropping started, and `depth_full` indicates the deepest level with "
            "complete data. "
            "Default token_budget=1000. "
            "Hub suppression (default on): nodes whose in-degree exceeds the computed threshold "
            "(p99 of in-degree distribution, floor 10) are treated as utility hubs. "
            "Their direct edges to the queried symbol are kept but BFS does not traverse further "
            "through them in the callers direction, preventing lateral expansion through shared "
            "utilities (e.g. call_llm, parse_json). "
            "The queried symbol itself is always exempt. "
            "Response always includes `hub_suppressed: list[str]` (FQNs of suppressed hub nodes; "
            "empty when no suppression occurred) and `hub_threshold: int` (the threshold used). "
            "Pass expand_hubs=true to disable suppression. Pass hub_threshold to override the "
            "per-query threshold (the response hub_threshold field echoes the override). "
            "Response includes result-scoped staleness: `stale: bool`, `stale_files: list[str]`, "
            "and `stale_action: str` when stale is true. "
            "Response includes `completeness: 'complete' | 'partial'`. "
            "'partial' means at least one FQN in the neighborhood has unresolved "
            "static-dispatch calls — verify with grep before treating as exhaustive. "
            "'complete' means no FQN in the neighborhood appears in the miss log. "
            "Returns {symbol, depth_full, depth_truncated?, edges: [[caller, callee], ...], "
            "truncated: bool, token_budget_used: int, completeness: str, "
            "hub_suppressed: list[str], hub_threshold: int, stale: bool, stale_files: [...]}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Fully-qualified name of the target symbol, e.g. pkg.mod.func",
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 2,
                    "description": "Maximum hop depth in each direction (default: 2)",
                },
                "token_budget": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1000,
                    "description": "Maximum tokens to return (4 chars/token; default: 1000)",
                },
                "expand_hubs": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Disable hub suppression and return the full lateral traversal. "
                        "hub_suppressed will be [] and hub_threshold still reports the computed value. "
                        "Default: false (suppression on)."
                    ),
                },
                "hub_threshold": {
                    "type": ["integer", "null"],
                    "default": None,
                    "description": (
                        "Override the in-degree threshold for hub suppression. "
                        "Nodes with in-degree > hub_threshold are treated as hubs. "
                        "The response hub_threshold field echoes the value used. "
                        "Default: null (use index-level p99 threshold)."
                    ),
                },
            },
            "required": ["symbol"],
        },
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
]

_TOOL_NAMES = {t["name"] for t in _TOOL_LIST}


# ---------------------------------------------------------------------------
# Index accessor
# ---------------------------------------------------------------------------
def _get_index() -> CallGraphIndex:
    global _INDEX
    if _INDEX is None:
        if _INDEX_PATH is None:
            raise RuntimeError("server started without an index path")
        _INDEX = CallGraphIndex.load(_INDEX_PATH)
    return _INDEX


def _text(payload: object) -> dict:
    """Wrap a JSON-serialisable payload in MCP tool-result shape."""
    return {"content": [{"type": "text", "text": _json.dumps(payload, indent=2)}], "isError": False}


def _error_result(message: str) -> dict:
    """Tool-level failure (not a JSON-RPC error) — isError=true."""
    return {"content": [{"type": "text", "text": message}], "isError": True}


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------
@_SERVER.method("tools/list")
async def _tools_list(id, params):  # noqa: A002
    # cursor / pagination not supported; never return nextCursor
    return {"tools": _TOOL_LIST}


# ---------------------------------------------------------------------------
# tools/call
# ---------------------------------------------------------------------------
@_SERVER.method("tools/call")
async def _tools_call(id, params):  # noqa: A002
    # Tool-level shape validation: non-dict params surface as isError:true
    # (MCP convention), not -32603 INTERNAL_ERROR. This guard is handler-local
    # — the _rpc dispatcher remains permissive per issue #40.
    if params is not None and not isinstance(params, dict):
        return _error_result("params must be an object")
    p = params or {}
    name = p.get("name")
    arguments = p.get("arguments")
    if arguments is not None and not isinstance(arguments, dict):
        return _error_result("arguments must be an object")
    if not arguments:
        arguments = {}

    if not isinstance(name, str) or name not in _TOOL_NAMES:
        # Unknown tool name → isError:true in result, NOT a JSON-RPC error
        return _error_result(f"unknown tool: {name!r}")

    try:
        return await _dispatch_tool(name, arguments)
    except RpcError:
        raise  # propagate JSON-RPC level errors (e.g. missing index path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tool %r raised: %s", name, exc)
        return _error_result(str(exc))


async def _dispatch_tool(name: str, arguments: dict) -> dict:
    global _INDEX

    if name == "reload":
        if _INDEX_PATH is None:
            raise RuntimeError("server started without an index path")
        _INDEX = CallGraphIndex.load(_INDEX_PATH)
        return _text(_INDEX.stats())

    idx = _get_index()

    if name == "stats":
        return _text(idx.stats())

    if name == "callers_of":
        fqn = arguments.get("fqn")
        if not fqn:
            return _error_result("callers_of requires 'fqn'")
        return _text(idx.callers_of(fqn, int(arguments.get("depth", 1))))

    if name == "callees_of":
        fqn = arguments.get("fqn")
        if not fqn:
            return _error_result("callees_of requires 'fqn'")
        return _text(idx.callees_of(fqn, int(arguments.get("depth", 1))))

    if name == "module_callers":
        module = arguments.get("module")
        if module is None:
            return _error_result("module_callers requires 'module'")
        return _text(idx.module_callers(module, int(arguments.get("depth", 1))))

    if name == "module_callees":
        module = arguments.get("module")
        if module is None:
            return _error_result("module_callees requires 'module'")
        return _text(idx.module_callees(module, int(arguments.get("depth", 1))))

    if name == "search":
        query = arguments.get("query")
        if query is None:
            return _error_result("search requires 'query'")
        return _text(idx.search(query, int(arguments.get("limit", 50))))

    if name == "file_skeleton":
        path = arguments.get("path")
        if not path:
            return _error_result("file_skeleton requires 'path'")
        result = idx.file_skeleton(path)
        if result.get("isError"):
            # Emit the full dict (including stale/stale_files) so callers can
            # branch on machine-readable staleness fields, not just the message string.
            return {"content": [{"type": "text", "text": _json.dumps(result, indent=2)}], "isError": True}
        return _text(result)

    if name == "neighborhood":
        symbol = arguments.get("symbol")
        if not symbol:
            return _error_result("neighborhood requires 'symbol'")
        depth = int(arguments.get("depth", 2))
        token_budget = int(arguments.get("token_budget", 1000))
        expand_hubs = bool(arguments.get("expand_hubs", False))
        hub_threshold_raw = arguments.get("hub_threshold")
        hub_threshold = int(hub_threshold_raw) if hub_threshold_raw is not None else None
        return _text(idx.neighborhood(symbol, depth, token_budget, expand_hubs=expand_hubs, hub_threshold=hub_threshold))

    # Should never reach here — guarded by _TOOL_NAMES check above
    return _error_result(f"unknown tool: {name!r}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_stdio(index_path: Path) -> None:
    """Load the index and start the JSON-RPC stdio loop (blocks until EOF/shutdown)."""
    global _INDEX_PATH, _INDEX
    _INDEX_PATH = Path(index_path)
    _INDEX = CallGraphIndex.load(_INDEX_PATH)
    asyncio.run(_SERVER.run())
