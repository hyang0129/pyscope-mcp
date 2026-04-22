from __future__ import annotations

import asyncio
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from pycg_mcp.graph import CallGraphIndex

_INDEX: CallGraphIndex | None = None
_ROOT = Path(os.environ.get("PYCG_MCP_ROOT", os.getcwd())).resolve()
_PACKAGE = os.environ.get("PYCG_MCP_PACKAGE")

app: Server = Server("pycg-mcp")


def _get_index() -> CallGraphIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = CallGraphIndex.build(_ROOT, package=_PACKAGE)
    return _INDEX


@app.list_tools()
async def list_tools() -> list[Tool]:
    depth = {"type": "integer", "minimum": 1, "maximum": 10, "default": 1}
    fqn = {"type": "string", "description": "Fully-qualified name, e.g. pkg.mod.func"}
    return [
        Tool(
            name="reindex",
            description="Re-run pycg over the configured repo root and rebuild the graph.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="stats",
            description="Return function/module node + edge counts for the current index.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="callers_of",
            description="List functions that (transitively, up to depth) call the given function.",
            inputSchema={
                "type": "object",
                "properties": {"fqn": fqn, "depth": depth},
                "required": ["fqn"],
            },
        ),
        Tool(
            name="callees_of",
            description="List functions (transitively, up to depth) called by the given function.",
            inputSchema={
                "type": "object",
                "properties": {"fqn": fqn, "depth": depth},
                "required": ["fqn"],
            },
        ),
        Tool(
            name="module_callers",
            description="List modules that import/call into the given module.",
            inputSchema={
                "type": "object",
                "properties": {"module": {"type": "string"}, "depth": depth},
                "required": ["module"],
            },
        ),
        Tool(
            name="module_callees",
            description="List modules that the given module imports/calls into.",
            inputSchema={
                "type": "object",
                "properties": {"module": {"type": "string"}, "depth": depth},
                "required": ["module"],
            },
        ),
        Tool(
            name="search",
            description="Substring search over known fully-qualified function names.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["query"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    global _INDEX
    if name == "reindex":
        _INDEX = CallGraphIndex.build(_ROOT, package=_PACKAGE)
        return _text(_INDEX.stats())

    idx = _get_index()
    if name == "stats":
        return _text(idx.stats())
    if name == "callers_of":
        return _text(idx.callers_of(arguments["fqn"], int(arguments.get("depth", 1))))
    if name == "callees_of":
        return _text(idx.callees_of(arguments["fqn"], int(arguments.get("depth", 1))))
    if name == "module_callers":
        return _text(idx.module_callers(arguments["module"], int(arguments.get("depth", 1))))
    if name == "module_callees":
        return _text(idx.module_callees(arguments["module"], int(arguments.get("depth", 1))))
    if name == "search":
        return _text(idx.search(arguments["query"], int(arguments.get("limit", 50))))
    raise ValueError(f"unknown tool: {name}")


def _text(payload) -> list[TextContent]:
    import json as _json

    return [TextContent(type="text", text=_json.dumps(payload, indent=2))]


async def _run() -> None:
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
