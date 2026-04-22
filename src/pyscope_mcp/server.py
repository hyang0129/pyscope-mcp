from __future__ import annotations

import asyncio
import json as _json
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from pyscope_mcp.graph import CallGraphIndex

_INDEX: CallGraphIndex | None = None
_INDEX_PATH: Path | None = None

app: Server = Server("pyscope-mcp")


def _get_index() -> CallGraphIndex:
    global _INDEX
    if _INDEX is None:
        if _INDEX_PATH is None:
            raise RuntimeError("server started without an index path")
        _INDEX = CallGraphIndex.load(_INDEX_PATH)
    return _INDEX


@app.list_tools()
async def list_tools() -> list[Tool]:
    depth = {"type": "integer", "minimum": 1, "maximum": 10, "default": 1}
    fqn = {"type": "string", "description": "Fully-qualified name, e.g. pkg.mod.func"}
    return [
        Tool(
            name="stats",
            description="Return function/module node + edge counts for the loaded index.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="reload",
            description="Re-read the index file from disk (use after running 'pyscope-mcp build').",
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
    if name == "reload":
        if _INDEX_PATH is None:
            raise RuntimeError("server started without an index path")
        _INDEX = CallGraphIndex.load(_INDEX_PATH)
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
    return [TextContent(type="text", text=_json.dumps(payload, indent=2))]


async def _run() -> None:
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


def run_stdio(index_path: Path) -> None:
    global _INDEX_PATH, _INDEX
    _INDEX_PATH = Path(index_path)
    _INDEX = CallGraphIndex.load(_INDEX_PATH)
    asyncio.run(_run())
