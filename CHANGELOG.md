# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 0.1.0 — 2026-04-28

First public release of `pyscope-mcp`.

### What's included

- **AST-based call-graph analyzer** (`pyscope-mcp build`): walks a Python package, resolves intra-package call edges without importing the target code, and writes a versioned JSON index to disk.
- **Precomputed index format** (v5): stores node/edge data plus file-level SHA digests and skeleton symbols. The `serve` path loads the index once at startup; it never re-runs analysis at query time.
- **MCP stdio server** (`pyscope-mcp serve`): implements JSON-RPC 2.0 over stdio (zero runtime dependencies — no `mcp`, `pydantic`, or `httpx`) with the following tools:
  - `stats` — node and edge counts for the loaded graph
  - `refers_to` — all typed references to a symbol (calls, imports, isinstance checks, annotations)
  - `callees_of` — what a function calls, up to a specified depth
  - `module_callees` — module-level dependency edges
  - `search` — substring search over all fully-qualified names
  - `file_skeleton` — top-level functions, classes, and methods defined in a file
  - `neighborhood` — bounded bidirectional subgraph around a symbol, rank-truncated to a token budget
  - `reload` — re-read the index from disk after running `build`
  - `build` — trigger a rebuild via subprocess and reload the in-process index
- **Opt-in query logger** (`PYSCOPE_MCP_LOG=1`): appends structured JSONL entries per tool call to a rotating log file. Off by default.
- **Per-file error isolation**: a parse failure on one file does not abort the build — the analyzer skips the file, warns, and continues.
