# pycg-mcp — Claude instructions

## What this repo is

A standalone pip-installable MCP server. Exposes pycg-derived function and module call graphs as MCP tools so agentic coding clients can query structure instead of grepping.

Package name on PyPI: `pycg-mcp`. Import path: `pycg_mcp`. Console script: `pycg-mcp`.

## Core convention: precomputed-and-saved index

**The server never runs pycg itself.** Analysis and serving are split:

- `pycg-mcp build` runs pycg and writes a JSON index to disk.
- `pycg-mcp serve` loads that index and answers MCP queries.

If the index file does not exist, `serve` exits with an error — it does not lazily build. This keeps MCP startup fast, makes the analysis step reproducible, and means the server process has no `python -m pycg` subprocess dependency at runtime.

Default index path: `.pycg-mcp/index.json` inside the target repo. Override with `--output` / `--index` flags or `$PYCG_MCP_INDEX`.

When you add a new MCP tool that depends on graph data, assume the index is already loaded. Do not introduce code paths that silently rebuild.

## Layout

- `src/pycg_mcp/graph.py` — `CallGraphIndex`: builds from pycg, serializes to JSON (`save`/`load`), wraps NetworkX digraphs for queries. The raw pycg dict is the source of truth; graphs are derived on load.
- `src/pycg_mcp/cli.py` — argparse entry point with `build` and `serve` subcommands.
- `src/pycg_mcp/server.py` — MCP stdio server. `run_stdio(index_path)` loads the index and runs the loop. Tools: `stats`, `reload`, `callers_of`, `callees_of`, `module_callers`, `module_callees`, `search`.
- `tests/` — pytest; use small fixture repos, not the real world.

## Conventions

- Python 3.10+. Use `from __future__ import annotations`.
- Dependencies stay minimal: `mcp`, `pycg`, `networkx`, `pydantic`. Do not add heavy deps without a reason.
- pycg is invoked as a subprocess (`python -m pycg`) during `build` — do not import its internals; the CLI contract is more stable.
- Index file format is versioned (`{"version": 1, "root": ..., "raw": {...}}`). Bump the version if the schema changes and handle old versions in `CallGraphIndex.load`.
- `reload` is the only way the running server picks up a new index — no filesystem watching.

## Environment variables

- `PYCG_MCP_ROOT` — repo to analyze / serve from (default: cwd).
- `PYCG_MCP_PACKAGE` — package name passed to pycg during `build` (default: root dir name).
- `PYCG_MCP_INDEX` — index file path (default: `.pycg-mcp/index.json` relative to root).

## Testing

```bash
pip install -e '.[dev]'
pytest
```

## Releasing

Bump `version` in `pyproject.toml` and `src/pycg_mcp/__init__.py`, tag, `python -m build`, `twine upload`.
