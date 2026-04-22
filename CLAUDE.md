# pycg-mcp — Claude instructions

## What this repo is

A standalone pip-installable MCP server. Exposes pycg-derived function and module call graphs as MCP tools so agentic coding clients can query structure instead of grepping.

Package name on PyPI: `pycg-mcp`. Import path: `pycg_mcp`. Console script: `pycg-mcp`.

## Layout

- `src/pycg_mcp/graph.py` — `CallGraphIndex`: shells out to `python -m pycg`, parses the JSON, wraps it in NetworkX DiGraphs for function- and module-level queries.
- `src/pycg_mcp/server.py` — MCP stdio server, registers tools, lazily builds the index on first query.
- `tests/` — pytest; use small fixture repos, not the real world.

## Conventions

- Python 3.10+. Use `from __future__ import annotations`.
- Dependencies stay minimal: `mcp`, `pycg`, `networkx`, `pydantic`. Do not add heavy deps without a reason.
- pycg is invoked as a subprocess (`python -m pycg`) — do not import its internals; the CLI contract is more stable.
- The server is stateless across processes but caches the graph in-process. `reindex` is the only mutation.

## Environment variables

- `PYCG_MCP_ROOT` — repo to analyze (default: cwd).
- `PYCG_MCP_PACKAGE` — package name passed to pycg (default: root dir name).

## Testing

```bash
pip install -e '.[dev]'
pytest
```

## Releasing

Bump `version` in `pyproject.toml` and `src/pycg_mcp/__init__.py`, tag, `python -m build`, `twine upload`.
