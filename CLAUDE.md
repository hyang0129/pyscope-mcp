# pyscope-mcp — Claude instructions

## What this repo is

A standalone pip-installable MCP server. Exposes function- and module-level call graphs of a target Python repo as MCP tools so agentic coding clients can query structure instead of grepping.

Package name on PyPI: `pyscope-mcp`. Import path: `pyscope_mcp`. Console script: `pyscope-mcp`.

## Current status

Scaffold only. The query layer, index format, CLI, MCP server, and tests are all in place. The analyzer — the thing that turns source code into the raw `{caller_fqn: [callee_fqn, ...]}` dict — is **not implemented**. `pyscope_mcp/analyzer.py::build_raw` raises `NotImplementedError`.

We started on pycg but dropped it: see [docs/prior-art.md](docs/prior-art.md) for the detailed reasons (Python 3.11 incompatibility, broken PyPI wheel, hard aborts on real repos) and for notes on the constraints the replacement must satisfy.

## The one contract the analyzer must satisfy

```python
def build_raw(root: Path, package: str) -> dict[str, list[str]]: ...
```

- Keys: fully-qualified caller names (e.g. `my_pkg.module.function` or `my_pkg.module.Class.method`).
- Values: lists of fully-qualified callees in the same form.
- Must run on Python 3.11+ without monkey-patching the import system.
- Must fail **per-file** rather than aborting the whole analysis when it can't parse something.

Everything downstream (`CallGraphIndex.from_raw`, save/load, the MCP tools) already works against this shape — it's tested in `tests/test_graph.py` with a synthetic raw dict.

## Core convention: precomputed-and-saved index

Analysis and serving are split:

- `pyscope-mcp build` runs the analyzer and writes a JSON index to disk.
- `pyscope-mcp serve` loads that index and answers MCP queries.

If the index file does not exist, `serve` exits with an error — it does not lazily build. This keeps MCP startup fast, makes the analysis step reproducible, and means the server process has no analyzer dependency at runtime.

Default index path: `.pyscope-mcp/index.json` inside the target repo. Override with `--output` / `--index` flags or `$PYSCOPE_MCP_INDEX`.

When adding a new MCP tool that needs graph data, assume the index is already loaded. Do not introduce code paths that silently rebuild.

## Layout

- `src/pyscope_mcp/graph.py` — `CallGraphIndex`: wraps raw dict in NetworkX digraphs, implements queries, save/load.
- `src/pyscope_mcp/analyzer.py` — **stub**. Implement `build_raw` here.
- `src/pyscope_mcp/cli.py` — argparse entry point with `build` and `serve` subcommands.
- `src/pyscope_mcp/server.py` — MCP stdio server. Tools: `stats`, `reload`, `callers_of`, `callees_of`, `module_callers`, `module_callees`, `search`.
- `tests/` — pytest; fixtures are synthetic raw dicts, not real repos.
- `docs/prior-art.md` — what we learned from pycg and what constrains the replacement.

## Conventions

- Python 3.10+ at minimum. The analyzer must run on 3.11+ cleanly.
- Use `from __future__ import annotations`.
- Dependencies stay minimal: `mcp`, `networkx`, `pydantic`. Do not add heavy deps without a reason — especially do not pull in unmaintained analysis libraries.
- Index file format is versioned (`{"version": 1, "root": ..., "raw": {...}}`). Bump the version if the schema changes and handle old versions in `CallGraphIndex.load`.
- `reload` is the only way the running server picks up a new index — no filesystem watching.

## Environment variables

- `PYSCOPE_MCP_ROOT` — repo to analyze / serve from (default: cwd).
- `PYSCOPE_MCP_PACKAGE` — root package name passed to the analyzer (default: root dir name).
- `PYSCOPE_MCP_INDEX` — index file path (default: `.pyscope-mcp/index.json` relative to root).

## Testing

```bash
pip install -e '.[dev]'
pytest
```

## Releasing

Bump `version` in `pyproject.toml` and `src/pyscope_mcp/__init__.py`, tag, `python -m build`, `twine upload`.
