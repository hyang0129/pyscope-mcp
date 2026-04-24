# pyscope-mcp — Claude instructions

## What this repo is

A standalone pip-installable MCP server. Exposes function- and module-level call graphs of a target Python repo as MCP tools so agentic coding clients can query structure instead of grepping.

Package name on PyPI: `pyscope-mcp`. Import path: `pyscope_mcp`. Console script: `pyscope-mcp`.

## Governing document — read before non-trivial changes

[CONSTITUTION.md](CONSTITUTION.md) defines the non-negotiables this project is checked against. Any PR that adds features, changes the graph surface, or touches the serve/build split must keep all four laws and pass the Review Heuristic at the bottom of that file. In one-line summary:

1. Never mislead the consumer — no uncertain edges, no silent truncation.
2. Minimal startup; the serve path stays thin.
3. Git-in-sync graph is the zero-friction default; drift is cheaply detectable.
4. Graph update ≤60 s on standard PRs.

When the conventions below appear to conflict with the constitution, the constitution wins. If that happens, flag it — the convention is out of date.

## Current status

Scaffold only. The query layer, index format, CLI, MCP server, and tests are all in place. The analyzer — the thing that turns source code into the raw `{caller_fqn: [callee_fqn, ...]}` dict — is **not implemented**. `pyscope_mcp/analyzer.py::build_raw` raises `NotImplementedError`.

We started on pycg but dropped it. See [docs/prior-art.md](docs/prior-art.md) for the full pivot rationale, the inherent limits of static analysis on dynamic Python, and a survey of how other code-graph MCPs solve these problems. The conventions below are the distilled takeaways — read prior-art.md for the reasoning.

## The one contract the analyzer must satisfy

```python
def build_raw(root: Path, package: str) -> dict[str, list[str]]: ...
```

- Keys: fully-qualified caller names, e.g. `my_pkg.module.function` or `my_pkg.module.Class.method`.
- Values: lists of fully-qualified callees in the same form.
- **Python 3.11+ native.** No monkey-patching the import system. AST-based, not import-based.
- **Per-file error isolation.** A parse failure on one file must not abort the run — skip, warn, continue.
- **No runtime import of target code.** pycg's fatal mistake; don't repeat it.
- **Deterministic output.** Same repo in, same JSON out.

Everything downstream (`CallGraphIndex.from_raw`, save/load, the MCP tools) already works against this shape — see `tests/test_graph.py`.

### Symbol naming

Fully-qualified dotted paths are the primary ID. Matches LSP and what downstream code expects. Don't invent alternative schemes (`{kind}:{relpath}:{name}` etc.) — they collide on overloads and break interop.

### Unresolved edges (forthcoming)

Dynamic patterns (string-keyed registries, `getattr`, duck typing, decorators returning callable-class instances, LLM tool-use) cannot be resolved statically. The current shape silently drops them. **Preferred direction:** keep these call sites in the graph but tag them with a confidence score (axon pattern) — either `{caller: [(callee, confidence), ...]}` or a parallel `low_confidence` dict. This is a schema change; bump the version when landing.

Until that lands, absence of an edge is **weak** evidence — treat the graph as "definite edges, silent false negatives."

## Core convention: precomputed-and-saved index

Analysis and serving are split:

- `pyscope-mcp build` runs the analyzer and writes a JSON index to disk.
- `pyscope-mcp serve` loads that index and answers MCP queries.

If the index file does not exist, `serve` exits with an error — it does not lazily build. This keeps MCP startup fast, makes the analysis step reproducible, and means the server process has no analyzer dependency at runtime.

Default index path: `.pyscope-mcp/index.json` inside the target repo. Override with `--output` / `--index` flags or `$PYSCOPE_MCP_INDEX`.

When adding a new MCP tool that needs graph data, assume the index is already loaded. Do not introduce code paths that silently rebuild.

## Layout

- `src/pyscope_mcp/graph.py` — `CallGraphIndex`: wraps raw dict in `_DiGraph` (inline plain-dict adjacency list), implements queries, save/load.
- `src/pyscope_mcp/analyzer.py` — **stub**. Implement `build_raw` here.
- `src/pyscope_mcp/cli.py` — argparse entry point with `build` and `serve` subcommands.
- `src/pyscope_mcp/server.py` — MCP stdio server.
- `tests/` — pytest; fixtures are synthetic raw dicts, not real repos.
- `docs/prior-art.md` — pycg post-mortem + survey of other code-graph MCPs.

## MCP tool surface

Currently shipping: `stats`, `reload`, `callers_of`, `callees_of`, `module_callers`, `module_callees`, `search`.

Missing but recurring across every other code-graph MCP — add in this order when the analyzer lands:

1. **`file_skeleton(path)`** — symbols + signatures, no bodies. Reportedly the single highest-leverage tool for agent context.
2. **`call_chain(src, dst)`** — BFS shortest path from caller to callee.
3. **`neighborhood(symbol, depth, token_budget)`** — bounded subgraph around a symbol, **rank-and-truncate** to the token budget (aider repo-map pattern: score candidates and drop low-rank entries until it fits). Do not dump raw subgraphs; they're useless to agents.
4. **`impact(symbol)`** — callers + transitive callers, split direct / indirect / transitive.

Conventions when adding tools:
- No raw-query endpoints (CPGQL-style). Agents write bad queries.
- All list-returning tools need a size cap. Large neighborhoods truncate by rank, not by arbitrary slice.
- Don't add filesystem watching. `reload` is the invalidation contract.

## Conventions

- Python 3.10+ at minimum. The analyzer must run on 3.11+ cleanly.
- Use `from __future__ import annotations`.
- Dependencies stay minimal: **zero runtime deps for the serve path** (issue #40 removed `mcp`, `pydantic` and all their transitive deps: `httpx`, `anyio`, `jsonschema`, `pydantic_settings`). The JSON-RPC 2.0 stdio transport is hand-rolled in `src/pyscope_mcp/_rpc.py`. Do not add heavy deps without a reason — especially do not pull in unmaintained analysis libraries. (No pickle caches, no regex "parsers," no LSIF.) `networkx` was removed in #37 — graph.py uses a minimal inline `_DiGraph` backed by plain dicts.
- Index file format is versioned (`{"version": 1, "root": ..., "raw": {...}}`). Bump the version if the schema changes and handle old versions in `CallGraphIndex.load`.
- `reload` is the only way the running server picks up a new index — no filesystem watching.
- JSON on disk is fine while indexes are small and human-inspectable. If we outgrow it, move to SQLite + FTS5 before anything more exotic. SCIP export is a later option for Sourcegraph interop; not a v1 concern.

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
