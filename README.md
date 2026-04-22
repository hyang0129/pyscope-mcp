# pyscope-mcp

A deployable MCP server that exposes Python function- and module-level call graphs over any Python repo, for use by agentic coding clients (Claude Code, etc.).

**Status: scaffold.** The index format, the MCP server, the CLI, and the query layer are in place and tested. The call-graph **analyzer is not yet implemented** — see [docs/prior-art.md](docs/prior-art.md) for why we dropped pycg and the design notes for the replacement. Running `pyscope-mcp build` currently raises `NotImplementedError`.

## What it will give an agent

Instead of grepping blindly:

- `callers_of(fqn, depth)` — who calls this function, transitively
- `callees_of(fqn, depth)` — what does this function reach
- `module_callers(module)` / `module_callees(module)` — module-level dependency edges
- `search(query)` — substring search over all FQNs
- `stats()` — sanity-check the loaded index
- `reload()` — re-read the index after a `pyscope-mcp build`

The graph is **precomputed and saved** — `pyscope-mcp build` runs the analyzer once and writes a JSON index; `pyscope-mcp serve` loads that index and answers MCP queries without re-running analysis.

## Install

```bash
pip install pyscope-mcp
```

## Build the index

```bash
pyscope-mcp build --root /path/to/repo --package my_pkg
```

Writes to `.pyscope-mcp/index.json` inside the repo by default. Override with `--output` or `$PYSCOPE_MCP_INDEX`.

## Serve

```bash
pyscope-mcp serve --root /path/to/repo
```

Errors out if no index exists. Env-var equivalents: `PYSCOPE_MCP_ROOT`, `PYSCOPE_MCP_INDEX`.

## Register with Claude Code

```json
{
  "mcpServers": {
    "pyscope": {
      "command": "pyscope-mcp",
      "args": ["serve"],
      "env": {
        "PYSCOPE_MCP_ROOT": "/abs/path/to/repo"
      }
    }
  }
}
```
