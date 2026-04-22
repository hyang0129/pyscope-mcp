# pycg-mcp

A deployable MCP server that exposes [pycg](https://github.com/vitsalis/PyCG) function- and module-level call graphs over any Python repo, for use by agentic coding clients (Claude Code, etc.).

The graph is **precomputed and saved** — `pycg-mcp build` runs pycg once and writes a JSON index; `pycg-mcp serve` loads that index and answers MCP queries without re-running pycg. This keeps the server cold-start cheap and decouples analysis from serving.

## What it gives an agent

Instead of grepping blindly:

- `callers_of(fqn, depth)` — who calls this function, transitively
- `callees_of(fqn, depth)` — what does this function reach
- `module_callers(module)` / `module_callees(module)` — module-level dependency edges
- `search(query)` — substring search over all FQNs pycg discovered
- `stats()` — sanity-check the loaded index
- `reload()` — re-read the index after a `pycg-mcp build`

## Install

```bash
pip install pycg-mcp
```

## Build the index

From the repo you want to analyze:

```bash
pycg-mcp build --root /path/to/repo --package my_pkg
```

Writes to `.pycg-mcp/index.json` inside the repo by default. Override with `--output` or `$PYCG_MCP_INDEX`.

Re-run after significant code changes. Check the index file into source control or leave it gitignored — your call.

## Serve

```bash
pycg-mcp serve --root /path/to/repo
```

Errors out if no index exists. Env-var equivalents: `PYCG_MCP_ROOT`, `PYCG_MCP_INDEX`.

## Register with Claude Code

```json
{
  "mcpServers": {
    "pycg": {
      "command": "pycg-mcp",
      "args": ["serve"],
      "env": {
        "PYCG_MCP_ROOT": "/abs/path/to/repo"
      }
    }
  }
}
```

## Status

Alpha. Works on any Python repo that pycg can analyze. pycg is a static analyzer, so dynamic dispatch and heavy metaprogramming will degrade recall — see [docs/pycg-limitations.md](docs/pycg-limitations.md) for the full list of patterns that defeat it and how to interpret the output.
