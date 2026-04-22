# pycg-mcp

A deployable MCP server that exposes [pycg](https://github.com/vitsalis/PyCG) function- and module-level call graphs over any Python repo, for use by agentic coding clients (Claude Code, etc.).

## What it gives an agent

Instead of grepping blindly, the agent can ask:

- `callers_of(fqn, depth)` — who calls this function, transitively
- `callees_of(fqn, depth)` — what does this function reach
- `module_callers(module)` / `module_callees(module)` — module-level dependency edges
- `search(query)` — substring search over all FQNs pycg discovered
- `stats()` — sanity-check the index
- `reindex()` — rebuild after large edits

The graph is built once per server process from `pycg`'s JSON output, then queried via NetworkX.

## Install

```bash
pip install pycg-mcp
```

## Run

```bash
PYCG_MCP_ROOT=/path/to/target/repo \
PYCG_MCP_PACKAGE=my_pkg \
pycg-mcp
```

`PYCG_MCP_PACKAGE` is optional; it defaults to the root directory name.

## Register with Claude Code

```json
{
  "mcpServers": {
    "pycg": {
      "command": "pycg-mcp",
      "env": {
        "PYCG_MCP_ROOT": "/abs/path/to/repo",
        "PYCG_MCP_PACKAGE": "my_pkg"
      }
    }
  }
}
```

## Status

Alpha. Works on any Python repo that pycg can analyze. pycg is a static analyzer, so dynamic dispatch and heavy metaprogramming will degrade recall.
