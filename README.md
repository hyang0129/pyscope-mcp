# pyscope-mcp

A deployable MCP server that exposes Python function- and module-level call graphs over any Python repo, for use by agentic coding clients (Claude Code, etc.).

## What it gives an agent

Instead of grepping blindly:

- `callers_of(fqn, depth)` — who calls this function, transitively
- `callees_of(fqn, depth)` — what does this function reach
- `module_callers(module)` / `module_callees(module)` — module-level dependency edges
- `neighborhood(symbol, depth, token_budget)` — bounded bidirectional subgraph around a symbol, rank-truncated to fit a token budget
- `file_skeleton(path)` — all top-level functions, classes, and methods defined in a file
- `search(query)` — substring search over all FQNs
- `stats()` — sanity-check the loaded index
- `reload()` — re-read the index from disk after running `pyscope-mcp build`
- `build()` — trigger a rebuild via subprocess and reload the in-process index in one call

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

## Setup for an agent in your repo (Claude Code)

If you are an agent setting up `pyscope-mcp` in a new repo, follow these steps:

1. **Install the package**

   ```bash
   pip install pyscope-mcp
   ```

2. **Build the call-graph index**

   ```bash
   pyscope-mcp build --root /abs/path/to/repo --package your_package_name
   ```

   This writes `.pyscope-mcp/index.json` inside the repo. Run this step first — `serve` will not build the index automatically.

3. **Register the MCP server in `.claude/settings.json`**

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

   If the index lives at a non-default path, add `"PYSCOPE_MCP_INDEX": "/abs/path/to/index.json"` to the `env` block.

4. **Verify the server loaded**

   After restarting Claude Code (or reloading the MCP server), ask Claude:

   > Call the `stats` tool on the pyscope-mcp server.

   You should see node and edge counts in the response. If the output is non-empty the index loaded correctly and the MCP tools are ready to use.

## Query logging

Every `tools/call` dispatch appends a structured JSONL entry to a local rotating log file so you can measure tool-use patterns, truncation rates, hub-suppression hits, and latency without parsing claude's session transcripts.

The logger is **on by default**. Set `PYSCOPE_MCP_LOG=0` to disable. Per-call overhead is ~20 µs (well under 0.1% of typical tool latency); the rotating file is capped at ~70 MB on disk (10 MB × 5 backups + current). On activation the server emits a one-time WARNING to stderr announcing the active log path so the behavior is never silent.

### Log location

Default: `.pyscope-mcp/query.jsonl` next to the index file (already `.gitignore`d).  
Override: `PYSCOPE_MCP_LOG_PATH=/abs/path/to/query.jsonl`.

### Enable / disable

```bash
# Enabled by default — no env var needed
pyscope-mcp serve ...

# Disable
PYSCOPE_MCP_LOG=0 pyscope-mcp serve ...
```

### Log entry schema (v1)

```json
{
  "v": 1,
  "ts": "2026-04-25T21:00:00.000+00:00",
  "server_id": "550e8400-e29b-41d4-a716-446655440000",
  "rpc_id": 3,
  "tool": "neighborhood",
  "args": {"symbol": "pkg.mod.Foo.run", "depth": 2, "token_budget": 500},
  "duration_ms": 12,
  "is_error": false,
  "truncated": true,
  "result_count": null,
  "edge_count": 7,
  "hub_suppressed_count": 2,
  "depth_full": 1,
  "token_budget_used": 487,
  "index_version": 5,
  "index_git_sha": "a1b2c3d4...",
  "index_content_hash": "abcdef12..."
}
```

`server_id` partitions entries by session (the MCP server is spawned per-session as a stdio subprocess by Claude Code). To join log entries with claude's own session transcript, match by `(ts, tool, args)` or `rpc_id`.

Rotation: files rotate at 10 MB; up to 5 historical files are kept (`query.jsonl.1` … `query.jsonl.5`), giving a ~50 MB ceiling.

### Index schema v5 note

The index format was bumped from v4 to v5 to add `git_sha` and `content_hash` header fields (used by the logger to tie each log entry to a specific graph version). Existing v4 index files are **not migrated** — re-run `pyscope-mcp build` to generate a v5 index.
