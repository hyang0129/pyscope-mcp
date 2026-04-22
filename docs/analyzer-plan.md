# Analyzer MVP plan

Goal: a working AST-based call-graph generator that satisfies the
`build_raw(root, package) -> dict[str, list[str]]` contract, validated by
running it against `video_agent_long/video_agent_long/` (~207 `.py` files).

MCP server work is **out of scope**. We only need `pyscope-mcp build` to
produce a usable `index.json` for that repo.

## Scope and non-goals

In scope:
- Pure-AST, stdlib-only analyzer (`ast` module).
- Intra-package call resolution: function-to-function, method-to-method,
  and module-to-module edges where statically determinable.
- Per-file error isolation (parse failures skipped with a warning).
- Deterministic output (sorted keys, sorted callee lists).
- End-to-end run against `video_agent_long`, producing a JSON index that
  `CallGraphIndex.from_raw` loads without error.

Explicitly out of scope (accepted limitations, per
[docs/prior-art.md](prior-art.md)):
- Dynamic dispatch through dict/registry lookup.
- `getattr` / `importlib.import_module` with non-literal names.
- Decorator-based registries, LLM tool-use dispatch.
- Full duck-typing resolution across unrelated classes.
- Metaclass / `__init_subclass__` effects, `exec` / `eval`.
- Confidence scores on edges (v1 shape only: `dict[str, list[str]]`).
- Runtime tracing, incremental rebuilds, SQLite storage, SCIP export.
- `file_skeleton`, `call_chain`, `neighborhood`, `impact` tools.
- The MCP server — leave `serve` untouched.

We accept "definite edges, silent false negatives" as the quality bar.

## Target validation repo

- Root: `/workspaces/hub_3/video_agent_long`
- Package: `video_agent_long` (the inner `video_agent_long/video_agent_long/`)
- Scale: ~207 `.py` files under `video_agent_long/`, `agents/`, `tools/`.
- Known-hard patterns present (from a skim):
  - Agent dispatch by string key (registry/tool-use) — will under-resolve.
  - Async / click / typer CLI entry points — usually fine.
  - Pydantic models with validators — usually fine.

Success = build completes end-to-end, index loads, spot-checks below pass.

## Architecture

Two passes, one file at a time. No cross-file symbol resolution pass —
we resolve names per-file against an import table plus the set of
in-package module symbols discovered in pass 1.

### Pass 1: module discovery

Walk the package root, collect every `.py` file, compute its dotted
module name relative to `package`:

- `pkg/__init__.py` → `pkg`
- `pkg/sub/mod.py` → `pkg.sub.mod`
- `pkg/sub/__init__.py` → `pkg.sub`

Parse each file; on `SyntaxError`, log and skip. From successful parses,
collect top-level defs: every `FunctionDef`, `AsyncFunctionDef`,
`ClassDef`, and nested `FunctionDef`s inside classes (one level — methods).
Store a flat set of known FQNs:

```
{"video_agent_long.main.main",
 "video_agent_long.agents.router.Router",
 "video_agent_long.agents.router.Router.dispatch",
 ...}
```

This set is the resolution oracle for pass 2: a name resolves only if it
maps to something in this set. External calls (stdlib, third-party) are
dropped, not emitted as `<builtin>.*` placeholders. Keeps the graph
focused on the target package, matching what downstream queries care about.

### Pass 2: per-file edge extraction

For each parsed file, build:

1. **Import table** — maps local names → FQNs. Handles:
   - `import foo.bar` → `{"foo": "foo", "foo.bar": "foo.bar"}` (attribute
     access `foo.bar.baz` resolves by prefix match).
   - `import foo.bar as fb` → `{"fb": "foo.bar"}`.
   - `from foo.bar import baz` → `{"baz": "foo.bar.baz"}`.
   - `from foo.bar import baz as b` → `{"b": "foo.bar.baz"}`.
   - `from . import x` and `from .sub import x` — resolve relative to the
     current module's package path.
   - Module-level only; imports inside function bodies are captured if
     they appear at the top of the body (best-effort), otherwise missed.

2. **Scope walker** — tracks the enclosing FQN as it descends:
   - Module scope → caller FQN is the module itself (`pkg.mod`).
   - `FunctionDef` / `AsyncFunctionDef` → `pkg.mod.func`.
   - `ClassDef` → pushes class name onto the stack; method defs inside
     produce `pkg.mod.Class.method`.
   - Nested functions → use dotted path including parent function name.
     Good enough; these rarely matter.

3. **Call resolution** — for each `ast.Call` node, resolve `node.func`:
   - `ast.Name("foo")` → look up `foo` in import table, then in
     module-local defs (functions/classes defined in this file).
   - `ast.Attribute(ast.Name("mod"), "func")` → look up `mod` in import
     table; if it maps to a known module FQN, try `<fqn>.func`.
   - `ast.Attribute(ast.Attribute(...), "x")` → flatten the attribute
     chain to a dotted string, try prefix-match against the import table.
   - `self.method(...)` inside a method → resolve to
     `<enclosing_class_fqn>.method` **if** that method is defined on the
     same class (no MRO / base-class lookup in v1).
   - Everything else → drop. Unresolved calls are silent false negatives;
     that's the accepted cost.

Emit an edge only if the resolved callee is in the pass-1 known-FQN set.
This filters out stdlib / third-party noise and avoids fabricated edges.

### Output assembly

```python
raw: dict[str, list[str]] = {}
for caller_fqn, callees in collected_edges.items():
    raw[caller_fqn] = sorted(set(callees))
return dict(sorted(raw.items()))
```

Deterministic: dict insertion order sorted, callee lists deduped and sorted.

### Miss report (sidecar)

The raw dict shape stays `dict[str, list[str]]` — unchanged, per the
contract. But `build_raw` is not the only thing the analyzer writes.
Alongside the index, emit a **miss report** so we can see after the run
what the graph is blind to. This is the "known unknowns" log; it turns
silent false negatives into visible ones.

Written to `.pyscope-mcp/misses.json` next to `index.json`. Shape:

```json
{
  "version": 1,
  "summary": {
    "files_total": 207,
    "files_parsed": 205,
    "files_skipped": 2,
    "calls_total": 4821,
    "calls_resolved_in_package": 1902,
    "calls_resolved_external": 1640,
    "calls_unresolved": 1279,
    "resolution_rate_in_package": 0.54
  },
  "skipped_files": [
    {"path": "pkg/broken.py", "reason": "SyntaxError: ..."}
  ],
  "unresolved_calls": [
    {
      "caller": "video_agent_long.agents.router.Router.dispatch",
      "file": "video_agent_long/agents/router.py",
      "line": 142,
      "pattern": "subscript_call",
      "snippet": "HANDLERS[task_type](payload)"
    },
    {
      "caller": "video_agent_long.tools.loader.load",
      "file": "video_agent_long/tools/loader.py",
      "line": 37,
      "pattern": "getattr_nonliteral",
      "snippet": "getattr(mod, name)()"
    }
  ],
  "pattern_counts": {
    "subscript_call": 48,
    "getattr_nonliteral": 22,
    "importlib_import_module": 9,
    "decorator_registry": 14,
    "self_method_unresolved": 61,
    "attr_chain_unresolved": 310,
    "bare_name_unresolved": 180,
    "call_on_call_result": 95,
    "exec_or_eval": 3,
    "lambda_or_local_def": 27
  }
}
```

Pattern classifier runs inside `_resolve_call` — whenever we **decide not
to emit an edge**, we also record *why*. Categories we tag:

- `subscript_call` — `expr[...]()`, the dict/registry dispatch case.
- `getattr_nonliteral` — `getattr(x, name)(...)` where `name` isn't a
  string literal.
- `importlib_import_module` / `__import__` — dynamic import.
- `decorator_registry` — call site is `registry.register("...")(...)`-shaped
  OR a `@decorator(...)`-decorated def where the decorator comes from a
  module we'd classify as a registry (heuristic: named `registry`,
  `router`, `dispatcher`, or has a `.register` / `.dispatch` attribute
  accessed elsewhere in the file). Best-effort; false-positive tolerant.
- `self_method_unresolved` — `self.foo(...)` where `foo` isn't defined
  on the enclosing class (MRO lookup would be needed).
- `attr_chain_unresolved` — `a.b.c(...)` where the chain doesn't
  prefix-match any known import or in-package FQN.
- `bare_name_unresolved` — `foo(...)` where `foo` isn't in the import
  table or module-local defs (probably a parameter or closure var).
- `call_on_call_result` — `f(...)(...)`; the callee is itself the
  result of a call (factories, decorators-as-callables).
- `exec_or_eval` — literal `exec(...)` / `eval(...)` / `compile(...)`
  seen in the file.
- `lambda_or_local_def` — call to a locally-bound lambda or def that's
  not exposed at module scope; we don't give these FQNs.

Unresolved call entries are capped per-pattern (e.g. keep first 50
exemplars of each) so the file stays readable; `pattern_counts` holds
the full tallies. Sampling is deterministic (first-N by file then line),
not random.

Additional rollups that are cheap to compute and useful to eyeball:

- **`dead_keys`** — in-package FQNs that appear as callers but are
  never called by anyone in the graph. Expected for CLI entry points
  and test helpers; a long list in weird places hints at router/registry
  blindness.
- **`unreferenced_modules`** — modules with zero inbound edges. Same
  caveat; CLI/plugin-loaded modules will legitimately appear here.

Both rollups live under `summary.rollups` in the same `misses.json`.

Why a sidecar and not inside the index: the index shape is the public
contract and downstream code (`CallGraphIndex.from_raw`, existing tests,
MCP tools) already works against it. Bolting a miss dict onto it would
force a schema-version bump for diagnostic data that the query layer
doesn't need. Sidecar keeps the contract stable.

The CLI prints a one-screen summary at the end of `build`:

```
pyscope-mcp build complete
  files:  205/207 parsed  (2 skipped — see misses.json)
  calls:  4821 total → 1902 in-package edges (39%)
          1640 external (dropped), 1279 unresolved (27%)
  top unresolved patterns:
    attr_chain_unresolved   310
    bare_name_unresolved    180
    call_on_call_result      95
    self_method_unresolved   61
    subscript_call           48
  full report: .pyscope-mcp/misses.json
```

This is the signal we use to decide whether the graph is worth trusting
on `video_agent_long` or whether a specific pattern is dominating.

## Implementation layout

All in [src/pyscope_mcp/analyzer.py](../src/pyscope_mcp/analyzer.py).
Target ~300 lines, stdlib only.

```
analyzer.py
├── build_raw(root, package)                  # public entry, returns raw
├── build_with_report(root, package)          # returns (raw, misses)
├── _discover_modules(root, package)          # pass 1 → {fqn: path}
├── _collect_defs(tree, module_fqn)           # pass 1 → set[str]
├── _build_import_table(tree, module_fqn)     # pass 2 helper
├── _resolve_call(node, ctx)                  # returns (fqn|None, miss_reason)
├── _classify_miss(node, ctx)                 # returns pattern tag
├── _EdgeVisitor(ast.NodeVisitor)             # pass 2 walker, feeds MissLog
├── MissLog                                   # accumulates misses + counts
└── _warn(msg)                                # stderr, structured
```

`build_raw` stays on the published contract (returns just the dict).
`build_with_report` is what the CLI calls — it writes both files. Tests
can call either.

No new dependencies. `from __future__ import annotations` at top.

## Testing

Add to existing `tests/` (pytest fixtures use synthetic raw dicts today —
this adds AST-level tests):

1. **Unit tests on tiny synthetic trees** — one file each exercising:
   - `import` / `from ... import` / relative imports.
   - Top-level function calling another top-level function.
   - Method calling a sibling method via `self`.
   - Call through an imported module attribute.
   - Call into the stdlib (must be dropped).
   - Registry-style dynamic dispatch (must produce no edge — regression
     guard, confirms we don't fabricate).
2. **Per-file isolation test** — package with one `SyntaxError` file;
   assert build returns edges from the other files and logs the skip.
3. **Determinism test** — run `build_raw` twice on the same fixture,
   assert identical output.
4. **Smoke test against `video_agent_long`** — gated behind a pytest
   marker (`@pytest.mark.slow` or env var) so CI stays fast. Asserts:
   - Non-zero edge count.
   - Index round-trips through `CallGraphIndex.save` / `load`.
   - A handful of manually-verified edges exist (e.g. `main.main` →
     some known callee we confirm by reading the source first).

## Validation against video_agent_long

Run order:

```
cd /workspaces/hub_3/pyscope_mcp
pip install -e '.[dev]'
pytest

cd /workspaces/hub_3/video_agent_long
PYSCOPE_MCP_PACKAGE=video_agent_long \
PYSCOPE_MCP_ROOT=/workspaces/hub_3/video_agent_long \
pyscope-mcp build
```

Expected artifact: `video_agent_long/.pyscope-mcp/index.json`.

Sanity checks on the index:
- Node count roughly matches def count across the package (within ~10%).
- CLI entry point (`video_agent_long.main.main` or similar) has
  non-empty callees.
- At least one class method → sibling method edge exists.
- No keys contain `<builtin>` or `None` — only in-package FQNs.
- File size is "small" (low MB); if it's huge, we're leaking stdlib.

Known failures we will accept without fixing:
- Agent router dispatch table will show as a dead-end — no edges into
  the registered handlers. Documented limitation.
- Tool-use `tools=[...]` lists won't produce edges to tool bodies.
- Duck-typed `stage.execute(ctx)` calls will often be unresolved.

## Milestones

1. **M1 — skeleton + pass 1.** Module discovery and def collection
   working, with unit tests. No edges yet.
2. **M2 — pass 2 simple cases.** Direct name calls and `from ... import`
   resolution. Unit tests green.
3. **M3 — method self-calls and attribute-chain resolution.** Covers the
   bulk of real edges.
4. **M4 — relative imports, error isolation, determinism.** Hardening.
5. **M5 — miss report + run against video_agent_long.** Wire `MissLog`
   through `_EdgeVisitor`, add `build_with_report` and the CLI summary
   print, write `misses.json`. Inspect output against the real repo,
   fix obvious under-resolutions that are cheap wins, and let the miss
   report tell us which patterns dominate so we know where the graph is
   blind. Commit the sanity-check script under `scripts/`.

Stop at M5. Anything beyond (confidence scores, `file_skeleton`, MCP tool
additions, SCIP export) is a separate piece of work.
