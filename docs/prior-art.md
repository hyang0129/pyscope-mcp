# Prior art: pycg, and what we learned from it

pyscope-mcp started as a thin MCP wrapper around [pycg](https://github.com/vitsalis/PyCG), a research static-analysis call-graph generator from the ICSE 2021 paper *"PyCG: Practical Call Graph Generation in Python."* We dropped pycg after hitting blockers that turn out to be fundamental to it as a dependency, not to static call graphs in general. This document is preserved both as an explanation of the pivot and as a constraint catalogue for whatever we build next.

## Why we dropped pycg

1. **Python 3.11 incompatibility.** pycg 0.0.8 installs a custom `importlib` finder whose `invalidate_caches` path recurses through 3.11's lazy imports of `importlib.metadata._adapters` / `_meta` and raises `ImportManagerError: Can't add edge to a non existing node` — on a 4-line toy file. Last release was 2022; no 3.11 fix in sight.
2. **Broken PyPI wheel on case-sensitive filesystems.** The 0.0.8 wheel ships a `PyCG/` directory but its own source does `from pycg import ...`. Works on macOS/Windows, fails on Linux. Reinstalling from git works around it but can't be expressed in a pyproject dependency cleanly.
3. **Hard-aborts on real-world repos.** Before we even hit the 3.11 issue, pycg was aborting on subsets of video_agent_long with `ImportManagerError`. No `--continue-on-error`, no per-file isolation — one bad file kills the whole run.
4. **Unmaintained.** Last commit, last release, and last issue response are all in 2022. Not a foundation to build on.

## What the replacement analyzer must do differently

Direct consequences of the failures above — these are hard requirements, not nice-to-haves:

- **Run on Python 3.11+ natively.** No monkey-patching the import system. An AST-based approach (`ast` module, or `libcst` / `parso`) avoids the whole import-hook mess.
- **Per-file isolation.** A parse failure on one file must not abort the run. Emit a warning for that file, skip it, keep going. The index is allowed to be partial.
- **No runtime import of target code.** pycg's fatal mistake was actually importing target modules to resolve names. Analysis should be purely textual/AST-based.
- **Deterministic output.** Same repo in, same JSON out. Makes the index diffable across commits.

## Patterns any static analyzer will miss

The rest of this doc is a mental model agents should internalize. These limitations are inherent to statically analyzing a dynamically-typed language — we cannot fix them by switching backends, only by being honest about what the graph means.

### Mental model

Treat the graph as:

> **"Definite edges, with silent false negatives."**

- If the graph says A calls B, it almost certainly does.
- If the graph shows no edge between A and B, that's **weak** evidence they're unrelated — the analyzer may simply not have been able to resolve the target.

Use it to answer "who definitely depends on X" — not "is X dead code."

### Patterns no static analyzer can resolve

#### 1. Dynamic dispatch through a table

```python
HANDLERS = {"tts": run_tts, "llm": run_llm}
HANDLERS[task_type](payload)
```

Static analysis sees a dict lookup and a call on the result. It does not know which entry was retrieved, so no edge is drawn to `run_tts` or `run_llm` from this call site.

#### 2. Strings-as-code

```python
mod = importlib.import_module(cfg["module"])
cls = getattr(mod, name)
cls()
```

The target is a string only known at runtime. `importlib.import_module`, `__import__`, and `getattr(obj, name)` with a non-literal name are all opaque.

#### 3. Decorator-based registries

```python
@registry.register("persona_a")
def run_a(ctx): ...

registry.dispatch("persona_a", ctx)
```

The registration stores `run_a` in a structure the analyzer does not track symbolically, and the dispatch call retrieves it by string key. Very common in agent, plugin, and router frameworks.

#### 4. LLM tool-use

```python
tools = [search_web, render_frame, write_file]
client.messages.create(tools=tools, ...)
```

The analyzer sees the list being built, but the decision of which tool to invoke happens inside the model response. There is no static edge from the calling code to the tool implementations.

#### 5. Duck-typed polymorphism

```python
def run_stage(stage):
    stage.execute(ctx)
```

If five classes implement `execute` and `stage` could be any of them, the analyzer may resolve to one, to none, or to a spurious target depending on how type information flows through the specific call site. Do not rely on the resolved target.

#### 6. Conditional and lazy imports

```python
def encode(frame, use_gpu):
    if use_gpu:
        from .gpu import encode_impl
    else:
        from .cpu import encode_impl
    return encode_impl(frame)
```

Module-level conditional imports are usually captured. Imports inside functions are often missed, especially when gated on runtime values.

#### 7. Metaclasses and `__init_subclass__`

Class hierarchies built by metaclasses, or side effects triggered in `__init_subclass__`, are opaque. Any call graph that depends on subclass registration will be incomplete.

#### 8. `exec`, `eval`, templated code

```python
exec(compile(source, "<str>", "exec"), namespace)
```

Zero visibility. Treat any module that runs generated code as a black box in the graph.

#### 9. C extensions and stdlib internals

Functions implemented in C (numpy operations, most of `builtins`, parts of `stdlib`) have no Python source to analyze. Calls into them terminate the trace. You may see edges like `<builtin>.dict.get` as placeholders.

#### 10. Complex decorator stacks

Decorators that return wrappers built from `functools.wraps` usually resolve correctly. Decorators that replace the function with an instance of a callable class, or that compose multiple layers dynamically, can cause misattribution.

## Failure modes in the output

1. **Missing edges.** Most common. `callees_of(foo)` returns a short, plausible-looking list that happens to be incomplete. There is no warning.
2. **Under-resolved names.** Targets appear as unqualified names or `<builtin>.*` placeholders when resolution failed. Treat these as "call site exists, target unknown."
3. **Skipped files.** The replacement analyzer is required to skip on parse error rather than abort. Expect a small number of files to be absent from the index in any real repo.

## Guidance for agentic use

- **Good questions for pyscope-mcp:**
  - "Who calls this helper I'm about to change?"
  - "What does this function reach, to scope a refactor?"
  - "Which modules import this module?"

- **Bad questions for pyscope-mcp:**
  - "Is this function dead code?" (a string-keyed registry may call it)
  - "Is this code path unreachable?" (dynamic dispatch may hit it)
  - "List every place that could invoke this LLM tool." (only the model knows)

- **When the graph looks too sparse:** the code is probably heavy on registries, tool-use, or metaclass magic. Fall back to grep for the function name — a textual search will find string-keyed call sites that static analysis missed.

- **When in doubt:** combine pyscope-mcp results with a grep pass before acting on the assumption that something is unused.

## Alternatives considered

**Scalpel** (Penn State research framework) is pitched as a more type-aware successor to pycg and emits a call graph in a comparable shape. Its type inference may resolve some of the duck-typed polymorphism and conditional-import cases above. It cannot address the fundamentally dynamic patterns — no static tool can. Unexplored whether it shares pycg's import-hook fragility; would need to be validated before adopting.

**Jedi / pyright / pylsp**: language servers, not call graph generators. Could be scripted into one, but the interface is built around cursor positions, not whole-repo analysis.

**Our own AST walker**: the most likely path. `ast.parse` per file, walk imports to build a module-scope symbol table, walk call expressions to emit edges. Gives up on patterns 1–10 above (as would any static tool) but gives us per-file error isolation, Python 3.11+ compatibility, and zero external dependencies. The MVP is maybe 200 lines.

## What other code-graph MCPs do

Survey of shipping MCP servers (and closely-related tools like aider's repo-map) that expose codebase structure to agents. The goal is to steal the conventions that keep recurring and skip the ones that don't survive real use.

### Servers surveyed

| Project | Indexer | Graph shape | Notable tools |
|---|---|---|---|
| **oraios/serena** | LSP (40+ langs) or JetBrains | symbols, refs, type hierarchy | `find_symbol`, `symbol_overview`, `find_referencing_symbols`, `find_declaration`, `find_implementations`, `replace_symbol_body`, `insert_after_symbol` |
| **harshkedia177/axon** | tree-sitter, 12-phase pipeline | File / Func / Class / Method + CALLS / CONTAINS / IMPORTS / EXTENDS / USES_TYPE, **confidence 0.0–1.0 on every edge** | `axon_query`, `axon_context`, `axon_impact` (direct/indirect/transitive), `axon_dead_code` |
| **rustic-ai/codeprism** | "universal AST" | graph-first | `repository_stats`, `trace_path`, `analyze_dependencies`, `find_unused_code` |
| **MikeRecognex/mcp-codebase-index** | Python `ast` (+regex for other langs — brittle) | symbols, imports, calls | `find_symbol`, `get_dependencies`, `get_dependents`, `get_call_chain` (BFS shortest path), `get_change_impact`, `reindex` |
| **cavenine/ctxpp** | tree-sitter + SQLite FTS5 + embeddings | symbols + calls + imports | `ctxpp_file_skeleton`, `ctxpp_feature_traverse` (BFS, depth 3 default), `ctxpp_blast_radius` |
| **Lekssays/codebadger** | Joern CPG (Dockerized) | full Code Property Graph | raw CPGQL, `get_call_graph`, taint / CFG / slicing. Hard 500 MB repo cap, 10-min gen timeout. |
| **MapleRook/rook** | ingests **SCIP** | SCIP nodes in Postgres + Redis | GraphQL + MCP wrapper |
| **drxddy/polymcp** | **SCIP** | — | MCP wrapper over scip indexes |
| **bnomei/frigg** | tree-sitter + **SCIP** + reranker | hybrid | — |
| **aider repo-map** (not MCP; widely copied) | tree-sitter + `.scm` tag queries | file graph, edges = symbol deps | NetworkX PageRank personalized on chat files + mentioned symbols, binary-searched to a token budget |

### Conventions that keep recurring

**Symbol naming.** Three camps: dotted FQN (what we already use — matches LSP), `{kind}:{relpath}:{name}` (axon — cheap and stable but collides on overloads), or **SCIP monikers** (`<scheme> <package> (<descriptor>)+` with suffixes `#` type, `.` term, `()` method, `/` namespace, plus `local <id>` for locals). SCIP is the only format with a real spec and cross-tool tooling; LSIF is being deprecated in Sourcegraph's direction. We keep dotted FQN as primary. SCIP export is a later option if we want Sourcegraph interop.

**Index storage.** SQLite with FTS5 for name search dominates for anything non-trivial. Postgres + Redis at scale (rook). Pickle is common and always a mistake. In-memory graphs are fine for small repos. We're on JSON because indexes are small and the file is human-inspectable; if that stops scaling, move to SQLite before anything else.

**Query primitives every server re-invents.** `find_symbol`, `file_skeleton` (symbols + signatures, no bodies — reportedly the single highest-leverage tool for agent context), `callers_of`, `callees_of`, `refs_of`, `call_chain` / `shortest_path` (always BFS), `neighborhood(symbol, depth, token_budget)`, `impact(symbol)` split into direct / indirect / transitive. Our current tool list (`stats`, `reload`, `callers_of`, `callees_of`, `module_callers`, `module_callees`, `search`) covers the core but is missing `file_skeleton`, `call_chain`, and `neighborhood` — worth adding once the analyzer lands.

**Confidence on edges (axon).** Keep unresolved / dynamically-dispatched call sites in the graph, but tag them with a confidence score. This is the cleanest way we've seen to handle the registry / duck-typing / decorator cases from earlier in this doc without either hiding them or pretending they resolve. Strongly consider adopting: raw dict becomes `{caller: [(callee, confidence), ...]}` or a parallel `low_confidence` dict. Bumps the schema version.

**Rank-then-truncate output (aider).** Large neighborhoods are useless to an agent if dumped raw. Aider's pattern: score every candidate by PageRank personalized on the user's focus (open files, mentioned symbols), then binary-search a token budget by dropping low-rank entries. None of the surveyed MCPs fully replicate this; the ones that don't truncate at all are the ones agents complain about. Worth stealing when we add `neighborhood`.

**Incremental updates.** Two workable patterns. Axon splits file-local phases (parse / imports / calls — run on file change) from global phases (communities, dead-code — batched every 30 s). mcp-codebase-index does a git-diff check before queries and rebuilds if ≤20 files changed. Serena offloads invalidation to the LSP. We currently require explicit `build` + `reload`; that's fine for now, and simpler than any of the above. Revisit if rebuild times get painful.

**Dynamic-Python handling.** No surveyed server handles it well. Best known workaround is augmenting static analysis with an **optional runtime trace hook** (`sys.setprofile` or `coverage.py`) and merging those edges in at `confidence=1.0`. Out of scope for v1 but worth knowing.

**What to skip.** Raw-query endpoints (CPGQL-style — agents write bad queries). Regex "parsers" for languages the server doesn't properly support (mcp-codebase-index does this for non-Python; brittle). Pickle caches. LSIF (use SCIP if anything). Filesystem watchers inside the MCP process — the build/reload split is cleaner.

### SCIP and MCP

Yes, people layer MCPs on SCIP: **MapleRook/rook**, **drxddy/polymcp**, **bnomei/frigg**. `scip-python` is a pyright fork that emits `index.scip` and supports `--project-namespace` for cross-repo nav. If we want interop with Sourcegraph tooling or to let other MCPs consume our index, SCIP export is the target. Not a v1 concern.

### Takeaways for pyscope-mcp

Concrete things to carry into the replacement analyzer and the surrounding server:

1. **Dotted FQN stays the primary ID.** Matches LSP, matches what `CallGraphIndex` already expects.
2. **Per-file error isolation in the analyzer.** Already a hard requirement from the pycg post-mortem; reinforced by every survivor in this survey.
3. **Surface unresolved edges with a confidence score.** Don't silently drop registry / dynamic-dispatch call sites. Schema change; bump the version when adding.
4. **Add `file_skeleton`, `call_chain`, `neighborhood` tools.** These keep recurring and our current surface is missing them.
5. **Rank-and-truncate large outputs** to a token budget when we add neighborhood queries.
6. **Defer:** SQLite storage, SCIP export, runtime tracing, incremental rebuild. All optional; none block v1.

### Source URLs

- github.com/oraios/serena
- github.com/harshkedia177/axon
- github.com/rustic-ai/codeprism
- github.com/MikeRecognex/mcp-codebase-index
- github.com/cavenine/ctxpp
- github.com/Lekssays/codebadger
- github.com/MapleRook/rook
- github.com/drxddy/polymcp
- github.com/bnomei/frigg
- github.com/sourcegraph/scip-python
- github.com/sourcegraph/scip/blob/main/docs/scip.md
- github.com/Aider-AI/aider/blob/main/aider/repomap.py
