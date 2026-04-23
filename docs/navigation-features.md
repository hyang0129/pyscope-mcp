# Graph navigation features: what to build, what to skip

Companion to [prior-art.md](prior-art.md) and [constitution.md](constitution.md).
Prior-art surveyed what other code-graph MCPs ship; the constitution set hard
constraints on what we're willing to ship. This document is the intersection:
given those constraints, which navigation features actually pay for themselves,
and in what order.

## Framing

The raw graph (`{caller_fqn: [callee_fqn, ...]}`) is not the product. It's the
substrate. The product is a small set of tool calls an agent can reach for
instead of grepping blindly, each of which answers a recurring real question
in one round-trip.

An agent's time budget on any given query is roughly *"one or two tool calls
before I give up and grep."* So each tool has to either fully answer a common
question, or narrow the search enough that the follow-up grep is targeted.
Tools that dump raw subgraphs fail this test — the agent then has to do work
the tool should have done.

## What survives the survey

From [prior-art.md §What other code-graph MCPs do](prior-art.md#what-other-code-graph-mcps-do),
the features that show up in *most* shipping servers — and that don't conflict
with the constitution — are:

| Feature                           | Shows up in                         | Why it keeps recurring                                  |
| --------------------------------- | ----------------------------------- | ------------------------------------------------------- |
| `find_symbol` / name search       | all of them                         | entry point for every other question                    |
| `file_skeleton`                   | serena, ctxpp, codeprism            | highest-leverage context primitive per reports          |
| `callers_of` / `callees_of`       | all of them                         | the "who depends on X" core question                    |
| `call_chain` (BFS shortest path)  | mcp-codebase-index, codeprism       | scopes a refactor; one hop is rarely enough             |
| `neighborhood(symbol, depth, $)`  | ctxpp, codeprism                    | "give me everything near X, bounded"                    |
| `impact(symbol)` split D/I/T      | axon, mcp-codebase-index            | change-review workflow, different from callers_of       |
| Rank-and-truncate output          | aider (copied by few; regretted)    | raw subgraphs above ~50 nodes are agent poison          |

Features that show up but we **reject** on constitutional grounds:

| Feature                      | Why others ship it             | Why we don't                                                |
| ---------------------------- | ------------------------------ | ----------------------------------------------------------- |
| Confidence-scored edges      | axon (cleanest dynamic-Python treatment seen) | Law 1. Agents round 0.7 → 1.0; the UX harm exceeds the recall gain. |
| Raw query language           | codebadger (CPGQL)             | Agents write bad queries; blast radius unbounded.           |
| `find_unused_code` / dead code | codeprism, axon              | Law 1 again. Static dead-code on dynamic Python is a lie.   |
| Filesystem watchers          | some LSP-backed servers        | Law 2. `reload` is the invalidation contract.               |
| Lazy/on-query rebuild        | mcp-codebase-index (git-diff) | Law 2. Startup determinism beats freshness convenience.     |
| Embeddings / semantic search | ctxpp, frigg                   | Different problem (NL → symbol). Orthogonal to call graph; out of scope. |

## The tool surface we should ship

Ordered by return-on-effort given everything already built.

### Already shipped

`stats`, `reload`, `callers_of`, `callees_of`, `module_callers`, `module_callees`,
`search`. These cover the spine. Nothing to redesign.

### 1. `file_skeleton(path) → signatures`

Return every top-level symbol in the file with its signature and line range,
no bodies. Docstring first line optional.

**Why first:** every surveyed server that added this reported it as the single
biggest context-window win. Agents that open a file to "see what's in it"
spend thousands of tokens on bodies they don't need. A skeleton answers "what
lives here" in tens of tokens.

**Constitutional fit:** pure structural query, no edges involved → law 1 trivially
satisfied. Data is already in the AST pass the analyzer does; just emit it.
No impact on startup (law 2) or index format beyond adding a `skeletons` section.

**Watch for:** overloaded signatures (@overload stubs), property getters/setters,
decorator stacks. Render decorators verbatim so agents can see `@cached_property`
without guessing.

### 2. `call_chain(src_fqn, dst_fqn, max_depth=6) → path | None`

BFS shortest path through definite edges. Return the path as a list of FQNs,
or `None` if no path exists within `max_depth`.

**Why second:** answers "how does A reach B?" which is the refactor-scoping
question agents currently solve by recursively grepping. One call replaces a
~5-turn interaction.

**Constitutional fit:** law 1 means "no path found" has the usual caveat —
*a dynamic dispatch might connect them*. Document this in the tool description;
don't let the agent read `None` as "definitely unreachable." Return `None` with
a `reason` string that encourages grep fallback when the hop count would have
been short.

**Watch for:** cycles (BFS handles, but cap visited-set size), and return paths
through `__init__` that look meaningful but are just constructor chains.

### 3. `neighborhood(symbol, depth=2, token_budget=2000) → ranked subgraph`

Bounded BFS outward in both directions (callers and callees), with
**rank-and-truncate** so the result fits `token_budget`.

**Why third:** this is the tool agents reach for when they want "everything
relevant to X." The surveyed MCPs that skipped the ranking step got complaints
about dumps; the ones that truncated by arbitrary slice (alphabetical, random)
got worse complaints about inconsistency.

**Ranking approach:** aider's personalized PageRank is the known-good reference
but needs a focus set (open files, mentioned symbols) that MCP doesn't naturally
expose. Simpler approximations worth trying first:
- Degree-based: prefer nodes with higher in+out-degree within the hosting module.
- Distance-weighted: score = `1 / (1 + depth_from_query)`, break ties by degree.
- Same-module bias: add a bonus for nodes in the same package subtree as the
  query — agents exploring `foo.bar.baz` usually care about other `foo.bar.*`
  more than `unrelated.*`.

Start with distance + same-module bias. Upgrade to PageRank only if complaints
land. Do *not* skip truncation; a fallback of "return up to N nodes" with N=50
is acceptable if ranking proves premature.

**Constitutional fit:** law 1 preserved (only definite edges traversed). Law 2
preserved (bounded BFS is fast). The agent needs to know the output was
truncated — include a `truncated: true, dropped: 37` field so they know to
narrow the query or fall back to grep.

### 4. `impact(symbol) → {direct, indirect, transitive}`

Three lists: direct callers (1 hop), indirect (2+ hops up to a cap), transitive
(everything reachable). Returned as counts + top-N by some rank, not dumped raw.

**Why fourth:** change-review workflow — "what breaks if I change X?" — is
distinct from `callers_of` because the agent wants to see the blast pattern,
not just the immediate wall. Splitting direct/indirect/transitive matches how
humans read review outputs.

**Constitutional fit:** subject to law 1's caveat — transitive impact through
a registry edge is invisible. Tool description must say so. Consider appending
a `registry_risk: true` flag when the symbol's module contains decorator-based
registration patterns detected by the analyzer's miss log — a cheap heuristic
that flags "your answer may be incomplete here" without fabricating edges.

### Later, if evidence accumulates

- **`reverse_import_graph(module)`** — who imports this module. Orthogonal to
  the call graph but agents ask for it constantly. Cheap to compute from the
  same AST pass.
- **SCIP export** — interop with Sourcegraph tooling. No v1 demand.
- **Runtime-trace augmentation** — merge `sys.setprofile` or `coverage.py`
  edges at `confidence=1.0`. Constitutionally permitted *only because* those
  edges come from actual execution — but the ergonomics (users have to run
  their test suite under trace) are bad enough that this defers until someone
  asks. If added, it replaces nothing; it adds a *second* source of definite
  edges with the same bar.

## Analyzer improvements that unlock more definite edges

The constitution (law 1) forbids low-confidence edges. That sounds like it
caps recall at whatever the current resolver can prove one-on-one. It doesn't
— a sizeable chunk of today's `misses.json` pile is *recoverable* as honest
1.0 edges with better analysis, not better scoring.

### The distinction that matters

Confidence-scored edges (the axon pattern) are sometimes framed as the answer
to duck-typed polymorphism — calls like `adapter.synthesize(...)` where
`adapter: TTSAdapter` and three classes implement it. That framing conflates
two separable things:

- **"Which target does this specific call hit?"** — one of the three. At any
  particular runtime, exactly one. This is the question that *sounds* like it
  wants a probability.
- **"Which targets is this call site capable of reaching?"** — all three.
  Over the lifetime of the codebase, across all possible `adapter` values, the
  call site genuinely connects to each subclass.

Call graphs answer the second question. "Who can reach X, what can X reach."
Under that semantics, the three subclass edges are **definite**, 1.0, no
scoring required. The miss is a resolver gap, not an ambiguity the analyzer
has to hedge.

### What the miss pile actually contains

Applying that distinction to [our miss inventory](.pyscope-mcp/misses.json)
(456 `attr_chain_unresolved`, 141 `self_method_unresolved`, 28 `bare_name_unresolved`):

- **Most of `attr_chain_unresolved`**: duck-typed dispatch where the receiver
  has a resolvable type (class attribute, constructor call, annotated
  parameter) and the method is defined on a known class hierarchy. **Recoverable
  as definite edges** via class-hierarchy analysis.
- **Most of `self_method_unresolved`**: `self.foo()` where `foo` is defined on
  the class or a superclass. Recoverable with a class-scope symbol table
  (cheap; just walk MRO during resolution).
- **Most of `bare_name_unresolved`**: imports missed by the current resolver
  (star imports, conditional imports, aliasing chains). Recoverable with a
  more careful import pass.

- **Genuinely irreducible** (the residue that would justify scoring if we
  accepted it): decorator-based registries with cross-module registration we
  can't prove is complete; `getattr(obj, runtime_str)`; `HANDLERS[key]()`
  where `HANDLERS` is assembled at runtime; metaclass / `__init_subclass__`
  registries; string-dispatched phases like our own `_run_phase_w*_*_*`. These
  stay in `misses.json` as telemetry; no edge is emitted.

The point: scoring would buy us the first group at confidence ~0.8 and the
second group at ~0.4. Doing more analyzer work buys us the first group at
1.0 and leaves the second group honestly unresolved. Law 1 prefers the second
trade; it also keeps the analyzer's output meaning single-valued.

### The analyzer work, in priority order

1. **Class-hierarchy analysis (CHA) for attribute calls.** Build a map from
   class → concrete methods defined on it and its MRO. When an attribute call
   `x.m(...)` has a receiver whose type is narrowable to one or more classes
   (from `__init__` body, annotations, or a return type at the construction
   site), emit one edge per concrete implementation reachable in the hierarchy.
   Empirically the biggest lever: should resolve the bulk of the `adapter.*`,
   `agent.*`, `model.*`, `store.*` miss clusters. Classic whole-program
   technique; well-understood complexity.

2. **Class-scope symbol resolution for `self.*` and `super().*`.** Cheap pass
   that sits on top of (1). Walk MRO; emit definite edges. Should clear
   ~nearly all `self_method_unresolved` and `super_unresolved`.

3. **Rapid Type Analysis (RTA) as a narrowing step on top of CHA.** CHA emits
   edges to *every* subclass that defines the method, even ones never
   instantiated in this program. RTA tracks which classes actually get
   constructed anywhere in the analyzed code and prunes CHA edges to that
   set. Tighter and still 1.0 — an edge only drops if the target class is
   provably never created. Optional; revisit if CHA produces too much noise
   for real queries.

4. **Import-resolution polish.** Aliasing chains, `from x import *` against
   `__all__`, conditional imports at module scope. Small wins each; worth
   doing once (1) and (2) are in so the improvements stack.

5. **Module-local registry detection for literal dicts/lists.** When a dict
   like `HANDLERS = {"a": foo, "b": bar}` is defined at module scope and the
   call is `HANDLERS[key](...)`, emit edges to every value. Still 1.0 under
   Semantics B: *some* invocation can reach each. Does not cover cross-module
   registries (those stay in misses).

### Out of scope, intentionally

- Flow-sensitive type inference beyond what's needed to narrow receiver types
  at construction sites. Full pyright-grade inference is a different project.
- Cross-module decorator-registry tracking. Doable in principle, but the
  completeness claim is fragile (a new registration in another file breaks
  the proof). Leave to `misses.json` and let agents grep.
- Runtime tracing. Covered elsewhere; remains a separate, orthogonal source
  of definite edges.

### Why this matters for the roadmap

Before building confidence-scored edges (constitutionally rejected anyway) or
shipping more tools on a lossy graph, the highest-leverage analyzer work is
(1) + (2). It converts our current 89.86% effective resolution into something
higher *without relaxing law 1*. Every tool downstream — `neighborhood`,
`impact`, `call_chain` — gets proportionally better, and the `completeness`
field (below) ends up reporting `complete` far more often.

## Cross-cutting concerns

### Ranking and truncation

Every list-returning tool needs a size cap. Three rules:

1. **Never return raw unbounded lists.** Even `callers_of` for a hot symbol
   can return 200+ entries; the current surface needs a cap retrofitted.
2. **Truncate by rank, not by slice.** Alphabetical truncation hides the
   interesting callers behind `zzz_*` test files.
3. **Tell the agent what was dropped.** Return a count, not silence.

### Symbol IDs

Dotted FQN stays primary, per the constitution and prior-art. Resist the
pressure to add `{kind}:{relpath}:{name}` as a secondary ID for overload
disambiguation — overloads in Python are rare enough to handle as a suffix
(`foo.bar#overload_2`) if they become a real problem.

### Storage

JSON is fine until it isn't. The transition trigger is law 2's 2-second
startup ceiling, not index file size per se. When we hit it, SQLite + FTS5
is the next step (every non-toy surveyed server landed there). Skip the
intermediate steps (pickle caches, in-memory marshaled blobs) that show up
in some MCPs — they're all regretted.

### Honesty surfaces

Several tools above include a "this answer may be incomplete because dynamic
dispatch" escape hatch. Standardize these: every tool that traverses edges
returns a `completeness` field:

```json
{"status": "complete" | "likely_partial" | "known_partial", "reason": "..."}
```

- `complete`: no miss-log entries touched any traversed symbol.
- `likely_partial`: query subtree overlaps modules with `misses.json` entries.
- `known_partial`: traversal terminated at a call site we know we can't resolve.

This is law 1 expressed at the API: we ship definite edges *and* an honest
signal about what we couldn't see. The agent can then decide whether to grep.

## Build order

1. Land the analyzer (blocks everything).
2. Retrofit `completeness` to existing tools; add size caps.
3. `file_skeleton` — small, high leverage, unblocks agent context workflows.
4. `call_chain` — small, high leverage, replaces common grep spirals.
5. `neighborhood` with rank-and-truncate — larger, but this is where the
   "graph beats grep" thesis gets proven.
6. `impact` — straightforward once `neighborhood` exists.

Features 1–4 are the v1 surface. 5–6 are v1.1. Anything not listed (confidence
edges, dead-code detection, CPGQL, embeddings) requires a constitutional
amendment, not a feature request.
