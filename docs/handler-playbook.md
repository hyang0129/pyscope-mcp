# Handler playbook

How to add a new resolution handler to the analyzer. The pattern is the
loop we used to land the MRO handler (`super()` + inherited `self.method`)
and the indirect-dispatch handler (`executor.submit`, `functools.partial`,
`run_in_executor`, `Pool.map`, etc.).

Every handler follows the same four beats: **diagnose, design, implement,
verify.** Skipping any step produces either a dead handler (no real-world
impact) or a noisy one (false positives that corrode trust in the graph).

## The loop

### 1. Diagnose — from real misses, not theory

Start from [`misses.json`](../README.md) on a representative target repo,
not a synthetic one. Categorize exemplars from `unresolved_calls` by what
it would take to resolve each:

- **Genuinely external** (stdlib, third-party) — leave alone.
- **Genuinely non-calls** (f-string attribute reads mis-classified as calls,
  etc.) — fix the classifier, not the resolver.
- **Recoverable with static facts we could collect** — candidate handler.

For a candidate pattern, answer:
- How many exemplars share it? (rough count from `pattern_counts`)
- Does it silently create false-positive `dead_keys` in the rollup? This is
  the highest-value signal: a recovered inbound edge often removes a
  misleading dead-key entry that an agent would otherwise act on.
- What's the *confidence* of a resolution if we did it? A method reference
  passed to `ThreadPoolExecutor.submit` is high-confidence (the reference
  is syntactically a callable). A string key looked up in a dict is low.
  Low-confidence patterns wait for the confidence-tagged-edges schema.

### 2. Design — resolver signature + pass-1 facts

Resolvers are pure functions in `analyzer/resolution.py` that take AST
nodes plus a `ResolveCtx` (static project facts) and return an FQN or
`None`. Before writing the resolver, ask:

- **What new static facts are needed?** If the pattern depends on class
  hierarchy, import aliasing, decorator identity, etc., that data has to
  be collected in pass 1 (`analyzer/discovery.py`) and threaded through
  `ResolveCtx`. Adding a field to `ResolveCtx` is cheap; collecting it
  later as a second analysis pass is not.
- **Where does the resolver get invoked?** Three slots in the visitor:
  1. As part of `_resolve_expr` (direct call targets, including `super()`
     and `self.method`).
  2. As an *extra* edge emitter in `visit_Call`, after the primary
     resolve (indirect dispatch — the dispatcher call still misses, but
     the callable argument gets its own edge).
  3. Future: decorator expressions, type-annotation-driven resolution.
- **What is the failure mode?** Returning `None` must be silent. Handlers
  never record misses — that's the visitor's job after all handlers have
  had a chance.

### 3. Implement — handler + pass-1 collector + visitor wiring

Split the change across three files even if the handler is small:

- `analyzer/discovery.py` — new collector function if new facts are needed
  (e.g. `collect_class_bases`). Runs over already-parsed ASTs; do not
  re-parse.
- `analyzer/resolution.py` — the pure resolver. Add a module-level
  whitelist/constant (e.g. `DISPATCHER_NAMES`) if the pattern has a small
  fixed vocabulary.
- `analyzer/visitor.py` — call the resolver from the appropriate slot.
  For extra-edge handlers, emit via `_emit` independently of the primary
  resolve; do **not** suppress the primary miss.

Naming convention: `resolve_<pattern>` for single-call resolvers,
`is_<pattern>_call` + `<pattern>_callable_arg` for argument extractors.

### 4. Verify — synthetic tests, then real-repo diff

Two layers:

**Synthetic unit tests.** One test file per handler group (e.g.
`tests/test_analyzer_m6.py`). Each test builds a minimal package with
`_make_package`, runs `build_raw`, and asserts a specific edge exists or
is absent. Always include a **false-positive guard** test (lambda arg,
call-result arg, external-parent super — something the handler must
*not* emit for).

**Real-repo delta.** Snapshot `misses.json` before, rebuild, diff the
summary:

```
in-package edges:   X -> Y  (+N)
unresolved calls:   X -> Y  (-N)
<pattern> count:    X -> Y
dead_keys:          before - after (recovered: {set diff})
```

Recovered dead keys are the headline number — they represent false-dead
verdicts the rollup was handing agents. If a handler adds edges but
recovers zero dead keys, re-examine whether the pattern actually
mattered.

## Worked example: MRO + indirect dispatch

The first application of this loop:

**Diagnosed:** 50 exemplars from `angle_brainstorm_agent.py` showed two
recoverable patterns: `super().__init__(run_dir=run_dir)` (one exemplar,
but representative of a broad class — every subclass constructor) and
`executor.submit(self._enrich_concept, ...)` (one exemplar, but
responsible for marking `_enrich_concept` as a dead key).

**Designed:** A `walk_mro` helper in `resolution.py` plus a
`collect_class_bases` collector in `discovery.py`, reused by both
`super().X()` and `self.X()` (latter as fallback when the direct
`{class_fqn}.X` lookup misses). A `DISPATCHER_NAMES` whitelist plus
`dispatcher_callable_arg` helper, emitted as an *extra* edge in
`visit_Call` without suppressing the primary `.submit` miss.

**Implemented:** Three-file change spanning `discovery.py` (pass-1
collector), `resolution.py` (MRO walker + dispatcher whitelist), and
`visitor.py` (both slots wired).

**Verified:** 11 new synthetic tests (same-module super, cross-module
super, external-parent skip, inherited self.method, override preference,
diamond MRO, `submit`, `partial`, `run_in_executor` second-arg, lambda
non-emission, call-result non-emission). On `video_agent_long`:
+58 in-package edges, `self_method_unresolved` 283 → 244, 6 recovered
dead keys including the motivating `_enrich_concept`.

## When this loop is the wrong tool

- **Pattern requires runtime information.** String-keyed dispatch, dict
  registries, `__init_subclass__`. Don't build a resolver that guesses;
  wait for confidence-tagged edges.
- **Pattern is one occurrence in one file.** Skip. Per-case resolvers
  rot. Only build handlers for patterns that recur across the repo.
- **Pattern is a classifier bug, not a resolution bug.** If the "miss"
  is a non-call (f-string attribute read, type annotation) being counted
  as a call, fix `classify_miss` and stop.
