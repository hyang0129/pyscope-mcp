# pyscope-mcp — Constitution (Mini)
<!-- Auto-derived from CONSTITUTION.md — do not edit directly -->

**Thesis:** An MCP graph server that lets Claude answer code-structure questions — who calls X, what does X reach, what's the shortest path from A to B — faster and more reliably than repeated `grep` + `read`.

## Laws

### Law 1 — Never mislead the consumer; every edge and every query answer declares its own certainty and completeness
**Anti-pattern:** Emitting an edge for any call site whose target cannot be proven by static analysis — unresolved `attr_chain`, duck-typed dispatch, string-keyed registries, `getattr`, metaclass magic. These call sites belong in `misses.json` for telemetry, not in the graph. A list-returning tool that truncates without a machine-readable signal the consumer can act on (e.g. `truncated: true, total: N`). A completeness-dependent query — dead-code detection, "is X unreachable", exhaustive caller list — returning confident results without labeling them as candidates that require grep confirmation. Any return value that looks like "the answer" when it is in fact "the answer modulo static-analysis blind spots and a size cap."

### Law 2 — Minimal startup time once set up (<500 ms target, <5 s ceiling)
**Anti-pattern:** Silently triggering a build when the index file is missing instead of erroring out. Filesystem watchers or background rebuild threads inside the serve process. Pickle caches or any on-disk format whose cold-load cost scales with payload complexity rather than payload size. Imports inside hot paths that pull analyzer dependencies into the serve process.

### Law 3 — Git-in-sync graph is the zero-friction default; drift is cheaply detectable
**Anti-pattern:** Shipping an index without a git SHA and content hash in the header. Requiring manual coordination (non-default flags, out-of-tree bookkeeping, after-merge rebuilds) to tie a PR's commit to its graph artifact. A `pyscope-mcp verify` command that takes flags, calls out over the network, has side effects, or returns anything other than 0 / non-zero with a machine-readable reason. An `.pyscope-mcp/index.ref` pointing at a URL whose content hash no longer resolves, or whose CI step didn't re-upload on the latest commit. A PR check that validates the reference *exists* without validating that its hash matches the current tree. Non-deterministic analyzer output (sort-order drift, timestamps in the payload) that makes the hash check meaningless.

### Law 4 — Graph update must not exceed 1 minute for standard PRs
**Anti-pattern:** An incremental rebuild path that produces different edges than a full rebuild on the same tree, without a CI diff catching it. A full rebuild that aborts on one parse failure instead of isolating the failing file. Any global phase that scales non-linearly with repo size and can't be re-run cheaply.

---

If any proposed change violates a law above: redesign required — not a carve-out.
