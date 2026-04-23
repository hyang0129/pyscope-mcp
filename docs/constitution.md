# pyscope-mcp constitution

The non-negotiables. Every design decision, tool, and PR should be checked against these.
If a feature can't be built without violating one, the feature loses.

## Purpose

An MCP graph server that lets Claude answer code-structure questions (who calls X,
what does X reach, what's the shortest path from A to B) faster and more reliably
than repeated `grep` + `read`. The graph is the product; MCP is just the delivery
surface.

## The five laws

### 1. Only definite edges. Never surface an uncertain one.

The graph is **"definite edges, with silent false negatives."** If we're not ≥99.9%
sure A calls B, no edge. An agent that trusts a wrong edge wastes more time than one
that has to fall back to grep.

- No edges from unresolved `attr_chain` / duck-typed dispatch / string-keyed registries /
  `getattr` / metaclass magic. These call sites are recorded in `misses.json` for
  telemetry, **not** merged into the graph.
- "Low-confidence edges with a score" (the axon pattern) is explicitly rejected for v1.
  Agents don't calibrate well to edge confidence; they round 0.7 to 1.0.
- Query responses must be honest about incompleteness: any tool whose correctness
  depends on completeness (dead-code detection, "is X unreachable") is either omitted
  or returns results labeled as *candidates requiring grep confirmation*.

### 2. Minimal startup time once set up.

MCP `serve` must open the index and be ready to answer within a human blink
(<500 ms target, <5 s ceiling) on a pre-built index. Analysis is a separate,
ahead-of-time step.

- No lazy build on first query. If the index file is missing, `serve` errors out —
  it does not silently trigger analysis.
- No filesystem watchers, no background rebuild threads. `reload` is the only
  invalidation path.
- JSON on disk while indexes are small and human-inspectable. Move to SQLite+FTS5
  before anything more exotic, and only when JSON startup stops meeting the ceiling.
- Server has zero analyzer dependency at runtime.

### 3. The code must make staleness detectable. Ops enforces the gate.

A graph that disagrees with `HEAD` is worse than no graph — it lies with a straight
face. This library cannot *prevent* a stale graph from being merged; that is
ultimately a CI / branch-protection decision the hosting project owns. What this
library **must** do is make "is this graph stale?" a cheap, unambiguous, scriptable
question, so the hosting project can wire it into a required check trivially.

Code-side obligations:

- Every index carries the git commit SHA it was built from, plus a content hash
  over the exact bytes analyzed (e.g. sorted `(relpath, sha256)` pairs of every
  `.py` file the analyzer consumed). Both live in the index header.
- `pyscope-mcp verify` recomputes the content hash against the working tree and
  exits 0 / non-zero with a machine-readable reason. No flags, no side effects,
  no network — just a pure comparison. This is the hook CI calls.
- Analyzer output is deterministic (see corollaries) so the hash is meaningful.

Out of scope for this repo:

- Deciding *when* verify runs, *which* branches it blocks, or *how* drift is
  surfaced in PR UI. That's `.github/workflows/*.yml` in the hosting project,
  not our problem.
- Hard-failing the MCP server on SHA mismatch. The server will warn loudly, but
  refusing to start would make dev loops miserable; enforcement belongs upstream
  of the running server.

### 4. Graph must ship with the PR.

A PR without a fresh graph is not mergeable. The graph does not have to live in the
git tree — a reference to an external artifact is fine, as long as the reference
itself is versioned with the PR.

- Acceptable forms: (a) committed `.pyscope-mcp/index.json`, (b) committed
  `.pyscope-mcp/index.ref` containing a URL + content hash pointing to a CI-uploaded
  artifact (S3 / R2 / GitHub Actions artifact / release asset).
- Whichever form, the reference is tied to the commit SHA. The PR check from law 3
  resolves the reference and validates the hash.
- Large indexes (>1 MB gzipped) should default to the external-reference form to
  keep git history clean. Small repos can commit the JSON directly.

### 5. Graph update must not exceed 1 minute for standard PRs.

If updating the graph is slow, contributors will skip it or automate around it
badly. Full rebuild on every PR is fine for small repos; beyond that, incrementality
becomes mandatory.

- Target: **≤60 s wall-clock** on a typical PR (≤50 changed files) on standard CI
  hardware.
- Full rebuild is the baseline and always available (`pyscope-mcp build`).
- Beyond the threshold where full rebuild exceeds 60 s, incremental rebuild is
  required: re-analyze changed files only, splice their new edges into the existing
  index, drop edges from deleted files. The global phase (dead-key rollup, stats) is
  cheap and always re-runs.
- Incremental correctness is non-negotiable: if incremental output ever disagrees
  with a full rebuild on the same tree, the feature is broken and reverts to full
  rebuild until fixed. A CI job periodically diffs incremental vs. full.

## Corollaries

These follow from the five laws; they're not independent constraints but they
catch common mistakes.

- **No runtime import of target code.** pycg's fatal mistake. Breaks per-file
  isolation (law 5's incrementality relies on it), breaks determinism (law 3's
  hash relies on it), and surfaces spurious edges when imports have side effects
  (law 1).
- **Deterministic output.** Same tree → same JSON, byte-for-byte. Required by
  law 3's hash check. Sort keys and lists; no timestamps in the payload.
- **Per-file error isolation in the analyzer.** A parse failure on one file
  skips that file, warns, and keeps going. Required for law 5 (one bad file
  can't blow the 60 s budget by aborting the run) and law 1 (a partial index
  of definite edges still satisfies the contract).
- **Dotted FQN as the primary symbol ID.** Matches LSP, matches the existing
  downstream code, survives refactors better than path-based schemes.

## What we explicitly do not ship

Not because they're bad ideas, but because they conflict with a law above or
add surface area we can't maintain cheaply:

- Confidence-scored edges (law 1).
- Lazy / on-query rebuild (law 2).
- Filesystem watchers in the server process (law 2).
- Raw query endpoints — CPGQL, Cypher, SQL. Agents write bad queries, and the
  blast radius is unbounded.
- Regex "parsers" for non-Python languages. Either we parse it properly or we
  don't claim to support it.
- Pickle caches. Versioned JSON or SQLite, nothing else.
- LSIF. SCIP is the successor; LSIF is deprecated upstream.

## Review heuristic

When a PR adds a feature, ask in order:
1. Does it keep law 1? (No uncertain edges leaking into query answers.)
2. Does it keep law 2? (Server startup still <5 s.)
3. Does it keep laws 3 and 4? (Staleness still cheaply detectable by CI, graph still shippable with the PR.)
4. Does it keep law 5? (Incremental build still ≤60 s on standard PRs.)

If any answer is "no or unclear," the feature needs redesign, not a carve-out.
