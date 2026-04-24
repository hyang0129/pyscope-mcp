# pyscope-mcp — Constitution

The non-negotiables. Every design decision, tool, and PR should be checked against
these. If a feature can't be built without violating one, the feature loses.

## Thesis

An MCP graph server that lets Claude answer code-structure questions — who calls X,
what does X reach, what's the shortest path from A to B — faster and more reliably
than repeated `grep` + `read`. The graph is the product; MCP is just the delivery
surface.

## Laws

## Law 1 — Never mislead the consumer; every edge and every query answer declares its own certainty and completeness

**Why:** An agent that gets a confident-looking wrong edge, or a silently-truncated
"complete" list, acts on bad information without the warning it needs to
compensate. We cannot prevent incompleteness — static analysis has inherent blind
spots (registries, duck typing, metaclass magic), and list-returning tools have
size caps — but we can refuse to misrepresent it. "I returned 50 of ~180 results"
is strictly more useful than "50 results" presented as exhaustive; "this dispatch
target can't be resolved — logged in `misses.json`" is strictly more useful than a
plausible-looking but unverifiable edge. Any answer a consumer relies on for
certainty or completeness must disclose when either is incomplete. Better to say
"I don't know" than to provide a half-baked answer the consumer treats as
complete.

**Rejected Alternative:** Considered two families of the same mistake. (a)
Confidence-scored edges — the "axon" pattern: low-confidence call targets shipped
with a numeric score, caller picks a threshold; rejected because agents don't
calibrate thresholds and round 0.7 to 1.0, paying for the wrong edge with a wrong
action. (b) Silent top-N truncation on list-returning tools — the default in
shipped MCPs: return the first page, no signal that more exists; rejected because
a partial list presented as complete is the exact failure this law exists to
prevent. In both cases the project takes explicit silence (omit the edge, label
the result) over spurious confidence.

**Anti-pattern:** Emitting an edge for any call site whose target cannot be
proven by static analysis — unresolved `attr_chain`, duck-typed dispatch,
string-keyed registries, `getattr`, metaclass magic. These call sites belong in
`misses.json` for telemetry, not in the graph. A list-returning tool that
truncates without a machine-readable signal the consumer can act on (e.g.
`truncated: true, total: N`). A completeness-dependent query — dead-code
detection, "is X unreachable", exhaustive caller list — returning confident
results without labeling them as candidates that require grep confirmation. Any
return value that looks like "the answer" when it is in fact "the answer modulo
static-analysis blind spots and a size cap."

## Law 2 — Minimal startup time once set up (<500 ms target, <5 s ceiling)

**Why:** Slow startup shows up as two failures that both erode trust in the tool.
First: Claude's first MCP tool call can time out before `serve` is ready, leaving
the user with a hard error on the very first interaction. Second, and more
corrosive: any warmup long enough for a human to notice pushes users to ask "is
it working?" — the kind of friction that ends in the tool being disabled. The
invariant isn't "fast is nice"; it's "startup must be invisible enough that
nobody considers whether the tool is alive."

**Rejected Alternative:** Considered: lazy / on-query build (pycg-style implicit
analysis — if the index is missing, build it on first query). Rejected because:
first-query latency would blow past the ceiling on any non-trivial repo, and the
serve path would inherit the analyzer's full dependency graph forever, making
cold start progressively slower as the analyzer grows.

**Anti-pattern:** Silently triggering a build when the index file is missing
instead of erroring out. Filesystem watchers or background rebuild threads inside
the serve process. Pickle caches or any on-disk format whose cold-load cost scales
with payload complexity rather than payload size. Imports inside hot paths that
pull analyzer dependencies into the serve process.

## Law 3 — Git-in-sync graph is the zero-friction default; drift is cheaply detectable

**Why:** A graph that disagrees with `HEAD` is worse than no graph — it lies with
a straight face. Agents treat graph answers as authoritative; when the graph is
stale they confidently recommend call paths that no longer exist, and the user
pays the debugging cost. Detection alone isn't enough: if keeping the graph fresh
is friction-heavy, contributors skip it or automate around it badly, and the
freshness contract erodes silently. This library can't *prevent* a stale graph
from being merged — that's a CI / branch-protection decision in the hosting
project — but it must make the in-sync path effectively free (near-zero
additional work to "do it the right way") and make drift a single-command,
machine-readable question.

**Rejected Alternative:** Considered: (a) a repo shape that requires active
maintenance to keep the graph in sync — e.g. maintaining one global graph
out-of-band and re-running `build` on merge to main — which would leave
reviewers querying a previous commit's structure during the window that matters
most; and (b) hard-failing the MCP server on SHA mismatch to concentrate
enforcement inside the library, which would make dev loops miserable and place
enforcement at the wrong layer. Rejected because the right-thing path must be
the default path. If there's any gap between "graph in sync with git" and "what
the contributor does by default," the gap gets exploited, and the staleness
contract becomes aspirational.

**Anti-pattern:** Shipping an index without a git SHA and content hash in the
header. Requiring manual coordination (non-default flags, out-of-tree bookkeeping,
after-merge rebuilds) to tie a PR's commit to its graph artifact. A
`pyscope-mcp verify` command that takes flags, calls out over the network, has
side effects, or returns anything other than 0 / non-zero with a machine-readable
reason. An `.pyscope-mcp/index.ref` pointing at a URL whose content hash no
longer resolves, or whose CI step didn't re-upload on the latest commit. A PR
check that validates the reference *exists* without validating that its hash
matches the current tree. Non-deterministic analyzer output (sort-order drift,
timestamps in the payload) that makes the hash check meaningless.

## Law 4 — Graph update must not exceed 1 minute for standard PRs

**Why:** If updating the graph is slow, contributors skip it or automate around
it badly — stale indexes land, CI checks get disabled, the freshness contract
silently erodes. A slow step gets defeated, not accepted, so the budget itself
is the invariant (and reinforces Law 3: the "zero-friction default" is only
zero-friction if it's also fast).

**Rejected Alternative:** Considered: full rebuild on every PR, regardless of
repo size, for simplicity. Rejected because: full rebuild is fine for small
repos but turns into a contributor-hostile tax once it exceeds 60 s — pushing
the project toward exactly the "skip or automate around" failure that motivates
this law.

**Anti-pattern:** An incremental rebuild path that produces different edges than
a full rebuild on the same tree, without a CI diff catching it. A full rebuild
that aborts on one parse failure instead of isolating the failing file. Any
global phase that scales non-linearly with repo size and can't be re-run
cheaply.

## Rejected Alternatives

Whole-project designs and tools evaluated against the laws above. Named because
they make the laws concrete — each entry is a project or pattern that took a
different call on one of these invariants. See [docs/prior-art.md](docs/prior-art.md)
for the detailed survey.

- **pycg** (ICSE 2021 reference implementation): pyscope-mcp started as a thin
  MCP wrapper around it. Dropped because it runtime-imports target code
  (Corollary 1.1: breaks Law 1's definite-edge guarantee when imports have side
  effects, breaks Law 3's hash determinism, breaks Law 4's per-file isolation),
  has no per-file error isolation (one bad file aborts the run), and is
  unmaintained since 2022 with no Python 3.11 fix.

- **axon** (harshkedia177/axon): confidence-scored edges 0.0–1.0 on every call,
  plus an `axon_dead_code` tool. Violates Law 1 — agents don't calibrate
  confidence and round to 1.0, and a dead-code tool whose correctness depends
  on graph completeness returns confident falsehoods. Its staged pipeline
  (file-local vs. global phases) is a genuinely good incrementality pattern
  and may be reconsiderable for Law 4; the confidence surface is not.

- **mcp-codebase-index** (MikeRecognex): Python `ast` plus *regex parsers for
  other languages*; does an implicit git-diff-and-rebuild on the query path if
  ≤20 files changed. Violates Law 1 (regex parsers surface uncertain edges as
  certain) and Law 2 (rebuild on query path blows the startup ceiling). We
  don't claim multi-language support and we don't silently rebuild.

- **codebadger** (Lekssays, Joern CPG): full Code Property Graph exposed through
  a raw CPGQL endpoint. Rejected even though CPG is strictly more expressive
  than our graph — a raw query language for agents has unbounded blast radius
  (agents write bad queries), and the 500 MB / 10-min gen limits would violate
  Law 4 on large repos.

- **SCIP-based MCPs** (MapleRook/rook, drxddy/polymcp, bnomei/frigg) on
  `scip-python`: real format, real tooling. **Deferred, not rejected.** Dotted
  FQN remains our primary ID (matches LSP, matches what `CallGraphIndex` already
  expects). SCIP export is a later option for Sourcegraph interop, not a v1
  concern.

- **LSIF**: predecessor to SCIP; deprecated upstream. Rejected as an interop
  target in favor of SCIP if we ever add cross-tool export.

- **Pickle-based indexes**: common in the ecosystem. Rejected categorically.
  Versioned JSON now; SQLite+FTS5 if JSON stops scaling; nothing else. Pickle's
  version-coupling and opacity violate the "human-inspectable while small"
  convention supporting Laws 2 and 3.

- **Filesystem watchers in the serve process** (pattern, not a project): used
  by several MCP servers to auto-pick-up changes. Rejected — violates Law 2
  (background work in the serve path) and muddles the invalidation contract
  that `reload` owns under Law 3.

- **Scalpel** (Penn State, type-aware pycg successor): plausible alternative
  static analyzer that may resolve some duck-typed cases. **Deferred** — its
  import-hook fragility is unvalidated, and we have a working AST-based path.

- **Raw-query endpoints for our own graph** (CPGQL, Cypher, SQL over the index):
  rejected for the same reason as codebadger — bad queries from agents produce
  bad answers confidently, violating Law 1's honesty contract at the query
  layer.

## Corollaries

These follow from the laws; they're not independent constraints but they catch
common mistakes.

**Corollary 1.1 / 3.1 / 4.1** — No runtime import of target code (pycg's fatal
mistake). Breaks per-file isolation that Law 4's incrementality relies on, breaks
the determinism Law 3's hash check needs, and surfaces spurious edges when imports
have side effects (Law 1).

**Corollary 3.2** — Deterministic output: same tree → same JSON, byte-for-byte.
Required by Law 3's hash check. Sort keys and lists; no timestamps in the payload.

**Corollary 1.2 / 4.2** — Per-file error isolation in the analyzer: a parse
failure on one file skips that file, warns, and keeps going. Required by Law 4
(one bad file can't blow the 60 s budget by aborting the run) and Law 1 (a partial
index of definite edges still satisfies the contract).

**Convention (not a corollary)** — Dotted FQN as the primary symbol ID. Matches
LSP and downstream code. Kept here for visibility but doesn't derive from a law;
may move to `CLAUDE.md` during a later amendment pass.

## Review Heuristic

When a PR adds a feature, ask in order:

1. Does this keep Law 1 (no misrepresentation — no uncertain edges in the graph, no silently-truncated query answers, no completeness-dependent tool returning confident results)?
2. Does this keep Law 2 (server startup still inside the ceiling, nothing
   silently shifted into the serve path)?
3. Does this keep Law 3 (git-in-sync graph still the zero-friction default,
   drift still cheaply detectable)?
4. Does this keep Law 4 (incremental build still ≤60 s on standard PRs, still
   agrees with full rebuild)?

If any answer is "no or unclear", the feature requires redesign — not a carve-out.
