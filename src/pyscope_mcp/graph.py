from __future__ import annotations

import hashlib
import json
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal, TypedDict

from pyscope_mcp.types import (
    CalleesResult,
    Completeness,
    ModuleResult,
    NeighborhoodResult,
    ReferencedByEntry,
    ReferencedByResult,
    SearchResult,
    StatsResult,
)

INDEX_VERSION = 5


class SymbolSummary(TypedDict):
    fqn: str
    kind: str  # "function" | "class" | "method"
    signature: str
    lineno: int


_SKELETON_CAP = 50


class GraphReader:
    """Thin facade over the site-keyed ``nodes`` dict.

    All traversal in :class:`CallGraphIndex` goes through this reader — no
    parallel adjacency structure, no separate storage.  Edge kinds are first-class:
    every method accepts an optional ``kinds`` filter so callers can restrict
    traversal to a subset of edge kinds (e.g. ``kinds=("call",)`` for call-only
    traversal, ``kinds=None`` for all kinds).

    The reader has zero state beyond the ``nodes`` reference passed to
    ``__init__``.  No caches, no rebuilt structures.  Caches (like
    ``_in_degree_threshold``) live on :class:`CallGraphIndex`, not here.
    """

    def __init__(self, nodes: dict[str, dict]) -> None:
        self.nodes = nodes

    # ---------------------------------------------------------------- existence

    def __contains__(self, fqn: object) -> bool:
        return fqn in self.nodes

    def nodes_iter(self) -> Iterator[str]:
        return iter(self.nodes)

    def num_nodes(self) -> int:
        return len(self.nodes)

    # ---------------------------------------------------------- kind-aware neighbors

    def successors(
        self, fqn: str, kinds: Iterable[str] | None = None
    ) -> Iterator[str]:
        """Iterate callees of *fqn*, optionally filtered to *kinds*."""
        calls = self.nodes.get(fqn, {}).get("calls", {}) or {}
        seen: set[str] = set()
        if kinds is None:
            for targets in calls.values():
                for t in targets:
                    if t not in seen:
                        seen.add(t)
                        yield t
        else:
            for kind in kinds:
                for t in calls.get(kind, ()):
                    if t not in seen:
                        seen.add(t)
                        yield t

    def predecessors(
        self, fqn: str, kinds: Iterable[str] | None = None
    ) -> Iterator[str]:
        """Iterate callers of *fqn*, optionally filtered to *kinds*."""
        called_by = self.nodes.get(fqn, {}).get("called_by", {}) or {}
        seen: set[str] = set()
        if kinds is None:
            for sources in called_by.values():
                for s in sources:
                    if s not in seen:
                        seen.add(s)
                        yield s
        else:
            for kind in kinds:
                for s in called_by.get(kind, ()):
                    if s not in seen:
                        seen.add(s)
                        yield s

    def out_edges(
        self, fqn: str, kinds: Iterable[str] | None = None
    ) -> Iterator[tuple[str, str, str]]:
        """Yield (src, dst, kind) tuples for all outgoing edges from *fqn*."""
        calls = self.nodes.get(fqn, {}).get("calls", {}) or {}
        if kinds is None:
            for kind, targets in calls.items():
                for t in targets:
                    yield (fqn, t, kind)
        else:
            for kind in kinds:
                for t in calls.get(kind, ()):
                    yield (fqn, t, kind)

    def in_edges(
        self, fqn: str, kinds: Iterable[str] | None = None
    ) -> Iterator[tuple[str, str, str]]:
        """Yield (src, dst, kind) tuples for all incoming edges to *fqn*."""
        called_by = self.nodes.get(fqn, {}).get("called_by", {}) or {}
        if kinds is None:
            for kind, sources in called_by.items():
                for s in sources:
                    yield (s, fqn, kind)
        else:
            for kind in kinds:
                for s in called_by.get(kind, ()):
                    yield (s, fqn, kind)

    # ------------------------------------------------------------------ degree

    def out_degree(self, fqn: str, kinds: Iterable[str] | None = None) -> int:
        """Return the number of distinct callees of *fqn* (kind-filtered)."""
        return sum(1 for _ in self.successors(fqn, kinds))

    def in_degree(self, fqn: str, kinds: Iterable[str] | None = None) -> int:
        """Return the number of distinct callers of *fqn* (kind-filtered)."""
        return sum(1 for _ in self.predecessors(fqn, kinds))

    # --------------------------------------------------------------- BFS primitive

    def bfs(
        self,
        start: str,
        depth: int,
        direction: Literal["calls", "called_by"],
        kinds: Iterable[str] | None = None,
    ) -> dict[str, tuple[int, dict[str, int]]]:
        """BFS from *start* up to *depth* hops in *direction*.

        Returns ``{fqn: (min_hop, {kind: hop_at_first_seen})}`` for each
        reachable node (excluding *start* itself).

        ``direction="calls"`` traverses successors (callees direction).
        ``direction="called_by"`` traverses predecessors (callers direction).

        ``kinds=None`` follows all edge kinds.  ``kinds=("call",)`` restricts
        traversal to call edges only.

        ``depth=0`` returns ``{}``.  If *start* is not in the graph, returns ``{}``.
        """
        if depth <= 0 or start not in self.nodes:
            return {}

        kinds_list: list[str] | None = list(kinds) if kinds is not None else None

        # seen: fqn -> (min_hop, {kind: hop_at_first_seen})
        seen: dict[str, tuple[int, dict[str, int]]] = {}
        frontier: list[str] = [start]

        for hop in range(1, depth + 1):
            nxt: list[str] = []
            for n in frontier:
                record = self.nodes.get(n, {})
                bucket = record.get(direction, {}) or {}
                if kinds_list is None:
                    neighbors_iter: Iterable[tuple[str, str]] = (
                        (neighbor, kind)
                        for kind, neighbors in bucket.items()
                        for neighbor in neighbors
                    )
                else:
                    neighbors_iter = (
                        (neighbor, kind)
                        for kind in kinds_list
                        for neighbor in bucket.get(kind, ())
                    )
                for neighbor, kind in neighbors_iter:
                    if neighbor == start:
                        continue
                    if neighbor not in seen:
                        seen[neighbor] = (hop, {kind: hop})
                        nxt.append(neighbor)
                    else:
                        cur_hop, cur_kinds = seen[neighbor]
                        if kind not in cur_kinds:
                            cur_kinds[kind] = hop
                        if hop < cur_hop:
                            seen[neighbor] = (hop, cur_kinds)
            frontier = nxt

        return seen

    # ------------------------------------------------------------------ counts

    def num_edges(self, kind: str | None = None) -> int:
        """Return total edge count, optionally restricted to *kind*.

        Counts unique (src, dst) pairs per kind — does not deduplicate across
        kinds when ``kind=None``.
        """
        total = 0
        for record in self.nodes.values():
            calls = record.get("calls", {}) or {}
            if kind is None:
                total += sum(len(v) for v in calls.values())
            else:
                total += len(calls.get(kind, ()))
        return total


def _compute_content_hash(nodes: dict[str, dict]) -> str:
    """Return SHA-256 hex digest of the deterministically serialised *nodes* dict.

    Hashes the canonical (sorted) projection of the call edges in the
    site-keyed shape: ``nodes[caller]["calls"]["call"]``.  Only ``call`` edges
    contribute today (matching pre-migration semantics so freshness checks
    survive the schema bump for repos that have not added other kinds yet).

    Sorted keys and sorted callee lists ensure the same graph always yields
    the same hash, regardless of insertion order.
    """
    projection: dict[str, list[str]] = {}
    for caller in sorted(nodes):
        record = nodes[caller]
        callees = list(record.get("calls", {}).get("call", ()))
        if callees:
            projection[caller] = sorted(callees)
    canonical = json.dumps(projection, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _raw_to_nodes(raw: dict[str, list[str]]) -> dict[str, dict]:
    """Convert legacy ``raw`` (caller → [callees]) to site-keyed ``nodes`` shape.

    Build-time inversion: every ``(caller, callee)`` pair in *raw* yields
    ``nodes[caller]["calls"]["call"]`` (forward) and
    ``nodes[callee]["called_by"]["call"]`` (reverse).  Both endpoints get a
    ``NodeRecord`` skeleton with sorted, deduplicated lists per kind so the
    serialisation determinism invariant (Corollary 3.2) holds regardless of
    insertion order.
    """
    forward: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {}
    for caller, callees in raw.items():
        forward.setdefault(caller, set()).update(callees)
        for callee in callees:
            reverse.setdefault(callee, set()).add(caller)
            forward.setdefault(callee, set())  # ensure callee node exists
        # Keep callers with empty callee list as nodes too.
        forward.setdefault(caller, set())

    all_symbols = set(forward) | set(reverse)
    nodes: dict[str, dict] = {}
    for sym in sorted(all_symbols):
        record: dict[str, dict[str, list[str]]] = {"calls": {}, "called_by": {}}
        callees = forward.get(sym, set())
        if callees:
            record["calls"]["call"] = sorted(callees)
        callers = reverse.get(sym, set())
        if callers:
            record["called_by"]["call"] = sorted(callers)
        nodes[sym] = record
    return nodes


def _nodes_to_raw(nodes: dict[str, dict]) -> dict[str, list[str]]:
    """Project the site-keyed ``nodes`` shape back to the legacy ``raw`` form.

    Used to support the deprecated ``raw`` property and ``from_raw`` constructor
    while Child #3 of epic #76 migrates the test corpus.  Only ``call`` edges
    appear in the projection — non-call kinds are not part of the legacy
    contract and are silently omitted (they remain available via
    ``CallGraphIndex.nodes``).
    """
    out: dict[str, list[str]] = {}
    for caller in sorted(nodes):
        callees = nodes[caller].get("calls", {}).get("call")
        if callees:
            out[caller] = sorted(callees)
    return out


@dataclass
class CallGraphIndex:
    """Function- and module-level call graph over a Python repo.

    The source of truth is ``nodes``: a site-keyed mapping
    ``{symbol_fqn: {"calls": {kind: [callees]}, "called_by": {kind: [callers]}}}``.
    All traversal goes through ``_reader``, a :class:`GraphReader` constructed
    from ``nodes``.  ``module_index`` maps each module FQN to the set of symbol
    FQNs it contains, supporting module-level traversal without a second graph.

    The legacy ``raw`` property projects the same call edges in the
    pre-migration shape for transitional callers.

    ``skeletons`` maps relative file paths → pre-computed lists of SymbolSummary
    dicts, populated during ``pyscope-mcp build`` (index version 2+).
    """

    root: Path
    # v5+: site-keyed nodes — replaces the legacy ``raw`` field.  The ``raw``
    # property below derives a call-only projection for transitional callers.
    nodes: dict[str, dict] = field(default_factory=dict)
    skeletons: dict[str, list[SymbolSummary]] = field(default_factory=dict)
    # None = pre-v3 index (no hashes stored); dict = v3/v5 index with per-file SHA256 digests.
    file_shas: dict[str, str] | None = field(default=None)
    # v4+: maps caller FQN → {pattern: count} for unresolved static-dispatch calls.
    # Empty dict on v4 indexes with zero misses. Present only in v4+; absent from
    # older indexes (which are rejected by load()).
    missed_callers: dict[str, dict[str, int]] = field(default_factory=dict)
    # v5+: git SHA of the repo at build time (None when build runs outside a git checkout).
    git_sha: str | None = field(default=None)
    # v5+: SHA-256 hex digest of the deterministically serialised raw dict.
    # Computed at from_raw time; persisted in the index header.
    content_hash: str = field(default="")
    # Derived at load/from_nodes time: maps FQN → relative file path (inverted from skeletons).
    # Not serialised — rebuilt on every load.
    _fqn_to_file: dict[str, str] = field(default_factory=dict, repr=False)
    # Computed at from_nodes time: p99 of in-degree distribution, floor of 10.
    # Used by neighborhood() for hub suppression. Not serialised — recomputed on load.
    _in_degree_threshold: int = field(default=10, repr=False)
    # Thin facade over nodes dict for all traversal — never serialised.
    _reader: GraphReader = field(default_factory=lambda: GraphReader({}), repr=False)
    # Module-level index: module_fqn → set of symbol FQNs.  Built once from nodes keys.
    # Replaces module_graph for module-level traversal. Not serialised.
    _module_index: dict[str, set[str]] = field(default_factory=dict, repr=False)

    @property
    def raw(self) -> dict[str, list[str]]:
        """Legacy projection of ``nodes`` to ``{caller: [callees]}`` (call edges only).

        Provided as a transitional read-only view for callers that have not
        migrated to the ``nodes`` shape yet (Child #3 of epic #76 migrates
        the remaining test fixtures).  New code should consume ``nodes``
        directly to keep access to non-call edge kinds.
        """
        return _nodes_to_raw(self.nodes)

    @classmethod
    def from_nodes(
        cls,
        root: str | Path,
        nodes: dict[str, dict],
        skeletons: dict[str, list[SymbolSummary]] | None = None,
        file_shas: dict[str, str] | None = None,
        missed_callers: dict[str, dict[str, int]] | None = None,
        git_sha: str | None = None,
    ) -> "CallGraphIndex":
        """Construct a CallGraphIndex from the site-keyed ``nodes`` shape.

        Builds a :class:`GraphReader` over *nodes* for all traversal and a
        ``module_index`` for module-level queries.  The O(E) ``_DiGraph``
        rebuild step that previously populated ``function_graph`` and
        ``module_graph`` is gone — traversal is now O(1) per lookup via the
        reader.
        """
        root = Path(root).resolve()
        reader = GraphReader(nodes)

        # Build module_index: module_fqn → set of symbol FQNs (O(N) over keys).
        module_index: dict[str, set[str]] = {}
        for sym in nodes:
            mod = _module_of(sym)
            module_index.setdefault(mod, set()).add(sym)

        skeletons = skeletons or {}
        # Invert skeletons → _fqn_to_file: {fqn: rel_path}
        fqn_to_file: dict[str, str] = {}
        for rel_path, symbols in skeletons.items():
            for sym in symbols:
                fqn_to_file[sym["fqn"]] = rel_path
        # Compute in-degree threshold: p99 of in-degree distribution, floor of 10.
        in_degree_threshold = _compute_in_degree_threshold(reader)
        # Compute content_hash from the canonical projection of call edges.
        content_hash = _compute_content_hash(nodes)
        return cls(
            root=root,
            nodes=nodes,
            skeletons=skeletons,
            file_shas=file_shas,
            missed_callers=missed_callers if missed_callers is not None else {},
            git_sha=git_sha,
            content_hash=content_hash,
            _fqn_to_file=fqn_to_file,
            _in_degree_threshold=in_degree_threshold,
            _reader=reader,
            _module_index=module_index,
        )

    @classmethod
    def from_raw(
        cls,
        root: str | Path,
        raw: dict[str, list[str]],
        skeletons: dict[str, list[SymbolSummary]] | None = None,
        file_shas: dict[str, str] | None = None,
        missed_callers: dict[str, dict[str, int]] | None = None,
        git_sha: str | None = None,
    ) -> "CallGraphIndex":
        """Deprecated — convert legacy ``raw`` shape to ``nodes`` and delegate.

        Retained for transitional callers (notably the test corpus that Child #3
        of epic #76 will migrate).  New code should call :meth:`from_nodes`
        directly so non-call edge kinds can be expressed.
        """
        nodes = _raw_to_nodes(raw)
        return cls.from_nodes(
            root,
            nodes,
            skeletons=skeletons,
            file_shas=file_shas,
            missed_callers=missed_callers,
            git_sha=git_sha,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic serialisation: sort node keys, sort kind keys within
        # each calls/called_by bucket, sort each kind's list.  json.dumps with
        # sort_keys=True handles the dict layers; we pre-sort the lists.
        canonical_nodes: dict[str, dict] = {}
        for sym in sorted(self.nodes):
            record = self.nodes[sym]
            calls_buckets = record.get("calls", {}) or {}
            called_by_buckets = record.get("called_by", {}) or {}
            canonical_nodes[sym] = {
                "calls": {
                    kind: sorted(calls_buckets[kind])
                    for kind in sorted(calls_buckets)
                },
                "called_by": {
                    kind: sorted(called_by_buckets[kind])
                    for kind in sorted(called_by_buckets)
                },
            }
        payload: dict = {
            "version": INDEX_VERSION,
            "root": str(self.root),
            "nodes": canonical_nodes,
            "skeletons": self.skeletons,
            "file_shas": self.file_shas if self.file_shas is not None else {},
            "missed_callers": self.missed_callers,
            "git_sha": self.git_sha,
            "content_hash": self.content_hash,
        }
        path.write_text(json.dumps(payload, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CallGraphIndex":
        path = Path(path)
        payload = json.loads(path.read_text())
        version = payload.get("version")
        if version != INDEX_VERSION:
            raise ValueError(
                f"index schema is v{version}, server requires v{INDEX_VERSION} — "
                "run 'pyscope-mcp build' to regenerate"
            )
        # Pre-migration v5 indexes wrote ``raw`` instead of ``nodes``.  Reject
        # them with the same clear-error contract used for older versions —
        # there is no migration shim (Corollary 1.3/4.3).
        if "nodes" not in payload:
            raise ValueError(
                f"index schema is v{version} but uses the legacy 'raw' field; "
                f"server requires v{INDEX_VERSION} with site-keyed 'nodes' — "
                "run 'pyscope-mcp build' to regenerate"
            )
        skeletons: dict[str, list[SymbolSummary]] = payload.get("skeletons", {})
        file_shas: dict[str, str] = payload.get("file_shas", {})
        missed_callers: dict[str, dict[str, int]] = payload.get("missed_callers", {})
        git_sha: str | None = payload.get("git_sha")
        return cls.from_nodes(
            Path(payload["root"]),
            payload["nodes"],
            skeletons=skeletons,
            file_shas=file_shas,
            missed_callers=missed_callers,
            git_sha=git_sha,
        )

    _STALE_ACTION = "Call the 'build' MCP tool to rebuild and reload the index."

    @staticmethod
    def _class_prefix(fqn: str) -> str | None:
        """Return the class-level FQN prefix for a method FQN, or None.

        A method FQN has ≥4 dotted segments (e.g. ``pkg.mod.MyClass.method``
        → ``pkg.mod.MyClass``).  Top-level functions have ≤3 segments
        (e.g. ``pkg.mod.func`` or ``pkg.func``) and return ``None`` —
        they never trip the class-prefix path.

        The "≥4 segments" rule derives from the minimum realistic Python FQN
        for a method: ``<package>.<module>.<Class>.<method>``.  Three-segment
        FQNs (e.g. ``pkg.mod.func``) are always top-level functions.

        Examples:
          ``pkg.mod.MyClass.method`` → ``pkg.mod.MyClass``  (4 segments)
          ``a.b.c.D.method``         → ``a.b.c.D``          (5 segments)
          ``pkg.mod.func``           → ``None``              (3 segments)
          ``pkg.func``               → ``None``              (2 segments)
        """
        parts = fqn.split(".")
        if len(parts) >= 4:
            return ".".join(parts[:-1])
        return None

    def completeness_for(self, fqns: list[str]) -> Completeness:
        """Return the completeness status for a set of traversed FQNs.

        Returns ``"partial"`` if any FQN in *fqns* is:
          - directly present as a key in ``self.missed_callers`` (direct hit), OR
          - a method of a class (≥3 dotted segments) whose class prefix (all but
            the last segment) is a prefix of any key in ``self.missed_callers``.
            E.g. ``a.b.C.bar`` is partial if ``a.b.C.foo`` is in missed_callers,
            because both share the class prefix ``a.b.C``.

        Top-level functions (≤2 dotted segments) NEVER trip the class-prefix path;
        they only return ``"partial"`` on a direct hit.

        Returns ``"complete"`` when ``missed_callers`` is empty or no FQN in
        *fqns* matches either condition.
        """
        if not self.missed_callers:
            return "complete"
        missed_keys = set(self.missed_callers.keys())
        for fqn in fqns:
            # Direct hit
            if fqn in missed_keys:
                return "partial"
            # Class-prefix hit (methods only)
            prefix = self._class_prefix(fqn)
            if prefix is not None:
                # Any missed_callers key that starts with "<prefix>." means a
                # sibling method in the same class has unresolved calls.
                prefix_dot = prefix + "."
                for key in missed_keys:
                    if key.startswith(prefix_dot):
                        return "partial"
        return "complete"

    def _staleness_for(self, fqns: list[str]) -> dict:
        """Compute result-scoped staleness for a list of FQNs.

        Returns a dict with uniform staleness shape:
          - ``{"stale": False, "stale_files": []}`` when all backing files are clean.
          - ``{"stale": True, "stale_files": [...], "stale_action": ...}`` when any
            file backing a result FQN has changed since the last build.
          - ``{"stale": True, "stale_files": [], "index_stale_reason": "index_format_incompatible",
            "stale_action": ...}`` for pre-v3 indexes (file_shas is None).

        Only files that back the provided FQNs are checked (result-scoped).
        Files in the index that do not appear in the result set are ignored.
        """
        if self.file_shas is None:
            return {
                "stale": True,
                "stale_files": [],
                "index_stale_reason": "index_format_incompatible",
                "stale_action": self._STALE_ACTION,
            }

        # Collect the unique file paths that back the provided FQNs.
        result_files: set[str] = set()
        for fqn in fqns:
            rel = self._fqn_to_file.get(fqn)
            if rel is not None:
                result_files.add(rel)

        stale_files: list[str] = []
        for rel in sorted(result_files):
            stored_sha = self.file_shas.get(rel)
            live_file = self.root / rel
            if not live_file.exists():
                stale_files.append(rel)
                continue
            live_sha = hashlib.sha256(live_file.read_bytes()).hexdigest()
            if stored_sha is None or live_sha != stored_sha:
                stale_files.append(rel)

        if stale_files:
            return {
                "stale": True,
                "stale_files": stale_files,
                "stale_action": self._STALE_ACTION,
            }
        return {"stale": False, "stale_files": []}

    def _commit_staleness(self) -> dict:
        """Return commit-level staleness fields for the loaded index.

        Compares ``self.git_sha`` (captured at build time) against the current
        HEAD of the repo at ``self.root`` via ``git rev-parse HEAD``.

        Returns a dict with three fields — always present, never omitted:
          - ``commit_stale: bool | None`` — True when HEAD has advanced past the
            index's build commit; None when the comparison is impossible.
          - ``index_git_sha: str | None`` — git SHA captured at build time
            (None when the index was built outside a git checkout).
          - ``head_git_sha: str | None`` — current HEAD SHA of the repo
            (None when git is unavailable or HEAD cannot be resolved).

        When the index was built outside a git checkout (``self.git_sha`` is
        None) or git is unavailable, all three fields are None.
        """
        index_sha = self.git_sha
        head_sha: str | None = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                head_sha = result.stdout.strip() or None
        except FileNotFoundError:
            pass  # git binary not installed

        if index_sha is None or head_sha is None:
            return {
                "commit_stale": None,
                "index_git_sha": None,
                "head_git_sha": None,
            }
        return {
            "commit_stale": head_sha != index_sha,
            "index_git_sha": index_sha,
            "head_git_sha": head_sha,
        }

    def file_skeleton(self, path: str, cap: int = _SKELETON_CAP) -> dict:
        """Return a compact symbol list for the given relative file path.

        Returns a dict with:
          - ``results``: list of SymbolSummary dicts sorted by lineno (capped at *cap*)
          - ``truncated``: True when the full symbol count exceeds *cap*
          - ``total``: total number of symbols before capping
          - ``stale``: True when the live file differs from what was indexed
          - ``stale_files``: list of stale relative paths (the queried file, or [])
          - ``stale_action``: guidance on how to fix staleness (present only when stale=True)
          - ``index_stale_reason``: ``"index_format_incompatible"`` for pre-v3 indexes

        Results are always returned when the path is in the index, even when stale.
        If the path is not in the index, returns an error dict with ``isError: True``.
        """
        commit = self._commit_staleness()

        # Scenario D / isError: path not in skeletons
        if path not in self.skeletons:
            result: dict = {
                "isError": True,
                "error_reason": "path_not_in_index",
                "stale": False,
                "stale_files": [],
                **commit,
            }
            return result

        symbols = self.skeletons[path]
        total = len(symbols)
        truncated = total > cap
        base_result: dict = {
            "results": symbols[:cap],
            "truncated": truncated,
            "total": total,
        }

        # Scenario E: pre-v3 index — no hashes available.
        if self.file_shas is None:
            base_result["stale"] = True
            base_result["stale_files"] = []
            base_result["index_stale_reason"] = "index_format_incompatible"
            base_result["stale_action"] = self._STALE_ACTION
            base_result.update(commit)
            return base_result

        # Scenarios A/B/C: v3 index — compare live file hash against stored hash.
        live_file = self.root / path
        if not live_file.exists():
            # Scenario C: file deleted since build.
            base_result["stale"] = True
            base_result["stale_files"] = [path]
            base_result["stale_action"] = self._STALE_ACTION
            base_result.update(commit)
            return base_result

        stored_sha = self.file_shas.get(path)
        live_sha = hashlib.sha256(live_file.read_bytes()).hexdigest()

        if stored_sha is None or live_sha != stored_sha:
            # Scenario B: file changed.
            base_result["stale"] = True
            base_result["stale_files"] = [path]
            base_result["stale_action"] = self._STALE_ACTION
            base_result.update(commit)
            return base_result

        # Scenario A: fresh — hashes match.
        base_result["stale"] = False
        base_result["stale_files"] = []
        base_result.update(commit)
        return base_result

    _CALLERS_CALLEES_CAP = 50

    def _rank_bfs_results(
        self,
        bfs_result: dict[str, tuple[int, dict[str, int]]],
        reader: "GraphReader",
    ) -> list[str]:
        """Rank BFS result nodes by (hop_depth ASC, -total_degree DESC, fqn ASC).

        ``total_degree`` = in-degree + out-degree of the result node in the
        underlying function graph (call edges only — proxy for importance).

        ``reader.out_degree`` reads from ``calls`` and ``reader.in_degree``
        reads from ``called_by`` — no direction flag needed.
        """

        def total_degree(n: str) -> int:
            return reader.out_degree(n, kinds=("call",)) + reader.in_degree(
                n, kinds=("call",)
            )

        degree_cache: dict[str, int] = {n: total_degree(n) for n in bfs_result}
        return sorted(
            bfs_result,
            key=lambda n: (bfs_result[n][0], -degree_cache[n], n),
        )

    def refers_to(
        self,
        fqn: str,
        kind: str = "all",
        granularity: str = "function",
        depth: int = 1,
    ) -> ReferencedByResult:
        """Return all typed AST references to *fqn* across the codebase.

        Replaces the deleted ``callers_of`` (call-only) and ``module_callers``
        (module-level call-only).  Covers all reference kinds stored in
        ``nodes[fqn]["called_by"]``: ``"call"``, ``"import"``, ``"except"``,
        ``"annotation"``, ``"isinstance"``.

        Parameters
        ----------
        fqn:
            Fully-qualified name of the symbol to look up.
        kind:
            ``"all"`` — all reference kinds (default, for refactor safety).
            ``"callers"`` — call edges only (replaces ``callers_of``).
        granularity:
            ``"function"`` — results are ``{fqn, context, depth}`` dicts
            (default).  Each function appears once; ``"call"`` context wins
            when a function references the symbol in multiple ways.
            ``"module"`` — flat deduplicated list of module FQN strings
            (replaces ``module_callers`` when used with ``kind="callers"``).
        depth:
            BFS hop depth (1 or 2).  Depth > 2 returns an error.  Depth > 1
            traverses transitively through the ``"call"`` edges in the reverse
            function graph (same BFS as the deleted ``callers_of``) for
            ``kind="callers"``, and through the nodes dict for ``kind="all"``.

        Returns
        -------
        :class:`ReferencedByResult` with ``results``, ``truncated``,
        ``dropped``, ``completeness``, and staleness fields.

        Error shapes
        ------------
        - ``fqn`` not in ``nodes``: ``{isError: true, error_reason:
          "fqn_not_in_graph", stale: false, stale_files: []}``
        - ``depth > 2``: ``{isError: true, error_reason: "depth_exceeds_max"}``
        - FQN present, zero references: ``{results: [], ...}`` — not an error.

        NOTE: Wildcard imports (``from module import *``) are not tracked and
        are silently absent from results. See issue #74.
        """
        if depth > 2:
            return {  # type: ignore[return-value]
                "isError": True,
                "error_reason": "depth_exceeds_max",
            }

        if fqn not in self.nodes:
            commit = self._commit_staleness()
            return {  # type: ignore[return-value]
                "isError": True,
                "error_reason": "fqn_not_in_graph",
                "stale": False,
                "stale_files": [],
                **commit,
            }

        if kind == "callers":
            # BFS over called_by["call"] edges only (call-edge traversal).
            bfs_result = self._reader.bfs(fqn, depth, direction="called_by", kinds=("call",))
            ranked_fqns = self._rank_bfs_results(bfs_result, self._reader)
            dropped = max(0, len(ranked_fqns) - self._CALLERS_CALLEES_CAP)
            truncated = dropped > 0
            ranked_fqns = ranked_fqns[: self._CALLERS_CALLEES_CAP]

            if granularity == "module":
                module_results = _dedup_modules(ranked_fqns)
                staleness = self._staleness_for_modules(module_results)
                symbol_fqns = self._expand_modules_to_symbols(module_results)
                commit = self._commit_staleness()
                return ReferencedByResult(
                    results=module_results,
                    truncated=truncated,
                    dropped=dropped,
                    completeness=self.completeness_for(symbol_fqns),
                    **staleness,  # type: ignore[arg-type]
                    **commit,  # type: ignore[arg-type]
                )

            # granularity == "function"
            entries: list[ReferencedByEntry] = [
                ReferencedByEntry(fqn=f, context="call", depth=bfs_result[f][0])
                for f in ranked_fqns
            ]
            staleness = self._staleness_for(ranked_fqns)
            commit = self._commit_staleness()
            return ReferencedByResult(
                results=entries,
                truncated=truncated,
                dropped=dropped,
                completeness=self.completeness_for(ranked_fqns),
                **staleness,  # type: ignore[arg-type]
                **commit,  # type: ignore[arg-type]
            )

        # kind == "all" — walk nodes["called_by"] for all edge kinds.
        # Context priority: "call" > all other kinds (first-encountered non-call).
        _CONTEXT_PRIORITY = ("call", "import", "except", "annotation", "isinstance")
        bfs_all = self._reader.bfs(fqn, depth, direction="called_by", kinds=None)
        # bfs_all: dict[referencing_fqn -> (min_hop, {kind: hop_at_which_seen})]
        # Rank by (hop ASC, -total_degree DESC, fqn ASC).
        # total_degree via reader (call edges only — proxy for importance).

        def _total_degree_all(n: str) -> int:
            return self._reader.out_degree(n, kinds=("call",)) + self._reader.in_degree(
                n, kinds=("call",)
            )

        sorted_fqns = sorted(
            bfs_all,
            key=lambda n: (bfs_all[n][0], -_total_degree_all(n), n),
        )
        dropped = max(0, len(sorted_fqns) - self._CALLERS_CALLEES_CAP)
        truncated = dropped > 0
        sorted_fqns = sorted_fqns[: self._CALLERS_CALLEES_CAP]

        if granularity == "module":
            module_results = _dedup_modules(sorted_fqns)
            staleness = self._staleness_for_modules(module_results)
            symbol_fqns = self._expand_modules_to_symbols(module_results)
            commit = self._commit_staleness()
            return ReferencedByResult(
                results=module_results,
                truncated=truncated,
                dropped=dropped,
                completeness=self.completeness_for(symbol_fqns),
                **staleness,  # type: ignore[arg-type]
                **commit,  # type: ignore[arg-type]
            )

        # granularity == "function" — pick highest-priority context per FQN.
        entries = []
        for ref_fqn in sorted_fqns:
            hop, kinds_seen = bfs_all[ref_fqn]
            chosen_context = "import"  # fallback if no call kind present
            for ctx in _CONTEXT_PRIORITY:
                if ctx in kinds_seen:
                    chosen_context = ctx
                    break
            entries.append(ReferencedByEntry(fqn=ref_fqn, context=chosen_context, depth=hop))

        staleness = self._staleness_for(sorted_fqns)
        commit = self._commit_staleness()
        return ReferencedByResult(
            results=entries,
            truncated=truncated,
            dropped=dropped,
            completeness=self.completeness_for(sorted_fqns),
            **staleness,  # type: ignore[arg-type]
            **commit,  # type: ignore[arg-type]
        )

    def callees_of(self, fqn: str, depth: int = 1) -> CalleesResult:
        """Return functions (transitively, up to depth) called by *fqn*.

        Results are capped at 50; ``truncated`` signals when the cap fires.
        ``dropped`` is the number of results cut by the cap (always present; 0
        when the cap does not fire).  Results are ranked by
        ``(hop_depth ASC, -total_degree DESC, fqn ASC)`` so depth-1 callees
        always precede depth-2 ones regardless of alphabetical order.
        Response includes uniform staleness fields (``stale``, ``stale_files``,
        and optionally ``stale_action`` / ``index_stale_reason``) and a
        ``completeness`` field (``"complete"`` or ``"partial"``).

        If *fqn* is not present in the graph at all, returns an error dict
        with ``isError: True``, ``error_reason: "fqn_not_in_graph"``, and
        ``stale: False``.  An FQN that is present but has zero callees returns
        ``results: []`` (not an error).
        """
        if fqn not in self._reader:
            commit = self._commit_staleness()
            return {  # type: ignore[return-value]
                "isError": True,
                "error_reason": "fqn_not_in_graph",
                "stale": False,
                "stale_files": [],
                **commit,
            }
        bfs_result = self._reader.bfs(fqn, depth, direction="calls", kinds=("call",))
        ranked = self._rank_bfs_results(bfs_result, self._reader)
        dropped = max(0, len(ranked) - self._CALLERS_CALLEES_CAP)
        truncated = dropped > 0
        results = ranked[: self._CALLERS_CALLEES_CAP]
        staleness = self._staleness_for(results)
        commit = self._commit_staleness()
        return CalleesResult(
            results=results,
            truncated=truncated,
            dropped=dropped,
            completeness=self.completeness_for(results),
            **staleness,  # type: ignore[arg-type]
            **commit,  # type: ignore[arg-type]
        )

    def neighborhood(
        self,
        symbol: str,
        depth: int = 2,
        token_budget: int = 1000,
        expand_hubs: bool = False,
        hub_threshold: int | None = None,
    ) -> NeighborhoodResult:
        """Return a bounded bidirectional subgraph around *symbol*.

        BFS outward (both callers and callees) up to *depth* hops.  Candidate
        edges are ranked by (hop_depth, -degree) — depth-first with degree
        tiebreak — then truncated to fit within *token_budget* (4 chars/token
        estimate).

        Hub suppression (default on):
          Nodes whose in-degree exceeds *hub_threshold* (default: index-load-time
          p99 of in-degree distribution, floor 10) are treated as utility hubs.
          Their direct edges to/from the queried symbol are kept, but BFS does
          not expand *through* them in the callers direction (in-degree only —
          out-degree hubs are pipeline siblings and are expanded normally).
          The queried *symbol* itself is exempt — calling neighborhood directly
          on a hub still returns its full neighborhood.

          Pass ``expand_hubs=True`` to disable suppression.  Pass a non-None
          ``hub_threshold`` to override the cached threshold for this query.

        Response keys:
          - ``symbol``: the queried FQN
          - ``depth_full``: deepest level with all edges intact (no drops)
          - ``depth_truncated``: first level where dropping started (only when ``truncated=True``)
          - ``edges``: list of ``[caller, callee]`` pairs
          - ``truncated``: True when the budget was hit
          - ``token_budget_used``: estimated tokens consumed (chars / 4)
          - ``hub_suppressed``: list of FQNs whose caller expansion was skipped (always present)
          - ``hub_threshold``: the in-degree threshold applied (always present)

        ``depth_full`` reflects the actual graph depth, not the declared *depth*
        parameter (if the graph is shallower, ``depth_full`` mirrors reality).
        """
        # Guard: FQN not in graph at all → clear not-found error (not ambiguous empty-edges)
        if symbol not in self._reader:
            commit = self._commit_staleness()
            return {  # type: ignore[return-value]
                "isError": True,
                "error_reason": "fqn_not_in_graph",
                "stale": False,
                "stale_files": [],
                **commit,
            }

        # Resolve effective threshold for this query
        effective_threshold: int = (
            self._in_degree_threshold if hub_threshold is None else hub_threshold
        )

        # ------------------------------------------------------------------
        # Phase 1: bidirectional BFS — collect (caller, callee, hop_depth)
        # ------------------------------------------------------------------
        # We collect all edges reachable within `depth` hops from symbol in
        # either direction.  Each edge is annotated with the minimum hop_depth
        # at which either endpoint was first encountered from symbol.

        # BFS state: maps node → min hop distance from symbol
        node_depth: dict[str, int] = {symbol: 0}
        frontier: deque[tuple[str, int]] = deque([(symbol, 0)])

        # Edge deduplication and depth tracking: maps (caller, callee) → min hop_depth
        edge_depths: dict[tuple[str, str], int] = {}  # min hop_depth for edge

        # Collect hub nodes whose expansion was suppressed (in-degree only)
        suppressed_hubs: set[str] = set()

        while frontier:
            node, d = frontier.popleft()
            if d >= depth:
                continue
            next_d = d + 1

            # Callees direction: node → callee (out-degree; always expand)
            for callee in self._reader.successors(node, kinds=("call",)):
                edge = (node, callee)
                if edge not in edge_depths or edge_depths[edge] > next_d:
                    edge_depths[edge] = next_d
                if callee not in node_depth:
                    node_depth[callee] = next_d
                    frontier.append((callee, next_d))

            # Callers direction: caller → node (in-degree; hub suppression applies)
            # A node is treated as a hub if its in-degree exceeds the threshold AND
            # it is not the queried symbol itself (the queried symbol is always exempt).
            node_in_degree = self._reader.in_degree(node, kinds=("call",))
            is_hub = (
                not expand_hubs
                and node != symbol
                and node_in_degree >= effective_threshold
            )
            if is_hub:
                suppressed_hubs.add(node)
                # Do not expand through hub callers — skip adding them to frontier.
                # The edge from node's own callers back to node is still collected
                # if node was reached as a direct neighbor; but we don't traverse
                # further through the hub.
                continue

            for caller in self._reader.predecessors(node, kinds=("call",)):
                edge = (caller, node)
                if edge not in edge_depths or edge_depths[edge] > next_d:
                    edge_depths[edge] = next_d
                if caller not in node_depth:
                    node_depth[caller] = next_d
                    frontier.append((caller, next_d))

        if not edge_depths:
            # Symbol not in graph or isolated node — return minimal result
            staleness = self._staleness_for([symbol])
            commit = self._commit_staleness()
            result = NeighborhoodResult(
                symbol=symbol,
                depth_full=0,
                edges=[],
                truncated=False,
                token_budget_used=0,
                completeness=self.completeness_for([symbol]),
                hub_suppressed=[],
                hub_threshold=effective_threshold,
            )
            result.update(staleness)  # type: ignore[arg-type]
            result.update(commit)  # type: ignore[arg-type]
            return result

        # ------------------------------------------------------------------
        # Phase 2: rank edges deterministically
        # Ranking key: (hop_depth ASC, -degree DESC, caller ASC, callee ASC)
        # degree = out-degree + in-degree of the caller node in the full graph
        # ------------------------------------------------------------------
        # Precompute degrees for all unique caller nodes to avoid O(E×degree)
        # redundant traversals during sort.
        caller_nodes = {e[0] for e in edge_depths}
        degree_cache: dict[str, int] = {
            n: self._reader.out_degree(n, kinds=("call",))
            + self._reader.in_degree(n, kinds=("call",))
            for n in caller_nodes
        }

        ranked_edges = sorted(
            edge_depths.keys(),
            key=lambda e: (edge_depths[e], -degree_cache.get(e[0], 0), e[0], e[1]),
        )

        # ------------------------------------------------------------------
        # Phase 3: truncate to token_budget (4 chars/token estimate)
        # ------------------------------------------------------------------
        char_budget = token_budget * 4
        kept_edges: list[tuple[str, str]] = []
        chars_used = 0
        truncated = False
        depth_truncated: int | None = None

        # Track per-depth completeness
        depth_complete_through = 0  # deepest level with ALL edges kept
        depth_edges_by_level: dict[int, list[tuple[str, str]]] = {}
        for edge in ranked_edges:
            d = edge_depths[edge]
            depth_edges_by_level.setdefault(d, []).append(edge)

        for level in sorted(depth_edges_by_level):
            level_edges = depth_edges_by_level[level]
            for edge in level_edges:
                edge_repr = json.dumps([edge[0], edge[1]])
                edge_chars = len(edge_repr) + 2  # +2 for comma + newline approx
                if chars_used + edge_chars > char_budget:
                    truncated = True
                    if depth_truncated is None:
                        depth_truncated = level
                    # Skip this edge — budget exhausted
                    continue
                kept_edges.append(edge)
                chars_used += edge_chars
            if not truncated:
                depth_complete_through = level

        # depth_full = deepest level where ALL edges were kept intact
        depth_full = depth_complete_through

        # ------------------------------------------------------------------
        # Phase 4: assemble result
        # ------------------------------------------------------------------
        # Collect unique FQNs from kept edges for result-scoped staleness check.
        unique_fqns: list[str] = list(
            {fqn for edge in kept_edges for fqn in edge}
        )
        staleness = self._staleness_for(unique_fqns)
        commit = self._commit_staleness()

        result = NeighborhoodResult(
            symbol=symbol,
            depth_full=depth_full,
            edges=[[c, e] for c, e in kept_edges],
            truncated=truncated,
            token_budget_used=chars_used // 4,
            completeness=self.completeness_for(unique_fqns),
            hub_suppressed=sorted(suppressed_hubs),
            hub_threshold=effective_threshold,
        )
        result.update(staleness)  # type: ignore[arg-type]
        result.update(commit)  # type: ignore[arg-type]
        if truncated and depth_truncated is not None:
            result["depth_truncated"] = depth_truncated

        return result

    def _rank_module_bfs_results(
        self,
        bfs_result: dict[str, int],
        reader: "GraphReader",
    ) -> list[str]:
        """Rank module BFS result nodes by (hop_depth ASC, -total_degree DESC, fqn ASC).

        ``total_degree`` = in-degree + out-degree of the result module, computed
        by counting how many distinct modules appear as callers/callees across all
        symbols in the module.  Uses the same ranking convention as
        :meth:`neighborhood`.
        """

        def total_degree(mod: str) -> int:
            # Out-degree: distinct callee modules reached from this module's symbols.
            callee_mods: set[str] = set()
            for sym in self._module_index.get(mod, ()):
                for callee in reader.successors(sym, kinds=("call",)):
                    callee_mods.add(_module_of(callee))
            callee_mods.discard(mod)
            # In-degree: distinct caller modules that reach into this module's symbols.
            caller_mods: set[str] = set()
            for sym in self._module_index.get(mod, ()):
                for caller in reader.predecessors(sym, kinds=("call",)):
                    caller_mods.add(_module_of(caller))
            caller_mods.discard(mod)
            return len(callee_mods) + len(caller_mods)

        degree_cache: dict[str, int] = {n: total_degree(n) for n in bfs_result}
        return sorted(
            bfs_result,
            key=lambda n: (bfs_result[n], -degree_cache[n], n),
        )

    def module_callees(self, module: str, depth: int = 1) -> ModuleResult:
        """Return callees of all modules whose FQN starts with *module* (prefix query).

        Symmetric to :meth:`module_callers` — see its docstring for semantics.
        ``dropped`` is the number of results cut by the cap (always present; 0
        when the cap does not fire).  Results are ranked by
        ``(hop_depth ASC, -total_degree DESC, fqn ASC)`` so depth-1 module
        callees always precede depth-2 ones regardless of alphabetical order.
        Completeness is computed over the expanded symbol FQNs of the result modules.
        """
        base = _prefix_module_bfs(self._reader, self._module_index, module, depth)
        raw_union: dict[str, int] = base["results"]  # type: ignore[assignment]
        ranked = self._rank_module_bfs_results(raw_union, self._reader)
        cap = _MODULE_BFS_CAP
        dropped: int = base["dropped"]  # type: ignore[assignment]
        result_modules = ranked[:cap]
        staleness = self._staleness_for_modules(result_modules)
        symbol_fqns = self._expand_modules_to_symbols(result_modules)
        commit = self._commit_staleness()
        return ModuleResult(
            results=result_modules,
            truncated=base["truncated"],  # type: ignore[typeddict-item]
            dropped=dropped,
            completeness=self.completeness_for(symbol_fqns),
            **staleness,  # type: ignore[arg-type]
            **commit,  # type: ignore[arg-type]
        )

    def _expand_modules_to_symbols(self, module_fqns: list[str]) -> list[str]:
        """Expand module FQNs to the symbol FQNs they contain.

        Module FQNs (e.g. ``pkg.mod``) are not directly stored in ``_fqn_to_file``;
        only function/class/method FQNs are.  Expansion finds all ``_fqn_to_file``
        keys that start with ``module_fqn + "."``.
        """
        symbol_fqns: list[str] = []
        for mod in module_fqns:
            if not isinstance(mod, str):
                raise TypeError(
                    f"_expand_modules_to_symbols expects str elements, "
                    f"got {type(mod)!r}: {mod!r}"
                )
            prefix = mod + "."
            for fqn in self._fqn_to_file:
                if fqn.startswith(prefix):
                    symbol_fqns.append(fqn)
        return symbol_fqns

    def _staleness_for_modules(self, module_fqns: list[str]) -> dict:
        """Expand module FQNs to symbol FQNs and delegate to ``_staleness_for``.

        Module FQNs (e.g. ``pkg.mod``) are not stored in ``_fqn_to_file``; only
        function/class/method FQNs are.  We expand by finding all ``_fqn_to_file``
        keys that start with ``module_fqn + "."``.
        """
        if self.file_shas is None:
            return {
                "stale": True,
                "stale_files": [],
                "index_stale_reason": "index_format_incompatible",
                "stale_action": self._STALE_ACTION,
            }
        return self._staleness_for(self._expand_modules_to_symbols(module_fqns))

    def search(self, substring: str, limit: int = 50) -> SearchResult:
        """Substring search over known fully-qualified function names.

        Returns a dict with:
          - ``results``: list of matching FQNs (capped at *limit*)
          - ``truncated``: True when the full match count exceeds *limit*
          - ``total_matched``: count of all matches before capping
          - ``stale``, ``stale_files``: result-scoped staleness info
        """
        s = substring.lower()
        all_matches = [n for n in self._reader.nodes_iter() if s in n.lower()]
        total = len(all_matches)
        results = all_matches[:limit]
        staleness = self._staleness_for(results)
        commit = self._commit_staleness()
        return SearchResult(
            results=results,
            truncated=total > limit,
            total_matched=total,
            **staleness,  # type: ignore[arg-type]
            **commit,  # type: ignore[arg-type]
        )

    def stats(self) -> StatsResult:
        commit = self._commit_staleness()
        num_modules = len(self._module_index)
        # Count module-level edges: distinct (src_module, dst_module) pairs from call edges.
        module_edge_set: set[tuple[str, str]] = set()
        for sym in self._reader.nodes_iter():
            src_mod = _module_of(sym)
            for callee in self._reader.successors(sym, kinds=("call",)):
                dst_mod = _module_of(callee)
                if src_mod != dst_mod:
                    module_edge_set.add((src_mod, dst_mod))
        return StatsResult(
            functions=self._reader.num_nodes(),
            function_edges=self._reader.num_edges(kind="call"),
            modules=num_modules,
            module_edges=len(module_edge_set),
            **commit,  # type: ignore[arg-type]
        )


_MODULE_BFS_CAP = 50

_IN_DEGREE_THRESHOLD_FLOOR = 10


def _compute_in_degree_threshold(reader: GraphReader) -> int:
    """Compute the in-degree hub-suppression threshold from *reader*.

    Returns the p99 of the in-degree distribution across all nodes in the
    function graph (call edges only), with a floor of
    ``_IN_DEGREE_THRESHOLD_FLOOR`` (10).

    If the graph has fewer than 2 nodes, returns the floor.

    This is called once during ``CallGraphIndex.from_nodes`` and the result is
    cached on the index instance as ``_in_degree_threshold``.  Never called
    per-query — see Law 2.
    """
    if reader.num_nodes() < 2:
        return _IN_DEGREE_THRESHOLD_FLOOR
    in_degrees = [reader.in_degree(n, kinds=("call",)) for n in reader.nodes_iter()]
    in_degrees.sort()
    n = len(in_degrees)
    # p99 index: the value at the 99th percentile (upper inclusive)
    p99_idx = int(0.99 * n)
    if p99_idx >= n:
        p99_idx = n - 1
    p99 = in_degrees[p99_idx]
    return max(_IN_DEGREE_THRESHOLD_FLOOR, p99)


def _module_of(fqn: str) -> str:
    return fqn.rsplit(".", 1)[0] if "." in fqn else fqn


def _prefix_module_bfs(
    reader: GraphReader,
    module_index: dict[str, set[str]],
    prefix: str,
    depth: int,
    cap: int = _MODULE_BFS_CAP,
) -> dict[str, object]:
    """BFS over the function graph for modules whose FQN starts with *prefix*.

    Uses *reader* for function-level traversal and *module_index* to map
    symbols to their parent modules.  Results from each matched seed module
    are unioned and deduplicated.  The matched seed modules themselves are
    excluded from the result set (same semantics as the former _bfs helper).
    Results are capped at *cap*; ``truncated`` reflects whether the full union
    exceeded the cap.  ``dropped`` is the number of results cut by the cap.

    When a result module is reachable from multiple seeds, its hop depth is
    the minimum across all seeds (closest path wins).

    ``prefix=""`` matches all module nodes.
    """
    # Collect all module nodes that start with the given prefix
    matched_seeds = [mod for mod in module_index if mod.startswith(prefix)]

    seed_set = set(matched_seeds)
    union: dict[str, int] = {}  # module_fqn -> min hop depth

    for seed_mod in matched_seeds:
        # Do a function-level BFS from all symbols in this seed module.
        for sym in module_index.get(seed_mod, ()):
            bfs_result = reader.bfs(sym, depth, direction="calls", kinds=("call",))
            for reachable_fqn, (hop, _) in bfs_result.items():
                reachable_mod = _module_of(reachable_fqn)
                if reachable_mod not in seed_set:
                    if reachable_mod not in union or hop < union[reachable_mod]:
                        union[reachable_mod] = hop

    dropped = max(0, len(union) - cap)
    truncated = dropped > 0
    return {
        "results": union,  # raw dict returned; callers apply ranking + slice
        "truncated": truncated,
        "dropped": dropped,
    }


def _dedup_modules(fqns: list[str]) -> list[str]:
    """Deduplicate function FQNs to their module FQNs (preserving order)."""
    seen: set[str] = set()
    result: list[str] = []
    for fqn in fqns:
        mod = _module_of(fqn)
        if mod not in seen:
            seen.add(mod)
            result.append(mod)
    return result
