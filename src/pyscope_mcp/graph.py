from __future__ import annotations

import hashlib
import json
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

INDEX_VERSION = 5

from pyscope_mcp.types import (
    CalleesResult,
    CallersResult,
    Completeness,
    ModuleResult,
    NeighborhoodResult,
    SearchResult,
    StatsResult,
)


class SymbolSummary(TypedDict):
    fqn: str
    kind: str  # "function" | "class" | "method"
    signature: str
    lineno: int


_SKELETON_CAP = 50


class _DiGraph:
    """Minimal directed-graph backed by plain dicts.

    Replaces networkx.DiGraph — only the operations actually used by
    CallGraphIndex are implemented.  Cold-start impact: ~7 s saved.
    """

    def __init__(self) -> None:
        self._succ: dict[str, set[str]] = {}
        self._pred: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ nodes
    def add_node(self, n: str) -> None:
        self._succ.setdefault(n, set())
        self._pred.setdefault(n, set())

    # ------------------------------------------------------------------ edges
    def add_edge(self, u: str, v: str) -> None:
        self.add_node(u)
        self.add_node(v)
        self._succ[u].add(v)
        self._pred[v].add(u)

    # ---------------------------------------------------------------- queries
    def successors(self, n: str):  # type: ignore[return]
        return iter(self._succ.get(n, ()))

    def predecessors(self, n: str):  # type: ignore[return]
        return iter(self._pred.get(n, ()))

    def __contains__(self, n: object) -> bool:
        return n in self._succ

    @property
    def nodes(self):  # type: ignore[return]
        return self._succ.keys()

    def number_of_nodes(self) -> int:
        return len(self._succ)

    def number_of_edges(self) -> int:
        return sum(len(v) for v in self._succ.values())

    # ----------------------------------------------------------------- views
    def reverse(self, copy: bool = False) -> "_DiGraphReverseView":
        """Return a view that swaps successor/predecessor lookup (no copy).

        ``copy=True`` is not supported — this implementation only holds a live
        view.  Passing ``copy=True`` raises ``NotImplementedError``.
        """
        if copy:
            raise NotImplementedError("_DiGraph.reverse(copy=True) is not supported")
        return _DiGraphReverseView(self)

    # --------------------------------------------------------------- display
    def __repr__(self) -> str:
        return (
            f"_DiGraph(nodes={self.number_of_nodes()}, edges={self.number_of_edges()})"
        )


class _DiGraphReverseView:
    """Read-only reversed view of a _DiGraph (swaps succ ↔ pred)."""

    def __init__(self, g: _DiGraph) -> None:
        self._g = g

    def successors(self, n: str):  # type: ignore[return]
        return iter(self._g._pred.get(n, ()))

    def predecessors(self, n: str):  # type: ignore[return]
        return iter(self._g._succ.get(n, ()))

    def __contains__(self, n: object) -> bool:
        return n in self._g._succ

    @property
    def nodes(self):  # type: ignore[return]
        return self._g._succ.keys()

    def number_of_nodes(self) -> int:
        return self._g.number_of_nodes()

    def number_of_edges(self) -> int:
        return self._g.number_of_edges()

    def __repr__(self) -> str:
        return (
            f"_DiGraphReverseView(nodes={self.number_of_nodes()},"
            f" edges={self.number_of_edges()})"
        )


def _compute_content_hash(raw: dict[str, list[str]]) -> str:
    """Return SHA-256 hex digest of the deterministically serialised *raw* dict.

    Sorted keys and sorted callee lists ensure the same graph always yields
    the same hash, regardless of insertion order.
    """
    canonical = json.dumps(
        {k: sorted(v) for k, v in sorted(raw.items())},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


@dataclass
class CallGraphIndex:
    """Function- and module-level call graph over a Python repo.

    The source of truth is `raw`: a mapping {caller_fqn: [callee_fqn, ...]}.
    Graphs are derived from `raw` on construction and on `load`. The backend
    that populates `raw` from source code lives in pyscope_mcp.analyzer
    (AST-based; fully implemented — see CLAUDE.md for the contract).

    ``skeletons`` maps relative file paths → pre-computed lists of SymbolSummary
    dicts, populated during ``pyscope-mcp build`` (index version 2+).
    """

    root: Path
    function_graph: _DiGraph = field(default_factory=_DiGraph)
    module_graph: _DiGraph = field(default_factory=_DiGraph)
    raw: dict[str, list[str]] = field(default_factory=dict)
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
    # Derived at load/from_raw time: maps FQN → relative file path (inverted from skeletons).
    # Not serialised — rebuilt on every load.
    _fqn_to_file: dict[str, str] = field(default_factory=dict, repr=False)
    # Computed at from_raw time: p99 of in-degree distribution, floor of 10.
    # Used by neighborhood() for hub suppression. Not serialised — recomputed on load.
    _in_degree_threshold: int = field(default=10, repr=False)

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
        root = Path(root).resolve()
        fg = _DiGraph()
        mg = _DiGraph()
        for caller, callees in raw.items():
            fg.add_node(caller)
            cm = _module_of(caller)
            mg.add_node(cm)
            for callee in callees:
                fg.add_edge(caller, callee)
                tm = _module_of(callee)
                if cm != tm:
                    mg.add_edge(cm, tm)
        skeletons = skeletons or {}
        # Invert skeletons → _fqn_to_file: {fqn: rel_path}
        fqn_to_file: dict[str, str] = {}
        for rel_path, symbols in skeletons.items():
            for sym in symbols:
                fqn_to_file[sym["fqn"]] = rel_path
        # Compute in-degree threshold: p99 of in-degree distribution, floor of 10.
        # One-time O(N) cost at index load; cached for O(1) lookup during BFS.
        in_degree_threshold = _compute_in_degree_threshold(fg)
        # Compute content_hash: SHA-256 of the deterministically serialised raw dict.
        # Sorted keys + sorted callee lists ensure same raw → same hash.
        content_hash = _compute_content_hash(raw)
        return cls(
            root=root,
            function_graph=fg,
            module_graph=mg,
            raw=raw,
            skeletons=skeletons,
            file_shas=file_shas,
            missed_callers=missed_callers if missed_callers is not None else {},
            git_sha=git_sha,
            content_hash=content_hash,
            _fqn_to_file=fqn_to_file,
            _in_degree_threshold=in_degree_threshold,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "version": INDEX_VERSION,
            "root": str(self.root),
            "raw": self.raw,
            "skeletons": self.skeletons,
            "file_shas": self.file_shas if self.file_shas is not None else {},
            "missed_callers": self.missed_callers,
            "git_sha": self.git_sha,
            "content_hash": self.content_hash,
        }
        path.write_text(json.dumps(payload))
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
        skeletons: dict[str, list[SymbolSummary]] = payload.get("skeletons", {})
        file_shas: dict[str, str] = payload.get("file_shas", {})
        missed_callers: dict[str, dict[str, int]] = payload.get("missed_callers", {})
        git_sha: str | None = payload.get("git_sha")
        return cls.from_raw(
            Path(payload["root"]),
            payload["raw"],
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
        bfs_result: dict[str, int],
        graph: _DiGraph | _DiGraphReverseView,
    ) -> list[str]:
        """Rank BFS result nodes by (hop_depth ASC, -total_degree DESC, fqn ASC).

        ``total_degree`` = in-degree + out-degree of the result node in the
        underlying function graph.  Uses the same ranking convention as
        :meth:`neighborhood`.

        For a ``_DiGraphReverseView``, the backing ``_DiGraph`` (``graph._g``)
        is used to look up both in-degree and out-degree so the degree reflects
        the real graph topology regardless of traversal direction.
        """
        # Resolve the underlying forward graph for degree computation.
        fwd: _DiGraph = graph._g if isinstance(graph, _DiGraphReverseView) else graph  # type: ignore[assignment]

        def total_degree(n: str) -> int:
            out_deg = sum(1 for _ in fwd.successors(n))
            in_deg = sum(1 for _ in fwd._pred.get(n, ()))
            return out_deg + in_deg

        degree_cache: dict[str, int] = {n: total_degree(n) for n in bfs_result}
        return sorted(
            bfs_result,
            key=lambda n: (bfs_result[n], -degree_cache[n], n),
        )

    def callers_of(self, fqn: str, depth: int = 1) -> CallersResult:
        """Return functions that (transitively, up to depth) call *fqn*.

        Results are capped at 50; ``truncated`` signals when the cap fires.
        ``dropped`` is the number of results cut by the cap (always present; 0
        when the cap does not fire).  Results are ranked by
        ``(hop_depth ASC, -total_degree DESC, fqn ASC)`` so depth-1 callers
        always precede depth-2 ones regardless of alphabetical order.
        Response includes uniform staleness fields (``stale``, ``stale_files``,
        and optionally ``stale_action`` / ``index_stale_reason``) and a
        ``completeness`` field (``"complete"`` or ``"partial"``).

        If *fqn* is not present in the graph at all, returns an error dict
        with ``isError: True``, ``error_reason: "fqn_not_in_graph"``, and
        ``stale: False``.  An FQN that is present but has zero callers returns
        ``results: []`` (not an error).
        """
        if fqn not in self.function_graph.nodes:
            commit = self._commit_staleness()
            return {  # type: ignore[return-value]
                "isError": True,
                "error_reason": "fqn_not_in_graph",
                "stale": False,
                "stale_files": [],
                **commit,
            }
        rev_fg = self.function_graph.reverse(copy=False)
        bfs_result = _bfs(rev_fg, fqn, depth)
        ranked = self._rank_bfs_results(bfs_result, rev_fg)
        dropped = max(0, len(ranked) - self._CALLERS_CALLEES_CAP)
        truncated = dropped > 0
        results = ranked[: self._CALLERS_CALLEES_CAP]
        staleness = self._staleness_for(results)
        commit = self._commit_staleness()
        return CallersResult(
            results=results,
            truncated=truncated,
            dropped=dropped,
            completeness=self.completeness_for(results),
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
        if fqn not in self.function_graph.nodes:
            commit = self._commit_staleness()
            return {  # type: ignore[return-value]
                "isError": True,
                "error_reason": "fqn_not_in_graph",
                "stale": False,
                "stale_files": [],
                **commit,
            }
        fg = self.function_graph
        bfs_result = _bfs(fg, fqn, depth)
        ranked = self._rank_bfs_results(bfs_result, fg)
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
        fg = self.function_graph
        rev_fg = fg.reverse(copy=False)

        # Guard: FQN not in graph at all → clear not-found error (not ambiguous empty-edges)
        if symbol not in fg.nodes:
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
            for callee in fg.successors(node):
                edge = (node, callee)
                if edge not in edge_depths or edge_depths[edge] > next_d:
                    edge_depths[edge] = next_d
                if callee not in node_depth:
                    node_depth[callee] = next_d
                    frontier.append((callee, next_d))

            # Callers direction: caller → node (in-degree; hub suppression applies)
            # A node is treated as a hub if its in-degree exceeds the threshold AND
            # it is not the queried symbol itself (the queried symbol is always exempt).
            node_in_degree = sum(1 for _ in rev_fg.successors(node))
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

            for caller in rev_fg.successors(node):
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
            n: sum(1 for _ in fg.successors(n)) + sum(1 for _ in rev_fg.successors(n))
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
        module_graph: _DiGraph,
    ) -> list[str]:
        """Rank module BFS result nodes by (hop_depth ASC, -total_degree DESC, fqn ASC).

        ``total_degree`` = in-degree + out-degree of the result module node in
        *module_graph*.  Uses the same ranking convention as :meth:`neighborhood`.
        """
        rev_mg = module_graph.reverse(copy=False)

        def total_degree(n: str) -> int:
            return (
                sum(1 for _ in module_graph.successors(n))
                + sum(1 for _ in rev_mg.successors(n))
            )

        degree_cache: dict[str, int] = {n: total_degree(n) for n in bfs_result}
        return sorted(
            bfs_result,
            key=lambda n: (bfs_result[n], -degree_cache[n], n),
        )

    def module_callers(self, module: str, depth: int = 1) -> ModuleResult:
        """Return callers of all modules whose FQN starts with *module* (prefix query).

        An exact FQN is a degenerate prefix and behaves identically to the
        pre-change implementation.  Results are capped at 50 items; the
        ``truncated`` key in the returned dict signals when the cap triggers.
        ``dropped`` is the number of results cut by the cap (always present; 0
        when the cap does not fire).  Results are ranked by
        ``(hop_depth ASC, -total_degree DESC, fqn ASC)`` so depth-1 module
        callers always precede depth-2 ones regardless of alphabetical order.
        An empty-string prefix matches all modules.  A prefix that matches no
        module nodes returns ``{"results": [], "truncated": false, "dropped": 0, ...}``.

        Staleness is result-scoped: module FQNs are expanded to file paths via
        prefix-matching against ``_fqn_to_file`` (find all symbol FQNs that start
        with ``result_module + "."``).  Completeness is computed over the same
        expanded symbol FQN set.
        """
        base = _prefix_module_bfs(
            self.module_graph.reverse(copy=False), self.module_graph, module, depth
        )
        raw_union: dict[str, int] = base["results"]  # type: ignore[assignment]
        ranked = self._rank_module_bfs_results(raw_union, self.module_graph)
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

    def module_callees(self, module: str, depth: int = 1) -> ModuleResult:
        """Return callees of all modules whose FQN starts with *module* (prefix query).

        Symmetric to :meth:`module_callers` — see its docstring for semantics.
        ``dropped`` is the number of results cut by the cap (always present; 0
        when the cap does not fire).  Results are ranked by
        ``(hop_depth ASC, -total_degree DESC, fqn ASC)`` so depth-1 module
        callees always precede depth-2 ones regardless of alphabetical order.
        Completeness is computed over the expanded symbol FQNs of the result modules.
        """
        base = _prefix_module_bfs(self.module_graph, self.module_graph, module, depth)
        raw_union: dict[str, int] = base["results"]  # type: ignore[assignment]
        ranked = self._rank_module_bfs_results(raw_union, self.module_graph)
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
        all_matches = [n for n in self.function_graph.nodes if s in n.lower()]
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
        return StatsResult(
            functions=self.function_graph.number_of_nodes(),
            function_edges=self.function_graph.number_of_edges(),
            modules=self.module_graph.number_of_nodes(),
            module_edges=self.module_graph.number_of_edges(),
            **commit,  # type: ignore[arg-type]
        )


_MODULE_BFS_CAP = 50

_IN_DEGREE_THRESHOLD_FLOOR = 10


def _compute_in_degree_threshold(fg: _DiGraph) -> int:
    """Compute the in-degree hub-suppression threshold for *fg*.

    Returns the p99 of the in-degree distribution across all nodes in the
    function graph, with a floor of ``_IN_DEGREE_THRESHOLD_FLOOR`` (10).

    If the graph has fewer than 2 nodes, returns the floor.

    This is called once during ``CallGraphIndex.from_raw`` and the result is
    cached on the index instance as ``_in_degree_threshold``.  Never called
    per-query — see Law 2.
    """
    if fg.number_of_nodes() < 2:
        return _IN_DEGREE_THRESHOLD_FLOOR
    in_degrees = [len(fg._pred.get(n, ())) for n in fg.nodes]
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
    query_graph: _DiGraph | _DiGraphReverseView,
    node_graph: _DiGraph,
    prefix: str,
    depth: int,
    cap: int = _MODULE_BFS_CAP,
) -> dict[str, object]:
    """BFS over *query_graph* for all nodes in *node_graph* whose FQN starts with *prefix*.

    Results from each matched seed node are unioned and deduplicated.  The matched
    seed nodes themselves are excluded from the result set (same semantics as _bfs).
    Results are capped at *cap*; ``truncated`` reflects whether the full union
    exceeded the cap.  ``dropped`` is the number of results cut by the cap (always
    present; 0 when the cap does not fire).

    When a result node is reachable from multiple seeds, its hop depth is the
    minimum across all seeds (closest path wins).

    ``prefix=""`` matches all module nodes.
    """
    # Collect all module nodes that start with the given prefix
    matched_seeds = [n for n in node_graph.nodes if n.startswith(prefix)]

    # Union BFS results across all matched seeds, excluding the seeds themselves.
    # Track min hop depth per result node.
    seed_set = set(matched_seeds)
    union: dict[str, int] = {}  # fqn -> min hop depth
    for seed in matched_seeds:
        for node, hop in _bfs(query_graph, seed, depth).items():
            if node not in seed_set:
                if node not in union or hop < union[node]:
                    union[node] = hop

    dropped = max(0, len(union) - cap)
    truncated = dropped > 0
    return {
        "results": union,  # raw dict returned; callers apply ranking + slice
        "truncated": truncated,
        "dropped": dropped,
    }


def _bfs(g: _DiGraph | _DiGraphReverseView, start: str, depth: int) -> dict[str, int]:
    """BFS from *start* up to *depth* hops.

    ``depth=0`` returns ``{}``.  ``depth=1`` returns direct neighbours only.
    ``depth=N`` returns all nodes reachable within N hops (excluding *start*).

    Returns a ``dict[fqn, min_hop_depth]`` mapping each reachable FQN to its
    minimum hop distance from *start*.  The traversal set is identical to the
    previous ``sorted(seen)`` implementation; only ordering and depth metadata
    are new.
    """
    if depth <= 0 or start not in g:
        return {}
    seen: dict[str, int] = {}
    frontier = [start]
    for hop in range(1, depth + 1):
        nxt: list[str] = []
        for n in frontier:
            for s in g.successors(n):
                if s not in seen and s != start:
                    seen[s] = hop
                    nxt.append(s)
        frontier = nxt
    return seen
