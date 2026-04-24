from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from pyscope_mcp.types import (
    CalleesResult,
    CallersResult,
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


@dataclass
class CallGraphIndex:
    """Function- and module-level call graph over a Python repo.

    The source of truth is `raw`: a mapping {caller_fqn: [callee_fqn, ...]}.
    Graphs are derived from `raw` on construction and on `load`. The backend
    that populates `raw` from source code lives in pyscope_mcp.analyzer
    (not yet implemented — see CLAUDE.md for the rewrite plan).

    ``skeletons`` maps relative file paths → pre-computed lists of SymbolSummary
    dicts, populated during ``pyscope-mcp build`` (index version 2+).
    """

    root: Path
    function_graph: _DiGraph = field(default_factory=_DiGraph)
    module_graph: _DiGraph = field(default_factory=_DiGraph)
    raw: dict[str, list[str]] = field(default_factory=dict)
    skeletons: dict[str, list[SymbolSummary]] = field(default_factory=dict)
    # None = pre-v3 index (no hashes stored); dict = v3 index with per-file SHA256 digests.
    file_shas: dict[str, str] | None = field(default=None)
    # Derived at load/from_raw time: maps FQN → relative file path (inverted from skeletons).
    # Not serialised — rebuilt on every load.
    _fqn_to_file: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_raw(
        cls,
        root: str | Path,
        raw: dict[str, list[str]],
        skeletons: dict[str, list[SymbolSummary]] | None = None,
        file_shas: dict[str, str] | None = None,
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
        return cls(
            root=root,
            function_graph=fg,
            module_graph=mg,
            raw=raw,
            skeletons=skeletons,
            file_shas=file_shas,
            _fqn_to_file=fqn_to_file,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "version": 3,
            "root": str(self.root),
            "raw": self.raw,
            "skeletons": self.skeletons,
            "file_shas": self.file_shas if self.file_shas is not None else {},
        }
        path.write_text(json.dumps(payload))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CallGraphIndex":
        path = Path(path)
        payload = json.loads(path.read_text())
        version = payload.get("version")
        if version not in (1, 2, 3):
            raise ValueError(f"unsupported index version: {version}")
        skeletons: dict[str, list[SymbolSummary]] = payload.get("skeletons", {})
        # v1/v2: no file_shas stored — use None as sentinel to signal pre-v3.
        # v3: read the stored file_shas dict.
        file_shas: dict[str, str] | None = None if version < 3 else payload.get("file_shas", {})
        return cls.from_raw(
            Path(payload["root"]),
            payload["raw"],
            skeletons=skeletons,
            file_shas=file_shas,
        )

    _STALE_ACTION = "Run 'pyscope-mcp build' then 'reload' to update the index."

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
        # Scenario D / isError: path not in skeletons
        if path not in self.skeletons:
            result: dict = {
                "isError": True,
                "stale": True,
                "stale_files": [],
                "stale_action": self._STALE_ACTION,
            }
            if self.file_shas is None:
                result["index_stale_reason"] = "index_format_incompatible"
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
            return base_result

        # Scenarios A/B/C: v3 index — compare live file hash against stored hash.
        live_file = self.root / path
        if not live_file.exists():
            # Scenario C: file deleted since build.
            base_result["stale"] = True
            base_result["stale_files"] = [path]
            base_result["stale_action"] = self._STALE_ACTION
            return base_result

        stored_sha = self.file_shas.get(path)
        live_sha = hashlib.sha256(live_file.read_bytes()).hexdigest()

        if stored_sha is None or live_sha != stored_sha:
            # Scenario B: file changed.
            base_result["stale"] = True
            base_result["stale_files"] = [path]
            base_result["stale_action"] = self._STALE_ACTION
            return base_result

        # Scenario A: fresh — hashes match.
        base_result["stale"] = False
        base_result["stale_files"] = []
        return base_result

    _CALLERS_CALLEES_CAP = 50

    def callers_of(self, fqn: str, depth: int = 1) -> CallersResult:
        """Return functions that (transitively, up to depth) call *fqn*.

        Results are capped at 50; ``truncated`` signals when the cap fires.
        Response includes uniform staleness fields (``stale``, ``stale_files``,
        and optionally ``stale_action`` / ``index_stale_reason``).
        """
        all_results = _bfs(self.function_graph.reverse(copy=False), fqn, depth)
        truncated = len(all_results) > self._CALLERS_CALLEES_CAP
        results = all_results[: self._CALLERS_CALLEES_CAP]
        staleness = self._staleness_for(results)
        return CallersResult(
            results=results,
            truncated=truncated,
            **staleness,  # type: ignore[arg-type]
        )

    def callees_of(self, fqn: str, depth: int = 1) -> CalleesResult:
        """Return functions (transitively, up to depth) called by *fqn*.

        Results are capped at 50; ``truncated`` signals when the cap fires.
        Response includes uniform staleness fields (``stale``, ``stale_files``,
        and optionally ``stale_action`` / ``index_stale_reason``).
        """
        all_results = _bfs(self.function_graph, fqn, depth)
        truncated = len(all_results) > self._CALLERS_CALLEES_CAP
        results = all_results[: self._CALLERS_CALLEES_CAP]
        staleness = self._staleness_for(results)
        return CalleesResult(
            results=results,
            truncated=truncated,
            **staleness,  # type: ignore[arg-type]
        )

    def neighborhood(
        self,
        symbol: str,
        depth: int = 2,
        token_budget: int = 1000,
    ) -> NeighborhoodResult:
        """Return a bounded bidirectional subgraph around *symbol*.

        BFS outward (both callers and callees) up to *depth* hops.  Candidate
        edges are ranked by (hop_depth, -degree) — depth-first with degree
        tiebreak — then truncated to fit within *token_budget* (4 chars/token
        estimate).

        Response keys:
          - ``symbol``: the queried FQN
          - ``depth_full``: deepest level with all edges intact (no drops)
          - ``depth_truncated``: first level where dropping started (only when ``truncated=True``)
          - ``edges``: list of ``[caller, callee]`` pairs
          - ``truncated``: True when the budget was hit
          - ``token_budget_used``: estimated tokens consumed (chars / 4)

        ``depth_full`` reflects the actual graph depth, not the declared *depth*
        parameter (if the graph is shallower, ``depth_full`` mirrors reality).
        """
        fg = self.function_graph
        rev_fg = fg.reverse(copy=False)

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

        while frontier:
            node, d = frontier.popleft()
            if d >= depth:
                continue
            next_d = d + 1

            # Callees direction: node → callee
            for callee in fg.successors(node):
                edge = (node, callee)
                if edge not in edge_depths or edge_depths[edge] > next_d:
                    edge_depths[edge] = next_d
                if callee not in node_depth:
                    node_depth[callee] = next_d
                    frontier.append((callee, next_d))

            # Callers direction: caller → node
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
            result = NeighborhoodResult(
                symbol=symbol,
                depth_full=0,
                edges=[],
                truncated=False,
                token_budget_used=0,
            )
            result.update(staleness)  # type: ignore[arg-type]
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

        result = NeighborhoodResult(
            symbol=symbol,
            depth_full=depth_full,
            edges=[[c, e] for c, e in kept_edges],
            truncated=truncated,
            token_budget_used=chars_used // 4,
        )
        result.update(staleness)  # type: ignore[arg-type]
        if truncated and depth_truncated is not None:
            result["depth_truncated"] = depth_truncated

        return result

    def module_callers(self, module: str, depth: int = 1) -> ModuleResult:
        """Return callers of all modules whose FQN starts with *module* (prefix query).

        An exact FQN is a degenerate prefix and behaves identically to the
        pre-change implementation.  Results are capped at 50 items; the
        ``truncated`` key in the returned dict signals when the cap triggers.
        An empty-string prefix matches all modules.  A prefix that matches no
        module nodes returns ``{"results": [], "truncated": false, "stale": ..., "stale_files": []}``.

        Staleness is result-scoped: module FQNs are expanded to file paths via
        prefix-matching against ``_fqn_to_file`` (find all symbol FQNs that start
        with ``result_module + "."``).
        """
        base = _prefix_module_bfs(
            self.module_graph.reverse(copy=False), self.module_graph, module, depth
        )
        result_modules: list[str] = base["results"]  # type: ignore[assignment]
        staleness = self._staleness_for_modules(result_modules)
        return ModuleResult(
            results=result_modules,
            truncated=base["truncated"],  # type: ignore[typeddict-item]
            **staleness,  # type: ignore[arg-type]
        )

    def module_callees(self, module: str, depth: int = 1) -> ModuleResult:
        """Return callees of all modules whose FQN starts with *module* (prefix query).

        Symmetric to :meth:`module_callers` — see its docstring for semantics.
        """
        base = _prefix_module_bfs(self.module_graph, self.module_graph, module, depth)
        result_modules: list[str] = base["results"]  # type: ignore[assignment]
        staleness = self._staleness_for_modules(result_modules)
        return ModuleResult(
            results=result_modules,
            truncated=base["truncated"],  # type: ignore[typeddict-item]
            **staleness,  # type: ignore[arg-type]
        )

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
        symbol_fqns: list[str] = []
        for mod in module_fqns:
            prefix = mod + "."
            for fqn in self._fqn_to_file:
                if fqn.startswith(prefix):
                    symbol_fqns.append(fqn)
        return self._staleness_for(symbol_fqns)

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
        return SearchResult(
            results=results,
            truncated=total > limit,
            total_matched=total,
            **staleness,  # type: ignore[arg-type]
        )

    def stats(self) -> StatsResult:
        return StatsResult(
            functions=self.function_graph.number_of_nodes(),
            function_edges=self.function_graph.number_of_edges(),
            modules=self.module_graph.number_of_nodes(),
            module_edges=self.module_graph.number_of_edges(),
        )


_MODULE_BFS_CAP = 50


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
    exceeded the cap.

    ``prefix=""`` matches all module nodes.
    """
    # Collect all module nodes that start with the given prefix
    matched_seeds = [n for n in node_graph.nodes if n.startswith(prefix)]

    # Union BFS results across all matched seeds, excluding the seeds themselves
    seed_set = set(matched_seeds)
    union: set[str] = set()
    for seed in matched_seeds:
        for node in _bfs(query_graph, seed, depth):
            if node not in seed_set:
                union.add(node)

    results_all = sorted(union)
    truncated = len(results_all) > cap
    return {
        "results": results_all[:cap],
        "truncated": truncated,
    }


def _bfs(g: _DiGraph | _DiGraphReverseView, start: str, depth: int) -> list[str]:
    """BFS from *start* up to *depth* hops.

    ``depth=0`` returns ``[]``.  ``depth=1`` returns direct neighbours only.
    ``depth=N`` returns all nodes reachable within N hops (excluding *start*).
    """
    if depth <= 0 or start not in g:
        return []
    seen: set[str] = {start}
    frontier = [start]
    for _ in range(depth):
        nxt: list[str] = []
        for n in frontier:
            for s in g.successors(n):
                if s not in seen:
                    seen.add(s)
                    nxt.append(s)
        frontier = nxt
    seen.discard(start)
    return sorted(seen)
