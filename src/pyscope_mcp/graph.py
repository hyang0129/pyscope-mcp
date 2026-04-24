from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict


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

    @classmethod
    def from_raw(
        cls,
        root: str | Path,
        raw: dict[str, list[str]],
        skeletons: dict[str, list[SymbolSummary]] | None = None,
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
        return cls(
            root=root,
            function_graph=fg,
            module_graph=mg,
            raw=raw,
            skeletons=skeletons or {},
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "version": 2,
            "root": str(self.root),
            "raw": self.raw,
            "skeletons": self.skeletons,
        }
        path.write_text(json.dumps(payload))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CallGraphIndex":
        path = Path(path)
        payload = json.loads(path.read_text())
        version = payload.get("version")
        if version not in (1, 2):
            raise ValueError(f"unsupported index version: {version}")
        skeletons: dict[str, list[SymbolSummary]] = payload.get("skeletons", {})
        return cls.from_raw(Path(payload["root"]), payload["raw"], skeletons=skeletons)

    def file_skeleton(self, path: str, cap: int = _SKELETON_CAP) -> dict:
        """Return a compact symbol list for the given relative file path.

        Returns a dict with:
          - ``results``: list of SymbolSummary dicts sorted by lineno (capped at *cap*)
          - ``truncated``: True when the full symbol count exceeds *cap*
          - ``total``: total number of symbols before capping

        If the path is not in the index, returns an error dict with ``isError: True``.
        """
        if path not in self.skeletons:
            return {
                "isError": True,
                "message": (
                    f"File '{path}' is not in the index. "
                    "Run 'pyscope-mcp build' and then 'reload' to update the index."
                ),
            }
        symbols = self.skeletons[path]
        total = len(symbols)
        truncated = total > cap
        return {
            "results": symbols[:cap],
            "truncated": truncated,
            "total": total,
        }

    def callers_of(self, fqn: str, depth: int = 1) -> list[str]:
        return _bfs(self.function_graph.reverse(copy=False), fqn, depth)

    def callees_of(self, fqn: str, depth: int = 1) -> list[str]:
        return _bfs(self.function_graph, fqn, depth)

    def module_callers(self, module: str, depth: int = 1) -> dict[str, object]:
        """Return callers of all modules whose FQN starts with *module* (prefix query).

        An exact FQN is a degenerate prefix and behaves identically to the
        pre-change implementation.  Results are capped at 50 items; the
        ``truncated`` key in the returned dict signals when the cap triggers.
        An empty-string prefix matches all modules.  A prefix that matches no
        module nodes returns ``{"results": [], "truncated": false}``.
        """
        return _prefix_module_bfs(
            self.module_graph.reverse(copy=False), self.module_graph, module, depth
        )

    def module_callees(self, module: str, depth: int = 1) -> dict[str, object]:
        """Return callees of all modules whose FQN starts with *module* (prefix query).

        Symmetric to :meth:`module_callers` — see its docstring for semantics.
        """
        return _prefix_module_bfs(self.module_graph, self.module_graph, module, depth)

    def search(self, substring: str, limit: int = 50) -> dict[str, object]:
        """Substring search over known fully-qualified function names.

        Returns a dict with:
          - ``results``: list of matching FQNs (capped at *limit*)
          - ``truncated``: True when the full match count exceeds *limit*
          - ``total_matched``: count of all matches before capping
        """
        s = substring.lower()
        all_matches = [n for n in self.function_graph.nodes if s in n.lower()]
        total = len(all_matches)
        return {
            "results": all_matches[:limit],
            "truncated": total > limit,
            "total_matched": total,
        }

    def stats(self) -> dict[str, int]:
        return {
            "functions": self.function_graph.number_of_nodes(),
            "function_edges": self.function_graph.number_of_edges(),
            "modules": self.module_graph.number_of_nodes(),
            "module_edges": self.module_graph.number_of_edges(),
        }


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
