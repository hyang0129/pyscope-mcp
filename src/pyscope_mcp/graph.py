from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


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
    """

    root: Path
    function_graph: _DiGraph = field(default_factory=_DiGraph)
    module_graph: _DiGraph = field(default_factory=_DiGraph)
    raw: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, root: str | Path, raw: dict[str, list[str]]) -> "CallGraphIndex":
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
        return cls(root=root, function_graph=fg, module_graph=mg, raw=raw)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "root": str(self.root),
            "raw": self.raw,
        }
        path.write_text(json.dumps(payload))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CallGraphIndex":
        path = Path(path)
        payload = json.loads(path.read_text())
        if payload.get("version") != 1:
            raise ValueError(f"unsupported index version: {payload.get('version')}")
        return cls.from_raw(Path(payload["root"]), payload["raw"])

    def callers_of(self, fqn: str, depth: int = 1) -> list[str]:
        return _bfs(self.function_graph.reverse(copy=False), fqn, depth)

    def callees_of(self, fqn: str, depth: int = 1) -> list[str]:
        return _bfs(self.function_graph, fqn, depth)

    def module_callers(self, module: str, depth: int = 1) -> list[str]:
        return _bfs(self.module_graph.reverse(copy=False), module, depth)

    def module_callees(self, module: str, depth: int = 1) -> list[str]:
        return _bfs(self.module_graph, module, depth)

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


def _module_of(fqn: str) -> str:
    return fqn.rsplit(".", 1)[0] if "." in fqn else fqn


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
