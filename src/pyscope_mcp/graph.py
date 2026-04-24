from __future__ import annotations

import hashlib
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
    # None = pre-v3 index (no hashes stored); dict = v3 index with per-file SHA256 digests.
    file_shas: dict[str, str] | None = field(default=None)

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
        return cls(
            root=root,
            function_graph=fg,
            module_graph=mg,
            raw=raw,
            skeletons=skeletons or {},
            file_shas=file_shas,
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

    def file_skeleton(self, path: str, cap: int = _SKELETON_CAP) -> dict:
        """Return a compact symbol list for the given relative file path.

        Returns a dict with:
          - ``results``: list of SymbolSummary dicts sorted by lineno (capped at *cap*)
          - ``truncated``: True when the full symbol count exceeds *cap*
          - ``total``: total number of symbols before capping
          - ``stale``: True when the live file differs from what was indexed
          - ``staleness_info``: dict with ``reason`` and ``action`` when ``stale`` is True

        Staleness reasons:
          - ``file_changed``: file exists but content has changed since build
          - ``file_not_found``: file was in the index but no longer exists on disk
          - ``file_not_in_index``: path not in index (also sets ``isError: True``)
          - ``index_format_incompatible``: index is pre-v3 and has no stored hashes

        Results are always returned when the path is in the index, even when stale.
        If the path is not in the index, returns an error dict with ``isError: True``.
        """
        # Scenario D / Scenario E (isError cases that may also carry stale info)
        if path not in self.skeletons:
            if self.file_shas is None:
                # Pre-v3 index: stale, but we can't distinguish not-in-index from
                # index_format_incompatible for paths absent from skeletons.
                # Since the path isn't in skeletons either, use file_not_in_index.
                return {
                    "isError": True,
                    "stale": True,
                    "staleness_info": {
                        "reason": "file_not_in_index",
                        "action": self._STALE_ACTION,
                    },
                }
            return {
                "isError": True,
                "stale": True,
                "staleness_info": {
                    "reason": "file_not_in_index",
                    "action": self._STALE_ACTION,
                },
            }

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
            base_result["staleness_info"] = {
                "reason": "index_format_incompatible",
                "action": self._STALE_ACTION,
            }
            return base_result

        # Scenarios A/B/C: v3 index — compare live file hash against stored hash.
        live_file = self.root / path
        if not live_file.exists():
            # Scenario C: file deleted since build.
            base_result["stale"] = True
            base_result["staleness_info"] = {
                "reason": "file_not_found",
                "action": self._STALE_ACTION,
            }
            return base_result

        stored_sha = self.file_shas.get(path)
        live_sha = hashlib.sha256(live_file.read_bytes()).hexdigest()

        if stored_sha is None or live_sha != stored_sha:
            # Scenario B: file changed.
            base_result["stale"] = True
            base_result["staleness_info"] = {
                "reason": "file_changed",
                "action": self._STALE_ACTION,
            }
            return base_result

        # Scenario A: fresh — hashes match.
        base_result["stale"] = False
        return base_result

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
