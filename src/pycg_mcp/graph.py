from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx


@dataclass
class CallGraphIndex:
    """Function- and module-level call graph over a Python repo, backed by pycg.

    pycg produces a JSON mapping {caller_fqn: [callee_fqn, ...]}. We wrap that
    in a NetworkX DiGraph so queries (callers, callees, reachability, module
    subgraphs) are cheap after the one-shot pycg run.
    """

    root: Path
    function_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    module_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    raw: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, root: str | Path, package: str | None = None) -> "CallGraphIndex":
        root = Path(root).resolve()
        files = [str(p) for p in root.rglob("*.py") if ".venv" not in p.parts]
        if not files:
            raise ValueError(f"no .py files found under {root}")

        cmd = [sys.executable, "-m", "pycg", "--package", package or root.name, *files]
        out = subprocess.check_output(cmd, cwd=root, text=True)
        raw = json.loads(out)
        return cls._from_raw(root, raw)

    @classmethod
    def _from_raw(cls, root: Path, raw: dict[str, list[str]]) -> "CallGraphIndex":
        fg: nx.DiGraph = nx.DiGraph()
        mg: nx.DiGraph = nx.DiGraph()
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
        return cls._from_raw(Path(payload["root"]), payload["raw"])

    def callers_of(self, fqn: str, depth: int = 1) -> list[str]:
        return _bfs(self.function_graph.reverse(copy=False), fqn, depth)

    def callees_of(self, fqn: str, depth: int = 1) -> list[str]:
        return _bfs(self.function_graph, fqn, depth)

    def module_callers(self, module: str, depth: int = 1) -> list[str]:
        return _bfs(self.module_graph.reverse(copy=False), module, depth)

    def module_callees(self, module: str, depth: int = 1) -> list[str]:
        return _bfs(self.module_graph, module, depth)

    def search(self, substring: str, limit: int = 50) -> list[str]:
        s = substring.lower()
        return [n for n in self.function_graph.nodes if s in n.lower()][:limit]

    def stats(self) -> dict[str, int]:
        return {
            "functions": self.function_graph.number_of_nodes(),
            "function_edges": self.function_graph.number_of_edges(),
            "modules": self.module_graph.number_of_nodes(),
            "module_edges": self.module_graph.number_of_edges(),
        }


def _module_of(fqn: str) -> str:
    return fqn.rsplit(".", 1)[0] if "." in fqn else fqn


def _bfs(g: nx.DiGraph, start: str, depth: int) -> list[str]:
    if start not in g:
        return []
    seen: set[str] = {start}
    frontier = [start]
    for _ in range(max(1, depth)):
        nxt: list[str] = []
        for n in frontier:
            for s in g.successors(n):
                if s not in seen:
                    seen.add(s)
                    nxt.append(s)
        frontier = nxt
    seen.discard(start)
    return sorted(seen)
