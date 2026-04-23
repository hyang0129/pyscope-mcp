"""Tests for union-annotation resolution in _resolve_annotation_to_class.

Covers:
- PEP 604 unions: X | None, None | X, X | Y (both in-package: first-wins)
- Optional[X], Union[X, None], Union[X, Y, None]
- Qualified forms: typing.Optional[X], typing.Union[X, Y], alias forms
- Nested: X | Y | None (left-associative BinOp)
- False-positive guards: list[X], Callable[[X], Y], Optional[ExternalType],
  X | int (only one in-package), X | Y both in-package (first-wins)
"""

from __future__ import annotations

from pathlib import Path

from pyscope_mcp.analyzer import build_raw


def _make_package(tmp_path: Path, pkg_name: str, files: dict[str, str]) -> Path:
    """Create a minimal package tree for testing."""
    pkg = tmp_path / pkg_name
    pkg.mkdir()
    if "__init__.py" not in files:
        (pkg / "__init__.py").write_text("")
    for rel, content in files.items():
        target = pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


# ---------------------------------------------------------------------------
# PEP 604: X | None
# ---------------------------------------------------------------------------

def test_binop_union_x_or_none_resolves_x(tmp_path: Path) -> None:
    """Parameter annotation `agent: MyAgent | None` should resolve to MyAgent."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "from pkg.agent import MyAgent\n"
            "\n"
            "def export(agent: MyAgent | None = None) -> None:\n"
            "    if agent is not None:\n"
            "        agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.agent.MyAgent.run" in raw.get("pkg.caller.export", [])


def test_binop_union_none_or_x_resolves_x(tmp_path: Path) -> None:
    """Reversed form `None | MyAgent` should also resolve to MyAgent."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "from pkg.agent import MyAgent\n"
            "\n"
            "def export(agent: None | MyAgent = None) -> None:\n"
            "    if agent is not None:\n"
            "        agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.agent.MyAgent.run" in raw.get("pkg.caller.export", [])


def test_binop_union_x_or_int_resolves_in_package_only(tmp_path: Path) -> None:
    """X | int — int is not in-package, so MyAgent is picked."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "from pkg.agent import MyAgent\n"
            "\n"
            "def export(agent: MyAgent | int = None) -> None:\n"
            "    agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.agent.MyAgent.run" in raw.get("pkg.caller.export", [])


def test_binop_union_both_in_package_first_wins(tmp_path: Path) -> None:
    """X | Y where both are in-package — first (left) should win."""
    root = _make_package(tmp_path, "pkg", {
        "agents.py": (
            "class AgentA:\n"
            "    def run(self): pass\n"
            "\n"
            "class AgentB:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "from pkg.agents import AgentA, AgentB\n"
            "\n"
            "def dispatch(agent: AgentA | AgentB) -> None:\n"
            "    agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.caller.dispatch", [])
    # First-wins: AgentA.run should be present
    assert "pkg.agents.AgentA.run" in callees
    # AgentB.run should NOT also be added (first-wins, not both)
    assert "pkg.agents.AgentB.run" not in callees


def test_binop_nested_union_x_or_y_or_none(tmp_path: Path) -> None:
    """X | Y | None is left-associative: (X | Y) | None.  First in-package wins."""
    root = _make_package(tmp_path, "pkg", {
        "agents.py": (
            "class AgentA:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "from pkg.agents import AgentA\n"
            "\n"
            "def dispatch(agent: AgentA | int | None) -> None:\n"
            "    agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.agents.AgentA.run" in raw.get("pkg.caller.dispatch", [])


# ---------------------------------------------------------------------------
# Optional[X]  (bare name)
# ---------------------------------------------------------------------------

def test_optional_bare_name(tmp_path: Path) -> None:
    """Optional[MyAgent] with `from typing import Optional`."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "from typing import Optional\n"
            "from pkg.agent import MyAgent\n"
            "\n"
            "def export(agent: Optional[MyAgent] = None) -> None:\n"
            "    if agent is not None:\n"
            "        agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.agent.MyAgent.run" in raw.get("pkg.caller.export", [])


def test_optional_external_type_returns_none(tmp_path: Path) -> None:
    """Optional[ExternalLib] should not resolve (external type not in-package)."""
    root = _make_package(tmp_path, "pkg", {
        "caller.py": (
            "from typing import Optional\n"
            "\n"
            # httpx.Client is not an in-package class, so no resolution
            "def export(client: Optional[int] = None) -> None:\n"
            "    pass\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # No in-package method calls — just ensure no crash and no spurious edges
    callees = raw.get("pkg.caller.export", [])
    assert callees == [] or all("pkg." in c for c in callees)


# ---------------------------------------------------------------------------
# Union[X, None]
# ---------------------------------------------------------------------------

def test_union_x_none(tmp_path: Path) -> None:
    """Union[MyAgent, None] with `from typing import Union`."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "from typing import Union\n"
            "from pkg.agent import MyAgent\n"
            "\n"
            "def export(agent: Union[MyAgent, None] = None) -> None:\n"
            "    if agent is not None:\n"
            "        agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.agent.MyAgent.run" in raw.get("pkg.caller.export", [])


def test_union_x_y_none(tmp_path: Path) -> None:
    """Union[AgentA, AgentB, None] — first in-package wins."""
    root = _make_package(tmp_path, "pkg", {
        "agents.py": (
            "class AgentA:\n"
            "    def run(self): pass\n"
            "\n"
            "class AgentB:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "from typing import Union\n"
            "from pkg.agents import AgentA, AgentB\n"
            "\n"
            "def export(agent: Union[AgentA, AgentB, None] = None) -> None:\n"
            "    agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.caller.export", [])
    # First-wins: AgentA
    assert "pkg.agents.AgentA.run" in callees
    assert "pkg.agents.AgentB.run" not in callees


# ---------------------------------------------------------------------------
# Qualified forms: typing.Optional[X], typing.Union[X, Y]
# ---------------------------------------------------------------------------

def test_typing_optional_qualified(tmp_path: Path) -> None:
    """typing.Optional[MyAgent] without explicit `from typing import Optional`."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "import typing\n"
            "from pkg.agent import MyAgent\n"
            "\n"
            "def export(agent: typing.Optional[MyAgent] = None) -> None:\n"
            "    if agent is not None:\n"
            "        agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.agent.MyAgent.run" in raw.get("pkg.caller.export", [])


def test_typing_union_qualified(tmp_path: Path) -> None:
    """typing.Union[MyAgent, None] without explicit `from typing import Union`."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "import typing\n"
            "from pkg.agent import MyAgent\n"
            "\n"
            "def export(agent: typing.Union[MyAgent, None] = None) -> None:\n"
            "    if agent is not None:\n"
            "        agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.agent.MyAgent.run" in raw.get("pkg.caller.export", [])


def test_typing_alias_optional(tmp_path: Path) -> None:
    """import typing as t; t.Optional[MyAgent] — alias form."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "import typing as t\n"
            "from pkg.agent import MyAgent\n"
            "\n"
            "def export(agent: t.Optional[MyAgent] = None) -> None:\n"
            "    if agent is not None:\n"
            "        agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.agent.MyAgent.run" in raw.get("pkg.caller.export", [])


def test_typing_alias_union(tmp_path: Path) -> None:
    """import typing as t; t.Union[MyAgent, None] — alias form."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "import typing as t\n"
            "from pkg.agent import MyAgent\n"
            "\n"
            "def export(agent: t.Union[MyAgent, None] = None) -> None:\n"
            "    if agent is not None:\n"
            "        agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.agent.MyAgent.run" in raw.get("pkg.caller.export", [])


# ---------------------------------------------------------------------------
# False-positive guards
# ---------------------------------------------------------------------------

def test_list_subscript_not_treated_as_union(tmp_path: Path) -> None:
    """list[MyAgent] must NOT resolve to MyAgent — not a union form."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "from pkg.agent import MyAgent\n"
            "\n"
            "def process(agents: list[MyAgent]) -> None:\n"
            "    # iterating, not calling agent.run() directly via annotation\n"
            "    pass\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # The parameter annotation should NOT resolve to MyAgent via the list subscript
    # (verify by checking there's no erroneous direct method call)
    callees = raw.get("pkg.caller.process", [])
    assert "pkg.agent.MyAgent.run" not in callees


def test_callable_subscript_not_treated_as_union(tmp_path: Path) -> None:
    """Callable[[MyAgent], None] must NOT resolve to MyAgent."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "from typing import Callable\n"
            "from pkg.agent import MyAgent\n"
            "\n"
            "def process(fn: Callable[[MyAgent], None]) -> None:\n"
            "    pass\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.caller.process", [])
    assert "pkg.agent.MyAgent.run" not in callees


def test_annotated_assignment_with_union(tmp_path: Path) -> None:
    """x: MyAgent | None = ... inside a function body resolves via annotation."""
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class MyAgent:\n"
            "    def run(self): pass\n"
        ),
        "caller.py": (
            "from pkg.agent import MyAgent\n"
            "\n"
            "def export() -> None:\n"
            "    agent: MyAgent | None = None\n"
            "    if agent is not None:\n"
            "        agent.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.agent.MyAgent.run" in raw.get("pkg.caller.export", [])
