"""M6: class-hierarchy (super / inherited self.method) + indirect dispatch."""

from __future__ import annotations

from pathlib import Path

from pyscope_mcp.analyzer import build_raw


def _make_package(tmp_path: Path, pkg_name: str, files: dict[str, str]) -> Path:
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
# super().__init__ / super().method
# ---------------------------------------------------------------------------

def test_super_init_resolves_to_parent_in_same_module(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Base:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "\n"
            "class Child(Base):\n"
            "    def __init__(self, x, y):\n"
            "        super().__init__(x)\n"
            "        self.y = y\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Base.__init__" in raw["pkg.mod.Child.__init__"]


def test_super_method_resolves_across_modules(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "base.py": (
            "class Base:\n"
            "    def greet(self):\n"
            "        return 'hi'\n"
        ),
        "child.py": (
            "from pkg.base import Base\n"
            "\n"
            "class Child(Base):\n"
            "    def greet(self):\n"
            "        return super().greet() + '!'\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.base.Base.greet" in raw["pkg.child.Child.greet"]


def test_super_skips_external_parent(tmp_path: Path) -> None:
    """`super().__init__()` where parent is external (e.g. object) emits nothing."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Thing:\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # Child __init__ may not even be a key if it emitted nothing; accept both.
    callees = raw.get("pkg.mod.Thing.__init__", [])
    assert not any(c.endswith(".__init__") and "pkg" in c.split(".")[:-1][:1]
                   for c in callees)


# ---------------------------------------------------------------------------
# self.method() with MRO fallback
# ---------------------------------------------------------------------------

def test_self_method_inherited_from_in_package_base(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Base:\n"
            "    def helper(self):\n"
            "        return 42\n"
            "\n"
            "class Child(Base):\n"
            "    def run(self):\n"
            "        return self.helper()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Base.helper" in raw["pkg.mod.Child.run"]


def test_self_method_prefers_own_override_over_base(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Base:\n"
            "    def helper(self):\n"
            "        return 'base'\n"
            "\n"
            "class Child(Base):\n"
            "    def helper(self):\n"
            "        return 'child'\n"
            "    def run(self):\n"
            "        return self.helper()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw["pkg.mod.Child.run"]
    assert "pkg.mod.Child.helper" in callees
    assert "pkg.mod.Base.helper" not in callees


def test_mro_diamond_depth_first_left(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class A:\n"
            "    def foo(self): ...\n"
            "class B(A):\n"
            "    def foo(self): ...\n"
            "class C(A):\n"
            "    pass\n"
            "class D(B, C):\n"
            "    def run(self):\n"
            "        return self.foo()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # DFS left-first: D -> B (has foo) wins.
    assert "pkg.mod.B.foo" in raw["pkg.mod.D.run"]
    assert "pkg.mod.A.foo" not in raw["pkg.mod.D.run"]


# ---------------------------------------------------------------------------
# Regression: nested-function scope + ABC base (issue #3)
# ---------------------------------------------------------------------------

def test_self_method_inherited_via_abc_base_cross_module(tmp_path: Path) -> None:
    """Regression: ContractChapterWriter-style failure.

    A subclass in one module extends an ABC-inheriting base from another
    module.  A *nested* helper function inside a method calls
    ``self.call_llm(...)``.  Before the fix, ``_enclosing_class_fqn`` walked
    the scope stack and returned the enclosing *method* FQN (which IS in
    ``known_fqns``) rather than the enclosing *class* FQN, so ``walk_mro``
    was invoked with a method key, found no bases, and the call was logged as
    ``self_method_unresolved``.
    """
    root = _make_package(tmp_path, "pkg", {
        "base.py": (
            "from abc import ABC, abstractmethod\n"
            "\n"
            "class LLMAgent(ABC):\n"
            "    @abstractmethod\n"
            "    def call_llm(self, prompt: str) -> str: ...\n"
        ),
        "writer.py": (
            "from pkg.base import LLMAgent\n"
            "\n"
            "class ContractChapterWriter(LLMAgent):\n"
            "    def call_llm(self, prompt: str) -> str:\n"
            "        return 'answer'\n"
            "    def write_chapter(self, text: str) -> str:\n"
            "        def _inner_helper() -> str:\n"
            "            return self.call_llm(text)\n"
            "        return _inner_helper()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # The nested helper must resolve self.call_llm to the subclass's own override.
    caller = "pkg.writer.ContractChapterWriter.write_chapter._inner_helper"
    assert caller in raw, f"expected caller key {caller!r} in raw; got {sorted(raw)}"
    assert "pkg.writer.ContractChapterWriter.call_llm" in raw[caller], (
        f"expected self.call_llm to resolve; callees were {raw[caller]}"
    )


def test_self_method_inherited_in_nested_fn_resolves_to_base(tmp_path: Path) -> None:
    """Regression companion: inherited method (not overridden) inside nested fn.

    Same topology but ``call_llm`` is only defined on the base, not overridden
    in the subclass, so the MRO walker must chase into the base class.
    """
    root = _make_package(tmp_path, "pkg", {
        "base.py": (
            "from abc import ABC\n"
            "\n"
            "class LLMAgent(ABC):\n"
            "    def call_llm(self, prompt: str) -> str:\n"
            "        return 'base'\n"
        ),
        "writer.py": (
            "from pkg.base import LLMAgent\n"
            "\n"
            "class ContractChapterWriter(LLMAgent):\n"
            "    def write_chapter(self, text: str) -> str:\n"
            "        def _call_section() -> str:\n"
            "            return self.call_llm(text)\n"
            "        return _call_section()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    caller = "pkg.writer.ContractChapterWriter.write_chapter._call_section"
    assert caller in raw, f"expected caller key {caller!r}; got {sorted(raw)}"
    assert "pkg.base.LLMAgent.call_llm" in raw[caller], (
        f"expected MRO resolution to base; callees were {raw[caller]}"
    )


# ---------------------------------------------------------------------------
# Indirect dispatch
# ---------------------------------------------------------------------------

def test_executor_submit_emits_edge_to_method(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from concurrent.futures import ThreadPoolExecutor\n"
            "\n"
            "class Worker:\n"
            "    def _job(self, i):\n"
            "        return i\n"
            "    def run(self):\n"
            "        with ThreadPoolExecutor() as ex:\n"
            "            fut = ex.submit(self._job, 1)\n"
            "        return fut\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Worker._job" in raw["pkg.mod.Worker.run"]


def test_functools_partial_emits_edge(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from functools import partial\n"
            "\n"
            "def handler(x, y):\n"
            "    return x + y\n"
            "\n"
            "def build():\n"
            "    return partial(handler, 1)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.handler" in raw["pkg.mod.build"]


def test_run_in_executor_uses_second_positional(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import asyncio\n"
            "\n"
            "def blocking():\n"
            "    return 1\n"
            "\n"
            "async def entry():\n"
            "    loop = asyncio.get_event_loop()\n"
            "    return await loop.run_in_executor(None, blocking)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.blocking" in raw["pkg.mod.entry"]


def test_dispatcher_lambda_arg_no_false_edge(tmp_path: Path) -> None:
    """Lambdas/inline expressions must not emit spurious edges."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from concurrent.futures import ThreadPoolExecutor\n"
            "\n"
            "def run():\n"
            "    with ThreadPoolExecutor() as ex:\n"
            "        ex.submit(lambda: 1)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # run should have no in-package callees.
    callees = raw.get("pkg.mod.run", [])
    assert all(not c.startswith("pkg.mod.") or c == "pkg.mod.run"
               for c in callees), callees


def test_dispatcher_call_result_arg_no_false_edge(tmp_path: Path) -> None:
    """`executor.submit(factory(), ...)` — arg is a Call result, not a ref."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from concurrent.futures import ThreadPoolExecutor\n"
            "\n"
            "def factory():\n"
            "    return lambda: 1\n"
            "\n"
            "def run():\n"
            "    with ThreadPoolExecutor() as ex:\n"
            "        ex.submit(factory())\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # run should call factory (the Call itself) but NOT emit factory via
    # dispatcher-arg logic (that branch requires Name/Attribute, not Call).
    # The primary edge from `factory()` already gives us pkg.mod.factory once.
    callees = raw.get("pkg.mod.run", [])
    assert callees.count("pkg.mod.factory") <= 1
