"""C2: call-on-constructor-result handler.

Pattern: ClassName(...).method() where the receiver is a constructor call
to an in-package class. The analyzer should emit an edge to the method,
walking MRO if the method is inherited.
"""

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
# Positive cases
# ---------------------------------------------------------------------------

def test_ctor_result_same_module_direct_method(tmp_path: Path) -> None:
    """Foo(a, b).run() where Foo is defined in same module → edge to pkg.mod.Foo.run."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Foo:\n"
            "    def run(self, a, b):\n"
            "        return a + b\n"
            "\n"
            "def entry(a, b):\n"
            "    return Foo(a, b).run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Foo.run" in raw["pkg.mod.entry"]


def test_ctor_result_inherited_method_via_mro(tmp_path: Path) -> None:
    """Foo().inherited() where inherited is only on Foo's parent → edge via MRO."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Parent:\n"
            "    def inherited(self):\n"
            "        return 'base'\n"
            "\n"
            "class Foo(Parent):\n"
            "    pass\n"
            "\n"
            "def entry():\n"
            "    return Foo().inherited()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Parent.inherited" in raw["pkg.mod.entry"]


def test_ctor_result_cross_module_import(tmp_path: Path) -> None:
    """from pkg.mod import Foo; Foo().run() from a sibling module → edge to pkg.mod.Foo.run."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Foo:\n"
            "    def run(self):\n"
            "        return 42\n"
        ),
        "caller.py": (
            "from pkg.mod import Foo\n"
            "\n"
            "def entry():\n"
            "    return Foo().run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Foo.run" in raw["pkg.caller.entry"]


def test_ctor_result_aliased_module(tmp_path: Path) -> None:
    """import pkg.mod as m; m.Foo().run() → edge to pkg.mod.Foo.run."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Foo:\n"
            "    def run(self):\n"
            "        return 42\n"
        ),
        "caller.py": (
            "import pkg.mod as m\n"
            "\n"
            "def entry():\n"
            "    return m.Foo().run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Foo.run" in raw["pkg.caller.entry"]


def test_ctor_result_nested_call_in_args(tmp_path: Path) -> None:
    """Foo(bar()).run() — outer .run resolves to Foo.run even though bar() is in args."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Foo:\n"
            "    def run(self):\n"
            "        return 1\n"
            "\n"
            "def bar():\n"
            "    return 99\n"
            "\n"
            "def entry():\n"
            "    return Foo(bar()).run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Foo.run" in raw["pkg.mod.entry"]


# ---------------------------------------------------------------------------
# Negative guards
# ---------------------------------------------------------------------------

def test_function_call_result_no_false_edge(tmp_path: Path) -> None:
    """some_function().run() where some_function is a function, not a class → no in-pkg edge."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def some_function():\n"
            "    return object()\n"
            "\n"
            "def entry():\n"
            "    return some_function().run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.entry", [])
    # some_function call itself may be resolved, but .run() on its result must not be
    assert "pkg.mod.some_function.run" not in callees
    # No edge like pkg.mod.Foo.run or similar
    assert not any(c.endswith(".run") for c in callees)


def test_dict_builtin_no_in_package_edge(tmp_path: Path) -> None:
    """dict(x=1).items() → no in-package edge; items continues to route through builtin."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def entry():\n"
            "    return dict(x=1).items()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.entry", [])
    assert not any(c.startswith("pkg.") for c in callees)


def test_external_class_no_edge(tmp_path: Path) -> None:
    """ExternalClass().method() where ExternalClass is from a third-party package → no edge."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from some_external_lib import ExternalClass\n"
            "\n"
            "def entry():\n"
            "    return ExternalClass().do_something()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.entry", [])
    assert not any(c.startswith("pkg.") for c in callees)


def test_unknown_name_call_result_no_edge(tmp_path: Path) -> None:
    """unknown_name().method() → no edge."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def entry():\n"
            "    return unknown_name().method()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.entry", [])
    assert not any(c.startswith("pkg.") for c in callees)


def test_self_method_result_no_false_edge(tmp_path: Path) -> None:
    """self.thing().method() — inner call is a method call, not a constructor → no false edge."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Foo:\n"
            "    def thing(self):\n"
            "        return object()\n"
            "    def entry(self):\n"
            "        return self.thing().method()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.Foo.entry", [])
    # self.thing() itself is an in-pkg call, but .method() on its result must not emit
    assert "pkg.mod.Foo.thing" in callees  # the self.thing() call resolves
    assert not any(c.endswith(".method") for c in callees)


# ---------------------------------------------------------------------------
# Regression: super() must still fire first
# ---------------------------------------------------------------------------

def test_super_init_still_resolves(tmp_path: Path) -> None:
    """Regression: super().__init__() still resolves correctly (super takes priority)."""
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


def test_super_cross_module_still_resolves(tmp_path: Path) -> None:
    """Regression: super().greet() across modules still resolves."""
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


def test_self_method_inherited_still_resolves(tmp_path: Path) -> None:
    """Regression: self.method() with MRO fallback still works."""
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
