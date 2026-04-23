"""Tests for cls() / cls.method() resolution inside @classmethod bodies."""

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
# 1. cls(...) → enclosing __init__
# ---------------------------------------------------------------------------

def test_cls_call_resolves_to_enclosing_init(tmp_path: Path) -> None:
    """cls(...) inside a classmethod resolves to the enclosing class __init__."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Widget:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "\n"
            "    @classmethod\n"
            "    def create(cls, x):\n"
            "        return cls(x)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Widget.__init__" in raw["pkg.mod.Widget.create"]


# ---------------------------------------------------------------------------
# 2. cls.from_dict(...) → enclosing class method
# ---------------------------------------------------------------------------

def test_cls_method_resolves_to_enclosing_class_method(tmp_path: Path) -> None:
    """cls.from_dict(...) inside a classmethod resolves to EnclosingClass.from_dict."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Record:\n"
            "    def __init__(self, data):\n"
            "        self.data = data\n"
            "\n"
            "    @classmethod\n"
            "    def from_dict(cls, d):\n"
            "        return cls(d)\n"
            "\n"
            "    @classmethod\n"
            "    def from_json(cls, s):\n"
            "        import json\n"
            "        return cls.from_dict(json.loads(s))\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Record.from_dict" in raw["pkg.mod.Record.from_json"]


# ---------------------------------------------------------------------------
# 3. Subclass cls.parent_method(...) → resolves via MRO
# ---------------------------------------------------------------------------

def test_cls_method_inherits_via_mro(tmp_path: Path) -> None:
    """cls.parent_method() inside subclass classmethod resolves via MRO."""
    root = _make_package(tmp_path, "pkg", {
        "base.py": (
            "class Base:\n"
            "    @classmethod\n"
            "    def build(cls, x):\n"
            "        return cls(x)\n"
            "\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
        ),
        "child.py": (
            "from pkg.base import Base\n"
            "\n"
            "class Child(Base):\n"
            "    @classmethod\n"
            "    def make(cls, x):\n"
            "        return cls.build(x)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.base.Base.build" in raw["pkg.child.Child.make"]


# ---------------------------------------------------------------------------
# 4. Override preference: child defines method → resolves to child version
# ---------------------------------------------------------------------------

def test_cls_method_prefers_child_override(tmp_path: Path) -> None:
    """cls.method() resolves to the child's own override, not the parent's."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Base:\n"
            "    @classmethod\n"
            "    def build(cls, x):\n"
            "        return x\n"
            "\n"
            "class Child(Base):\n"
            "    @classmethod\n"
            "    def build(cls, x):\n"
            "        return x * 2\n"
            "\n"
            "    @classmethod\n"
            "    def make(cls, x):\n"
            "        return cls.build(x)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw["pkg.mod.Child.make"]
    assert "pkg.mod.Child.build" in callees
    assert "pkg.mod.Base.build" not in callees


# ---------------------------------------------------------------------------
# 5. FP guard: plain method with first arg named cls but no @classmethod decorator
# ---------------------------------------------------------------------------

def test_no_edge_plain_method_named_cls_no_decorator(tmp_path: Path) -> None:
    """A plain method whose first arg is named 'cls' (no @classmethod) must NOT resolve."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Thing:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "\n"
            "    def weird(cls, x):\n"
            "        return cls(x)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.Thing.weird", [])
    assert "pkg.mod.Thing.__init__" not in callees


# ---------------------------------------------------------------------------
# 6. FP guard: @classmethod decorator but first arg is not 'cls'
# ---------------------------------------------------------------------------

def test_no_edge_classmethod_first_arg_not_cls(tmp_path: Path) -> None:
    """@classmethod with non-'cls' first arg must NOT trigger cls resolution."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Factory:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "\n"
            "    @classmethod\n"
            "    def create(klass, x):\n"
            "        return klass(x)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.Factory.create", [])
    assert "pkg.mod.Factory.__init__" not in callees


# ---------------------------------------------------------------------------
# 7. FP guard: cls(...) in module-scope function (no enclosing class)
# ---------------------------------------------------------------------------

def test_no_edge_cls_in_module_scope_function(tmp_path: Path) -> None:
    """cls used in a module-level function (not inside any class) must not resolve."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class MyClass:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "\n"
            "def factory(cls, x):\n"
            "    return cls(x)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.factory", [])
    assert "pkg.mod.MyClass.__init__" not in callees


# ---------------------------------------------------------------------------
# 8. cls.a.b(...) 3-part chain → no edge (out of scope)
# ---------------------------------------------------------------------------

def test_no_edge_cls_three_part_chain(tmp_path: Path) -> None:
    """cls.a.b(...) three-part chain is out of scope and must not emit an edge."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Registry:\n"
            "    _store = {}\n"
            "\n"
            "    @classmethod\n"
            "    def register(cls, key, val):\n"
            "        cls._store[key] = val\n"
            "\n"
            "    @classmethod\n"
            "    def do_something(cls):\n"
            "        cls._store.update({'a': 1})\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.Registry.do_something", [])
    # Must not resolve cls._store.update as an in-package edge
    assert not any(c.startswith("pkg.mod.Registry") for c in callees)


# ---------------------------------------------------------------------------
# 9. Local variable 'cls = ...' inside classmethod — documents current behavior
# ---------------------------------------------------------------------------

def test_local_cls_shadow_resolves_to_enclosing_class(tmp_path: Path) -> None:
    """A local variable named 'cls' inside a classmethod shadows the parameter.

    Current behavior (by design): the resolver uses the enclosing function's
    classmethod context, not dataflow, so cls(...) still resolves to the
    enclosing class __init__. This test asserts current behavior and documents
    the known limitation.
    """
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Widget:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "\n"
            "    @classmethod\n"
            "    def create(cls, x):\n"
            "        cls = Widget  # shadows the cls parameter\n"
            "        return cls(x)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # Documented limitation: resolver uses classmethod context, not dataflow.
    # cls(x) still resolves to Widget.__init__ even though cls was rebound.
    assert "pkg.mod.Widget.__init__" in raw["pkg.mod.Widget.create"]
