"""Unit tests for M1 analyzer primitives: _discover_modules, _collect_defs, build_raw."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pyscope_mcp.analyzer import _collect_defs, _discover_modules, build_raw


# ---------------------------------------------------------------------------
# _discover_modules
# ---------------------------------------------------------------------------


def test_discover_modules_flat(tmp_path: Path) -> None:
    """Flat package: __init__ + two sibling modules."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod_a.py").write_text("")
    (pkg / "mod_b.py").write_text("")

    result = _discover_modules(pkg, "mypkg")
    assert set(result.keys()) == {"mypkg", "mypkg.mod_a", "mypkg.mod_b"}


def test_discover_modules_nested(tmp_path: Path) -> None:
    """Nested sub-package: values must be Path objects."""
    pkg = tmp_path / "mypkg"
    sub = pkg / "sub"
    sub.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (sub / "__init__.py").write_text("")
    (sub / "deep.py").write_text("")

    result = _discover_modules(pkg, "mypkg")
    assert set(result.keys()) == {"mypkg", "mypkg.sub", "mypkg.sub.deep"}
    for v in result.values():
        assert isinstance(v, Path), f"Expected Path, got {type(v)}"


# ---------------------------------------------------------------------------
# _collect_defs
# ---------------------------------------------------------------------------


def test_collect_defs_functions() -> None:
    """Top-level functions (sync and async) are collected."""
    source = "def foo(): pass\nasync def bar(): pass\n"
    tree = ast.parse(source)
    result = _collect_defs(tree, "mypkg.mod")
    assert result == {"mypkg.mod.foo", "mypkg.mod.bar"}


def test_collect_defs_class_and_methods() -> None:
    """Class name + one-level-deep methods are all collected."""
    source = (
        "class MyClass:\n"
        "    def method_a(self): pass\n"
        "    def method_b(self): pass\n"
    )
    tree = ast.parse(source)
    result = _collect_defs(tree, "mypkg.mod")
    assert result == {
        "mypkg.mod.MyClass",
        "mypkg.mod.MyClass.method_a",
        "mypkg.mod.MyClass.method_b",
    }


def test_collect_defs_mixed() -> None:
    """File with both top-level functions and a class."""
    source = (
        "def standalone(): pass\n"
        "\n"
        "class Widget:\n"
        "    def render(self): pass\n"
        "    async def update(self): pass\n"
        "\n"
        "async def helper(): pass\n"
    )
    tree = ast.parse(source)
    result = _collect_defs(tree, "mypkg.mod")
    assert result == {
        "mypkg.mod.standalone",
        "mypkg.mod.Widget",
        "mypkg.mod.Widget.render",
        "mypkg.mod.Widget.update",
        "mypkg.mod.helper",
    }


# ---------------------------------------------------------------------------
# build_raw
# ---------------------------------------------------------------------------


def test_build_raw_syntax_error_isolated(tmp_path: Path) -> None:
    """A bad file is skipped; build_raw does not raise and returns {}."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "good.py").write_text("def ok(): pass\n")
    (pkg / "bad.py").write_text("def broken(\n")  # deliberate SyntaxError

    # Must not raise
    result = build_raw(tmp_path, "mypkg")
    assert result == {}


def test_build_raw_returns_empty_m1(tmp_path: Path) -> None:
    """M1 contract: build_raw always returns {} (no edges yet)."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def init_func(): pass\n")
    (pkg / "alpha.py").write_text("def alpha(): pass\n")

    result = build_raw(tmp_path, "mypkg")
    assert result == {}
