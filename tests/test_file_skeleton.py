"""Tests for the file_skeleton MCP tool and supporting infrastructure.

Covers:
- CallGraphIndex.file_skeleton() query method
- Skeleton extraction via _extract_skeletons() in the pipeline
- Index round-trip (save/load) with skeletons (version 2)
- Backward compatibility: version 1 indexes load with empty skeletons
- Truncation at 50 symbols
- isError: true for unknown path
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from pyscope_mcp.graph import CallGraphIndex
from pyscope_mcp.analyzer.pipeline import _extract_skeletons, _first_def_line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index_with_skeletons(skeletons: dict) -> CallGraphIndex:
    """Build a minimal CallGraphIndex pre-loaded with the given skeletons dict."""
    return CallGraphIndex.from_raw("/tmp/test", {}, skeletons=skeletons)


def _parse(source: str) -> ast.Module:
    return ast.parse(textwrap.dedent(source))


# ---------------------------------------------------------------------------
# _first_def_line
# ---------------------------------------------------------------------------

def test_first_def_line_function() -> None:
    tree = _parse("def foo(x: int, y: str) -> bool: pass")
    node = tree.body[0]
    assert _first_def_line(node) == "def foo(x: int, y: str) -> bool:"


def test_first_def_line_async() -> None:
    tree = _parse("async def bar(a, b=None): pass")
    node = tree.body[0]
    assert _first_def_line(node) == "async def bar(a, b=None):"


def test_first_def_line_class_no_bases() -> None:
    tree = _parse("class MyClass: pass")
    node = tree.body[0]
    assert _first_def_line(node) == "class MyClass:"


def test_first_def_line_class_with_bases() -> None:
    tree = _parse("class Child(Parent, Mixin): pass")
    node = tree.body[0]
    assert _first_def_line(node) == "class Child(Parent, Mixin):"


# ---------------------------------------------------------------------------
# _extract_skeletons
# ---------------------------------------------------------------------------

def test_extract_skeletons_basic(tmp_path: Path) -> None:
    """Extracts top-level function, class, and methods; sorts by lineno."""
    source = textwrap.dedent("""\
        def standalone(x):
            pass

        class MyClass:
            def method_a(self):
                pass

            def method_b(self, value: int) -> str:
                return str(value)
    """)
    file_path = tmp_path / "mymodule.py"
    file_path.write_text(source)
    tree = ast.parse(source, filename=str(file_path))

    parsed = [("pkg.mymodule", tree, file_path, {})]
    skeletons = _extract_skeletons(tmp_path, parsed)

    rel = "mymodule.py"
    assert rel in skeletons
    symbols = skeletons[rel]

    # All 4 symbols: standalone, MyClass, method_a, method_b
    assert len(symbols) == 4

    # Sorted by lineno
    linenos = [s["lineno"] for s in symbols]
    assert linenos == sorted(linenos)

    # Check kinds
    kinds = {s["fqn"].split(".")[-1]: s["kind"] for s in symbols}
    assert kinds["standalone"] == "function"
    assert kinds["MyClass"] == "class"
    assert kinds["method_a"] == "method"
    assert kinds["method_b"] == "method"

    # FQNs rooted at module FQN
    fqns = {s["fqn"] for s in symbols}
    assert "pkg.mymodule.standalone" in fqns
    assert "pkg.mymodule.MyClass" in fqns
    assert "pkg.mymodule.MyClass.method_a" in fqns
    assert "pkg.mymodule.MyClass.method_b" in fqns

    # Signatures are first-def-lines only (no bodies)
    standalone_sym = next(s for s in symbols if s["fqn"].endswith("standalone"))
    assert "pass" not in standalone_sym["signature"]
    assert standalone_sym["signature"].startswith("def standalone")


def test_extract_skeletons_excludes_nested_functions(tmp_path: Path) -> None:
    """Nested functions inside other functions are not included."""
    source = textwrap.dedent("""\
        def outer():
            def inner():
                pass
            return inner
    """)
    file_path = tmp_path / "nested.py"
    file_path.write_text(source)
    tree = ast.parse(source)

    parsed = [("pkg.nested", tree, file_path, {})]
    skeletons = _extract_skeletons(tmp_path, parsed)

    symbols = skeletons.get("nested.py", [])
    fqns = {s["fqn"] for s in symbols}
    assert "pkg.nested.outer" in fqns
    assert "pkg.nested.outer.inner" not in fqns  # inner must be excluded


def test_extract_skeletons_empty_file(tmp_path: Path) -> None:
    """An empty module produces an empty skeleton list (not an error)."""
    file_path = tmp_path / "empty.py"
    file_path.write_text("")
    tree = ast.parse("")

    parsed = [("pkg.empty", tree, file_path, {})]
    skeletons = _extract_skeletons(tmp_path, parsed)
    assert skeletons.get("empty.py") == []


# ---------------------------------------------------------------------------
# CallGraphIndex.file_skeleton()
# ---------------------------------------------------------------------------

def _make_symbol(fqn: str, kind: str, lineno: int) -> dict:
    return {"fqn": fqn, "kind": kind, "signature": f"def {fqn.split('.')[-1]}():", "lineno": lineno}


def test_file_skeleton_returns_symbols() -> None:
    symbols = [
        _make_symbol("pkg.mod.ClassA", "class", 1),
        _make_symbol("pkg.mod.ClassA.method_x", "method", 5),
        _make_symbol("pkg.mod.helper", "function", 10),
    ]
    idx = _make_index_with_skeletons({"mod.py": symbols})
    result = idx.file_skeleton("mod.py")

    assert result.get("isError") is None or result.get("isError") is False
    assert result["truncated"] is False
    assert result["total"] == 3
    assert len(result["results"]) == 3


def test_file_skeleton_unknown_path_returns_error() -> None:
    idx = _make_index_with_skeletons({"mod.py": []})
    result = idx.file_skeleton("nonexistent/file.py")

    assert result["isError"] is True
    assert "rebuild" in result["message"].lower() or "build" in result["message"].lower()


def test_file_skeleton_truncation_at_50() -> None:
    """When a file has >50 symbols, truncated=True and results are capped at 50."""
    symbols = [_make_symbol(f"pkg.mod.fn{i}", "function", i) for i in range(60)]
    idx = _make_index_with_skeletons({"large.py": symbols})
    result = idx.file_skeleton("large.py")

    assert result["truncated"] is True
    assert result["total"] == 60
    assert len(result["results"]) == 50


def test_file_skeleton_exactly_50_not_truncated() -> None:
    """Exactly 50 symbols: truncated=False (boundary condition)."""
    symbols = [_make_symbol(f"pkg.mod.fn{i}", "function", i) for i in range(50)]
    idx = _make_index_with_skeletons({"exact.py": symbols})
    result = idx.file_skeleton("exact.py")

    assert result["truncated"] is False
    assert result["total"] == 50
    assert len(result["results"]) == 50


def test_file_skeleton_51_symbols_truncated() -> None:
    """51 symbols: truncated=True."""
    symbols = [_make_symbol(f"pkg.mod.fn{i}", "function", i) for i in range(51)]
    idx = _make_index_with_skeletons({"mod51.py": symbols})
    result = idx.file_skeleton("mod51.py")

    assert result["truncated"] is True
    assert result["total"] == 51
    assert len(result["results"]) == 50


def test_file_skeleton_empty_file() -> None:
    """Empty file (no symbols) returns results=[], truncated=False, total=0."""
    idx = _make_index_with_skeletons({"empty.py": []})
    result = idx.file_skeleton("empty.py")

    assert result["truncated"] is False
    assert result["total"] == 0
    assert result["results"] == []


# ---------------------------------------------------------------------------
# Index save/load round-trip (version 2)
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_with_skeletons(tmp_path: Path) -> None:
    """Skeleton data survives save/load intact."""
    symbols = [
        _make_symbol("pkg.mod.MyClass", "class", 1),
        _make_symbol("pkg.mod.MyClass.run", "method", 5),
    ]
    idx = CallGraphIndex.from_raw("/tmp/test", {"pkg.mod.MyClass.run": []}, skeletons={"mod.py": symbols})
    out = tmp_path / "index.json"
    idx.save(out)

    loaded = CallGraphIndex.load(out)
    assert loaded.skeletons == {"mod.py": symbols}

    result = loaded.file_skeleton("mod.py")
    assert len(result["results"]) == 2
    assert result["truncated"] is False


def test_save_version_is_2(tmp_path: Path) -> None:
    """Saved index must have version=2."""
    import json as _json
    idx = CallGraphIndex.from_raw("/tmp/test", {})
    out = tmp_path / "index.json"
    idx.save(out)
    payload = _json.loads(out.read_text())
    assert payload["version"] == 2


def test_load_version_1_backward_compat(tmp_path: Path) -> None:
    """Version 1 index loads successfully with empty skeletons dict."""
    import json as _json
    payload = {"version": 1, "root": "/tmp/test", "raw": {}}
    out = tmp_path / "v1_index.json"
    out.write_text(_json.dumps(payload))

    loaded = CallGraphIndex.load(out)
    assert loaded.skeletons == {}
    result = loaded.file_skeleton("any.py")
    assert result["isError"] is True
