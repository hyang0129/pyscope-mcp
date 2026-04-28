"""Tests for the file_skeleton MCP tool and supporting infrastructure.

Covers:
- CallGraphIndex.file_skeleton() query method
- Skeleton extraction via _extract_skeletons() in the pipeline
- Index round-trip (save/load) with skeletons (version 3)
- SHA staleness detection: fresh / file_changed / file_not_found / file_not_in_index
- Backward compatibility: version 1/2 indexes → index_format_incompatible
- Truncation at 50 symbols
- isError: true for unknown path (also stale: true, file_not_in_index)
"""

from __future__ import annotations

import ast
import hashlib
import json as _json
import textwrap
from pathlib import Path

import pytest

from pyscope_mcp.graph import CallGraphIndex
from conftest import make_nodes
from pyscope_mcp.analyzer.pipeline import _extract_skeletons, _first_def_line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index_with_skeletons(skeletons: dict, file_shas: dict | None = None) -> CallGraphIndex:
    """Build a minimal CallGraphIndex pre-loaded with the given skeletons dict.

    Pass ``file_shas=None`` (the default) to simulate a pre-v3 index.
    Pass ``file_shas={}`` (or a populated dict) to simulate a v3 index.
    """
    return CallGraphIndex.from_nodes("/tmp/test", make_nodes({}), skeletons=skeletons, file_shas=file_shas)


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
    """Happy path: symbols returned; stale is false when file_shas not set (pre-v3 behaviour
    with file_shas=None triggers index_format_incompatible, so we use an empty file_shas
    dict to get a fresh-ish result — actual disk check is covered by dedicated tests)."""
    symbols = [
        _make_symbol("pkg.mod.ClassA", "class", 1),
        _make_symbol("pkg.mod.ClassA.method_x", "method", 5),
        _make_symbol("pkg.mod.helper", "function", 10),
    ]
    # Use a pre-v3 index (file_shas=None) just to verify the base shape still returns results.
    idx = _make_index_with_skeletons({"mod.py": symbols}, file_shas=None)
    result = idx.file_skeleton("mod.py")

    assert result.get("isError") is None or result.get("isError") is False
    assert result["truncated"] is False
    assert result["total"] == 3
    assert len(result["results"]) == 3
    # Pre-v3: stale=True with index_format_incompatible (uniform shape)
    assert result["stale"] is True
    assert result["stale_files"] == []
    assert result["index_stale_reason"] == "index_format_incompatible"
    assert "staleness_info" not in result


def test_file_skeleton_unknown_path_returns_error() -> None:
    """Unknown path returns isError:true, error_reason:'path_not_in_index', stale:false with no stale_action."""
    idx = _make_index_with_skeletons({"mod.py": []}, file_shas={})
    result = idx.file_skeleton("nonexistent/file.py")

    assert result["isError"] is True
    assert result["error_reason"] == "path_not_in_index"
    assert result["stale"] is False
    assert result["stale_files"] == []
    assert "stale_action" not in result
    assert "staleness_info" not in result


def test_file_skeleton_truncation_at_50() -> None:
    """When a file has >50 symbols, truncated=True and results are capped at 50."""
    symbols = [_make_symbol(f"pkg.mod.fn{i}", "function", i) for i in range(60)]
    idx = _make_index_with_skeletons({"large.py": symbols}, file_shas=None)
    result = idx.file_skeleton("large.py")

    assert result["truncated"] is True
    assert result["total"] == 60
    assert len(result["results"]) == 50


def test_file_skeleton_exactly_50_not_truncated() -> None:
    """Exactly 50 symbols: truncated=False (boundary condition)."""
    symbols = [_make_symbol(f"pkg.mod.fn{i}", "function", i) for i in range(50)]
    idx = _make_index_with_skeletons({"exact.py": symbols}, file_shas=None)
    result = idx.file_skeleton("exact.py")

    assert result["truncated"] is False
    assert result["total"] == 50
    assert len(result["results"]) == 50


def test_file_skeleton_51_symbols_truncated() -> None:
    """51 symbols: truncated=True."""
    symbols = [_make_symbol(f"pkg.mod.fn{i}", "function", i) for i in range(51)]
    idx = _make_index_with_skeletons({"mod51.py": symbols}, file_shas=None)
    result = idx.file_skeleton("mod51.py")

    assert result["truncated"] is True
    assert result["total"] == 51
    assert len(result["results"]) == 50


def test_file_skeleton_empty_file() -> None:
    """Empty file (no symbols) returns results=[], truncated=False, total=0."""
    idx = _make_index_with_skeletons({"empty.py": []}, file_shas=None)
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
    idx = CallGraphIndex.from_nodes("/tmp/test", make_nodes({"pkg.mod.MyClass.run": []}), skeletons={"mod.py": symbols})
    out = tmp_path / "index.json"
    idx.save(out)

    loaded = CallGraphIndex.load(out)
    assert loaded.skeletons == {"mod.py": symbols}

    result = loaded.file_skeleton("mod.py")
    assert len(result["results"]) == 2
    assert result["truncated"] is False


def test_save_version_is_current(tmp_path: Path) -> None:
    """Saved index must have the current INDEX_VERSION."""
    from pyscope_mcp.graph import INDEX_VERSION
    idx = CallGraphIndex.from_nodes("/tmp/test", make_nodes({}))
    out = tmp_path / "index.json"
    idx.save(out)
    payload = _json.loads(out.read_text())
    assert payload["version"] == INDEX_VERSION


def test_save_includes_file_shas(tmp_path: Path) -> None:
    """Saved index includes file_shas key."""
    shas = {"mod.py": "abc123"}
    idx = CallGraphIndex.from_nodes("/tmp/test", make_nodes({}), file_shas=shas)
    out = tmp_path / "index.json"
    idx.save(out)
    payload = _json.loads(out.read_text())
    assert "file_shas" in payload
    assert payload["file_shas"] == shas


def test_save_load_roundtrip_file_shas(tmp_path: Path) -> None:
    """file_shas survive save() → load() intact."""
    shas = {"a.py": "deadbeef", "b.py": "cafebabe"}
    idx = CallGraphIndex.from_nodes("/tmp/test", make_nodes({}), file_shas=shas)
    out = tmp_path / "index.json"
    idx.save(out)
    loaded = CallGraphIndex.load(out)
    assert loaded.file_shas == shas


def test_load_version_1_raises(tmp_path: Path) -> None:
    """Version 1 index raises a clear error (no backward compat)."""
    payload = {"version": 1, "root": "/tmp/test", "raw": {}, "skeletons": {"any.py": []}}
    out = tmp_path / "v1_index.json"
    out.write_text(_json.dumps(payload))

    with pytest.raises(ValueError, match="v1"):
        CallGraphIndex.load(out)


def test_load_version_2_raises(tmp_path: Path) -> None:
    """Version 2 index raises a clear error (no backward compat)."""
    payload = {"version": 2, "root": "/tmp/test", "raw": {}, "skeletons": {}}
    out = tmp_path / "v2_index.json"
    out.write_text(_json.dumps(payload))

    with pytest.raises(ValueError, match="v2"):
        CallGraphIndex.load(out)


def test_load_version_3_raises(tmp_path: Path) -> None:
    """Version 3 index raises a clear error (no backward compat)."""
    payload = {"version": 3, "root": "/tmp/test", "raw": {}, "skeletons": {}, "file_shas": {}}
    out = tmp_path / "v3_index.json"
    out.write_text(_json.dumps(payload))

    with pytest.raises(ValueError, match="v3"):
        CallGraphIndex.load(out)


def test_load_version_future_raises(tmp_path: Path) -> None:
    """Unknown future version raises ValueError."""
    payload = {"version": 99, "root": "/tmp/test", "raw": {}, "skeletons": {}, "file_shas": {}}
    out = tmp_path / "v99_index.json"
    out.write_text(_json.dumps(payload))

    with pytest.raises(ValueError, match="v99"):
        CallGraphIndex.load(out)


# ---------------------------------------------------------------------------
# SHA staleness scenarios (unit tests using tmp_path for live files)
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_file_skeleton_stale_sha_match(tmp_path: Path) -> None:
    """Scenario A: v3 index, file on disk matches stored SHA → stale: false."""
    content = b"def foo(): pass\n"
    live_file = tmp_path / "mod.py"
    live_file.write_bytes(content)

    symbols = [_make_symbol("pkg.mod.foo", "function", 1)]
    shas = {"mod.py": _sha256(content)}
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes({}), skeletons={"mod.py": symbols}, file_shas=shas)

    result = idx.file_skeleton("mod.py")
    assert result["stale"] is False
    assert result["stale_files"] == []
    assert "staleness_info" not in result
    assert "stale_action" not in result
    assert len(result["results"]) == 1


def test_file_skeleton_stale_sha_mismatch(tmp_path: Path) -> None:
    """Scenario B: v3 index, file on disk has different content → stale: true, stale_files=[path]."""
    original = b"def foo(): pass\n"
    live_file = tmp_path / "mod.py"
    live_file.write_bytes(b"def foo(): return 42\n")  # different content

    symbols = [_make_symbol("pkg.mod.foo", "function", 1)]
    shas = {"mod.py": _sha256(original)}  # stored hash is of original content
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes({}), skeletons={"mod.py": symbols}, file_shas=shas)

    result = idx.file_skeleton("mod.py")
    assert result["stale"] is True
    assert result["stale_files"] == ["mod.py"]
    assert "build" in result["stale_action"].lower()
    assert "staleness_info" not in result
    # Results are still returned
    assert len(result["results"]) == 1


def test_file_skeleton_stale_file_not_found(tmp_path: Path) -> None:
    """Scenario C: v3 index, file absent from disk → stale: true, stale_files=[path]."""
    symbols = [_make_symbol("pkg.mod.foo", "function", 1)]
    shas = {"mod.py": "somehex"}
    # Don't create the file on disk
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes({}), skeletons={"mod.py": symbols}, file_shas=shas)

    result = idx.file_skeleton("mod.py")
    assert result["stale"] is True
    assert result["stale_files"] == ["mod.py"]
    assert "staleness_info" not in result
    # Results still returned (from index)
    assert len(result["results"]) == 1


def test_file_skeleton_stale_file_not_in_index(tmp_path: Path) -> None:
    """Scenario D: v3 index, path not in skeletons → isError: true, error_reason: 'path_not_in_index', stale: false."""
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes({}), skeletons={}, file_shas={})

    result = idx.file_skeleton("new_mod.py")
    assert result["isError"] is True
    assert result["error_reason"] == "path_not_in_index"
    assert result["stale"] is False
    assert result["stale_files"] == []
    assert "stale_action" not in result
    assert "staleness_info" not in result


def test_file_skeleton_stale_pre_v3_index(tmp_path: Path) -> None:
    """Scenario E: pre-v3 index (file_shas=None) → stale: true, index_format_incompatible."""
    symbols = [_make_symbol("pkg.mod.foo", "function", 1)]
    # file_shas=None simulates loading a v1 or v2 index
    idx = CallGraphIndex.from_nodes(str(tmp_path), make_nodes({}), skeletons={"mod.py": symbols}, file_shas=None)

    result = idx.file_skeleton("mod.py")
    assert result["stale"] is True
    assert result["stale_files"] == []
    assert result["index_stale_reason"] == "index_format_incompatible"
    assert "staleness_info" not in result
    # Results are still returned
    assert len(result["results"]) == 1


# ---------------------------------------------------------------------------
# build_with_report computes file_shas
# ---------------------------------------------------------------------------

def test_build_stores_file_shas(tmp_path: Path) -> None:
    """Integration: build_with_report returns file_shas; hashes match live file bytes."""
    from pyscope_mcp.analyzer.pipeline import build_with_report

    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("# init\n")
    (pkg / "core.py").write_text("def foo(): pass\n")

    _raw, _report, _skeletons, file_shas = build_with_report(str(tmp_path), package="mypkg")

    assert isinstance(file_shas, dict)
    assert len(file_shas) > 0

    # Verify each hash matches the live file bytes
    for rel_path, stored_sha in file_shas.items():
        live_bytes = (tmp_path / rel_path).read_bytes()
        assert _sha256(live_bytes) == stored_sha, f"SHA mismatch for {rel_path}"
