"""Unit tests for M2 analyzer features: _build_import_table, _EdgeVisitor, build_raw edges."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pyscope_mcp.analyzer import (
    _build_import_table,
    _EdgeVisitor,
    _collect_defs,
    build_raw,
)


# ---------------------------------------------------------------------------
# _build_import_table
# ---------------------------------------------------------------------------


def test_import_table_absolute() -> None:
    """import foo.bar and import baz as b produce correct table entries."""
    source = "import foo.bar\nimport baz as b\n"
    tree = ast.parse(source)
    table = _build_import_table(tree, "mypkg.mod")

    # import foo.bar -> prefix entries "foo" and "foo.bar"
    assert table["foo"] == "foo"
    assert table["foo.bar"] == "foo.bar"
    # import baz as b -> alias entry only
    assert table["b"] == "baz"
    # "baz" itself must NOT appear as a key (only alias is registered)
    assert "baz" not in table


def test_import_table_from_import() -> None:
    """from foo.bar import baz and from foo.bar import qux as q."""
    source = "from foo.bar import baz\nfrom foo.bar import qux as q\n"
    tree = ast.parse(source)
    table = _build_import_table(tree, "mypkg.mod")

    assert table == {"baz": "foo.bar.baz", "q": "foo.bar.qux"}


# ---------------------------------------------------------------------------
# build_raw — edge extraction via real on-disk packages
# ---------------------------------------------------------------------------


def _make_package(tmp_path: Path, pkg_name: str, files: dict[str, str]) -> Path:
    """Create a package directory with __init__.py and given files."""
    pkg = tmp_path / pkg_name
    pkg.mkdir()
    if "__init__.py" not in files:
        (pkg / "__init__.py").write_text("")
    for rel, content in files.items():
        target = pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


def test_top_level_function_calls_sibling(tmp_path: Path) -> None:
    """main() calls helper() in same module — expect one intra-module edge."""
    root = _make_package(
        tmp_path,
        "mypkg",
        {
            "mod.py": (
                "def helper(): pass\n"
                "def main():\n"
                "    helper()\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    assert "mypkg.mod.main" in result
    assert result["mypkg.mod.main"] == ["mypkg.mod.helper"]


def test_call_via_imported_module(tmp_path: Path) -> None:
    """from mypkg.utils import helper; run() calls helper() -> resolved edge."""
    root = _make_package(
        tmp_path,
        "mypkg",
        {
            "utils.py": "def helper(): pass\n",
            "main.py": (
                "from mypkg.utils import helper\n"
                "def run():\n"
                "    helper()\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    assert "mypkg.main.run" in result
    assert result["mypkg.main.run"] == ["mypkg.utils.helper"]


def test_call_via_module_attribute(tmp_path: Path) -> None:
    """import mypkg.utils; run() calls mypkg.utils.helper() -> resolved edge."""
    root = _make_package(
        tmp_path,
        "mypkg",
        {
            "utils.py": "def helper(): pass\n",
            "main.py": (
                "import mypkg.utils\n"
                "def run():\n"
                "    mypkg.utils.helper()\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    assert "mypkg.main.run" in result
    assert result["mypkg.main.run"] == ["mypkg.utils.helper"]


def test_stdlib_call_dropped(tmp_path: Path) -> None:
    """Calls to os.path.join and print must not appear as callee edges."""
    root = _make_package(
        tmp_path,
        "mypkg",
        {
            "mod.py": (
                "import os.path\n"
                "def work():\n"
                "    result = os.path.join('a', 'b')\n"
                "    print(result)\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    # Flatten all callee lists to check no stdlib leaks
    all_callees: list[str] = [c for callees in result.values() for c in callees]
    assert not any("os" in c for c in all_callees), (
        f"stdlib callee leaked into graph: {all_callees}"
    )
    assert not any("print" in c for c in all_callees), (
        f"builtin print leaked into graph: {all_callees}"
    )


def test_no_edges_for_unresolved(tmp_path: Path) -> None:
    """Calling an undefined name mystery() must not emit an edge or raise."""
    root = _make_package(
        tmp_path,
        "mypkg",
        {
            "mod.py": (
                "def work():\n"
                "    mystery()\n"
            ),
        },
    )
    # Must not raise
    result = build_raw(root, "mypkg")

    all_callees: list[str] = [c for callees in result.values() for c in callees]
    assert "mystery" not in all_callees
    assert not any("mystery" in c for c in all_callees)


def test_determinism(tmp_path: Path) -> None:
    """Running build_raw twice on the same package produces identical output."""
    root = _make_package(
        tmp_path,
        "mypkg",
        {
            "utils.py": "def helper(): pass\n",
            "main.py": (
                "from mypkg.utils import helper\n"
                "def run():\n"
                "    helper()\n"
                "def also_run():\n"
                "    helper()\n"
            ),
        },
    )
    result1 = build_raw(root, "mypkg")
    result2 = build_raw(root, "mypkg")

    assert result1 == result2
    # Verify the lists are sorted (determinism contract)
    for callees in result1.values():
        assert callees == sorted(callees)
