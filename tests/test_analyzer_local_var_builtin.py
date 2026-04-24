"""PR 4: local-var handler extended to cover builtin/pathlib sentinel types."""

from __future__ import annotations

from pathlib import Path

from pyscope_mcp.analyzer import build_raw, build_with_report


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


def _accepted(report: dict, pattern: str) -> int:
    return report["summary"]["accepted_counts"].get(pattern, 0)


# ---------------------------------------------------------------------------
# d: dict = {} — AnnAssign with dict annotation
# ---------------------------------------------------------------------------

def test_local_var_ann_assign_dict_get_is_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def f():\n"
            "    d: dict = {}\n"
            "    return d.get('x')\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert _accepted(report, "builtin_method_call") >= 1
    assert report["pattern_counts"].get("attr_chain_unresolved", 0) == 0


# ---------------------------------------------------------------------------
# def f(p: Path): p.mkdir() — param annotation Path
# ---------------------------------------------------------------------------

def test_local_var_param_path_mkdir_is_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from pathlib import Path\n"
            "\n"
            "def f(p: Path):\n"
            "    p.mkdir()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert _accepted(report, "pathlib_method_call") >= 1
    assert report["pattern_counts"].get("attr_chain_unresolved", 0) == 0


# ---------------------------------------------------------------------------
# x = {} — bare dict literal inferred as builtins.dict
# ---------------------------------------------------------------------------

def test_local_var_dict_literal_assign_get_is_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def f():\n"
            "    x = {}\n"
            "    return x.get('y')\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert _accepted(report, "builtin_method_call") >= 1


# ---------------------------------------------------------------------------
# x = [] — list literal inferred as builtins.list
# ---------------------------------------------------------------------------

def test_local_var_list_literal_append_is_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def f():\n"
            "    items = []\n"
            "    items.append(1)\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert _accepted(report, "builtin_method_call") >= 1


# ---------------------------------------------------------------------------
# Path() constructor call in local var
# ---------------------------------------------------------------------------

def test_local_var_path_call_exists_is_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from pathlib import Path\n"
            "\n"
            "def f(s):\n"
            "    p = Path(s)\n"
            "    return p.exists()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert _accepted(report, "pathlib_method_call") >= 1


# ---------------------------------------------------------------------------
# False-positive guards
# ---------------------------------------------------------------------------

def test_fpg_factory_return_type_unknown_no_false_in_package_edge(tmp_path: Path) -> None:
    """x = factory() — factory return type is unknown; x.get('y') must not emit a
    false in-package edge. The call may be accepted by classify_miss (method-name
    heuristic), but it must NOT produce an in-package edge to some fabricated callee.
    """
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def factory():\n"
            "    return {}\n"
            "\n"
            "def f():\n"
            "    x = factory()\n"
            "    return x.get('y')\n"
        ),
    })
    raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    # No in-package callee should be emitted for x.get() — factory() does not
    # record a sentinel for x (it is an in-package Call, not a builtin literal/ctor).
    callees = raw.get("pkg.mod.f", [])
    # The only in-package edge from f() should be factory() itself.
    assert "pkg.mod.factory" in callees
    # No callee like "builtins.dict.get" or similar should appear.
    non_factory = [c for c in callees if c != "pkg.mod.factory"]
    assert non_factory == [], f"Unexpected extra callees: {non_factory}"


def test_fpg_for_loop_var_not_tracked_no_false_in_package_edge(tmp_path: Path) -> None:
    """for p in paths: p.mkdir() — loop vars are not tracked per existing design.
    p.mkdir() must not produce a false in-package edge.
    """
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from pathlib import Path\n"
            "\n"
            "def f(paths):\n"
            "    for p in paths:\n"
            "        p.mkdir()\n"
        ),
    })
    raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    # f() emits no in-package edges — no callee of p.mkdir() should be in raw.
    callees = raw.get("pkg.mod.f", [])
    assert callees == [], f"Expected no in-package edges from f, got: {callees}"
