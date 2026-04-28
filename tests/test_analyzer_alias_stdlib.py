"""End-to-end tests for aliased / injected stdlib imports (issue #22).

Covers three scenarios:
1. Import alias:  ``import sys as sys_module; sys_module.exit(0)``
2. Direct import: ``import os.path as op; op.join('a', 'b')``
3. Parameter injection: ``def f(sys_module): sys_module.exit(1)``
   — common testable-CLI pattern where the caller passes ``sys`` as an arg.

All three should land in an accepted bucket (``stdlib_method_call`` or
``calls_resolved_external``) rather than ``attr_chain_unresolved``.

FP guard: a variable whose name ends in ``_module`` but whose stem is NOT a
stdlib module must still land in ``attr_chain_unresolved``.
"""

from __future__ import annotations

from pathlib import Path

from pyscope_mcp.analyzer import build_with_report


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
# Import-alias path: import sys as sys_module
# ---------------------------------------------------------------------------

def test_import_alias_sys_exit_is_not_attr_chain_unresolved(tmp_path: Path) -> None:
    """``import sys as sys_module; sys_module.exit(0)`` must not land in
    ``attr_chain_unresolved``.  Via _resolve_external it becomes
    ``calls_resolved_external``; via classify_miss it becomes
    ``stdlib_method_call`` (both are correct).
    """
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import sys as sys_module\n"
            "\n"
            "def f():\n"
            "    sys_module.exit(0)\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    unresolved = report["pattern_counts"]
    assert "attr_chain_unresolved" not in unresolved or unresolved.get("attr_chain_unresolved", 0) == 0, (
        f"sys_module.exit(0) should not be attr_chain_unresolved; got {unresolved}"
    )


def test_import_alias_os_path_join_is_not_attr_chain_unresolved(tmp_path: Path) -> None:
    """``import os.path as op; op.join('a', 'b')`` must not land in
    ``attr_chain_unresolved``.
    """
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import os.path as op\n"
            "\n"
            "def f(x, y):\n"
            "    return op.join(x, y)\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    unresolved = report["pattern_counts"]
    assert "attr_chain_unresolved" not in unresolved or unresolved.get("attr_chain_unresolved", 0) == 0, (
        f"op.join should not be attr_chain_unresolved; got {unresolved}"
    )


def test_import_alias_json_loads_is_not_attr_chain_unresolved(tmp_path: Path) -> None:
    """``import json as j; j.loads('{}')`` must not land in
    ``attr_chain_unresolved``.
    """
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import json as j\n"
            "\n"
            "def f(s):\n"
            "    return j.loads(s)\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    unresolved = report["pattern_counts"]
    assert "attr_chain_unresolved" not in unresolved or unresolved.get("attr_chain_unresolved", 0) == 0, (
        f"j.loads should not be attr_chain_unresolved; got {unresolved}"
    )


# ---------------------------------------------------------------------------
# Parameter injection: def f(sys_module): sys_module.exit(...)
# — matches STDLIB_MODULES stem + "_module" / "_lib" suffix heuristic
# ---------------------------------------------------------------------------

def test_parameter_injection_sys_module_exit_is_accepted(tmp_path: Path) -> None:
    """``def f(sys_module): sys_module.exit(2)`` — sys_module is a parameter,
    not an import, but the name strongly implies it holds the ``sys`` stdlib
    module.  Must land in ``stdlib_method_call`` (accepted) not
    ``attr_chain_unresolved``.
    """
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def gate(phases, sys_module) -> None:\n"
            "    if not phases:\n"
            "        sys_module.exit(2)\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    accepted = report["summary"]["accepted_counts"]
    pattern_counts = report["pattern_counts"]
    assert accepted.get("stdlib_method_call", 0) >= 1, (
        f"sys_module.exit(2) should be accepted as stdlib_method_call; "
        f"accepted={accepted}, pattern_counts={pattern_counts}"
    )
    assert pattern_counts.get("attr_chain_unresolved", 0) == 0, (
        f"sys_module.exit(2) should not be attr_chain_unresolved; "
        f"pattern_counts={pattern_counts}"
    )


def test_parameter_injection_os_lib_is_accepted(tmp_path: Path) -> None:
    """``def f(os_lib): os_lib.getcwd()`` — ``_lib`` suffix variant."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def f(os_lib):\n"
            "    return os_lib.getcwd()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    accepted = report["summary"]["accepted_counts"]
    assert accepted.get("stdlib_method_call", 0) >= 1, (
        f"os_lib.getcwd() should be accepted as stdlib_method_call; accepted={accepted}"
    )


# ---------------------------------------------------------------------------
# FP guard: non-stdlib_module suffix must NOT false-accept
# ---------------------------------------------------------------------------

def test_fp_guard_non_stdlib_stem_is_attr_chain_unresolved(tmp_path: Path) -> None:
    """``myapp_module.exit()`` — ``myapp`` is not a stdlib module; must not be
    accepted as ``stdlib_method_call``.  Should land in ``attr_chain_unresolved``.
    """
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def f(myapp_module):\n"
            "    myapp_module.exit()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    accepted = report["summary"]["accepted_counts"]
    pattern_counts = report["pattern_counts"]
    assert accepted.get("stdlib_method_call", 0) == 0, (
        f"myapp_module.exit() must NOT be accepted as stdlib_method_call; accepted={accepted}"
    )
    assert pattern_counts.get("attr_chain_unresolved", 0) >= 1, (
        f"myapp_module.exit() should land in attr_chain_unresolved; pattern_counts={pattern_counts}"
    )
