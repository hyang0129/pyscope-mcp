"""Classifier tests for stdlib_method_call tag (PR 2).

Verifies that aliased and direct stdlib module calls (sys.exit,
os.path.join, etc.) are accepted without inflating actionable miss counts.
False-positive guard: in-package module aliases must NOT be classified as
stdlib_method_call.
"""

from __future__ import annotations

import ast

from pyscope_mcp.analyzer import _classify_miss
from pyscope_mcp.analyzer.misses import ACCEPTED_PATTERNS, STDLIB_MODULES


def _parse_call(src: str) -> ast.Call:
    tree = ast.parse(src)
    expr = tree.body[0]
    assert isinstance(expr, ast.Expr)
    call = expr.value
    assert isinstance(call, ast.Call)
    return call


# ---------------------------------------------------------------------------
# Aliased import: import sys as sys_module; sys_module.exit(1)
# ---------------------------------------------------------------------------

def test_aliased_sys_exit_is_stdlib_method_call() -> None:
    call = _parse_call("sys_module.exit(1)")
    # import_table: sys_module -> sys
    assert _classify_miss(call, import_table={"sys_module": "sys"}) == "stdlib_method_call"


def test_aliased_os_getcwd_is_stdlib_method_call() -> None:
    call = _parse_call("operating_system.getcwd()")
    assert _classify_miss(call, import_table={"operating_system": "os"}) == "stdlib_method_call"


def test_aliased_json_loads_is_stdlib_method_call() -> None:
    call = _parse_call("js.loads(data)")
    assert _classify_miss(call, import_table={"js": "json"}) == "stdlib_method_call"


# ---------------------------------------------------------------------------
# Plain import: import sys; sys.exit(1)
# ---------------------------------------------------------------------------

def test_plain_sys_exit_is_stdlib_method_call() -> None:
    """import sys → import_table has {"sys": "sys"} or sys not in import_table."""
    call = _parse_call("sys.exit(1)")
    # When sys is in import_table mapping to itself
    assert _classify_miss(call, import_table={"sys": "sys"}) == "stdlib_method_call"


def test_plain_os_path_join_is_stdlib_method_call() -> None:
    """import os; os.path.join(a, b) — chain root 'os' resolves to stdlib."""
    call = _parse_call("os.path.join(a, b)")
    assert _classify_miss(call, import_table={"os": "os"}) == "stdlib_method_call"


def test_direct_sys_without_import_table_entry_is_stdlib_method_call() -> None:
    """sys.exit() where sys is NOT in import_table but is a stdlib name."""
    call = _parse_call("sys.exit(1)")
    # Empty import_table — chain[0]=='sys' is directly in STDLIB_MODULES
    assert _classify_miss(call, import_table={}) == "stdlib_method_call"


def test_plain_logging_getLogger_is_stdlib_method_call() -> None:
    call = _parse_call("logging.getLogger(__name__)")
    assert _classify_miss(call, import_table={"logging": "logging"}) == "stdlib_method_call"


def test_plain_json_dumps_is_stdlib_method_call() -> None:
    call = _parse_call("json.dumps(data)")
    assert _classify_miss(call, import_table={"json": "json"}) == "stdlib_method_call"


# ---------------------------------------------------------------------------
# False-positive guard — in-package alias must NOT become stdlib_method_call
# ---------------------------------------------------------------------------

def test_inpackage_alias_not_stdlib() -> None:
    """import my_pkg as mp; mp.thing() — mp maps to in-package FQN, not stdlib."""
    call = _parse_call("mp.thing()")
    assert _classify_miss(call, import_table={"mp": "my_pkg"}) != "stdlib_method_call"


def test_unknown_root_not_stdlib() -> None:
    """factory.create() with no import_table entry → attr_chain_unresolved."""
    call = _parse_call("factory.create()")
    result = _classify_miss(call, import_table={})
    assert result != "stdlib_method_call"


def test_no_import_table_does_not_crash() -> None:
    """classify_miss without import_table kwarg stays backward-compatible.

    When import_table is None (omitted), the stdlib block is skipped entirely.
    sys.exit(1) then falls through all method-name whitelists ('exit' matches
    none of them) and lands on attr_chain_unresolved.
    """
    call = _parse_call("sys.exit(1)")
    result = _classify_miss(call)
    assert result == "attr_chain_unresolved"


# ---------------------------------------------------------------------------
# ACCEPTED_PATTERNS linkage
# ---------------------------------------------------------------------------

def test_stdlib_method_call_is_accepted() -> None:
    assert "stdlib_method_call" in ACCEPTED_PATTERNS


def test_stdlib_modules_contains_expected_entries() -> None:
    for mod in ("sys", "os", "json", "re", "pathlib", "logging", "asyncio"):
        assert mod in STDLIB_MODULES, f"{mod!r} missing from STDLIB_MODULES"
