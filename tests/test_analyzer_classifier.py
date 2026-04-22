"""Classifier tests for builtin_function_call tag (PR 1).

Verifies that Python builtins are accepted without inflating the actionable
miss counts, while preserving existing dedicated tags (exec_or_eval,
super_unresolved) and the false-positive guard (non-builtin bare names stay
bare_name_unresolved).
"""

from __future__ import annotations

import ast

from pyscope_mcp.analyzer import _classify_miss
from pyscope_mcp.analyzer.misses import ACCEPTED_PATTERNS, BUILTIN_FUNCTION_NAMES


def _parse_call(src: str) -> ast.Call:
    tree = ast.parse(src)
    expr = tree.body[0]
    assert isinstance(expr, ast.Expr)
    call = expr.value
    assert isinstance(call, ast.Call)
    return call


# ---------------------------------------------------------------------------
# Positive cases — common builtins must return builtin_function_call
# ---------------------------------------------------------------------------

def test_len_is_builtin_function_call() -> None:
    call = _parse_call("len(x)")
    assert _classify_miss(call) == "builtin_function_call"


def test_str_is_builtin_function_call() -> None:
    call = _parse_call("str(x)")
    assert _classify_miss(call) == "builtin_function_call"


def test_isinstance_is_builtin_function_call() -> None:
    call = _parse_call("isinstance(x, int)")
    assert _classify_miss(call) == "builtin_function_call"


def test_RuntimeError_is_builtin_function_call() -> None:
    call = _parse_call('RuntimeError("m")')
    assert _classify_miss(call) == "builtin_function_call"


def test_list_is_builtin_function_call() -> None:
    call = _parse_call("list(iterable)")
    assert _classify_miss(call) == "builtin_function_call"


def test_dict_is_builtin_function_call() -> None:
    call = _parse_call("dict(a=1)")
    assert _classify_miss(call) == "builtin_function_call"


def test_int_is_builtin_function_call() -> None:
    call = _parse_call("int(s)")
    assert _classify_miss(call) == "builtin_function_call"


# ---------------------------------------------------------------------------
# Preserved dedicated tags — must NOT be rerouted to builtin_function_call
# ---------------------------------------------------------------------------

def test_exec_stays_exec_or_eval() -> None:
    call = _parse_call("exec(code)")
    assert _classify_miss(call) == "exec_or_eval"


def test_eval_stays_exec_or_eval() -> None:
    call = _parse_call("eval(expr)")
    assert _classify_miss(call) == "exec_or_eval"


def test_compile_stays_exec_or_eval() -> None:
    call = _parse_call("compile(src, '<string>', 'exec')")
    assert _classify_miss(call) == "exec_or_eval"


def test_super_bare_stays_bare_name_unresolved() -> None:
    """super() as a bare Name call is excluded from BUILTIN_FUNCTION_NAMES,
    so classify_miss falls through to bare_name_unresolved (not builtin_function_call).
    The visitor resolves super() chains before classify_miss is called; this
    tests the classifier's own behavior when it receives an unresolved super() node."""
    call = _parse_call("super()")
    assert _classify_miss(call) == "bare_name_unresolved"


# ---------------------------------------------------------------------------
# False-positive guard — non-builtin bare names must stay bare_name_unresolved
# ---------------------------------------------------------------------------

def test_non_builtin_stays_bare_name_unresolved() -> None:
    """parse_json(x) is not a builtin — must NOT become builtin_function_call."""
    call = _parse_call("parse_json(x)")
    assert _classify_miss(call) == "bare_name_unresolved"


def test_call_llm_stays_bare_name_unresolved() -> None:
    call = _parse_call("call_llm(prompt)")
    assert _classify_miss(call) == "bare_name_unresolved"


def test_write_json_stays_bare_name_unresolved() -> None:
    call = _parse_call("write_json(path, data)")
    assert _classify_miss(call) == "bare_name_unresolved"


# ---------------------------------------------------------------------------
# ACCEPTED_PATTERNS linkage
# ---------------------------------------------------------------------------

def test_builtin_function_call_is_accepted() -> None:
    assert "builtin_function_call" in ACCEPTED_PATTERNS


def test_builtin_function_names_excludes_exec_eval_compile_super_getattr() -> None:
    excluded = {"exec", "eval", "compile", "super", "getattr"}
    assert not (excluded & BUILTIN_FUNCTION_NAMES), (
        f"These names should be excluded: {excluded & BUILTIN_FUNCTION_NAMES}"
    )
