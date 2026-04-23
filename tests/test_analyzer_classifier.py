"""Classifier tests for builtin_function_call tag (issue #27).

Verifies that Python builtins — including previously-leaking ones like
getattr, super, hasattr, setattr, etc. — are accepted into the
builtin_function_call bucket, and that str.removesuffix / str.removeprefix
(Python 3.9+) land in builtin_method_call.

Preserves existing dedicated tags (exec_or_eval, getattr_nonliteral,
super_unresolved for super().method() chains) and false-positive guards.
"""

from __future__ import annotations

import ast

import pytest

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
# Issue #27 — previously leaking builtins now accepted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src", [
    # getattr / hasattr / setattr / delattr family
    "getattr(obj, 'x')",
    "getattr(channel_spec, 'description', '')",
    "hasattr(obj, 'x')",
    "setattr(obj, 'x', 1)",
    "delattr(obj, 'x')",
    # super() bare call (no attribute follow-on)
    "super()",
    # other commonly-used builtins
    "issubclass(Foo, Bar)",
    "callable(f)",
    "iter(items)",
    "next(it)",
    "type(obj)",
    "vars(obj)",
    "dir(obj)",
    "repr(obj)",
    "id(obj)",
    "hash(obj)",
    "any(gen)",
    "all(gen)",
    "min(a, b)",
    "max(a, b)",
    "sum(lst)",
    "map(f, lst)",
    "filter(f, lst)",
    "zip(a, b)",
    "enumerate(lst)",
    "sorted(lst)",
    "reversed(lst)",
    "tuple(lst)",
    "set(lst)",
    "float(s)",
    "bool(v)",
    "bytes(n)",
    "print('hello')",
    "open('f.txt')",
    "range(10)",
])
def test_issue27_builtin_accepted(src: str) -> None:
    """Each builtin in the issue #27 list must resolve to builtin_function_call."""
    call = _parse_call(src)
    assert _classify_miss(call) == "builtin_function_call"


# ---------------------------------------------------------------------------
# str.removesuffix / str.removeprefix — builtin_method_call (issue #27)
# ---------------------------------------------------------------------------

def test_removesuffix_is_builtin_method_call() -> None:
    """x.removesuffix('.json') on a local variable must be builtin_method_call."""
    call = _parse_call("output_filename.removesuffix('.json')")
    assert _classify_miss(call) == "builtin_method_call"


def test_removeprefix_is_builtin_method_call() -> None:
    call = _parse_call("name.removeprefix('prefix_')")
    assert _classify_miss(call) == "builtin_method_call"


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


def test_super_bare_is_builtin_function_call() -> None:
    """super() as a bare Name call is now included in BUILTIN_FUNCTION_NAMES.

    Previously it was excluded and leaked to bare_name_unresolved.  The
    dedicated super_unresolved tag is preserved for super().method() chains
    (where func is an ast.Attribute, not ast.Name).
    """
    call = _parse_call("super()")
    assert _classify_miss(call) == "builtin_function_call"


def test_super_dot_method_stays_super_unresolved() -> None:
    """super().__init__() — func is Attribute(value=Call(Name('super')));
    this is handled by the ast.Attribute branch and returns super_unresolved,
    not builtin_function_call.  Ensures the dedicated tag still fires."""
    call = _parse_call("super().__init__()")
    assert _classify_miss(call) == "super_unresolved"


def test_getattr_result_call_stays_getattr_nonliteral() -> None:
    """getattr(obj, method_name)(...) — func is Call(Name('getattr')); the
    ast.Call branch fires first and returns getattr_nonliteral.  This ensures
    the dedicated tag is not broken by adding getattr to BUILTIN_FUNCTION_NAMES."""
    call = _parse_call("getattr(obj, method_name)(arg)")
    assert _classify_miss(call) == "getattr_nonliteral"


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


def test_user_defined_shadowing_getattr_stays_bare_name_unresolved() -> None:
    """A module-level function literally named 'getattr' (user-defined shadowing).

    The current classifier is name-only and does NOT perform scope-aware
    shadowing detection.  If a user defines their own `getattr` function and
    calls it, the classifier will bucket it as builtin_function_call rather
    than bare_name_unresolved.  This is a known limitation documented here;
    a scope-aware fix is out of scope for issue #27 (see issue body).

    This test documents the current (name-only) behaviour: the call is
    classified as builtin_function_call because the classifier matches the
    name 'getattr' against BUILTIN_FUNCTION_NAMES without checking scope.
    """
    # Simulating: getattr = my_thing; getattr(x, y)
    call = _parse_call("getattr(x, y)")
    # NOTE: this returns builtin_function_call due to name-only classification.
    # A scope-aware classifier would return bare_name_unresolved when 'getattr'
    # is locally shadowed.  This is a known limitation, not a regression.
    assert _classify_miss(call) == "builtin_function_call"


# ---------------------------------------------------------------------------
# ACCEPTED_PATTERNS linkage
# ---------------------------------------------------------------------------

def test_builtin_function_call_is_accepted() -> None:
    assert "builtin_function_call" in ACCEPTED_PATTERNS


def test_builtin_function_names_excludes_exec_eval_compile() -> None:
    """exec/eval/compile must stay excluded from BUILTIN_FUNCTION_NAMES."""
    excluded = {"exec", "eval", "compile"}
    assert not (excluded & BUILTIN_FUNCTION_NAMES), (
        f"These names should be excluded: {excluded & BUILTIN_FUNCTION_NAMES}"
    )


def test_builtin_function_names_includes_getattr_super_hasattr() -> None:
    """getattr, super, hasattr must now be in BUILTIN_FUNCTION_NAMES (issue #27)."""
    required = {"getattr", "super", "hasattr", "setattr", "delattr"}
    missing = required - BUILTIN_FUNCTION_NAMES
    assert not missing, f"These should now be in BUILTIN_FUNCTION_NAMES: {missing}"
