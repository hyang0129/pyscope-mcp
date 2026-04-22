"""Pure resolver helpers operating on AST nodes + static project facts.

Handlers here take AST nodes plus a `ResolveCtx` (or equivalent static facts)
and return an FQN string or None. They do not mutate state and do not know
about MissLog. The visitor composes them.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


def attr_chain(node: ast.expr) -> list[str] | None:
    """Flatten a chain of ast.Attribute/ast.Name into a dotted list, or None."""
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


# ---------------------------------------------------------------------------
# Class hierarchy (MRO) resolution
# ---------------------------------------------------------------------------

def walk_mro(
    class_fqn: str,
    method: str,
    class_bases: dict[str, list[str]],
    known_fqns: set[str],
    _seen: frozenset[str] = frozenset(),
) -> str | None:
    """Walk the class hierarchy depth-first, left-to-right, looking for
    the first in-package ancestor whose `{ancestor}.{method}` is a known FQN.

    Returns the resolved FQN or None. Cycles are broken via `_seen`.
    External (non-in-package) bases are silently skipped — they don't define
    in-package methods.
    """
    if class_fqn in _seen:
        return None
    seen = _seen | {class_fqn}
    for base in class_bases.get(class_fqn, []):
        if base not in known_fqns:
            # External base; can't chase it, but a later in-package base might hit.
            continue
        candidate = f"{base}.{method}"
        if candidate in known_fqns:
            return candidate
        # Recurse into grandparents.
        deeper = walk_mro(base, method, class_bases, known_fqns, seen)
        if deeper is not None:
            return deeper
    return None


# ---------------------------------------------------------------------------
# Indirect dispatch
# ---------------------------------------------------------------------------

# Dispatchers whose first positional argument is a callable reference that
# will be invoked later. Matching is by trailing attribute name — receiver
# type is usually un-inferrable statically.
#
# Not included (intentional):
#   - asyncio.ensure_future / create_task: first arg is usually a coroutine
#     expression (already-called), not a bare reference.
#   - Generic decorator patterns like @retry(fn): too many false positives.
DISPATCHER_NAMES: frozenset[str] = frozenset({
    "submit",
    "map",
    "starmap",
    "imap",
    "imap_unordered",
    "apply",
    "apply_async",
    "run_in_executor",
    "partial",
})


def is_dispatcher_call(call: ast.Call) -> bool:
    """True if `call` targets a known dispatcher (by trailing name)."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr in DISPATCHER_NAMES
    if isinstance(func, ast.Name):
        return func.id in DISPATCHER_NAMES
    return False


def dispatcher_callable_arg(call: ast.Call) -> ast.expr | None:
    """Return the callable-reference argument of a dispatcher call, or None.

    Convention: the callable is the first positional argument for every
    dispatcher in `DISPATCHER_NAMES`. `loop.run_in_executor(executor, fn, ...)`
    is the one oddball — its callable is the *second* arg. We handle that
    specially: if the trailing name is `run_in_executor` and there are at
    least two positional args, return args[1]; otherwise args[0].
    """
    if not call.args:
        return None
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else (
        func.id if isinstance(func, ast.Name) else None
    )
    if name == "run_in_executor":
        if len(call.args) >= 2:
            return call.args[1]
        return None
    return call.args[0]


# ---------------------------------------------------------------------------
# ResolveCtx — static facts threaded into visitor/resolvers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolveCtx:
    """Static facts about the project + current file, passed to resolvers."""

    module_fqn: str
    import_table: dict[str, str]
    known_fqns: set[str]
    class_bases: dict[str, list[str]]
    known_classes: set[str] = None  # type: ignore[assignment]
    # {class_fqn: {attr_name: inferred_class_fqn}} — populated in pass 1
    self_attr_types: dict[str, dict[str, str]] = None  # type: ignore[assignment]
    # {func_fqn: {var_name: class_fqn}} — populated in pass 1 by collect_local_var_types
    local_types: dict[str, dict[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Default to empty set/dict so callers that don't supply it still work.
        if self.known_classes is None:
            object.__setattr__(self, "known_classes", set())
        if self.self_attr_types is None:
            object.__setattr__(self, "self_attr_types", {})
        if self.local_types is None:
            object.__setattr__(self, "local_types", {})


# ---------------------------------------------------------------------------
# Constructor-result type inference
# ---------------------------------------------------------------------------

def infer_call_class_type(call: ast.Call, ctx: "ResolveCtx") -> str | None:
    """If `call` is a constructor call to an in-package class, return its FQN.

    Mirrors _infer_type Case 2 from discovery.py but takes a ResolveCtx.
    Returns None for non-class callables (functions, unknowns, builtins).
    Only returns the FQN if it is in ctx.known_classes (not just known_fqns —
    must be a class, not a function).
    """
    func = call.func
    candidate: str | None = None

    if isinstance(func, ast.Name):
        name = func.id
        if name in ctx.import_table:
            candidate = ctx.import_table[name]
        else:
            candidate = f"{ctx.module_fqn}.{name}"
    else:
        chain = attr_chain(func)
        if chain is not None:
            # Longest-prefix lookup against import table.
            for prefix_len in range(len(chain) - 1, 0, -1):
                prefix = ".".join(chain[:prefix_len])
                if prefix in ctx.import_table:
                    base_fqn = ctx.import_table[prefix]
                    remainder = chain[prefix_len:]
                    candidate = ".".join([base_fqn] + remainder)
                    break
            else:
                candidate = ".".join(chain)

    if candidate is None:
        return None
    # Must be a known class, not just any known FQN (no functions/modules).
    if candidate in ctx.known_classes:
        return candidate
    return None


# ---------------------------------------------------------------------------
# self.<attr>.<method>() resolution via __init__ type tracking
# ---------------------------------------------------------------------------

def resolve_self_attr_method(
    attr_name: str,
    method: str,
    class_fqn: str,
    ctx: ResolveCtx,
) -> str | None:
    """Resolve ``self.<attr>.<method>(...)`` inside a method of ``class_fqn``.

    1. Look up ``attr_name`` in ``self_attr_types[class_fqn]``.
    2. Walk MRO of the inferred attribute class to find ``method``.
    3. Return the FQN or None (silent on any miss).
    """
    attr_class = ctx.self_attr_types.get(class_fqn, {}).get(attr_name)
    if attr_class is None:
        return None
    # Direct hit first (avoids MRO traversal overhead).
    candidate = f"{attr_class}.{method}"
    if candidate in ctx.known_fqns:
        return candidate
    return walk_mro(attr_class, method, ctx.class_bases, ctx.known_fqns)


# ---------------------------------------------------------------------------
# Local-variable type resolution
# ---------------------------------------------------------------------------

def resolve_local_var_method(
    node: ast.Attribute,
    enclosing_func_fqn: str,
    ctx: ResolveCtx,
) -> str | None:
    """For ``var.method`` attribute-chain calls where ``var`` is a local variable
    statically bound to an in-package class via ``ctx.local_types``.

    1. Extract the variable name from ``node.value`` (must be a bare ``ast.Name``).
    2. Look up the class FQN in ``ctx.local_types[enclosing_func_fqn]``.
    3. Walk MRO to find the method. Return FQN or None.

    Returns None silently on any failure. Does not mutate state.
    """
    if not isinstance(node.value, ast.Name):
        return None
    var_name = node.value.id
    method = node.attr
    func_vars = ctx.local_types.get(enclosing_func_fqn)
    if func_vars is None:
        return None
    class_fqn = func_vars.get(var_name)
    if class_fqn is None:
        return None
    candidate = f"{class_fqn}.{method}"
    if candidate in ctx.known_fqns:
        return candidate
    return walk_mro(class_fqn, method, ctx.class_bases, ctx.known_fqns)


def resolve_call_result_method(
    node: ast.Attribute,
    ctx: ResolveCtx,
) -> str | None:
    """For ``ClassName(...).method`` chains where the Call's func resolves to an
    in-package class constructor; walk_mro from that class.

    This covers the ``ClassName(...).method()`` pattern using the same
    ``infer_call_class_type`` machinery but exposed as a standalone resolver
    for use as a late fallback in the visitor.

    Returns None silently on any failure. Does not mutate state.
    """
    if not isinstance(node.value, ast.Call):
        return None
    inner_call = node.value
    # super() is handled by a separate resolver; skip it here.
    if isinstance(inner_call.func, ast.Name) and inner_call.func.id == "super":
        return None
    class_fqn = infer_call_class_type(inner_call, ctx)
    if class_fqn is None:
        return None
    method = node.attr
    candidate = f"{class_fqn}.{method}"
    if candidate in ctx.known_fqns:
        return candidate
    return walk_mro(class_fqn, method, ctx.class_bases, ctx.known_fqns)
