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
