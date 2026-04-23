"""Pass 1: discover modules, collect definitions, build project-wide index."""

from __future__ import annotations

import ast
from pathlib import Path

from .imports import build_import_table
from .resolution import attr_chain

# Sentinel FQN strings for builtin/pathlib types that are NOT in known_fqns but
# are meaningful to the visitor for accepted-pattern routing.
SENTINEL_BUILTIN_TYPES: frozenset[str] = frozenset({
    "builtins.dict",
    "builtins.list",
    "builtins.set",
    "builtins.tuple",
    "pathlib.Path",
})

# Annotation names that map to builtin sentinel FQNs.
_BUILTIN_ANNOTATION_NAMES: dict[str, str] = {
    "dict": "builtins.dict",
    "list": "builtins.list",
    "set": "builtins.set",
    "tuple": "builtins.tuple",
    "Dict": "builtins.dict",
    "List": "builtins.list",
    "Set": "builtins.set",
    "Tuple": "builtins.tuple",
}


def _infer_sentinel_from_annotation(
    annotation: ast.expr,
    import_table: dict[str, str],
) -> str | None:
    """Return a sentinel FQN if annotation is a recognised builtin or pathlib.Path type.

    Handles: bare Name (dict, list, set, tuple, Path), and subscripted forms
    like dict[str, Any] (ast.Subscript whose value is a Name).
    """
    target = annotation
    # Unwrap subscript: dict[str, Any] → dict
    if isinstance(target, ast.Subscript):
        target = target.value

    if isinstance(target, ast.Name):
        name = target.id
        if name in _BUILTIN_ANNOTATION_NAMES:
            return _BUILTIN_ANNOTATION_NAMES[name]
        # Path — only if import_table explicitly maps it to pathlib.Path.
        # Do NOT assume bare 'Path' without an explicit import; user code may
        # define its own Path class.
        if import_table.get(name) == "pathlib.Path":
            return "pathlib.Path"
    return None


def _infer_sentinel_from_rhs(
    rhs: ast.expr,
    import_table: dict[str, str],
) -> str | None:
    """Return a sentinel FQN if rhs is a recognised builtin literal or Path() call."""
    # Literal dict: {}
    if isinstance(rhs, ast.Dict):
        return "builtins.dict"
    # Literal list: []
    if isinstance(rhs, ast.List):
        return "builtins.list"
    # Literal set: set() or {1, 2}
    if isinstance(rhs, ast.Set):
        return "builtins.set"
    # Literal tuple: ()
    if isinstance(rhs, ast.Tuple):
        return "builtins.tuple"
    # Constructor call: Path(...), dict(...), list(...), set(...), tuple(...)
    if isinstance(rhs, ast.Call):
        func = rhs.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in _BUILTIN_ANNOTATION_NAMES:
                return _BUILTIN_ANNOTATION_NAMES[name]
            # Path(...) — only if import_table explicitly maps it to pathlib.Path.
            if import_table.get(name) == "pathlib.Path":
                return "pathlib.Path"
    return None


def discover_modules(root: Path, package: str) -> dict[str, Path]:
    """Walk root, return mapping of dotted FQN -> path for every .py file."""
    result: dict[str, Path] = {}
    for py_file in sorted(root.rglob("*.py")):
        rel = py_file.relative_to(root.parent)
        parts = list(rel.parts)
        parts[-1] = parts[-1][:-3]  # strip .py
        if parts[-1] == "__init__":
            parts = parts[:-1]
        fqn = ".".join(parts)
        result[fqn] = py_file
    return result


def collect_nested_defs(
    tree: ast.Module,
    module_fqn: str,
) -> dict[str, dict[str, tuple[str, int]]]:
    """Collect nested-function definitions for bare-name resolution.

    Returns a two-level map::

        {enclosing_fqn: {nested_name: (nested_fqn, def_lineno)}}

    where ``enclosing_fqn`` is the FQN of the *direct* enclosing function (or
    method), using the same plain-dot convention as the visitor's scope stack::

        pkg.mod.outer            → {_table: (pkg.mod.outer._table, lineno)}
        pkg.mod.MyClass.method   → {_helper: (pkg.mod.MyClass.method._helper, lineno)}

    The ``def_lineno`` is the first line of the nested ``def`` statement.  It
    is used by the resolver to enforce Python's name-binding rule: a nested
    function is not in scope before its ``def`` statement executes.

    Only nested *functions* are recorded (not nested classes — see issue #28
    out-of-scope note).  Each function is recorded under its direct enclosing
    function's FQN; the resolver walks outward through the scope chain if the
    calling site is itself a nested function.
    """
    result: dict[str, dict[str, tuple[str, int]]] = {}
    # Walk module-level statements, descending into class and function bodies.
    _collect_nested_defs_toplevel(tree.body, module_fqn, result)
    return result


def _collect_nested_defs_toplevel(
    stmts: list[ast.stmt],
    scope_fqn: str,
    result: dict[str, dict[str, tuple[str, int]]],
) -> None:
    """Walk statements at module or class level, entering function bodies."""
    for stmt in stmts:
        if isinstance(stmt, ast.ClassDef):
            class_fqn = f"{scope_fqn}.{stmt.name}"
            # Descend into class body; methods inside are function scopes
            _collect_nested_defs_toplevel(stmt.body, class_fqn, result)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_fqn = f"{scope_fqn}.{stmt.name}"
            # Scan this function's body for nested defs
            _collect_nested_defs_in_func(stmt.body, func_fqn, result)


def _collect_nested_defs_in_func(
    stmts: list[ast.stmt],
    enclosing_fqn: str,
    result: dict[str, dict[str, tuple[str, int]]],
) -> None:
    """Scan a function body for nested FunctionDefs, recording them under enclosing_fqn.

    Uses the same plain-dot FQN convention as the visitor's scope stack (no
    ``<locals>`` segment).  This keeps nested_defs keys consistent with what
    ``_enclosing_func_fqns`` produces in the visitor.

    Descends into if/for/while/with/try bodies (where a def can appear), but
    does NOT descend into nested function bodies — those are separate scopes.
    """
    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_fqn = f"{enclosing_fqn}.{stmt.name}"
            # Record: bare name → (nested_fqn, def_lineno)
            entries = result.setdefault(enclosing_fqn, {})
            entries[stmt.name] = (nested_fqn, stmt.lineno)
            # Recurse into nested function body as a new scope
            _collect_nested_defs_in_func(stmt.body, nested_fqn, result)

        elif isinstance(stmt, ast.ClassDef):
            # Nested class inside a function — out of scope for issue #28.
            # Recurse into its methods so their nested defs are found.
            class_fqn = f"{enclosing_fqn}.{stmt.name}"
            _collect_nested_defs_toplevel(stmt.body, class_fqn, result)

        elif isinstance(stmt, ast.If):
            _collect_nested_defs_in_func(stmt.body + stmt.orelse, enclosing_fqn, result)
        elif isinstance(stmt, ast.For):
            _collect_nested_defs_in_func(stmt.body + stmt.orelse, enclosing_fqn, result)
        elif isinstance(stmt, ast.While):
            _collect_nested_defs_in_func(stmt.body + stmt.orelse, enclosing_fqn, result)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            _collect_nested_defs_in_func(stmt.body, enclosing_fqn, result)
        elif isinstance(stmt, ast.Try):
            all_stmts = stmt.body + stmt.orelse + stmt.finalbody
            for handler in stmt.handlers:
                all_stmts += handler.body
            _collect_nested_defs_in_func(all_stmts, enclosing_fqn, result)


def collect_defs(tree: ast.Module, module_fqn: str) -> set[str]:
    """Collect top-level defs and one-level-deep methods from a parsed AST."""
    defs: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.add(f"{module_fqn}.{node.name}")
        elif isinstance(node, ast.ClassDef):
            class_fqn = f"{module_fqn}.{node.name}"
            defs.add(class_fqn)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs.add(f"{class_fqn}.{child.name}")
    return defs


def collect_classes(tree: ast.Module, module_fqn: str) -> set[str]:
    """Collect FQNs of top-level class definitions in this module.

    Used to distinguish class FQNs from method FQNs when searching for an
    enclosing class in the scope stack.  Only top-level classes are needed
    because ``collect_defs`` only records one level of nesting (class →
    method); nested classes-inside-methods are not tracked.
    """
    classes: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            classes.add(f"{module_fqn}.{node.name}")
    return classes


def collect_class_bases(
    tree: ast.Module,
    module_fqn: str,
    import_table: dict[str, str],
) -> dict[str, list[str]]:
    """Map `{class_fqn: [base_fqn, ...]}` for top-level classes in this module.

    Base expressions are resolved via the module's import table. Bases that
    are themselves attribute chains (e.g. `mod.Parent`) are resolved via
    longest-prefix lookup. Unresolvable bases are dropped silently — the MRO
    walker skips non-in-package entries anyway.

    Only positional bases are recorded. Keyword metaclass/mixin forms
    (`class C(metaclass=M)`) are ignored.
    """
    result: dict[str, list[str]] = {}
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        class_fqn = f"{module_fqn}.{node.name}"
        resolved: list[str] = []
        for base in node.bases:
            resolved_base = _resolve_base(base, module_fqn, import_table)
            if resolved_base is not None:
                resolved.append(resolved_base)
        result[class_fqn] = resolved
    return result


def collect_self_attr_types(
    tree: ast.Module,
    module_fqn: str,
    import_table: dict[str, str],
    known_fqns: set[str],
) -> dict[str, dict[str, str]]:
    """Map ``{class_fqn: {attr_name: inferred_class_fqn}}`` for top-level classes.

    For each class, inspect ``__init__`` (and ``__post_init__``) for assignments
    of the form ``self.X = Y`` where ``Y``'s type is statically resolvable:

    1. RHS is a direct constructor call: ``self.X = Foo(...)`` → ``Foo`` via
       existing import/alias machinery.
    2. RHS is a bare name that matches an annotated ``__init__`` parameter:
       ``def __init__(self, foo: Foo)`` + ``self.X = foo`` → ``Foo``.
    3. RHS is a builtin literal (``{}``, ``[]``, ``set()``, ``tuple()``) or
       ``Path(...)`` call → sentinel FQN (e.g. ``"builtins.dict"``).
    4. Class-body ``AnnAssign`` ``self.X: T`` (typed attribute declaration).
    5. Parameter annotation is a builtin/pathlib type: ``cfg: dict`` + ``self.x = cfg``.

    First-assignment-wins; no flow-sensitive reassignment tracking.
    No tuple/dict unpacking, no factory-function return-type resolution.
    Returns attrs whose inferred type is in ``known_fqns`` OR is a sentinel type.
    """
    result: dict[str, dict[str, str]] = {}

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        class_fqn = f"{module_fqn}.{node.name}"
        attr_types: dict[str, str] = result.setdefault(class_fqn, {})

        # Source 1: class-body AnnAssign nodes — e.g. ``x: dict[str, Any]``
        for stmt in ast.iter_child_nodes(node):
            if not isinstance(stmt, ast.AnnAssign):
                continue
            # We want ``self.x: T`` (Attribute target) or bare class-level ``x: T``.
            # Class-level ``x: T`` is a common pattern for typed attrs (dataclass-style).
            target = stmt.target
            if isinstance(target, ast.Name):
                attr_name = target.id
                if attr_name in attr_types:
                    continue
                sentinel = _infer_sentinel_from_annotation(stmt.annotation, import_table)
                if sentinel is not None:
                    attr_types[attr_name] = sentinel
                else:
                    resolved = _resolve_annotation(
                        stmt.annotation, module_fqn, import_table, known_fqns
                    )
                    if resolved is not None:
                        attr_types[attr_name] = resolved
            elif (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                attr_name = target.attr
                if attr_name in attr_types:
                    continue
                sentinel = _infer_sentinel_from_annotation(stmt.annotation, import_table)
                if sentinel is not None:
                    attr_types[attr_name] = sentinel
                else:
                    resolved = _resolve_annotation(
                        stmt.annotation, module_fqn, import_table, known_fqns
                    )
                    if resolved is not None:
                        attr_types[attr_name] = resolved

        # Source 2: __init__ / __post_init__ body assignments.
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if child.name not in {"__init__", "__post_init__"}:
                continue

            # Build param-name → annotated-type FQN map from the signature.
            param_types: dict[str, str] = {}
            for arg in child.args.args:
                if arg.arg == "self":
                    continue
                if arg.annotation is not None:
                    # Try in-package first, then sentinel.
                    resolved = _resolve_annotation(
                        arg.annotation, module_fqn, import_table, known_fqns
                    )
                    if resolved is not None:
                        param_types[arg.arg] = resolved
                    else:
                        sentinel = _infer_sentinel_from_annotation(
                            arg.annotation, import_table
                        )
                        if sentinel is not None:
                            param_types[arg.arg] = sentinel

            # Walk the body for ``self.X = ...`` assignments.
            for stmt in ast.walk(child):
                if not isinstance(stmt, ast.Assign):
                    continue
                if len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    continue
                attr_name = target.attr
                if attr_name in attr_types:
                    continue  # first-assignment-wins

                inferred = _infer_type(
                    stmt.value, module_fqn, import_table, known_fqns, param_types
                )
                if inferred is not None:
                    attr_types[attr_name] = inferred

    return result


def _resolve_annotation(
    annotation: ast.expr,
    module_fqn: str,
    import_table: dict[str, str],
    known_fqns: set[str],
) -> str | None:
    """Resolve a type annotation expression to an in-package FQN, or None."""
    if isinstance(annotation, ast.Name):
        name = annotation.id
        if name in import_table:
            candidate = import_table[name]
            if candidate in known_fqns:
                return candidate
        candidate = f"{module_fqn}.{name}"
        if candidate in known_fqns:
            return candidate
        return None

    chain = attr_chain(annotation)
    if chain is None:
        return None
    for prefix_len in range(len(chain) - 1, 0, -1):
        prefix = ".".join(chain[:prefix_len])
        if prefix in import_table:
            base_fqn = import_table[prefix]
            remainder = chain[prefix_len:]
            candidate = ".".join([base_fqn] + remainder)
            if candidate in known_fqns:
                return candidate
    dotted = ".".join(chain)
    if dotted in known_fqns:
        return dotted
    return None


def _infer_type(
    rhs: ast.expr,
    module_fqn: str,
    import_table: dict[str, str],
    known_fqns: set[str],
    param_types: dict[str, str],
) -> str | None:
    """Infer the class type of an RHS expression (best-effort, v1 scope only).

    Returns an in-package FQN, a sentinel FQN (e.g. ``"builtins.dict"``), or None.
    """
    # Case 1: bare name → parameter annotation (may be sentinel or in-package).
    if isinstance(rhs, ast.Name):
        return param_types.get(rhs.id)

    # Case 1b: builtin literal → sentinel.
    sentinel = _infer_sentinel_from_rhs(rhs, import_table)
    if sentinel is not None:
        return sentinel

    # Case 2: constructor call: Foo(...) or pkg.Foo(...) or aliased.Foo(...)
    if isinstance(rhs, ast.Call):
        func = rhs.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in import_table:
                candidate = import_table[name]
                if candidate in known_fqns:
                    return candidate
            candidate = f"{module_fqn}.{name}"
            if candidate in known_fqns:
                return candidate
            return None
        chain = attr_chain(func)
        if chain is not None:
            for prefix_len in range(len(chain) - 1, 0, -1):
                prefix = ".".join(chain[:prefix_len])
                if prefix in import_table:
                    base_fqn = import_table[prefix]
                    remainder = chain[prefix_len:]
                    candidate = ".".join([base_fqn] + remainder)
                    if candidate in known_fqns:
                        return candidate
            dotted = ".".join(chain)
            if dotted in known_fqns:
                return dotted
    return None


def collect_external_local_var_types(
    tree: ast.AST,
    module_fqn: str,
    import_table: dict[str, str],
    external_factories: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Map ``{func_fqn: {var_name: external_factory_fqn}}`` for locals bound
    from a whitelisted external factory call.

    Mirrors ``collect_local_var_types`` but targets external FQNs (not
    in-package classes).  Only records bindings where the resolved factory
    FQN appears in ``external_factories`` (the whitelist).

    Also handles second-order (chained) bindings: if the RHS is a method
    call on an already-typed external local (e.g. ``paginator =
    client.get_paginator(...)``) and that ``(factory_fqn, method)`` pair
    maps to a known return type in ``EXTERNAL_RETURNS``, the result type
    is recorded too.

    Scope/last-write-wins rules are identical to ``collect_local_var_types``.
    """
    from .resolution import EXTERNAL_RETURNS  # avoid circular at module level
    result: dict[str, dict[str, str]] = {}
    # Collect module-level external factory bindings (e.g. `_cli = typer.Typer()` at top level).
    # These are keyed under the module_fqn so the visitor can find them when
    # a function-level lookup misses.
    module_vars: dict[str, str] = {}
    if isinstance(tree, ast.Module):
        _walk_body_for_external_bindings(
            tree.body, module_fqn, import_table, external_factories,
            EXTERNAL_RETURNS, module_vars,
        )
        if module_vars:
            result[module_fqn] = module_vars
    _collect_func_external_types(
        tree, module_fqn, module_fqn, import_table, external_factories, result
    )
    return result


def _resolve_external_factory(
    rhs: ast.expr,
    import_table: dict[str, str],
    external_factories: dict[str, str],
) -> str | None:
    """If ``rhs`` is a call to a whitelisted external factory, return the factory FQN."""
    if not isinstance(rhs, ast.Call):
        return None
    func = rhs.func
    if isinstance(func, ast.Name):
        name = func.id
        if name in import_table:
            fqn = import_table[name]
            if fqn in external_factories:
                return external_factories[fqn]
        return None
    chain = attr_chain(func)
    if chain is not None:
        for prefix_len in range(len(chain) - 1, 0, -1):
            prefix = ".".join(chain[:prefix_len])
            if prefix in import_table:
                base_fqn = import_table[prefix]
                remainder = chain[prefix_len:]
                candidate = ".".join([base_fqn] + remainder)
                if candidate in external_factories:
                    return external_factories[candidate]
        dotted = ".".join(chain)
        if dotted in external_factories:
            return external_factories[dotted]
    return None


def _resolve_chained_external_factory(
    rhs: ast.expr,
    partial_vars: dict[str, str],
    external_returns: dict[tuple[str, str], str],
) -> str | None:
    """If ``rhs`` is ``var.method(...)`` and ``var`` is already a typed external
    local, look up ``(factory_fqn, method)`` in ``external_returns``.
    """
    if not isinstance(rhs, ast.Call):
        return None
    func = rhs.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name):
        return None
    var_name = func.value.id
    method = func.attr
    factory_fqn = partial_vars.get(var_name)
    if factory_fqn is None:
        return None
    return external_returns.get((factory_fqn, method))


def _collect_func_external_types(
    node: ast.AST,
    module_fqn: str,
    func_fqn: str,
    import_table: dict[str, str],
    external_factories: dict[str, str],
    result: dict[str, dict[str, str]],
) -> None:
    """Recursively collect external factory var types for each function scope."""
    from .resolution import EXTERNAL_RETURNS
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            class_fqn = (
                f"{func_fqn}.{child.name}"
                if func_fqn != module_fqn
                else f"{module_fqn}.{child.name}"
            )
            _collect_func_external_types(
                child, module_fqn, class_fqn, import_table, external_factories, result
            )
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            child_fqn = f"{func_fqn}.{child.name}"
            _scan_function_external_bindings(
                child, module_fqn, child_fqn, import_table, external_factories,
                EXTERNAL_RETURNS, result
            )


def _scan_function_external_bindings(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_fqn: str,
    func_fqn: str,
    import_table: dict[str, str],
    external_factories: dict[str, str],
    external_returns: dict[tuple[str, str], str],
    result: dict[str, dict[str, str]],
) -> None:
    """Scan a single function body for external-factory bindings (non-recursive)."""
    var_types: dict[str, str] = {}
    _walk_body_for_external_bindings(
        func_node.body, module_fqn, import_table, external_factories,
        external_returns, var_types,
    )
    if var_types:
        result[func_fqn] = var_types

    # Recurse into nested functions.
    for nested in _iter_direct_nested_funcs(func_node.body):
        nested_fqn = f"{func_fqn}.{nested.name}"
        _scan_function_external_bindings(
            nested, module_fqn, nested_fqn, import_table, external_factories,
            external_returns, result
        )


def _walk_body_for_external_bindings(
    stmts: list[ast.stmt],
    module_fqn: str,
    import_table: dict[str, str],
    external_factories: dict[str, str],
    external_returns: dict[tuple[str, str], str],
    var_types: dict[str, str],
) -> None:
    """Walk a statement list collecting external-factory Assign bindings."""
    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                var_name = stmt.targets[0].id
                # Direct factory call
                resolved = _resolve_external_factory(
                    stmt.value, import_table, external_factories
                )
                if resolved is None:
                    # Chained: var.method(...) where var is already typed
                    resolved = _resolve_chained_external_factory(
                        stmt.value, var_types, external_returns
                    )
                # last-write-wins: always overwrite
                if resolved is not None:
                    var_types[var_name] = resolved
                elif var_name in var_types:
                    # var was rebound to something non-factory → shadow, remove
                    del var_types[var_name]

        elif isinstance(stmt, ast.For):
            _walk_body_for_external_bindings(
                stmt.body + stmt.orelse, module_fqn, import_table,
                external_factories, external_returns, var_types,
            )
        elif isinstance(stmt, ast.While):
            _walk_body_for_external_bindings(
                stmt.body + stmt.orelse, module_fqn, import_table,
                external_factories, external_returns, var_types,
            )
        elif isinstance(stmt, ast.If):
            _walk_body_for_external_bindings(
                stmt.body + stmt.orelse, module_fqn, import_table,
                external_factories, external_returns, var_types,
            )
        elif isinstance(stmt, ast.With):
            _walk_body_for_external_bindings(
                stmt.body, module_fqn, import_table,
                external_factories, external_returns, var_types,
            )
        elif isinstance(stmt, ast.AsyncWith):
            _walk_body_for_external_bindings(
                stmt.body, module_fqn, import_table,
                external_factories, external_returns, var_types,
            )
        elif isinstance(stmt, ast.Try):
            all_stmts = stmt.body + stmt.orelse + stmt.finalbody
            for handler in stmt.handlers:
                all_stmts += handler.body
            _walk_body_for_external_bindings(
                all_stmts, module_fqn, import_table,
                external_factories, external_returns, var_types,
            )


def collect_local_var_types(
    tree: ast.AST,
    module_fqn: str,
    import_table: dict[str, str],
    known_classes: set[str],
) -> dict[str, dict[str, str]]:
    """Map ``{func_fqn: {var_name: class_fqn}}`` for local variable bindings.

    Scans each function/method body for statically resolvable variable
    bindings of the form:

    - ``x = ClassName(...)`` — constructor call (Assign).
    - ``x: ClassName = ...`` — annotated assignment (AnnAssign), RHS ignored.
    - Parameter annotations: ``def f(self, x: ClassName)`` — x tracked.
      ``self`` / ``cls`` are skipped.

    Last-write-wins: later bindings overwrite earlier ones (by source line).
    Only records bindings where the class resolves to an in-package class FQN
    (i.e. the FQN is in ``known_classes``).

    Scope rules:
    - Each function/method/lambda gets its own dict keyed by its FQN.
    - Nested function bodies are NOT descended into (they get their own entry).
    - Loop targets (for x in xs) are NOT tracked.
    - Comprehension targets are NOT tracked.
    - Tuple-unpack targets (a, b = X(), Y()) are NOT tracked. TODO: v2.
    - Lambda parameters are NOT tracked (type annotations unsupported on lambdas).
    """
    result: dict[str, dict[str, str]] = {}
    _collect_func_local_types(tree, module_fqn, module_fqn, import_table, known_classes, result)
    return result


def _is_none_literal(node: ast.expr) -> bool:
    """Return True if node is the literal ``None`` (ast.Constant(value=None))."""
    return isinstance(node, ast.Constant) and node.value is None


def _resolve_union_lhs(
    value: ast.expr,
    import_table: dict[str, str],
) -> str | None:
    """Resolve the LHS of a Subscript to a normalised typing FQN.

    Returns one of ``"typing.Optional"``, ``"typing.Union"``, or ``None``.
    We accept both bare names (``Optional``, ``Union``) and qualified forms
    (``typing.Optional``, ``t.Union``) by checking the import table.
    """
    # Bare name: Optional[X] or Union[X, Y]
    if isinstance(value, ast.Name):
        name = value.id
        if name in ("Optional", "Union"):
            # Check import_table to see if it really maps to typing.*
            target = import_table.get(name)
            if target in ("typing.Optional", "typing.Union"):
                return target
            # Bare name with no confirmed import mapping.  In practice, bare
            # `Optional` and `Union` nearly always come from `from typing import
            # Optional/Union` — accept them.  A user-defined class named
            # Optional/Union subscripted in an annotation is pathological and
            # not guarded here.
            if name == "Optional":
                return "typing.Optional"
            if name == "Union":
                return "typing.Union"
        return None

    # Qualified form: typing.Optional[X], t.Union[X, Y]
    chain = attr_chain(value)
    if chain is None:
        return None
    for prefix_len in range(len(chain) - 1, 0, -1):
        prefix = ".".join(chain[:prefix_len])
        target = import_table.get(prefix)
        if target is not None:
            remainder = chain[prefix_len:]
            resolved = ".".join([target] + remainder)
            if resolved in ("typing.Optional", "typing.Union"):
                return resolved
    # Fully qualified without import alias: e.g. typing.Optional without import
    dotted = ".".join(chain)
    if dotted in ("typing.Optional", "typing.Union"):
        return dotted
    return None


def _resolve_annotation_to_class(
    annotation: ast.expr | str,
    module_fqn: str,
    import_table: dict[str, str],
    known_classes: set[str],
) -> str | None:
    """Resolve a type annotation (or forward-ref string) to an in-package class FQN.

    Handles:
    - ``ast.Name`` / ``ast.Attribute`` / dotted chains — simple name lookup.
    - ``ast.Constant(str)`` — forward-reference string annotations.
    - ``ast.BinOp(op=BitOr)`` — PEP 604 union (``X | None``, ``X | Y``).
    - ``ast.Subscript`` with typing.Optional or typing.Union LHS.

    False-positive guards: ``list[X]``, ``dict[K, V]``, ``Callable[...]``,
    ``Annotated[X, ...]`` all return ``None`` — only Union-shaped subscripts
    are peeled.
    """
    # ------------------------------------------------------------------ #
    # PEP 604 union: X | Y  (left-associative; None literals are skipped) #
    # ------------------------------------------------------------------ #
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        # Filter None on either side immediately.
        if not _is_none_literal(annotation.left):
            result = _resolve_annotation_to_class(
                annotation.left, module_fqn, import_table, known_classes
            )
            if result is not None:
                return result
        if not _is_none_literal(annotation.right):
            result = _resolve_annotation_to_class(
                annotation.right, module_fqn, import_table, known_classes
            )
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # Subscript: Optional[X] or Union[X, Y, ...]                         #
    # Guard: only peel recognised typing forms; list[X], Callable, etc.  #
    # must NOT be treated as unions.                                      #
    # ------------------------------------------------------------------ #
    if isinstance(annotation, ast.Subscript):
        typing_form = _resolve_union_lhs(annotation.value, import_table)
        if typing_form == "typing.Optional":
            # Optional[X] → recurse on slice.
            return _resolve_annotation_to_class(
                annotation.slice, module_fqn, import_table, known_classes
            )
        if typing_form == "typing.Union":
            # Union[X, Y, ...] → try each elt in order, first in-package wins.
            slc = annotation.slice
            elts: list[ast.expr]
            if isinstance(slc, ast.Tuple):
                elts = slc.elts
            else:
                # Union[X] (degenerate but legal) — treat slice as single elt.
                elts = [slc]
            for elt in elts:
                if _is_none_literal(elt):
                    continue
                result = _resolve_annotation_to_class(
                    elt, module_fqn, import_table, known_classes
                )
                if result is not None:
                    return result
            return None
        # Not a recognised union form — return None (guards list[X], Callable, etc.)
        return None

    # ------------------------------------------------------------------ #
    # Forward-reference string annotation e.g. "ClassName" or "mod.C"   #
    # ------------------------------------------------------------------ #
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        raw = annotation.value.strip()
        # Try as a simple name first
        if "." not in raw:
            if raw in import_table:
                candidate = import_table[raw]
                if candidate in known_classes:
                    return candidate
            candidate = f"{module_fqn}.{raw}"
            if candidate in known_classes:
                return candidate
            return None
        # Dotted string: try longest-prefix lookup
        parts = raw.split(".")
        for prefix_len in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:prefix_len])
            if prefix in import_table:
                base_fqn = import_table[prefix]
                remainder = parts[prefix_len:]
                candidate = ".".join([base_fqn] + remainder)
                if candidate in known_classes:
                    return candidate
        dotted = raw
        if dotted in known_classes:
            return dotted
        return None

    # ------------------------------------------------------------------ #
    # Bare name                                                           #
    # ------------------------------------------------------------------ #
    if isinstance(annotation, ast.Name):
        name = annotation.id
        if name in import_table:
            candidate = import_table[name]
            if candidate in known_classes:
                return candidate
        candidate = f"{module_fqn}.{name}"
        if candidate in known_classes:
            return candidate
        return None

    # ------------------------------------------------------------------ #
    # Attribute chain: mod.ClassName, pkg.sub.ClassName                  #
    # ------------------------------------------------------------------ #
    chain = attr_chain(annotation)
    if chain is None:
        return None
    for prefix_len in range(len(chain) - 1, 0, -1):
        prefix = ".".join(chain[:prefix_len])
        if prefix in import_table:
            base_fqn = import_table[prefix]
            remainder = chain[prefix_len:]
            candidate = ".".join([base_fqn] + remainder)
            if candidate in known_classes:
                return candidate
    dotted = ".".join(chain)
    if dotted in known_classes:
        return dotted
    return None


def _infer_constructor_class(
    rhs: ast.expr,
    module_fqn: str,
    import_table: dict[str, str],
    known_classes: set[str],
) -> str | None:
    """Infer class FQN if RHS is a constructor call to an in-package class.

    Handles two forms:
    - ``ClassName(...)`` / ``pkg.ClassName(...)`` — direct constructor call.
    - ``ClassName.__new__(ClassName)`` — bypass-__init__ constructor pattern.
      The ``.__new__`` attr is stripped; the receiver expression is resolved
      to a class FQN using the same longest-prefix import-table logic.
    """
    if not isinstance(rhs, ast.Call):
        return None
    func = rhs.func

    # ClassName.__new__(...) — treat as constructor for ClassName.
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "__new__"
    ):
        inner = func.value
        if isinstance(inner, ast.Name):
            name = inner.id
            if name in import_table:
                candidate = import_table[name]
                if candidate in known_classes:
                    return candidate
            candidate = f"{module_fqn}.{name}"
            if candidate in known_classes:
                return candidate
            return None
        inner_chain = attr_chain(inner)
        if inner_chain is not None:
            for prefix_len in range(len(inner_chain) - 1, 0, -1):
                prefix = ".".join(inner_chain[:prefix_len])
                if prefix in import_table:
                    base_fqn = import_table[prefix]
                    remainder = inner_chain[prefix_len:]
                    candidate = ".".join([base_fqn] + remainder)
                    if candidate in known_classes:
                        return candidate
            dotted = ".".join(inner_chain)
            if dotted in known_classes:
                return dotted
        return None

    if isinstance(func, ast.Name):
        name = func.id
        if name in import_table:
            candidate = import_table[name]
            if candidate in known_classes:
                return candidate
        candidate = f"{module_fqn}.{name}"
        if candidate in known_classes:
            return candidate
        return None
    chain = attr_chain(func)
    if chain is not None:
        for prefix_len in range(len(chain) - 1, 0, -1):
            prefix = ".".join(chain[:prefix_len])
            if prefix in import_table:
                base_fqn = import_table[prefix]
                remainder = chain[prefix_len:]
                candidate = ".".join([base_fqn] + remainder)
                if candidate in known_classes:
                    return candidate
        dotted = ".".join(chain)
        if dotted in known_classes:
            return dotted
    return None


def _collect_func_local_types(
    node: ast.AST,
    module_fqn: str,
    func_fqn: str,
    import_table: dict[str, str],
    known_classes: set[str],
    result: dict[str, dict[str, str]],
) -> None:
    """Recursively collect local var types for each function scope.

    Descends into ClassDef to find methods, but stops descent when entering
    a nested FunctionDef/AsyncFunctionDef (that scope gets its own entry).
    """
    # For module-level: walk top-level nodes.
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            class_fqn = f"{func_fqn}.{child.name}" if func_fqn != module_fqn else f"{module_fqn}.{child.name}"
            _collect_func_local_types(child, module_fqn, class_fqn, import_table, known_classes, result)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            child_fqn = f"{func_fqn}.{child.name}"
            _scan_function_body(child, module_fqn, child_fqn, import_table, known_classes, result)


def _iter_direct_nested_funcs(
    stmts: list[ast.stmt],
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield FunctionDef/AsyncFunctionDef nodes that are directly (shallowly) nested.

    Descends into if/for/while/with/try bodies (where a def can appear) but
    does NOT descend into nested function bodies — those are separate scopes.
    """
    found: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(stmt)
        elif isinstance(stmt, ast.If):
            found.extend(_iter_direct_nested_funcs(stmt.body + stmt.orelse))
        elif isinstance(stmt, ast.For):
            found.extend(_iter_direct_nested_funcs(stmt.body + stmt.orelse))
        elif isinstance(stmt, ast.While):
            found.extend(_iter_direct_nested_funcs(stmt.body + stmt.orelse))
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            found.extend(_iter_direct_nested_funcs(stmt.body))
        elif isinstance(stmt, ast.Try):
            all_stmts = stmt.body + stmt.orelse + stmt.finalbody
            for handler in stmt.handlers:
                all_stmts += handler.body
            found.extend(_iter_direct_nested_funcs(all_stmts))
    return found


def _scan_function_body(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_fqn: str,
    func_fqn: str,
    import_table: dict[str, str],
    known_classes: set[str],
    result: dict[str, dict[str, str]],
) -> None:
    """Scan a single function body (non-recursive into nested funcs).

    Collects bindings ordered by source line (last-write-wins).
    Also recurses into nested functions as separate scopes.
    """
    var_types: dict[str, str] = {}

    # Collect parameter annotations (skip self/cls).
    SKIP_PARAMS = {"self", "cls"}
    for arg in func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs:
        if arg.arg in SKIP_PARAMS:
            continue
        if arg.annotation is not None:
            resolved = _resolve_annotation_to_class(
                arg.annotation, module_fqn, import_table, known_classes
            )
            if resolved is None:
                resolved = _infer_sentinel_from_annotation(arg.annotation, import_table)
            if resolved is not None:
                var_types[arg.arg] = resolved

    # Walk the direct body — collect Assign and AnnAssign, but do NOT descend
    # into nested function bodies (collect those as separate scopes).
    _walk_body_for_bindings(
        func_node.body, module_fqn, func_fqn, import_table, known_classes,
        var_types, result,
    )

    if var_types:
        result[func_fqn] = var_types

    # Recurse into directly-nested functions as separate scopes.
    # We must NOT use ast.walk here — it descends transitively and would visit
    # doubly-nested functions twice (once from this scope's walk, and again
    # when the intermediate scope's _scan_function_body runs its own walk).
    # Instead collect only direct-child function defs from the body stmts.
    for nested in _iter_direct_nested_funcs(func_node.body):
        nested_fqn = f"{func_fqn}.{nested.name}"
        _scan_function_body(nested, module_fqn, nested_fqn, import_table, known_classes, result)


def _walk_body_for_bindings(
    stmts: list[ast.stmt],
    module_fqn: str,
    func_fqn: str,
    import_table: dict[str, str],
    known_classes: set[str],
    var_types: dict[str, str],
    result: dict[str, dict[str, str]],
) -> None:
    """Walk a statement list collecting Assign/AnnAssign bindings.

    Does NOT descend into nested function/async-function bodies.
    Descends into if/for/while/with/try bodies for assignments in those blocks,
    but for-loop targets are skipped (element type is unknown).
    """
    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Nested function — do NOT descend; it's a separate scope.
            continue

        if isinstance(stmt, ast.ClassDef):
            # Nested class — skip; not a variable binding in this scope.
            continue

        if isinstance(stmt, ast.Assign):
            # Only handle single, simple-name targets (skip tuple-unpack). TODO: v2 tuple-unpack.
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                var_name = stmt.targets[0].id
                resolved = _infer_constructor_class(
                    stmt.value, module_fqn, import_table, known_classes
                )
                if resolved is None:
                    resolved = _infer_sentinel_from_rhs(stmt.value, import_table)
                if resolved is not None:
                    var_types[var_name] = resolved

        elif isinstance(stmt, ast.AnnAssign):
            # x: ClassName = ... — use annotation, ignore RHS.
            if isinstance(stmt.target, ast.Name) and stmt.annotation is not None:
                var_name = stmt.target.id
                resolved = _resolve_annotation_to_class(
                    stmt.annotation, module_fqn, import_table, known_classes
                )
                if resolved is None:
                    resolved = _infer_sentinel_from_annotation(stmt.annotation, import_table)
                if resolved is not None:
                    var_types[var_name] = resolved

        elif isinstance(stmt, ast.For):
            # for x in xs: — skip the loop target but descend into body.
            _walk_body_for_bindings(
                stmt.body + stmt.orelse,
                module_fqn, func_fqn, import_table, known_classes,
                var_types, result,
            )

        elif isinstance(stmt, ast.While):
            _walk_body_for_bindings(
                stmt.body + stmt.orelse,
                module_fqn, func_fqn, import_table, known_classes,
                var_types, result,
            )

        elif isinstance(stmt, ast.If):
            _walk_body_for_bindings(
                stmt.body + stmt.orelse,
                module_fqn, func_fqn, import_table, known_classes,
                var_types, result,
            )

        elif isinstance(stmt, ast.With):
            _walk_body_for_bindings(
                stmt.body,
                module_fqn, func_fqn, import_table, known_classes,
                var_types, result,
            )

        elif isinstance(stmt, ast.AsyncWith):
            _walk_body_for_bindings(
                stmt.body,
                module_fqn, func_fqn, import_table, known_classes,
                var_types, result,
            )

        elif isinstance(stmt, ast.Try):
            all_stmts = stmt.body + stmt.orelse + stmt.finalbody
            for handler in stmt.handlers:
                all_stmts += handler.body
            _walk_body_for_bindings(
                all_stmts,
                module_fqn, func_fqn, import_table, known_classes,
                var_types, result,
            )


def _resolve_base(
    base: ast.expr,
    module_fqn: str,
    import_table: dict[str, str],
) -> str | None:
    """Resolve a base-class expression to an FQN (best-effort).

    Mirrors the Name / Attribute resolution used for call targets.
    """
    if isinstance(base, ast.Name):
        name = base.id
        if name in import_table:
            return import_table[name]
        # Same-module sibling class
        return f"{module_fqn}.{name}"

    chain = attr_chain(base)
    if chain is None:
        return None
    # Longest prefix against import table.
    for prefix_len in range(len(chain) - 1, 0, -1):
        prefix = ".".join(chain[:prefix_len])
        if prefix in import_table:
            base_fqn = import_table[prefix]
            remainder = chain[prefix_len:]
            return ".".join([base_fqn] + remainder)
    # Bare dotted chain with no import — probably unresolvable.
    return ".".join(chain)
