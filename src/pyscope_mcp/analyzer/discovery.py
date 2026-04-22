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
        # Path — only if it resolves to pathlib.Path via import_table
        if name == "Path":
            resolved = import_table.get("Path")
            if resolved in ("pathlib.Path",):
                return "pathlib.Path"
            # bare 'Path' with no import_table entry but commonly means pathlib.Path
            if resolved is None:
                # conservative: only if no other mapping for 'Path' exists
                return "pathlib.Path"
        # pathlib.Path annotation
        if name in import_table:
            if import_table[name] == "pathlib.Path":
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
            if name == "Path":
                resolved = import_table.get("Path")
                if resolved in ("pathlib.Path",) or resolved is None:
                    return "pathlib.Path"
            if name in import_table and import_table[name] == "pathlib.Path":
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


def _resolve_annotation_to_class(
    annotation: ast.expr | str,
    module_fqn: str,
    import_table: dict[str, str],
    known_classes: set[str],
) -> str | None:
    """Resolve a type annotation (or forward-ref string) to an in-package class FQN."""
    # Handle forward-reference string annotations e.g. "ClassName" or "mod.ClassName"
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
    """Infer class FQN if RHS is a constructor call to an in-package class."""
    if not isinstance(rhs, ast.Call):
        return None
    func = rhs.func
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
                if resolved is not None:
                    var_types[var_name] = resolved

        elif isinstance(stmt, ast.AnnAssign):
            # x: ClassName = ... — use annotation, ignore RHS.
            if isinstance(stmt.target, ast.Name) and stmt.annotation is not None:
                var_name = stmt.target.id
                resolved = _resolve_annotation_to_class(
                    stmt.annotation, module_fqn, import_table, known_classes
                )
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
