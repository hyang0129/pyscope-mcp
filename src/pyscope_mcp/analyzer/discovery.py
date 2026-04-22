"""Pass 1: discover modules, collect definitions, build project-wide index."""

from __future__ import annotations

import ast
from pathlib import Path

from .imports import build_import_table
from .resolution import attr_chain


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

    First-assignment-wins; no flow-sensitive reassignment tracking.
    No tuple/dict unpacking, no factory-function return-type resolution.
    Returns only attrs whose inferred type is in ``known_fqns``.
    """
    result: dict[str, dict[str, str]] = {}

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        class_fqn = f"{module_fqn}.{node.name}"

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
                    resolved = _resolve_annotation(
                        arg.annotation, module_fqn, import_table, known_fqns
                    )
                    if resolved is not None:
                        param_types[arg.arg] = resolved

            # Walk the body for ``self.X = ...`` assignments.
            attr_types: dict[str, str] = result.setdefault(class_fqn, {})
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
    """Infer the class type of an RHS expression (best-effort, v1 scope only)."""
    # Case 1: bare name → parameter annotation.
    if isinstance(rhs, ast.Name):
        return param_types.get(rhs.id)

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
