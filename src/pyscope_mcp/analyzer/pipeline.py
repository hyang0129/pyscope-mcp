"""Orchestrator: pass 1 + pass 2. The public `build_raw` / `build_with_report`."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

from .discovery import (
    collect_class_bases,
    collect_classes,
    collect_defs,
    collect_external_local_var_types,
    collect_local_var_types,
    collect_nested_defs,
    collect_self_attr_types,
    discover_modules,
)
from .imports import build_import_table
from .misses import MissLog
from .resolution import EXTERNAL_FACTORIES
from .visitor import EdgeVisitor


def _warn(msg: str) -> None:
    print(f"[pyscope-mcp] {msg}", file=sys.stderr)


def _extract_skeletons(
    root: Path,
    parsed: list[tuple[str, ast.Module, Path, dict[str, str]]],
) -> dict[str, list[dict]]:
    """Extract symbol skeletons (fqn, kind, signature, lineno) from parsed ASTs.

    Skeleton data is keyed by file path relative to ``root``.  Each entry is a
    list of SymbolSummary-compatible dicts sorted by lineno.

    Only top-level functions, top-level classes, and methods (functions defined
    directly inside a class body) are included.  Nested functions inside other
    functions are excluded — they are not part of the public structural surface.

    The ``signature`` field is the first ``def`` or ``class`` line only — no body.
    """
    skeletons: dict[str, list[dict]] = {}
    for module_fqn, tree, file_path, _import_table in parsed:
        try:
            rel = str(file_path.relative_to(root))
        except ValueError:
            rel = str(file_path)

        symbols: list[dict] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            # Only process top-level nodes and methods (depth == 1 inside a class)
            # We iterate ast.walk which is unordered; we need explicit parent traversal.
            break  # exit early — we do explicit traversal below

        # Explicit two-level traversal: top-level defs + class bodies
        for top_node in ast.iter_child_nodes(tree):
            if isinstance(top_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = top_node.name
                fqn = f"{module_fqn}.{name}"
                sig = _first_def_line(top_node)
                symbols.append({
                    "fqn": fqn,
                    "kind": "function",
                    "signature": sig,
                    "lineno": top_node.lineno,
                })
            elif isinstance(top_node, ast.ClassDef):
                cls_name = top_node.name
                cls_fqn = f"{module_fqn}.{cls_name}"
                symbols.append({
                    "fqn": cls_fqn,
                    "kind": "class",
                    "signature": _first_def_line(top_node),
                    "lineno": top_node.lineno,
                })
                # Methods inside the class body (one level deep)
                for class_node in ast.iter_child_nodes(top_node):
                    if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_name = class_node.name
                        method_fqn = f"{cls_fqn}.{method_name}"
                        symbols.append({
                            "fqn": method_fqn,
                            "kind": "method",
                            "signature": _first_def_line(class_node),
                            "lineno": class_node.lineno,
                        })

        symbols.sort(key=lambda s: s["lineno"])
        skeletons[rel] = symbols

    return skeletons


def _first_def_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """Return the first line of the def/class statement as a signature string.

    We reconstruct from the AST rather than slicing source lines so that the
    pipeline stays stateless (no need to pass raw source strings around).
    The result is ``def name(args...):`` or ``class Name(bases...):`` — no body.
    """
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        base_str = f"({bases})" if bases else ""
        return f"class {node.name}{base_str}:"

    # FunctionDef / AsyncFunctionDef
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args_str = ast.unparse(node.args)
    ret = ""
    if node.returns is not None:
        ret = f" -> {ast.unparse(node.returns)}"
    return f"{prefix} {node.name}({args_str}){ret}:"


def build_with_report(
    root: str | Path,
    package: str,
) -> tuple[dict[str, list[str]], dict, dict[str, list[dict]], dict[str, str]]:
    """Pass 1 + pass 2 with MissLog. Returns (raw_edges, miss_report_dict, skeletons, file_shas).

    ``skeletons`` maps relative file paths to pre-computed symbol lists suitable
    for storage in the index under the ``skeletons`` key (version 3 schema).

    ``file_shas`` maps relative file paths to SHA256 hex digests of the file
    bytes at build time, used by ``file_skeleton()`` to detect stale results.
    """
    root = Path(root)
    pkg_root = root / package if (root / package).is_dir() else root

    modules = discover_modules(pkg_root, package)

    miss_log = MissLog()
    miss_log.files_total = len(modules)

    # Pass 1: parse all files; collect defs, import tables, class bases.
    parsed: list[tuple[str, ast.Module, Path, dict[str, str]]] = []
    file_shas: dict[str, str] = {}
    known_fqns: set[str] = set()
    known_classes: set[str] = set()
    class_bases: dict[str, list[str]] = {}
    self_attr_types: dict[str, dict[str, str]] = {}
    local_types: dict[str, dict[str, str]] = {}
    external_local_types: dict[str, dict[str, str]] = {}
    nested_defs: dict[str, dict[str, tuple[str, int]]] = {}

    for fqn, path in modules.items():
        try:
            raw_bytes = path.read_bytes()
            source = raw_bytes.decode("utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
            known_fqns.update(collect_defs(tree, fqn))
            known_classes.update(collect_classes(tree, fqn))
            import_table = build_import_table(tree, fqn)
            parsed.append((fqn, tree, path, import_table))
            # Compute SHA256 from the raw bytes already in memory — no second disk read.
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = str(path)
            file_shas[rel] = hashlib.sha256(raw_bytes).hexdigest()
        except SyntaxError as exc:
            reason = f"SyntaxError: {exc}"
            _warn(f"skipping {path}: {reason}")
            miss_log.record_skip(str(path), reason)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            _warn(f"skipping {path}: {reason}")
            miss_log.record_skip(str(path), reason)

    # Second sweep over parsed files to build class-bases, self-attr types, and
    # local-variable type bindings, now that known_fqns/known_classes are fully
    # populated (types may reference classes in other modules).
    for fqn, tree, _path, import_table in parsed:
        class_bases.update(collect_class_bases(tree, fqn, import_table))
        self_attr_types.update(
            collect_self_attr_types(tree, fqn, import_table, known_fqns)
        )
        local_types.update(
            collect_local_var_types(tree, fqn, import_table, known_classes)
        )
        external_local_types.update(
            collect_external_local_var_types(tree, fqn, import_table, EXTERNAL_FACTORIES)
        )
        nested_defs.update(
            collect_nested_defs(tree, fqn)
        )

    miss_log.files_parsed = len(parsed)

    # Pass 2: extract edges.
    all_edges: dict[str, set[str]] = {}
    for fqn, tree, path, import_table in parsed:
        try:
            visitor = EdgeVisitor(
                fqn,
                import_table,
                known_fqns,
                class_bases,
                file_path=str(path),
                miss_log=miss_log,
                known_classes=known_classes,
                self_attr_types=self_attr_types,
                local_types=local_types,
                external_local_types=external_local_types,
                nested_defs=nested_defs,
            )
            visitor.visit(tree)
            for caller, callees in visitor.edges.items():
                all_edges.setdefault(caller, set()).update(callees)
        except Exception as exc:  # noqa: BLE001
            _warn(f"error processing {fqn} in pass 2: {type(exc).__name__}: {exc}")

    raw = {caller: sorted(callees) for caller, callees in sorted(all_edges.items())}
    report = miss_log.to_dict(raw, known_fqns)
    skeletons = _extract_skeletons(root, parsed)
    return raw, report, skeletons, file_shas


def build_raw(root: str | Path, package: str) -> dict[str, list[str]]:
    """Pass 1 + pass 2: discover modules, collect defs, extract call edges.

    Returns just the raw edge dict (public contract, unchanged).
    """
    raw, _report, _skeletons, _file_shas = build_with_report(root, package)
    return raw
