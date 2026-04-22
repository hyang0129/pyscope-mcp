"""Orchestrator: pass 1 + pass 2. The public `build_raw` / `build_with_report`."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from .discovery import collect_class_bases, collect_defs, discover_modules
from .imports import build_import_table
from .misses import MissLog
from .visitor import EdgeVisitor


def _warn(msg: str) -> None:
    print(f"[pyscope-mcp] {msg}", file=sys.stderr)


def build_with_report(
    root: str | Path,
    package: str,
) -> tuple[dict[str, list[str]], dict]:
    """Pass 1 + pass 2 with MissLog. Returns (raw_edges, miss_report_dict)."""
    root = Path(root)
    pkg_root = root / package if (root / package).is_dir() else root

    modules = discover_modules(pkg_root, package)

    miss_log = MissLog()
    miss_log.files_total = len(modules)

    # Pass 1: parse all files; collect defs, import tables, class bases.
    parsed: list[tuple[str, ast.Module, Path, dict[str, str]]] = []
    known_fqns: set[str] = set()
    class_bases: dict[str, list[str]] = {}

    for fqn, path in modules.items():
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
            known_fqns.update(collect_defs(tree, fqn))
            import_table = build_import_table(tree, fqn)
            parsed.append((fqn, tree, path, import_table))
        except SyntaxError as exc:
            reason = f"SyntaxError: {exc}"
            _warn(f"skipping {path}: {reason}")
            miss_log.record_skip(str(path), reason)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            _warn(f"skipping {path}: {reason}")
            miss_log.record_skip(str(path), reason)

    # Second sweep over parsed files to build class-bases, now that
    # known_fqns is fully populated (bases in one module may reference
    # classes in another).
    for fqn, tree, _path, import_table in parsed:
        class_bases.update(collect_class_bases(tree, fqn, import_table))

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
            )
            visitor.visit(tree)
            for caller, callees in visitor.edges.items():
                all_edges.setdefault(caller, set()).update(callees)
        except Exception as exc:  # noqa: BLE001
            _warn(f"error processing {fqn} in pass 2: {type(exc).__name__}: {exc}")

    raw = {caller: sorted(callees) for caller, callees in sorted(all_edges.items())}
    report = miss_log.to_dict(raw, known_fqns)
    return raw, report


def build_raw(root: str | Path, package: str) -> dict[str, list[str]]:
    """Pass 1 + pass 2: discover modules, collect defs, extract call edges.

    Returns just the raw edge dict (public contract, unchanged).
    """
    raw, _report = build_with_report(root, package)
    return raw
