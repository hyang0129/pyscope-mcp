"""Call-graph analyzer package.

Public API:
    build_raw(root, package) -> dict[str, list[str]]
    build_with_report(root, package) -> tuple[dict[str, list[str]], dict]

Layout:
    discovery    — pass 1 (modules, defs, class bases)
    imports      — module-level import-table construction
    resolution   — pure resolver helpers (attr_chain, MRO walk, dispatcher whitelist)
    visitor      — EdgeVisitor (pass 2 AST walker)
    misses       — MissLog + miss classification
    pipeline     — orchestrator tying pass 1 + pass 2 together
"""

from __future__ import annotations

from .discovery import (
    collect_class_bases,
    collect_defs as _collect_defs,
    discover_modules as _discover_modules,
)
from .imports import build_import_table as _build_import_table
from .misses import MissLog, classify_miss as _classify_miss
from .pipeline import build_nodes_with_report, build_raw, build_with_report
from .visitor import EdgeVisitor as _EdgeVisitor

__all__ = [
    "build_raw",
    "build_with_report",
    "build_nodes_with_report",
    "MissLog",
    # Private names re-exported for existing tests / callers:
    "_collect_defs",
    "_discover_modules",
    "_build_import_table",
    "_classify_miss",
    "_EdgeVisitor",
    "collect_class_bases",
]
