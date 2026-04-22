"""Call-graph analyzer.

Replaces the pycg backend we started with. Not implemented yet — the plan
lives in CLAUDE.md and docs/prior-art.md. The contract this module owes
the rest of the codebase is minimal:

    build_raw(root, package) -> dict[str, list[str]]

Where keys are fully-qualified caller names and values are lists of
fully-qualified callees. `CallGraphIndex.from_raw` takes it from there.
"""

from __future__ import annotations

from pathlib import Path


def build_raw(root: str | Path, package: str) -> dict[str, list[str]]:
    raise NotImplementedError(
        "pyscope-mcp analyzer is not implemented yet. "
        "See CLAUDE.md for the rewrite plan."
    )
