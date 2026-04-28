from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pyscope_mcp.graph import CallGraphIndex

DEFAULT_INDEX_PATH = ".pyscope-mcp/index.json"


def _index_path(value: str | None) -> Path:
    return Path(value or os.environ.get("PYSCOPE_MCP_INDEX") or DEFAULT_INDEX_PATH).resolve()


def _print_summary(report: dict, misses_path: Path) -> None:
    """Print a one-screen build summary to stderr."""
    s = report["summary"]
    files_total = s["files_total"]
    files_parsed = s["files_parsed"]
    files_skipped = s["files_skipped"]
    calls_total = s["calls_total"]
    calls_in_pkg = s["calls_resolved_in_package"]
    calls_ext = s["calls_resolved_external"]
    calls_unres = s["calls_unresolved"]

    in_pkg_pct = round(calls_in_pkg / calls_total * 100) if calls_total > 0 else 0
    unres_pct = round(calls_unres / calls_total * 100) if calls_total > 0 else 0

    skip_note = f"  ({files_skipped} skipped — see misses.json)" if files_skipped else ""

    lines = [
        "pyscope-mcp build complete",
        f"  files:  {files_parsed}/{files_total} parsed{skip_note}",
        f"  calls:  {calls_total} total → {calls_in_pkg} in-package edges ({in_pkg_pct}%)",
        f"          {calls_ext} external (dropped), {calls_unres} unresolved ({unres_pct}%)",
    ]

    pattern_counts: dict[str, int] = report.get("pattern_counts", {})
    if pattern_counts:
        top5 = sorted(pattern_counts.items(), key=lambda kv: -kv[1])[:5]
        lines.append("  top unresolved patterns:")
        for pattern, count in top5:
            lines.append(f"    {pattern:<30} {count}")

    lines.append(f"  full report: {misses_path}")

    print("\n".join(lines), file=sys.stderr)


def cmd_build(args: argparse.Namespace) -> int:
    root = Path(args.root or os.environ.get("PYSCOPE_MCP_ROOT") or os.getcwd()).resolve()
    package = args.package or os.environ.get("PYSCOPE_MCP_PACKAGE")
    out = _index_path(args.output)
    if not out.is_absolute():
        out = (root / out).resolve()

    print(f"[pyscope-mcp] building index: root={root} package={package or root.name}", file=sys.stderr)
    from pyscope_mcp.analyzer import build_nodes_with_report

    nodes, miss_report, skeletons, file_shas = build_nodes_with_report(
        root, package=package or root.name
    )

    # Inline missed_callers into index.json at build time (Law 3: single artifact).
    # Aggregate per-caller pattern counts from the flat unresolved_calls list.
    # The miss_report["unresolved_calls"] is a flat list of dicts with keys:
    #   "caller", "pattern", "file", "line", "snippet"
    missed_callers: dict[str, dict[str, int]] = {}
    for entry in miss_report.get("unresolved_calls", []):
        caller = entry["caller"]
        pattern = entry["pattern"]
        if caller not in missed_callers:
            missed_callers[caller] = {}
        missed_callers[caller][pattern] = missed_callers[caller].get(pattern, 0) + 1

    # Capture git SHA at build time (Law 3: git-in-sync graph).
    # Failure to capture (not a git checkout, git absent) → git_sha=None, not an error.
    git_sha: str | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip() or None
    except FileNotFoundError:
        pass  # git binary not installed

    idx = CallGraphIndex.from_nodes(
        root, nodes, skeletons=skeletons, file_shas=file_shas, missed_callers=missed_callers,
        git_sha=git_sha,
    )
    idx.save(out)

    # Write misses sidecar next to index.json
    misses_path = out.parent / "misses.json"
    misses_path.write_text(json.dumps(miss_report, indent=2))

    stats = idx.stats()
    print(
        f"[pyscope-mcp] wrote {out} "
        f"({stats['functions']} functions, {stats['function_edges']} edges, "
        f"{stats['modules']} modules)",
        file=sys.stderr,
    )

    _print_summary(miss_report, misses_path)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from pyscope_mcp import server

    root = Path(args.root or os.environ.get("PYSCOPE_MCP_ROOT") or os.getcwd()).resolve()
    index = _index_path(args.index)
    if not index.is_absolute():
        index = (root / index).resolve()

    # Do NOT exit early for missing/stale index. Pass the path to run_stdio
    # unconditionally; it will enter deferred-error mode and keep all tools
    # registered. This prevents the silent tool-disappearance on stale schema
    # or first-run before 'pyscope-mcp build' has been executed.
    server.run_stdio(index_path=index)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyscope-mcp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="compute the call graph and save it to disk")
    b.add_argument("--root", help="repo root (default: $PYSCOPE_MCP_ROOT or cwd)")
    b.add_argument("--package", help="root package name (default: root dir name)")
    b.add_argument("--output", help=f"index file (default: $PYSCOPE_MCP_INDEX or {DEFAULT_INDEX_PATH})")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("serve", help="start the MCP stdio server using a prebuilt index")
    s.add_argument("--root", help="repo root (default: $PYSCOPE_MCP_ROOT or cwd)")
    s.add_argument("--index", help=f"index file (default: $PYSCOPE_MCP_INDEX or {DEFAULT_INDEX_PATH})")
    s.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
