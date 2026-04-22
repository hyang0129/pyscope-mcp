from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pycg_mcp.graph import CallGraphIndex

DEFAULT_INDEX_PATH = ".pycg-mcp/index.json"


def _index_path(value: str | None) -> Path:
    return Path(value or os.environ.get("PYCG_MCP_INDEX") or DEFAULT_INDEX_PATH).resolve()


def cmd_build(args: argparse.Namespace) -> int:
    root = Path(args.root or os.environ.get("PYCG_MCP_ROOT") or os.getcwd()).resolve()
    package = args.package or os.environ.get("PYCG_MCP_PACKAGE")
    out = _index_path(args.output)
    if not out.is_absolute():
        out = (root / out).resolve()

    print(f"[pycg-mcp] building index: root={root} package={package or root.name}", file=sys.stderr)
    idx = CallGraphIndex.build(root, package=package)
    idx.save(out)
    stats = idx.stats()
    print(
        f"[pycg-mcp] wrote {out} "
        f"({stats['functions']} functions, {stats['function_edges']} edges, "
        f"{stats['modules']} modules)",
        file=sys.stderr,
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from pycg_mcp import server

    root = Path(args.root or os.environ.get("PYCG_MCP_ROOT") or os.getcwd()).resolve()
    index = _index_path(args.index)
    if not index.is_absolute():
        index = (root / index).resolve()
    if not index.exists():
        print(
            f"[pycg-mcp] no index at {index}. Run 'pycg-mcp build' first.",
            file=sys.stderr,
        )
        return 2

    server.run_stdio(index_path=index)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pycg-mcp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="compute the call graph and save it to disk")
    b.add_argument("--root", help="repo root (default: $PYCG_MCP_ROOT or cwd)")
    b.add_argument("--package", help="package name for pycg (default: root dir name)")
    b.add_argument("--output", help=f"index file (default: $PYCG_MCP_INDEX or {DEFAULT_INDEX_PATH})")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("serve", help="start the MCP stdio server using a prebuilt index")
    s.add_argument("--root", help="repo root (default: $PYCG_MCP_ROOT or cwd)")
    s.add_argument("--index", help=f"index file (default: $PYCG_MCP_INDEX or {DEFAULT_INDEX_PATH})")
    s.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
