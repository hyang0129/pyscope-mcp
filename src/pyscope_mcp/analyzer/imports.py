"""Import-table construction for a single module.

`build_import_table` maps a local name used in a module to the absolute
dotted FQN it resolves to, for both absolute and relative imports.
"""

from __future__ import annotations

import ast


def _resolve_relative_base(module_fqn: str, level: int) -> str | None:
    """Compute the base package FQN for a relative import.

    Strips `level` segments from the right of `module_fqn`:
      - level=1 strips the last segment (current file → current package)
      - level=2 strips two segments (go up one more package level)

    Returns None if level exceeds the number of segments.
    """
    parts = module_fqn.split(".")
    if level > len(parts):
        return None
    return ".".join(parts[:-level])


def build_import_table(tree: ast.Module, module_fqn: str) -> dict[str, str]:
    """Map local names -> resolved absolute FQNs for module-level imports.

    Handles absolute imports and relative imports (level > 0).
    """
    table: dict[str, str] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # import foo.bar        -> {"foo": "foo", "foo.bar": "foo.bar"}
                # import foo.bar as fb  -> {"fb": "foo.bar"}
                if alias.asname:
                    table[alias.asname] = alias.name
                else:
                    parts = alias.name.split(".")
                    for i in range(len(parts)):
                        prefix = ".".join(parts[: i + 1])
                        table[prefix] = prefix
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            if level > 0:
                base = _resolve_relative_base(module_fqn, level)
                if base is None:
                    continue
                if node.module:
                    base = f"{base}.{node.module}"
                for alias in node.names:
                    local = alias.asname if alias.asname else alias.name
                    table[local] = f"{base}.{alias.name}"
            else:
                if node.module is None:
                    continue
                for alias in node.names:
                    local = alias.asname if alias.asname else alias.name
                    table[local] = f"{node.module}.{alias.name}"
    return table
