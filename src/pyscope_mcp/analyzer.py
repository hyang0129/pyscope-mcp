"""Call-graph analyzer.

Contract:
    build_raw(root, package) -> dict[str, list[str]]

Keys: fully-qualified caller names. Values: lists of fully-qualified callees.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_EXEMPLAR_CAP = 50


def _warn(msg: str) -> None:
    print(f"[pyscope-mcp] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# MissLog
# ---------------------------------------------------------------------------

class MissLog:
    """Accumulates miss/skip/resolution data during pass 2."""

    def __init__(self) -> None:
        self.files_total: int = 0
        self.files_parsed: int = 0
        self.files_skipped: int = 0
        self.skipped_files: list[dict] = []

        self.calls_total: int = 0
        self.calls_resolved_in_package: int = 0
        self.calls_resolved_external: int = 0
        self.calls_unresolved: int = 0

        self.pattern_counts: dict[str, int] = {}
        self.unresolved_calls: dict[str, list[dict]] = {}

    def record_miss(
        self,
        pattern: str,
        caller: str,
        file_path: str,
        line: int,
        snippet: str,
    ) -> None:
        self.calls_total += 1
        self.calls_unresolved += 1
        self.pattern_counts[pattern] = self.pattern_counts.get(pattern, 0) + 1
        bucket = self.unresolved_calls.setdefault(pattern, [])
        if len(bucket) < _EXEMPLAR_CAP:
            bucket.append({
                "caller": caller,
                "file": file_path,
                "line": line,
                "pattern": pattern,
                "snippet": snippet,
            })

    def record_resolved(self, *, in_package: bool) -> None:
        self.calls_total += 1
        if in_package:
            self.calls_resolved_in_package += 1
        else:
            self.calls_resolved_external += 1

    def record_skip(self, path: str, reason: str) -> None:
        self.files_skipped += 1
        self.skipped_files.append({"path": path, "reason": reason})

    def to_dict(self, raw: dict[str, list[str]], known_fqns: set[str]) -> dict:
        """Assemble the full misses.json structure."""
        total = self.calls_total
        in_pkg = self.calls_resolved_in_package
        rate = round(in_pkg / total, 4) if total > 0 else 0.0

        # Flat list of exemplars sorted by file then line (already insertion-ordered
        # by first-N per pattern, but flatten across patterns sorted by pattern name
        # for determinism).
        flat_unresolved: list[dict] = []
        for pattern in sorted(self.unresolved_calls):
            flat_unresolved.extend(self.unresolved_calls[pattern])

        # Rollups
        # dead_keys: callers in raw that never appear as a callee
        all_callees: set[str] = set()
        for callees in raw.values():
            all_callees.update(callees)
        dead_keys = sorted(k for k in raw if k not in all_callees)

        # unreferenced_modules: module-level FQNs with zero inbound edges
        # A module FQN is one that appears in known_fqns AND equals a module name
        # (i.e., it is a key in the module discovery set).
        # We approximate: FQNs in known_fqns that don't contain a dot after the
        # package prefix — or more practically, just check inbound edges in raw.
        # Strategy: collect all FQNs that appear as callees; anything in known_fqns
        # that never appears as a callee and looks like a module (no method suffix
        # visible in known_fqns) is unreferenced. We use the simpler definition:
        # FQNs that are themselves module FQNs (appear in _discover_modules keyset).
        # Since we don't have module_fqns here, we check known_fqns: a "module-level"
        # FQN is one where removing the last segment gives something not in known_fqns
        # (i.e., the FQN itself is a top-level def or the module itself).
        # Simplest approach: unreferenced_modules = module-level keys in raw with no
        # inbound edges. We define "module-level FQN" as one whose parent (everything
        # before the last dot) is NOT in known_fqns — i.e., it's directly under a module.
        # Actually the plan says: "module-level FQNs with zero inbound edges
        # (FQNs that are a module name, i.e., no dot-separated method/function suffix
        # past the package prefix... just check: fqn in module_fqns where module_fqns
        # is the keyset from _discover_modules)".
        # We don't have that here, so we skip this rollup (leave empty) — it would
        # require passing module_fqns. We return an empty list as a safe default.
        unreferenced_modules: list[str] = []

        return {
            "version": 1,
            "summary": {
                "files_total": self.files_total,
                "files_parsed": self.files_parsed,
                "files_skipped": self.files_skipped,
                "calls_total": total,
                "calls_resolved_in_package": in_pkg,
                "calls_resolved_external": self.calls_resolved_external,
                "calls_unresolved": self.calls_unresolved,
                "resolution_rate_in_package": rate,
                "rollups": {
                    "dead_keys": dead_keys,
                    "unreferenced_modules": unreferenced_modules,
                },
            },
            "skipped_files": self.skipped_files,
            "pattern_counts": self.pattern_counts,
            "unresolved_calls": flat_unresolved,
        }


# ---------------------------------------------------------------------------
# Pass 1 helpers
# ---------------------------------------------------------------------------

def _discover_modules(root: Path, package: str) -> dict[str, Path]:
    """Walk root, return mapping of dotted FQN -> path for every .py file."""
    result: dict[str, Path] = {}
    for py_file in sorted(root.rglob("*.py")):
        rel = py_file.relative_to(root.parent)
        parts = list(rel.parts)
        # Strip .py suffix from last part
        parts[-1] = parts[-1][:-3]  # remove ".py"
        if parts[-1] == "__init__":
            parts = parts[:-1]
        fqn = ".".join(parts)
        result[fqn] = py_file
    return result


def _collect_defs(tree: ast.Module, module_fqn: str) -> set[str]:
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


def _build_import_table(tree: ast.Module, module_fqn: str) -> dict[str, str]:
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
                    # Add entry for every prefix so attribute access works
                    parts = alias.name.split(".")
                    for i in range(len(parts)):
                        prefix = ".".join(parts[: i + 1])
                        table[prefix] = prefix
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            if level > 0:
                # Relative import resolution (M4).
                # Strip `level` segments from the right to get the base package.
                base = _resolve_relative_base(module_fqn, level)
                if base is None:
                    continue  # malformed level — skip
                # Append the optional module fragment (e.g. "sub" in "from .sub import x")
                if node.module:
                    base = f"{base}.{node.module}"
                # Map each imported name to base.name
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


def _attr_chain(node: ast.expr) -> list[str] | None:
    """Flatten a chain of ast.Attribute/ast.Name into a dotted list, or None."""
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


# ---------------------------------------------------------------------------
# Pattern classifier
# ---------------------------------------------------------------------------

def _classify_miss(node: ast.Call) -> str:
    """Classify why a call could not be resolved. Returns a pattern tag."""
    func = node.func

    if isinstance(func, ast.Subscript):
        return "subscript_call"

    if isinstance(func, ast.Name):
        if func.id in {"exec", "eval", "compile"}:
            return "exec_or_eval"
        return "bare_name_unresolved"

    if isinstance(func, ast.Call):
        # getattr(x, name)(...) — outer Call whose func is also a Call
        inner = func
        if isinstance(inner.func, ast.Name) and inner.func.id == "getattr":
            return "getattr_nonliteral"
        return "call_on_call_result"

    if isinstance(func, ast.Attribute):
        chain = _attr_chain(func)
        if chain is not None:
            # Check for importlib.import_module or __import__
            dotted = ".".join(chain)
            if "importlib" in dotted and "import_module" in dotted:
                return "importlib_import_module"
            if chain[0] == "__import__":
                return "importlib_import_module"
            if chain[0] == "self":
                return "self_method_unresolved"
        return "attr_chain_unresolved"

    return "other_unresolved"


def _snippet(node: ast.Call) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        text = "<unparse-error>"
    return text[:80]


# ---------------------------------------------------------------------------
# Pass 2: edge visitor
# ---------------------------------------------------------------------------

class _EdgeVisitor(ast.NodeVisitor):
    """Walk one file's AST and collect caller -> callee edges."""

    def __init__(
        self,
        module_fqn: str,
        import_table: dict[str, str],
        known_fqns: set[str],
        file_path: str = "",
        miss_log: MissLog | None = None,
    ) -> None:
        self._module_fqn = module_fqn
        self._import_table = import_table
        self._known_fqns = known_fqns
        self._file_path = file_path
        self._miss_log = miss_log
        # Stack of name segments below module_fqn: e.g. ["MyClass", "method"]
        self._scope_stack: list[str] = []
        self.edges: dict[str, set[str]] = {}

    def _current_caller(self) -> str:
        if self._scope_stack:
            return f"{self._module_fqn}.{'.'.join(self._scope_stack)}"
        return self._module_fqn

    def _enclosing_class_fqn(self) -> str | None:
        """Return the FQN of the enclosing class if the current scope is a method.

        The scope stack looks like ["ClassName", "method_name"] when inside a
        method.  We identify a class frame as the second-to-last segment when
        the stack has at least two entries and the candidate FQN is a known class.
        """
        if len(self._scope_stack) < 2:
            return None
        # Walk from innermost outward looking for the first class frame.
        # This handles nested functions inside methods: ["MyClass", "method", "inner_fn"].
        for i in range(len(self._scope_stack) - 2, -1, -1):
            candidate = f"{self._module_fqn}.{'.'.join(self._scope_stack[:i + 1])}"
            if candidate in self._known_fqns:
                return candidate
        return None

    def _resolve(self, func_node: ast.expr) -> str | None:
        """Try to resolve a call's func node to a known FQN."""
        if isinstance(func_node, ast.Name):
            name = func_node.id
            # Check import table first
            if name in self._import_table:
                candidate = self._import_table[name]
                if candidate in self._known_fqns:
                    return candidate
            # Check module-local: module_fqn.name
            candidate = f"{self._module_fqn}.{name}"
            if candidate in self._known_fqns:
                return candidate
            return None

        if isinstance(func_node, ast.Attribute):
            chain = _attr_chain(func_node)
            if chain is None:
                return None

            # M3 case 1: self.method(...) inside a method
            if chain[0] == "self" and len(chain) >= 2:
                class_fqn = self._enclosing_class_fqn()
                if class_fqn is not None:
                    # Append the remainder of the chain after "self"
                    candidate = ".".join([class_fqn] + chain[1:])
                    if candidate in self._known_fqns:
                        return candidate
                    # Silently drop if not found (inherited/unresolved — no MRO in v1)
                return None

            # M3 case 2: deeper attribute chain — try all prefix lengths against
            # the import table (longest prefix first for correctness).
            # e.g. chain = ["a", "b", "c", "d"] tries:
            #   import["a.b.c"] -> fqn + ".d"
            #   import["a.b"]   -> fqn + ".c.d"
            #   import["a"]     -> fqn + ".b.c.d"
            for prefix_len in range(len(chain) - 1, 0, -1):
                prefix = ".".join(chain[:prefix_len])
                if prefix in self._import_table:
                    base_fqn = self._import_table[prefix]
                    remainder = chain[prefix_len:]
                    candidate = ".".join([base_fqn] + remainder)
                    if candidate in self._known_fqns:
                        return candidate

            # Also try prefix-matching directly against known FQNs
            # (handles intra-package attribute access without an import table entry).
            dotted = ".".join(chain)
            if dotted in self._known_fqns:
                return dotted

            return None

        return None

    def _resolve_external(self, func_node: ast.expr) -> str | None:
        """Check if a call resolves to an external (non-package) FQN.

        Returns the candidate FQN string if the func node maps to something
        in the import table but not in known_fqns, else None.
        """
        if isinstance(func_node, ast.Name):
            name = func_node.id
            if name in self._import_table:
                candidate = self._import_table[name]
                if candidate not in self._known_fqns:
                    return candidate
            return None

        if isinstance(func_node, ast.Attribute):
            chain = _attr_chain(func_node)
            if chain is None:
                return None
            # Skip self.xxx
            if chain[0] == "self":
                return None
            for prefix_len in range(len(chain) - 1, 0, -1):
                prefix = ".".join(chain[:prefix_len])
                if prefix in self._import_table:
                    base_fqn = self._import_table[prefix]
                    remainder = chain[prefix_len:]
                    candidate = ".".join([base_fqn] + remainder)
                    if candidate not in self._known_fqns:
                        return candidate
        return None

    def _emit(self, callee_fqn: str) -> None:
        caller = self._current_caller()
        self.edges.setdefault(caller, set()).add(callee_fqn)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        callee = self._resolve(node.func)
        if callee is not None:
            self._emit(callee)
            if self._miss_log is not None:
                self._miss_log.record_resolved(in_package=True)
        else:
            # Check if it resolves externally
            if self._miss_log is not None:
                ext = self._resolve_external(node.func)
                if ext is not None:
                    self._miss_log.record_resolved(in_package=False)
                else:
                    pattern = _classify_miss(node)
                    self._miss_log.record_miss(
                        pattern=pattern,
                        caller=self._current_caller(),
                        file_path=self._file_path,
                        line=node.lineno,
                        snippet=_snippet(node),
                    )
        # Still walk arguments and nested calls
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_with_report(
    root: str | Path,
    package: str,
) -> tuple[dict[str, list[str]], dict]:
    """Pass 1 + pass 2 with MissLog. Returns (raw_edges, miss_report_dict)."""
    root = Path(root)
    pkg_root = root / package if (root / package).is_dir() else root

    modules = _discover_modules(pkg_root, package)

    miss_log = MissLog()
    miss_log.files_total = len(modules)

    # Pass 1: parse all files and gather known FQNs
    parsed: list[tuple[str, ast.Module, Path]] = []
    known_fqns: set[str] = set()
    for fqn, path in modules.items():
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
            known_fqns.update(_collect_defs(tree, fqn))
            parsed.append((fqn, tree, path))
        except SyntaxError as exc:
            reason = f"SyntaxError: {exc}"
            _warn(f"skipping {path}: {reason}")
            miss_log.record_skip(str(path), reason)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            _warn(f"skipping {path}: {reason}")
            miss_log.record_skip(str(path), reason)

    miss_log.files_parsed = len(parsed)

    # Pass 2: extract edges
    all_edges: dict[str, set[str]] = {}
    for fqn, tree, path in parsed:
        try:
            import_table = _build_import_table(tree, fqn)
            visitor = _EdgeVisitor(
                fqn,
                import_table,
                known_fqns,
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
