"""MissLog and miss-pattern classification."""

from __future__ import annotations

import ast
import collections
from collections import Counter

from .resolution import attr_chain

_EXEMPLAR_CAP = 50

# Canonical method names for built-in container / string types.
# These are unresolvable via static analysis (the receiver could be anything);
# we accept them as a known-unresolvable bucket rather than polluting the
# actionable miss counts.
BUILTIN_COLLECTION_METHODS: frozenset[str] = frozenset({
    # list
    "append", "extend", "insert", "pop", "remove", "clear", "sort",
    "reverse", "index", "count",
    # dict
    "get", "keys", "values", "items", "update", "setdefault", "copy",
    # set
    "add", "discard", "union", "intersection", "difference",
    "issubset", "issuperset",
    # str
    "format", "format_map", "join", "split", "rsplit", "splitlines",
    "strip", "lstrip", "rstrip", "replace", "startswith", "endswith",
    "lower", "upper", "title", "capitalize", "casefold", "encode",
    "find", "rfind",
})

# Heuristic: method-name-only whitelists for stable external libraries.
# We cannot infer receiver types statically, so we accept any attribute call
# whose method name appears in these sets.  This is a noise-reduction measure
# (moves calls from unresolved_calls to accepted_counts) — it does NOT produce
# resolved edges.  False-positive rate is low for these names in practice, but
# any in-package class that defines the same method will be resolved earlier by
# the self/MRO path and never reach classify_miss.

PATHLIB_METHODS: frozenset[str] = frozenset({
    "exists", "mkdir", "read_text", "read_bytes", "write_text", "write_bytes",
    "stat", "relative_to", "glob", "iterdir", "resolve", "unlink",
    "is_file", "is_dir", "parent", "with_suffix", "joinpath",
})

FUTURES_METHODS: frozenset[str] = frozenset({
    "result", "shutdown", "cancel", "done", "add_done_callback",
})

PYDANTIC_METHODS: frozenset[str] = frozenset({
    "model_dump", "model_dump_json", "model_validate", "model_validate_json",
    "model_copy", "model_fields",
})


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
        self.calls_accepted: int = 0

        self.pattern_counts: dict[str, int] = {}
        self.unresolved_calls: dict[str, list[dict]] = {}
        self.accepted_counts: dict[str, int] = {}
        self.builtin_method_modules: Counter[str] = Counter()

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

    def record_accepted(self, pattern: str, file_path: str) -> None:
        """Record a call that matched an accepted-miss pattern (e.g. builtin method)."""
        self.calls_total += 1
        self.calls_accepted += 1
        self.accepted_counts[pattern] = self.accepted_counts.get(pattern, 0) + 1
        if pattern == "builtin_method_call":
            self.builtin_method_modules[file_path] += 1

    def to_dict(self, raw: dict[str, list[str]], known_fqns: set[str]) -> dict:
        """Assemble the full misses.json structure."""
        total = self.calls_total
        in_pkg = self.calls_resolved_in_package
        rate = round(in_pkg / total, 4) if total > 0 else 0.0

        flat_unresolved: list[dict] = []
        for pattern in sorted(self.unresolved_calls):
            flat_unresolved.extend(self.unresolved_calls[pattern])

        all_callees: set[str] = set()
        for callees in raw.values():
            all_callees.update(callees)
        dead_keys = sorted(k for k in raw if k not in all_callees)

        # unreferenced_modules: requires module_fqns keyset not threaded here.
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
                "calls_accepted": self.calls_accepted,
                "accepted_counts": dict(self.accepted_counts),
                "resolution_rate_in_package": rate,
                "rollups": {
                    "dead_keys": dead_keys,
                    "unreferenced_modules": unreferenced_modules,
                    "builtin_method_modules": [
                        {"module": p, "count": n}
                        for p, n in self.builtin_method_modules.most_common(5)
                    ],
                },
            },
            "skipped_files": self.skipped_files,
            "pattern_counts": self.pattern_counts,
            "unresolved_calls": flat_unresolved,
        }


def classify_miss(node: ast.Call) -> str:
    """Classify why a call could not be resolved. Returns a pattern tag."""
    func = node.func

    if isinstance(func, ast.Subscript):
        return "subscript_call"

    if isinstance(func, ast.Name):
        if func.id in {"exec", "eval", "compile"}:
            return "exec_or_eval"
        return "bare_name_unresolved"

    if isinstance(func, ast.Call):
        inner = func
        if isinstance(inner.func, ast.Name) and inner.func.id == "getattr":
            return "getattr_nonliteral"
        # super().__init__() and similar — super() returns a proxy; if the
        # outer call couldn't be resolved, tag it distinctly so we notice
        # unresolved super dispatch (non-in-package parent).
        if isinstance(inner.func, ast.Name) and inner.func.id == "super":
            return "super_unresolved"
        return "call_on_call_result"

    if isinstance(func, ast.Attribute):
        # Walk to the chain root to detect literal method calls (e.g. '\n'.join(...))
        cur: ast.expr = func
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Constant):
            return "literal_method_call"
        # super().foo() — value of the outer Attribute is a Call to super()
        if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name) and cur.func.id == "super":
            return "super_unresolved"

        chain = attr_chain(func)
        if chain is not None:
            dotted = ".".join(chain)
            if "importlib" in dotted and "import_module" in dotted:
                return "importlib_import_module"
            if chain[0] == "__import__":
                return "importlib_import_module"
            if chain[0] == "self":
                return "self_method_unresolved"
            method = chain[-1]
            if chain[0] != "self":
                if method in BUILTIN_COLLECTION_METHODS:
                    return "builtin_method_call"
                if method in PATHLIB_METHODS:
                    return "pathlib_method_call"
                if method in FUTURES_METHODS:
                    return "futures_method_call"
                if method in PYDANTIC_METHODS:
                    return "pydantic_method_call"
        return "attr_chain_unresolved"

    return "other_unresolved"


def snippet(node: ast.Call) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        text = "<unparse-error>"
    return text[:80]
