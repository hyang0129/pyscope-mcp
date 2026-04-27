"""MissLog and miss-pattern classification."""

from __future__ import annotations

import ast
import builtins
import collections
from collections import Counter

from .resolution import attr_chain

_EXEMPLAR_CAP = 50

# Python builtin callable names that are safe to accept without resolution.
# exec/eval/compile are intentionally excluded (exec_or_eval bucket — users
# want those visible).  Names starting with "_" (including __import__) are
# already filtered out by the startswith("_") guard above.
# getattr and super were previously excluded because they had "dedicated tags",
# but those tags only apply when getattr/super is the *inner* function of a
# nested Call node (e.g. getattr(...)() or super().__init__()).  A bare call
# like getattr(obj, 'x') or super() has func=Name(id='getattr'/'super') and
# must be accepted here instead of leaking to bare_name_unresolved.
BUILTIN_FUNCTION_NAMES: frozenset[str] = (
    frozenset(n for n in dir(builtins) if not n.startswith("_"))
    - {"exec", "eval", "compile"}
)

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
    "fromkeys",
    # set
    "add", "discard", "union", "intersection", "difference",
    "issubset", "issuperset",
    # str
    "format", "format_map", "join", "split", "rsplit", "splitlines",
    "strip", "lstrip", "rstrip", "replace", "startswith", "endswith",
    "lower", "upper", "title", "capitalize", "casefold", "encode",
    "find", "rfind",
    # str — additional predicates and transformations
    "isalnum", "isdigit", "isalpha", "isspace", "isnumeric", "isidentifier",
    "isdecimal", "isprintable", "istitle", "islower", "isupper",
    "decode", "from_bytes", "to_bytes",
    "zfill", "center", "ljust", "rjust", "expandtabs", "swapcase",
    "partition", "rpartition", "translate", "maketrans",
    # str — Python 3.9+ additions (removesuffix / removeprefix)
    "removesuffix", "removeprefix",
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
    "is_file", "is_dir", "with_suffix", "joinpath",
    # additional pathlib.Path methods
    "with_name", "with_stem", "rename", "absolute", "touch", "rmdir", "chmod",
    "is_absolute", "is_relative_to", "samefile", "expanduser", "home", "cwd",
    "replace", "symlink_to", "hardlink_to", "readlink", "owner", "group",
    "lstat", "match",
    "as_posix", "as_uri",
})

FUTURES_METHODS: frozenset[str] = frozenset({
    "submit", "map", "result", "shutdown", "cancel", "done", "add_done_callback",
})

# PIL / Pillow Image and ImageDraw method names.
PIL_METHODS: frozenset[str] = frozenset({
    "new", "open", "save", "convert", "resize", "paste", "putalpha", "close",
    "textbbox", "textsize", "text", "load_default", "truetype",
    "Draw", "alpha_composite", "crop", "rotate", "transpose", "thumbnail",
    "getpixel", "putpixel", "fromarray",
    # ImageDraw drawing primitives
    "rectangle", "line", "ellipse", "polygon", "arc", "chord",
    "pieslice", "point", "rounded_rectangle",
})

# loguru Logger method names.
LOGURU_METHODS: frozenset[str] = frozenset({
    "info", "debug", "warning", "error", "critical", "exception",
    "trace", "success", "log", "opt", "bind", "patch", "level",
    "configure", "add", "remove", "complete",
})

# re module method names — covers both re.Pattern and re.Match objects.
RE_METHODS: frozenset[str] = frozenset({
    "search", "match", "fullmatch", "finditer", "findall",
    "sub", "subn", "split",
    "start", "end", "group", "groups", "groupdict", "span", "expand",
})

# datetime / date / time method names.
# Deliberately excludes "replace" to avoid clobbering str.replace / Path.replace.
DATETIME_METHODS: frozenset[str] = frozenset({
    "now", "today", "utcnow", "isoformat", "strftime", "strptime",
    "fromisoformat", "fromtimestamp", "timestamp", "date", "time",
    "astimezone", "combine", "weekday",
})

# difflib SequenceMatcher / Differ method names.
DIFFLIB_METHODS: frozenset[str] = frozenset({
    "ratio", "get_matching_blocks", "get_opcodes",
    "quick_ratio", "real_quick_ratio",
})

# wave module handle method names.
WAVE_METHODS: frozenset[str] = frozenset({
    "getnframes", "getnchannels", "getsampwidth", "getframerate",
    "getcomptype", "getcompname", "readframes",
    "setnframes", "setnchannels", "setsampwidth", "setframerate",
    "writeframes", "writeframesraw", "rewind", "tell",
})

# Pydantic BaseModel base-name heuristic.  A class is treated as a BaseModel
# subclass if any (transitive) base's *short name* matches one of these.  We
# use short names because the full FQN of the external base is not known
# without runtime import resolution.
PYDANTIC_BASE_NAMES: frozenset[str] = frozenset({"BaseModel", "RootModel"})

PYDANTIC_METHODS: frozenset[str] = frozenset({
    "model_dump", "model_dump_json", "model_validate", "model_validate_json",
    "model_copy", "model_fields",
    "model_rebuild", "model_json_schema", "model_parametrized_name",
})

# File-like object method names.  Gated conservatively: only classified as
# io_method_call when chain[0] is a known file-var name OR when chain is None
# (receiver is a Call result like open(p).read()).
IO_METHODS: frozenset[str] = frozenset({
    "read", "write", "readline", "readlines", "writelines", "seek", "tell",
    "flush", "close", "truncate", "readable", "writable", "seekable",
    "getvalue", "getbuffer",
})

# hashlib hash-object method names.  These are distinctive enough to classify
# without any chain-root guard.
HASHLIB_METHODS: frozenset[str] = frozenset({
    "hexdigest", "digest", "update", "copy",
})

# random / Random instance method names.
RANDOM_METHODS: frozenset[str] = frozenset({
    "random", "randint", "choice", "choices", "sample", "shuffle", "uniform",
    "seed", "randrange", "getrandbits", "gauss",
})

# argparse ArgumentParser / subparsers method names.  Distinctive enough to
# classify without chain-root guard.
ARGPARSE_METHODS: frozenset[str] = frozenset({
    "add_argument", "add_subparsers", "add_parser", "parse_args",
    "parse_known_args", "set_defaults", "add_argument_group",
    "add_mutually_exclusive_group", "print_help",
})

# Anthropic SDK client method names.  Generic names (create/list/retrieve)
# are only classified as anthropic when the chain contains a known client-var
# or an intermediate Anthropic namespace attr (messages/beta/completions).
ANTHROPIC_METHODS: frozenset[str] = frozenset({
    "create", "stream", "count_tokens", "retrieve", "list",
})

# Known Anthropic client variable names that make ANTHROPIC_METHODS unambiguous.
_ANTHROPIC_CLIENT_VARS: frozenset[str] = frozenset({
    "client", "anthropic_client", "_client",
})

# Intermediate attribute names that indicate an Anthropic SDK chain.
_ANTHROPIC_ATTRS: frozenset[str] = frozenset({
    "messages", "beta", "completions",
})

# requests Response method names — gated on chain-root variable name to avoid
# false positives on in-package .json() style methods.
REQUESTS_METHODS: frozenset[str] = frozenset({
    "json", "raise_for_status", "iter_content", "iter_lines",
})

# Known requests response variable names.
_REQUESTS_RESP_VARS: frozenset[str] = frozenset({
    "r", "resp", "response",
})

# Known IO file variable names — used to gate io_method_call conservatively.
_IO_FILE_VARS: frozenset[str] = frozenset({
    "f", "fh", "fp", "file", "tmp", "buf", "stream",
})

# Curated list of stdlib top-level module names.  Pinned rather than
# enumerated at runtime (sys.stdlib_module_names) for determinism.
STDLIB_MODULES: frozenset[str] = frozenset({
    "sys", "os", "json", "re", "pathlib", "functools", "itertools",
    "collections", "typing", "dataclasses", "math", "time", "datetime",
    "shutil", "subprocess", "logging", "tempfile", "uuid", "hashlib",
    "base64", "enum", "asyncio", "contextlib", "inspect", "warnings",
    "copy", "traceback", "threading", "queue", "pickle", "csv", "io",
})

# Pattern tags from classify_miss that route to record_accepted (vs record_miss).
ACCEPTED_PATTERNS: frozenset[str] = frozenset({
    "builtin_function_call",
    "stdlib_method_call",
    "builtin_method_call", "pathlib_method_call", "futures_method_call",
    "pydantic_method_call", "pil_method_call", "wave_method_call",
    "loguru_method_call", "re_method_call", "datetime_method_call",
    "difflib_method_call",
    # M8 additions
    "io_method_call", "hashlib_method_call", "random_method_call",
    "argparse_method_call", "anthropic_method_call", "requests_method_call",
    # Handler #19: external-factory local-var accept buckets
    "httpx_method_call", "boto3_method_call", "typer_method_call", "googleapi_method_call",
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

        # v5 (epic #76 child #1): dead_keys are callers with no incoming
        # call edges.  We accumulate the caller / callee universes here so
        # to_dict() can derive dead_keys without taking the legacy ``raw``
        # dict as a parameter.
        self._call_callers: set[str] = set()
        self._call_callees: set[str] = set()

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

    def record_call_edge(self, caller: str, callee: str) -> None:
        """Track a resolved call edge for dead_keys computation.

        Pipeline calls this for every (caller, callee) pair that lands in the
        final edge dict.  ``to_dict()`` derives ``dead_keys`` as
        ``sorted(callers - callees)`` from the accumulated sets — the same
        semantics the pre-migration ``to_dict(raw, known_fqns)`` produced
        from the ``raw`` dict.
        """
        self._call_callers.add(caller)
        self._call_callees.add(callee)

    def to_dict(self) -> dict:
        """Assemble the full misses.json structure."""
        total = self.calls_total
        in_pkg = self.calls_resolved_in_package
        ext = self.calls_resolved_external
        unres = self.calls_unresolved
        rate = round(in_pkg / total, 4) if total > 0 else 0.0
        unresolved_rate = round(unres / total, 4) if total > 0 else 0.0
        effective_denom = in_pkg + ext + unres
        effective_resolution = (
            round((in_pkg + ext) / effective_denom, 4) if effective_denom > 0 else 0.0
        )

        flat_unresolved: list[dict] = []
        for pattern in sorted(self.unresolved_calls):
            flat_unresolved.extend(self.unresolved_calls[pattern])

        # dead_keys: callers that no edge points to — derived from per-edge
        # tracking in record_call_edge (pre-migration this was computed
        # from the ``raw`` dict argument).
        dead_keys = sorted(self._call_callers - self._call_callees)

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
                "unresolved_rate": unresolved_rate,
                "effective_resolution": effective_resolution,
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


def _has_pydantic_ancestor(
    class_fqn: str,
    class_bases: dict[str, list[str]],
    _seen: frozenset[str] = frozenset(),
) -> bool:
    """Shallow heuristic: True if `class_fqn` has any base whose short name is
    in PYDANTIC_BASE_NAMES, walking in-package ancestors transitively.

    External bases (not tracked in class_bases) are checked by short name
    only — so `class Foo(BaseModel)` matches even though `BaseModel` is not
    in class_bases.
    """
    if class_fqn in _seen:
        return False
    seen = _seen | {class_fqn}
    for base in class_bases.get(class_fqn, []):
        short = base.rsplit(".", 1)[-1]
        if short in PYDANTIC_BASE_NAMES:
            return True
        if _has_pydantic_ancestor(base, class_bases, seen):
            return True
    return False


def _chain_has_anthropic_attr(chain: list[str]) -> bool:
    """Return True if any intermediate element of chain is an Anthropic namespace attr."""
    # chain[-1] is the method, chain[0] is the root — check the middle attrs
    return bool(set(chain[1:-1]) & _ANTHROPIC_ATTRS)


def classify_miss(
    node: ast.Call,
    *,
    enclosing_class_fqn: str | None = None,
    class_bases: dict[str, list[str]] | None = None,
    import_table: dict[str, str] | None = None,
) -> str:
    """Classify why a call could not be resolved. Returns a pattern tag.

    The optional kwargs let the classifier recognise `super().method(...)`
    on pydantic BaseModel subclasses and route them to pydantic_method_call
    instead of super_unresolved, and aliased stdlib calls (e.g.
    `import sys as s; s.exit()`) to stdlib_method_call.
    """
    func = node.func

    if isinstance(func, ast.Subscript):
        return "subscript_call"

    if isinstance(func, ast.Name):
        if func.id in {"exec", "eval", "compile"}:
            return "exec_or_eval"
        if func.id in BUILTIN_FUNCTION_NAMES:
            return "builtin_function_call"
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
            # A method called on a known literal type (str, bytes, int, float,
            # list, dict, set, tuple) — route to builtin_method_call when the
            # method name is in the whitelist; otherwise keep literal_method_call.
            if isinstance(func, ast.Attribute) and func.attr in BUILTIN_COLLECTION_METHODS:
                return "builtin_method_call"
            return "literal_method_call"
        # super().foo() — value of the outer Attribute is a Call to super()
        if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name) and cur.func.id == "super":
            # Pydantic BaseModel subclass calling super().__init__(**data) etc.
            # Route to pydantic_method_call if the enclosing class has a
            # BaseModel ancestor (shallow base-name heuristic).
            if (
                enclosing_class_fqn is not None
                and class_bases is not None
                and _has_pydantic_ancestor(enclosing_class_fqn, class_bases)
            ):
                return "pydantic_method_call"
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
            # Stdlib alias check: import sys as s; s.exit() or import os; os.path.join()
            # None (kwarg omitted) → skip entirely for backward-compat; {} (passed empty) → direct-name fallback still fires.
            if import_table is not None:
                root_fqn = import_table.get(chain[0])
                if root_fqn is not None and root_fqn.split(".")[0] in STDLIB_MODULES:
                    return "stdlib_method_call"
                # Also handle direct module names: `import sys` → chain[0]=='sys' in STDLIB_MODULES
                if chain[0] in STDLIB_MODULES and chain[0] not in import_table:
                    return "stdlib_method_call"
                # Parameter injection pattern: def f(sys_module): sys_module.exit(...)
                # Common in testable CLI code where the caller passes `sys` as an argument
                # to allow tests to intercept exit(). Recognize <stdlib_name>_module and
                # <stdlib_name>_lib naming conventions.
                root = chain[0]
                for suffix in ("_module", "_lib"):
                    if root.endswith(suffix):
                        stem = root[: -len(suffix)]
                        if stem in STDLIB_MODULES:
                            return "stdlib_method_call"
            method = chain[-1]
            if chain[0] != "self":
                # ClassName.__new__(...) — inherited from object/type; never a
                # user-defined in-package method worth tracking.  Accept as a
                # known builtin-method call so it doesn't pollute the miss log.
                # Note: if the receiver resolved to an in-package class, the
                # visitor's _resolve_expr already returned it via
                # infer_call_class_type and this branch is never reached.
                if method == "__new__":
                    return "builtin_method_call"
                # Chain-root special case: known logger variable names
                if chain[0] in {"logger", "log", "logging"}:
                    return "loguru_method_call"
                # Chain-root special case: random / rng
                if chain[0] in {"random", "rng"}:
                    return "random_method_call"
                # Chain-root special case: requests response variables
                # Only route when both chain-root AND method match.
                if chain[0] in _REQUESTS_RESP_VARS and method in REQUESTS_METHODS:
                    return "requests_method_call"
                if method in BUILTIN_COLLECTION_METHODS:
                    return "builtin_method_call"
                if method in PATHLIB_METHODS:
                    return "pathlib_method_call"
                if method in FUTURES_METHODS:
                    return "futures_method_call"
                if method in PYDANTIC_METHODS:
                    return "pydantic_method_call"
                if method in PIL_METHODS:
                    return "pil_method_call"
                if method in WAVE_METHODS:
                    return "wave_method_call"
                if method in LOGURU_METHODS:
                    return "loguru_method_call"
                if method in RE_METHODS:
                    return "re_method_call"
                if method in DATETIME_METHODS:
                    return "datetime_method_call"
                if method in DIFFLIB_METHODS:
                    return "difflib_method_call"
                if method in HASHLIB_METHODS:
                    return "hashlib_method_call"
                if method in RANDOM_METHODS:
                    return "random_method_call"
                if method in ARGPARSE_METHODS:
                    return "argparse_method_call"
                # Anthropic: gate strictly — only when chain-root is a known
                # client var OR an intermediate attr is an Anthropic namespace.
                if method in ANTHROPIC_METHODS:
                    if chain[0] in _ANTHROPIC_CLIENT_VARS or _chain_has_anthropic_attr(chain):
                        return "anthropic_method_call"
                # IO: gate conservatively on chain[0] being a known file-var name.
                if method in IO_METHODS and chain[0] in _IO_FILE_VARS:
                    return "io_method_call"
        else:
            # chain is None: receiver is a non-Name/non-Attribute expression
            # (e.g. BinOp, Call, Subscript).  Fall back to method-name lookup.
            method = func.attr
            if method in PATHLIB_METHODS:
                return "pathlib_method_call"
            if method in PIL_METHODS:
                return "pil_method_call"
            if method in WAVE_METHODS:
                return "wave_method_call"
            if method in BUILTIN_COLLECTION_METHODS:
                return "builtin_method_call"
            if method in FUTURES_METHODS:
                return "futures_method_call"
            if method in PYDANTIC_METHODS:
                return "pydantic_method_call"
            if method in LOGURU_METHODS:
                return "loguru_method_call"
            if method in RE_METHODS:
                return "re_method_call"
            if method in DATETIME_METHODS:
                return "datetime_method_call"
            if method in DIFFLIB_METHODS:
                return "difflib_method_call"
            if method in HASHLIB_METHODS:
                return "hashlib_method_call"
            if method in RANDOM_METHODS:
                return "random_method_call"
            if method in ARGPARSE_METHODS:
                return "argparse_method_call"
            # IO chain-None: receiver is a call result like open(p).read() —
            # safe to classify without chain-root guard.
            if method in IO_METHODS:
                return "io_method_call"
            # NOTE: anthropic and requests are NOT classified in chain-None
            # fallback — their method names are too generic without chain context.
        return "attr_chain_unresolved"

    return "other_unresolved"


def snippet(node: ast.Call) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        text = "<unparse-error>"
    return text[:80]
