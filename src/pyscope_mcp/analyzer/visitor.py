"""AST visitor: walk one file, emit caller -> callee edges."""

from __future__ import annotations

import ast

from .discovery import SENTINEL_BUILTIN_TYPES, resolve_annotation
from .misses import (
    ACCEPTED_PATTERNS,
    BUILTIN_COLLECTION_METHODS,
    PATHLIB_METHODS,
    MissLog,
    classify_miss,
    snippet,
)
from .resolution import (
    EXTERNAL_FACTORY_BUCKETS,
    EXTERNAL_SELF_RETURNING,
    ResolveCtx,
    attr_chain,
    dispatcher_callable_arg,
    infer_call_class_type,
    is_classmethod_context,
    is_dispatcher_call,
    resolve_cls_call,
    resolve_local_var_method,
    resolve_nested_def,
    resolve_self_attr_method,
    walk_mro,
)

# Map sentinel FQN → accepted pattern tag + valid method set.
# Keys must exactly match SENTINEL_BUILTIN_TYPES in discovery.py (single source of truth).
_SENTINEL_ACCEPTED: dict[str, tuple[str, frozenset[str]]] = {
    "builtins.dict": ("builtin_method_call", BUILTIN_COLLECTION_METHODS),
    "builtins.list": ("builtin_method_call", BUILTIN_COLLECTION_METHODS),
    "builtins.set": ("builtin_method_call", BUILTIN_COLLECTION_METHODS),
    "builtins.tuple": ("builtin_method_call", BUILTIN_COLLECTION_METHODS),
    "pathlib.Path": ("pathlib_method_call", PATHLIB_METHODS),
}

assert set(_SENTINEL_ACCEPTED) == SENTINEL_BUILTIN_TYPES, (
    "visitor._SENTINEL_ACCEPTED keys diverged from discovery.SENTINEL_BUILTIN_TYPES"
)


def _warn_kind_failure(kind: str, file_path: str, exc: BaseException) -> None:
    """Emit a per-kind visitor failure warning to stderr.

    Used by the new edge-kind visitors (import / except / annotation /
    isinstance) under their per-method ``try/except`` so a failure in one kind
    does not poison other kinds' edges from the same file (Corollary 1.2/4.2
    extended per-kind, per epic #76).
    """
    import sys

    where = file_path or "<unknown>"
    print(
        f"[pyscope-mcp] {kind}-edge visitor failed in {where}: "
        f"{type(exc).__name__}: {exc}",
        file=sys.stderr,
    )


class EdgeVisitor(ast.NodeVisitor):
    """Walk one file's AST and collect caller -> callee edges.

    Resolution is composed from handlers in `resolution`:
      1. Direct call target via `_resolve_expr`
      2. If that miss'd, try indirect dispatch: callable arg to a known dispatcher
      3. If nothing resolved, record the miss.
    """

    def __init__(
        self,
        module_fqn: str,
        import_table: dict[str, str],
        known_fqns: set[str],
        class_bases: dict[str, list[str]],
        file_path: str = "",
        miss_log: MissLog | None = None,
        known_classes: set[str] | None = None,
        self_attr_types: dict[str, dict[str, str]] | None = None,
        local_types: dict[str, dict[str, str]] | None = None,
        external_local_types: dict[str, dict[str, str]] | None = None,
        nested_defs: dict[str, dict[str, tuple[str, int]]] | None = None,
    ) -> None:
        self._ctx = ResolveCtx(
            module_fqn=module_fqn,
            import_table=import_table,
            known_fqns=known_fqns,
            class_bases=class_bases,
            known_classes=known_classes or set(),
            self_attr_types=self_attr_types or {},
            local_types=local_types or {},
            external_local_types=external_local_types or {},
            nested_defs=nested_defs or {},
        )
        self._file_path = file_path
        self._miss_log = miss_log
        self._scope_stack: list[str] = []
        self._func_node_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        # Kind-tagged edges: caller_fqn -> kind -> set[callee_fqn].
        # Edge kinds: "call", "import", "except", "annotation", "isinstance".
        # The legacy ``self.edges`` view (call edges only) remains available via
        # the property below so call-edge consumers (pipeline aggregation,
        # legacy tests) continue to work unmodified.
        self.kind_edges: dict[str, dict[str, set[str]]] = {}

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def _current_caller(self) -> str:
        if self._scope_stack:
            return f"{self._ctx.module_fqn}.{'.'.join(self._scope_stack)}"
        return self._ctx.module_fqn

    def _current_func_fqn(self) -> str | None:
        """Return the FQN of the innermost enclosing function/method scope, or None.

        Walks the scope stack inside-out, skipping scopes that are known classes.
        Returns the first scope FQN that is NOT a known class (i.e. a function or
        nested function). Returns None at module level or if we're only inside classes.

        Note: nested functions may not be in ``known_fqns`` (``collect_defs`` only
        goes two levels deep), so we cannot rely on ``known_fqns`` membership.
        Instead we use ``known_classes`` as a negative filter: if a scope name at
        a given depth IS a known class FQN, skip it; otherwise treat it as a function.
        """
        for i in range(len(self._scope_stack) - 1, -1, -1):
            candidate = f"{self._ctx.module_fqn}.{'.'.join(self._scope_stack[:i + 1])}"
            # Skip known-class scopes; everything else is a function scope.
            if candidate not in self._ctx.known_classes:
                return candidate
        return None

    def _enclosing_func_fqns(self) -> list[str]:
        """Return a list of all enclosing function-scope FQNs, innermost first.

        Used by ``resolve_nested_def`` to walk outward through nested scopes.
        Class scopes are excluded (same logic as ``_current_func_fqn``).
        """
        result: list[str] = []
        for i in range(len(self._scope_stack) - 1, -1, -1):
            candidate = f"{self._ctx.module_fqn}.{'.'.join(self._scope_stack[:i + 1])}"
            if candidate not in self._ctx.known_classes:
                result.append(candidate)
        return result

    def _enclosing_func_node(self) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """Return the innermost enclosing function AST node, or None."""
        return self._func_node_stack[-1] if self._func_node_stack else None

    def _enclosing_class_fqn(self) -> str | None:
        """FQN of the enclosing class, walking inside-out through the scope stack.

        Only returns a candidate if it is a known *class* FQN.  Without this
        check a nested function such as ``Class.method._inner`` would match
        ``Class.method`` first (a method, not a class) and hand the wrong FQN
        to ``walk_mro``, causing all inherited ``self.X`` calls inside nested
        helpers to be mis-classified as ``self_method_unresolved``.
        """
        if len(self._scope_stack) < 2:
            return None
        for i in range(len(self._scope_stack) - 2, -1, -1):
            candidate = f"{self._ctx.module_fqn}.{'.'.join(self._scope_stack[:i + 1])}"
            if candidate in self._ctx.known_classes:
                return candidate
        return None

    # ------------------------------------------------------------------
    # Resolvers
    # ------------------------------------------------------------------

    def _resolve_expr(self, func_node: ast.expr, call_lineno: int = 0) -> str | None:
        """Resolve any callable-reference expression (a Call.func, or a
        callable-argument to a dispatcher) to an in-package FQN or None.

        ``call_lineno`` is the source line of the call site; used by the
        nested-def resolver to enforce forward-reference rules.  Pass 0
        when the line is unknown (nested-def check is then skipped).
        """
        # super().method(...) — Attribute whose value is Call(super).
        if isinstance(func_node, ast.Attribute) and isinstance(func_node.value, ast.Call):
            inner_call = func_node.value
            if isinstance(inner_call.func, ast.Name) and inner_call.func.id == "super":
                return self._resolve_super_method(func_node.attr)

        if isinstance(func_node, ast.Name):
            name = func_node.id
            if name in self._ctx.import_table:
                candidate = self._ctx.import_table[name]
                if candidate in self._ctx.known_fqns:
                    return candidate
            # cls(...) inside a @classmethod → resolve to enclosing class __init__.
            if name == "cls":
                enc_func = self._enclosing_func_node()
                enc_class = self._enclosing_class_fqn()
                if enc_func is not None and enc_class is not None and is_classmethod_context(enc_func):
                    return resolve_cls_call(enc_class, None, self._ctx)
            # Nested-def lookup: check enclosing scopes before module-global.
            # Only attempt when we have a call_lineno (pass 2 call sites always
            # have one; dispatcher-arg resolution may not).
            if call_lineno > 0 and self._ctx.nested_defs:
                scope_fqns = self._enclosing_func_fqns()
                nested = resolve_nested_def(name, call_lineno, scope_fqns, self._ctx)
                if nested is not None:
                    return nested
            candidate = f"{self._ctx.module_fqn}.{name}"
            if candidate in self._ctx.known_fqns:
                return candidate
            return None

        if isinstance(func_node, ast.Attribute):
            # ClassName(...).method(...) — call on constructor result.
            # Must be checked BEFORE attr_chain (which bails on Call receivers).
            if isinstance(func_node.value, ast.Call):
                inner_call = func_node.value
                # super() is already handled above; skip it here.
                if not (
                    isinstance(inner_call.func, ast.Name)
                    and inner_call.func.id == "super"
                ):
                    class_fqn = infer_call_class_type(inner_call, self._ctx)
                    if class_fqn is not None:
                        method = func_node.attr
                        candidate = f"{class_fqn}.{method}"
                        if candidate in self._ctx.known_fqns:
                            return candidate
                        return walk_mro(
                            class_fqn,
                            method,
                            self._ctx.class_bases,
                            self._ctx.known_fqns,
                        )

            chain = attr_chain(func_node)
            if chain is None:
                return None

            # cls.method(...) inside a @classmethod.
            if chain[0] == "cls" and len(chain) == 2:
                enc_func = self._enclosing_func_node()
                enc_class = self._enclosing_class_fqn()
                if enc_func is not None and enc_class is not None and is_classmethod_context(enc_func):
                    return resolve_cls_call(enc_class, chain[1], self._ctx)

            # self.method(...) with MRO fallback for inherited methods.
            if chain[0] == "self" and len(chain) >= 2:
                class_fqn = self._enclosing_class_fqn()
                if class_fqn is None:
                    return None
                if len(chain) == 2:
                    method = chain[1]
                    candidate = f"{class_fqn}.{method}"
                    if candidate in self._ctx.known_fqns:
                        return candidate
                    return walk_mro(
                        class_fqn,
                        method,
                        self._ctx.class_bases,
                        self._ctx.known_fqns,
                    )
                # self.<attr>.<method>(...) — three-part chain via attr type tracking.
                if len(chain) == 3:
                    attr_name, method = chain[1], chain[2]
                    return resolve_self_attr_method(
                        attr_name, method, class_fqn, self._ctx
                    )
                return None

            # Local-variable type tracking fallback: var.method() where var is
            # a local variable statically bound to an in-package class.
            # Only applies to 2-part chains (var.method); longer chains go through
            # the import-table prefix resolution below.
            if len(chain) == 2 and chain[0] != "self":
                func_fqn = self._current_func_fqn()
                if func_fqn is not None:
                    resolved = resolve_local_var_method(func_node, func_fqn, self._ctx)
                    if resolved is not None:
                        return resolved

            # Deeper attribute chain — longest-prefix against import table.
            for prefix_len in range(len(chain) - 1, 0, -1):
                prefix = ".".join(chain[:prefix_len])
                if prefix in self._ctx.import_table:
                    base_fqn = self._ctx.import_table[prefix]
                    remainder = chain[prefix_len:]
                    candidate = ".".join([base_fqn] + remainder)
                    if candidate in self._ctx.known_fqns:
                        return candidate

            dotted = ".".join(chain)
            if dotted in self._ctx.known_fqns:
                return dotted
            return None

        return None

    def _resolve_super_method(self, method: str) -> str | None:
        """Resolve `super().{method}()` to the MRO-next in-package definition."""
        class_fqn = self._enclosing_class_fqn()
        if class_fqn is None:
            return None
        return walk_mro(
            class_fqn,
            method,
            self._ctx.class_bases,
            self._ctx.known_fqns,
        )

    def _resolve_external(self, func_node: ast.expr) -> str | None:
        """Check if a call resolves to an external (non-package) FQN."""
        if isinstance(func_node, ast.Name):
            name = func_node.id
            if name in self._ctx.import_table:
                candidate = self._ctx.import_table[name]
                if candidate not in self._ctx.known_fqns:
                    return candidate
            return None

        if isinstance(func_node, ast.Attribute):
            chain = attr_chain(func_node)
            if chain is None:
                return None
            if chain[0] == "self":
                return None
            for prefix_len in range(len(chain) - 1, 0, -1):
                prefix = ".".join(chain[:prefix_len])
                if prefix in self._ctx.import_table:
                    base_fqn = self._ctx.import_table[prefix]
                    remainder = chain[prefix_len:]
                    candidate = ".".join([base_fqn] + remainder)
                    if candidate not in self._ctx.known_fqns:
                        return candidate
        return None

    # ------------------------------------------------------------------
    # Edge emission
    # ------------------------------------------------------------------

    def _emit(self, callee_fqn: str, kind: str = "call", caller: str | None = None) -> None:
        """Record an edge from *caller* (or the current caller) to *callee_fqn*
        under the given *kind* bucket.

        ``kind`` is one of the supported edge-kind strings — ``call``, ``import``,
        ``except``, ``annotation``, ``isinstance``.  Adding a new kind is a
        single-line additive change to the call site that emits it; the
        visitor stores all kinds in the same per-caller dict.

        ``caller`` overrides the current scope-derived caller when supplied,
        which lets module-level emitters (imports) attribute edges to the
        module FQN regardless of how the AST walk's scope stack happens to
        be configured at the visit site.
        """
        if caller is None:
            caller = self._current_caller()
        bucket = self.kind_edges.setdefault(caller, {})
        bucket.setdefault(kind, set()).add(callee_fqn)

    @property
    def edges(self) -> dict[str, set[str]]:
        """Backward-compatible view: caller -> set[callee] for ``call`` edges only.

        The legacy ``self.edges`` shape (flat call-only) is preserved as a
        derived view over ``self.kind_edges``.  Pipeline aggregation and the
        existing test corpus consume this view; non-call edge kinds are
        accessible directly via ``self.kind_edges``.
        """
        return {
            caller: set(buckets["call"])
            for caller, buckets in self.kind_edges.items()
            if "call" in buckets and buckets["call"]
        }

    def _try_accept_self_attr_sentinel(self, node: ast.Call) -> str | None:
        """For ``self.<attr>.<method>(...)`` calls where ``<attr>`` is typed as a
        sentinel (builtin or pathlib.Path), return the accepted-pattern tag if the
        method is in the corresponding whitelist; otherwise return None.

        Called before classify_miss so these don't land in self_method_unresolved.
        """
        func = node.func
        if not isinstance(func, ast.Attribute):
            return None
        chain = attr_chain(func)
        if chain is None or len(chain) != 3 or chain[0] != "self":
            return None
        attr_name, method = chain[1], chain[2]
        class_fqn = self._enclosing_class_fqn()
        if class_fqn is None:
            return None
        attr_class = self._ctx.self_attr_types.get(class_fqn, {}).get(attr_name)
        if attr_class is None or attr_class not in _SENTINEL_ACCEPTED:
            return None
        accepted_tag, valid_methods = _SENTINEL_ACCEPTED[attr_class]
        if method in valid_methods:
            return accepted_tag
        return None

    def _try_accept_local_var_sentinel(self, node: ast.Call) -> str | None:
        """For ``var.<method>(...)`` calls where ``var`` is a local variable typed
        as a sentinel (builtin or pathlib.Path), return the accepted-pattern tag if
        the method is in the corresponding whitelist; otherwise return None.
        """
        func = node.func
        if not isinstance(func, ast.Attribute):
            return None
        chain = attr_chain(func)
        if chain is None or len(chain) != 2 or chain[0] == "self":
            return None
        var_name, method = chain[0], chain[1]
        func_fqn = self._current_func_fqn()
        if func_fqn is None:
            return None
        var_class = self._ctx.local_types.get(func_fqn, {}).get(var_name)
        if var_class is None or var_class not in _SENTINEL_ACCEPTED:
            return None
        accepted_tag, valid_methods = _SENTINEL_ACCEPTED[var_class]
        if method in valid_methods:
            return accepted_tag
        return None

    def _try_accept_external_local_var(self, node: ast.Call) -> str | None:
        """For ``var.<method>(...)`` 2-part chains where ``var`` is a local variable
        (or module-level variable) bound to a whitelisted external factory, return
        the accepted-pattern bucket tag.

        Only fires for 2-part chains (``var.method``).  Longer chains (e.g.
        ``service.channels().list().execute()``) are not handled here — they fall
        through to ``attr_chain_unresolved`` as before.

        Lookup order:
          1. Current function-level bindings.
          2. Module-level bindings (for `_cli = typer.Typer()` at module scope,
             called from inside a function like `_cli.command()`).

        Returns the accepted-bucket name (e.g. ``"httpx_method_call"``) or None.
        """
        func = node.func
        if not isinstance(func, ast.Attribute):
            return None
        chain = attr_chain(func)
        if chain is None or len(chain) != 2 or chain[0] == "self":
            return None
        var_name = chain[0]
        # 1. Function-level lookup.
        func_fqn = self._current_func_fqn()
        if func_fqn is not None:
            factory_fqn = self._ctx.external_local_types.get(func_fqn, {}).get(var_name)
            if factory_fqn is not None:
                return EXTERNAL_FACTORY_BUCKETS.get(factory_fqn)
        # 2. Module-level fallback: vars bound at module scope are visible inside functions.
        factory_fqn = self._ctx.external_local_types.get(self._ctx.module_fqn, {}).get(var_name)
        if factory_fqn is not None:
            return EXTERNAL_FACTORY_BUCKETS.get(factory_fqn)
        return None

    def _try_accept_external_chained(self, node: ast.Call) -> str | None:
        """For inline chained calls rooted at a self-returning external local var,
        return the accepted-pattern bucket tag.

        Handles patterns like::

            service.channels().list(part="snippet").execute()

        where ``service`` is bound to a factory FQN in ``EXTERNAL_SELF_RETURNING``.
        The chain is walked from the outermost Call inward: each segment is an
        Attribute node whose value is either a Call (keep walking) or a Name (root).

        Only fires when the root variable's factory FQN is in
        ``EXTERNAL_SELF_RETURNING``.  Returns the bucket tag or None.

        False-positive guards:
        - Root variable bound to a NON-self-returning factory → None.
        - Root variable unbound (unknown) → None.
        - Chain rooted at ``self`` → None (handled by self-attr resolvers).
        - Chain rooted at an in-package class local var → None (local_types wins
          via ``_resolve_expr`` before this is called; this is a last-resort path).
        """
        func = node.func
        # We need at least one intermediate Call in the chain, so func must be
        # an Attribute whose value is a Call (not a plain Name or static chain).
        if not isinstance(func, ast.Attribute):
            return None
        if not isinstance(func.value, ast.Call):
            return None

        # Walk the chain inward to find the root Name.
        # The chain looks like:  Name.attr().(attr().)*.attr  — outermost Call
        # node's func is Attribute(value=Call(...), attr=terminal).
        # We don't need the terminal method name for bucket lookup — we only need
        # to confirm the root variable is self-returning.
        cur: ast.expr = func.value  # the innermost-so-far Call
        while True:
            if isinstance(cur, ast.Call):
                inner_func = cur.func
                if isinstance(inner_func, ast.Attribute):
                    if isinstance(inner_func.value, ast.Name):
                        # Root reached: inner_func.value is the variable name.
                        root_name = inner_func.value.id
                        break
                    elif isinstance(inner_func.value, ast.Call):
                        # Another Call layer — keep descending.
                        cur = inner_func.value
                        continue
                # Anything else (e.g. subscript, complex expression) → give up.
                return None
            else:
                return None

        # Guard: self-rooted chains are handled elsewhere.
        if root_name == "self":
            return None

        # Look up the root variable's factory FQN.
        factory_fqn: str | None = None
        func_fqn = self._current_func_fqn()
        if func_fqn is not None:
            factory_fqn = self._ctx.external_local_types.get(func_fqn, {}).get(root_name)
        if factory_fqn is None:
            # Module-level fallback (e.g. module-scope typer.Typer instance).
            factory_fqn = self._ctx.external_local_types.get(
                self._ctx.module_fqn, {}
            ).get(root_name)
        if factory_fqn is None:
            return None

        # Only fire for self-returning types.
        if factory_fqn not in EXTERNAL_SELF_RETURNING:
            return None

        return EXTERNAL_FACTORY_BUCKETS.get(factory_fqn)

    def _try_emit_dispatcher_edge(self, call: ast.Call) -> bool:
        """If `call` looks like `dispatcher(fn, ...)` where fn is an in-package
        callable reference, emit an extra edge to fn. Returns True iff we emitted.

        This runs in ADDITION to the normal edge for the dispatcher call
        itself — it doesn't suppress misses on the dispatcher.
        """
        if not is_dispatcher_call(call):
            return False
        arg = dispatcher_callable_arg(call)
        if arg is None:
            return False
        if not isinstance(arg, (ast.Name, ast.Attribute)):
            return False
        # Pass the dispatcher call's lineno so nested-def resolution can apply
        # its forward-reference guard (the callable arg is referenced at this line).
        resolved = self._resolve_expr(arg, call_lineno=call.lineno)
        if resolved is None:
            return False
        self._emit(resolved)
        return True

    # ------------------------------------------------------------------
    # Visit methods
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_stack.append(node.name)
        self._func_node_stack.append(node)
        # Annotation edges from this function's signature are emitted with the
        # function's own FQN as the caller — push the scope first.
        self._scan_function_annotations(node)
        self.generic_visit(node)
        self._func_node_stack.pop()
        self._scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scope_stack.append(node.name)
        self._func_node_stack.append(node)
        self._scan_function_annotations(node)
        self.generic_visit(node)
        self._func_node_stack.pop()
        self._scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    # ------------------------------------------------------------------
    # New edge-kind visitors (epic #76 child #2 / issue #85)
    #
    # Per-kind error isolation: each new visitor wraps its body in
    # try/except so a failure in (say) annotation resolution does not drop
    # call/import/except edges from the same file.  The outer per-file guard
    # in pipeline.py remains as the safety net for ``visit_Call`` itself
    # (which is hot and has its own intricate resolution path).
    # ------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        try:
            module_caller = self._ctx.module_fqn
            for alias in node.names:
                # ``import foo.bar`` → emit an import edge to ``foo.bar``.
                # ``import foo.bar as fb`` → still emit ``foo.bar`` (the imported
                # symbol's canonical FQN, not the local rebinding).
                target = alias.name
                if target:
                    self._emit(target, kind="import", caller=module_caller)
        except Exception as exc:  # noqa: BLE001
            _warn_kind_failure("import", self._file_path, exc)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        try:
            module_caller = self._ctx.module_fqn
            level = node.level or 0
            if level > 0:
                # Relative import — resolve via import_table: each alias.local
                # was registered there by build_import_table.
                for alias in node.names:
                    local = alias.asname if alias.asname else alias.name
                    target = self._ctx.import_table.get(local)
                    if target:
                        self._emit(target, kind="import", caller=module_caller)
            else:
                base = node.module
                if base:
                    for alias in node.names:
                        # ``from foo.bar import *`` — alias.name == "*".  Skip:
                        # we don't have visibility into the wildcard expansion
                        # at the visitor layer.  (Wildcard imports are #74.)
                        if alias.name == "*":
                            continue
                        target = f"{base}.{alias.name}"
                        self._emit(target, kind="import", caller=module_caller)
        except Exception as exc:  # noqa: BLE001
            _warn_kind_failure("import", self._file_path, exc)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        try:
            exc_type = node.type
            if exc_type is not None:
                # ``except (A, B):`` — tuple of exception types.
                if isinstance(exc_type, ast.Tuple):
                    for elt in exc_type.elts:
                        self._emit_except_target(elt)
                else:
                    self._emit_except_target(exc_type)
        except Exception as exc:  # noqa: BLE001
            _warn_kind_failure("except", self._file_path, exc)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        try:
            self._scan_annotation(node.annotation)
        except Exception as exc:  # noqa: BLE001
            _warn_kind_failure("annotation", self._file_path, exc)
        self.generic_visit(node)

    # ---- annotation helpers ----

    def _scan_function_annotations(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Emit ``annotation`` edges from this function/method to every type
        named in its argument and return annotations.

        Wrapped in its own try/except for per-kind isolation.
        """
        try:
            args = node.args
            for arg in (
                *args.args,
                *args.kwonlyargs,
                *args.posonlyargs,
            ):
                if arg.annotation is not None:
                    self._scan_annotation(arg.annotation)
            if args.vararg is not None and args.vararg.annotation is not None:
                self._scan_annotation(args.vararg.annotation)
            if args.kwarg is not None and args.kwarg.annotation is not None:
                self._scan_annotation(args.kwarg.annotation)
            if node.returns is not None:
                self._scan_annotation(node.returns)
        except Exception as exc:  # noqa: BLE001
            _warn_kind_failure("annotation", self._file_path, exc)

    def _scan_annotation(self, annotation: ast.expr) -> None:
        """Resolve *annotation* to in-package type FQNs and emit annotation
        edges.  Recurses into ``Subscript`` (``Optional[X]``, ``list[X]``,
        ``Union[A, B]``) and ``BinOp`` (``A | B`` PEP 604 unions).  Bare
        ``ast.Name`` / ``ast.Attribute`` chains are resolved via
        ``resolve_annotation`` (the discovery helper, exposed for reuse).
        """
        # Subscript: peel off the subscript and recurse into both value and slice.
        if isinstance(annotation, ast.Subscript):
            self._scan_annotation(annotation.value)
            slice_node = annotation.slice
            # Tuple slice: ``Union[A, B]`` -> Tuple(A, B); ``Dict[K, V]`` similar.
            if isinstance(slice_node, ast.Tuple):
                for elt in slice_node.elts:
                    self._scan_annotation(elt)
            else:
                self._scan_annotation(slice_node)
            return
        # PEP 604 unions: ``A | B`` is BinOp(BitOr).  Recurse on both sides.
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            self._scan_annotation(annotation.left)
            self._scan_annotation(annotation.right)
            return
        # Bare names and attribute chains: resolve via the shared helper.
        if isinstance(annotation, (ast.Name, ast.Attribute)):
            resolved = resolve_annotation(
                annotation,
                self._ctx.module_fqn,
                self._ctx.import_table,
                self._ctx.known_fqns,
            )
            if resolved is not None:
                # Emit from the innermost enclosing function; module scope falls
                # back to the module FQN via _current_caller().
                func_fqn = self._current_func_fqn()
                caller = func_fqn if func_fqn is not None else self._current_caller()
                self._emit(resolved, kind="annotation", caller=caller)

    def _emit_except_target(self, type_node: ast.expr) -> None:
        """Resolve an ``except T:`` clause's type expression to an in-package
        FQN and emit an ``except`` edge.  Falls back to ``import_table`` lookup
        for bare names so ``except SomeError:`` resolves correctly even when
        ``SomeError`` is brought in via ``from x import SomeError``.
        """
        target: str | None = None
        if isinstance(type_node, ast.Name):
            name = type_node.id
            candidate = self._ctx.import_table.get(name)
            if candidate is not None:
                target = candidate
            else:
                local_candidate = f"{self._ctx.module_fqn}.{name}"
                if local_candidate in self._ctx.known_fqns:
                    target = local_candidate
        elif isinstance(type_node, ast.Attribute):
            # Reuse the annotation resolver — it handles dotted chains via
            # the import table the same way exception names work.
            target = resolve_annotation(
                type_node,
                self._ctx.module_fqn,
                self._ctx.import_table,
                self._ctx.known_fqns,
            )
        if target is not None:
            func_fqn = self._current_func_fqn()
            caller = func_fqn if func_fqn is not None else self._current_caller()
            self._emit(target, kind="except", caller=caller)

    def _try_emit_isinstance_edge(self, node: ast.Call) -> bool:
        """If *node* is an ``isinstance(obj, T)`` call, emit ``isinstance``
        edges to T's resolved FQN(s) and return True.  Otherwise return False.

        T may be a tuple ``isinstance(obj, (A, B))`` — one edge per element.
        Edges are attributed to the innermost enclosing function (consistent
        with the call-edge convention), falling back to module scope.

        Resolution failures inside this method (e.g. unresolved name) are
        non-fatal — return True regardless so the caller skips the normal
        miss-classification path for the ``isinstance`` builtin itself.
        """
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "isinstance"):
            return False
        if len(node.args) < 2:
            return True  # malformed isinstance — still skip miss recording
        try:
            second = node.args[1]
            type_exprs: list[ast.expr]
            if isinstance(second, ast.Tuple):
                type_exprs = list(second.elts)
            else:
                type_exprs = [second]
            func_fqn = self._current_func_fqn()
            caller = func_fqn if func_fqn is not None else self._current_caller()
            for expr in type_exprs:
                if isinstance(expr, (ast.Name, ast.Attribute)):
                    resolved = resolve_annotation(
                        expr,
                        self._ctx.module_fqn,
                        self._ctx.import_table,
                        self._ctx.known_fqns,
                    )
                    if resolved is not None:
                        self._emit(resolved, kind="isinstance", caller=caller)
        except Exception as exc:  # noqa: BLE001
            _warn_kind_failure("isinstance", self._file_path, exc)
        return True

    def visit_Call(self, node: ast.Call) -> None:
        # isinstance carve-out: emit isinstance kind edges to the checked
        # type(s) and skip both the normal call-resolution path and the miss
        # classification (isinstance is a builtin — emitting a "call" edge
        # would be a false positive, and recording it as an unresolved miss
        # would pollute the miss log).  Still recurse into children for
        # nested calls inside the isinstance arguments.
        if self._try_emit_isinstance_edge(node):
            self.generic_visit(node)
            return
        callee = self._resolve_expr(node.func, call_lineno=node.lineno)
        if callee is not None:
            self._emit(callee)
            if self._miss_log is not None:
                self._miss_log.record_resolved(in_package=True)
        else:
            if self._miss_log is not None:
                ext = self._resolve_external(node.func)
                if ext is not None:
                    self._miss_log.record_resolved(in_package=False)
                else:
                    # Check self-attr and local-var sentinel calls before falling
                    # through to classify_miss.
                    sentinel_tag = (
                        self._try_accept_self_attr_sentinel(node)
                        or self._try_accept_local_var_sentinel(node)
                        or self._try_accept_external_local_var(node)
                        or self._try_accept_external_chained(node)
                    )
                    if sentinel_tag is not None:
                        self._miss_log.record_accepted(sentinel_tag, self._file_path)
                    else:
                        pattern = classify_miss(
                            node,
                            enclosing_class_fqn=self._enclosing_class_fqn(),
                            class_bases=self._ctx.class_bases,
                            import_table=self._ctx.import_table,
                        )
                        if pattern in ACCEPTED_PATTERNS:
                            self._miss_log.record_accepted(pattern, self._file_path)
                        else:
                            self._miss_log.record_miss(
                                pattern=pattern,
                                caller=self._current_caller(),
                                file_path=self._file_path,
                                line=node.lineno,
                                snippet=snippet(node),
                            )

        # Indirect-dispatch extra edge: if the dispatcher's callable arg is an
        # in-package reference, emit an edge to it. Runs regardless of whether
        # the primary dispatcher call resolved.
        self._try_emit_dispatcher_edge(node)

        self.generic_visit(node)
