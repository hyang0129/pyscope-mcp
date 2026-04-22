"""AST visitor: walk one file, emit caller -> callee edges."""

from __future__ import annotations

import ast

from .misses import MissLog, classify_miss, snippet
from .resolution import (
    ResolveCtx,
    attr_chain,
    dispatcher_callable_arg,
    is_dispatcher_call,
    resolve_self_attr_method,
    walk_mro,
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
    ) -> None:
        self._ctx = ResolveCtx(
            module_fqn=module_fqn,
            import_table=import_table,
            known_fqns=known_fqns,
            class_bases=class_bases,
            known_classes=known_classes or set(),
            self_attr_types=self_attr_types or {},
        )
        self._file_path = file_path
        self._miss_log = miss_log
        self._scope_stack: list[str] = []
        self.edges: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def _current_caller(self) -> str:
        if self._scope_stack:
            return f"{self._ctx.module_fqn}.{'.'.join(self._scope_stack)}"
        return self._ctx.module_fqn

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

    def _resolve_expr(self, func_node: ast.expr) -> str | None:
        """Resolve any callable-reference expression (a Call.func, or a
        callable-argument to a dispatcher) to an in-package FQN or None.
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
            candidate = f"{self._ctx.module_fqn}.{name}"
            if candidate in self._ctx.known_fqns:
                return candidate
            return None

        if isinstance(func_node, ast.Attribute):
            chain = attr_chain(func_node)
            if chain is None:
                return None

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

    def _emit(self, callee_fqn: str) -> None:
        caller = self._current_caller()
        self.edges.setdefault(caller, set()).add(callee_fqn)

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
        resolved = self._resolve_expr(arg)
        if resolved is None:
            return False
        self._emit(resolved)
        return True

    # ------------------------------------------------------------------
    # Visit methods
    # ------------------------------------------------------------------

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
        callee = self._resolve_expr(node.func)
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
                    pattern = classify_miss(
                        node,
                        enclosing_class_fqn=self._enclosing_class_fqn(),
                        class_bases=self._ctx.class_bases,
                    )
                    if pattern in {
                        "builtin_method_call",
                        "pathlib_method_call",
                        "futures_method_call",
                        "pydantic_method_call",
                    }:
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
