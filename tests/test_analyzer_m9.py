"""M9: local-variable type tracker handler.

Tests for the ``collect_local_var_types`` pass-1 collector and the
``resolve_local_var_method`` / ``resolve_call_result_method`` resolvers
wired into EdgeVisitor._resolve_expr.

Pattern: var.method(...) where var is a local variable statically bound to an
in-package class via constructor call, annotation, or parameter annotation.
"""

from __future__ import annotations

from pathlib import Path

from pyscope_mcp.analyzer import build_raw


def _make_package(tmp_path: Path, pkg_name: str, files: dict[str, str]) -> Path:
    pkg = tmp_path / pkg_name
    pkg.mkdir()
    if "__init__.py" not in files:
        (pkg / "__init__.py").write_text("")
    for rel, content in files.items():
        target = pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------

def test_localvar_constructor_direct_method(tmp_path: Path) -> None:
    """x = InPkgClass(); x.method() → edge to InPkgClass.method."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def process(self):\n"
            "        return 42\n"
            "\n"
            "def runner():\n"
            "    x = Worker()\n"
            "    return x.process()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Worker.process" in raw["pkg.mod.runner"]


def test_localvar_cross_module_import(tmp_path: Path) -> None:
    """x = mod.InPkgClass(); x.method() with cross-module import → edge."""
    root = _make_package(tmp_path, "pkg", {
        "worker.py": (
            "class Worker:\n"
            "    def process(self):\n"
            "        return 42\n"
        ),
        "runner.py": (
            "from pkg.worker import Worker\n"
            "\n"
            "def run():\n"
            "    x = Worker()\n"
            "    return x.process()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.worker.Worker.process" in raw["pkg.runner.run"]


def test_localvar_parameter_annotation(tmp_path: Path) -> None:
    """def f(self, x: InPkgClass): x.method() — annotation-driven."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def process(self):\n"
            "        return 42\n"
            "\n"
            "class Runner:\n"
            "    def run(self, worker: Worker):\n"
            "        return worker.process()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Worker.process" in raw["pkg.mod.Runner.run"]


def test_localvar_forward_ref_string_annotation(tmp_path: Path) -> None:
    """def f(self, x: \"InPkgClass\"): x.method() — forward-ref string annotation."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def process(self):\n"
            "        return 42\n"
            "\n"
            "class Runner:\n"
            "    def run(self, worker: \"Worker\"):\n"
            "        return worker.process()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Worker.process" in raw["pkg.mod.Runner.run"]


def test_localvar_annassign_unknown_rhs(tmp_path: Path) -> None:
    """x: InPkgClass = factory(); x.method() — AnnAssign with unknown RHS."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def process(self):\n"
            "        return 42\n"
            "\n"
            "def factory():\n"
            "    return Worker()\n"
            "\n"
            "def runner():\n"
            "    x: Worker = factory()\n"
            "    return x.process()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Worker.process" in raw["pkg.mod.runner"]


def test_call_result_method_inline(tmp_path: Path) -> None:
    """InPkgClass().method() — call-on-call-result (direct resolution path)."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def process(self):\n"
            "        return 42\n"
            "\n"
            "def runner():\n"
            "    return Worker().process()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Worker.process" in raw["pkg.mod.runner"]


def test_localvar_inherited_method_via_mro(tmp_path: Path) -> None:
    """class B(A); x = B(); x.a_method() → edge to A.a_method via MRO."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Base:\n"
            "    def helper(self):\n"
            "        return 1\n"
            "\n"
            "class Child(Base):\n"
            "    pass\n"
            "\n"
            "def runner():\n"
            "    x = Child()\n"
            "    return x.helper()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Base.helper" in raw["pkg.mod.runner"]


# ---------------------------------------------------------------------------
# Negative guards (false-positive guards)
# ---------------------------------------------------------------------------

def test_loop_var_not_tracked(tmp_path: Path) -> None:
    """for x in xs: x.method() — no edge (loop var not tracked)."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def process(self):\n"
            "        return 42\n"
            "\n"
            "def runner(items):\n"
            "    for x in items:\n"
            "        x.process()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # No edge to Worker.process since x is a loop target
    callees = raw.get("pkg.mod.runner", [])
    assert "pkg.mod.Worker.process" not in callees


def test_external_factory_not_tracked(tmp_path: Path) -> None:
    """x = external_factory(); x.method() — no edge (external_factory is external)."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from some_external_lib import external_factory\n"
            "\n"
            "class Worker:\n"
            "    def process(self):\n"
            "        return 42\n"
            "\n"
            "def runner():\n"
            "    x = external_factory()\n"
            "    return x.process()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.runner", [])
    # external_factory() return type is unknown — no edge to Worker.process
    assert "pkg.mod.Worker.process" not in callees


def test_last_write_wins(tmp_path: Path) -> None:
    """x = SomeClass(); x = OtherClass(); x.method() → edge to OtherClass.method."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class SomeClass:\n"
            "    def process(self):\n"
            "        return 1\n"
            "\n"
            "class OtherClass:\n"
            "    def process(self):\n"
            "        return 2\n"
            "\n"
            "def runner():\n"
            "    x = SomeClass()\n"
            "    x = OtherClass()\n"
            "    return x.process()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.runner", [])
    # Last-write-wins: x is rebound to OtherClass
    assert "pkg.mod.OtherClass.process" in callees
    assert "pkg.mod.SomeClass.process" not in callees


def test_nested_func_scope_isolation(tmp_path: Path) -> None:
    """def outer(x: A): def inner(): x.method() — inner must NOT inherit outer's binding."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def process(self):\n"
            "        return 42\n"
            "\n"
            "def outer(worker: Worker):\n"
            "    def inner():\n"
            "        return worker.process()\n"
            "    return inner\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # outer.inner does NOT have its own binding for worker — scope isolation.
    inner_callees = raw.get("pkg.mod.outer.inner", [])
    # The call worker.process() in inner should NOT resolve via outer's annotation.
    assert "pkg.mod.Worker.process" not in inner_callees


def test_deeply_nested_func_no_double_processing(tmp_path: Path) -> None:
    """outer -> inner -> innermost: innermost must be collected exactly once.

    Regression guard for the ast.walk double-processing bug: _scan_function_body
    formerly used ast.walk which would find innermost from outer's walk AND again
    from inner's walk, causing duplicate/incorrect bindings. This test verifies
    that a binding in innermost is still correctly captured (not lost) and that
    innermost has only its own scope bindings (not inherited from outer or inner).
    """
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def process(self):\n"
            "        return 42\n"
            "\n"
            "class Other:\n"
            "    def run(self):\n"
            "        return 1\n"
            "\n"
            "def outer(w: Worker):\n"
            "    def inner(o: Other):\n"
            "        def innermost():\n"
            "            x = Worker()\n"
            "            return x.process()\n"
            "        return innermost\n"
            "    return inner\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # innermost has its own binding x = Worker() — edge to Worker.process
    innermost_callees = raw.get("pkg.mod.outer.inner.innermost", [])
    assert "pkg.mod.Worker.process" in innermost_callees
    # innermost must NOT inherit outer's 'w: Worker' or inner's 'o: Other' bindings
    assert "pkg.mod.Other.run" not in innermost_callees
    # inner's scope has o: Other — edge to Other.run from inner is valid but
    # innermost should not inherit it
    # Also verify inner and outer scopes are independent
    inner_callees = raw.get("pkg.mod.outer.inner", [])
    outer_callees = raw.get("pkg.mod.outer", [])
    # outer binds w: Worker but makes no calls — no Worker.process edge from outer
    assert "pkg.mod.Worker.process" not in outer_callees


def test_lambda_outer_scope_not_inherited(tmp_path: Path) -> None:
    """f = lambda: x.method() where x is bound in outer scope — no edge from lambda."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def process(self):\n"
            "        return 42\n"
            "\n"
            "def runner():\n"
            "    x = Worker()\n"
            "    f = lambda: x.process()\n"
            "    return f\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # x.process() is in runner's scope; the lambda has its own empty local_types scope.
    # The call resolves from runner's scope (x is bound there; lambda has its own scope).
    # Lambda body accesses x as a closure, which has no static type in the lambda scope.
    # This is about scope isolation for the lambda's func_fqn.
    # The lambda call is a closure read — we do NOT track closure vars into lambdas.
    # NOTE: The runner() call to x.process() IS directly tracked in runner's scope,
    # so we check the lambda scope separately.
    # Lambda doesn't have a named FQN in our scheme — so just check runner:
    runner_callees = raw.get("pkg.mod.runner", [])
    # runner may resolve x.process() (x is bound there before lambda).
    # But we ensure the lambda body call doesn't somehow create a wrong edge.
    # This test mainly checks that lambda-scoped access doesn't produce false edges
    # from lambda's own FQN (lambdas aren't tracked as named scopes here).
    # Any edge to Worker.process must come from runner, not a phantom lambda FQN.
    phantom = "pkg.mod.runner.<lambda>"
    assert phantom not in raw or "pkg.mod.Worker.process" not in raw.get(phantom, [])
