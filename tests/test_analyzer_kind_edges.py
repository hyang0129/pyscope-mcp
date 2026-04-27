"""Tests for the new edge kinds added in epic #76 child #2 (issue #85).

Covers the four new edge kinds — ``import``, ``except``, ``annotation``,
``isinstance`` — and the build-time inversion phase that produces
``called_by`` per symbol.

Per-kind error isolation has its own test below: a deliberately-broken
visitor method for one kind must not drop edges of other kinds from the
same file (Corollary 1.2/4.2 extended per-kind, per epic #76).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pyscope_mcp.analyzer import build_nodes_with_report


def _write(root: Path, rel: str, src: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src).lstrip("\n"))


# ----------------------------------------------------------------------
# Scenario 1 — import edge emission
# ----------------------------------------------------------------------

def test_import_from_emits_import_edge(tmp_path: Path) -> None:
    """``from pkg.lib import MyFunc`` produces an ``import`` edge to the
    fully-qualified imported symbol."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/lib.py", "def MyFunc():\n    pass\n")
    _write(
        tmp_path,
        "pkg/consumer.py",
        """
        from pkg.lib import MyFunc

        def use_it():
            MyFunc()
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    consumer = nodes["pkg.consumer"]
    assert "pkg.lib.MyFunc" in consumer["calls"].get("import", [])


def test_plain_import_emits_module_edge(tmp_path: Path) -> None:
    """``import pkg.lib`` emits an ``import`` edge to the module FQN."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/lib.py", "X = 1\n")
    _write(
        tmp_path,
        "pkg/consumer.py",
        """
        import pkg.lib
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    consumer = nodes["pkg.consumer"]
    assert "pkg.lib" in consumer["calls"].get("import", [])


def test_import_star_is_skipped(tmp_path: Path) -> None:
    """Wildcard imports are dropped at the visitor layer (issue #74's scope)."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/lib.py", "X = 1\n")
    _write(
        tmp_path,
        "pkg/consumer.py",
        """
        from pkg.lib import *
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    # If pkg.consumer is in nodes at all, it must not list a ``*`` import.
    consumer = nodes.get("pkg.consumer", {"calls": {}, "called_by": {}})
    imports = consumer.get("calls", {}).get("import", [])
    assert not any(name.endswith(".*") for name in imports)


# ----------------------------------------------------------------------
# Scenario 2 — except edge emission
# ----------------------------------------------------------------------

def test_except_handler_emits_except_edge(tmp_path: Path) -> None:
    """``except SomeError:`` emits an ``except`` edge from the enclosing
    function to the exception class FQN (resolved via import_table)."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/errors.py",
        """
        class BadThing(Exception):
            pass
        """,
    )
    _write(
        tmp_path,
        "pkg/handler.py",
        """
        from pkg.errors import BadThing

        def process():
            try:
                ...
            except BadThing:
                ...
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    excepts = nodes["pkg.handler.process"]["calls"].get("except", [])
    assert "pkg.errors.BadThing" in excepts


def test_except_tuple_emits_one_edge_per_member(tmp_path: Path) -> None:
    """``except (A, B):`` emits one edge per element."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/errors.py",
        """
        class A(Exception):
            pass

        class B(Exception):
            pass
        """,
    )
    _write(
        tmp_path,
        "pkg/handler.py",
        """
        from pkg.errors import A, B

        def process():
            try:
                ...
            except (A, B):
                ...
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    excepts = set(nodes["pkg.handler.process"]["calls"].get("except", []))
    assert "pkg.errors.A" in excepts
    assert "pkg.errors.B" in excepts


def test_bare_except_emits_no_edge(tmp_path: Path) -> None:
    """``except:`` (no type) must not emit any edge."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/handler.py",
        """
        def process():
            try:
                ...
            except:
                ...
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    proc = nodes.get("pkg.handler.process", {"calls": {}, "called_by": {}})
    # The function may not have a node at all if no edges of any kind were
    # emitted — accept either "no node" or "node with no except bucket".
    assert not proc.get("calls", {}).get("except")


# ----------------------------------------------------------------------
# Scenario 3 — annotation edge emission
# ----------------------------------------------------------------------

def test_function_param_and_return_annotations_emit_edges(tmp_path: Path) -> None:
    """Argument annotations and the return annotation both produce
    ``annotation`` edges from the function FQN to the annotated types."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/models.py",
        """
        class MyClass:
            pass

        class OtherClass:
            pass
        """,
    )
    _write(
        tmp_path,
        "pkg/typed.py",
        """
        from pkg.models import MyClass, OtherClass

        def f(x: MyClass) -> OtherClass:
            ...
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    annotations = set(nodes["pkg.typed.f"]["calls"].get("annotation", []))
    assert "pkg.models.MyClass" in annotations
    assert "pkg.models.OtherClass" in annotations


def test_ann_assign_emits_annotation_edge(tmp_path: Path) -> None:
    """``y: MyClass = ...`` inside a function emits an ``annotation`` edge."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/models.py", "class MyClass:\n    pass\n")
    _write(
        tmp_path,
        "pkg/typed.py",
        """
        from pkg.models import MyClass

        def f():
            y: MyClass = MyClass()
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    annotations = nodes["pkg.typed.f"]["calls"].get("annotation", [])
    assert "pkg.models.MyClass" in annotations


def test_subscripted_annotation_unwraps_to_inner_type(tmp_path: Path) -> None:
    """``x: Optional[MyClass]`` — the inner type is captured."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/models.py", "class MyClass:\n    pass\n")
    _write(
        tmp_path,
        "pkg/typed.py",
        """
        from typing import Optional
        from pkg.models import MyClass

        def f(x: Optional[MyClass]) -> None:
            ...
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    annotations = nodes["pkg.typed.f"]["calls"].get("annotation", [])
    assert "pkg.models.MyClass" in annotations


def test_pep604_union_unwraps_both_sides(tmp_path: Path) -> None:
    """``x: A | B`` — both A and B captured as annotation edges."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/models.py",
        """
        class A:
            pass

        class B:
            pass
        """,
    )
    _write(
        tmp_path,
        "pkg/typed.py",
        """
        from pkg.models import A, B

        def f(x: A | B) -> None:
            ...
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    annotations = set(nodes["pkg.typed.f"]["calls"].get("annotation", []))
    assert "pkg.models.A" in annotations
    assert "pkg.models.B" in annotations


# ----------------------------------------------------------------------
# Scenario 4 — isinstance edge emission
# ----------------------------------------------------------------------

def test_isinstance_emits_isinstance_edge(tmp_path: Path) -> None:
    """``isinstance(obj, Widget)`` emits an ``isinstance`` edge from the
    enclosing function to ``Widget``'s FQN (NOT a ``call`` edge for the
    isinstance builtin itself)."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/models.py", "class Widget:\n    pass\n")
    _write(
        tmp_path,
        "pkg/checks.py",
        """
        from pkg.models import Widget

        def validate(obj):
            return isinstance(obj, Widget)
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    rec = nodes["pkg.checks.validate"]
    assert "pkg.models.Widget" in rec["calls"].get("isinstance", [])
    # Critically: no spurious ``call`` edge for the isinstance builtin.
    assert "isinstance" not in rec["calls"].get("call", [])


def test_isinstance_with_tuple_emits_per_element(tmp_path: Path) -> None:
    """``isinstance(x, (A, B))`` — one edge per type."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/models.py",
        """
        class A:
            pass

        class B:
            pass
        """,
    )
    _write(
        tmp_path,
        "pkg/checks.py",
        """
        from pkg.models import A, B

        def validate(obj):
            return isinstance(obj, (A, B))
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    isinstances = set(nodes["pkg.checks.validate"]["calls"].get("isinstance", []))
    assert "pkg.models.A" in isinstances
    assert "pkg.models.B" in isinstances


# ----------------------------------------------------------------------
# Scenario 5 — build-time inversion produces called_by entries
# ----------------------------------------------------------------------

def test_build_time_inversion_populates_called_by(tmp_path: Path) -> None:
    """Build-time inversion: forward edges produce matching ``called_by``
    entries on the callee, sorted and per-kind."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/b.py",
        """
        def g():
            pass
        """,
    )
    _write(
        tmp_path,
        "pkg/a.py",
        """
        from pkg.b import g

        def f():
            g()
        """,
    )
    _write(
        tmp_path,
        "pkg/c.py",
        """
        import pkg.b
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    # Forward call edge
    assert "pkg.b.g" in nodes["pkg.a.f"]["calls"].get("call", [])
    # Reverse call edge — this is the inversion's responsibility
    assert "pkg.a.f" in nodes["pkg.b.g"]["called_by"].get("call", [])
    # Reverse import edge from pkg.c → pkg.b appears under pkg.b.called_by.import
    assert "pkg.c" in nodes["pkg.b"]["called_by"].get("import", [])


def test_inversion_lists_are_sorted(tmp_path: Path) -> None:
    """Determinism (Corollary 3.2): callee/caller lists per kind are sorted."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/target.py",
        """
        def t():
            pass
        """,
    )
    _write(
        tmp_path,
        "pkg/zeta.py",
        """
        from pkg.target import t

        def use():
            t()
        """,
    )
    _write(
        tmp_path,
        "pkg/alpha.py",
        """
        from pkg.target import t

        def use():
            t()
        """,
    )
    nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    callers = nodes["pkg.target.t"]["called_by"].get("call", [])
    assert callers == sorted(callers)


# ----------------------------------------------------------------------
# Scenario 6 — per-kind error isolation
# ----------------------------------------------------------------------

def test_per_kind_isolation_failure_in_annotation_does_not_drop_call_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the annotation visitor raises while processing one file, other
    kinds' edges from the same file (call, import, except) must still be
    emitted.  This verifies the per-method try/except guard rather than a
    single coarse per-file guard.
    """
    from pyscope_mcp.analyzer import visitor as visitor_mod

    original = visitor_mod.EdgeVisitor._scan_annotation

    def boom(self, annotation):  # noqa: ANN001 — patched method
        raise RuntimeError("induced annotation failure")

    monkeypatch.setattr(visitor_mod.EdgeVisitor, "_scan_annotation", boom)

    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/lib.py",
        """
        class Boom(Exception):
            pass

        def helper():
            pass
        """,
    )
    _write(
        tmp_path,
        "pkg/handler.py",
        """
        from pkg.lib import helper, Boom

        def process(x: int) -> None:
            try:
                helper()
            except Boom:
                ...
        """,
    )
    try:
        nodes, _report, _skel, _shas = build_nodes_with_report(tmp_path, "pkg")
    finally:
        # The monkeypatch fixture restores automatically, but be explicit.
        monkeypatch.setattr(visitor_mod.EdgeVisitor, "_scan_annotation", original)

    proc = nodes.get("pkg.handler.process", {})
    calls = proc.get("calls", {})
    # Call edge survived
    assert "pkg.lib.helper" in calls.get("call", [])
    # Except edge survived
    assert "pkg.lib.Boom" in calls.get("except", [])
    # Annotation edges suppressed (broken visitor) — bucket is absent or empty.
    assert not calls.get("annotation")
    # Sanity: the import edge from the module FQN survives too.
    assert "pkg.lib.helper" in nodes["pkg.handler"]["calls"].get("import", [])


# ----------------------------------------------------------------------
# Determinism — same source → byte-identical inversion output
# ----------------------------------------------------------------------

def test_inversion_is_deterministic(tmp_path: Path) -> None:
    """Running the build twice on the same tree yields identical nodes."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/m1.py",
        """
        from pkg.m2 import g
        from pkg.m3 import h

        def f():
            g()
            h()
        """,
    )
    _write(tmp_path, "pkg/m2.py", "def g():\n    pass\n")
    _write(tmp_path, "pkg/m3.py", "def h():\n    pass\n")
    a, _, _, _ = build_nodes_with_report(tmp_path, "pkg")
    b, _, _, _ = build_nodes_with_report(tmp_path, "pkg")
    import json
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
