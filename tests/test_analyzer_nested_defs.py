"""Synthetic tests for nested-function bare-name resolution (issue #28).

Tests cover:
1. Nested def called from outer body (direct case).
2. Nested def A calls nested def B defined earlier in same outer scope.
3. Nested def shadows a module-level name with the same identifier — nested wins.
4. FP guard: nested def in outer1 is NOT visible to sibling outer2.
5. FP guard: forward reference — nested def defined after the call site must NOT resolve.
6. Nested def inside a method (class scope).
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
# Case 1: nested def called from outer body (direct case)
# ---------------------------------------------------------------------------

def test_nested_def_called_from_outer_body(tmp_path: Path) -> None:
    """Bare call to a nested function from the enclosing outer function resolves."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def outer():\n"
            "    def _table(x):\n"
            "        return x\n"
            "    _table(1)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.outer._table" in raw.get("pkg.mod.outer", [])


# ---------------------------------------------------------------------------
# Case 2: nested def A calls nested def B defined earlier in same outer scope
# ---------------------------------------------------------------------------

def test_sibling_nested_def_calls_earlier_sibling(tmp_path: Path) -> None:
    """Nested def A can call nested def B if B was defined before A in same scope."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def outer():\n"
            "    def _make_row(x):\n"
            "        return x\n"
            "    def _render():\n"
            "        _make_row(1)\n"
            "    _render()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # _render calls _make_row (defined earlier in same outer scope)
    assert "pkg.mod.outer._make_row" in raw.get("pkg.mod.outer._render", [])
    # outer calls _render
    assert "pkg.mod.outer._render" in raw.get("pkg.mod.outer", [])


# ---------------------------------------------------------------------------
# Case 3: nested def shadows a module-level name
# ---------------------------------------------------------------------------

def test_nested_def_shadows_module_level(tmp_path: Path) -> None:
    """When a nested def has the same name as a module-level function, the nested
    def wins inside the enclosing scope (innermost scope first)."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def _helper():\n"
            "    pass\n"
            "\n"
            "def outer():\n"
            "    def _helper():\n"
            "        pass\n"
            "    _helper()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.outer", [])
    # The nested _helper should be resolved, not the module-level one.
    # (Module-level is pkg.mod._helper; nested is pkg.mod.outer._helper.)
    assert "pkg.mod.outer._helper" in callees


# ---------------------------------------------------------------------------
# FP guard 1: sibling-scope isolation
# ---------------------------------------------------------------------------

def test_sibling_scope_isolation(tmp_path: Path) -> None:
    """A nested def in outer1 must NOT resolve from a bare call in outer2."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def outer1():\n"
            "    def _secret():\n"
            "        pass\n"
            "\n"
            "def outer2():\n"
            "    _secret()\n"  # _secret is NOT in scope here
        ),
    })
    raw = build_raw(root, "pkg")
    # outer2 calls _secret as a bare name; it must NOT resolve to outer1._secret.
    callees = raw.get("pkg.mod.outer2", [])
    assert "pkg.mod.outer1._secret" not in callees


# ---------------------------------------------------------------------------
# FP guard 2: forward reference — def after the call site must NOT resolve
# ---------------------------------------------------------------------------

def test_forward_reference_does_not_resolve(tmp_path: Path) -> None:
    """A call to a bare name that refers to a nested def defined *after* the call
    site must not resolve (Python would raise UnboundLocalError at runtime)."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def outer():\n"
            "    _helper()       # line 2: call BEFORE def\n"
            "    def _helper():\n"  # line 3: def AFTER call
            "        pass\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.outer", [])
    # _helper is defined after the call site on line 2 — must NOT resolve.
    assert "pkg.mod.outer._helper" not in callees


# ---------------------------------------------------------------------------
# Case 6: nested def inside a method (class scope)
# ---------------------------------------------------------------------------

def test_nested_def_inside_method(tmp_path: Path) -> None:
    """Nested function defined inside a method resolves from a bare call in that method."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class MyClass:\n"
            "    def render(self):\n"
            "        def _row(x):\n"
            "            return x\n"
            "        _row(1)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.MyClass.render._row" in raw.get("pkg.mod.MyClass.render", [])


# ---------------------------------------------------------------------------
# Additional guard: no false positive for external / unrelated bare names
# ---------------------------------------------------------------------------

def test_unrelated_bare_name_not_resolved_as_nested(tmp_path: Path) -> None:
    """A bare call to a name that has no nested def should not produce a spurious edge."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def outer():\n"
            "    some_external_call()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # outer calls some_external_call — it must not appear as any in-package edge.
    callees = raw.get("pkg.mod.outer", [])
    assert not any("some_external_call" in c for c in callees)


# ---------------------------------------------------------------------------
# Additional guard: nested def correctly scoped to its direct enclosing function
# ---------------------------------------------------------------------------

def test_nested_def_not_visible_to_grandparent(tmp_path: Path) -> None:
    """A nested def inside a doubly-nested function is not visible from the
    outer-outer function when that name doesn't exist in the outer-outer scope."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def outer():\n"
            "    def middle():\n"
            "        def _deep():\n"
            "            pass\n"
            "        _deep()         # OK: _deep defined in middle\n"
            "    _deep()             # NOT OK: _deep is in middle, not outer\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # middle() can call _deep — that must resolve.
    assert "pkg.mod.outer.middle._deep" in raw.get("pkg.mod.outer.middle", [])
    # outer() calls _deep (line 6) — _deep is NOT defined in outer's scope.
    outer_callees = raw.get("pkg.mod.outer", [])
    assert "pkg.mod.outer.middle._deep" not in outer_callees


# ---------------------------------------------------------------------------
# Motivating exemplar structure: _render_chapter_contract pattern
# ---------------------------------------------------------------------------

def test_motivating_exemplar_render_contract_pattern(tmp_path: Path) -> None:
    """Reproduces the motivating _table inside _render_chapter_contract pattern."""
    root = _make_package(tmp_path, "pkg", {
        "writer.py": (
            "class ContractWriter:\n"
            "    def render(self, cc):\n"
            "        def _table(title, rows):\n"
            "            return [title] + [str(r) for r in rows]\n"
            "        lines = []\n"
            "        lines.extend(_table('Threads', cc.threads))\n"
            "        lines.extend(_table('Facts', cc.facts))\n"
            "        return lines\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # The render method calls _table (nested inside render) — must resolve.
    assert "pkg.writer.ContractWriter.render._table" in raw.get(
        "pkg.writer.ContractWriter.render", []
    )
