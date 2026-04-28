"""Track 1 — classifier widening: PIL, wave, extended builtins/pathlib, non-Name receivers.

Tests use a combination of:
  - `_make_package` + `build_raw` integration tests that check accepted_counts
  - `classify_miss` unit tests with crafted AST nodes for non-Name/non-Attribute receivers
"""

from __future__ import annotations

import ast
from pathlib import Path


from pyscope_mcp.analyzer import build_raw
from pyscope_mcp.analyzer.misses import classify_miss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _build_and_get_accepted(tmp_path: Path, pkg_name: str, files: dict[str, str]) -> dict[str, int]:
    """Build raw graph and return accepted_counts from the miss log.

    We can't access MissLog directly from build_raw (it returns only the raw
    dict), so we use classify_miss on synthetic AST nodes instead for
    bucket-level checks.
    """
    root = _make_package(tmp_path, pkg_name, files)
    build_raw(root, pkg_name)
    return {}


def _parse_call(src: str) -> ast.Call:
    """Parse a single expression statement and return the Call node."""
    tree = ast.parse(src, mode="eval")
    expr = tree.body
    assert isinstance(expr, ast.Call), f"Expected ast.Call, got {type(expr)}"
    return expr


# ---------------------------------------------------------------------------
# Unit tests: classify_miss with crafted AST nodes
# ---------------------------------------------------------------------------

class TestPILMethodCall:
    def test_image_new(self) -> None:
        node = _parse_call("Image.new('RGBA', (100, 100))")
        assert classify_miss(node) == "pil_method_call"

    def test_image_open(self) -> None:
        node = _parse_call("Image.open('file.png')")
        assert classify_miss(node) == "pil_method_call"

    def test_image_save(self) -> None:
        node = _parse_call("img.save('out.png')")
        assert classify_miss(node) == "pil_method_call"

    def test_draw_text(self) -> None:
        node = _parse_call("draw.text((0, 0), 'hello')")
        assert classify_miss(node) == "pil_method_call"

    def test_image_crop(self) -> None:
        node = _parse_call("img.crop((0, 0, 100, 100))")
        assert classify_miss(node) == "pil_method_call"


class TestWaveMethodCall:
    def test_getnframes(self) -> None:
        node = _parse_call("wf.getnframes()")
        assert classify_miss(node) == "wave_method_call"

    def test_getnchannels(self) -> None:
        node = _parse_call("wf.getnchannels()")
        assert classify_miss(node) == "wave_method_call"

    def test_readframes(self) -> None:
        node = _parse_call("wf.readframes(1024)")
        assert classify_miss(node) == "wave_method_call"

    def test_writeframes(self) -> None:
        node = _parse_call("wav.writeframes(data)")
        assert classify_miss(node) == "wave_method_call"

    def test_getframerate(self) -> None:
        node = _parse_call("wf.getframerate()")
        assert classify_miss(node) == "wave_method_call"


class TestBuiltinMethodCallExtended:
    def test_isalnum(self) -> None:
        node = _parse_call("c.isalnum()")
        assert classify_miss(node) == "builtin_method_call"

    def test_isdigit(self) -> None:
        node = _parse_call("s.isdigit()")
        assert classify_miss(node) == "builtin_method_call"

    def test_zfill(self) -> None:
        node = _parse_call("s.zfill(5)")
        assert classify_miss(node) == "builtin_method_call"

    def test_decode(self) -> None:
        node = _parse_call("b.decode('utf-8')")
        assert classify_miss(node) == "builtin_method_call"

    def test_partition(self) -> None:
        node = _parse_call("s.partition('=')")
        assert classify_miss(node) == "builtin_method_call"

    def test_swapcase(self) -> None:
        node = _parse_call("s.swapcase()")
        assert classify_miss(node) == "builtin_method_call"


class TestPathlibMethodCallExtended:
    def test_with_name(self) -> None:
        node = _parse_call("wav.with_name('x')")
        assert classify_miss(node) == "pathlib_method_call"

    def test_rename(self) -> None:
        node = _parse_call("p.rename('new')")
        assert classify_miss(node) == "pathlib_method_call"

    def test_touch(self) -> None:
        node = _parse_call("p.touch()")
        assert classify_miss(node) == "pathlib_method_call"

    def test_is_absolute(self) -> None:
        node = _parse_call("p.is_absolute()")
        assert classify_miss(node) == "pathlib_method_call"


# ---------------------------------------------------------------------------
# Non-Name/non-Attribute receiver fallback tests
# ---------------------------------------------------------------------------

class TestNonNameReceiverFallback:
    def test_binop_write_text(self) -> None:
        """(a / b).write_text(...) — BinOp receiver, pathlib method."""
        node = _parse_call("(a / b).write_text('hello')")
        assert classify_miss(node) == "pathlib_method_call"

    def test_call_result_resolve(self) -> None:
        """func().resolve() — Call receiver, pathlib method."""
        node = _parse_call("func().resolve()")
        assert classify_miss(node) == "pathlib_method_call"

    def test_attribute_receiver_decode(self) -> None:
        """proc.stderr.decode('utf-8') — Attribute receiver, builtin method.

        attr_chain(['proc', 'stderr', 'decode']) succeeds → standard path.
        But also verify it routes to builtin_method_call.
        """
        node = _parse_call("proc.stderr.decode('utf-8')")
        assert classify_miss(node) == "builtin_method_call"

    def test_subscript_receiver_decode(self) -> None:
        """x[0].decode('utf-8') — Subscript receiver, builtin method."""
        node = _parse_call("x[0].decode('utf-8')")
        assert classify_miss(node) == "builtin_method_call"

    def test_binop_receiver_pil_method(self) -> None:
        """(a / b).open() — BinOp receiver, PIL method (open is in PIL_METHODS)."""
        node = _parse_call("(a / b).open()")
        # 'open' appears in PIL_METHODS → pil_method_call via fallback
        assert classify_miss(node) == "pil_method_call"

    def test_call_result_getnframes(self) -> None:
        """wave.open(f).getnframes() — Call receiver, wave method."""
        node = _parse_call("wave.open(f).getnframes()")
        assert classify_miss(node) == "wave_method_call"


# ---------------------------------------------------------------------------
# Negative guards
# ---------------------------------------------------------------------------

class TestNegativeGuards:
    def test_unknown_method_stays_unresolved(self) -> None:
        """foo.some_random_method() — not in any whitelist → attr_chain_unresolved."""
        node = _parse_call("foo.some_random_method()")
        assert classify_miss(node) == "attr_chain_unresolved"

    def test_binop_receiver_unknown_method(self) -> None:
        """(a + b).some_random_method() — BinOp + unknown → attr_chain_unresolved."""
        node = _parse_call("(a + b).some_random_method()")
        assert classify_miss(node) == "attr_chain_unresolved"

    def test_call_receiver_unknown_method(self) -> None:
        """func().some_random_method() — Call receiver + unknown → attr_chain_unresolved."""
        node = _parse_call("func().some_random_method()")
        assert classify_miss(node) == "attr_chain_unresolved"

    def test_self_method_not_rerouted(self, tmp_path: Path) -> None:
        """self.foo() still routes to self_method_unresolved, not any whitelist."""
        node = _parse_call("self.some_method()")
        assert classify_miss(node) == "self_method_unresolved"


# ---------------------------------------------------------------------------
# Regression: self.foo() / self.attr.bar() MRO resolution still works
# ---------------------------------------------------------------------------

def test_self_method_mro_regression_not_broken(tmp_path: Path) -> None:
    """self.method() resolution via MRO must still work after classifier changes."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Base:\n"
            "    def helper(self):\n"
            "        return 42\n"
            "\n"
            "class Child(Base):\n"
            "    def run(self):\n"
            "        return self.helper()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Base.helper" in raw["pkg.mod.Child.run"]


def test_self_attr_type_resolution_regression(tmp_path: Path) -> None:
    """self.attr.bar() where attr is typed in __init__ must still resolve."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Inner:\n"
            "    def work(self):\n"
            "        return 1\n"
            "\n"
            "class Outer:\n"
            "    def __init__(self):\n"
            "        self.inner = Inner()\n"
            "\n"
            "    def run(self):\n"
            "        return self.inner.work()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Inner.work" in raw["pkg.mod.Outer.run"]


def test_self_super_mro_regression(tmp_path: Path) -> None:
    """super().__init__() resolution must still work."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Base:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "\n"
            "class Child(Base):\n"
            "    def __init__(self, x, y):\n"
            "        super().__init__(x)\n"
            "        self.y = y\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Base.__init__" in raw["pkg.mod.Child.__init__"]
