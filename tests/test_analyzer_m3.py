"""Unit tests for M3 analyzer features: self-call resolution, deep attr chains, nested scopes."""

from __future__ import annotations

from pathlib import Path


from pyscope_mcp.analyzer import build_raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_package(tmp_path: Path, pkg_name: str, files: dict[str, str]) -> Path:
    """Create a package directory with __init__.py and given files."""
    pkg = tmp_path / pkg_name
    pkg.mkdir()
    if "__init__.py" not in files:
        (pkg / "__init__.py").write_text("")
    for rel, content in files.items():
        target = pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


def _make_nested_package(tmp_path: Path, structure: dict[str, str]) -> Path:
    """Create arbitrary file structure under tmp_path.

    Keys are paths relative to tmp_path (e.g. "mypkg/utils.py").
    Returns tmp_path so build_raw can be called with it.
    """
    for rel, content in structure.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


# ---------------------------------------------------------------------------
# Test 1: self.other() inside sibling method resolves to same-class FQN
# ---------------------------------------------------------------------------


def test_self_call_sibling_method(tmp_path: Path) -> None:
    """Class with two methods; caller() calls self.other() — edge should resolve."""
    root = _make_package(
        tmp_path,
        "mypkg",
        {
            "service.py": (
                "class MyService:\n"
                "    def other(self):\n"
                "        pass\n"
                "\n"
                "    def caller(self):\n"
                "        self.other()\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    assert "mypkg.service.MyService.caller" in result, (
        f"Expected caller FQN in result keys. Got: {list(result.keys())}"
    )
    assert "mypkg.service.MyService.other" in result["mypkg.service.MyService.caller"], (
        f"Expected edge to sibling method. Got: {result.get('mypkg.service.MyService.caller')}"
    )


# ---------------------------------------------------------------------------
# Test 2: self.inherited_method() not defined on class → no edge, no crash
# ---------------------------------------------------------------------------


def test_self_call_unresolved_dropped(tmp_path: Path) -> None:
    """self.inherited_method() where inherited_method is not on the class — drop silently."""
    root = _make_package(
        tmp_path,
        "mypkg",
        {
            "child.py": (
                "class Child:\n"
                "    def do_work(self):\n"
                "        self.inherited_method()\n"
            ),
        },
    )
    # Must not raise
    result = build_raw(root, "mypkg")

    all_callees: list[str] = [c for callees in result.values() for c in callees]
    assert not any("inherited_method" in c for c in all_callees), (
        f"Unresolved inherited_method leaked into graph: {all_callees}"
    )


# ---------------------------------------------------------------------------
# Test 3: deep attribute chain mypkg.utils.process() resolved via import table
# ---------------------------------------------------------------------------


def test_deep_attr_chain_cross_module(tmp_path: Path) -> None:
    """import mypkg.utils; func calls mypkg.utils.process() — edge resolved."""
    root = _make_nested_package(
        tmp_path,
        {
            "mypkg/__init__.py": "",
            "mypkg/utils.py": "def process(): pass\n",
            "mypkg/main.py": (
                "import mypkg.utils\n"
                "\n"
                "def run():\n"
                "    mypkg.utils.process()\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    assert "mypkg.main.run" in result, (
        f"Expected mypkg.main.run in result. Got: {list(result.keys())}"
    )
    assert "mypkg.utils.process" in result["mypkg.main.run"], (
        f"Expected edge to mypkg.utils.process. Got: {result.get('mypkg.main.run')}"
    )


# ---------------------------------------------------------------------------
# Test 4: nested function inside a method — no crash, inner calls captured
# ---------------------------------------------------------------------------


def test_nested_function_no_crash(tmp_path: Path) -> None:
    """Inner function inside a regular function makes calls — no exception, edges captured."""
    root = _make_package(
        tmp_path,
        "mypkg",
        {
            "nested.py": (
                "def helper(): pass\n"
                "\n"
                "def outer():\n"
                "    def inner():\n"
                "        helper()\n"
                "    inner()\n"
            ),
        },
    )
    # Must not raise
    result = build_raw(root, "mypkg")

    # helper() called from inside inner() should appear somewhere
    all_callees: list[str] = [c for callees in result.values() for c in callees]
    assert "mypkg.nested.helper" in all_callees, (
        f"Expected helper to be in callees from nested scope. Got: {all_callees}"
    )


# ---------------------------------------------------------------------------
# Test 5: class method calls a module-level function in the same file
# ---------------------------------------------------------------------------


def test_method_calls_top_level_in_same_module(tmp_path: Path) -> None:
    """A method calls a top-level function in the same module — edge captured."""
    root = _make_package(
        tmp_path,
        "mypkg",
        {
            "worker.py": (
                "def validate(data): pass\n"
                "\n"
                "class Worker:\n"
                "    def run(self, data):\n"
                "        validate(data)\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    assert "mypkg.worker.Worker.run" in result, (
        f"Expected Worker.run in result. Got: {list(result.keys())}"
    )
    assert "mypkg.worker.validate" in result["mypkg.worker.Worker.run"], (
        f"Expected edge to module-level validate. Got: {result.get('mypkg.worker.Worker.run')}"
    )


# ---------------------------------------------------------------------------
# Test 6: two classes with self.helper() — each resolves to its own class only
# ---------------------------------------------------------------------------


def test_multiple_classes_self_calls_isolated(tmp_path: Path) -> None:
    """Two classes each call self.helper(); each resolves to its own helper, not the other's."""
    root = _make_package(
        tmp_path,
        "mypkg",
        {
            "duo.py": (
                "class Alpha:\n"
                "    def helper(self): pass\n"
                "\n"
                "    def run(self):\n"
                "        self.helper()\n"
                "\n"
                "class Beta:\n"
                "    def helper(self): pass\n"
                "\n"
                "    def run(self):\n"
                "        self.helper()\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    alpha_run = result.get("mypkg.duo.Alpha.run", [])
    beta_run = result.get("mypkg.duo.Beta.run", [])

    # Each run() must resolve only to its own class's helper
    assert "mypkg.duo.Alpha.helper" in alpha_run, (
        f"Alpha.run should call Alpha.helper. Got: {alpha_run}"
    )
    assert "mypkg.duo.Beta.helper" in beta_run, (
        f"Beta.run should call Beta.helper. Got: {beta_run}"
    )

    # Cross-class edges must not exist
    assert "mypkg.duo.Beta.helper" not in alpha_run, (
        f"Alpha.run must not reference Beta.helper. Got: {alpha_run}"
    )
    assert "mypkg.duo.Alpha.helper" not in beta_run, (
        f"Beta.run must not reference Alpha.helper. Got: {beta_run}"
    )
