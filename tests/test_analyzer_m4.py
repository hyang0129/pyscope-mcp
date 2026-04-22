"""Unit tests for M4 analyzer features: relative imports, error isolation, determinism."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyscope_mcp.analyzer import build_raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_structure(tmp_path: Path, structure: dict[str, str | bytes]) -> Path:
    """Create an arbitrary file structure under tmp_path.

    Keys are paths relative to tmp_path.  Values may be str (text) or bytes
    (written raw — used to inject invalid UTF-8).
    Returns tmp_path so build_raw can be called with it.
    """
    for rel, content in structure.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)
    return tmp_path


# ---------------------------------------------------------------------------
# Test 1: from . import utils  →  utils.helper()
# ---------------------------------------------------------------------------


def test_relative_import_same_package(tmp_path: Path) -> None:
    """`from . import utils` then `utils.helper()` resolves to mypkg.utils.helper."""
    root = _make_structure(
        tmp_path,
        {
            "mypkg/__init__.py": "",
            "mypkg/utils.py": "def helper(): pass\n",
            "mypkg/main.py": (
                "from . import utils\n"
                "\n"
                "def run():\n"
                "    utils.helper()\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    assert "mypkg.main.run" in result, (
        f"Expected mypkg.main.run in result keys. Got: {list(result.keys())}"
    )
    assert "mypkg.utils.helper" in result["mypkg.main.run"], (
        f"Expected edge to mypkg.utils.helper. Got: {result.get('mypkg.main.run')}"
    )


# ---------------------------------------------------------------------------
# Test 2: from .sub.utils import helper  →  helper()
# ---------------------------------------------------------------------------


def test_relative_import_from_sub(tmp_path: Path) -> None:
    """`from .sub.utils import helper` in main.py; `helper()` in run() resolves correctly."""
    root = _make_structure(
        tmp_path,
        {
            "mypkg/__init__.py": "",
            "mypkg/sub/__init__.py": "",
            "mypkg/sub/utils.py": "def helper(): pass\n",
            "mypkg/main.py": (
                "from .sub.utils import helper\n"
                "\n"
                "def run():\n"
                "    helper()\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    assert "mypkg.main.run" in result, (
        f"Expected mypkg.main.run in result keys. Got: {list(result.keys())}"
    )
    assert "mypkg.sub.utils.helper" in result["mypkg.main.run"], (
        f"Expected edge to mypkg.sub.utils.helper. Got: {result.get('mypkg.main.run')}"
    )


# ---------------------------------------------------------------------------
# Test 3: from .sub import utils as u  →  u.func()
# ---------------------------------------------------------------------------


def test_relative_import_alias(tmp_path: Path) -> None:
    """`from .sub import utils as u` then `u.func()` resolves to the correct FQN."""
    root = _make_structure(
        tmp_path,
        {
            "mypkg/__init__.py": "",
            "mypkg/sub/__init__.py": "",
            "mypkg/sub/utils.py": "def func(): pass\n",
            "mypkg/main.py": (
                "from .sub import utils as u\n"
                "\n"
                "def caller():\n"
                "    u.func()\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    assert "mypkg.main.caller" in result, (
        f"Expected mypkg.main.caller in result keys. Got: {list(result.keys())}"
    )
    # `from .sub import utils as u` maps `u` -> `mypkg.sub.utils`
    # then `u.func()` should resolve to `mypkg.sub.utils.func`
    assert "mypkg.sub.utils.func" in result["mypkg.main.caller"], (
        f"Expected edge to mypkg.sub.utils.func. Got: {result.get('mypkg.main.caller')}"
    )


# ---------------------------------------------------------------------------
# Test 4: from .. import util  (level-2 relative import from nested module)
# ---------------------------------------------------------------------------


def test_relative_import_level2(tmp_path: Path) -> None:
    """`from .. import util` in mypkg/sub/deep.py resolves to mypkg.util."""
    root = _make_structure(
        tmp_path,
        {
            "mypkg/__init__.py": "",
            "mypkg/util.py": "def top_fn(): pass\n",
            "mypkg/sub/__init__.py": "",
            "mypkg/sub/deep.py": (
                "from .. import util\n"
                "\n"
                "def go():\n"
                "    util.top_fn()\n"
            ),
        },
    )
    result = build_raw(root, "mypkg")

    assert "mypkg.sub.deep.go" in result, (
        f"Expected mypkg.sub.deep.go in result keys. Got: {list(result.keys())}"
    )
    assert "mypkg.util.top_fn" in result["mypkg.sub.deep.go"], (
        f"Expected edge to mypkg.util.top_fn. Got: {result.get('mypkg.sub.deep.go')}"
    )


# ---------------------------------------------------------------------------
# Test 5: error isolation — invalid UTF-8 file is skipped, others still analyzed
# ---------------------------------------------------------------------------


def test_error_isolation_unicode(tmp_path: Path) -> None:
    """A file with invalid UTF-8 bytes must be skipped; good files are still analyzed."""
    # Write raw bytes that are not valid UTF-8 as "source"
    invalid_bytes = b"def bad(): pass\n\x80\x81\xff\xfe invalid bytes here\n"

    root = _make_structure(
        tmp_path,
        {
            "mypkg/__init__.py": "",
            # Good file that defines a real edge
            "mypkg/good.py": (
                "def helper(): pass\n"
                "\n"
                "def caller():\n"
                "    helper()\n"
            ),
            # Bad file with invalid UTF-8
            "mypkg/bad.py": invalid_bytes,
        },
    )

    # Must not raise, even with an unreadable / unparseable file
    result = build_raw(root, "mypkg")

    # The good file's edge must still be present
    assert "mypkg.good.caller" in result, (
        f"Expected mypkg.good.caller from the good file. Got: {list(result.keys())}"
    )
    assert "mypkg.good.helper" in result["mypkg.good.caller"], (
        f"Expected edge from good.caller to good.helper. Got: {result.get('mypkg.good.caller')}"
    )


# ---------------------------------------------------------------------------
# Test 6: determinism — identical output across multiple runs
# ---------------------------------------------------------------------------


def test_determinism_full(tmp_path: Path) -> None:
    """build_raw returns identical dicts with sorted keys and sorted value lists on every call."""
    root = _make_structure(
        tmp_path,
        {
            "mypkg/__init__.py": "",
            "mypkg/alpha.py": (
                "def a1(): pass\n"
                "def a2(): pass\n"
                "\n"
                "def run():\n"
                "    a1()\n"
                "    a2()\n"
            ),
            "mypkg/beta.py": (
                "from .alpha import a1, a2\n"
                "\n"
                "def go():\n"
                "    a1()\n"
                "    a2()\n"
            ),
            "mypkg/gamma.py": (
                "from . import alpha\n"
                "\n"
                "class Worker:\n"
                "    def work(self):\n"
                "        alpha.a1()\n"
            ),
        },
    )

    runs = [build_raw(root, "mypkg") for _ in range(3)]

    # All three runs must be identical
    assert runs[0] == runs[1] == runs[2], (
        f"build_raw produced non-deterministic output across 3 runs.\n"
        f"Run 0: {runs[0]}\nRun 1: {runs[1]}\nRun 2: {runs[2]}"
    )

    # Keys must be sorted
    keys = list(runs[0].keys())
    assert keys == sorted(keys), (
        f"Result keys are not sorted. Got: {keys}"
    )

    # Each callee list must be sorted
    for caller, callees in runs[0].items():
        assert callees == sorted(callees), (
            f"Callee list for {caller!r} is not sorted. Got: {callees}"
        )
