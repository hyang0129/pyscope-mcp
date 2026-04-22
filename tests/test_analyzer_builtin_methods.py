"""Tests for the builtin-method-call accepted-miss classifier."""

from __future__ import annotations

import ast
from pathlib import Path

from pyscope_mcp.analyzer import build_raw, build_with_report, MissLog, _classify_miss


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


def _parse_call(src: str) -> ast.Call:
    """Parse a single expression statement and return the Call node."""
    tree = ast.parse(src)
    expr = tree.body[0]
    assert isinstance(expr, ast.Expr)
    call = expr.value
    assert isinstance(call, ast.Call)
    return call


# ---------------------------------------------------------------------------
# Classifier unit tests (pattern tag only, no full pipeline needed)
# ---------------------------------------------------------------------------

def test_list_append_is_accepted() -> None:
    call = _parse_call("lines.append(x)")
    assert _classify_miss(call) == "builtin_method_call"


def test_dict_get_is_accepted() -> None:
    call = _parse_call('d.get("k")')
    assert _classify_miss(call) == "builtin_method_call"


def test_str_format_is_accepted() -> None:
    call = _parse_call("s.format(x=1)")
    assert _classify_miss(call) == "builtin_method_call"


def test_pathlib_write_text_stays_attr_chain() -> None:
    """write_text is not in the whitelist — must stay attr_chain_unresolved."""
    call = _parse_call("path.write_text('hello')")
    assert _classify_miss(call) == "attr_chain_unresolved"


def test_get_value_method_stays_attr_chain() -> None:
    """get_value is not in the whitelist (whole name, not prefix)."""
    call = _parse_call("obj.get_value()")
    assert _classify_miss(call) == "attr_chain_unresolved"


# ---------------------------------------------------------------------------
# Pipeline tests — self.append is resolved in-package when class defines it
# ---------------------------------------------------------------------------

def test_self_append_when_class_defines_it_resolves_in_package(tmp_path: Path) -> None:
    """A class with def append should resolve self.append as in-package, no miss."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class MyList:\n"
            "    def append(self, x):\n"
            "        self._data = getattr(self, '_data', [])\n"
            "        self._data.append(x)\n"
            "    def add(self, x):\n"
            "        self.append(x)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    # self.append in add() should resolve to pkg.mod.MyList.append
    assert "pkg.mod.MyList.append" in raw.get("pkg.mod.MyList.add", [])


# ---------------------------------------------------------------------------
# MissLog.to_dict rollup tests
# ---------------------------------------------------------------------------

def test_to_dict_rollup_top5() -> None:
    log = MissLog()
    # Record accepted calls: a=3, b=5, c=1, d=2, e=4, f=6  (total=21)
    for _ in range(3):
        log.record_accepted("builtin_method_call", "a.py")
    for _ in range(5):
        log.record_accepted("builtin_method_call", "b.py")
    for _ in range(1):
        log.record_accepted("builtin_method_call", "c.py")
    for _ in range(2):
        log.record_accepted("builtin_method_call", "d.py")
    for _ in range(4):
        log.record_accepted("builtin_method_call", "e.py")
    for _ in range(6):
        log.record_accepted("builtin_method_call", "f.py")

    report = log.to_dict({}, set())
    summary = report["summary"]

    assert summary["calls_accepted"] == 21
    assert summary["accepted_counts"] == {"builtin_method_call": 21}

    top5 = summary["rollups"]["builtin_method_modules"]
    assert top5 == [
        {"module": "f.py", "count": 6},
        {"module": "b.py", "count": 5},
        {"module": "e.py", "count": 4},
        {"module": "a.py", "count": 3},
        {"module": "d.py", "count": 2},
    ]


# ---------------------------------------------------------------------------
# calls_total invariant: total == in_package + external + accepted + unresolved
# ---------------------------------------------------------------------------

def test_calls_total_invariant(tmp_path: Path) -> None:
    """calls_total must equal the sum of all four accounting buckets."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            # in-package resolved: helper()
            "def helper():\n"
            "    return 1\n"
            "\n"
            # external resolved: len(x)  — actually bare_name; use imported external
            "from os.path import join as path_join\n"
            "\n"
            "def caller(x, lines, d):\n"
            "    helper()          # in-package\n"
            "    path_join('a', 'b')  # external\n"
            "    lines.append(x)   # accepted\n"
            "    d.get('k')        # accepted\n"
            "    unknown_fn(x)     # unresolved bare_name\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    s = report["summary"]
    total = s["calls_total"]
    parts = (
        s["calls_resolved_in_package"]
        + s["calls_resolved_external"]
        + s["calls_accepted"]
        + s["calls_unresolved"]
    )
    assert total == parts, (
        f"Invariant broken: total={total} != "
        f"in_pkg={s['calls_resolved_in_package']} + "
        f"ext={s['calls_resolved_external']} + "
        f"accepted={s['calls_accepted']} + "
        f"unresolved={s['calls_unresolved']}"
    )
