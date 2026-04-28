"""PR 3: self-attr handler extended to cover builtin/pathlib sentinel types."""

from __future__ import annotations

from pathlib import Path

from pyscope_mcp.analyzer import build_with_report


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


def _accepted(report: dict, pattern: str) -> int:
    return report["summary"]["accepted_counts"].get(pattern, 0)


# ---------------------------------------------------------------------------
# self._config = {} — dict literal inferred as builtins.dict
# ---------------------------------------------------------------------------

def test_self_attr_dict_literal_get_is_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def __init__(self):\n"
            "        self._config = {}\n"
            "    def run(self):\n"
            "        return self._config.get('key')\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert _accepted(report, "builtin_method_call") >= 1
    assert report["pattern_counts"].get("self_method_unresolved", 0) == 0


# ---------------------------------------------------------------------------
# self._raw_dir = Path(path) — Path() call inferred as pathlib.Path
# ---------------------------------------------------------------------------

def test_self_attr_path_call_mkdir_is_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from pathlib import Path\n"
            "\n"
            "class Worker:\n"
            "    def __init__(self, path):\n"
            "        self._raw_dir = Path(path)\n"
            "    def prepare(self):\n"
            "        self._raw_dir.mkdir()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert _accepted(report, "pathlib_method_call") >= 1
    assert report["pattern_counts"].get("self_method_unresolved", 0) == 0


# ---------------------------------------------------------------------------
# self._config: dict[str, Any] — class-body annotation
# ---------------------------------------------------------------------------

def test_self_attr_class_body_annotation_dict_is_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from typing import Any\n"
            "\n"
            "class Worker:\n"
            "    _config: dict[str, Any]\n"
            "\n"
            "    def run(self):\n"
            "        return self._config.get('key')\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert _accepted(report, "builtin_method_call") >= 1
    assert report["pattern_counts"].get("self_method_unresolved", 0) == 0


# ---------------------------------------------------------------------------
# self._cfg = cfg where cfg: dict is a param annotation
# ---------------------------------------------------------------------------

def test_self_attr_param_annotation_dict_is_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def __init__(self, cfg: dict):\n"
            "        self._cfg = cfg\n"
            "    def run(self):\n"
            "        return self._cfg.get('key')\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert _accepted(report, "builtin_method_call") >= 1
    assert report["pattern_counts"].get("self_method_unresolved", 0) == 0


# ---------------------------------------------------------------------------
# list and set literals
# ---------------------------------------------------------------------------

def test_self_attr_list_literal_append_is_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Collector:\n"
            "    def __init__(self):\n"
            "        self._items = []\n"
            "    def add(self, x):\n"
            "        self._items.append(x)\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert _accepted(report, "builtin_method_call") >= 1


def test_self_attr_set_literal_add_is_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Tracker:\n"
            "    def __init__(self):\n"
            "        self._seen = set()\n"
            "    def mark(self, x):\n"
            "        self._seen.add(x)\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert _accepted(report, "builtin_method_call") >= 1


# ---------------------------------------------------------------------------
# False-positive guards — must NOT be accepted
# ---------------------------------------------------------------------------

def test_fpg_unknown_factory_rhs_stays_unresolved(tmp_path: Path) -> None:
    """self._cfg = something_unknown() — factory return type is unknown; must miss."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def something_unknown():\n"
            "    return {}\n"
            "\n"
            "class Worker:\n"
            "    def __init__(self):\n"
            "        self._cfg = something_unknown()\n"
            "    def run(self):\n"
            "        return self._cfg.get('key')\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    # .get() should NOT be accepted as builtin_method_call via sentinel path
    assert _accepted(report, "builtin_method_call") == 0
    # It should remain unresolved (self_method_unresolved or builtin_method_call via classify_miss)
    total_misses = report["summary"]["calls_unresolved"] + report["summary"]["calls_accepted"]
    assert total_misses > 0


def test_fpg_string_rhs_path_method_stays_unresolved(tmp_path: Path) -> None:
    """self._path = some_string — no chain to Path; mkdir must not be accepted."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Worker:\n"
            "    def __init__(self, s: str):\n"
            "        self._path = s\n"
            "    def prepare(self):\n"
            "        self._path.mkdir()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    # .mkdir() on a str-typed attr must NOT route to pathlib_method_call via sentinel
    # (str param annotation → no sentinel, so no intercept before classify_miss)
    assert _accepted(report, "pathlib_method_call") == 0
