"""M7: pathlib / concurrent.futures / pydantic accepted-miss whitelists.

Each test family confirms a method-name is classified into its accepted bucket,
not left in unresolved_calls.  The false-positive guard verifies that a
user-defined class with a same-named method is resolved as in-package and
never reaches the whitelist classifier.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pyscope_mcp.analyzer import build_raw, build_with_report, _classify_miss


def _parse_call(src: str) -> ast.Call:
    tree = ast.parse(src)
    expr = tree.body[0]
    assert isinstance(expr, ast.Expr)
    call = expr.value
    assert isinstance(call, ast.Call)
    return call


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
# Classifier unit tests — pathlib
# ---------------------------------------------------------------------------

def test_pathlib_exists_accepted() -> None:
    call = _parse_call("p.exists()")
    assert _classify_miss(call) == "pathlib_method_call"


def test_pathlib_read_text_accepted() -> None:
    call = _parse_call("p.read_text(encoding='utf-8')")
    assert _classify_miss(call) == "pathlib_method_call"


def test_pathlib_mkdir_accepted() -> None:
    call = _parse_call("output_dir.mkdir(parents=True, exist_ok=True)")
    assert _classify_miss(call) == "pathlib_method_call"


def test_pathlib_iterdir_accepted() -> None:
    call = _parse_call("base.iterdir()")
    assert _classify_miss(call) == "pathlib_method_call"


def test_pathlib_write_bytes_accepted() -> None:
    call = _parse_call("f.write_bytes(data)")
    assert _classify_miss(call) == "pathlib_method_call"


def test_pathlib_glob_accepted() -> None:
    call = _parse_call("root.glob('**/*.py')")
    assert _classify_miss(call) == "pathlib_method_call"


def test_pathlib_joinpath_accepted() -> None:
    call = _parse_call("base.joinpath('sub', 'file.txt')")
    assert _classify_miss(call) == "pathlib_method_call"


# ---------------------------------------------------------------------------
# Classifier unit tests — concurrent.futures
# ---------------------------------------------------------------------------

def test_futures_result_accepted() -> None:
    call = _parse_call("fut.result(timeout=5)")
    assert _classify_miss(call) == "futures_method_call"


def test_futures_shutdown_accepted() -> None:
    call = _parse_call("executor.shutdown(wait=True)")
    assert _classify_miss(call) == "futures_method_call"


def test_futures_cancel_accepted() -> None:
    call = _parse_call("f.cancel()")
    assert _classify_miss(call) == "futures_method_call"


def test_futures_done_accepted() -> None:
    call = _parse_call("future.done()")
    assert _classify_miss(call) == "futures_method_call"


def test_futures_add_done_callback_accepted() -> None:
    call = _parse_call("fut.add_done_callback(cb)")
    assert _classify_miss(call) == "futures_method_call"


def test_futures_submit_accepted() -> None:
    call = _parse_call("executor.submit(fn, 1)")
    assert _classify_miss(call) == "futures_method_call"


def test_futures_map_accepted() -> None:
    call = _parse_call("executor.map(fn, items)")
    assert _classify_miss(call) == "futures_method_call"


# ---------------------------------------------------------------------------
# Classifier unit tests — pydantic BaseModel
# ---------------------------------------------------------------------------

def test_pydantic_model_dump_accepted() -> None:
    call = _parse_call("obj.model_dump()")
    assert _classify_miss(call) == "pydantic_method_call"


def test_pydantic_model_dump_json_accepted() -> None:
    call = _parse_call("obj.model_dump_json(indent=2)")
    assert _classify_miss(call) == "pydantic_method_call"


def test_pydantic_model_validate_accepted() -> None:
    call = _parse_call("MyModel.model_validate(data)")
    assert _classify_miss(call) == "pydantic_method_call"


def test_pydantic_model_copy_accepted() -> None:
    call = _parse_call("record.model_copy(update={'x': 1})")
    assert _classify_miss(call) == "pydantic_method_call"


def test_pydantic_model_validate_json_accepted() -> None:
    call = _parse_call("MyModel.model_validate_json(raw)")
    assert _classify_miss(call) == "pydantic_method_call"


# ---------------------------------------------------------------------------
# Pipeline tests — new patterns appear in accepted_counts, not unresolved_calls
# ---------------------------------------------------------------------------

def test_pathlib_method_in_accepted_counts(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from pathlib import Path\n"
            "\n"
            "def read_config(cfg: Path) -> str:\n"
            "    return cfg.read_text(encoding='utf-8')\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    summary = report["summary"]
    assert summary["accepted_counts"].get("pathlib_method_call", 0) >= 1
    patterns = {e["pattern"] for e in report["unresolved_calls"]}
    assert "pathlib_method_call" not in patterns


def test_futures_method_in_accepted_counts(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from concurrent.futures import Future\n"
            "\n"
            "def collect(fut: Future) -> object:\n"
            "    return fut.result(timeout=10)\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    summary = report["summary"]
    assert summary["accepted_counts"].get("futures_method_call", 0) >= 1
    patterns = {e["pattern"] for e in report["unresolved_calls"]}
    assert "futures_method_call" not in patterns


def test_pydantic_super_init_routed_to_accepted(tmp_path: Path) -> None:
    """super().__init__(**data) on a BaseModel subclass is accepted, not super_unresolved."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from pydantic import BaseModel\n"
            "\n"
            "class Config(BaseModel):\n"
            "    name: str\n"
            "\n"
            "    def __init__(self, **data):\n"
            "        super().__init__(**data)\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    summary = report["summary"]
    assert summary["accepted_counts"].get("pydantic_method_call", 0) >= 1
    # super_unresolved for the BaseModel subclass init should not be recorded.
    patterns = {e["pattern"] for e in report["unresolved_calls"]}
    assert "super_unresolved" not in patterns


def test_pydantic_super_init_transitive_base(tmp_path: Path) -> None:
    """super().__init__() routed to accepted when BaseModel is a grandparent."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from pydantic import BaseModel\n"
            "\n"
            "class Parent(BaseModel):\n"
            "    pass\n"
            "\n"
            "class Child(Parent):\n"
            "    def __init__(self, **data):\n"
            "        super().__init__(**data)\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    summary = report["summary"]
    assert summary["accepted_counts"].get("pydantic_method_call", 0) >= 1


def test_non_pydantic_super_init_stays_super_unresolved(tmp_path: Path) -> None:
    """False-positive guard: class with no BaseModel ancestor stays super_unresolved."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Thing:\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert report["summary"]["accepted_counts"].get("pydantic_method_call", 0) == 0


def test_pydantic_method_in_accepted_counts(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from pydantic import BaseModel\n"
            "\n"
            "class Config(BaseModel):\n"
            "    name: str\n"
            "\n"
            "def dump(cfg: Config) -> dict:\n"
            "    return cfg.model_dump()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    summary = report["summary"]
    assert summary["accepted_counts"].get("pydantic_method_call", 0) >= 1
    patterns = {e["pattern"] for e in report["unresolved_calls"]}
    assert "pydantic_method_call" not in patterns


# ---------------------------------------------------------------------------
# False-positive guard — user-defined class with same method name must resolve
# in-package, never hit the whitelist
# ---------------------------------------------------------------------------

def test_user_class_submit_resolves_in_package_not_accepted(tmp_path: Path) -> None:
    """A user class with its own .submit must resolve via self-method, not whitelist."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class MyQueue:\n"
            "    def submit(self, item):\n"
            "        pass\n"
            "\n"
            "    def enqueue(self, item):\n"
            "        self.submit(item)\n"
        ),
    })
    raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    # self.submit resolves to pkg.mod.MyQueue.submit (in-package)
    assert "pkg.mod.MyQueue.submit" in raw.get("pkg.mod.MyQueue.enqueue", [])
    # Nothing should be in futures_method_call accepted bucket for this package
    assert report["summary"]["accepted_counts"].get("futures_method_call", 0) == 0


def test_user_class_model_dump_resolves_in_package_not_accepted(tmp_path: Path) -> None:
    """A user class that defines model_dump resolves via self-method, not pydantic whitelist."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class MySerializer:\n"
            "    def model_dump(self) -> dict:\n"
            "        return {}\n"
            "\n"
            "    def serialize(self):\n"
            "        return self.model_dump()\n"
        ),
    })
    raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert "pkg.mod.MySerializer.model_dump" in raw.get("pkg.mod.MySerializer.serialize", [])
    assert report["summary"]["accepted_counts"].get("pydantic_method_call", 0) == 0
