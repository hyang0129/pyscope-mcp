"""Integration test for issue #18: all four pass-1 extension handlers working together.

Tests that a package using patterns from all four handlers simultaneously produces
the correct resolved edges and accepted buckets.

Handlers covered:
  H1 - External-factory local-var accept (httpx, boto3, typer, googleapi)
  H2 - PEP 604 union + Optional/Union subscript in annotations
  H3 - ClassName.__new__(...) as constructor
  H4 - Import-alias stdlib investigation (verified as working, not a new handler)
"""

from __future__ import annotations

from pathlib import Path

from pyscope_mcp.analyzer import build_raw, build_with_report


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
# Combined: all 4 handlers in one package
# ---------------------------------------------------------------------------

def test_all_four_handlers_combined(tmp_path: Path) -> None:
    """Realistic package that exercises H1/H2/H3/H4 simultaneously.

    - H2: agent param annotation `WorkerAgent | None = None`
    - H3: `_tmp = WorkerAgent.__new__(WorkerAgent)` binding
    - H1: `client = httpx.Client(); client.post(...)`
    - H4: `import sys as sys_mod; sys_mod.exit(...)` (alias)
    """
    root = _make_package(tmp_path, "pkg", {
        "agent.py": (
            "class WorkerAgent:\n"
            "    def run(self) -> str:\n"
            "        return 'ok'\n"
            "    def compute(self) -> int:\n"
            "        return 42\n"
        ),
        "pipeline.py": (
            "import sys as sys_mod\n"
            "import httpx\n"
            "from typing import Optional\n"
            "from pkg.agent import WorkerAgent\n"
            "\n"
            # H2: union annotation resolves WorkerAgent
            "def process(agent: WorkerAgent | None = None) -> None:\n"
            "    if agent is not None:\n"
            "        agent.run()\n"
            "\n"
            # H3: __new__ binding + downstream method call
            "def bypass_init() -> None:\n"
            "    _tmp = WorkerAgent.__new__(WorkerAgent)\n"
            "    _tmp.compute()\n"
            "\n"
            # H1: external factory local var
            "def upload(url: str, data: bytes) -> None:\n"
            "    client = httpx.Client()\n"
            "    client.post(url, content=data)\n"
            "\n"
            # H4: import alias for stdlib
            "def maybe_exit(code: int) -> None:\n"
            "    if code != 0:\n"
            "        sys_mod.exit(code)\n"
        ),
    })
    raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    summary = report["summary"]

    # H2: agent.run() from union annotation
    assert "pkg.agent.WorkerAgent.run" in raw.get("pkg.pipeline.process", []), (
        f"H2: agent.run() not resolved; callees={raw.get('pkg.pipeline.process')}"
    )

    # H3: _tmp.compute() from __new__ binding
    assert "pkg.agent.WorkerAgent.compute" in raw.get("pkg.pipeline.bypass_init", []), (
        f"H3: compute() not resolved via __new__; callees={raw.get('pkg.pipeline.bypass_init')}"
    )

    # H1: httpx_method_call accepted
    assert summary["accepted_counts"].get("httpx_method_call", 0) >= 1, (
        f"H1: httpx_method_call not accepted; accepted={summary['accepted_counts']}"
    )

    # H4: sys_mod.exit() not in attr_chain_unresolved
    unresolved_patterns = report["pattern_counts"]
    assert unresolved_patterns.get("attr_chain_unresolved", 0) == 0, (
        f"H4: attr_chain_unresolved should be 0; pattern_counts={unresolved_patterns}"
    )


def test_h2_optional_subscript_with_h3_new(tmp_path: Path) -> None:
    """H2 Optional[X] annotation + H3 __new__ binding, both in same function."""
    root = _make_package(tmp_path, "pkg", {
        "model.py": (
            "class MyModel:\n"
            "    def save(self): pass\n"
            "    def load(self): pass\n"
        ),
        "ops.py": (
            "from typing import Optional\n"
            "from pkg.model import MyModel\n"
            "\n"
            "def restore(model: Optional[MyModel] = None) -> None:\n"
            "    if model is not None:\n"
            "        model.save()\n"
            "    raw = MyModel.__new__(MyModel)\n"
            "    raw.load()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.ops.restore", [])
    assert "pkg.model.MyModel.save" in callees, (
        f"H2: save() not resolved; callees={callees}"
    )
    assert "pkg.model.MyModel.load" in callees, (
        f"H3: load() not resolved via __new__; callees={callees}"
    )


def test_h1_boto3_and_h2_union_no_false_positives(tmp_path: Path) -> None:
    """H1 boto3 local var + H2 union annotation — both fire, no false positives."""
    root = _make_package(tmp_path, "pkg", {
        "worker.py": (
            "class DataWorker:\n"
            "    def process(self, data): pass\n"
        ),
        "runner.py": (
            "import boto3\n"
            "from pkg.worker import DataWorker\n"
            "\n"
            "def run(worker: DataWorker | None = None) -> None:\n"
            "    if worker is not None:\n"
            "        worker.process('data')\n"
            "    s3 = boto3.client('s3')\n"
            "    s3.put_object(Bucket='b', Key='k', Body=b'v')\n"
        ),
    })
    raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")

    # H2: worker.process() resolved in-package
    assert "pkg.worker.DataWorker.process" in raw.get("pkg.runner.run", []), (
        f"H2: process() not resolved; callees={raw.get('pkg.runner.run')}"
    )

    # H1: boto3_method_call accepted (not unresolved)
    assert report["summary"]["accepted_counts"].get("boto3_method_call", 0) >= 1, (
        f"H1: boto3_method_call not accepted; accepted={report['summary']['accepted_counts']}"
    )

    # FP guard: no attr_chain_unresolved
    assert report["pattern_counts"].get("attr_chain_unresolved", 0) == 0, (
        f"Unexpected attr_chain_unresolved; pattern_counts={report['pattern_counts']}"
    )
