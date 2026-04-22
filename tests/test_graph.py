from __future__ import annotations

from pathlib import Path

import pytest

from pycg_mcp.graph import CallGraphIndex


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "sample").mkdir()
    (tmp_path / "sample" / "__init__.py").write_text("")
    (tmp_path / "sample" / "a.py").write_text(
        "from sample.b import helper\n"
        "def top():\n"
        "    return helper()\n"
    )
    (tmp_path / "sample" / "b.py").write_text(
        "def helper():\n"
        "    return inner()\n"
        "def inner():\n"
        "    return 1\n"
    )
    return tmp_path


def test_build_and_basic_queries(sample_repo: Path) -> None:
    idx = CallGraphIndex.build(sample_repo, package="sample")
    stats = idx.stats()
    assert stats["functions"] > 0
    hits = idx.search("helper")
    assert any("helper" in h for h in hits)
