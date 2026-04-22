from __future__ import annotations

from pathlib import Path

from pyscope_mcp.graph import CallGraphIndex


def _sample_raw() -> dict[str, list[str]]:
    return {
        "sample.a.top": ["sample.b.helper"],
        "sample.b.helper": ["sample.b.inner"],
        "sample.b.inner": [],
    }


def test_from_raw_builds_graphs() -> None:
    idx = CallGraphIndex.from_raw("/tmp/sample", _sample_raw())
    stats = idx.stats()
    assert stats["functions"] >= 3
    assert stats["function_edges"] == 2
    assert stats["modules"] >= 2


def test_queries() -> None:
    idx = CallGraphIndex.from_raw("/tmp/sample", _sample_raw())
    assert "sample.b.helper" in idx.callees_of("sample.a.top", depth=1)
    assert "sample.b.inner" in idx.callees_of("sample.a.top", depth=2)
    assert "sample.a.top" in idx.callers_of("sample.b.helper", depth=1)
    assert idx.search("helper") == ["sample.b.helper"]


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    idx = CallGraphIndex.from_raw("/tmp/sample", _sample_raw())
    out = tmp_path / "index.json"
    idx.save(out)

    loaded = CallGraphIndex.load(out)
    assert loaded.stats() == idx.stats()
    assert loaded.search("helper") == idx.search("helper")
