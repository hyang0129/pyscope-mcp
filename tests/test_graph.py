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
    result = idx.search("helper")
    assert result["results"] == ["sample.b.helper"]
    assert result["truncated"] is False
    assert result["total_matched"] == 1


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    idx = CallGraphIndex.from_raw("/tmp/sample", _sample_raw())
    out = tmp_path / "index.json"
    idx.save(out)

    loaded = CallGraphIndex.load(out)
    assert loaded.stats() == idx.stats()
    assert loaded.search("helper") == idx.search("helper")


def test_search_truncation() -> None:
    """When matches exceed the cap, truncated=True and total_matched reflects the full count."""
    # Build a raw graph with 5 symbols all containing "fn"
    raw = {f"pkg.mod.fn{i}": [] for i in range(5)}
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)

    # Cap at 3 — should truncate
    result = idx.search("fn", limit=3)
    assert len(result["results"]) == 3
    assert result["truncated"] is True
    assert result["total_matched"] == 5

    # Cap at 10 — should not truncate
    result_all = idx.search("fn", limit=10)
    assert len(result_all["results"]) == 5
    assert result_all["truncated"] is False
    assert result_all["total_matched"] == 5
