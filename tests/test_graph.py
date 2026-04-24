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


def _prefix_raw() -> dict[str, list[str]]:
    """Raw data for prefix-match tests.

    Module graph derived from this:
      pkg.agents.writer  →  pkg.io.disk
      pkg.agents.reader  →  pkg.io.network
      pkg.core           →  pkg.agents.writer, pkg.agents.reader
    """
    return {
        "pkg.agents.writer.write": ["pkg.io.disk.save"],
        "pkg.agents.reader.read": ["pkg.io.network.fetch"],
        "pkg.core.run": ["pkg.agents.writer.write", "pkg.agents.reader.read"],
        "pkg.io.disk.save": [],
        "pkg.io.network.fetch": [],
    }


def test_module_callers_prefix_match() -> None:
    """module_callers with a prefix returns callers of all matched modules."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callers("pkg.agents")
    assert result["truncated"] is False
    # pkg.core calls into pkg.agents.writer and pkg.agents.reader
    assert "pkg.core" in result["results"]
    # The matched seeds themselves must NOT appear in results
    assert "pkg.agents.writer" not in result["results"]
    assert "pkg.agents.reader" not in result["results"]


def test_module_callees_prefix_match() -> None:
    """module_callees with a prefix returns callees of all matched modules."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callees("pkg.agents")
    assert result["truncated"] is False
    # pkg.agents.writer calls pkg.io.disk; pkg.agents.reader calls pkg.io.network
    assert "pkg.io.disk" in result["results"]
    assert "pkg.io.network" in result["results"]


def test_module_callers_exact_match_backward_compat() -> None:
    """An exact FQN returns the same callers as before, now in structured form."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callers("pkg.agents.writer")
    assert isinstance(result, dict)
    assert "results" in result
    assert "truncated" in result
    assert result["truncated"] is False
    assert "pkg.core" in result["results"]


def test_module_callers_no_match() -> None:
    """A prefix matching no module nodes returns empty results, no error."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callers("nonexistent.pkg")
    assert result == {"results": [], "truncated": False}


def test_module_callees_no_match() -> None:
    """A prefix matching no module nodes returns empty results, no error."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callees("nonexistent.pkg")
    assert result == {"results": [], "truncated": False}


def test_module_callers_empty_prefix() -> None:
    """Empty-string prefix matches all modules; returns structured dict with no error."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callers("")
    # All modules are seeds — callers within the same set are excluded.
    # The important invariants are shape and no error.
    assert isinstance(result, dict)
    assert "results" in result
    assert "truncated" in result
    assert isinstance(result["results"], list)
    assert isinstance(result["truncated"], bool)


def test_module_callees_empty_prefix() -> None:
    """Empty-string prefix matches all modules; returns structured dict with no error."""
    idx = CallGraphIndex.from_raw("/tmp/sample", _prefix_raw())
    result = idx.module_callees("")
    assert isinstance(result, dict)
    assert "results" in result
    assert "truncated" in result
    assert isinstance(result["results"], list)
    assert isinstance(result["truncated"], bool)


def test_module_callers_truncation() -> None:
    """When more than 50 distinct callers exist, truncated=True and len==50."""
    # Build a graph where one "target" module has 60 distinct callers
    raw: dict[str, list[str]] = {}
    for i in range(60):
        raw[f"callers.mod{i}.fn"] = [f"target.core.fn"]
    raw["target.core.fn"] = []
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)
    result = idx.module_callers("target")
    assert result["truncated"] is True
    assert len(result["results"]) == 50


def test_module_callees_truncation() -> None:
    """When more than 50 distinct callees exist, truncated=True and len==50."""
    raw: dict[str, list[str]] = {
        "source.mod.fn": [f"target.mod{i}.fn" for i in range(60)]
    }
    for i in range(60):
        raw[f"target.mod{i}.fn"] = []
    idx = CallGraphIndex.from_raw("/tmp/sample", raw)
    result = idx.module_callees("source")
    assert result["truncated"] is True
    assert len(result["results"]) == 50


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

    # Cap exactly equal to match count — truncated must be False (condition is total > limit, not >=)
    result_exact = idx.search("fn", limit=5)
    assert len(result_exact["results"]) == 5
    assert result_exact["truncated"] is False
    assert result_exact["total_matched"] == 5
