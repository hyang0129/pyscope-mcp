"""Integration test: depth-aware BFS ranking + dropped field through the analyzer.

Tests the issue #59 fix end-to-end: real Python source -> build_raw() ->
CallGraphIndex -> callers_of/callees_of.  Existing tests in test_graph.py cover
the ranking via from_raw() with hand-crafted dicts; this test exercises the
analyzer pipeline so that any analyzer-level edge-resolution bug would surface
as a failure here, not just at the graph layer.

Fixture topology (callers direction)
------------------------------------
  target.target_fn          <- called by 3 depth-1 callers in zdepth1.py
  zdepth1.z_caller_01..03   <- called by 55 depth-2 callers in adepth2.py

Total reachable from target_fn (reverse BFS, depth=2): 3 + 55 = 58
Cap = 50, so dropped = 8.  ``a_caller_*`` sorts alphabetically BEFORE
``z_caller_*``, so an unranked impl would keep 50 a_caller_* (depth-2) and
drop all three z_caller_* (depth-1) — the exact bug from issue #59.

Symmetric topology (callees direction)
--------------------------------------
  source.source_fn -> 3 depth-1 callees in zdepth1_callees.py
  zdepth1_callees.z_callee_01..03 -> 55 depth-2 callees in adepth2_callees.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyscope_mcp.analyzer import build_raw
from pyscope_mcp.graph import CallGraphIndex


_N_DEPTH1 = 3
_N_DEPTH2 = 55  # > (50 - _N_DEPTH1) so the cap fires and would drop depth-1 without ranking
_CAP = 50
_EXPECTED_DROPPED = (_N_DEPTH1 + _N_DEPTH2) - _CAP


def _make_callers_fixture(tmp_path: Path, pkg: str) -> Path:
    root = tmp_path / pkg
    root.mkdir()
    (root / "__init__.py").write_text("")

    (root / "target.py").write_text(
        "def target_fn():\n"
        "    pass\n"
    )

    zlines = [f"from {pkg}.target import target_fn\n\n"]
    for i in range(1, _N_DEPTH1 + 1):
        zlines.append(f"def z_caller_{i:02d}():\n    target_fn()\n\n")
    (root / "zdepth1.py").write_text("".join(zlines))

    alines = [f"from {pkg}.zdepth1 import " + ", ".join(
        f"z_caller_{i:02d}" for i in range(1, _N_DEPTH1 + 1)
    ) + "\n\n"]
    for i in range(1, _N_DEPTH2 + 1):
        # round-robin each a_caller across the depth-1 callers
        z_target = f"z_caller_{((i - 1) % _N_DEPTH1) + 1:02d}"
        alines.append(f"def a_caller_{i:02d}():\n    {z_target}()\n\n")
    (root / "adepth2.py").write_text("".join(alines))

    return tmp_path


def _make_callees_fixture(tmp_path: Path, pkg: str) -> Path:
    root = tmp_path / pkg
    root.mkdir()
    (root / "__init__.py").write_text("")

    alines = []
    for i in range(1, _N_DEPTH2 + 1):
        alines.append(f"def a_callee_{i:02d}():\n    pass\n\n")
    (root / "adepth2_callees.py").write_text("".join(alines))

    zlines = [f"from {pkg}.adepth2_callees import " + ", ".join(
        f"a_callee_{i:02d}" for i in range(1, _N_DEPTH2 + 1)
    ) + "\n\n"]
    for i in range(1, _N_DEPTH1 + 1):
        body = "\n".join(
            f"    a_callee_{j:02d}()"
            for j in range(1, _N_DEPTH2 + 1)
            if (j - 1) % _N_DEPTH1 == (i - 1)
        )
        zlines.append(f"def z_callee_{i:02d}():\n{body}\n\n")
    (root / "zdepth1_callees.py").write_text("".join(zlines))

    src = [f"from {pkg}.zdepth1_callees import " + ", ".join(
        f"z_callee_{i:02d}" for i in range(1, _N_DEPTH1 + 1)
    ) + "\n\n"]
    src.append("def source_fn():\n")
    for i in range(1, _N_DEPTH1 + 1):
        src.append(f"    z_callee_{i:02d}()\n")
    (root / "source.py").write_text("".join(src))

    return tmp_path


@pytest.fixture()
def callers_idx(tmp_path: Path) -> CallGraphIndex:
    pkg = "ranking_callers_fixture"
    root = _make_callers_fixture(tmp_path, pkg)
    raw = build_raw(root, pkg)
    return CallGraphIndex.from_raw(str(root), raw)


@pytest.fixture()
def callees_idx(tmp_path: Path) -> CallGraphIndex:
    pkg = "ranking_callees_fixture"
    root = _make_callees_fixture(tmp_path, pkg)
    raw = build_raw(root, pkg)
    return CallGraphIndex.from_raw(str(root), raw)


def test_analyzer_resolves_all_depth1_callers(callers_idx: CallGraphIndex) -> None:
    """Sanity: analyzer must produce edges from each depth-1 caller into target_fn.

    If the analyzer fails to resolve the cross-module call, the ranking test
    below would pass for the wrong reason (depth-1 callers absent because the
    edge never existed).
    """
    target = "ranking_callers_fixture.target.target_fn"
    direct_callers = {
        fqn
        for fqn, node in callers_idx.nodes.items()
        if target in node.get("calls", {}).get("call", [])
    }
    expected = {f"ranking_callers_fixture.zdepth1.z_caller_{i:02d}"
                for i in range(1, _N_DEPTH1 + 1)}
    missing = expected - direct_callers
    assert not missing, (
        f"Analyzer failed to resolve depth-1 caller edges into target_fn: "
        f"missing={missing}. Ranking test below would not be meaningful."
    )


def test_callers_of_preserves_depth1_under_cap_via_analyzer(
    callers_idx: CallGraphIndex,
) -> None:
    """Issue #59 regression: depth-1 callers must survive the 50-item cap
    even when 55 depth-2 callers sort alphabetically before them."""
    result = callers_idx.refers_to(
        "ranking_callers_fixture.target.target_fn", kind="callers", depth=2
    )
    fqns = [e["fqn"] for e in result["results"]]

    assert result["truncated"] is True
    assert len(result["results"]) == _CAP
    assert result["dropped"] == _EXPECTED_DROPPED

    for i in range(1, _N_DEPTH1 + 1):
        fqn = f"ranking_callers_fixture.zdepth1.z_caller_{i:02d}"
        assert fqn in fqns, (
            f"depth-1 caller {fqn} dropped by cap — ranking did not run "
            f"before truncation"
        )

    z_indices = [
        fqns.index(f"ranking_callers_fixture.zdepth1.z_caller_{i:02d}")
        for i in range(1, _N_DEPTH1 + 1)
    ]
    last_depth1 = max(z_indices)
    for item in fqns:
        if "adepth2.a_caller_" in item:
            assert fqns.index(item) > last_depth1, (
                f"depth-2 caller {item} ranked before a depth-1 caller"
            )


def test_callees_of_preserves_depth1_under_cap_via_analyzer(
    callees_idx: CallGraphIndex,
) -> None:
    """Symmetric regression test for callees_of through the analyzer."""
    result = callees_idx.callees_of(
        "ranking_callees_fixture.source.source_fn", depth=2
    )

    assert result["truncated"] is True
    assert len(result["results"]) == _CAP
    assert result["dropped"] == _EXPECTED_DROPPED

    for i in range(1, _N_DEPTH1 + 1):
        fqn = f"ranking_callees_fixture.zdepth1_callees.z_callee_{i:02d}"
        assert fqn in result["results"], (
            f"depth-1 callee {fqn} dropped by cap — ranking did not run "
            f"before truncation"
        )

    z_indices = [
        result["results"].index(
            f"ranking_callees_fixture.zdepth1_callees.z_callee_{i:02d}"
        )
        for i in range(1, _N_DEPTH1 + 1)
    ]
    last_depth1 = max(z_indices)
    for item in result["results"]:
        if "adepth2_callees.a_callee_" in item:
            assert result["results"].index(item) > last_depth1, (
                f"depth-2 callee {item} ranked before a depth-1 callee"
            )


def test_callers_of_dropped_zero_when_under_cap_via_analyzer(
    callers_idx: CallGraphIndex,
) -> None:
    """Querying a depth-1-only result set must yield dropped=0, not absent."""
    result = callers_idx.refers_to(
        "ranking_callers_fixture.target.target_fn", kind="callers", depth=1
    )
    assert result["truncated"] is False
    assert result["dropped"] == 0
    assert len(result["results"]) == _N_DEPTH1
