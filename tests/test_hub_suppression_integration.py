"""Integration test: hub suppression fires end-to-end through the analyzer.

Tests the full pipeline: real Python source → build_raw() → CallGraphIndex →
neighborhood().  All existing hub-suppression tests use from_raw() with
hand-crafted dicts; this test exercises the analyzer so that edge-counting bugs
(e.g. self_method_unresolved undercount) would surface as failures here.

Fixture topology
----------------
  utils.shared_helper    <- called by entry_point + 15 callers in callers.py (16 total)
  app.entry_point        -> shared_helper, private_step
  app.private_step       (no callees)
  callers.caller_01..15  -> shared_helper

With default suppression on and 16 callers >> floor-10 threshold:
  neighborhood(entry_point, depth=2) must suppress shared_helper and hide the
  15 sibling callers.

With expand_hubs=True the 15 sibling callers must reappear.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyscope_mcp.analyzer import build_raw
from pyscope_mcp.graph import CallGraphIndex


_N_EXTRA_CALLERS = 15  # plus entry_point = 16 total callers of shared_helper


def _make_fixture_package(tmp_path: Path, pkg: str) -> Path:
    root = tmp_path / pkg
    root.mkdir()
    (root / "__init__.py").write_text("")

    (root / "utils.py").write_text(
        "def shared_helper():\n"
        "    pass\n"
    )

    callers_lines = [f"from {pkg}.utils import shared_helper\n\n"]
    for i in range(1, _N_EXTRA_CALLERS + 1):
        callers_lines.append(f"def caller_{i:02d}():\n    shared_helper()\n\n")
    (root / "callers.py").write_text("".join(callers_lines))

    (root / "app.py").write_text(
        f"from {pkg}.utils import shared_helper\n"
        "\n"
        "def private_step():\n"
        "    pass\n"
        "\n"
        "def entry_point():\n"
        "    shared_helper()\n"
        "    private_step()\n"
    )
    return tmp_path


@pytest.fixture()
def hub_idx(tmp_path: Path) -> CallGraphIndex:
    pkg = "hub_fixture"
    root = _make_fixture_package(tmp_path, pkg)
    raw = build_raw(root, pkg)
    return CallGraphIndex.from_raw(str(root), raw)


def test_analyzer_resolves_hub_callers(hub_idx: CallGraphIndex) -> None:
    """Analyzer must produce >= N_EXTRA_CALLERS+1 in-edges for shared_helper."""
    in_degree = sum(
        1
        for callees in hub_idx.raw.values()
        if "hub_fixture.utils.shared_helper" in callees
    )
    assert in_degree >= _N_EXTRA_CALLERS + 1, (
        f"Expected >= {_N_EXTRA_CALLERS + 1} callers of shared_helper, "
        f"analyzer produced {in_degree}. "
        f"Hub suppression will never fire if the analyzer undercounts edges."
    )


def test_hub_suppression_fires_on_real_graph(hub_idx: CallGraphIndex) -> None:
    """Hub suppression fires end-to-end: shared_helper lands in hub_suppressed."""
    result = hub_idx.neighborhood(
        "hub_fixture.app.entry_point", depth=2, token_budget=100_000
    )
    assert "hub_fixture.utils.shared_helper" in result["hub_suppressed"], (
        f"shared_helper must be in hub_suppressed (in-degree={_N_EXTRA_CALLERS + 1} "
        f"> threshold={result['hub_threshold']}). "
        f"hub_suppressed={result['hub_suppressed']}"
    )


def test_sibling_callers_hidden_by_default(hub_idx: CallGraphIndex) -> None:
    """With suppression on, sibling callers of the hub must not appear in edges."""
    result = hub_idx.neighborhood(
        "hub_fixture.app.entry_point", depth=2, token_budget=100_000
    )
    edge_nodes = {n for edge in result["edges"] for n in edge}
    sibling_callers = {f"hub_fixture.callers.caller_{i:02d}" for i in range(1, _N_EXTRA_CALLERS + 1)}
    leaked = sibling_callers & edge_nodes
    assert not leaked, (
        f"Hub suppression should hide sibling callers of shared_helper; "
        f"leaked into result: {leaked}"
    )


def test_own_callees_still_present(hub_idx: CallGraphIndex) -> None:
    """Suppression must not drop entry_point's own callees (private_step)."""
    result = hub_idx.neighborhood(
        "hub_fixture.app.entry_point", depth=2, token_budget=100_000
    )
    edge_nodes = {n for edge in result["edges"] for n in edge}
    assert "hub_fixture.app.private_step" in edge_nodes, (
        "private_step (non-hub callee of entry_point) must still appear in neighborhood"
    )


def test_expand_hubs_restores_sibling_callers(hub_idx: CallGraphIndex) -> None:
    """expand_hubs=True must restore sibling callers that suppression hides."""
    result = hub_idx.neighborhood(
        "hub_fixture.app.entry_point", depth=2, token_budget=100_000, expand_hubs=True
    )
    edge_nodes = {n for edge in result["edges"] for n in edge}
    sibling_callers = {f"hub_fixture.callers.caller_{i:02d}" for i in range(1, _N_EXTRA_CALLERS + 1)}
    missing = sibling_callers - edge_nodes
    assert not missing, (
        f"expand_hubs=True should restore all sibling callers; "
        f"still missing: {missing}"
    )
    assert result["hub_suppressed"] == [], (
        "hub_suppressed must be [] when expand_hubs=True"
    )
