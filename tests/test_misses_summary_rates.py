"""Summary-level rate metrics emitted by MissLog.to_dict.

Covers: resolution_rate_in_package, unresolved_rate, effective_resolution.
effective_resolution treats external resolutions as successful and excludes
accepted-pattern calls from both numerator and denominator — it measures
resolver power, unaffected by whitelist widening.
"""

from __future__ import annotations

from pyscope_mcp.analyzer.misses import MissLog


def _make_log(
    in_pkg: int = 0, external: int = 0, unresolved: int = 0, accepted: int = 0
) -> MissLog:
    log = MissLog()
    for _ in range(in_pkg):
        log.record_resolved(in_package=True)
    for _ in range(external):
        log.record_resolved(in_package=False)
    for _ in range(unresolved):
        log.record_miss(
            pattern="attr_chain_unresolved",
            caller="m.f",
            file_path="/t.py",
            line=1,
            snippet="x.y()",
        )
    for _ in range(accepted):
        log.record_accepted("builtin_method_call", "/t.py")
    return log


def _summary(log: MissLog) -> dict:
    return log.to_dict()["summary"]


def test_empty_log_emits_zero_rates() -> None:
    s = _summary(MissLog())
    assert s["resolution_rate_in_package"] == 0.0
    assert s["unresolved_rate"] == 0.0
    assert s["effective_resolution"] == 0.0


def test_resolution_rate_in_package_includes_accepted_in_denominator() -> None:
    log = _make_log(in_pkg=10, external=20, unresolved=30, accepted=40)
    s = _summary(log)
    assert s["calls_total"] == 100
    assert s["resolution_rate_in_package"] == 0.1


def test_unresolved_rate_is_unresolved_over_total() -> None:
    log = _make_log(in_pkg=10, external=20, unresolved=30, accepted=40)
    assert _summary(log)["unresolved_rate"] == 0.3


def test_effective_resolution_excludes_accepted() -> None:
    log = _make_log(in_pkg=10, external=20, unresolved=30, accepted=40)
    # (10 + 20) / (10 + 20 + 30) = 30 / 60 = 0.5
    assert _summary(log)["effective_resolution"] == 0.5


def test_effective_resolution_whitelist_widening_is_invisible() -> None:
    """Adding 1000 accepted calls must not change effective_resolution."""
    before = _summary(_make_log(in_pkg=10, external=20, unresolved=30, accepted=0))
    after = _summary(_make_log(in_pkg=10, external=20, unresolved=30, accepted=1000))
    assert before["effective_resolution"] == after["effective_resolution"] == 0.5
    # Raw in-package rate, in contrast, collapses from 0.2 → ~0.01.
    assert before["resolution_rate_in_package"] > after["resolution_rate_in_package"]


def test_effective_resolution_all_resolved_is_one() -> None:
    log = _make_log(in_pkg=5, external=5, unresolved=0, accepted=0)
    assert _summary(log)["effective_resolution"] == 1.0


def test_effective_resolution_all_unresolved_is_zero() -> None:
    log = _make_log(in_pkg=0, external=0, unresolved=10, accepted=0)
    assert _summary(log)["effective_resolution"] == 0.0
