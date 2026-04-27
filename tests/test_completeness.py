"""Tests for the completeness field on all edge-traversing tools.

Covers:
  - completeness_for() helper directly
  - callers_of completeness
  - callees_of completeness
  - neighborhood completeness (isolated node + main path)
  - module_callers completeness (via symbol expansion)
  - module_callees completeness
  - v4 save/load round-trip for missed_callers
  - v3 (and older) index rejection

Acceptance scenarios from the refined spec (#60):
  (a) "complete" when missed_callers={}
  (b) "partial" on direct hit
  (c) "partial" on class-prefix hit (method whose class sibling has misses)
  (d) "complete" for top-level function whose module sibling has misses (negative case)
  (e) all five tools route through completeness_for()
  (f) completeness_for() exercised directly
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyscope_mcp.graph import CallGraphIndex, SymbolSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym(fqn: str, kind: str = "function", lineno: int = 1) -> SymbolSummary:
    return SymbolSummary(
        fqn=fqn,
        kind=kind,
        signature=f"def {fqn.split('.')[-1]}(): ...",
        lineno=lineno,
    )


def _make_idx(
    raw: dict[str, list[str]],
    missed_callers: dict[str, dict[str, int]] | None = None,
    skeletons: dict[str, list[SymbolSummary]] | None = None,
    file_shas: dict[str, str] | None = None,
    git_sha: str | None = None,
) -> CallGraphIndex:
    return CallGraphIndex.from_raw(
        "/tmp/test",
        raw,
        skeletons=skeletons or {},
        file_shas=file_shas or {},
        missed_callers=missed_callers or {},
        git_sha=git_sha,
    )


# ---------------------------------------------------------------------------
# 1. completeness_for() helper — direct unit tests (scenario f)
# ---------------------------------------------------------------------------

class TestCompletenessFor:
    def test_empty_missed_callers_returns_complete(self) -> None:
        """(a) No missed_callers → always 'complete'."""
        idx = _make_idx({}, missed_callers={})
        assert idx.completeness_for([]) == "complete"
        assert idx.completeness_for(["pkg.mod.func"]) == "complete"
        assert idx.completeness_for(["pkg.mod.MyClass.method"]) == "complete"

    def test_direct_hit_returns_partial(self) -> None:
        """(b) FQN directly in missed_callers → 'partial'."""
        missed = {"pkg.mod.MyClass.foo": {"getattr_nonliteral": 3}}
        idx = _make_idx({}, missed_callers=missed)
        assert idx.completeness_for(["pkg.mod.MyClass.foo"]) == "partial"

    def test_class_prefix_hit_returns_partial(self) -> None:
        """(c) Method whose class sibling is in missed_callers → 'partial'."""
        missed = {"pkg.mod.MyClass.foo": {"getattr_nonliteral": 3}}
        idx = _make_idx({}, missed_callers=missed)
        # pkg.mod.MyClass.bar is not directly missed, but shares class pkg.mod.MyClass
        assert idx.completeness_for(["pkg.mod.MyClass.bar"]) == "partial"

    def test_top_level_function_module_sibling_returns_complete(self) -> None:
        """(d) Top-level function whose module sibling has misses → 'complete'.

        Top-level functions (≤2 dotted segments after module part) NEVER trip the
        class-prefix path — only direct hits flip them.
        """
        missed = {"pkg.mod.dirty_func": {"bare_name_unresolved": 1}}
        idx = _make_idx({}, missed_callers=missed)
        # pkg.mod.clean_func is top-level, same module — must NOT be flagged partial
        assert idx.completeness_for(["pkg.mod.clean_func"]) == "complete"

    def test_method_class_prefix_negative_no_sibling(self) -> None:
        """Method with class prefix that doesn't match any missed key → 'complete'."""
        missed = {"pkg.mod.OtherClass.method": {"getattr_nonliteral": 1}}
        idx = _make_idx({}, missed_callers=missed)
        # pkg.mod.MyClass.bar has class prefix pkg.mod.MyClass — no key starts with that
        assert idx.completeness_for(["pkg.mod.MyClass.bar"]) == "complete"

    def test_empty_fqn_list_returns_complete(self) -> None:
        """Empty FQN list → 'complete' regardless of missed_callers."""
        missed = {"pkg.mod.MyClass.foo": {"getattr_nonliteral": 3}}
        idx = _make_idx({}, missed_callers=missed)
        assert idx.completeness_for([]) == "complete"

    def test_multiple_fqns_one_hit_returns_partial(self) -> None:
        """Mixed list: one direct hit → 'partial'."""
        missed = {"pkg.mod.MyClass.foo": {"getattr_nonliteral": 3}}
        idx = _make_idx({}, missed_callers=missed)
        assert idx.completeness_for(["pkg.other.clean", "pkg.mod.MyClass.foo"]) == "partial"


# ---------------------------------------------------------------------------
# 2. _class_prefix() helper
# ---------------------------------------------------------------------------

class TestClassPrefix:
    def test_method_returns_class_prefix(self) -> None:
        # 4 segments: pkg.mod.MyClass.method → pkg.mod.MyClass
        assert CallGraphIndex._class_prefix("a.b.C.method") == "a.b.C"

    def test_top_level_two_segments_returns_none(self) -> None:
        assert CallGraphIndex._class_prefix("pkg.func") is None

    def test_top_level_one_segment_returns_none(self) -> None:
        assert CallGraphIndex._class_prefix("func") is None

    def test_top_level_three_segments_returns_none(self) -> None:
        # pkg.mod.func has 3 segments — top-level function, not a method
        assert CallGraphIndex._class_prefix("pkg.mod.func") is None

    def test_nested_class_method(self) -> None:
        # a.b.c.D.method → prefix is a.b.c.D (5 segments)
        assert CallGraphIndex._class_prefix("a.b.c.D.method") == "a.b.c.D"

    def test_exactly_four_segments(self) -> None:
        # pkg.mod.Cls.method → prefix is pkg.mod.Cls (4 segments = minimum method shape)
        assert CallGraphIndex._class_prefix("pkg.mod.Cls.method") == "pkg.mod.Cls"


# ---------------------------------------------------------------------------
# 3. callers_of completeness (scenario e + acceptance scenario for callers_of)
# ---------------------------------------------------------------------------

class TestCallersOfCompleteness:
    _RAW = {
        "pkg.mod.MyClass.foo": ["pkg.mod.helper"],
        "pkg.mod.bar": ["pkg.mod.helper"],
        "pkg.mod.helper": [],
    }

    def test_complete_when_no_misses(self) -> None:
        idx = _make_idx(self._RAW, missed_callers={})
        result = idx.callers_of("pkg.mod.helper")
        assert result["completeness"] == "complete"

    def test_partial_when_result_has_direct_miss(self) -> None:
        missed = {"pkg.mod.MyClass.foo": {"getattr_nonliteral": 3}}
        idx = _make_idx(self._RAW, missed_callers=missed)
        result = idx.callers_of("pkg.mod.helper")
        # pkg.mod.MyClass.foo is in results and directly in missed_callers
        assert "pkg.mod.MyClass.foo" in result["results"]
        assert result["completeness"] == "partial"

    def test_partial_when_result_has_class_sibling_miss(self) -> None:
        # pkg.mod.MyClass.bar is NOT in results but pkg.mod.MyClass.foo IS
        # and they share the class prefix pkg.mod.MyClass
        raw = {
            "pkg.mod.MyClass.bar": ["pkg.mod.helper"],
            "pkg.mod.helper": [],
        }
        missed = {"pkg.mod.MyClass.foo": {"getattr_nonliteral": 3}}
        idx = _make_idx(raw, missed_callers=missed)
        result = idx.callers_of("pkg.mod.helper")
        assert "pkg.mod.MyClass.bar" in result["results"]
        assert result["completeness"] == "partial"

    def test_complete_for_top_level_sibling_miss(self) -> None:
        # Only pkg.mod.bar in results — a top-level func
        # dirty_func is top-level sibling in same module — must not flag bar
        raw = {
            "pkg.mod.bar": ["pkg.mod.helper"],
            "pkg.mod.helper": [],
        }
        missed = {"pkg.mod.dirty_func": {"bare_name_unresolved": 1}}
        idx = _make_idx(raw, missed_callers=missed)
        result = idx.callers_of("pkg.mod.helper")
        assert result["completeness"] == "complete"

    def test_completeness_field_always_present(self) -> None:
        idx = _make_idx(self._RAW, missed_callers={})
        result = idx.callers_of("pkg.mod.helper")
        assert "completeness" in result


# ---------------------------------------------------------------------------
# 4. callees_of completeness
# ---------------------------------------------------------------------------

class TestCalleesOfCompleteness:
    _RAW = {
        "pkg.mod.MyClass.foo": ["pkg.mod.helper", "pkg.mod.other"],
        "pkg.mod.helper": [],
        "pkg.mod.other": [],
    }

    def test_complete_when_no_misses(self) -> None:
        idx = _make_idx(self._RAW, missed_callers={})
        result = idx.callees_of("pkg.mod.MyClass.foo")
        assert result["completeness"] == "complete"

    def test_partial_when_result_has_direct_miss(self) -> None:
        # helper itself has misses
        missed = {"pkg.mod.helper": {"getattr_nonliteral": 1}}
        idx = _make_idx(self._RAW, missed_callers=missed)
        result = idx.callees_of("pkg.mod.MyClass.foo")
        assert "pkg.mod.helper" in result["results"]
        assert result["completeness"] == "partial"

    def test_completeness_field_always_present(self) -> None:
        idx = _make_idx(self._RAW, missed_callers={})
        result = idx.callees_of("pkg.mod.MyClass.foo")
        assert "completeness" in result


# ---------------------------------------------------------------------------
# 5. neighborhood completeness
# ---------------------------------------------------------------------------

class TestNeighborhoodCompleteness:
    _RAW = {
        "pkg.mod.MyClass.foo": ["pkg.mod.helper"],
        "pkg.mod.helper": ["pkg.mod.leaf"],
        "pkg.mod.leaf": [],
    }

    def test_complete_when_no_misses(self) -> None:
        idx = _make_idx(self._RAW, missed_callers={})
        result = idx.neighborhood("pkg.mod.helper")
        assert result["completeness"] == "complete"

    def test_partial_when_neighborhood_fqn_directly_missed(self) -> None:
        missed = {"pkg.mod.MyClass.foo": {"getattr_nonliteral": 2}}
        idx = _make_idx(self._RAW, missed_callers=missed)
        result = idx.neighborhood("pkg.mod.helper", depth=2)
        assert result["completeness"] == "partial"

    def test_isolated_node_completeness_present(self) -> None:
        """Symbol not in graph returns isError:true, error_reason:'fqn_not_in_graph'."""
        idx = _make_idx({}, missed_callers={})
        result = idx.neighborhood("nonexistent.symbol")
        # Post-fix: not-in-graph returns an error dict, not a normal neighborhood result
        assert result["isError"] is True
        assert result["error_reason"] == "fqn_not_in_graph"
        assert result["stale"] is False

    def test_isolated_node_partial_when_directly_missed(self) -> None:
        """Symbol not in graph returns error dict even if it appears in missed_callers."""
        missed = {"nonexistent.symbol": {"bare_name_unresolved": 1}}
        idx = _make_idx({}, missed_callers=missed)
        result = idx.neighborhood("nonexistent.symbol")
        # Post-fix: not-in-graph returns an error dict; missed_callers does not override this
        assert result["isError"] is True
        assert result["error_reason"] == "fqn_not_in_graph"

    def test_completeness_field_always_present(self) -> None:
        idx = _make_idx(self._RAW, missed_callers={})
        result = idx.neighborhood("pkg.mod.helper")
        assert "completeness" in result


# ---------------------------------------------------------------------------
# 6. module_callers / module_callees completeness
# ---------------------------------------------------------------------------

class TestModuleCompleteness:
    # Module graph: pkg.other → pkg.target (pkg.other.SomeClass.method calls pkg.target.helper)
    _RAW = {
        "pkg.other.SomeClass.method": ["pkg.target.helper"],
        "pkg.target.helper": [],
    }
    _SKELETONS: dict[str, list[SymbolSummary]] = {
        "pkg/other.py": [
            _sym("pkg.other.SomeClass.method", kind="method"),
        ],
        "pkg/target.py": [
            _sym("pkg.target.helper"),
        ],
    }

    def test_module_callers_complete_when_no_misses(self) -> None:
        idx = _make_idx(self._RAW, missed_callers={}, skeletons=self._SKELETONS)
        result = idx.module_callers("pkg.target")
        assert result["completeness"] == "complete"

    def test_module_callers_partial_when_symbol_in_result_module_is_missed(self) -> None:
        # pkg.other.SomeClass.method has unresolved calls;
        # _module_of("pkg.other.SomeClass.method") == "pkg.other.SomeClass", so that
        # is what appears in the module graph and in the module_callers result.
        missed = {"pkg.other.SomeClass.method": {"getattr_nonliteral": 2}}
        idx = _make_idx(self._RAW, missed_callers=missed, skeletons=self._SKELETONS)
        result = idx.module_callers("pkg.target")
        # The module graph reports the caller as "pkg.other.SomeClass"
        assert "pkg.other.SomeClass" in result["results"]
        assert result["completeness"] == "partial"

    def test_module_callers_partial_via_class_prefix_expansion(self) -> None:
        # pkg.other.SomeClass.other_method is in the same class as the missed FQN.
        # Expanding pkg.other reveals both. Class-prefix match fires.
        missed = {"pkg.other.SomeClass.foo": {"getattr_nonliteral": 1}}
        skeletons: dict[str, list[SymbolSummary]] = {
            "pkg/other.py": [
                _sym("pkg.other.SomeClass.method", kind="method"),
                _sym("pkg.other.SomeClass.bar", kind="method"),
            ],
            "pkg/target.py": [_sym("pkg.target.helper")],
        }
        idx = _make_idx(self._RAW, missed_callers=missed, skeletons=skeletons)
        result = idx.module_callers("pkg.target")
        assert result["completeness"] == "partial"

    def test_module_callees_complete_when_no_misses(self) -> None:
        idx = _make_idx(self._RAW, missed_callers={}, skeletons=self._SKELETONS)
        result = idx.module_callees("pkg.other")
        assert result["completeness"] == "complete"

    def test_module_callees_partial_when_callee_module_symbol_missed(self) -> None:
        # pkg.target.helper has misses; pkg.target is in result of module_callees("pkg.other")
        missed = {"pkg.target.helper": {"bare_name_unresolved": 1}}
        idx = _make_idx(self._RAW, missed_callers=missed, skeletons=self._SKELETONS)
        result = idx.module_callees("pkg.other")
        assert "pkg.target" in result["results"]
        assert result["completeness"] == "partial"

    def test_module_completeness_field_always_present(self) -> None:
        idx = _make_idx(self._RAW, missed_callers={}, skeletons=self._SKELETONS)
        result = idx.module_callers("pkg.target")
        assert "completeness" in result
        result2 = idx.module_callees("pkg.other")
        assert "completeness" in result2


# ---------------------------------------------------------------------------
# 7. save/load v5 round-trip + old-version rejection
# ---------------------------------------------------------------------------

class TestV5Schema:
    def test_save_writes_current_version(self, tmp_path: Path) -> None:
        from pyscope_mcp.graph import INDEX_VERSION
        idx = _make_idx(
            {"pkg.mod.func": ["pkg.mod.other"]},
            missed_callers={"pkg.mod.func": {"getattr_nonliteral": 2}},
        )
        out = tmp_path / "index.json"
        idx.save(out)
        payload = json.loads(out.read_text())
        assert payload["version"] == INDEX_VERSION

    def test_save_writes_missed_callers(self, tmp_path: Path) -> None:
        missed = {"pkg.mod.func": {"getattr_nonliteral": 2, "bare_name_unresolved": 1}}
        idx = _make_idx({}, missed_callers=missed)
        out = tmp_path / "index.json"
        idx.save(out)
        payload = json.loads(out.read_text())
        assert "missed_callers" in payload
        assert payload["missed_callers"] == missed

    def test_save_writes_git_sha_and_content_hash(self, tmp_path: Path) -> None:
        """v5: git_sha and content_hash are present in the saved payload."""
        idx = _make_idx({"pkg.mod.func": []}, git_sha="deadbeef" * 5)
        out = tmp_path / "index.json"
        idx.save(out)
        payload = json.loads(out.read_text())
        assert payload["git_sha"] == "deadbeef" * 5
        assert isinstance(payload["content_hash"], str) and len(payload["content_hash"]) == 64

    def test_load_populates_missed_callers(self, tmp_path: Path) -> None:
        missed = {"pkg.mod.MyClass.foo": {"getattr_nonliteral": 3}}
        idx = _make_idx({}, missed_callers=missed)
        out = tmp_path / "index.json"
        idx.save(out)
        loaded = CallGraphIndex.load(out)
        assert loaded.missed_callers == missed

    def test_load_completeness_for_works_after_roundtrip(self, tmp_path: Path) -> None:
        missed = {"pkg.mod.MyClass.foo": {"getattr_nonliteral": 3}}
        idx = _make_idx({}, missed_callers=missed)
        out = tmp_path / "index.json"
        idx.save(out)
        loaded = CallGraphIndex.load(out)
        assert loaded.completeness_for(["pkg.mod.MyClass.bar"]) == "partial"
        assert loaded.completeness_for(["pkg.other.func"]) == "complete"

    def test_load_empty_missed_callers(self, tmp_path: Path) -> None:
        idx = _make_idx({}, missed_callers={})
        out = tmp_path / "index.json"
        idx.save(out)
        loaded = CallGraphIndex.load(out)
        assert loaded.missed_callers == {}
        assert loaded.completeness_for(["anything"]) == "complete"

    @pytest.mark.parametrize("old_version", [1, 2, 3, 4])
    def test_load_rejects_old_versions(self, tmp_path: Path, old_version: int) -> None:
        from pyscope_mcp.graph import INDEX_VERSION
        payload = {
            "version": old_version,
            "root": "/tmp/test",
            "raw": {},
            "skeletons": {},
            "file_shas": {},
        }
        out = tmp_path / "index.json"
        out.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match=f"v{old_version}"):
            CallGraphIndex.load(out)

    def test_load_rejects_unknown_version(self, tmp_path: Path) -> None:
        payload = {"version": 99, "root": "/tmp", "raw": {}, "skeletons": {}, "file_shas": {}}
        out = tmp_path / "index.json"
        out.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="v99"):
            CallGraphIndex.load(out)
