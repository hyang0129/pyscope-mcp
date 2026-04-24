"""Unit tests for staleness helpers and per-tool stale/clean paths.

Covers:
  - _fqn_to_file population after from_raw()
  - _staleness_for() helper: dirty-in-result, dirty-not-in-result, pre-v3
  - callers_of / callees_of stale and clean paths
  - search stale and clean paths
  - neighborhood stale and clean paths
  - module_callers / module_callees stale and clean paths
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pyscope_mcp.graph import CallGraphIndex, SymbolSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sym(fqn: str, kind: str = "function", lineno: int = 1) -> SymbolSummary:
    return SymbolSummary(fqn=fqn, kind=kind, signature=f"def {fqn.split('.')[-1]}(): ...", lineno=lineno)


def _make_idx(
    tmp_path: Path,
    raw: dict[str, list[str]],
    skeletons: dict[str, list[SymbolSummary]],
    file_shas: dict[str, str] | None,
) -> CallGraphIndex:
    return CallGraphIndex.from_raw(str(tmp_path), raw, skeletons=skeletons, file_shas=file_shas)


# ---------------------------------------------------------------------------
# 1. _fqn_to_file population
# ---------------------------------------------------------------------------

def test_fqn_to_file_populated(tmp_path: Path) -> None:
    """_fqn_to_file is built by inverting skeletons at from_raw() time."""
    skeletons = {
        "src/pkg/mod.py": [_sym("pkg.mod.fn_a"), _sym("pkg.mod.fn_b")],
        "src/pkg/other.py": [_sym("pkg.other.fn_c")],
    }
    idx = CallGraphIndex.from_raw(str(tmp_path), {}, skeletons=skeletons, file_shas={})
    assert idx._fqn_to_file["pkg.mod.fn_a"] == "src/pkg/mod.py"
    assert idx._fqn_to_file["pkg.mod.fn_b"] == "src/pkg/mod.py"
    assert idx._fqn_to_file["pkg.other.fn_c"] == "src/pkg/other.py"


def test_fqn_to_file_empty_skeletons(tmp_path: Path) -> None:
    """Empty skeletons produce an empty _fqn_to_file."""
    idx = CallGraphIndex.from_raw(str(tmp_path), {}, skeletons={}, file_shas={})
    assert idx._fqn_to_file == {}


# ---------------------------------------------------------------------------
# 2. _staleness_for() helper
# ---------------------------------------------------------------------------

def test_staleness_for_clean(tmp_path: Path) -> None:
    """Clean file backing a result FQN → stale: false, stale_files: []."""
    content = b"def fn_a(): pass\n"
    (tmp_path / "mod.py").write_bytes(content)
    skeletons = {"mod.py": [_sym("pkg.mod.fn_a")]}
    shas = {"mod.py": _sha256(content)}
    idx = _make_idx(tmp_path, {}, skeletons, shas)

    result = idx._staleness_for(["pkg.mod.fn_a"])
    assert result["stale"] is False
    assert result["stale_files"] == []
    assert "stale_action" not in result


def test_staleness_for_dirty_file_in_result(tmp_path: Path) -> None:
    """Dirty file that backs a result FQN appears in stale_files."""
    (tmp_path / "mod.py").write_bytes(b"def fn_a(): return 42\n")
    skeletons = {"mod.py": [_sym("pkg.mod.fn_a")]}
    shas = {"mod.py": _sha256(b"def fn_a(): pass\n")}  # stored hash differs
    idx = _make_idx(tmp_path, {}, skeletons, shas)

    result = idx._staleness_for(["pkg.mod.fn_a"])
    assert result["stale"] is True
    assert "mod.py" in result["stale_files"]
    assert "build" in result["stale_action"].lower()


def test_staleness_for_dirty_file_not_in_result(tmp_path: Path) -> None:
    """Dirty file that backs NO result FQN must NOT appear in stale_files."""
    content_a = b"def fn_a(): pass\n"
    (tmp_path / "mod_a.py").write_bytes(content_a)
    (tmp_path / "mod_b.py").write_bytes(b"def fn_b(): return 99\n")  # dirty
    skeletons = {
        "mod_a.py": [_sym("pkg.mod_a.fn_a")],
        "mod_b.py": [_sym("pkg.mod_b.fn_b")],
    }
    shas = {
        "mod_a.py": _sha256(content_a),
        "mod_b.py": _sha256(b"def fn_b(): pass\n"),  # stored hash differs from disk
    }
    idx = _make_idx(tmp_path, {}, skeletons, shas)

    # Only asking about fn_a — mod_b.py is dirty but not in the result set
    result = idx._staleness_for(["pkg.mod_a.fn_a"])
    assert result["stale"] is False
    assert result["stale_files"] == []


def test_staleness_for_pre_v3_index(tmp_path: Path) -> None:
    """Pre-v3 index (file_shas=None) → short-circuit to index_format_incompatible."""
    skeletons = {"mod.py": [_sym("pkg.mod.fn_a")]}
    idx = _make_idx(tmp_path, {}, skeletons, file_shas=None)

    result = idx._staleness_for(["pkg.mod.fn_a"])
    assert result["stale"] is True
    assert result["stale_files"] == []
    assert result["index_stale_reason"] == "index_format_incompatible"


def test_staleness_for_fqn_not_in_skeletons(tmp_path: Path) -> None:
    """FQN with no skeleton entry (not in _fqn_to_file) yields clean result."""
    idx = _make_idx(tmp_path, {}, skeletons={}, file_shas={})
    result = idx._staleness_for(["pkg.mod.fn_unknown"])
    assert result["stale"] is False
    assert result["stale_files"] == []


# ---------------------------------------------------------------------------
# 3. callers_of / callees_of
# ---------------------------------------------------------------------------

def _raw_ab() -> dict[str, list[str]]:
    return {
        "pkg.mod.fn_a": ["pkg.mod.fn_b"],
        "pkg.mod.fn_b": [],
    }


def test_callers_of_clean(tmp_path: Path) -> None:
    """callers_of returns stale: false when backing files are clean."""
    content = b"def fn_b(): pass\n"
    (tmp_path / "mod.py").write_bytes(content)
    skeletons = {"mod.py": [_sym("pkg.mod.fn_a"), _sym("pkg.mod.fn_b")]}
    shas = {"mod.py": _sha256(content)}
    idx = _make_idx(tmp_path, _raw_ab(), skeletons, shas)

    result = idx.callers_of("pkg.mod.fn_b", depth=1)
    assert result["stale"] is False
    assert result["stale_files"] == []


def test_callers_of_stale(tmp_path: Path) -> None:
    """callers_of returns stale: true when a backing file changed."""
    (tmp_path / "mod.py").write_bytes(b"def fn_a(): return fn_b()\n")
    skeletons = {"mod.py": [_sym("pkg.mod.fn_a"), _sym("pkg.mod.fn_b")]}
    shas = {"mod.py": _sha256(b"def fn_a(): pass\n")}  # stored differs from disk
    idx = _make_idx(tmp_path, _raw_ab(), skeletons, shas)

    result = idx.callers_of("pkg.mod.fn_b", depth=1)
    assert result["stale"] is True
    assert "mod.py" in result["stale_files"]


def test_callees_of_clean(tmp_path: Path) -> None:
    """callees_of returns stale: false when backing files are clean."""
    content = b"def fn_a(): pass\n"
    (tmp_path / "mod.py").write_bytes(content)
    skeletons = {"mod.py": [_sym("pkg.mod.fn_a"), _sym("pkg.mod.fn_b")]}
    shas = {"mod.py": _sha256(content)}
    idx = _make_idx(tmp_path, _raw_ab(), skeletons, shas)

    result = idx.callees_of("pkg.mod.fn_a", depth=1)
    assert result["stale"] is False
    assert result["stale_files"] == []


def test_callees_of_stale(tmp_path: Path) -> None:
    """callees_of returns stale: true when a backing file changed."""
    (tmp_path / "mod.py").write_bytes(b"def fn_b(): return 42\n")
    skeletons = {"mod.py": [_sym("pkg.mod.fn_a"), _sym("pkg.mod.fn_b")]}
    shas = {"mod.py": _sha256(b"def fn_b(): pass\n")}  # stored differs from disk
    idx = _make_idx(tmp_path, _raw_ab(), skeletons, shas)

    result = idx.callees_of("pkg.mod.fn_a", depth=1)
    assert result["stale"] is True
    assert "mod.py" in result["stale_files"]


def test_callers_of_pre_v3(tmp_path: Path) -> None:
    """callers_of pre-v3 index returns index_format_incompatible."""
    skeletons = {"mod.py": [_sym("pkg.mod.fn_a"), _sym("pkg.mod.fn_b")]}
    idx = _make_idx(tmp_path, _raw_ab(), skeletons, file_shas=None)
    result = idx.callers_of("pkg.mod.fn_b")
    assert result["stale"] is True
    assert result["index_stale_reason"] == "index_format_incompatible"


# ---------------------------------------------------------------------------
# 4. search
# ---------------------------------------------------------------------------

def test_search_clean(tmp_path: Path) -> None:
    """search returns stale: false when backing files are clean."""
    content = b"def fn_a(): pass\n"
    (tmp_path / "mod.py").write_bytes(content)
    skeletons = {"mod.py": [_sym("pkg.mod.fn_a")]}
    shas = {"mod.py": _sha256(content)}
    idx = _make_idx(tmp_path, {"pkg.mod.fn_a": []}, skeletons, shas)

    result = idx.search("fn_a")
    assert result["stale"] is False
    assert result["stale_files"] == []
    assert "stale_action" not in result


def test_search_stale(tmp_path: Path) -> None:
    """search returns stale: true when a matched FQN's file changed."""
    (tmp_path / "mod.py").write_bytes(b"def fn_a(): return 42\n")
    skeletons = {"mod.py": [_sym("pkg.mod.fn_a")]}
    shas = {"mod.py": _sha256(b"def fn_a(): pass\n")}
    idx = _make_idx(tmp_path, {"pkg.mod.fn_a": []}, skeletons, shas)

    result = idx.search("fn_a")
    assert result["stale"] is True
    assert "mod.py" in result["stale_files"]


def test_search_dirty_non_result_not_included(tmp_path: Path) -> None:
    """search stale_files excludes dirty files not backing any matched FQN."""
    content_a = b"def fn_a(): pass\n"
    (tmp_path / "mod_a.py").write_bytes(content_a)
    (tmp_path / "mod_b.py").write_bytes(b"def fn_b(): return 99\n")  # dirty
    skeletons = {
        "mod_a.py": [_sym("pkg.mod_a.fn_a")],
        "mod_b.py": [_sym("pkg.mod_b.fn_b")],
    }
    shas = {
        "mod_a.py": _sha256(content_a),
        "mod_b.py": _sha256(b"def fn_b(): pass\n"),
    }
    idx = _make_idx(tmp_path, {"pkg.mod_a.fn_a": [], "pkg.mod_b.fn_b": []}, skeletons, shas)

    # Search only matches fn_a; fn_b is in a dirty file but not in results
    result = idx.search("fn_a")
    assert result["stale"] is False
    assert result["stale_files"] == []


# ---------------------------------------------------------------------------
# 5. neighborhood
# ---------------------------------------------------------------------------

def _raw_neighborhood() -> dict[str, list[str]]:
    return {
        "pkg.mod.hub": ["pkg.mod.spoke_a", "pkg.mod.spoke_b"],
        "pkg.mod.spoke_a": [],
        "pkg.mod.spoke_b": [],
    }


def test_neighborhood_clean(tmp_path: Path) -> None:
    """neighborhood returns stale: false when backing files are clean."""
    content = b"def hub(): pass\n"
    (tmp_path / "mod.py").write_bytes(content)
    skeletons = {
        "mod.py": [
            _sym("pkg.mod.hub"),
            _sym("pkg.mod.spoke_a"),
            _sym("pkg.mod.spoke_b"),
        ]
    }
    shas = {"mod.py": _sha256(content)}
    idx = _make_idx(tmp_path, _raw_neighborhood(), skeletons, shas)

    result = idx.neighborhood("pkg.mod.hub", depth=1)
    assert result["stale"] is False
    assert result["stale_files"] == []


def test_neighborhood_stale(tmp_path: Path) -> None:
    """neighborhood returns stale: true when an edge node's file changed."""
    (tmp_path / "mod.py").write_bytes(b"def hub(): return spoke_a()\n")
    skeletons = {
        "mod.py": [
            _sym("pkg.mod.hub"),
            _sym("pkg.mod.spoke_a"),
            _sym("pkg.mod.spoke_b"),
        ]
    }
    shas = {"mod.py": _sha256(b"def hub(): pass\n")}
    idx = _make_idx(tmp_path, _raw_neighborhood(), skeletons, shas)

    result = idx.neighborhood("pkg.mod.hub", depth=1)
    assert result["stale"] is True
    assert "mod.py" in result["stale_files"]


# ---------------------------------------------------------------------------
# 6. module_callers / module_callees
# ---------------------------------------------------------------------------

def _raw_module() -> dict[str, list[str]]:
    """pkg.core calls pkg.util functions."""
    return {
        "pkg.core.run": ["pkg.util.helper"],
        "pkg.util.helper": [],
    }


def test_module_callers_clean(tmp_path: Path) -> None:
    """module_callers returns stale: false when backing files are clean."""
    content = b"def run(): pass\n"
    (tmp_path / "core.py").write_bytes(content)
    skeletons = {"core.py": [_sym("pkg.core.run")]}
    shas = {"core.py": _sha256(content)}
    idx = _make_idx(tmp_path, _raw_module(), skeletons, shas)

    # pkg.core calls pkg.util; module_callers("pkg.util") → ["pkg.core"]
    result = idx.module_callers("pkg.util")
    assert result["stale"] is False
    assert result["stale_files"] == []


def test_module_callers_stale(tmp_path: Path) -> None:
    """module_callers returns stale: true when a result module's file changed."""
    (tmp_path / "core.py").write_bytes(b"def run(): return helper()\n")
    skeletons = {"core.py": [_sym("pkg.core.run")]}
    shas = {"core.py": _sha256(b"def run(): pass\n")}
    idx = _make_idx(tmp_path, _raw_module(), skeletons, shas)

    result = idx.module_callers("pkg.util")
    assert result["stale"] is True
    assert "core.py" in result["stale_files"]


def test_module_callees_clean(tmp_path: Path) -> None:
    """module_callees returns stale: false when backing files are clean."""
    content = b"def helper(): pass\n"
    (tmp_path / "util.py").write_bytes(content)
    skeletons = {"util.py": [_sym("pkg.util.helper")]}
    shas = {"util.py": _sha256(content)}
    idx = _make_idx(tmp_path, _raw_module(), skeletons, shas)

    # module_callees("pkg.core") → ["pkg.util"]
    result = idx.module_callees("pkg.core")
    assert result["stale"] is False
    assert result["stale_files"] == []


def test_module_callees_stale(tmp_path: Path) -> None:
    """module_callees returns stale: true when a result module's file changed."""
    (tmp_path / "util.py").write_bytes(b"def helper(): return 42\n")
    skeletons = {"util.py": [_sym("pkg.util.helper")]}
    shas = {"util.py": _sha256(b"def helper(): pass\n")}
    idx = _make_idx(tmp_path, _raw_module(), skeletons, shas)

    result = idx.module_callees("pkg.core")
    assert result["stale"] is True
    assert "util.py" in result["stale_files"]


def test_module_callers_pre_v3(tmp_path: Path) -> None:
    """module_callers with pre-v3 index returns index_format_incompatible."""
    skeletons = {"core.py": [_sym("pkg.core.run")]}
    idx = _make_idx(tmp_path, _raw_module(), skeletons, file_shas=None)
    result = idx.module_callers("pkg.util")
    assert result["stale"] is True
    assert result["index_stale_reason"] == "index_format_incompatible"
