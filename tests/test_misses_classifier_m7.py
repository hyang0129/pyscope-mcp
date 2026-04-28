"""M7 classifier extension: loguru / re / datetime / difflib accepted-miss buckets.

Each test verifies a method-name is classified into the correct accepted bucket.
False-positive guards verify that self.method, super().method, and unknown
attr chains are NOT stolen by the new whitelists.
"""

from __future__ import annotations

import ast

from pyscope_mcp.analyzer import _classify_miss


def _parse_call(src: str) -> ast.Call:
    """Parse a single-expression snippet and return the outermost Call node."""
    tree = ast.parse(src)
    expr = tree.body[0]
    assert isinstance(expr, ast.Expr)
    call = expr.value
    assert isinstance(call, ast.Call)
    return call


def _first_call_in_class(src: str) -> ast.Call:
    """Find the first ast.Call inside a class body (for self.* tests)."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            return node
    raise AssertionError("No Call node found")


# ---------------------------------------------------------------------------
# loguru positive cases
# ---------------------------------------------------------------------------

def test_logger_info_is_loguru_method_call() -> None:
    call = _parse_call('logger.info("x")')
    assert _classify_miss(call) == "loguru_method_call"


def test_logger_opt_warning_outer_is_loguru_method_call() -> None:
    """logger.opt(exception=True).warning("x") — outer Call.
    The method is 'warning' which is in LOGURU_METHODS (chain-None fallback
    because the receiver is a Call result), AND the chain-root check fires
    for the inner call which has chain[0]=='logger'."""
    call = _parse_call('logger.opt(exception=True).warning("x")')
    assert _classify_miss(call) == "loguru_method_call"


def test_log_log_is_loguru_method_call() -> None:
    call = _parse_call('log.log("x")')
    assert _classify_miss(call) == "loguru_method_call"


# ---------------------------------------------------------------------------
# re positive cases
# ---------------------------------------------------------------------------

def test_re_search_is_re_method_call() -> None:
    call = _parse_call("_RE.search(line)")
    assert _classify_miss(call) == "re_method_call"


def test_match_start_is_re_method_call() -> None:
    call = _parse_call("match.start()")
    assert _classify_miss(call) == "re_method_call"


def test_pattern_sub_is_re_method_call() -> None:
    call = _parse_call('pattern.sub("a", "b")')
    assert _classify_miss(call) == "re_method_call"


# ---------------------------------------------------------------------------
# datetime positive cases
# ---------------------------------------------------------------------------

def test_datetime_now_isoformat_outer_is_datetime_method_call() -> None:
    """datetime.now(UTC).isoformat() — outer call has chain=None (receiver is Call).
    'isoformat' is in DATETIME_METHODS so fallback catches it."""
    call = _parse_call("datetime.now(UTC).isoformat()")
    assert _classify_miss(call) == "datetime_method_call"


def test_datetime_now_is_datetime_method_call() -> None:
    call = _parse_call("datetime.now()")
    assert _classify_miss(call) == "datetime_method_call"


# ---------------------------------------------------------------------------
# builtin extension: fromkeys
# ---------------------------------------------------------------------------

def test_dict_fromkeys_is_builtin_method_call() -> None:
    call = _parse_call("dict.fromkeys(xs)")
    assert _classify_miss(call) == "builtin_method_call"


# ---------------------------------------------------------------------------
# pathlib extension: as_posix (chain-None fallback)
# ---------------------------------------------------------------------------

def test_path_as_posix_chain_none_fallback_is_pathlib_method_call() -> None:
    """Path('/x').relative_to('/').as_posix() — outer call, chain is None."""
    call = _parse_call("Path('/x').relative_to('/').as_posix()")
    assert _classify_miss(call) == "pathlib_method_call"


# ---------------------------------------------------------------------------
# difflib positive case (chain-None fallback)
# ---------------------------------------------------------------------------

def test_difflib_ratio_chain_none_is_difflib_method_call() -> None:
    """difflib.SequenceMatcher(None, 'a', 'b').ratio() — chain=None."""
    call = _parse_call("difflib.SequenceMatcher(None, 'a', 'b').ratio()")
    assert _classify_miss(call) == "difflib_method_call"


# ---------------------------------------------------------------------------
# PIL extension: draw.rectangle
# ---------------------------------------------------------------------------

def test_draw_rectangle_is_pil_method_call() -> None:
    call = _parse_call("draw.rectangle([(0, 0), (1, 1)])")
    assert _classify_miss(call) == "pil_method_call"


# ---------------------------------------------------------------------------
# False-positive guards
# ---------------------------------------------------------------------------

def test_self_search_is_self_method_unresolved() -> None:
    """self.search(x) inside a class must remain self_method_unresolved.
    The chain[0]=='self' branch fires before any re/loguru whitelist check."""
    src = (
        "class Finder:\n"
        "    def run(self, x):\n"
        "        self.search(x)\n"
    )
    tree = ast.parse(src)
    # Find the self.search(...) call
    call_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "search"
        ):
            call_node = node
            break
    assert call_node is not None
    assert _classify_miss(call_node) == "self_method_unresolved"


def test_super_info_is_super_unresolved_not_loguru() -> None:
    """super().info(...) — the super() branch fires first, returns super_unresolved."""
    call = _parse_call("super().info(msg)")
    assert _classify_miss(call) == "super_unresolved"


def test_foo_bar_unknown_stays_attr_chain_unresolved() -> None:
    """foo.bar() where neither foo nor bar matches any whitelist."""
    call = _parse_call("foo.bar()")
    assert _classify_miss(call) == "attr_chain_unresolved"


# ---------------------------------------------------------------------------
# Cross-file linkage: M7 tags must be present in ACCEPTED_PATTERNS
# ---------------------------------------------------------------------------

def test_m7_tags_are_all_accepted() -> None:
    """Every tag produced by M7 positive cases must be in ACCEPTED_PATTERNS.

    This guards against adding a new accepted bucket in classify_miss while
    forgetting to update ACCEPTED_PATTERNS, which would silently mis-route
    calls to record_miss instead of record_accepted.
    """
    from pyscope_mcp.analyzer.misses import ACCEPTED_PATTERNS
    for tag in {"loguru_method_call", "re_method_call", "datetime_method_call", "difflib_method_call"}:
        assert tag in ACCEPTED_PATTERNS
