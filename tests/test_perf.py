"""Benchmarks for the constitutional Law 2 / Law 4 budgets.

Two budgets are protected here:

- **Law 2 — cold-load p95 < 500 ms** on the self-index.  Measured by
  spawning 10 fresh ``python`` subprocesses and timing
  ``CallGraphIndex.load`` end-to-end (process startup + JSON parse +
  ``from_nodes`` rebuild).  We use subprocesses to defeat any in-process
  caching and replicate what an MCP client experiences when the server
  starts.

- **Law 4 — full build < 60 s** on the self-index.  A single in-process
  measurement of ``build_nodes_with_report`` + ``CallGraphIndex.save``.
  Build cost is dominated by file I/O and AST parsing; an in-process
  measurement is faithful to what ``pyscope-mcp build`` does.

These tests are environment-sensitive — they target real wall-clock
budgets, so they may be noisy on a hot CI runner but they are the only
way to falsify the kill criterion in epic #76's intent doc.  When the
self-index does not exist on disk yet, the cold-load test builds it
once at module setup.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "pyscope_mcp"
SRC_ROOT = REPO_ROOT / "src"
INDEX_PATH = REPO_ROOT / ".pyscope-mcp" / "index.json"

# Number of subprocess timings for the cold-load p95 test.  10 is the
# minimum sample size that lets ``statistics.quantiles(n=20)[18]`` resolve
# the 95th percentile usefully — fewer samples and the metric is noisy.
_COLD_LOAD_SAMPLES = 10

# Per-attempt timeout (seconds) for the cold-load subprocess.  A real
# breach of the 500 ms budget will still complete in well under a second;
# the timeout here just keeps a hung subprocess from stalling the suite.
_COLD_LOAD_PROCESS_TIMEOUT = 10.0


def _build_self_index_inplace() -> None:
    """Build the pyscope-mcp self-index at the canonical location.

    Used by the cold-load test as a setup step when no index exists yet.
    Writes to the same path the CLI's ``build`` subcommand would write.
    """
    from pyscope_mcp.analyzer.pipeline import build_nodes_with_report
    from pyscope_mcp.graph import CallGraphIndex

    nodes, _report, skeletons, file_shas = build_nodes_with_report(
        SRC_ROOT, PACKAGE
    )
    idx = CallGraphIndex.from_nodes(
        SRC_ROOT, nodes, skeletons=skeletons, file_shas=file_shas
    )
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    idx.save(INDEX_PATH)


def _ensure_self_index_exists() -> None:
    if not INDEX_PATH.exists():
        _build_self_index_inplace()


def test_cold_load_p95_under_500ms() -> None:
    """Law 2 — cold-load p95 must stay under 500 ms on the self-index.

    Spawns 10 fresh Python subprocesses, each timing
    ``CallGraphIndex.load(INDEX_PATH)`` end-to-end and printing the
    elapsed seconds to stdout.  Asserts ``p95 < 0.5``.

    This is the canonical falsifier for epic #76's kill criterion: a
    sustained breach here means the site-keyed shape lost the Law 2 budget
    and the bucket-by-kind fallback should be revisited.
    """
    _ensure_self_index_exists()

    timings: list[float] = []
    snippet = (
        "import time, sys; "
        "from pyscope_mcp.graph import CallGraphIndex; "
        f"_p = r'{INDEX_PATH}'; "
        "_t = time.perf_counter(); "
        "CallGraphIndex.load(_p); "
        "sys.stdout.write(f'{time.perf_counter() - _t:.6f}')"
    )
    for _ in range(_COLD_LOAD_SAMPLES):
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            timeout=_COLD_LOAD_PROCESS_TIMEOUT,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"cold-load subprocess failed: {result.stderr}"
        )
        out = result.stdout.strip()
        assert out, "cold-load subprocess produced no timing output"
        timings.append(float(out))

    timings.sort()
    # Compute p95.  ``statistics.quantiles(n=20)`` returns the 19 cut points
    # of the distribution; index 18 is the 95th percentile.
    p95 = statistics.quantiles(timings, n=20)[18]
    median = statistics.median(timings)
    print(
        f"cold-load timings (s): median={median:.4f} p95={p95:.4f} "
        f"min={min(timings):.4f} max={max(timings):.4f} "
        f"all={[f'{t:.4f}' for t in timings]}"
    )
    assert p95 < 0.5, (
        f"Law 2 budget breached: cold-load p95 = {p95*1000:.1f} ms "
        f"(>= 500 ms ceiling).  See epic #76 kill criterion."
    )


def test_full_build_under_60s() -> None:
    """Law 4 — a fresh build of the self-index must finish within 60 s.

    Times ``build_nodes_with_report`` + ``CallGraphIndex.save`` on the
    pyscope-mcp source tree.  This is the single-shot equivalent of what
    ``pyscope-mcp build`` does; the budget includes pass 1 (discovery),
    pass 2 (visitors), build-time inversion, and serialisation.
    """
    from pyscope_mcp.analyzer.pipeline import build_nodes_with_report
    from pyscope_mcp.graph import CallGraphIndex

    start = time.perf_counter()
    nodes, _report, skeletons, file_shas = build_nodes_with_report(
        SRC_ROOT, PACKAGE
    )
    idx = CallGraphIndex.from_nodes(
        SRC_ROOT, nodes, skeletons=skeletons, file_shas=file_shas
    )
    # Save to a temporary location to avoid clobbering a pre-existing
    # canonical index that other tests in the same run may rely on.
    tmp_out = REPO_ROOT / ".pyscope-mcp" / "index.benchmark.json"
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        idx.save(tmp_out)
        elapsed = time.perf_counter() - start
    finally:
        if tmp_out.exists():
            tmp_out.unlink()

    print(f"full build time: {elapsed:.2f} s")
    assert elapsed < 60.0, (
        f"Law 4 budget breached: full build took {elapsed:.1f} s "
        f"(>= 60 s ceiling).  Investigate analyzer regressions."
    )
