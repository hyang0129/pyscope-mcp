"""Tests for chained googleapiclient.discovery.Resource call acceptance.

Covers inline chains of arbitrary depth rooted at a local variable whose
factory FQN is in EXTERNAL_SELF_RETURNING.

Pattern:
    service = build("youtube", "v3", ...)
    service.channels().list(part="snippet").execute()

Each intermediate segment (channels(), list(), execute()) should be accepted
into the "googleapi_method_call" bucket instead of landing as attr_chain_unresolved.
"""

from __future__ import annotations

from pathlib import Path

from pyscope_mcp.analyzer import build_with_report


def _make_package(tmp_path: Path, pkg_name: str, files: dict[str, str]) -> Path:
    pkg = tmp_path / pkg_name
    pkg.mkdir()
    if "__init__.py" not in files:
        (pkg / "__init__.py").write_text("")
    for rel, content in files.items():
        target = pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


# ---------------------------------------------------------------------------
# Core feature: 3-segment inline chain
# ---------------------------------------------------------------------------

def test_googleapi_three_segment_chain_accepted(tmp_path: Path) -> None:
    """service.channels().list(...) — 3-segment chain accepted as googleapi_method_call."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from googleapiclient import discovery\n"
            "\n"
            "def get_channels(api_key: str):\n"
            "    service = discovery.build('youtube', 'v3', developerKey=api_key)\n"
            "    result = service.channels().list(part='snippet')\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    summary = report["summary"]
    # Both service.channels() (2-part, handled by existing path) and
    # service.channels().list(...) (3-part chain) should be accepted.
    assert summary["accepted_counts"].get("googleapi_method_call", 0) >= 2, (
        f"Expected >=2 googleapi_method_call; got: {summary['accepted_counts']}"
    )
    patterns = {e["pattern"] for e in report["unresolved_calls"]}
    assert "attr_chain_unresolved" not in patterns, (
        f"Chained call should not be attr_chain_unresolved; unresolved: {report['unresolved_calls']}"
    )


def test_googleapi_four_segment_chain_accepted(tmp_path: Path) -> None:
    """service.channels().list(...).execute() — 4-segment inline chain accepted."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from googleapiclient import discovery\n"
            "\n"
            "def run(api_key: str):\n"
            "    service = discovery.build('youtube', 'v3', developerKey=api_key)\n"
            "    service.channels().list(part='snippet').execute()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    summary = report["summary"]
    # service.channels() → accepted (2-part, existing handler)
    # service.channels().list(...) → accepted (3-part, new handler)
    # service.channels().list(...).execute() → accepted (4-part, new handler)
    assert summary["accepted_counts"].get("googleapi_method_call", 0) >= 3, (
        f"Expected >=3 googleapi_method_call; got: {summary['accepted_counts']}"
    )
    patterns = {e["pattern"] for e in report["unresolved_calls"]}
    assert "attr_chain_unresolved" not in patterns, (
        f"Chained calls should not be attr_chain_unresolved; unresolved: {report['unresolved_calls']}"
    )


def test_googleapi_videos_insert_chain_accepted(tmp_path: Path) -> None:
    """service.videos().insert(...).execute() — different resource, still accepted."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from googleapiclient.discovery import build\n"
            "\n"
            "def upload_video(api_key: str):\n"
            "    youtube = build('youtube', 'v3', developerKey=api_key)\n"
            "    youtube.videos().insert(part='snippet', body={}).execute()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    summary = report["summary"]
    assert summary["accepted_counts"].get("googleapi_method_call", 0) >= 3, (
        f"Expected >=3 googleapi_method_call; got: {summary['accepted_counts']}"
    )
    patterns = {e["pattern"] for e in report["unresolved_calls"]}
    assert "attr_chain_unresolved" not in patterns


def test_googleapi_module_level_service_chain(tmp_path: Path) -> None:
    """Module-level service variable — chain from inside a function is accepted."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from googleapiclient import discovery\n"
            "\n"
            "_service = discovery.build('youtube', 'v3')\n"
            "\n"
            "def query():\n"
            "    _service.channels().list(part='snippet').execute()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    summary = report["summary"]
    # The inline chain calls (channels().list().execute()) should be accepted.
    assert summary["accepted_counts"].get("googleapi_method_call", 0) >= 1, (
        f"Expected >=1 googleapi_method_call; got: {summary['accepted_counts']}"
    )


# ---------------------------------------------------------------------------
# False-positive guard: non-self-returning factory must NOT chain-walk
# ---------------------------------------------------------------------------

def test_httpx_chain_not_self_returning(tmp_path: Path) -> None:
    """httpx.Client is NOT self-returning; inline chained call should NOT accept via chain walker."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import httpx\n"
            "\n"
            "def run():\n"
            "    client = httpx.Client()\n"
            "    # This contrived chain won't be accepted by chain-walker\n"
            "    client.post('http://example.com').raise_for_status()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    # httpx.Client is NOT in EXTERNAL_SELF_RETURNING, so chain walker must not fire.
    # client.post(...) is still accepted via the 2-part handler.
    # client.post(...).raise_for_status() is an unknown chain — that's fine, it
    # stays unresolved; we just must NOT accept it as httpx_method_call via chain-walker.
    # The 2-part call client.post(...) itself is accepted as httpx_method_call.
    summary = report["summary"]
    # Count of httpx_method_call accepted calls: only client.post() qualifies (1 call).
    # The outer raise_for_status() is on the result of post(), which is NOT httpx.Client.
    # So the chain walker must not incorrectly accept that outer call.
    # We verify the outer call (raise_for_status on response) is NOT accepted as
    # httpx_method_call by checking unresolved_calls contains something from this file.
    # The important assertion: the chain walker must not fire for non-self-returning
    # factories. The googleapi bucket must remain at 0.
    assert summary["accepted_counts"].get("googleapi_method_call", 0) == 0, (
        "Chain walker must not accept httpx chain as googleapi_method_call"
    )


def test_boto3_chain_not_self_returning(tmp_path: Path) -> None:
    """boto3.client is NOT self-returning; chain walker must not apply."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import boto3\n"
            "\n"
            "def run():\n"
            "    client = boto3.client('s3')\n"
            "    # Contrived inline chain\n"
            "    client.put_object(Bucket='b', Key='k', Body=b'data').something()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    summary = report["summary"]
    # Chain walker must not fire for boto3.client (not self-returning).
    # The googleapi bucket must remain at 0.
    assert summary["accepted_counts"].get("googleapi_method_call", 0) == 0, (
        "Chain walker must not accept boto3 chain as googleapi_method_call"
    )


# ---------------------------------------------------------------------------
# False-positive guard: unknown root variable must not accept
# ---------------------------------------------------------------------------

def test_unknown_root_var_not_accepted(tmp_path: Path) -> None:
    """Chain rooted at an unbound variable must not produce googleapi_method_call."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def run(unknown_service):\n"
            "    unknown_service.channels().list(part='snippet').execute()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    assert report["summary"]["accepted_counts"].get("googleapi_method_call", 0) == 0, (
        "Unknown root var should not produce googleapi_method_call"
    )


# ---------------------------------------------------------------------------
# False-positive guard: in-package class local var must not be intercepted
# ---------------------------------------------------------------------------

def test_inpackage_class_chain_not_intercepted(tmp_path: Path) -> None:
    """An in-package class chain uses existing resolvers, not the chain walker."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Service:\n"
            "    def channels(self):\n"
            "        return self\n"
            "    def list(self, part: str):\n"
            "        pass\n"
            "\n"
            "def run():\n"
            "    svc = Service()\n"
            "    svc.channels()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    # svc.channels() should be resolved as an in-package call, not googleapi_method_call.
    assert report["summary"]["accepted_counts"].get("googleapi_method_call", 0) == 0
    # Verify it resolved in-package.
    assert "pkg.mod.Service.channels" in _raw.get("pkg.mod.run", [])


# ---------------------------------------------------------------------------
# False-positive guard: self as chain root must not be intercepted
# ---------------------------------------------------------------------------

def test_self_root_chain_not_intercepted(tmp_path: Path) -> None:
    """self.something().method() should not be caught by the chain walker."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Processor:\n"
            "    def _build(self):\n"
            "        return self\n"
            "    def run(self):\n"
            "        self._build().run()\n"
        ),
    })
    _raw, report, _skeletons, _file_shas = build_with_report(root, "pkg")
    # Must not accept via googleapi chain walker.
    assert report["summary"]["accepted_counts"].get("googleapi_method_call", 0) == 0
