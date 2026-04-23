"""Handler #19: external-factory local-var accept buckets.

Tests that method calls on locals bound from whitelisted external factories
(httpx, boto3, typer, googleapiclient) are classified into the correct
accepted bucket rather than emitted as attr_chain_unresolved.

Each factory family has:
  - A positive test: correct bucket, no unresolved entry.
  - FP guards: shadowed local, non-whitelisted factory, function param.
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
# httpx.Client
# ---------------------------------------------------------------------------

def test_httpx_client_post_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import httpx\n"
            "\n"
            "def upload(url: str, data: bytes) -> None:\n"
            "    client = httpx.Client()\n"
            "    client.post(url, content=data)\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    summary = report["summary"]
    assert summary["accepted_counts"].get("httpx_method_call", 0) >= 1, (
        f"Expected httpx_method_call in accepted, got: {summary['accepted_counts']}"
    )
    patterns = {e["pattern"] for e in report["unresolved_calls"]}
    assert "attr_chain_unresolved" not in patterns, (
        f"client.post() should not be attr_chain_unresolved; unresolved: {report['unresolved_calls']}"
    )


def test_httpx_async_client_get_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import httpx\n"
            "\n"
            "async def fetch(url: str) -> bytes:\n"
            "    client = httpx.AsyncClient()\n"
            "    resp = await client.get(url)\n"
            "    return resp\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    summary = report["summary"]
    assert summary["accepted_counts"].get("httpx_method_call", 0) >= 1


# ---------------------------------------------------------------------------
# boto3.client / boto3.resource
# ---------------------------------------------------------------------------

def test_boto3_client_put_object_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import boto3\n"
            "\n"
            "def store(bucket: str, key: str, body: bytes) -> None:\n"
            "    client = boto3.client('s3')\n"
            "    client.put_object(Bucket=bucket, Key=key, Body=body)\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    summary = report["summary"]
    assert summary["accepted_counts"].get("boto3_method_call", 0) >= 1
    patterns = {e["pattern"] for e in report["unresolved_calls"]}
    assert "attr_chain_unresolved" not in patterns


def test_boto3_client_head_object_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import boto3\n"
            "\n"
            "def exists(bucket: str, key: str) -> bool:\n"
            "    client = boto3.client('s3')\n"
            "    client.head_object(Bucket=bucket, Key=key)\n"
            "    return True\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    assert report["summary"]["accepted_counts"].get("boto3_method_call", 0) >= 1


# ---------------------------------------------------------------------------
# boto3 chained factory: get_paginator → paginator.paginate
# ---------------------------------------------------------------------------

def test_boto3_paginator_paginate_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import boto3\n"
            "\n"
            "def list_objects(bucket: str):\n"
            "    client = boto3.client('s3')\n"
            "    paginator = client.get_paginator('list_objects_v2')\n"
            "    for page in paginator.paginate(Bucket=bucket):\n"
            "        pass\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    summary = report["summary"]
    # get_paginator itself is boto3_method_call; paginator.paginate is also boto3_method_call
    assert summary["accepted_counts"].get("boto3_method_call", 0) >= 2, (
        f"Expected >=2 boto3_method_call, got {summary['accepted_counts']}"
    )


# ---------------------------------------------------------------------------
# typer.Typer
# ---------------------------------------------------------------------------

def test_typer_command_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import typer\n"
            "\n"
            "_cli = typer.Typer()\n"
            "\n"
            "def setup():\n"
            "    _cli.command()(lambda: None)\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    summary = report["summary"]
    assert summary["accepted_counts"].get("typer_method_call", 0) >= 1


def test_typer_command_in_function_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import typer\n"
            "\n"
            "def make_app():\n"
            "    app = typer.Typer()\n"
            "    app.command()(lambda: None)\n"
            "    return app\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    assert report["summary"]["accepted_counts"].get("typer_method_call", 0) >= 1


# ---------------------------------------------------------------------------
# googleapiclient.discovery.build
# ---------------------------------------------------------------------------

def test_googleapi_channels_list_accepted(tmp_path: Path) -> None:
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "from googleapiclient import discovery\n"
            "\n"
            "def get_channels(api_key: str):\n"
            "    service = discovery.build('youtube', 'v3', developerKey=api_key)\n"
            "    service.channels()\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    assert report["summary"]["accepted_counts"].get("googleapi_method_call", 0) >= 1


# ---------------------------------------------------------------------------
# False-positive guard: shadowed local — client reassigned to non-factory value
# ---------------------------------------------------------------------------

def test_shadowed_local_not_accepted(tmp_path: Path) -> None:
    """After `client = 42`, client.post() should NOT be accepted as httpx_method_call."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import httpx\n"
            "\n"
            "def mixed():\n"
            "    client = httpx.Client()\n"
            "    client = 42\n"
            "    client.post('http://example.com')\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    # After shadow, client is int — should NOT be accepted as httpx_method_call.
    assert report["summary"]["accepted_counts"].get("httpx_method_call", 0) == 0, (
        "Shadowed local should not be accepted as httpx_method_call"
    )


# ---------------------------------------------------------------------------
# False-positive guard: non-whitelisted factory
# ---------------------------------------------------------------------------

def test_non_whitelisted_factory_not_accepted(tmp_path: Path) -> None:
    """A call like SomeOtherLib.Client() → .post() must NOT produce httpx_method_call."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import some_other_lib\n"
            "\n"
            "def run():\n"
            "    client = some_other_lib.Client()\n"
            "    client.post('http://example.com')\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    assert report["summary"]["accepted_counts"].get("httpx_method_call", 0) == 0
    assert report["summary"]["accepted_counts"].get("boto3_method_call", 0) == 0
    assert report["summary"]["accepted_counts"].get("typer_method_call", 0) == 0
    assert report["summary"]["accepted_counts"].get("googleapi_method_call", 0) == 0


# ---------------------------------------------------------------------------
# False-positive guard: client is a parameter (no local binding)
# ---------------------------------------------------------------------------

def test_parameter_not_accepted(tmp_path: Path) -> None:
    """If `client` is a function parameter with no type binding, .post() stays unresolved."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "def upload(client, url: str) -> None:\n"
            "    client.post(url)\n"
        ),
    })
    _raw, report = build_with_report(root, "pkg")
    # No factory binding — should not be httpx_method_call accepted
    assert report["summary"]["accepted_counts"].get("httpx_method_call", 0) == 0


# ---------------------------------------------------------------------------
# False-positive guard: in-package class with same name wins over factory whitelist
# ---------------------------------------------------------------------------

def test_inpackage_class_wins_over_whitelist(tmp_path: Path) -> None:
    """An in-package `Client` class is resolved via local_types, not httpx whitelist."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Client:\n"
            "    def post(self, url: str):\n"
            "        pass\n"
            "\n"
            "def run():\n"
            "    c = Client()\n"
            "    c.post('http://example.com')\n"
        ),
    })
    raw, report = build_with_report(root, "pkg")
    # Resolved in-package, not accepted as httpx_method_call
    assert "pkg.mod.Client.post" in raw.get("pkg.mod.run", [])
    assert report["summary"]["accepted_counts"].get("httpx_method_call", 0) == 0
