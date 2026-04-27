"""Regression guard for server.py instructions= string (issue #70).

Asserts that the MCP server instructions contain the task-type-triggered
guidance phrases required by the issue spec. Tests both the in-process
object (_SERVER._instructions) and the wire-level initialize response.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import pyscope_mcp.server as srv
from pyscope_mcp._rpc import RpcServer


# ---------------------------------------------------------------------------
# Anchor phrases required in the instructions string.
# Each tuple is (label, phrase) where phrase is checked case-insensitively.
# ---------------------------------------------------------------------------
REQUIRED_ANCHORS = [
    ("trigger-framing keyword", "before"),
    ("refactor trigger keyword", "refactor"),
    ("callers_of anchor (refactor + delete triggers)", "callers_of"),
    ("delete-dead-code trigger", "delet"),     # matches "Deleting" and "delete"
    ("file_skeleton anchor (read-unfamiliar trigger)", "file_skeleton"),
    ("neighborhood anchor (read-unfamiliar trigger)", "neighborhood"),
    ("module_callers anchor (module-move trigger)", "module_callers"),
    ("completeness reference (limits paragraph)", "completeness"),
    ("partial reference (limits paragraph)", "partial"),
    ("import carve-out (grep-stays-correct)", "import"),
    ("except carve-out (grep-stays-correct)", "except"),
]


# ---------------------------------------------------------------------------
# Unit test: in-process object
# ---------------------------------------------------------------------------

class TestServerInstructionsObject:
    """Read _SERVER._instructions directly — fastest possible regression guard."""

    def _instructions(self) -> str:
        return srv._SERVER._instructions  # type: ignore[attr-defined]

    def test_instructions_is_non_empty_string(self):
        inst = self._instructions()
        assert isinstance(inst, str)
        assert inst.strip(), "_SERVER._instructions must be non-empty"

    @pytest.mark.parametrize("label,phrase", REQUIRED_ANCHORS)
    def test_anchor_present(self, label: str, phrase: str):
        inst = self._instructions().lower()
        assert phrase.lower() in inst, (
            f"Missing anchor [{label}]: expected '{phrase}' to appear in "
            f"_SERVER._instructions. "
            f"This guard ensures the task-type-triggered guidance (issue #70) "
            f"has not been reverted to the old descriptive blurb."
        )


# ---------------------------------------------------------------------------
# Wire-level test harness (mirrors test_rpc_server.py pattern)
# ---------------------------------------------------------------------------

class _FakeReader:
    """Feeds pre-encoded lines to the RPC loop as if they came from stdin."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)
        self._pos = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._pos >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._pos]
        self._pos += 1
        return line + b"\n"


class _FakeWriter:
    """Captures bytes written by the RPC loop (synchronous write, async drain)."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        pass

    def responses(self) -> list[dict]:
        result = []
        for chunk in self.chunks:
            for line in chunk.split(b"\n"):
                line = line.strip()
                if line:
                    result.append(json.loads(line))
        return result


def _req(method: str, params: Any = None, req_id: int = 1) -> bytes:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg).encode()


async def _run(server: RpcServer, lines: list[bytes]) -> list[dict]:
    reader = _FakeReader(lines)
    writer = _FakeWriter()
    await server._loop(reader, writer)
    return writer.responses()


@pytest.fixture()
def wire_server() -> RpcServer:
    """Return the live server instance used by the MCP server module."""
    return srv._SERVER  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_initialize_wire_instructions_anchors(wire_server: RpcServer):
    """Wire-level: initialize response instructions field contains all anchor phrases."""
    lines = [_req("initialize", {"protocolVersion": "2025-06-18"}, req_id=1)]
    responses = await _run(wire_server, lines)
    assert responses, "No responses from initialize"
    result = responses[0].get("result", {})
    assert "instructions" in result, "instructions key missing from initialize response"
    inst: str = result["instructions"]
    assert inst.strip(), "instructions must be non-empty in wire response"

    inst_lower = inst.lower()
    missing = [
        f"'{phrase}' (label: {label})"
        for label, phrase in REQUIRED_ANCHORS
        if phrase.lower() not in inst_lower
    ]
    assert not missing, (
        "Wire-level initialize response is missing required instruction anchors:\n"
        + "\n".join(f"  - {m}" for m in missing)
    )
