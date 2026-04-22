"""M8: self.<attr>.<method>() resolution via __init__ attribute type tracking."""

from __future__ import annotations

from pathlib import Path

from pyscope_mcp.analyzer import build_raw


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
# Case 1: constructor-RHS
# ---------------------------------------------------------------------------

def test_constructor_rhs_same_module(tmp_path: Path) -> None:
    """self.X = Foo() in __init__ → self.X.method() resolves to Foo.method."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Logger:\n"
            "    def log(self, msg):\n"
            "        pass\n"
            "\n"
            "class Agent:\n"
            "    def __init__(self):\n"
            "        self.logger = Logger()\n"
            "\n"
            "    def run(self):\n"
            "        self.logger.log('hello')\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.Logger.log" in raw.get("pkg.mod.Agent.run", [])


def test_constructor_rhs_cross_module(tmp_path: Path) -> None:
    """Constructor RHS imported from another module."""
    root = _make_package(tmp_path, "pkg", {
        "logger.py": (
            "class Logger:\n"
            "    def log(self, msg):\n"
            "        pass\n"
        ),
        "agent.py": (
            "from pkg.logger import Logger\n"
            "\n"
            "class Agent:\n"
            "    def __init__(self):\n"
            "        self._logger = Logger()\n"
            "\n"
            "    def run(self):\n"
            "        self._logger.log('hi')\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.logger.Logger.log" in raw.get("pkg.agent.Agent.run", [])


# ---------------------------------------------------------------------------
# Case 2: annotated-parameter case
# ---------------------------------------------------------------------------

def test_annotated_param_rhs(tmp_path: Path) -> None:
    """self.X = foo where foo: Foo in __init__ signature → self.X.method()."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class TTS:\n"
            "    def synthesize(self, text):\n"
            "        pass\n"
            "\n"
            "class Runner:\n"
            "    def __init__(self, tts: TTS):\n"
            "        self._tts = tts\n"
            "\n"
            "    def speak(self, text):\n"
            "        self._tts.synthesize(text)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.TTS.synthesize" in raw.get("pkg.mod.Runner.speak", [])


def test_annotated_param_cross_module(tmp_path: Path) -> None:
    """Annotated param type imported from another module."""
    root = _make_package(tmp_path, "pkg", {
        "packager.py": (
            "class Packager:\n"
            "    def package(self, data):\n"
            "        pass\n"
        ),
        "pipeline.py": (
            "from pkg.packager import Packager\n"
            "\n"
            "class Pipeline:\n"
            "    def __init__(self, packager: Packager):\n"
            "        self._packager = packager\n"
            "\n"
            "    def run(self, data):\n"
            "        self._packager.package(data)\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.packager.Packager.package" in raw.get("pkg.pipeline.Pipeline.run", [])


# ---------------------------------------------------------------------------
# Case 3: MRO walk up the attr's class
# ---------------------------------------------------------------------------

def test_mro_walk_on_attr_class(tmp_path: Path) -> None:
    """Attr class inherits method from a base → MRO walk finds it."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class LoggerBase:\n"
            "    def log_llm(self, data):\n"
            "        pass\n"
            "\n"
            "class ScriptLogger(LoggerBase):\n"
            "    pass\n"
            "\n"
            "class Agent:\n"
            "    def __init__(self):\n"
            "        self.script_logger = ScriptLogger()\n"
            "\n"
            "    def run(self):\n"
            "        self.script_logger.log_llm({'tokens': 5})\n"
        ),
    })
    raw = build_raw(root, "pkg")
    assert "pkg.mod.LoggerBase.log_llm" in raw.get("pkg.mod.Agent.run", [])


# ---------------------------------------------------------------------------
# False-positive guards
# ---------------------------------------------------------------------------

def test_unresolvable_rhs_returns_none(tmp_path: Path) -> None:
    """attr assigned from an unresolvable name → must NOT emit a fake edge."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Agent:\n"
            "    def __init__(self):\n"
            "        self.helper = some_factory()\n"  # 'some_factory' is unknown
            "\n"
            "    def run(self):\n"
            "        self.helper.do_thing()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.Agent.run", [])
    # Must not fabricate any edge to a non-existent method.
    assert not any("do_thing" in c for c in callees)


def test_external_type_attr_returns_none(tmp_path: Path) -> None:
    """Attr annotated with an external (non-package) type → return None silently."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "import logging\n"
            "\n"
            "class Agent:\n"
            "    def __init__(self, logger: logging.Logger):\n"
            "        self._logger = logger\n"
            "\n"
            "    def run(self):\n"
            "        self._logger.info('hello')\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.Agent.run", [])
    # External logger.info should not appear as a fake in-package edge.
    assert not any("logging" in c for c in callees)
    assert not any("info" in c and "pkg" in c for c in callees)


def test_same_method_name_on_unrelated_class_not_picked(tmp_path: Path) -> None:
    """A method with the same name on an unrelated class must NOT be returned."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Unrelated:\n"
            "    def process(self):\n"
            "        pass\n"
            "\n"
            "class Helper:\n"
            "    pass\n"
            "    # no 'process' method\n"
            "\n"
            "class Agent:\n"
            "    def __init__(self):\n"
            "        self.h = Helper()\n"
            "\n"
            "    def run(self):\n"
            "        self.h.process()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.Agent.run", [])
    # Unrelated.process must not appear; h.process should just miss silently.
    assert "pkg.mod.Unrelated.process" not in callees


def test_no_attr_in_init_returns_none(tmp_path: Path) -> None:
    """Attr not assigned in __init__ → call on it misses silently, no fake edge."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class TTS:\n"
            "    def synthesize(self, text):\n"
            "        pass\n"
            "\n"
            "class Agent:\n"
            "    def run(self):\n"
            "        self._tts.synthesize('hi')\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.Agent.run", [])
    assert "pkg.mod.TTS.synthesize" not in callees


def test_first_assignment_wins(tmp_path: Path) -> None:
    """Reassignment of self.X later in __init__ is ignored; first type wins."""
    root = _make_package(tmp_path, "pkg", {
        "mod.py": (
            "class Alpha:\n"
            "    def run(self):\n"
            "        pass\n"
            "\n"
            "class Beta:\n"
            "    def run(self):\n"
            "        pass\n"
            "\n"
            "class Agent:\n"
            "    def __init__(self):\n"
            "        self.worker = Alpha()\n"
            "        self.worker = Beta()  # reassignment — ignored\n"
            "\n"
            "    def go(self):\n"
            "        self.worker.run()\n"
        ),
    })
    raw = build_raw(root, "pkg")
    callees = raw.get("pkg.mod.Agent.go", [])
    # First assignment (Alpha) wins; Beta.run should NOT be the result.
    # Alpha.run must appear (it's the first-wins type).
    assert "pkg.mod.Alpha.run" in callees
