"""docs/ARCHITECTURE.md names only things that exist.

The seam map sends an agent from a concept to the module that owns it and to
the test that enforces the rule. A map naming a deleted module or a renamed
test is worse than none, and nothing else would notice the drift: every
backticked repo path in it must exist (a glob must match), and every
`tests/x.py::test_name` must name a function defined in that file.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAP = REPO / "docs" / "ARCHITECTURE.md"

PREFIXES = (
    "dw/",
    "dw_mcp/",
    "ui/",
    "scripts/",
    "tests/",
    "docs/",
    "workflows/",
    "prompts/",
    ".github/",
)

_TOKEN = re.compile(r"`([^`\s]+)`")


def map_paths(text):
    """Every backticked token in `text` that starts with a repo prefix, as
    (path, names). A `:name` suffix (a function in a module) is split off and
    not checked; a `::Class::test_name` suffix gives the names, each of which
    the file must define."""
    found = []
    for token in _TOKEN.findall(text):
        if not token.startswith(PREFIXES):
            continue
        path, _, suffix = token.partition(":")
        names = suffix[1:].split("::") if suffix.startswith(":") else []
        found.append((path, [name for name in names if name]))
    return found


def map_faults(text, root=REPO):
    """What in a map text names nothing: a missing path, an empty glob, or a
    test function its file does not define."""
    faults = []
    for path, names in map_paths(text):
        if "*" in path:
            if not any(root.glob(path)):
                faults.append(f"{path}: the glob matches nothing")
            continue
        target = root / path
        if not target.exists():
            faults.append(f"{path}: no such file or directory")
            continue
        if path.startswith("tests/") and names and target.is_file():
            source = target.read_text()
            *classes, function = names
            for name in classes:
                if not re.search(rf"^class {re.escape(name)}\b", source, re.M):
                    faults.append(f"{path}::{name}: no such test class")
            if not re.search(rf"def {re.escape(function)}\(", source):
                faults.append(f"{path}::{function}: no such test function")
    return faults


def test_the_checker_reports_a_missing_path_and_a_missing_test(tmp_path):
    (tmp_path / "dw").mkdir()
    (tmp_path / "dw" / "real.py").write_text("")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_real.py").write_text(
        "class TestX:\n    def test_here(self):\n        pass\n"
    )
    text = (
        "| concept | `dw/real.py`: owner | `dw/gone.py` |\n"
        "| x | `tests/test_real.py::TestX::test_here` | "
        "`tests/test_real.py::test_absent` | `tests/test_real.py::TestY::test_here` |\n"
        "| y | `dw/*.py` | `dw_mcp/tools_*.py` | `pathlib` | `dw/real.py:fn` |\n"
    )
    assert map_faults(text, tmp_path) == [
        "dw/gone.py: no such file or directory",
        "tests/test_real.py::test_absent: no such test function",
        "tests/test_real.py::TestY: no such test class",
        "dw_mcp/tools_*.py: the glob matches nothing",
    ]


def test_every_path_the_architecture_map_names_exists():
    text = MAP.read_text()
    assert map_paths(text), "the map names no repo path"
    assert map_faults(text) == []
