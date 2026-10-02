"""docs/ARCHITECTURE.md names only things that exist.

The seam map sends an agent from a concept to the module that owns it and to
the test that enforces the rule. A map naming a deleted module, a renamed
function or a renamed test is worse than none, and nothing else would notice
the drift. So every backticked repo path in it must exist (a glob must
match). Every name written after a path - `path`: `name`, `name` - must be
defined or assigned in that file, or be a key of a `.json` file. Every
`tests/x.py::Class::test_name` must name a test defined in that class.
"""

import ast
import json
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
_NAME = r"`[A-Za-z_][\w.]*`"
# `path`: `name`, `name` - the names a path is followed by
_OWNED = re.compile(rf"`([^`\s]+)`:\s*({_NAME}(?:,\s*{_NAME})*)")


def map_paths(text):
    """Every backticked repo path in `text`, as (path, test segments, owned
    names). `::Class::test` segments come from the token itself; owned names
    are the identifiers written after it as `path`: `name`, `name`."""
    owned = {}
    for match in _OWNED.finditer(text):
        names = re.findall(r"`([^`]+)`", match.group(2))
        owned.setdefault(match.start(), names)
    found = []
    for match in _TOKEN.finditer(text):
        token = match.group(1)
        if not token.startswith(PREFIXES):
            continue
        path, _, suffix = token.partition("::")
        segments = [s for s in suffix.split("::") if s] if suffix else []
        found.append((path, segments, owned.get(match.start(), [])))
    return found


def _python_names(tree):
    """Every name a module defines or assigns, at any depth."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _defines_test(tree, segments):
    """Whether `Class::test` (or a module-level `test`) is defined there: the
    test inside that class, not anywhere in the file."""
    *classes, function = segments
    body = tree.body
    for name in classes:
        match = [n for n in body if isinstance(n, ast.ClassDef) and n.name == name]
        if not match:
            return False
        body = match[0].body
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function
        for n in body
    )


def _defines(target, name):
    """Whether a non-python file defines `name`: a key of a JSON object, or a
    function, class or binding in a script."""
    source = target.read_text()
    if target.suffix == ".json":
        return name in json.loads(source)
    pattern = rf"\b(?:function|class|const|let|var|def)\s+{re.escape(name)}\b"
    return re.search(pattern, source) is not None


def map_faults(text, root=REPO):
    """What in a map text names nothing: a missing path, an empty glob, a
    name its file does not define, or a test its file does not hold."""
    faults = []
    for path, segments, names in map_paths(text):
        if "*" in path:
            if not any(root.glob(path)):
                faults.append(f"{path}: the glob matches nothing")
            continue
        target = root / path
        if not target.exists():
            faults.append(f"{path}: no such file or directory")
            continue
        if not target.is_file() or not (segments or names):
            continue
        tree = ast.parse(target.read_text()) if target.suffix == ".py" else None
        if segments and not (tree and _defines_test(tree, segments)):
            faults.append(f"{path}::{'::'.join(segments)}: no such test")
        defined = _python_names(tree) if tree else None
        for name in names:
            parts = name.split(".")
            if defined is not None:
                ok = all(part in defined for part in parts)
            else:
                ok = all(_defines(target, part) for part in parts)
            if not ok:
                faults.append(f"{path}: {name}: not defined there")
    return faults


def test_the_checker_reports_what_names_nothing(tmp_path):
    (tmp_path / "dw").mkdir()
    (tmp_path / "dw" / "real.py").write_text(
        "LIMIT = 3\n\nclass Thing:\n    def method(self):\n        pass\n"
    )
    (tmp_path / "base.json").write_text('{"import_cycles": 0}')
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_real.py").write_text(
        "class TestX:\n    def test_here(self):\n        pass\n\n"
        "def test_top():\n    pass\n"
    )
    text = (
        "| a | `dw/real.py`: `LIMIT`, `Thing.method` | `dw/gone.py` |\n"
        "| b | `dw/real.py`: `absent` | `tests/test_real.py::TestX::test_here` |\n"
        "| c | `tests/test_real.py::test_top` | `tests/test_real.py::test_absent` |\n"
        "| d | `tests/test_real.py::TestY::test_here` |"
        " `tests/test_real.py::test_here` |\n"
        "| e | `dw/*.py` | `dw_mcp/tools_*.py` | `pathlib` |\n"
    )
    assert map_faults(text, tmp_path) == [
        "dw/gone.py: no such file or directory",
        "dw/real.py: absent: not defined there",
        "tests/test_real.py::test_absent: no such test",
        "tests/test_real.py::TestY::test_here: no such test",
        "tests/test_real.py::test_here: no such test",
        "dw_mcp/tools_*.py: the glob matches nothing",
    ]


def test_every_ratchet_the_map_names_is_a_baseline_key():
    baseline = REPO / "docs" / "stabilization" / "baseline.json"
    keys = set(json.loads(baseline.read_text()))
    text = MAP.read_text()
    named = {
        name
        for path, _, names in map_paths(text)
        if path == "docs/stabilization/baseline.json"
        for name in names
    }
    assert named, "the map names no ratchet"
    assert named <= keys


def test_every_path_the_architecture_map_names_exists():
    text = MAP.read_text()
    assert map_paths(text), "the map names no repo path"
    assert map_faults(text) == []
