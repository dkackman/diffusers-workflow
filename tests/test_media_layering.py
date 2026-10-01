"""The layering the media and DSP modules exist to keep (stage 3d).

`dw/media.py` is the one module that opens a media container, so a decode
is one `av.open` that a test can count and a patch can intercept; `dw/dsp.py`
is pure numerics with no engine behind it; and `dw/media.py` hands back
arrays and rates, never a step type. Each is read off the source tree rather
than counted, so a new module that opens a container fails here.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DW = ROOT / "dw"
SCANNED = (DW, ROOT / "dw_mcp")


def sources(skip=("community_pipelines",)):
    for base in SCANNED:
        for path in sorted(base.rglob("*.py")):
            if any(part in skip for part in path.relative_to(base).parts):
                continue
            yield path, ast.parse(path.read_text(encoding="utf-8"))


def imported_modules(tree, package):
    """Every module a tree imports, relative ones resolved against `package`
    (the dotted package the file sits in, e.g. "dw" or "dw.tasks")."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package.split(".")
            if node.level:
                base = base[: len(base) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            names.append(module)
            names.extend(f"{module}.{alias.name}" for alias in node.names)
    return names


def dotted(node):
    """`a.b.c` for a Name/Attribute chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


def av_open_lines(tree):
    """Line numbers of every call that opens a PyAV container, however the
    name was reached: `av.open`, `av.container.open`, an alias of `av` (or of
    `av.container`), or `open` imported from either."""
    roots = {"av"}  # names bound to the av package
    containers = {"av.container"}  # dotted prefixes bound to av.container
    openers = set()  # bare names bound to an av open function
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "av":
                    roots.add(alias.asname or "av")
                elif alias.name == "av.container":
                    if alias.asname:
                        containers.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            for alias in node.names:
                bound = alias.asname or alias.name
                if node.module == "av":
                    if alias.name == "open":
                        openers.add(bound)
                    elif alias.name == "container":
                        containers.add(bound)
                elif node.module == "av.container" and alias.name == "open":
                    openers.add(bound)
    containers |= {f"{root}.container" for root in roots}
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in openers:
            lines.append(node.lineno)
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            owner = dotted(func.value)
            if owner in roots or owner in containers:
                lines.append(node.lineno)
    return lines


def test_av_open_appears_only_in_media():
    offenders = []
    for path, tree in sources():
        relative = path.relative_to(ROOT).as_posix()
        if relative == "dw/media.py":
            continue
        offenders.extend(f"{relative}:{line}" for line in av_open_lines(tree))
    assert offenders == [], f"av.open outside dw/media.py: {offenders}"


@pytest.mark.parametrize(
    "source",
    [
        "import av\nav.open(p)",
        "import av as pyav\npyav.open(p)",
        "from av import open\nopen(p)",
        "from av import open as opener\nopener(p)",
        "import av\nav.container.open(p)",
        "from av import container\ncontainer.open(p)",
        "import av as pyav\npyav.container.open(p)",
        "from av.container import open as o\no(p)",
    ],
)
def test_the_detector_catches_every_spelling(source):
    assert av_open_lines(ast.parse(source)) == [2]


@pytest.mark.parametrize(
    "source",
    ["open(p)", "import io\nio.open(p)", "import av\nav.AudioFrame.from_ndarray(a)"],
)
def test_the_detector_ignores_other_opens(source):
    assert av_open_lines(ast.parse(source)) == []


def test_dsp_imports_nothing_from_dw():
    tree = ast.parse((DW / "dsp.py").read_text(encoding="utf-8"))
    engine = [
        name
        for name in imported_modules(tree, "dw")
        if name == "dw" or name.startswith("dw.")
    ]
    assert engine == [], f"dw/dsp.py imports the engine: {engine}"


def test_media_imports_no_step_types():
    tree = ast.parse((DW / "media.py").read_text(encoding="utf-8"))
    forbidden = [
        name
        for name in imported_modules(tree, "dw")
        if name in ("dw.tasks", "dw.result")
        or name.startswith("dw.tasks.")
        or name.startswith("dw.result.")
    ]
    assert forbidden == [], f"dw/media.py imports a step type: {forbidden}"
