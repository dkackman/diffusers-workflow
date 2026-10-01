"""The layering the media and DSP modules exist to keep (stage 3d).

`dw/media.py` is the one module that opens a media container, so a decode
is one `av.open` that a test can count and a patch can intercept; `dw/dsp.py`
is pure numerics with no engine behind it; and `dw/media.py` hands back
arrays and rates, never a step type. Each is read off the source tree rather
than counted, so a new module that opens a container fails here.
"""

import ast
from pathlib import Path

DW = Path(__file__).resolve().parent.parent / "dw"


def sources(skip=("community_pipelines",)):
    for path in sorted(DW.rglob("*.py")):
        if any(part in skip for part in path.relative_to(DW).parts):
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


def test_av_open_appears_only_in_media():
    offenders = []
    for path, tree in sources():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "open"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "av"
            ):
                offenders.append(f"{path.relative_to(DW)}:{node.lineno}")
    offenders = [o for o in offenders if not o.startswith("media.py:")]
    assert offenders == [], f"av.open outside dw/media.py: {offenders}"


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
