"""dw/pipeline_processors/h3_rules.py stays torch- and diffusers-free (#771).

`dw/__init__` imports torch on purpose, so the import runs in a fresh
interpreter with `dw` and `dw.pipeline_processors` registered as bare packages
on their real directories: every module `h3_rules` imports is loaded from its
own file, and nothing `dw/__init__` would pull in rides along.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PROBE = """
import sys, types

for name, path in (("dw", {dw!r}), ("dw.pipeline_processors", {pp!r})):
    package = types.ModuleType(name)
    package.__path__ = [path]
    sys.modules[name] = package

import dw.pipeline_processors.h3_rules  # noqa: F401

print(",".join(sorted(m for m in ("torch", "diffusers") if m in sys.modules)))
"""


def _imported_heavy_modules():
    probe = PROBE.format(
        dw=str(REPO / "dw"), pp=str(REPO / "dw" / "pipeline_processors")
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_h3_rules_imports_neither_torch_nor_diffusers():
    assert _imported_heavy_modules() == ""
