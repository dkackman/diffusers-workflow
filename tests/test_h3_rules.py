"""dw/pipeline_processors/h3_rules.py stays torch- and diffusers-free (#771), and
h3_hold and h3_guides diffusers-free until a block is asked for (#790).

`dw/__init__` imports torch on purpose, so the import runs in a fresh
interpreter with `dw` and `dw.pipeline_processors` registered as bare packages
on their real directories: every module `h3_rules` imports is loaded from its
own file, and nothing `dw/__init__` would pull in rides along.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

PROBE = """
import sys, types

for name, path in (("dw", {dw!r}), ("dw.pipeline_processors", {pp!r})):
    package = types.ModuleType(name)
    package.__path__ = [path]
    sys.modules[name] = package

import dw.pipeline_processors.{module}  # noqa: F401

print(",".join(sorted(m for m in ("torch", "diffusers") if m in sys.modules)))
"""


def _imported_heavy_modules(module="h3_rules"):
    probe = PROBE.format(
        dw=str(REPO / "dw"),
        pp=str(REPO / "dw" / "pipeline_processors"),
        module=module,
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


@pytest.mark.parametrize("module", ["h3_hold", "h3_guides"])
def test_the_block_owners_import_no_diffusers(module):
    # Their blocks are h3_hold_steps and h3_guide_steps, which import diffusers
    # when they load: validation imports h3_guides, so those wait for a block
    assert "diffusers" not in _imported_heavy_modules(module).split(",")
