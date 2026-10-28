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


# refine_tiles (#1043): the pure rules, on a canvas with a shape and nothing else


class _Canvas:
    """A `latents` stand-in: only the shape h3_rules reads, no torch."""

    def __init__(self, height, width):
        self.shape = (1, 16, 8, height // 16, width // 16)


def _tiled(tiles, **overrides):
    arguments = {
        "refine_strength": 0.2,
        "latents": _Canvas(1088, 1920),
        "hold_audio": "previous_result:base.audio",
        "num_inference_steps": 5,
        "refine_tiles": tiles,
    }
    arguments.update(overrides)
    return {k: v for k, v in arguments.items() if v is not None}


def _problems(tiles, **overrides):
    from dw.pipeline_processors.h3_rules import refine_problems

    return refine_problems(_tiled(tiles, **overrides))


@pytest.mark.parametrize(
    "tiles",
    [
        {"width": 1344, "height": 768, "overlap": 256},
        {"width": 768, "height": 1088, "overlap": 192},
        {"width": 768, "height": 768, "overlap": 352},
    ],
)
def test_a_tile_inside_the_limit_is_clean(tiles):
    assert _problems(tiles) == []


def test_a_tile_over_the_latent_canvas_is_refused():
    problems = _problems(
        {"width": 1344, "height": 768, "overlap": 256}, latents=_Canvas(704, 1280)
    )
    assert len(problems) == 1
    assert "1280x704 canvas" in problems[0]


def test_a_missing_strength_is_the_only_problem():
    problems = _problems(
        {"width": 1344, "height": 768, "overlap": 256}, refine_strength=None
    )
    assert len(problems) == 1
    assert problems[0].startswith("refine_tiles tiles a refine_strength")
