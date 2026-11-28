"""#1809: a 2x IC-LoRA reference must reach the model at the size it was made.

`LTX2InContextPipeline` preprocesses its reference at height/factor x
width/factor, and diffusers' `VideoProcessor` floors that to a multiple of 32
without saying so. `upscale-clip` at 960x544 handed it a 480x272 reference,
which was squashed to 480x256 and so covered 512 of the 544 output rows.
"""

import glob
import json
import os

import numpy
import pytest
from diffusers.video_processor import VideoProcessor

from dw.media_types import AudioVideo
from dw.tasks.fit import fit_to_model, restore_to_source
from dw.variable_constraints import (
    apply_constraints,
    constraint_errors,
    effective,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# LTX-2.5's video VAE spatial compression, what the pipeline's processor floors to
VAE_SCALE = 32


def downscaled_references():
    """{template name: (definition, factor)} for every catalog workflow whose
    reference is encoded below the output size."""
    found = {}
    pattern = os.path.join(REPO_ROOT, "workflows", "**", "*.json")
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as handle:
            definition = json.load(handle)
        for step in definition.get("steps", []):
            arguments = step.get("pipeline", {}).get("arguments", {})
            factor = arguments.get("reference_downscale_factor", 1)
            if factor != 1:
                name = os.path.splitext(os.path.basename(path))[0]
                found[name] = (definition, factor)
    return found


TEMPLATES = downscaled_references()


def test_the_two_upscalers_are_found():
    assert {"upscale-clip", "generative-upscale"} <= set(TEMPLATES)


@pytest.mark.parametrize("name", sorted(TEMPLATES))
@pytest.mark.parametrize("dimension", ["width", "height"])
def test_every_accepted_size_halves_onto_the_vae_grid(name, dimension):
    definition, factor = TEMPLATES[name]
    rule = definition["variable_constraints"][dimension]

    assert rule["modulus"] % (VAE_SCALE * factor) == 0
    assert rule["remainder"] == 0
    assert definition["variables"][dimension] % (VAE_SCALE * factor) == 0


def test_upscale_clip_rounds_544_up_and_generative_upscale_refuses_it():
    upscale, _ = TEMPLATES["upscale-clip"]
    generative, _ = TEMPLATES["generative-upscale"]

    assert effective(544, upscale["variable_constraints"]["height"]) == 576
    assert constraint_errors(upscale, {"height": 544}, {"height"}) == []
    (error,) = constraint_errors(generative, {"height": 544}, {"height"})
    assert error["path"] == "arguments.height"


def test_a_run_asking_for_544_renders_576_and_says_so(monkeypatch):
    warned = []
    monkeypatch.setattr(
        "dw.events.emit_warning", lambda message, **fields: warned.append(fields)
    )
    upscale, _ = TEMPLATES["upscale-clip"]
    variables = {"width": 960, "height": 544}

    apply_constraints(upscale, variables)

    assert variables == {"width": 960, "height": 576}
    assert [(w["kind"], w["variable"], w["value"]) for w in warned] == [
        ("value_snapped", "height", 576)
    ]


@pytest.mark.parametrize("asked", [(960, 576), (960, 544)])
def test_a_480x272_source_keeps_its_composition(asked):
    """The real fit, the real reference preprocess and the real restore: the
    pipeline's 2x render stood in for by a nearest-neighbour 2x of what the
    model is actually given."""
    definition, factor = TEMPLATES["upscale-clip"]
    rules = definition["variable_constraints"]
    width, height = (
        effective(value, rules[name]) for value, name in zip(asked, ("width", "height"))
    )
    # A vertical ramp: a row's value says which source row it came from
    ramp = numpy.linspace(0.0, 1.0, 272, dtype=numpy.float32)
    source = numpy.broadcast_to(ramp[None, :, None, None], (9, 272, 480, 3)).copy()
    fitted = fit_to_model(
        AudioVideo(source, None, None, fps=24.0),
        width,
        height,
        9,
        downscale=factor,
    )

    processor = VideoProcessor(vae_scale_factor=VAE_SCALE, resample="bilinear")
    reference = processor.preprocess_video(
        numpy.asarray(fitted["video"]),
        height // factor,
        width // factor,
        resize_mode="crop",
    )
    assert reference.shape[-2:] == fitted["video"].shape[1:3]

    seen = (reference[0].permute(1, 2, 3, 0).numpy() + 1.0) / 2.0
    rendered = seen.repeat(factor, axis=1).repeat(factor, axis=2)
    restored = restore_to_source(
        AudioVideo(rendered, None, None, fps=24.0), fitted["fit"]
    ).frames

    assert restored.shape[1:3] == (544, 960)
    expected = numpy.linspace(0.0, 1.0, 544)
    # Within a source row: nothing is shifted or compressed
    assert numpy.abs(restored[0, :, 0, 0] - expected).max() < 1.0 / 272
