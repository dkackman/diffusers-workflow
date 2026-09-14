"""LTX-2.5's diffusion decoder, driven from a template rather than unused.

The family decoded every clip with the convolutional VAE and never touched
`LTX2VideoDiffusionDecodePipeline`, which upstream treats as the production
path. `templates/ltx2/diffusion-decode` exists to answer whether it is worth
its cost here, which needs a run - what is pinned here is that the template
drives the pipeline correctly, since a template built on a misread signature
would cost a 22B download to find out (#153).
"""

import inspect
import json
import os

import pytest

from tests.test_examples import REPO_ROOT

PATH = os.path.join(
    REPO_ROOT, "workflows", "templates", "ltx2", "diffusion-decode.json"
)
BASELINE = os.path.join(
    REPO_ROOT, "workflows", "templates", "ltx2", "text-to-video.json"
)


@pytest.fixture
def definition():
    with open(PATH, encoding="utf-8") as handle:
        return json.load(handle)


def test_every_argument_is_one_the_pipeline_takes(definition):
    from diffusers import LTX2Pipeline, LTX2VideoDiffusionDecodePipeline

    generate, decode = definition["steps"]
    for step, pipeline_class in (
        (generate, LTX2Pipeline),
        (decode, LTX2VideoDiffusionDecodePipeline),
    ):
        accepted = set(inspect.signature(pipeline_class.__call__).parameters)
        unknown = [a for a in step["pipeline"]["arguments"] if a not in accepted]
        assert not unknown, f"{step['name']} passes {unknown}"


def test_the_latents_are_denormalized_exactly_once(definition):
    """LTX2Pipeline applies the latent statistics on its way out at
    `output_type='latent'`, so the decoder must not apply them again."""
    decode = definition["steps"][1]["pipeline"]["arguments"]
    assert decode["denormalize"] is False

    from diffusers import LTX2Pipeline

    source = inspect.getsource(LTX2Pipeline.__call__)
    marker = source.index('output_type == "latent"')
    assert "_denormalize_latents" in source[marker : marker + 400]


def test_the_decoder_reads_what_the_generator_returns(definition):
    """`frames` holds the latents at `output_type='latent'`, and `latents` is
    what the decoder takes - the two names do not match by accident."""
    from diffusers.pipelines.ltx2.pipeline_output import LTX2PipelineOutput

    generate, decode = definition["steps"]
    assert generate["pipeline"]["arguments"]["output_type"] == "{latent}"
    assert decode["pipeline"]["arguments"]["latents"] == "previous_result:latents.frames"
    assert "frames" in LTX2PipelineOutput.__dataclass_fields__


def test_the_comparison_is_against_the_baseline_and_only_the_decode_differs(
    definition,
):
    """The experiment is only an experiment if one thing changed."""
    with open(BASELINE, encoding="utf-8") as handle:
        baseline = json.load(handle)

    assert definition["seed"] == baseline["seed"]
    for name in ("prompt", "width", "height", "num_frames", "frame_rate"):
        assert definition["variables"][name] == baseline["variables"][name], name

    generated = dict(definition["steps"][0]["pipeline"]["arguments"])
    reference = dict(baseline["steps"][0]["pipeline"]["arguments"])
    assert generated.pop("output_type") == "{latent}"
    assert reference.pop("output_type") == "{np}"
    assert generated == reference


def test_the_transformer_is_released_before_the_decoder_loads(definition):
    """A second model on top of a 22B transformer, on a 24 GB card."""
    assert definition["steps"][0]["release_pipeline"] is True


def test_no_cost_is_claimed(definition):
    """Nothing has measured it; `cost` is measured and never derived."""
    assert "cost" not in definition
