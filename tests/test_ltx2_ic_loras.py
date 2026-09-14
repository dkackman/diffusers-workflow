"""The LTX-2.5 IC-LoRA templates, pinned to what the vendor cards state.

#151 and #152. Each of these templates is a checkpoint plus a conditioning
recipe, and every number in the recipe is the card's: the reference
downscale factor, the trained bucket, the LoRA strength, and the caption
form. Getting one wrong produces something that validates, generates, and
is quietly worse - the failure mode hardest to notice - so they are pinned
here rather than trusted to a reading.

Sources, read from `lem` with its own Hugging Face token on 2026-09-14:
Lightricks/LTX-2.5-22b-IC-LoRA-{Ingredients,Deblur,Decompression}.
"""

import json
import os

import pytest

from tests.test_examples import REPO_ROOT

TEMPLATES = os.path.join(REPO_ROOT, "workflows", "templates", "ltx2")

# repo -> (weight file, reference downscale factor, trained width, height,
#          trained frames, lora strength)
CARDS = {
    "reference-sheet": (
        "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
        "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
        1,
        768,
        448,
        121,
        1.0,
    ),
    "restore-deblur": (
        "Lightricks/LTX-2.5-22b-IC-LoRA-Deblur",
        "ltx-2.5-22b-ic-lora-deblur-0.9.safetensors",
        1,
        960,
        544,
        121,
        1.0,
    ),
    "restore-decompression": (
        "Lightricks/LTX-2.5-22b-IC-LoRA-Decompression",
        "ltx-2.5-22b-ic-lora-decompression-0.9.safetensors",
        1,
        960,
        544,
        121,
        1.0,
    ),
}


def definition(name):
    with open(os.path.join(TEMPLATES, f"{name}.json"), encoding="utf-8") as handle:
        return json.load(handle)


def conditioned_step(name):
    """The step that drives the IC-LoRA - the last one, since the reference
    sheet template has a `loop_frames` step ahead of it."""
    return [
        step for step in definition(name)["steps"] if "pipeline" in step
    ][-1]


@pytest.mark.parametrize("name", sorted(CARDS))
class TestEachTemplateMatchesItsCard:
    def test_it_loads_the_checkpoint_the_card_names(self, name):
        repo, weight, *_ = CARDS[name]
        (lora,) = conditioned_step(name)["pipeline"]["loras"]

        assert lora["model_name"] == repo
        assert lora["weight_name"] == weight

    def test_the_strength_is_the_cards_default(self, name):
        *_, strength = CARDS[name]
        (lora,) = conditioned_step(name)["pipeline"]["loras"]
        variables = definition(name)["variables"]

        assert variables[lora["scale"].removeprefix("variable:")] == strength

    def test_the_reference_is_encoded_at_the_output_resolution(self, name):
        """Every one of these three states downscale factor 1 - unlike the
        spatial upscaler, whose reference is half size."""
        _repo, _weight, factor, *_ = CARDS[name]
        arguments = conditioned_step(name)["pipeline"]["arguments"]

        assert arguments["reference_downscale_factor"] == factor

    def test_the_defaults_are_the_trained_bucket(self, name):
        _repo, _weight, _factor, width, height, frames, _strength = CARDS[name]
        variables = definition(name)["variables"]

        assert (variables["width"], variables["height"]) == (width, height)
        assert variables["num_frames"] == frames
        assert variables["frame_rate"] == 24.0

    def test_it_conditions_through_the_in_context_pipeline(self, name):
        pipeline = conditioned_step(name)["pipeline"]

        assert pipeline["configuration"]["component_type"] == "LTX2InContextPipeline"
        (reference,) = pipeline["arguments"]["reference_conditions"]
        assert reference["reference_type"].endswith("LTX2ReferenceCondition")


class TestTheReferenceSheetIsHeldForTheWholeClip:
    """The Ingredients card: the sheet is supplied as a static video looped
    to the output's length and frame rate, and 'the reference static video
    must be >= 121 frames; shorter references break the reference-encoding
    bucket'."""

    def test_the_sheet_is_looped_into_a_static_video(self):
        steps = definition("reference-sheet")["steps"]

        assert steps[0]["task"]["command"] == "loop_frames"
        assert steps[0]["task"]["arguments"]["num_frames"] == (
            "variable:reference_frames"
        )
        (reference,) = steps[1]["pipeline"]["arguments"]["reference_conditions"]
        assert reference["from_arguments"]["frames"] == (
            "previous_result:static_sheet"
        )

    def test_the_bucket_floor_is_declared_rather_than_hoped_for(self):
        rule = definition("reference-sheet")["variable_constraints"][
            "reference_frames"
        ]

        assert rule["min_frames"] == 121

    def test_the_default_sheet_length_satisfies_it(self):
        variables = definition("reference-sheet")["variables"]

        assert variables["reference_frames"] >= 121


class TestTheRestorationTemplatesReadFootageTheyDidNotGenerate:
    """The point of both: a reference the workflow did not make. Every other
    IC-LoRA use in this catalog conditions on an earlier step's output."""

    @pytest.mark.parametrize("name", ["restore-deblur", "restore-decompression"])
    def test_the_reference_is_the_callers_own_clip(self, name):
        (reference,) = conditioned_step(name)["pipeline"]["arguments"][
            "reference_conditions"
        ]
        frames = reference["from_arguments"]["frames"]

        assert frames["media_type"] == "video"
        assert frames["location"] == "variable:source_video"
        assert definition(name)["variables"]["source_video"].startswith("asset:")

    @pytest.mark.parametrize(
        "name,marker",
        [
            ("restore-deblur", "DEBLUR"),
            ("restore-decompression", "ENHANCE QUALITY"),
        ],
    )
    def test_the_default_prompt_is_in_the_trained_form(self, name, marker):
        reference = definition(name)["variables"]["prompt"]
        assert reference.startswith("prompt:")
        path = os.path.join(
            REPO_ROOT, "prompts", f"{reference.removeprefix('prompt:')}.json"
        )
        with open(path, encoding="utf-8") as handle:
            text = json.load(handle)["text"]

        assert "Reference shows" in text
        assert marker in text
