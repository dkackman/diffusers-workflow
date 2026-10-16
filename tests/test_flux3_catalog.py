"""The FLUX 3 Image Klein catalog: the upsampler builtins, the templates' settings
and their placement. The vendor values here are the ones the design states
(docs/superpowers/specs/2026-10-15-flux-3-klein-catalog-design.md)."""

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILTINS = os.path.join(ROOT, "dw", "workflows")
PUP = "black-forest-labs/flux-3-image-klein-pup"
PUP_PIN = "3f3edb52b2c55de0d49feab28598b68f7f8e4830"


def load(path):
    with open(path) as f:
        return json.load(f)


def generation_steps(definition):
    return [
        s
        for s in definition["steps"]
        if s.get("task", {}).get("command") == "text_generation"
    ]


@pytest.mark.parametrize(
    "name,key",
    [
        ("flux3_klein_upsample.json", "t2i"),
        ("flux3_klein_upsample_edit.json", "i2i"),
    ],
)
def test_each_upsampler_runs_the_pinned_pup_as_bfl_does(name, key):
    definition = load(os.path.join(BUILTINS, name))
    (step,) = generation_steps(definition)
    args = step["task"]["arguments"]
    assert args["model_name"] == PUP
    assert args["revision"] == PUP_PIN
    assert args["system_prompt"] == {
        "from_hub": {
            "repo": PUP,
            "file": "system_prompts.json",
            "revision": PUP_PIN,
            "key": key,
        }
    }
    assert args["chat_template_kwargs"] == {"enable_thinking": False}
    assert args["max_new_tokens"] == 3584


def test_the_edit_upsampler_wraps_the_instruction_in_bfl_s_scaffold():
    definition = load(os.path.join(BUILTINS, "flux3_klein_upsample_edit.json"))
    compose = next(
        s for s in definition["steps"] if s["task"]["command"] == "compose_text"
    )
    parts = compose["task"]["arguments"]["parts"]
    assert parts == [
        "Reference images follow in order (the first image, the second image, ...).\nUser request: ",
        "variable:instruction",
        "\nOutput the caption.",
    ]
    assert compose["task"]["arguments"]["separator"] == ""
    (step,) = generation_steps(definition)
    assert step["task"]["arguments"]["image"] == "variable:images"
    assert step["task"]["arguments"]["repetition_penalty"] == 1.0
    assert "bbfe317d" in definition["description"]
