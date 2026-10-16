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


TEMPLATES = os.path.join(ROOT, "workflows", "templates", "flux3")
KLEIN = "black-forest-labs/flux-3-image-klein"


def klein_steps(definition):
    return [
        s
        for s in definition["steps"]
        if s.get("pipeline", {}).get("from_pretrained_arguments", {}).get("model_name")
        == KLEIN
    ]


def flux3_definitions():
    paths = [
        os.path.join(TEMPLATES, n)
        for n in sorted(os.listdir(TEMPLATES))
        if n.endswith(".json")
    ]
    paths.append(os.path.join(ROOT, "workflows", "models", "flux3-klein.json"))
    return paths


@pytest.mark.parametrize("path", flux3_definitions())
def test_every_flux3_pipeline_step_places_every_component(path):
    steps = klein_steps(load(path))
    assert steps, f"{path} runs no Klein step"
    for step in steps:
        components = step["pipeline"]["configuration"]["components"]
        for name in ("transformer", "text_encoder", "vae"):
            assert components.get(name, {}).get("device") == "cuda", (path, name)
        assert components["text_encoder"].get("residency") is None


@pytest.mark.parametrize("path", flux3_definitions())
def test_every_klein_step_runs_the_distilled_schedule_without_guidance(path):
    for step in klein_steps(load(path)):
        args = step["pipeline"]["arguments"]
        assert args["num_inference_steps"] == 16
        assert "guidance_scale" not in args and "negative_prompt" not in args
        assert args["output"] == "images"


@pytest.mark.parametrize("path", flux3_definitions())
def test_size_variables_are_held_to_the_16_pixel_grid(path):
    definition = load(path)
    constraints = definition.get("variable_constraints", {})
    for name in ("width", "height"):
        if name in definition.get("variables", {}):
            assert (
                constraints[name]["modulus"] == 16
                and constraints[name].get("remainder", 0) == 0
            )


def test_the_16_step_default_is_the_distilled_pipeline_s():
    from diffusers.modular_pipelines.flux3.modular_blocks_flux3_distilled import (
        Flux3DistilledCoreDenoiseStep,
    )

    defaults = {p.name: p.default for p in Flux3DistilledCoreDenoiseStep().inputs}
    assert defaults["num_inference_steps"] == 16


def test_the_16_pixel_grid_is_the_vae_scale_times_the_2x2_packing():
    import inspect

    from diffusers.modular_pipelines.flux3 import before_denoise

    # Source check: the multiple is derived from the loaded VAE, so it cannot
    # be read without weights; 8 (VAE) * 2 (packing) = 16 is the vendor's.
    assert "vae_scale_factor * 2" in inspect.getsource(before_denoise)
