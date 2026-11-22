"""#1545 (feature #1518 stage A): the Ref2VA environment-swap probe's three arms.

Each arm validates, conditions on exactly one video reference and no image
reference (no guides, no source_video), and the arms differ only in `prompt`
(and `seed` across seeds); C is A with its `retention_analysis` block removed,
so A against C isolates that block.
"""

import copy
import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench" / "workflows"
ARMS = {arm: BENCH / f"env-swap-{arm}.json" for arm in ("a", "b", "c")}
VIDEO_REF = "diffusers.modular_pipelines.minimax_h3.MiniMaxH3VideoReference"
IMAGE_REF = "diffusers.modular_pipelines.minimax_h3.MiniMaxH3ImageReference"


def _load(arm):
    return json.loads(ARMS[arm].read_text(encoding="utf-8"))


def _without(definition, *variables):
    stripped = copy.deepcopy(definition)
    for name in ("id", "summary", "description"):
        stripped.pop(name, None)
    for name in variables:
        stripped["variables"].pop(name, None)
    return stripped


def _drop_retention(prompt):
    """A's prompt minus the `retention_analysis:` header and its lines, through the blank line."""
    lines, skipping = [], False
    for line in prompt.split("\n"):
        if line == "retention_analysis:":
            skipping = True
        elif skipping:
            skipping = line != ""
        else:
            lines.append(line)
    return "\n".join(lines)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_validates(arm):
    from dw.workflow import Workflow

    definition = _load(arm)
    assert Workflow(definition, None, str(ARMS[arm])).validation_errors({}) == []


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_has_one_video_reference_and_nothing_else(arm):
    definition = _load(arm)
    (step,) = definition["steps"]
    pipeline = step["pipeline"]
    assert pipeline["from_pretrained_arguments"]["workflow"] == "ref2va"
    arguments = pipeline["arguments"]
    types = [r["reference_type"] for r in arguments["references"]]
    assert types == [VIDEO_REF]
    assert IMAGE_REF not in json.dumps(definition)
    assert "guides" not in arguments
    assert "source_video" not in arguments
    assert arguments["references"][0]["from_file"] == "variable:source"
    variables = definition["variables"]
    assert variables["source"] == "asset:bench/1518/env-swap-source.mp4"
    assert (variables["num_frames"], variables["width"], variables["height"]) == (
        124,
        960,
        544,
    )
    assert variables["num_inference_steps"] == 9
    assert variables["weights_dtype"] == "{int4}"
    assert (
        variables["lora_weight_name"]
        == "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"
    )


def test_the_arms_differ_only_in_prompt_and_seed():
    a = _without(_load("a"), "prompt", "seed")
    assert _without(_load("b"), "prompt", "seed") == a
    assert _without(_load("c"), "prompt", "seed") == a
    prompts = {arm: _load(arm)["variables"]["prompt"] for arm in ARMS}
    assert len(set(prompts.values())) == 3


def test_c_is_a_without_its_retention_analysis_block():
    a = _load("a")["variables"]["prompt"]
    c = _load("c")["variables"]["prompt"]
    assert "retention_analysis:" in a
    assert "retention_analysis" not in c
    assert c == _drop_retention(a)
    for line in (
        "<Video 1>: attribute_transfer",
        "environment: fully_replaced",
        "weather: fully_replaced",
        "lighting: partially_modified",
        "motion: fully_preserved",
        "camera: fully_preserved",
    ):
        assert line in a.split("\n")
        assert line not in c.split("\n")


def test_a_carries_the_edit_structure_and_b_is_one_plain_paragraph():
    a = _load("a")["variables"]["prompt"]
    for marker in (
        "subject_definitions:",
        "[reference generation] Edit <Video 1> by replacing the original environment and weather with",
        "detailed_description:",
        "CHANGE:",
        "PRESERVE:",
        "do not improvise or add movements",
        "Consistency:",
        "overall_soundscape:\npreserve the original paired soundtrack of <Video 1>.",
        "non_diegetic_music:\npreserve the original paired soundtrack of <Video 1>.",
    ):
        assert marker.lower() in a.lower(), marker
    b = _load("b")["variables"]["prompt"]
    assert "\n" not in b and ":" not in b and "<Video 1>" not in b
    assert "snow" in b
