"""MiniMax-H3's sigma shift and LoRA alpha are declared by the workflow.

Both are properties of the checkpoint being run, not of the engine. The base
scheduler ships `shift = 12.0`, which is the 544p figure; the 768p FL2VA turbo
LoRAs are trained at shift 6 and the 768p Ref2VA one at 12, and every one of
them records `alpha: 8` in its `__metadata__` at rank 128 while upstream runs
the 768p files at alpha 128. So a checkpoint swap that moved only
`lora_weight_name` would run a 768p LoRA on the 544p schedule at a sixteenth
of its trained strength - three silent quality failures, each costing a full
run to discover (#147).

Every number here is pinned to the diffusers symbol it derives from, the way
tests/test_variable_constraints.py pins the frame grid.
"""

import glob
import json
import os

import pytest

from tests.test_examples import REPO_ROOT

H3_MODEL = "MiniMaxAI/MiniMax-H3"

TEMPLATES = sorted(
    glob.glob(os.path.join(REPO_ROOT, "workflows", "templates", "minimax", "*.json"))
)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def h3_steps(definition):
    for step in definition.get("steps", []):
        pipeline = step.get("pipeline") or {}
        arguments = pipeline.get("from_pretrained_arguments", {})
        if arguments.get("model_name") == H3_MODEL:
            yield step, pipeline


H3_TEMPLATES = [p for p in TEMPLATES if any(h3_steps(load(p)))]


def test_the_family_was_found():
    assert len(H3_TEMPLATES) >= 15, "the H3 family moved - this sweep found almost none"


def test_the_schedulers_default_is_the_544p_figure():
    """What the workflow has to override, and why an unset shift is not safe
    for a 768p checkpoint."""
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    assert MiniMaxH3Scheduler().shift == 12.0
    assert MiniMaxH3Scheduler(shift=6.0).shift == 6.0


def test_the_scheduler_takes_the_shift_the_engine_sets():
    """load_and_configure_scheduler calls set_shift; a scheduler that lost it
    would make every `shift` in the catalog a silent no-op."""
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    scheduler = MiniMaxH3Scheduler()
    scheduler.set_shift(6.0)
    assert scheduler.shift == 6.0
    with pytest.raises(ValueError):
        scheduler.set_shift(0)


@pytest.mark.parametrize("path", H3_TEMPLATES)
def test_every_h3_step_declares_both_schedules(path):
    """H3 denoises video and audio against two schedulers in one transformer
    call, and only one of them is the one a few-step schedule has to lower."""
    definition = load(path)
    for step, pipeline in h3_steps(definition):
        where = f"{os.path.basename(path)}:{step['name']}"
        assert pipeline.get("scheduler", {}).get("shift") == "variable:video_shift", (
            where
        )
        assert (
            pipeline.get("audio_scheduler", {}).get("shift") == "variable:audio_shift"
        ), where
    # The three combinations upstream publishes: 12/3 for the 544p FL2VA and
    # the 768p Ref2VA checkpoints, 6/3 for the 768p FL2VA ones. A shift that
    # belongs to no checkpoint is the failure this catches.
    assert definition["variables"]["video_shift"] in (12.0, 6.0)
    assert definition["variables"]["audio_shift"] == 3.0


@pytest.mark.parametrize("path", H3_TEMPLATES)
def test_every_lora_takes_a_declared_alpha(path):
    """Left to the file, the 768p checkpoints load at alpha 8 over rank 128."""
    definition = load(path)
    loras = [
        lora
        for _, pipeline in h3_steps(definition)
        for lora in pipeline.get("loras", [])
    ]
    if not loras:
        pytest.skip("no adapter on this template")
    for lora in loras:
        assert lora["alpha"] == "variable:lora_alpha", os.path.basename(path)
    assert "lora_alpha" in definition["variables"]


@pytest.mark.parametrize("path", H3_TEMPLATES)
def test_the_reference_path_never_loads_an_fl2va_adapter(path):
    """Upstream: "Do not use an FL2VA LoRA checkpoint for [the ref] command
    unless it was specifically trained for `transformer_ref`." A ref2va step
    holds transformer_ref alone, so diffusers routes whatever is handed to it
    straight there - nothing downstream would complain."""
    definition = load(path)
    for step, pipeline in h3_steps(definition):
        if pipeline["from_pretrained_arguments"].get("workflow") != "ref2va":
            continue
        for lora in pipeline.get("loras", []):
            name = lora.get("weight_name", "")
            if name.startswith("variable:"):
                name = definition["variables"][name.removeprefix("variable:")]
            assert "ref2v" in name, (
                f"{os.path.basename(path)}:{step['name']} is a ref2va step "
                f"loading {name}, which is not distilled for transformer_ref"
            )


def test_a_step_count_is_one_more_than_the_adapters_nfe():
    """Why every turbo template says 9 for an 8-step LoRA.

    The scheduler counts sigma grid points and the terminal zero is one of
    them, so `num_inference_steps` is one higher than the NFE the checkpoint
    was distilled for. Reading upstream's `--inference-steps 8` literally
    would under-step every turbo template by one.
    """
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    scheduler = MiniMaxH3Scheduler()
    scheduler.set_timesteps(num_inference_steps=9, device="cpu")
    assert len(scheduler.timesteps) == 8


@pytest.mark.parametrize("path", H3_TEMPLATES)
def test_no_template_still_runs_the_base_model_at_twenty_steps(path):
    """The six reference templates ran 20 steps with no adapter, because the
    only turbo LoRA was distilled against the base transformer while they load
    the reference one. `minimax_h3_ref2v_turbo_8step_v1.0_768p` ended that
    (#149)."""
    definition = load(path)
    if not any(True for _ in h3_steps(definition)):
        pytest.skip("no H3 step")
    assert definition["variables"]["num_inference_steps"] != 20
