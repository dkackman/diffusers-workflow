"""Catalog defaults that silently multiply a run's cost (speed audit 2026-10-10)."""

import glob
import json
import os

import pytest

from dw.step_passes import step_list_passes

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = sorted(
    glob.glob(
        os.path.join(REPO_ROOT, "workflows", "templates", "**", "*.json"),
        recursive=True,
    )
)


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _default(definition, name):
    variable = definition.get("variables", {}).get(name)
    return variable.get("default") if isinstance(variable, dict) else variable


def _pipeline_steps(definition):
    for step in definition.get("steps", []):
        if "pipeline" in step:
            yield step


MINIMAX = [p for p in TEMPLATES if os.sep + "minimax" + os.sep in p]
# H3 and the families over its weights: Pika H3 Flex runs H3 under a fixed list
H3_FAMILIES = MINIMAX + [p for p in TEMPLATES if os.sep + "pika-h3" + os.sep in p]


def _few_step(definition, step):
    """The step's schedule is few-step: a turbo LoRA at 9 steps or fewer, or a
    pipeline whose stated `denoising_step_list` fixes the passes (#1316)."""
    passes = step_list_passes(step["pipeline"])
    if passes is not None:
        return passes
    weight = _default(definition, "lora_weight_name") or ""
    steps = _default(definition, "num_inference_steps")
    if "turbo" in weight and steps is not None and steps <= 9:
        return steps
    return None


def _template_id(path):
    return os.path.relpath(path, os.path.join(REPO_ROOT, "workflows", "templates"))


@pytest.mark.parametrize("path", H3_FAMILIES, ids=_template_id)
def test_few_step_schedules_carry_no_block_cache(path):
    definition = _load(path)
    few = [
        (step, steps)
        for step in _pipeline_steps(definition)
        if (steps := _few_step(definition, step)) is not None
    ]
    if not few:
        pytest.skip("not a few-step schedule: turbo or a step-list pipeline")
    for step, steps in few:
        configuration = step["pipeline"].get("configuration", {})
        # A `variable:cache` defaulting to null is no cache: the effective
        # default is what counts, not whether the key is written (#865)
        cache = configuration.get("cache")
        if isinstance(cache, str) and cache.startswith("variable:"):
            cache = _default(definition, cache[len("variable:") :])
        assert cache is None, (
            f"{_template_id(path)} step '{step['name']}' carries a cache block on a "
            f"{steps}-step few-step schedule; RECIPES_24GB.md says it never skips there"
        )


def test_the_pika_templates_are_few_step():
    paths = [p for p in H3_FAMILIES if os.sep + "pika-h3" + os.sep in p]
    assert len(paths) == 3
    for path in paths:
        definition = _load(path)
        assert [_few_step(definition, s) for s in _pipeline_steps(definition)] == [6]


def test_compose_workflows_video_steps_match_the_turbo_default():
    compose = _load(
        os.path.join(REPO_ROOT, "workflows", "templates", "compose-workflows.json")
    )
    i2v = _load(
        os.path.join(
            REPO_ROOT, "workflows", "templates", "minimax", "image-to-video.json"
        )
    )
    assert _default(compose, "video_num_inference_steps") == _default(
        i2v, "num_inference_steps"
    )


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: os.path.relpath(p, REPO_ROOT))
def test_schnell_runs_its_distilled_schedule(path):
    definition = _load(path)
    for step in _pipeline_steps(definition):
        pipeline = step["pipeline"]
        model = pipeline.get("from_pretrained_arguments", {}).get("model_name", "")
        if "FLUX.1-schnell" not in str(model):
            continue
        arguments = pipeline.get("arguments", {})
        assert arguments.get("num_inference_steps", 4) <= 4, path
        assert arguments.get("guidance_scale", 0.0) == 0.0, path
