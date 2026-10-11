"""Catalog defaults that silently multiply a run's cost (speed audit 2026-10-10)."""

import glob
import json
import os

import pytest

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


@pytest.mark.parametrize("path", MINIMAX, ids=os.path.basename)
def test_turbo_schedules_carry_no_block_cache(path):
    definition = _load(path)
    weight = _default(definition, "lora_weight_name") or ""
    steps = _default(definition, "num_inference_steps")
    if "turbo" not in weight or steps is None or steps > 9:
        pytest.skip("not a turbo schedule")
    for step in _pipeline_steps(definition):
        configuration = step["pipeline"].get("configuration", {})
        assert "cache" not in configuration, (
            f"{os.path.basename(path)} step '{step['name']}' carries a cache block on a "
            f"{steps}-step turbo schedule; RECIPES_24GB.md says it never skips there"
        )


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
