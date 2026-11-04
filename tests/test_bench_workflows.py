"""The long-form bench's six scenes: every workflow validates, and every arm
and scene names files that exist (bench/driver.py loads them)."""

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import driver  # noqa: E402

WORKFLOWS = sorted((BENCH / "workflows").glob("s*.json"))
IDS = [
    "s1-static",
    "s2-arc",
    "s3-dialogue-walk",
    "s4-montage",
    "s5-ltx-action",
    "s6-narration",
]


def _errors(path):
    from dw.workflow import Workflow

    definition = json.loads(path.read_text(encoding="utf-8"))
    return Workflow(definition, None, str(path)).validation_errors({})


def test_the_six_scene_workflows_exist():
    assert [p.stem for p in WORKFLOWS] == IDS


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.stem)
def test_every_bench_workflow_validates(path):
    assert _errors(path) == []


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.stem)
def test_chain_workflows_expose_the_arm_knobs(path):
    variables = json.loads(path.read_text(encoding="utf-8"))["variables"]
    assert "seed" in variables
    if path.stem in ("s1-static", "s2-arc", "s3-dialogue-walk"):
        for knob in (
            "continuity",
            "guide_frames",
            "segments",
            "anchor",
            "guide_noise",
            "width",
            "height",
            "num_inference_steps",
            "weights_dtype",
            "lora_model_name",
            "attention_backend",
        ):
            assert knob in variables, knob


def test_every_arm_names_a_scene_and_a_workflow_that_exist():
    arms = driver.load_arms(BENCH / "arms" / "baseline.json")
    scenes = driver.load_scenes(BENCH / "scenes.json")
    assert sorted(scenes) == IDS
    assert sorted(a.scene for a in arms) == IDS
    for arm in arms:
        assert arm.scene in scenes
        assert (BENCH / "workflows" / arm.workflow).is_file()
        assert arm.seed == 1166


def test_scenes_are_well_formed():
    for scene in driver.load_scenes(BENCH / "scenes.json").values():
        x, y, width, height = scene.crop  # the media routes' [x, y, w, h]
        assert x >= 0 and y >= 0 and width >= 128 and height >= 128
        assert len(scene.moments) == 3
        assert all(m.startswith("frame:") for m in scene.moments)
        frames = [int(m.split(":")[1]) for m in scene.moments]
        assert frames == sorted(frames)


def test_s2_reproduces_the_1175_arc_crop():
    scenes = driver.load_scenes(BENCH / "scenes.json")
    assert scenes["s2-arc"].crop == [0, 0, 240, 150]


def test_a_workflow_declaring_prompts_references_it():
    for path in sorted((BENCH / "workflows").glob("*.json")):
        definition = json.loads(path.read_text())
        if "prompts" in (definition.get("variables") or {}):
            steps = json.dumps(definition.get("steps"))
            assert "variable:prompts" in steps, path.name


ARM_FILES = sorted((BENCH / "arms").glob("*.json"))


@pytest.mark.parametrize("path", ARM_FILES, ids=lambda p: p.stem)
def test_every_arm_file_loads_and_each_arm_validates(path):
    """Each arm names a scene and workflow that exist, passes only variables
    its workflow declares, and validates with those arguments."""
    from dw.workflow import Workflow

    scenes = driver.load_scenes(BENCH / "scenes.json")
    arms = driver.load_arms(path)
    assert arms
    names = [a.name for a in arms]
    assert len(names) == len(set(names)), "arm names must be unique"
    for arm in arms:
        assert arm.scene in scenes, arm.name
        workflow_path = BENCH / "workflows" / arm.workflow
        definition = json.loads(workflow_path.read_text(encoding="utf-8"))
        undeclared = set(arm.arguments) - set(definition["variables"])
        assert not undeclared, (arm.name, sorted(undeclared))
        errors = Workflow(definition, None, str(workflow_path)).validation_errors(
            arguments=dict(arm.arguments)
        )
        assert errors == [], (arm.name, errors)


def test_the_counter_round_arms_exist():
    controls = driver.load_arms(BENCH / "arms" / "counters-controls.json")
    round1 = driver.load_arms(BENCH / "arms" / "counters-round1.json")
    assert {a.seed for a in controls} == {1166, 2}
    assert {a.arguments["continuity"] for a in controls} == {"guide", "last_frame"}
    assert all(a.arguments["segments"] == 6 for a in controls + round1)
