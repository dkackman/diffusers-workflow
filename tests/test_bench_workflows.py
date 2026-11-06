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
    # a scene's variants (s5-ltx-extend, ...) may sit beside it; each is
    # still validated below, but the six scenes themselves must all be here
    stems = {p.stem for p in WORKFLOWS}
    assert [i for i in IDS if i not in stems] == []


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
def test_every_arm_file_loads_and_each_arm_validates(path, monkeypatch, tmp_path):
    """Each arm names a scene and workflow that exist, passes only variables
    its workflow declares, and validates with those arguments - on the bench
    server's settings, which turn the experimental spikes' flag on and put
    the catalog on the search path (an arm composing a template by its
    catalog name resolves it there, and opens it under an output dir)."""
    from dw.library import LIBRARY_PATH_ENV_VARS, WORKFLOWS_KIND
    from dw.pipeline_processors.h3_rules import REFINE_WINDOWS_ENV_VAR
    from dw.workflow import Workflow

    monkeypatch.setenv(REFINE_WINDOWS_ENV_VAR, "1")
    monkeypatch.setenv(
        LIBRARY_PATH_ENV_VARS[WORKFLOWS_KIND], str(BENCH.parent / "workflows")
    )

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
        errors = Workflow(
            definition, str(tmp_path), str(workflow_path)
        ).validation_errors(arguments=dict(arm.arguments))
        assert errors == [], (arm.name, errors)


def test_the_counter_round_arms_exist():
    controls = driver.load_arms(BENCH / "arms" / "counters-controls.json")
    round1 = driver.load_arms(BENCH / "arms" / "counters-round1.json")
    assert {a.seed for a in controls} == {1166, 2}
    assert {a.arguments["continuity"] for a in controls} == {"guide", "last_frame"}
    assert all(a.arguments["segments"] == 6 for a in controls + round1)


H3_CHAINS = ["s1-static", "s2-arc", "s3-dialogue-walk", "lf-s1-static", "lf-s2-arc"]


def _chain_block(node):
    if isinstance(node, dict):
        if isinstance(node.get("chain"), dict):
            return node["chain"]
        node = list(node.values())
    if isinstance(node, list):
        for item in node:
            found = _chain_block(item)
            if found is not None:
                return found
    return None


@pytest.mark.parametrize("stem", H3_CHAINS)
def test_duration_seconds_is_an_off_by_default_chain_knob(stem):
    """A two-minute arm asks for seconds; left null, the chain keeps its
    segment count, so every earlier arm is unchanged."""
    definition = json.loads((BENCH / "workflows" / f"{stem}.json").read_text())
    assert definition["variables"]["duration_seconds"] is None
    chain = _chain_block(definition["steps"])
    assert chain["duration_seconds"] == "variable:duration_seconds"
    assert chain["segments"] == "variable:segments"


def _effective(arm):
    definition = json.loads((BENCH / "workflows" / arm.workflow).read_text())
    values = dict(definition["variables"])
    values.update(arm.arguments)
    values.pop("seed", None)
    return values


# Round 2: each arm, the control it is read against, and the only variables
# it may change from that control (seed aside: a seed-2 arm meets a seed-2
# control)
ROUND2 = {
    "counters-round2-confirm.json": {
        # the round-1 control again, past the step cache: the round-2
        # workflows gained a null knob, and the comparison needs the same
        # output from the same seed
        "ctl-guide-s1-s1166-rerun": ("ctl-guide-s1-s1166", set()),
        "r2-anchor25-s1-seed2": ("ctl-guide-s1-s2", {"anchor", "anchor_strength"}),
        "r2-anchor50-s1-seed2": ("ctl-guide-s1-s2", {"anchor", "anchor_strength"}),
        "r2-keystone-s1-seed2": ("ctl-guide-s1-s2", {"keystone"}),
        "r2-lf-lookhold-s1-seed2": ("ctl-lastframe-s1-s2", {"look_hold"}),
        "r2-lf-lookhold-s2-seed2": ("ctl-lastframe-s2-s2", {"look_hold"}),
    },
    "counters-round2-combos.json": {
        **{
            f"r2-anchor{s}-keystone-s1-{seed}": (
                f"ctl-guide-s1-{seed}",
                {"anchor", "anchor_strength", "keystone"},
            )
            for s in ("25", "50")
            for seed in ("s1166", "s2")
        },
        **{
            f"r2-lf-lookhold-anchor{s}-s1-{seed}": (
                f"ctl-lastframe-s1-{seed}",
                {"look_hold", "anchor", "anchor_strength"},
            )
            for s in ("25", "50")
            for seed in ("s1166", "s2")
        },
    },
    "counters-round2-long.json": {
        "ctl-guide-s1-120s": ("ctl-guide-s1-s1166", {"duration_seconds"}),
        "r2-anchor50-s1-120s": (
            "ctl-guide-s1-s1166",
            {"duration_seconds", "anchor", "anchor_strength"},
        ),
    },
}


@pytest.mark.parametrize("filename", sorted(ROUND2))
def test_round2_arms_differ_from_their_control_in_the_stated_counters_only(
    filename,
):
    controls = {
        a.name: a for a in driver.load_arms(BENCH / "arms" / "counters-controls.json")
    }
    arms = {a.name: a for a in driver.load_arms(BENCH / "arms" / filename)}
    assert set(arms) == set(ROUND2[filename])
    for name, (control_name, counters) in ROUND2[filename].items():
        arm, control = arms[name], controls[control_name]
        assert arm.scene == control.scene, name
        if "120s" not in name:
            assert arm.seed == control.seed, name
            assert arm.moments == control.moments, name
        mine, theirs = _effective(arm), _effective(control)
        changed = {k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)}
        assert changed == counters, (name, sorted(changed))


def test_round2_two_minute_arms_share_one_geometry():
    arms = driver.load_arms(BENCH / "arms" / "counters-round2-long.json")
    assert {a.arguments["duration_seconds"] for a in arms} == {120}
    assert len({tuple(a.moments) for a in arms}) == 1
    assert {a.seed for a in arms} == {1166}
