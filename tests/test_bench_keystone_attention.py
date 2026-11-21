"""#1394 (stage B of #1391): the keystone-attention bench's workflows.

The control is seam-restore-control.json (the #1325 control) with two changes,
and the attention arm is the control plus one argument. These tests rebuild that
claim from the files, so the record's "two differences" stays true of what ran.
"""

import copy
import json
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parent.parent / "bench" / "workflows"
ARGUMENT = "experimental_keystone_attention"
KEYS = ("key02", "key03", "key04", "key05", "key06")
FILES = (
    "keystone-attention-control.json",
    "keystone-attention.json",
    "keystone-attention-control-early.json",
    "keystone-attention-early.json",
    "keystone-probe-seg01.json",
    "keystone-probe.json",
    "keystone-probe-noarg.json",
)


def _load(name):
    return json.loads((WORKFLOWS / name).read_text(encoding="utf-8"))


def _steps(workflow):
    return {s["name"]: s for s in workflow["steps"]}


def _transformer(step):
    return step["pipeline"]["configuration"]["components"]["transformer"]


def _pipelines(workflow):
    return [s for s in workflow["steps"] if "pipeline" in s]


@pytest.fixture(autouse=True)
def _experimental(monkeypatch):
    from dw.pipeline_processors.h3_rules import REFINE_WINDOWS_ENV_VAR

    monkeypatch.setenv(REFINE_WINDOWS_ENV_VAR, "1")


@pytest.mark.parametrize("name", FILES)
def test_the_workflow_validates(name):
    from dw.workflow import Workflow

    path = WORKFLOWS / name
    errors = Workflow(_load(name), None, str(path)).validation_errors({})
    assert errors == []


def test_the_control_is_seam_restore_with_a_plain_keystone_on_dense_attention():
    source = _load("seam-restore-control.json")
    control = _load("keystone-attention-control.json")

    # undo the two changes on a copy of the control; what is left is the source
    undone = copy.deepcopy(control)
    steps = undone["steps"]
    assert [s["name"] for s in steps].count("key01") == 1
    key01 = _steps(undone)["key01"]
    assert key01["task"] == {
        "command": "get_first_frame",
        "arguments": {"video": "previous_result:seg01"},
    }
    steps.remove(key01)
    for step in _pipelines(undone):
        transformer = _transformer(step)
        assert "attention_backend" not in transformer
        assert "sparse_tau" not in transformer and "sparse_topk" not in transformer
        if step["name"] != "seg01":
            assert step["pipeline"]["arguments"]["image"] == "previous_result:key01"
    for step in _pipelines(source):
        _transformer(step).pop("attention_backend")
        _transformer(step).pop("sparse_tau")
        _transformer(step).pop("sparse_topk")
        if step["name"] != "seg01":
            del step["pipeline"]["arguments"]["image"]
    for step in _pipelines(undone):
        if step["name"] != "seg01":
            del step["pipeline"]["arguments"]["image"]
    source["steps"] = [s for s in source["steps"] if s["name"] not in KEYS]
    for key in ("id", "summary", "description"):
        source.pop(key)
        undone.pop(key)
    assert undone == source


def test_the_carried_tail_is_still_each_segments_guide():
    steps = _steps(_load("keystone-attention-control.json"))
    for k in range(2, 7):
        guides = steps[f"seg0{k}"]["pipeline"]["arguments"]["guides"]
        assert guides == [{"video": f"previous_result:tail0{k - 1}", "frame": 0}]


@pytest.mark.parametrize(
    "arm, control",
    [
        ("keystone-attention.json", "keystone-attention-control.json"),
        ("keystone-attention-early.json", "keystone-attention-control-early.json"),
    ],
)
def test_the_attention_arm_is_the_control_plus_the_argument_on_segments_2_on(
    arm, control
):
    arm, control = _load(arm), _load(control)
    seen = []
    for step in _pipelines(arm):
        arguments = step["pipeline"]["arguments"]
        if step["name"] == "seg01":
            assert ARGUMENT not in arguments  # no guide to re-split against
            continue
        value = arguments.pop(ARGUMENT)
        assert value["balance"] == 20
        seen.append(value["blocks"])
    assert seen and len(set(seen)) == 1  # one block set for the whole arm
    for key in ("id", "summary", "description"):
        arm.pop(key)
        control.pop(key)
    assert arm == control


def test_the_early_jobs_are_the_full_ones_cut_after_seg02():
    full = _load("keystone-attention-control.json")
    early = _load("keystone-attention-control-early.json")
    names = [s["name"] for s in early["steps"]]
    assert names == [s["name"] for s in full["steps"]][: names.index("seg02") + 1] + [
        "take"
    ]
    assert early["steps"][:-1] == full["steps"][: len(names) - 1]
    take = _steps(early)["take"]["task"]
    assert take["arguments"]["videos"] == [
        "previous_result:seg01",
        "previous_result:seg02",
    ]
    full_take = dict(_steps(full)["take"]["task"]["arguments"])
    full_take.pop("videos")
    assert {k: v for k, v in take["arguments"].items() if k != "videos"} == full_take


def test_the_probe_segments_are_the_controls_with_their_inputs_as_variables():
    control = _steps(_load("keystone-attention-control.json"))
    seg01 = _steps(_load("keystone-probe-seg01.json"))
    assert seg01["seg01"] == control["seg01"]
    assert seg01["aud01"] == control["aud01"]

    for name, with_argument in (
        ("keystone-probe.json", True),
        ("keystone-probe-noarg.json", False),
    ):
        probe = _load(name)
        steps = _steps(probe)
        assert steps["tail01"]["task"]["arguments"]["video"] == "variable:seg01"
        assert steps["aud02"] == control["aud02"]
        seg02 = copy.deepcopy(steps["seg02"])
        arguments = seg02["pipeline"]["arguments"]
        assert arguments.pop("image") == "variable:keystone"
        if with_argument:
            assert arguments.pop(ARGUMENT) == {
                "balance": 20,
                "blocks": "variable:blocks",
            }
            assert "blocks" in probe["variables"]
        else:
            assert ARGUMENT not in arguments
        expected = copy.deepcopy(control["seg02"])
        del expected["pipeline"]["arguments"]["image"]
        assert seg02 == expected


def test_the_test_keystones_are_frame_0_brightened_and_sharpened():
    steps = _steps(_load("keystone-probe-seg01.json"))
    assert steps["normal"]["task"]["command"] == "get_first_frame"
    bright = steps["bright"]["task"]
    assert bright["command"] == "grade"
    assert 2 ** bright["arguments"]["exposure"] == pytest.approx(1.3, abs=1e-3)
    sharp = steps["sharp"]["task"]
    assert sharp["command"] == "sharpen"
    assert sharp["arguments"]["amount"] == 0.36 and sharp["arguments"]["radius"] == 2.0
    for name in ("bright", "sharp"):
        assert steps[name]["task"]["arguments"]["media"] == "previous_result:normal"
