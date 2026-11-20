"""#1506 (stage A of #1505): the masked-keystone bench's workflows.

Segment 2 of the keystone control, made an unchained fl2va regional edit: the
source is the keystone as one frame, then segment 1's 22-frame tail, and the
mask holds the source's first frames black (kept) on whole latent cells. These
tests rebuild that claim from the files, so the record's description stays true
of what ran.
"""

import copy
import json
import math
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parent.parent / "bench" / "workflows"
ARGUMENT = "experimental_keystone_attention"
PROBE = "masked-keystone-probe.json"
CONTROL = "masked-keystone-control-early.json"
ATTENTION = "masked-keystone-attention-early.json"
FILES = (PROBE, CONTROL, ATTENTION)
FPS = 24
NUM_FRAMES = 192
LATENT_FRAMES = 57  # #1322: 192 decoded frames on H3's 57-frame latent grid
MASKED = {
    "source_video": "previous_result:src02",
    "mask_video": "previous_result:mask02",
    "mask_grow": 0,
    "refine_strength": 1.0,
}


def _load(name):
    return json.loads((WORKFLOWS / name).read_text(encoding="utf-8"))


def _steps(workflow):
    return {s["name"]: s for s in workflow["steps"]}


def _frames(seconds):
    # stills_to_video's rule: each still is shown max(1, round(d * fps)) frames
    return max(1, round(seconds * FPS))


def _cell(i):
    # decoded frames latent frame i covers, as #1322 laid the grid out
    return (
        math.floor(i * NUM_FRAMES / LATENT_FRAMES),
        math.ceil((i + 1) * NUM_FRAMES / LATENT_FRAMES),
    )


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


@pytest.mark.parametrize("name", FILES)
def test_segment_2_is_the_keystone_probes_made_a_masked_regional_edit(name):
    noarg = _steps(_load("keystone-probe-noarg.json"))
    steps = _steps(_load(name))
    assert steps["tail01"] == noarg["tail01"]
    assert steps["aud02"] == noarg["aud02"]

    seg02 = copy.deepcopy(steps["seg02"])
    arguments = seg02["pipeline"]["arguments"]
    arguments.pop(ARGUMENT, None)
    for key, value in MASKED.items():
        assert arguments.pop(key) == value
    assert seg02 == noarg["seg02"]
    # unchained: no latents carried in, the guide is still the plain tail
    assert "latents" not in steps["seg02"]["pipeline"]["arguments"]
    assert steps["seg02"]["pipeline"]["arguments"]["guides"] == [
        {"video": "previous_result:tail01", "frame": 0}
    ]
    assert steps["seg02"]["pipeline"]["arguments"]["image"] == "variable:keystone"


@pytest.mark.parametrize("name", FILES)
def test_the_source_is_the_keystone_frame_then_the_tail(name):
    steps = _steps(_load(name))
    tail = steps["tail01"]["task"]
    assert tail["command"] == "trim_video"
    assert tail["arguments"] == {
        "video": "variable:seg01",
        "start_frame": 170,
        "num_frames": 22,
    }
    key = steps["key02"]["task"]
    assert key["command"] == "stills_to_video"
    assert key["arguments"]["images"] == ["variable:keystone"]
    assert _frames(key["arguments"]["durations"]) == 1
    src = steps["src02"]["task"]
    assert src["command"] == "concat_videos"
    assert src["arguments"]["videos"] == [
        "previous_result:key02",
        "previous_result:tail01",
    ]
    assert src["arguments"]["trim_frames"] == 0  # no frame dropped at the join


@pytest.mark.parametrize("name", FILES)
def test_the_masks_held_frames_end_on_a_latent_cell_boundary(name):
    steps = _steps(_load(name))
    for still, push in (("black", "blacks"), ("white", "whites")):
        task = steps[still]["task"]
        assert task["command"] == "grade"
        assert task["arguments"]["media"] == "variable:keystone"
        assert task["arguments"]["contrast"] == 0.0  # one flat grey
    # grade's grey 0.5 pushed by a full blacks/whites sits either side of 128
    assert steps["black"]["task"]["arguments"]["blacks"] == -1.0
    assert steps["white"]["task"]["arguments"]["whites"] == 1.0

    mask = steps["mask02"]["task"]
    assert mask["command"] == "stills_to_video"
    assert mask["arguments"]["images"] == [
        "previous_result:black",
        "previous_result:white",
    ]
    held, regenerated = (_frames(d) for d in mask["arguments"]["durations"])
    assert held + regenerated == NUM_FRAMES
    assert held == 22  # #1322's mask-22
    # whole latent cells: the first 6 cells sit inside the held frames, the 7th
    # reaches past them, so the pooled mask keeps exactly 6 latent frames
    kept = [i for i in range(LATENT_FRAMES) if _cell(i)[1] <= held]
    assert kept == list(range(6))
    # and the held region is within the source (keystone + 22-frame tail)
    assert held <= 1 + 22


def test_the_arms_differ_only_in_the_attention_argument():
    control, attention = _load(CONTROL), _load(ATTENTION)
    seg02 = _steps(attention)["seg02"]["pipeline"]["arguments"]
    assert seg02.pop(ARGUMENT) == {"balance": 20, "blocks": "30-32,34-37,39,42,44"}
    assert ARGUMENT not in _steps(control)["seg02"]["pipeline"]["arguments"]
    for key in ("id", "summary", "description"):
        control.pop(key)
        attention.pop(key)
    assert attention == control


def test_the_probe_is_the_control_without_the_join():
    probe, control = _load(PROBE), _load(CONTROL)
    assert ARGUMENT not in _steps(probe)["seg02"]["pipeline"]["arguments"]
    assert control["steps"][: len(probe["steps"])] == probe["steps"]
    assert [s["name"] for s in control["steps"][len(probe["steps"]) :]] == [
        "head",
        "take",
    ]


def test_the_join_drops_the_held_frames():
    control = _load(CONTROL)
    steps = _steps(control)
    head = steps["head"]["task"]["arguments"]
    assert head == {
        "video": "variable:seg01",
        "start_frame": 0,
        "num_frames": "variable:head_frames",
    }
    take = steps["take"]["task"]["arguments"]
    assert take["videos"] == ["previous_result:head", "previous_result:seg02"]
    assert take["trim_frames"] == "variable:held"
    # seg02 frame j >= 1 is seg01 frame 169 + j, so keeping seg01 through frame
    # 168 + held and dropping held frames from seg02 leaves no frame twice
    variables = control["variables"]
    assert variables["head_frames"] == 169 + variables["held"]
