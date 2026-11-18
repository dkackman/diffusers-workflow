"""The track_people backend A/B bench (#1397 stage B, #1405): the workflow
validates with the examples root pinned, as the bench server pins it, and the
arms pair up per fixture - one grounding-dino arm and one sam3 arm on the same
clip, each naming its threshold, differing only in backend and threshold."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
sys.path.insert(0, str(BENCH))
import driver  # noqa: E402

WORKFLOW = "track-people-backend-ab.json"
ARMS = BENCH / "arms" / WORKFLOW
FIXTURES = {
    "asset:qa-cast/ep65-shot-kf-priya-hal.mp4",
    "asset:qa-cast/ep62-episode.mp4",
    "asset:qa-cast/ep49-episode.mp4",
}
# track_people's own defaults (dw/tasks/person_track.py DEFAULT_THRESHOLDS).
THRESHOLDS = {"grounding-dino": 0.3, "sam3": 0.5}


@pytest.fixture
def examples_pinned(monkeypatch):
    from dw.library import LIBRARY_PATH_ENV_VARS, WORKFLOWS_KIND

    monkeypatch.setenv(LIBRARY_PATH_ENV_VARS[WORKFLOWS_KIND], str(ROOT / "workflows"))


def _definition():
    return json.loads((BENCH / "workflows" / WORKFLOW).read_text(encoding="utf-8"))


def test_the_ab_workflow_validates(examples_pinned, tmp_path):
    from dw.workflow import Workflow

    path = BENCH / "workflows" / WORKFLOW
    assert Workflow(_definition(), str(tmp_path), str(path)).validation_errors({}) == []


@pytest.mark.parametrize("arm_index", range(6))
def test_every_arm_validates_with_its_arguments(arm_index, examples_pinned, tmp_path):
    from dw.workflow import Workflow

    arm = driver.load_arms(ARMS)[arm_index]
    path = BENCH / "workflows" / arm.workflow
    arguments = {**arm.arguments, "seed": arm.seed}
    assert (
        Workflow(_definition(), str(tmp_path), str(path)).validation_errors(arguments)
        == []
    )


def test_the_workflow_runs_track_people_with_the_arm_variables():
    (step,) = _definition()["steps"]
    assert step["task"]["command"] == "track_people"
    arguments = step["task"]["arguments"]
    for name in ("video", "backend", "threshold", "prompt", "max_people"):
        assert arguments[name] == f"variable:{name}"


def test_the_thresholds_are_the_backends_own_defaults():
    from dw.tasks.person_track import DEFAULT_THRESHOLDS

    assert dict(DEFAULT_THRESHOLDS) == THRESHOLDS


def test_the_arms_pair_dino_and_sam3_on_each_fixture():
    scenes = driver.load_scenes(BENCH / "scenes.json")
    arms = driver.load_arms(ARMS)
    assert len(arms) == 6
    assert len({arm.name for arm in arms}) == 6
    pairs = {}
    for arm in arms:
        assert arm.scene in scenes
        assert arm.workflow == WORKFLOW
        assert set(arm.arguments) == {"video", "backend", "threshold"}, arm.name
        backend = arm.arguments["backend"]
        assert arm.name.startswith({"grounding-dino": "dino-", "sam3": "sam3-"}[backend])
        assert arm.arguments["threshold"] == THRESHOLDS[backend], arm.name
        pairs.setdefault(arm.arguments["video"], set()).add(backend)
    assert set(pairs) == FIXTURES
    assert all(backends == set(THRESHOLDS) for backends in pairs.values())
