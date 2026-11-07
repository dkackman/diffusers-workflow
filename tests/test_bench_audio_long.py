"""The S6 long-audio bench arms (#1199 Task 22): the workflows compose the
shipped templates by catalog name and validate with the examples root on the
search path, as the bench server pins it, and every arm names a scene and a
workflow that exist."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
sys.path.insert(0, str(BENCH))
import driver  # noqa: E402

AUDIO_WORKFLOWS = [
    "narrator-reference-s6.json",
    "narration-long-s6.json",
    "music-long-s6.json",
]


@pytest.fixture
def examples_pinned(monkeypatch):
    from dw.library import LIBRARY_PATH_ENV_VARS, WORKFLOWS_KIND

    monkeypatch.setenv(LIBRARY_PATH_ENV_VARS[WORKFLOWS_KIND], str(ROOT / "workflows"))


@pytest.mark.parametrize("name", AUDIO_WORKFLOWS)
def test_the_audio_bench_workflows_validate(name, examples_pinned, tmp_path):
    from dw.workflow import Workflow

    path = BENCH / "workflows" / name
    definition = json.loads(path.read_text(encoding="utf-8"))
    assert Workflow(definition, str(tmp_path), str(path)).validation_errors({}) == []


@pytest.mark.parametrize("arms_file", ["audio-long.json", "audio-long-prep.json"])
def test_every_audio_arm_names_a_scene_and_a_workflow(arms_file):
    scenes = driver.load_scenes(BENCH / "scenes.json")
    arms = driver.load_arms(BENCH / "arms" / arms_file)
    assert arms
    for arm in arms:
        assert arm.scene == "s6-narration"
        assert arm.scene in scenes
        assert arm.workflow in AUDIO_WORKFLOWS


def test_the_long_arms_are_past_one_take_and_one_track():
    """Narration over 3 minutes at the template's rate, music over 300 s."""
    narration = json.loads(
        (BENCH / "workflows" / "narration-long-s6.json").read_text(encoding="utf-8")
    )["variables"]
    words = len(narration["text"].split())
    assert words / narration["words_per_second"] > 180

    music = json.loads(
        (BENCH / "workflows" / "music-long-s6.json").read_text(encoding="utf-8")
    )["variables"]
    sections = music["sections"]
    joined = (
        sum(s["seconds"] for s in sections)
        - (len(sections) - 1) * music["crossfade_ms"] / 1000
    )
    assert joined > 300
    for section in sections:
        assert section["audio_duration"] <= 300
        assert section["audio_duration"] >= 1.5 * section["seconds"]
