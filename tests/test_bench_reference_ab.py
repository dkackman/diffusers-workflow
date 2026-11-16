"""The narration reference A/B bench (#1384 stage A, #1386): the workflows
validate with the examples root pinned, as the bench server pins it, and the
three arms speak the same text with the same seed, differing only in the
reference (synthetic or human) or in sampling (temperature above the model's
effective default)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
sys.path.insert(0, str(BENCH))
import driver  # noqa: E402

WORKFLOWS = ["narration-reference-ab.json", "narrator-human-librivox.json"]
ARMS = BENCH / "arms" / "narration-reference-ab.json"
# Audio8-TTS-Realistic-0.6b generation_config.json (revision 1e857568) and the
# vendored generate()'s fallback both sample at 0.7.
EFFECTIVE_DEFAULT_TEMPERATURE = 0.7
SYNTHETIC = "asset:bench/narrator.wav"
HUMAN = "asset:bench/narrator-human.wav"


@pytest.fixture
def examples_pinned(monkeypatch):
    from dw.library import LIBRARY_PATH_ENV_VARS, WORKFLOWS_KIND

    monkeypatch.setenv(LIBRARY_PATH_ENV_VARS[WORKFLOWS_KIND], str(ROOT / "workflows"))


def _variables(name):
    path = BENCH / "workflows" / name
    return json.loads(path.read_text(encoding="utf-8"))["variables"]


@pytest.mark.parametrize("name", WORKFLOWS)
def test_the_reference_ab_workflows_validate(name, examples_pinned, tmp_path):
    from dw.workflow import Workflow

    path = BENCH / "workflows" / name
    definition = json.loads(path.read_text(encoding="utf-8"))
    assert Workflow(definition, str(tmp_path), str(path)).validation_errors({}) == []


@pytest.mark.parametrize("arm_index", [0, 1, 2])
def test_every_arm_validates_with_its_arguments(arm_index, examples_pinned, tmp_path):
    from dw.workflow import Workflow

    arm = driver.load_arms(ARMS)[arm_index]
    path = BENCH / "workflows" / arm.workflow
    definition = json.loads(path.read_text(encoding="utf-8"))
    arguments = {**arm.arguments, "seed": arm.seed}
    assert (
        Workflow(definition, str(tmp_path), str(path)).validation_errors(arguments)
        == []
    )


def test_the_passage_is_one_take():
    assert len(_variables("narration-reference-ab.json")["text"].split()) <= 111


def test_the_arms_name_a_scene_and_the_ab_workflow():
    scenes = driver.load_scenes(BENCH / "scenes.json")
    arms = driver.load_arms(ARMS)
    assert len(arms) == 3
    assert len({arm.name for arm in arms}) == 3
    for arm in arms:
        assert arm.scene in scenes
        assert arm.workflow == "narration-reference-ab.json"


def test_the_arms_differ_only_in_reference_or_sampling():
    arms = driver.load_arms(ARMS)
    defaults = _variables("narration-reference-ab.json")
    allowed = {"reference_audio", "reference_text", "generate_kwargs"}
    for arm in arms:
        assert set(arm.arguments) <= allowed, arm.name
        assert arm.seed == defaults["seed"]
    synthetic, human, sampling = arms

    assert synthetic.arguments["reference_audio"] == SYNTHETIC
    assert synthetic.arguments["generate_kwargs"] == {}
    assert synthetic.arguments["reference_text"] == defaults["reference_text"]

    # Human: only the reference (audio and its transcript) moves.
    assert human.arguments["reference_audio"] == HUMAN
    assert human.arguments["reference_text"] != synthetic.arguments["reference_text"]
    assert human.arguments["generate_kwargs"] == {}

    # Sampling: the synthetic reference, only the temperature moves, and above
    # the effective default.
    assert sampling.arguments["reference_audio"] == SYNTHETIC
    assert sampling.arguments["reference_text"] == synthetic.arguments["reference_text"]
    assert set(sampling.arguments["generate_kwargs"]) == {"temperature"}
    assert (
        sampling.arguments["generate_kwargs"]["temperature"]
        > EFFECTIVE_DEFAULT_TEMPERATURE
    )


def test_the_human_clip_is_5_to_10_seconds_of_https_librivox():
    human = _variables("narrator-human-librivox.json")
    assert human["source_url"].startswith("https://archive.org/download/")
    assert "librivox" in human["source_url"]
    assert 5 <= human["duration_seconds"] <= 10
    arms = driver.load_arms(ARMS)
    assert arms[1].arguments["reference_text"] == human["reference_text"]
