"""The variables a workflow takes files through, and the kind each takes.

`input_kinds` reads the argument a variable feeds - the same list the
`needs-input-media` trait reads - so the run form can filter a reference
picker by kind and a catalog card can say what a workflow needs.
"""

import glob
import json
import os
import shutil

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.catalog_shape import (
    _MEDIA_ARGUMENT_KINDS,
    derive_catalog_metadata,
    input_kinds,
)
from tests.test_server import ScriptedWorkerManager
from dw.server.jobs import JobManager

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def step(arguments, name="s"):
    return {"name": name, "task": {"command": "noop", "arguments": arguments}}


def test_a_variable_feeding_an_image_argument_is_an_image_input():
    assert input_kinds([step({"image": "variable:photo"})], {}) == {"photo": "image"}


def test_mask_and_video_and_audio_arguments_name_their_kinds():
    steps = [
        step(
            {
                "mask_image": "variable:mask",
                "source_video": "variable:take",
                "reference_audio": "variable:voice",
                "urls": ["variable:one", "variable:two"],
                "clip": "variable:cut",
                "prompt": "variable:prompt",
            }
        )
    ]
    assert input_kinds(steps, {}) == {
        "cut": "video",
        "mask": "image",
        "one": "image",
        "take": "video",
        "two": "image",
        "voice": "audio",
    }
    assert list(input_kinds(steps, {})) == sorted(input_kinds(steps, {}))


def test_a_variable_whose_default_is_a_step_result_is_not_an_input():
    steps = [step({"audio": "variable:song"})]
    assert input_kinds(steps, {"song": "previous_result:make_song"}) == {}


def test_a_location_object_takes_its_media_type():
    steps = [
        step({"video": {"media_type": "video/mp4", "location": "variable:take"}}),
        step({"source": {"media_type": "image", "location": "variable:sheet"}}),
    ]
    assert input_kinds(steps, {}) == {"take": "video", "sheet": "image"}


def test_a_location_object_without_one_is_a_file():
    steps = [step({"thing": {"location": "variable:blob"}})]
    assert input_kinds(steps, {}) == {"blob": "file"}


def test_the_first_kind_a_variable_takes_wins():
    steps = [step({"image": "variable:x", "video": "variable:x"})]
    assert input_kinds(steps, {}) == {"x": "image"}


def test_metadata_does_not_carry_inputs():
    # derive_catalog_metadata's keys are pinned elsewhere; inputs is the detail's
    assert "inputs" not in derive_catalog_metadata({"steps": []})


@pytest.fixture
def client(tmp_path):
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    shutil.copy(
        os.path.join(REPO_ROOT, "workflows", "templates", "inpaint.json"),
        workflows / "inpaint.json",
    )
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(lambda *args, **kwargs: None),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(workflows),
    )
    app = create_app(
        workflow_dir=str(workflows),
        output_dir=str(tmp_path / "outputs"),
        prompt_dir=str(tmp_path / "prompts"),
        job_manager=manager,
    )
    with TestClient(app, base_url="http://localhost") as client:
        yield client


def test_the_listing_says_nothing_about_inputs_unless_asked(client):
    details = client.get("/api/workflows").json()["details"]
    assert details
    assert all("inputs" not in detail for detail in details.values())
    # a cached detail from an asked listing does not leak into the next plain one
    client.get("/api/workflows", params={"inputs": "true"})
    details = client.get("/api/workflows").json()["details"]
    assert all("inputs" not in detail for detail in details.values())


def test_the_listing_carries_inputs_when_asked(client):
    details = client.get("/api/workflows", params={"inputs": "true"}).json()["details"]
    assert details["inpaint"]["inputs"] == {"image": "image", "mask_image": "image"}


def test_the_needs_input_media_trait_is_unchanged():
    # The trait reads the keys of _MEDIA_ARGUMENT_KINDS; pin them to the set
    # the trait has always read, and every workflow that names an input has it
    assert set(_MEDIA_ARGUMENT_KINDS) == {
        "image", "video", "audio", "mask_image", "source_video",
        "urls", "videos", "clip", "reference_audio",
    }  # fmt: skip
    paths = glob.glob(
        os.path.join(REPO_ROOT, "workflows", "**", "*.json"), recursive=True
    )
    assert paths
    for path in paths:
        with open(path, encoding="utf-8") as file:
            definition = json.load(file)
        if not isinstance(definition, dict):
            continue
        metadata = derive_catalog_metadata(definition)
        variables = definition.get("variables")
        inputs = input_kinds(
            [s for s in definition.get("steps", []) if isinstance(s, dict)],
            variables if isinstance(variables, dict) else {},
        )
        if inputs and "traits" not in definition:
            assert "needs-input-media" in metadata["traits"], path
