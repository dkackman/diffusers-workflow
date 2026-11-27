"""`templates/preferred/text-to-image` (#1627 stage A, #1724): the catalog's
named workflow for a picture request that names no model. It composes
`preference:text_to_image` and its `tier` preset picks the rung.

Runs the real resolver over the shipped store (dw/preference_defaults.json)
on a fixed host, so the targets are the shipped ladder's. No GPU, no model
download, no network.
"""

import json
import os
from unittest.mock import patch

import pytest

from dw import preferences
from dw.preferences import Host
from dw.presets import preset_errors
from dw.workflow import workflow_from_file

from tests.test_examples import REPO_ROOT

PATH = os.path.join(
    REPO_ROOT, "workflows", "templates", "preferred", "text-to-image.json"
)


@pytest.fixture
def cuda_session():
    session = preferences.Session(
        host=Host(device_type="cuda", smallest_card_gb=48, host_ram_gb=128)
    )
    token = preferences.activate(session)
    yield session
    preferences.deactivate(token)


def definition():
    with open(PATH, encoding="utf-8") as file:
        return json.load(file)


def workflow(tmp_path):
    return workflow_from_file(PATH, str(tmp_path))


def main_step(tmp_path, arguments=None):
    return workflow(tmp_path).expanded_definition(arguments)["steps"][0]["workflow"]


def test_variables_are_the_contract_plus_tier():
    variables = definition()["variables"]
    assert set(variables) == {"prompt", "seed", "width", "height", "tier"}
    assert variables["tier"] == "standard"
    assert sorted(definition()["presets"]["tier"]) == ["best", "draft", "standard"]
    assert preset_errors(definition()) == []


def test_one_step_composes_the_preference():
    steps = definition()["steps"]
    assert len(steps) == 1
    assert steps[0]["workflow"]["path"] == "preference:text_to_image"
    assert steps[0]["result"]["subfolder"] == "final"


def test_every_contract_variable_reaches_the_target():
    with open(
        os.path.join(REPO_ROOT, "dw", "preference_defaults.json"), encoding="utf-8"
    ) as file:
        contract = json.load(file)["contracts"]["text_to_image"]
    passed = definition()["steps"][0]["workflow"]["arguments"]
    for name in contract["required"]:
        assert passed[name] == f"variable:{name}"


def test_it_validates(tmp_path, cuda_session):
    assert workflow(tmp_path).validation_errors() == []


def test_standard_is_the_default_and_its_plan_line_names_the_target(
    tmp_path, cuda_session
):
    flow = workflow(tmp_path)
    assert (
        flow.expanded_definition()["steps"][0]["workflow"]["path"] == "models/z-image"
    )
    lines = flow.preference_lines()
    assert len(lines) == 1
    assert lines[0].startswith(
        "preference:text_to_image (tier standard) -> models/z-image"
    )


def test_draft_resolves_the_sd15_baseline_at_its_native_size(tmp_path, cuda_session):
    step = main_step(tmp_path, {"tier": "draft"})
    assert step["path"] == "templates/text-to-image"
    assert (step["arguments"]["width"], step["arguments"]["height"]) == (512, 512)


def test_best_resolves_the_best_rung(tmp_path, cuda_session):
    assert main_step(tmp_path, {"tier": "best"})["path"] == "models/minimax-h3-image"


def test_an_explicit_size_wins_over_the_tier(tmp_path, cuda_session):
    step = main_step(tmp_path, {"tier": "draft", "width": 768})
    assert step["arguments"]["width"] == 768
    assert step["arguments"]["height"] == 512


def test_a_run_makes_one_image(tmp_path, cuda_session):
    # z-image's own default is 4 per prompt; the template asks for one
    assert main_step(tmp_path)["arguments"]["num_images_per_prompt"] == 1


def test_an_unknown_tier_is_refused(tmp_path, cuda_session):
    errors = workflow(tmp_path).validation_errors({"tier": "ultra"})
    assert errors
    assert any("ultra" in error["message"] for error in errors)


# --- over the server (#1762 stage B) -------------------------------------------


def test_the_validate_route_answers_the_plan_preferences_line(tmp_path):
    """`plan.preferences` is declared on the `Plan` response model: under
    DW_STRICT_RESPONSES (tests/conftest.py) an undeclared field fails the
    response, so the route itself is the test, not `build_plan`."""
    from fastapi.testclient import TestClient

    from dw.server.app import create_app
    from dw.server.jobs import JobManager
    from tests.test_server import ScriptedWorkerManager, success_script

    workflows = os.path.join(REPO_ROOT, "workflows")
    outputs = str(tmp_path / "outputs")
    manager = JobManager(
        outputs,
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=workflows,
    )
    app = create_app(workflow_dir=workflows, output_dir=outputs, job_manager=manager)
    # A fixed host, so the shipped ladder resolves the same on any machine:
    # the device and RAM from the probe, the card from the job manager's pool
    host = Host(device_type="cuda", smallest_card_gb=48, host_ram_gb=128)
    with (
        patch.object(preferences, "default_host", return_value=host),
        patch.object(manager, "smallest_card_gb", return_value=48.0),
        TestClient(app, base_url="http://localhost") as client,
    ):
        response = client.post(
            "/api/validate",
            json={
                "workflow_path": "templates/preferred/text-to-image",
                "arguments": {"tier": "draft"},
            },
        )
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["valid"] is True, answer
    (line,) = answer["plan"]["preferences"]
    assert line.startswith("preference:text_to_image (tier draft) -> ")
