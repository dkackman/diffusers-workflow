"""Validating the arguments a caller is about to run with.

`POST /api/validate` used to answer about the stored definition and its
stock defaults - everything about a run except the part the caller actually
wrote. A bad `asset:` reference or a variable renamed out of the template
only surfaced once the run had been paid for (36 minutes, in the report that
prompted this). The same fold-the-arguments-in check now runs at submission
too, so a bad argument is a 400 rather than a job that fails after queuing.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.variables import argument_errors
from dw.workspace import Workspace

from .test_server import ScriptedWorkerManager, success_script, valid_workflow


def typed_workflow():
    """Variables of three shapes: text, a number, and a reference-carrying
    string - the three an argument can get wrong."""
    workflow = valid_workflow("typed")
    workflow["variables"] = {"prompt": "d", "steps": 25, "image": "asset:iris.png"}
    return workflow


@pytest.fixture
def server(tmp_path):
    root = Workspace(tmp_path / "studio", "flag").ensure()
    with open(os.path.join(root.workflows, "Typed.json"), "w") as file:
        json.dump(typed_workflow(), file)
    with open(os.path.join(root.assets, "iris.png"), "wb") as file:
        file.write(b"not really a png, but it is a file under that name")
    with open(os.path.join(root.prompts, "hero.json"), "w") as file:
        json.dump({"text": "a hero, centred"}, file)

    def make(script=success_script):
        manager = JobManager(
            root.outputs,
            worker_manager=ScriptedWorkerManager(script),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=root.workflows,
        )
        app = create_app(
            workflow_dir=root.workflows,
            output_dir=root.outputs,
            job_manager=manager,
            prompt_dir=root.prompts,
            asset_dir=root.assets,
            workspace=root.root,
        )
        return TestClient(app, base_url="http://localhost")

    return make


def validate(client, **body):
    return client.post("/api/validate", json={"workflow_path": "Typed", **body}).json()


class TestArgumentErrors:
    """The engine-level check, with no server around it."""

    def test_nothing_to_say_without_arguments(self):
        assert argument_errors(typed_workflow(), {}) == []
        assert argument_errors(typed_workflow(), None) == []

    def test_an_undeclared_name_is_reported_with_the_declared_ones(self):
        errors = argument_errors(typed_workflow(), {"prmopt": "x"})

        assert [error["path"] for error in errors] == ["arguments.prmopt"]
        assert "image, prompt, steps" in errors[0]["message"]

    def test_every_bad_argument_is_reported_not_just_the_first(self):
        errors = argument_errors(typed_workflow(), {"a": 1, "b": 2})

        assert [error["path"] for error in errors] == ["arguments.a", "arguments.b"]

    def test_a_value_that_will_not_coerce_is_reported(self):
        errors = argument_errors(typed_workflow(), {"steps": "twenty"})

        assert [error["path"] for error in errors] == ["arguments.steps"]

    def test_a_workflow_with_no_variables_takes_no_arguments(self):
        """The one case the run itself never reported: Workflow.run only
        substitutes when a variables block exists, so these were dropped."""
        bare = {"id": "bare", "steps": []}

        errors = argument_errors(bare, {"prompt": "x"})

        assert [error["path"] for error in errors] == ["arguments.prompt"]
        assert "declares no variables" in errors[0]["message"]


class TestValidateRoute:
    def test_good_arguments_validate_and_are_named(self, server):
        with server() as client:
            result = validate(
                client,
                arguments={"prompt": "a cat", "steps": 30, "image": "asset:iris.png"},
            )

        assert result["valid"] is True
        assert result["checked_arguments"] == ["image", "prompt", "steps"]

    def test_the_stored_definition_alone_says_so(self, server):
        """No arguments, no claim about any: the old answer, unchanged."""
        with server() as client:
            result = validate(client)

        assert result["valid"] is True and "checked_arguments" not in result

    def test_an_undeclared_argument_fails_validation(self, server):
        with server() as client:
            result = validate(client, arguments={"prmopt": "a cat"})

        assert result["valid"] is False
        assert result["errors"][0]["path"] == "arguments.prmopt"
        assert result["checked_arguments"] == ["prmopt"]

    def test_a_missing_asset_reference_fails_validation(self, server):
        with server() as client:
            result = validate(client, arguments={"image": "asset:iirs.png"})

        assert result["valid"] is False
        assert result["errors"][0]["path"] == "arguments.image"
        assert "iirs.png" in result["errors"][0]["message"]

    def test_an_asset_that_is_there_passes(self, server):
        with server() as client:
            assert validate(client, arguments={"image": "asset:iris.png"})["valid"]

    def test_a_missing_prompt_reference_fails_validation(self, server):
        with server() as client:
            missing = validate(client, arguments={"prompt": "prompt:villain"})
            found = validate(client, arguments={"prompt": "prompt:hero"})

        assert missing["valid"] is False
        assert missing["errors"][0]["path"] == "arguments.prompt"
        assert found["valid"] is True

    def test_a_traversal_in_a_reference_is_refused_not_resolved(self, server):
        with server() as client:
            result = validate(client, arguments={"image": "asset:../../etc/passwd"})

        assert result["valid"] is False
        assert result["errors"][0]["path"] == "arguments.image"

    def test_arguments_are_checked_on_an_inline_definition_too(self, server):
        with server() as client:
            result = client.post(
                "/api/validate",
                json={"workflow": typed_workflow(), "arguments": {"nope": 1}},
            ).json()

        assert result["valid"] is False
        assert result["errors"][0]["path"] == "arguments.nope"

    def test_validate_expands_for_each_with_the_callers_list(self, server):
        workflow = {
            "id": "fe",
            "variables": {"shots": [{"name": "a", "text": "A"}]},
            "steps": [
                {
                    "name": "shot",
                    "for_each": "variable:shots",
                    "task": {"command": "compose_text", "arguments": {"parts": ["item:text"]}},
                    "result": {"content_type": "text/plain"},
                },
                {
                    "name": "edit",
                    "task": {"command": "compose_text", "arguments": {"parts": "gather:shot"}},
                    "result": {"content_type": "text/plain"},
                },
            ],
        }
        with server() as client:
            ok = client.post("/api/validate", json={"workflow": workflow}).json()
            assert ok["valid"] is True

            bad = client.post(
                "/api/validate",
                json={
                    "workflow": workflow,
                    "arguments": {"shots": [{"name": "x"}, {"name": "x"}]},
                },
            ).json()

        assert bad["valid"] is False
        assert bad["errors"][0]["path"] == "steps[0].for_each[1].name"


class TestSubmission:
    def test_a_bad_argument_is_refused_before_the_job_is_queued(self, server):
        with server() as client:
            response = client.post(
                "/api/jobs",
                json={"workflow_path": "Typed", "arguments": {"prmopt": "a cat"}},
            )

            assert response.status_code == 400
            assert "prmopt" in response.json()["detail"]
            assert client.get("/api/jobs").json()["jobs"] == []

    def test_good_arguments_still_queue(self, server):
        with server() as client:
            response = client.post(
                "/api/jobs",
                json={"workflow_path": "Typed", "arguments": {"prompt": "a cat"}},
            )

        assert response.status_code == 201
