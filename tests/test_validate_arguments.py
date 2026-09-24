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


def null_variable_workflow():
    """A required task argument fed by `variable:audio`, where `audio`'s
    declared default is null - the #364 repro. A fine document (the step
    does supply the argument, just not yet a value); a run left as-is
    would fail."""
    return {
        "id": "null_var",
        "variables": {"audio": None},
        "steps": [
            {
                "name": "n",
                "task": {
                    "command": "normalize_audio",
                    "arguments": {"audio": "variable:audio"},
                },
                "result": {"content_type": "audio/wav"},
            }
        ],
    }


def placeholder_workflow():
    """A stored default that names no file in this workspace, the shape of
    `templates/ltx2/reference-sheet` and its siblings (#166): a bare call
    must fail the same way an explicit one naming the same value would."""
    workflow = valid_workflow("placeholder")
    workflow["variables"]["image"] = "asset:not-shipped.png"
    workflow["steps"][0]["pipeline"]["arguments"]["image"] = "variable:image"
    return workflow


@pytest.fixture
def server(tmp_path):
    root = Workspace(tmp_path / "studio", "flag").ensure()
    with open(os.path.join(root.workflows, "Typed.json"), "w") as file:
        json.dump(typed_workflow(), file)
    with open(os.path.join(root.workflows, "Placeholder.json"), "w") as file:
        json.dump(placeholder_workflow(), file)
    with open(os.path.join(root.workflows, "NullVariable.json"), "w") as file:
        json.dump(null_variable_workflow(), file)
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
                    "task": {
                        "command": "compose_text",
                        "arguments": {"parts": ["item:text"]},
                    },
                    "result": {"content_type": "text/plain"},
                },
                {
                    "name": "edit",
                    "task": {
                        "command": "compose_text",
                        "arguments": {"parts": "gather:shot"},
                    },
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

    def test_a_missing_reference_inside_a_list_argument_fails_validation(self, server):
        """A `shots` list entry's `references[].from_file` is checked the
        same as a top-level string argument - `_argument_reference_errors`
        walks into list/dict argument values rather than skipping them."""
        workflow = {
            "id": "list-refs",
            "variables": {"shots": [{"name": "a", "references": []}]},
            "steps": [
                {
                    "name": "shot",
                    "for_each": "variable:shots",
                    "task": {
                        "command": "compose_text",
                        "arguments": {"parts": ["item:name"]},
                    },
                    "result": {"content_type": "text/plain"},
                }
            ],
        }
        with server() as client:
            result = client.post(
                "/api/validate",
                json={
                    "workflow": workflow,
                    "arguments": {
                        "shots": [
                            {
                                "name": "a",
                                "references": [{"from_file": "asset:missing.wav"}],
                            }
                        ]
                    },
                },
            ).json()

        assert result["valid"] is False
        assert (
            result["errors"][0]["path"] == "arguments.shots[0].references[0].from_file"
        )
        assert "missing.wav" in result["errors"][0]["message"]

    def test_a_bad_stored_default_fails_a_bare_call(self, server):
        """`validate_workflow(name="Placeholder")` with no arguments at all
        used to answer valid, because only `arguments` was checked - the
        same value handed back explicitly already failed. One run, one
        verdict (#166)."""
        with server() as client:
            bare = client.post(
                "/api/validate", json={"workflow_path": "Placeholder"}
            ).json()
            explicit = validate(
                client,
                workflow_path="Placeholder",
                arguments={"image": "asset:not-shipped.png"},
            )

        assert bare["valid"] is False
        assert bare["errors"][0]["path"] == "variables.image"
        assert "not-shipped.png" in bare["errors"][0]["message"]
        assert explicit["valid"] is False

    def test_an_override_of_a_bad_default_is_checked_as_the_override(self, server):
        """A caller who overrides the bad default is judged on their own
        value, not on the default it replaced."""
        with server() as client:
            result = client.post(
                "/api/validate",
                json={
                    "workflow_path": "Placeholder",
                    "arguments": {"image": "asset:iris.png"},
                },
            ).json()

        assert result["valid"] is True

    def test_a_reference_inside_a_list_argument_that_exists_passes(self, server):
        workflow = {
            "id": "list-refs",
            "variables": {"shots": [{"name": "a", "references": []}]},
            "steps": [
                {
                    "name": "shot",
                    "for_each": "variable:shots",
                    "task": {
                        "command": "compose_text",
                        "arguments": {"parts": ["item:name"]},
                    },
                    "result": {"content_type": "text/plain"},
                }
            ],
        }
        with server() as client:
            result = client.post(
                "/api/validate",
                json={
                    "workflow": workflow,
                    "arguments": {
                        "shots": [
                            {
                                "name": "a",
                                "references": [{"from_file": "asset:iris.png"}],
                            }
                        ]
                    },
                },
            ).json()

        assert result["valid"] is True


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

    def test_a_bad_reference_is_refused_before_the_job_is_queued(self, server):
        """The name half of the check was refused at submission and the
        reference half was not, so a typo'd asset came back as a job id and
        died on the first step - a success-shaped answer to a question
        validate could answer for free."""
        with server() as client:
            response = client.post(
                "/api/jobs",
                json={
                    "workflow_path": "Typed",
                    "arguments": {"image": "asset:iirs.png"},
                },
            )

            assert response.status_code == 400
            detail = response.json()["detail"]
            assert "arguments.image" in detail and "iirs.png" in detail
            assert client.get("/api/jobs").json()["jobs"] == []

    def test_a_reference_that_resolves_still_queues(self, server):
        with server() as client:
            response = client.post(
                "/api/jobs",
                json={
                    "workflow_path": "Typed",
                    "arguments": {"image": "asset:iris.png"},
                },
            )

        assert response.status_code == 201

    def test_an_inline_workflow_is_checked_the_same_way(self, server):
        with server() as client:
            response = client.post(
                "/api/jobs",
                json={
                    "workflow": typed_workflow(),
                    "arguments": {"image": "asset:nowhere.png"},
                },
            )

        assert response.status_code == 400
        assert "arguments.image" in response.json()["detail"]

    def test_a_bad_stored_default_is_refused_before_the_job_is_queued(self, server):
        """The same gap at submission: a bare submit used to queue a job
        that could only fail on its first step (#166)."""
        with server() as client:
            response = client.post("/api/jobs", json={"workflow_path": "Placeholder"})

            assert response.status_code == 400
            assert "variables.image" in response.json()["detail"]
            assert client.get("/api/jobs").json()["jobs"] == []

    def test_submission_and_validation_give_the_same_message(self, server):
        """The ticket was a consistency gap, not a missing check: what the
        free pre-flight says is what submission says."""
        arguments = {"image": "asset:iirs.png"}
        with server() as client:
            validated = validate(client, arguments=arguments)
            refused = client.post(
                "/api/jobs", json={"workflow_path": "Typed", "arguments": arguments}
            )

        assert validated["errors"][0]["message"] in refused.json()["detail"]


class TestNullVariableArgument:
    """#364: a required task argument fed by `variable:name` where `name`'s
    declared value is null is a fine document - `save_workflow` accepts it,
    and `validate_workflow` called with no `arguments` at all must agree,
    since that is the same "check the document" question. The moment the
    caller names arguments of their own - even `{}` - it is a real run
    being checked, and one that leaves the variable null is a hard error.
    """

    def test_no_arguments_key_at_all_is_a_warning_not_an_error(self, server):
        """The literal repro: no `arguments` field in the request body."""
        with server() as client:
            response = client.post(
                "/api/validate", json={"workflow_path": "NullVariable"}
            ).json()

        assert response["valid"] is True
        assert response["errors"] == []
        assert any("audio" in warning for warning in response["warnings"])

    def test_an_inline_document_with_no_arguments_is_a_warning_too(self, server):
        with server() as client:
            response = client.post(
                "/api/validate", json={"workflow": null_variable_workflow()}
            ).json()

        assert response["valid"] is True
        assert any("audio" in warning for warning in response["warnings"])

    def test_an_explicit_empty_arguments_dict_is_still_a_hard_error(self, server):
        """`{}` names a run with no values supplied, distinct from omitting
        `arguments` entirely - the variable is still null for that run."""
        with server() as client:
            response = client.post(
                "/api/validate",
                json={"workflow_path": "NullVariable", "arguments": {}},
            ).json()

        assert response["valid"] is False
        assert response["errors"][0]["variable"] == "audio"

    def test_arguments_that_still_leave_it_null_are_a_hard_error(self, server):
        with server() as client:
            response = client.post(
                "/api/validate",
                json={
                    "workflow_path": "NullVariable",
                    "arguments": {"audio": None},
                },
            ).json()

        assert response["valid"] is False
        assert response["errors"][0]["variable"] == "audio"

    def test_arguments_that_supply_a_value_pass(self, server):
        with server() as client:
            response = client.post(
                "/api/validate",
                json={
                    "workflow_path": "NullVariable",
                    "arguments": {"audio": "asset:iris.png"},
                },
            ).json()

        assert response["valid"] is True
        assert response["errors"] == []

    def test_save_workflow_accepts_the_document_with_a_warning(self, server):
        with server() as client:
            response = client.put(
                "/api/workflows/NullVariableSaved",
                json={"workflow": null_variable_workflow()},
            )

        assert response.status_code == 200
        assert any("audio" in warning for warning in response.json()["warnings"])


def test_an_entry_key_no_step_reads_is_a_warning_not_an_error(server):
    workflow = {
        "id": "cut",
        "variables": {"shots": [{"name": "a", "text": "p"}]},
        "steps": [
            {
                "name": "shot",
                "for_each": "variable:shots",
                "task": {
                    "command": "compose_text",
                    "arguments": {"parts": ["item:text"]},
                },
                "result": {"content_type": "text/plain"},
            }
        ],
    }
    with server() as client:
        response = client.post(
            "/api/validate",
            json={
                "workflow": workflow,
                "arguments": {"shots": [{"name": "a", "text": "p", "txt": "q"}]},
            },
        )
    body = response.json()
    assert body["valid"] is True
    assert any(
        w.startswith("arguments.shots[0]: entry 'a' carries 'txt'")
        for w in body["warnings"]
    )
