"""One admission service for validate, submit and rerun.

A request used to be loaded and checked in several places - the validate
route, `_candidate_for`, `JobManager.submit` - each building its own
Workflow, expanding the definition again, activating the workspace's asset
library around some of its checks and not others, and computing a different
subset of the warnings. `dw.server.admission.admit` loads it once, checks it
once, with the library active for all of it.
"""

import copy
import os

import pytest
from fastapi.testclient import TestClient

from dw import assets, validation
from dw.library import library_path
from dw.server import admission as admission_module
from dw.server.admission import admit
from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.server.job_record import TERMINAL_STATES
from dw.workflow import Workflow
from dw.workspace import Workspace

from .test_prepare_pipeline import DEFINITION as prepare_definition
from .test_server import (
    ScriptedWorkerManager,
    success_script,
    valid_workflow,
    wait_for_status,
)


def for_each_workflow():
    """A task-only workflow with a for_each: expansion is where the work
    goes, and a default entry carries a key no step reads, so a warning
    derived from the expansion is there to report."""
    return {
        "id": "fe",
        "variables": {
            "shots": [
                {"name": "a", "text": "A", "mood": "calm"},
                {"name": "b", "text": "B", "mood": "calm"},
            ]
        },
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


def image_workflow():
    workflow = valid_workflow("imaged")
    workflow["variables"]["image"] = "asset:present.png"
    return workflow


@pytest.fixture
def root(tmp_path):
    root = Workspace(tmp_path / "studio", "flag").ensure()
    with open(os.path.join(root.assets, "present.png"), "wb") as file:
        file.write(b"png")
    return root


@pytest.fixture
def server(root, tmp_path):
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


@pytest.fixture
def folds(monkeypatch):
    """Every call of Workflow._fold, the stage that expansion starts with."""
    calls = []
    original = Workflow._fold

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Workflow, "_fold", counting)
    return calls


def test_a_submit_expands_once(server, folds):
    with server() as client:
        response = client.post(
            "/api/jobs",
            json={
                "workflow": for_each_workflow(),
                "arguments": {"shots": [{"name": "x", "text": "X"}]},
            },
        )
        assert response.status_code == 201, response.text
        assert len(folds) == 1


def test_a_validate_expands_once(server, folds):
    with server() as client:
        answer = client.post(
            "/api/validate?sizes=false",
            json={
                "workflow": for_each_workflow(),
                "arguments": {"shots": [{"name": "x", "text": "X"}]},
            },
        ).json()
        assert answer["valid"], answer
        assert answer["plan"] is not None
        assert len(folds) == 1


def test_a_validate_without_arguments_expands_once(server, folds):
    """Omitted arguments check the document (validation_errors is asked
    with None) while the warnings and the plan are asked with {} - the same
    fold, which the expansion memo has to key as one."""
    with server() as client:
        answer = client.post(
            "/api/validate?sizes=false", json={"workflow": for_each_workflow()}
        ).json()
        assert answer["valid"], answer
        assert answer["plan"] is not None
        assert len(folds) == 1


def test_a_bound_submit_expands_once(server, folds):
    """The bound acknowledgement's plan rides the admission's own fold."""
    body = {
        "workflow": for_each_workflow(),
        "arguments": {"shots": [{"name": "x", "text": "X"}]},
    }
    with server() as client:
        plan = client.post("/api/validate?sizes=false", json=body).json()["plan"]
        assert plan is not None
        folds.clear()

        response = client.post(
            "/api/jobs",
            json={
                **body,
                "acknowledged_cost": {
                    "fingerprint": plan["fingerprint"],
                    "minutes": plan["estimate"]["minutes"],
                    "downloads": plan["downloads_required"],
                },
            },
        )

        # A 409 (plan None) would be one fold too - so the 201 first
        assert response.status_code == 201, response.text
        assert response.json()["acknowledged"] == "bound"
        assert len(folds) == 1


def test_a_rerun_expands_once(server, folds):
    with server() as client:
        submitted = client.post(
            "/api/jobs",
            json={
                "workflow": for_each_workflow(),
                "arguments": {"shots": [{"name": "x", "text": "X"}]},
            },
        )
        assert submitted.status_code == 201, submitted.text
        job_id = submitted.json()["id"]
        wait_for_status(client, job_id, TERMINAL_STATES)
        folds.clear()

        rerun = client.post(f"/api/jobs/{job_id}/rerun")

        assert rerun.status_code == 201, rerun.text
        assert len(folds) == 1


def _swap_warning_check(monkeypatch, name, run):
    """Replace one entry of the warning registry for this test."""
    registry = list(validation.WARNING_CHECKS)
    index = [check.name for check in registry].index(name)
    original = registry[index]
    registry[index] = validation.Check(name, run)
    monkeypatch.setattr(validation, "WARNING_CHECKS", registry)
    return original


WARNING_CRASH = (
    "internal: warning check 'shot_span_warnings' failed (RuntimeError) - "
    "the server log has the detail"
)


def test_a_warning_source_that_raises_is_said_and_refuses_nothing(server, monkeypatch):
    """A warning never refuses, and a source that fails is no longer
    silently dropped (B10): it is one internal warning, and the sources
    after it still report - on validate and on the queued job alike."""

    def exploding(_context):
        raise RuntimeError("probe exploded")

    _swap_warning_check(monkeypatch, "shot_span_warnings", exploding)
    # The source after the one that raises - its warning must still arrive
    _swap_warning_check(
        monkeypatch,
        "inherited_vram_warnings",
        lambda _context: ["sentinel after the failure"],
    )
    with server() as client:
        answer = client.post(
            "/api/validate?sizes=false", json={"workflow": valid_workflow()}
        ).json()
        assert answer["valid"] is True, answer
        assert any("sets no 'seed'" in w for w in answer["warnings"])
        assert WARNING_CRASH in answer["warnings"]
        assert "sentinel after the failure" in answer["warnings"]
        # the exception's own text stays in the log
        assert not any("probe exploded" in w for w in answer["warnings"])

        submitted = client.post("/api/jobs", json={"workflow": valid_workflow()})
        assert submitted.status_code == 201, submitted.text
        assert any("sets no 'seed'" in w for w in submitted.json()["warnings"])
        assert WARNING_CRASH in submitted.json()["warnings"]
        assert "sentinel after the failure" in submitted.json()["warnings"]


def test_a_check_failing_after_loading_is_not_called_a_construction_failure(
    server, monkeypatch
):
    """The Workflow was built; an argument check raising is the validator
    failing, answered as such - not as a workflow that could not be
    constructed."""

    def exploding(*args, **kwargs):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(admission_module, "argument_reference_errors", exploding)
    with server() as client:
        response = client.post(
            "/api/validate?sizes=false", json={"workflow": valid_workflow()}
        )
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["valid"] is False
    assert "could not be validated" in answer["error"]
    assert "constructed" not in answer["error"]


def test_an_undeclared_frame_snap_through_a_variable_is_reported_at_its_path(
    server,
):
    workflow = copy.deepcopy(prepare_definition)
    workflow["variables"]["snap"] = "constraint:nope"
    workflow["steps"][1]["task"]["arguments"]["frame_snap"] = "variable:snap"
    with server() as client:
        answer = client.post(
            "/api/validate?sizes=false", json={"workflow": workflow}
        ).json()
        assert answer["valid"] is False
        assert [error["path"] for error in answer["errors"]] == [
            "steps[1].task.arguments.frame_snap"
        ]
        assert "'constraint:nope' names no entry" in answer["errors"][0]["message"]


def test_every_admission_check_sees_the_workspace_asset_library(
    server, root, monkeypatch
):
    seen = []
    original = None

    def spy(context):
        seen.append(assets._active_asset_dir.get())
        return original.run(context)

    original = _swap_warning_check(monkeypatch, "shot_span_warnings", spy)
    shots_assets = os.path.join(root.root, "shots", "assets")
    with server() as client:
        client.post("/api/workspaces", json={"name": "shots"})
        body = {"workflow": valid_workflow(), "workspace": "shots"}

        assert client.post("/api/validate?sizes=false", json=body).json()["valid"]
        assert seen == [shots_assets]

        assert client.post("/api/jobs", json=body).status_code == 201
        assert seen == [shots_assets, shots_assets]


def test_bad_arguments_are_refused_before_the_queue_and_validate_still_expands(
    server, folds
):
    body = {"workflow": for_each_workflow(), "arguments": {"prmopt": "a cat"}}
    with server() as client:
        manager = client.app.state.job_manager

        response = client.post("/api/jobs", json=body)
        assert response.status_code == 400
        assert "arguments.prmopt" in response.json()["detail"]
        assert manager.worker_manager.commands == []

        folds.clear()
        answer = client.post("/api/validate?sizes=false", json=body).json()
        assert answer["valid"] is False
        assert answer["errors"][0]["path"] == "arguments.prmopt"
        # the expansion ran over the defaults, whose entries carry a key no
        # step reads - so its warning is there beside the error
        assert any("mood" in warning for warning in answer["warnings"])
        assert len(folds) == 1


def test_a_rerun_rechecks_its_references(server, root):
    with open(os.path.join(root.assets, "gone.png"), "wb") as file:
        file.write(b"png")
    with server() as client:
        submitted = client.post(
            "/api/jobs",
            json={
                "workflow": image_workflow(),
                "arguments": {"image": "asset:gone.png"},
            },
        )
        assert submitted.status_code == 201, submitted.text
        job_id = submitted.json()["id"]
        wait_for_status(client, job_id, TERMINAL_STATES)
        os.remove(os.path.join(root.assets, "gone.png"))
        executed = [
            command
            for command in client.app.state.job_manager.worker_manager.commands
            if command["type"] == "execute"
        ]

        rerun = client.post(f"/api/jobs/{job_id}/rerun")

        assert rerun.status_code == 400
        assert "arguments.image" in rerun.json()["detail"]
        assert "gone.png" in rerun.json()["detail"]
        assert [
            command
            for command in client.app.state.job_manager.worker_manager.commands
            if command["type"] == "execute"
        ] == executed


def test_a_submitted_job_records_the_warnings_validate_reports(server):
    with server() as client:
        validated = client.post(
            "/api/validate?sizes=false", json={"workflow": valid_workflow()}
        ).json()
        unseeded = [w for w in validated["warnings"] if "sets no 'seed'" in w]
        assert unseeded

        submitted = client.post("/api/jobs", json={"workflow": valid_workflow()})

        assert submitted.status_code == 201
        assert unseeded[0] in submitted.json()["warnings"]


def test_admit_answers_a_schema_failure_with_no_warnings(root):
    broken = {"id": "broken", "steps": [{"name": "s"}]}

    admission = admit(
        workflow_path=None,
        workflow=broken,
        arguments={},
        base_dir=None,
        workspace=root,
        ceiling_index={},
        output_dir=root.outputs,
        workflow_dir=root.workflows,
        asset_library=library_path("assets", root),
        prompt_library=library_path("prompts", root),
    )

    assert not admission.ok
    assert admission.schema_errors
    assert admission.warnings == []
    assert admission.errors == [
        {
            "path": "steps[0]",
            "message": "{'name': 's'} is not valid under any of the given schemas",
        }
    ]


def test_admission_checks_content_type_against_the_callers_arguments(root):
    """A document-default 'text/html' content_type that the caller's own
    argument overrides to 'text/plain' must be admitted - JobManager.submit
    used to validate the unsubstituted document (loaded.validate(), no
    arguments), refusing a run that validate_workflow had already accepted
    for the same call (#415, the run_workflow mirror of #414). Moved from
    tests/test_server_jobs.py when admission left submit."""
    definition = {
        "id": "se-415",
        "variables": {"ct": "text/html"},
        "steps": [
            {
                "name": "t",
                "task": {"command": "compose_text", "arguments": {"parts": ["x"]}},
                "result": {"content_type": "variable:ct"},
            }
        ],
    }

    def admitted(arguments):
        return admit(
            workflow_path=None,
            workflow=definition,
            arguments=arguments,
            base_dir=None,
            workspace=root,
            ceiling_index={},
            output_dir=root.outputs,
            workflow_dir=root.workflows,
            asset_library=library_path("assets", root),
            prompt_library=library_path("prompts", root),
        )

    assert admitted({"ct": "text/plain"}).ok
    assert not admitted({}).ok


def _a_crashing_select_check(monkeypatch):
    def boom(_context):
        raise RuntimeError("check exploded")

    registry = list(validation.ERROR_CHECKS)
    index = [check.name for check in registry].index("select")
    registry[index] = validation.Check("select", boom)
    monkeypatch.setattr(validation, "ERROR_CHECKS", registry)


CRASH_MESSAGE = "check 'select' failed (RuntimeError) - the server log has the detail"


def test_a_crashing_check_is_an_invalid_verdict_on_validate(server, monkeypatch):
    _a_crashing_select_check(monkeypatch)
    with server() as client:
        response = client.post(
            "/api/validate?sizes=false", json={"workflow": valid_workflow()}
        )
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["valid"] is False
    assert [error["path"] for error in answer["errors"]] == [None]
    assert CRASH_MESSAGE in answer["errors"][0]["message"]


def test_a_crashing_check_admits_no_job(server, monkeypatch):
    _a_crashing_select_check(monkeypatch)
    with server() as client:
        response = client.post("/api/jobs", json={"workflow": valid_workflow()})
        assert response.status_code == 400, response.text
        # a finding with no path is its bare message, not "None: ..."
        assert response.json()["detail"] == CRASH_MESSAGE
        assert "check exploded" not in response.text
        assert client.get("/api/jobs").json()["jobs"] == []


SECRET = "/srv/secret/workspace/x.json"


def _raise_with_a_path(*_args, **_kwargs):
    raise RuntimeError(f"cannot read {SECRET}")


@pytest.mark.parametrize(
    "target, name",
    [(Workflow, "validation_context"), (admission_module, "argument_errors")],
    ids=["context", "argument_errors"],
)
def test_a_failing_gate_names_only_its_exception_type(
    server, monkeypatch, caplog, target, name
):
    """The 400 a submit answers when validation itself fails carries the
    exception's type, not its text - which can name a server path - and the
    log keeps the text with its traceback."""
    monkeypatch.setattr(target, name, _raise_with_a_path)
    with caplog.at_level("ERROR", logger="dw"):
        with server() as client:
            response = client.post("/api/jobs", json={"workflow": valid_workflow()})
            assert response.status_code == 400, response.text
            assert SECRET not in response.text
            assert "RuntimeError" in response.json()["detail"]
            assert client.get("/api/jobs").json()["jobs"] == []
    assert any(
        record.exc_info and SECRET in str(record.exc_info[1])
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "arguments, level",
    [({}, "ERROR"), ({"undeclared": 1}, "DEBUG")],
    ids=["admissible", "refused"],
)
def test_a_failing_warning_source_logs_loudly_only_when_admissible(
    root, monkeypatch, caplog, arguments, level
):
    """A warning source tripping over arguments already refused is expected,
    so it logs at DEBUG; on an otherwise admissible request it is a bug worth
    an ERROR. The internal warning is said either way."""

    def boom(_context):
        raise RuntimeError("warning exploded")

    monkeypatch.setattr(validation, "WARNING_CHECKS", [validation.Check("noisy", boom)])
    with caplog.at_level("DEBUG", logger="dw"):
        admission = admit(
            workflow_path=None,
            workflow=valid_workflow(),
            arguments=arguments,
            base_dir=None,
            workspace=root,
            ceiling_index={},
            output_dir=root.outputs,
            workflow_dir=root.workflows,
            asset_library=library_path("assets", root),
            prompt_library=library_path("prompts", root),
        )

    assert admission.ok == (level == "ERROR")
    assert [w for w in admission.warnings if "check 'noisy' failed" in w]
    records = [r for r in caplog.records if "noisy" in r.getMessage()]
    assert [r.levelname for r in records] == [level]
    assert records[0].exc_info is not None


def unexpandable_workflow():
    """Passes the schema; its for_each names a variable holding no list."""
    workflow = for_each_workflow()
    workflow["variables"]["shots"] = "not a list"
    return workflow


def undeclared_workflow():
    workflow = valid_workflow()
    workflow["steps"][0]["pipeline"]["arguments"]["prompt"] = "variable:nope"
    return workflow


@pytest.mark.parametrize(
    "workflow, path",
    [
        (unexpandable_workflow(), "steps[0].for_each"),
        (undeclared_workflow(), "steps[0].pipeline.arguments.prompt"),
    ],
    ids=["for_each", "undeclared_variable"],
)
def test_an_expansion_failure_is_a_finding_on_both_routes(server, workflow, path):
    """The request's one context is built ahead of the gates, and must not
    turn what the gates answer as a finding into a validator failure."""
    with server() as client:
        answer = client.post(
            "/api/validate?sizes=false", json={"workflow": workflow}
        ).json()
        assert answer["valid"] is False
        assert [error["path"] for error in answer["errors"]] == [path]
        assert "could not be validated" not in answer["errors"][0]["message"]

        response = client.post("/api/jobs", json={"workflow": workflow})
        assert response.status_code == 400, response.text
        assert response.json()["detail"].startswith(f"{path}: ")
        assert client.get("/api/jobs").json()["jobs"] == []


def test_one_request_probes_a_file_once_across_errors_and_warnings(
    server, root, monkeypatch
):
    """A dissolve (the error pass) and an analyze step (the warning pass)
    naming one asset share the request's probe cache (B9)."""
    import av

    from .test_dissolve_frame_errors import write_mp4

    for name in ("a.mp4", "b.mp4"):
        write_mp4(os.path.join(root.assets, name), frames=24)
    workflow = {
        "id": "probed",
        "steps": [
            {
                "name": "join",
                "task": {
                    "command": "dissolve_videos",
                    "arguments": {
                        "videos": ["asset:a.mp4", "asset:b.mp4"],
                        "dissolve_frames": 6,
                    },
                },
                "result": {"content_type": "video/mp4", "fps": 6},
            },
            {
                "name": "seams",
                "task": {
                    "command": "analyze_seams",
                    "arguments": {
                        "video": "asset:a.mp4",
                        "shots": [{"name": "s", "start_frame": 0, "num_frames": 30}],
                    },
                },
                "result": {"content_type": "application/json"},
            },
        ],
    }
    opened = []
    real_open = av.open

    def counting_open(file, *args, **kwargs):
        opened.append(os.path.basename(str(file)))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(av, "open", counting_open)
    with server() as client:
        answer = client.post(
            "/api/validate?sizes=false", json={"workflow": workflow}
        ).json()

    assert answer["valid"] is True, answer
    # the warning pass read the probe: the shots record reaches past a.mp4
    assert any("past the file's 24 frames" in w for w in answer["warnings"]), answer
    assert opened.count("a.mp4") == 1, opened
