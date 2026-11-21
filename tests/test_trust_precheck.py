"""The worker's trust pre-check (workflow_run.require_trusted_pipelines), the
gate it runs before it releases the previous workflow's models (#1333).

It reads the definition's keys only, so it loads no media and, trusted,
does no work at all (#1510); and it refuses everything the loader's own
pre-flight refuses - pre_load_modules included - as validate and submit do,
so a refused job neither queues nor evicts the warm models (#1511).
"""

import queue
from unittest.mock import patch

import pytest

from dw import arguments, workflow_run
from dw.security import UntrustedWorkflowError
from dw.workflow import Workflow


def _media_workflow(tmp_path, **pipeline_extra):
    (tmp_path / "clip.mp4").write_bytes(b"placeholder")
    (tmp_path / "still.png").write_bytes(b"placeholder")
    definition = {
        "id": "media",
        "steps": [
            {
                "name": "gen",
                "pipeline": {
                    "configuration": {"component_type": "StableDiffusionPipeline"},
                    "from_pretrained_arguments": {"model_name": "a/b"},
                    "arguments": {
                        "prompt": "x",
                        "image": "still.png",
                        "source_video": "clip.mp4",
                    },
                    **pipeline_extra,
                },
            }
        ],
    }
    return Workflow(definition, str(tmp_path), str(tmp_path / "w.json"), None)


@pytest.fixture
def media_calls(monkeypatch):
    calls = []

    def fetch_image(value, base_dir, key):
        calls.append(("image", value))
        return value

    def fetch_video(value, base_dir, key, with_audio=False):
        calls.append(("video", value))
        return value

    monkeypatch.setattr(arguments, "fetch_image_with_context", fetch_image)
    monkeypatch.setattr(arguments, "fetch_video_with_context", fetch_video)
    return calls


class TestThePreCheckLoadsNoMedia:
    """#1510"""

    def test_untrusted_reads_keys_only(self, tmp_path, monkeypatch, media_calls):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
        workflow_run.require_trusted_pipelines(_media_workflow(tmp_path), {})
        assert media_calls == []

    def test_untrusted_still_refuses_remote_code(
        self, tmp_path, monkeypatch, media_calls
    ):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
        workflow = _media_workflow(tmp_path)
        pipeline = workflow.workflow_definition["steps"][0]["pipeline"]
        pipeline["from_pretrained_arguments"]["trust_remote_code"] = True
        with pytest.raises(UntrustedWorkflowError, match="trust_remote_code"):
            workflow_run.require_trusted_pipelines(workflow, {})
        assert media_calls == []

    def test_trusted_does_no_work(self, tmp_path, monkeypatch, media_calls):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        workflow = _media_workflow(tmp_path)
        # The pre-check swallows a preparation failure, so count the calls
        # rather than raising from them
        with (
            patch.object(workflow_run, "prepare_definition") as prepared,
            patch.object(Workflow, "expanded_definition") as expanded,
        ):
            workflow_run.require_trusted_pipelines(workflow, {})
        assert prepared.call_count == 0 and expanded.call_count == 0
        assert media_calls == []


# --- #1511: pre_load_modules, and the refusals beside it -------------------


def _refused_definition(kind):
    pipeline = {
        "configuration": {"component_type": "StableDiffusionPipeline"},
        "from_pretrained_arguments": {"model_name": "a/b"},
        "arguments": {"prompt": "x"},
    }
    if kind == "pre_load_modules":
        pipeline["configuration"]["pre_load_modules"] = ["json"]
    elif kind == "dotted type":
        pipeline["configuration"]["scheduler_type"] = "os.path.join"
    elif kind == "constant":
        pipeline["arguments"]["separator"] = "constant:os.sep"
    elif kind == "variable constant":
        pipeline["arguments"]["separator"] = "variable:sep"
        return {
            "id": "hostile",
            "variables": {"sep": "constant:os.sep"},
            "steps": [{"name": "gen", "pipeline": pipeline}],
        }
    return {"id": "hostile", "steps": [{"name": "gen", "pipeline": pipeline}]}


KINDS = ["pre_load_modules", "dotted type", "constant", "variable constant"]


@pytest.mark.parametrize("kind", KINDS)
def test_the_pre_check_refuses(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    workflow = Workflow(
        _refused_definition(kind), str(tmp_path), str(tmp_path / "w.json"), None
    )
    with pytest.raises(UntrustedWorkflowError):
        workflow_run.require_trusted_pipelines(workflow, {})


@pytest.mark.parametrize("kind", KINDS)
def test_trusted_lets_each_through(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
    workflow = Workflow(
        _refused_definition(kind), str(tmp_path), str(tmp_path / "w.json"), None
    )
    workflow_run.require_trusted_pipelines(workflow, {})


def test_an_allowed_pre_load_module_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    definition = _refused_definition("pre_load_modules")
    definition["steps"][0]["pipeline"]["configuration"]["pre_load_modules"] = ["sdnq"]
    workflow = Workflow(definition, str(tmp_path), str(tmp_path / "w.json"), None)
    workflow_run.require_trusted_pipelines(workflow, {})


@pytest.mark.parametrize("kind", KINDS)
def test_a_refused_job_keeps_the_warm_models(tmp_path, monkeypatch, kind):
    """#1333's rule through the worker: no release, no run, just the refusal."""
    from dw import worker as worker_module

    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    with patch("dw.worker.setup_logging"):
        worker = worker_module.WorkflowWorker(queue.Queue(), queue.Queue())
    worker.workflow_identity = ("path", "/w/warm.json")
    worker.loaded_pipelines["warm"] = object()
    definition = _refused_definition(kind)
    command = {
        "definition": definition,
        "file_spec": "/w/hostile.json",
        "source": "path",
        "arguments": {},
        "output_dir": str(tmp_path),
    }
    with patch.object(worker_module, "workflow_from_snapshot") as admit:
        admit.return_value = Workflow(
            definition, str(tmp_path), "/w/hostile.json", None
        )
        worker._handle_execute(command)
    messages = []
    while not worker.result_queue.empty():
        messages.append(worker.result_queue.get_nowait())
    assert "warm" in worker.loaded_pipelines
    assert worker.workflow_identity == ("path", "/w/warm.json")
    assert not any("Workflow changed" in str(m) for m in messages)
    assert not any("run_start" in str(m) for m in messages)
    failed = [m for m in messages if m["type"] == "error"]
    assert failed and "--trust-workflows" in str(failed[0])


def _validate(client, workflow):
    response = client.post("/api/validate?sizes=false", json={"workflow": workflow})
    assert response.status_code == 200, response.text
    return response.json()


def test_validate_reports_a_refused_pre_load_module(gallery_client, monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    body = _validate(gallery_client, _refused_definition("pre_load_modules"))
    assert body["valid"] is False
    errors = [e for e in body["errors"] if "pre_load_modules" in e["message"]]
    assert errors and errors[0]["path"] == "steps[0].pipeline", body["errors"]
    assert "--trust-workflows" in errors[0]["message"]


def test_trusted_validate_passes_a_pre_load_module(gallery_client, monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
    body = _validate(gallery_client, _refused_definition("pre_load_modules"))
    assert not any("pre_load_modules" in e["message"] for e in body["errors"])


def test_submit_refuses_a_pre_load_module_with_no_job(gallery_client, monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    response = gallery_client.post(
        "/api/jobs", json={"workflow": _refused_definition("pre_load_modules")}
    )
    assert response.status_code == 400, response.text
    assert "pre_load_modules" in response.text
    assert gallery_client.get("/api/jobs").json()["jobs"] == []
