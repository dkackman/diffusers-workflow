"""Phase 1 server: job lifecycle over the API, SSE replay, cancellation,
request validation, and workflow browsing confinement."""

import json
import queue
import time

import pytest
from fastapi.testclient import TestClient

from dw.server.jobs import JobManager
from dw.server.app import create_app


def valid_workflow(job_id="server_test"):
    return {
        "id": job_id,
        "variables": {"prompt": "d"},
        "steps": [
            {
                "name": "gen",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}", "no_generator": True},
                    "from_pretrained_arguments": {"model_name": "m"},
                    "arguments": {"prompt": "variable:prompt"},
                },
            }
        ],
    }


class ScriptedWorkerManager:
    """Answers execute commands with a scripted message sequence."""

    def __init__(self, script=None):
        self.script = script
        self.commands = []
        self.worker_active = False
        self.worker_process = None
        self._results = queue.Queue()

    def ensure_worker(self, log_level="INFO"):
        self.worker_active = True

    def send_command(self, command):
        self.commands.append(command)
        if command["type"] == "execute":
            for message in self.script(command):
                self._results.put(message)
        elif command["type"] == "cancel":
            self._results.put({"type": "cancelled", "message": "cancelled"})
        elif command["type"] == "memory_status":
            self._results.put(
                {"type": "memory_status", "info": {"gpu_available": True}}
            )

    def get_result(self, timeout=None):
        return self._results.get(timeout=timeout if timeout is not None else 10)

    def shutdown_worker(self):
        self.worker_active = False

    def cancel(self):
        self.send_command({"type": "cancel"})


def success_script(command):
    yield {
        "type": "progress",
        "event": "step_start",
        "step": "gen",
        "index": 0,
        "total_steps": 1,
    }
    yield {"type": "progress", "event": "pipeline_step", "step": 1, "total_steps": 2}
    yield {
        "type": "progress",
        "event": "step_end",
        "step": "gen",
        "index": 0,
        "total_steps": 1,
        "files": ["/out/a.png"],
    }
    yield {
        "type": "success",
        "message": "ok",
        "run_count": 1,
        "manifest": [{"step": "gen", "files": ["/out/a.png"]}],
    }


def hanging_script(command):
    # Emits progress but no terminal message - the job ends only on cancel
    yield {
        "type": "progress",
        "event": "step_start",
        "step": "gen",
        "index": 0,
        "total_steps": 1,
    }


@pytest.fixture
def server(tmp_path):
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir()
    (workflow_dir / "Basic.json").write_text(json.dumps(valid_workflow("basic")))

    def make(script):
        manager = JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(script),
            history_path=str(tmp_path / "jobs.sqlite"),
        )
        app = create_app(
            workflow_dir=str(workflow_dir),
            output_dir=str(tmp_path / "outputs"),
            job_manager=manager,
        )
        return TestClient(app)

    return make


def wait_for_status(client, job_id, statuses, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        detail = client.get(f"/api/jobs/{job_id}").json()
        if detail["status"] in statuses:
            return detail
        time.sleep(0.02)
    raise AssertionError(f"job never reached {statuses}: {detail}")


def test_job_lifecycle_success(server):
    with server(success_script) as client:
        response = client.post(
            "/api/jobs",
            json={"workflow": valid_workflow(), "arguments": {"prompt": "hi"}},
        )
        assert response.status_code == 201
        job = response.json()
        assert job["workflow"] == "server_test"

        detail = wait_for_status(client, job["id"], ["succeeded"])
        assert detail["manifest"] == [{"step": "gen", "files": ["/out/a.png"]}]

        # the worker received the inline definition and the arguments
        manager = client.app.state.job_manager
        execute = [c for c in manager.worker_manager.commands if c["type"] == "execute"]
        assert execute[0]["workflow"]["id"] == "server_test"
        assert execute[0]["arguments"] == {"prompt": "hi"}


def test_sse_stream_and_replay(server):
    with server(success_script) as client:
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, job["id"], ["succeeded"])

        def read_events(url):
            events = []
            with client.stream("GET", url) as response:
                assert response.headers["content-type"].startswith("text/event-stream")
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[len("data: ") :]))
            return events

        events = read_events(f"/api/jobs/{job['id']}/events")
        assert [e["seq"] for e in events] == list(range(len(events)))
        names = [e["event"] for e in events]
        assert "pipeline_step" in names
        assert names[-1] == "job_status" and events[-1]["status"] == "succeeded"

        # reconnecting after seq N replays only what followed
        replay = read_events(f"/api/jobs/{job['id']}/events?after={events[2]['seq']}")
        assert [e["seq"] for e in replay] == [e["seq"] for e in events[3:]]


def test_cancel_running_and_queued(server):
    with server(hanging_script) as client:
        running = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, running["id"], ["running"])
        queued = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()

        # queued job cancels immediately, without ever running
        response = client.post(f"/api/jobs/{queued['id']}/cancel")
        assert response.json()["status"] == "cancelled"

        # running job cancels through the worker
        client.post(f"/api/jobs/{running['id']}/cancel")
        detail = wait_for_status(client, running["id"], ["cancelled"])
        assert detail["status"] == "cancelled"

        manager = client.app.state.job_manager
        executes = [
            c for c in manager.worker_manager.commands if c["type"] == "execute"
        ]
        assert len(executes) == 1, "the cancelled queued job must never execute"


def test_submit_validation(server):
    with server(success_script) as client:
        # neither and both workflow sources
        assert client.post("/api/jobs", json={}).status_code == 400
        assert (
            client.post(
                "/api/jobs",
                json={"workflow": valid_workflow(), "workflow_path": "x.json"},
            ).status_code
            == 400
        )
        # schema-invalid inline workflow
        assert (
            client.post("/api/jobs", json={"workflow": {"steps": "nope"}}).status_code
            == 400
        )
        # nothing reached the worker
        assert client.app.state.job_manager.worker_manager.commands == []


def test_workflow_browsing_and_confinement(server):
    with server(success_script) as client:
        listing = client.get("/api/workflows").json()
        assert listing["workflows"] == ["Basic"]

        workflow = client.get("/api/workflows/Basic").json()
        assert workflow["id"] == "basic"

        assert client.get("/api/workflows/../secret").status_code == 404
        assert client.get("/api/workflows/nope").status_code == 404


def test_health_and_memory(server):
    with server(success_script) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok"

        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, job["id"], ["succeeded"])
        memory = client.get("/api/memory").json()
        assert memory["live"] is True
        assert memory["info"]["gpu_available"] is True


def test_introspection_endpoints(server):
    with server(success_script) as client:
        pipelines = client.get("/api/pipelines").json()["pipelines"]
        assert "ZImagePipeline" in pipelines

        description = client.get("/api/pipelines/ZImagePipeline").json()
        names = [p["name"] for p in description["parameters"]]
        assert "prompt" in names and "num_inference_steps" in names

        assert client.get("/api/pipelines/NoSuchPipeline").status_code == 404
        assert client.get("/api/pipelines/..%2Fetc").status_code == 404

        tasks = client.get("/api/tasks").json()
        assert tasks["commands"] and tasks["image_processors"]


def test_validate_endpoint_flags_signature_typos(server):
    with server(success_script) as client:
        workflow = valid_workflow()
        workflow["steps"][0]["pipeline"]["configuration"][
            "component_type"
        ] = "ZImagePipeline"
        workflow["steps"][0]["pipeline"]["arguments"]["guidance_scael"] = 3

        result = client.post("/api/validate", json={"workflow": workflow}).json()
        assert result["valid"] is True
        assert any("guidance_scael" in w for w in result["warnings"])

        # schema failure reports invalid, not a warning
        result = client.post(
            "/api/validate", json={"workflow": {"steps": "nope"}}
        ).json()
        assert result["valid"] is False and result["error"]


def test_submission_carries_argument_warnings(server):
    with server(success_script) as client:
        workflow = valid_workflow()
        workflow["steps"][0]["pipeline"]["configuration"][
            "component_type"
        ] = "ZImagePipeline"
        workflow["steps"][0]["pipeline"]["arguments"]["guidance_scael"] = 3
        job = client.post("/api/jobs", json={"workflow": workflow}).json()
        assert any("guidance_scael" in w for w in job["warnings"])
        # a warning does not block the run
        assert wait_for_status(client, job["id"], ["succeeded"])


def test_foreign_origin_requests_are_rejected(server):
    with server(success_script) as client:
        response = client.post(
            "/api/jobs",
            json={"workflow": valid_workflow()},
            headers={"Origin": "https://evil.example"},
        )
        assert response.status_code == 403

        # local origins and origin-less requests (curl, scripts) pass
        assert (
            client.get(
                "/api/health", headers={"Origin": "http://localhost:8765"}
            ).status_code
            == 200
        )
        assert client.get("/api/health").status_code == 200


def test_save_workflow_roundtrip_and_confinement(server, tmp_path):
    with server(success_script) as client:
        workflow = valid_workflow("saved")
        response = client.put("/api/workflows/sub/Saved", json={"workflow": workflow})
        assert response.status_code == 200
        assert client.get("/api/workflows/sub/Saved").json()["id"] == "saved"
        assert "sub/Saved" in client.get("/api/workflows").json()["workflows"]

        # schema-invalid definitions never reach disk
        response = client.put(
            "/api/workflows/Broken", json={"workflow": {"steps": "nope"}}
        )
        assert response.status_code == 400
        assert client.get("/api/workflows/Broken").status_code == 404

        # delete: removes exactly the named workflow, confined the same way
        assert client.delete("/api/workflows/sub/Saved").status_code == 200
        assert client.get("/api/workflows/sub/Saved").status_code == 404
        assert client.delete("/api/workflows/sub/Saved").status_code == 404
        assert client.delete("/api/workflows/..%2Fconftest").status_code == 404

        # writes stay confined to the workflow directory: a literal ../ is
        # normalized away before routing; an encoded one reaches the route
        # and must be refused by path validation
        for evasion in ("/api/workflows/../escape", "/api/workflows/..%2Fescape"):
            response = client.put(evasion, json={"workflow": valid_workflow()})
            assert response.status_code in (400, 404, 405), evasion
        assert not (tmp_path / "escape.json").exists()


def test_gallery_lists_media_and_reads_metadata(server, tmp_path):
    from PIL import Image
    from dw.result import Result, read_embedded_metadata

    with server(success_script) as client:
        outputs = tmp_path / "outputs"

        # a plain image, a video stand-in, and a non-media file
        Image.new("RGB", (4, 4)).save(outputs / "plain-gen.0-0.0.png")
        (outputs / "clip-gen.0-0.0.mp4").write_bytes(b"\x00" * 10)
        (outputs / "notes.txt").write_text("not media")

        # an image saved the way the engine saves it, metadata embedded
        result = Result({"content_type": "image/png", "embed_metadata": True})
        result.set_metadata(
            {"step_name": "gen", "workflow": valid_workflow("from_image")}
        )
        result._save_image_with_metadata(
            Image.new("RGB", (4, 4)), str(outputs / "meta-gen.0-0.0.png"), "image/png"
        )

        listing = client.get("/api/gallery").json()
        names = {f["name"] for f in listing["files"]}
        assert names == {
            "plain-gen.0-0.0.png",
            "clip-gen.0-0.0.mp4",
            "meta-gen.0-0.0.png",
        }
        kinds = {f["name"]: f["kind"] for f in listing["files"]}
        assert kinds["clip-gen.0-0.0.mp4"] == "video"

        # metadata endpoint: full workflow comes back for the editor
        response = client.get("/api/gallery/meta-gen.0-0.0.png/metadata").json()
        assert response["metadata"]["workflow"]["id"] == "from_image"
        # an image without metadata answers null, not an error
        assert (
            client.get("/api/gallery/plain-gen.0-0.0.png/metadata").json()["metadata"]
            is None
        )

        # confinement
        assert client.get("/api/gallery/..%2Fsecret.png/metadata").status_code == 404

        # the read-side mirror also handles EXIF (jpeg) round trips
        result._save_image_with_metadata(
            Image.new("RGB", (4, 4)), str(outputs / "meta.jpg"), "image/jpeg"
        )
        assert read_embedded_metadata(str(outputs / "meta.jpg"))["step_name"] == "gen"


def test_embed_metadata_carries_the_workflow_definition(tmp_path):
    """A run with embed_metadata writes an image the gallery can reopen."""
    from unittest.mock import patch
    from dw.workflow import Workflow
    from dw.pipeline_processors.pipeline import Pipeline
    from dw.result import read_embedded_metadata
    from tests.test_events import FakePipeline

    workflow_def = valid_workflow("reopenable")
    workflow_def["steps"][0]["result"] = {
        "content_type": "image/png",
        "embed_metadata": True,
    }

    def mock_load(self, shared_components):
        self.pipeline = FakePipeline()

    workflow = Workflow(workflow_def, str(tmp_path), "test.json")
    with patch.object(Pipeline, "load", mock_load):
        with patch("dw.workflow.empty_device_cache"):
            workflow.run({}, previous_pipelines={})

    saved = workflow.manifest[0]["files"][0]
    metadata = read_embedded_metadata(saved)
    assert metadata["workflow"]["id"] == "reopenable"
    assert metadata["step_name"] == "gen"
    # the seed travels with the recipe, so reopening reproduces this image
    assert isinstance(metadata["seed"], int)


def test_job_history_survives_restart_and_reruns(tmp_path):
    """Finished jobs persist; a new manager lists them and can rerun them."""
    from dw.server.jobs import JobManager

    history = str(tmp_path / "jobs.sqlite")

    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=history,
    )
    job = manager.submit(workflow=valid_workflow(), arguments={"prompt": "hi"})
    deadline = time.time() + 5
    while job.status not in ("succeeded", "failed") and time.time() < deadline:
        time.sleep(0.02)
    assert job.status == "succeeded"
    manager.shutdown()

    # a fresh manager - the restarted server - sees the finished job
    revived = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=history,
    )
    summaries = revived.list()
    assert [s["id"] for s in summaries] == [job.id]
    assert summaries[0]["historical"] is True

    detail = revived.get(job.id)
    assert detail["status"] == "succeeded"
    assert detail["manifest"] == [{"step": "gen", "files": ["/out/a.png"]}]
    assert detail["arguments"] == {"prompt": "hi"}

    # and can run it again from the stored spec
    rerun = revived.rerun(job.id)
    assert rerun is not None and rerun.id != job.id
    assert rerun.spec["arguments"] == {"prompt": "hi"}
    revived.shutdown()

    assert revived.rerun("nonexistent") is None


def test_gallery_delete_and_job_linkage(server, tmp_path):
    from PIL import Image

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        Image.new("RGB", (4, 4)).save(outputs / "victim.png")

        # a finished job whose manifest names an output file links back to it
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, job["id"], ["succeeded"])
        Image.new("RGB", (4, 4)).save(outputs / "a.png")  # matches /out/a.png name
        linked = client.get("/api/gallery/a.png/metadata").json()
        assert linked["job"]["id"] == job["id"]

        # delete removes exactly the named file, confined to outputs
        assert client.delete("/api/gallery/victim.png").status_code == 200
        assert not (outputs / "victim.png").exists()
        assert client.delete("/api/gallery/victim.png").status_code == 404
        # encoded traversal: refused by routing (405) or validation (404)
        assert client.delete("/api/gallery/..%2Fjobs.sqlite").status_code in (404, 405)
        assert (tmp_path / "jobs.sqlite").exists()


def test_workflow_listing_carries_details(server):
    with server(success_script) as client:
        workflow = valid_workflow("detailed")
        workflow["steps"][0]["result"] = {"content_type": "image/png"}
        client.put("/api/workflows/Detailed", json={"workflow": workflow})

        listing = client.get("/api/workflows").json()
        assert listing["details"]["Detailed"] == {"kinds": ["image"], "variables": 1}
        assert listing["details"]["Basic"] == {"kinds": [], "variables": 1}
