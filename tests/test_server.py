"""Phase 1 server: job lifecycle over the API, SSE replay, cancellation,
request validation, and workflow browsing confinement."""

import json
import queue
import time

import pytest
from fastapi.testclient import TestClient

from dw.server.jobs import JobManager, TERMINAL_STATES
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

    def mark_crashed(self):
        self.worker_active = False
        self.crashed = True


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


def failing_script(command):
    # The worker caught an exception in the run and reported it
    yield {
        "type": "progress",
        "event": "step_start",
        "step": "gen",
        "index": 0,
        "total_steps": 1,
    }
    yield {
        "type": "error",
        "message": "Workflow execution error: CUDA out of memory",
        "traceback": "Traceback (most recent call last):\n  ...\nOutOfMemoryError",
    }


def crashing_script(command):
    # The worker process itself died - the message the watcher synthesizes
    yield {"type": "worker_crashed", "message": "exit code -11", "traceback": None}


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
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()

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
            prompt_dir=str(prompt_dir),
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


def test_failed_job_surfaces_the_error_and_traceback(server):
    """The failure path is what every user sees when a run goes wrong -
    the error and traceback must reach the detail, the event log must
    close with a terminal status, and history must remember the outcome."""
    with server(failing_script) as client:
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        detail = wait_for_status(client, job["id"], TERMINAL_STATES)
        assert detail["status"] == "failed"
        assert "CUDA out of memory" in detail["error"]
        assert detail["traceback"].startswith("Traceback")
        assert detail["manifest"] == []

        events = []
        with client.stream("GET", f"/api/jobs/{job['id']}/events") as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: ") :]))
        assert events[-1] == {
            "seq": events[-1]["seq"],
            "event": "job_status",
            "status": "failed",
        }

        manager = client.app.state.job_manager
        remembered = manager.history.get(job["id"])
        assert remembered["status"] == "failed"
        assert "CUDA out of memory" in remembered["error"]
        # the manager is idle again - a failure must not wedge the queue
        assert not manager.is_busy()


def test_a_worker_crash_fails_the_job_and_the_next_one_still_runs(server):
    """A crashed worker is marked so the next job respawns it rather than
    waiting forever on a dead process."""
    with server(crashing_script) as client:
        manager = client.app.state.job_manager
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        detail = wait_for_status(client, job["id"], TERMINAL_STATES)
        assert detail["status"] == "failed"
        assert detail["error"].startswith("Worker crashed")
        assert manager.worker_manager.crashed is True

        # the runner recovers: ensure_worker brings the (scripted) worker
        # back and the next submission runs to a terminal state
        manager.worker_manager.script = success_script
        again = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        assert wait_for_status(client, again["id"], TERMINAL_STATES)["status"] == (
            "succeeded"
        )


def test_job_detail_carries_its_queue_position_while_waiting(server):
    """The per-job detail says where a waiting job stands - the job page
    and the enhancer show it - and drops the field once it runs."""
    with server(hanging_script) as client:
        running = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, running["id"], ["running"])
        first = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        second = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()

        # the submission response already says so
        assert first["queue_position"] == 0
        assert second["queue_position"] == 1
        assert "queue_position" not in client.get(f"/api/jobs/{running['id']}").json()
        assert client.get(f"/api/jobs/{second['id']}").json()["queue_position"] == 1

        client.post(f"/api/jobs/{first['id']}/cancel")
        assert client.get(f"/api/jobs/{second['id']}").json()["queue_position"] == 0
        client.post(f"/api/jobs/{running['id']}/cancel")
        client.post(f"/api/jobs/{second['id']}/cancel")


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
        workflow["description"] = "Renders a small test image."
        client.put("/api/workflows/Detailed", json={"workflow": workflow})

        listing = client.get("/api/workflows").json()
        assert listing["details"]["Detailed"] == {
            "kinds": ["image"],
            "steps": 1,
            "variables": 1,
            "description": "Renders a small test image.",
            "prompt_refs": [],
        }
        assert listing["details"]["Basic"]["kinds"] == []


def test_workflow_details_name_their_prompt_references(server):
    """The listing says which stored prompts a workflow leans on - deleting
    a prompt warns from exactly this."""
    with server(success_script) as client:
        workflow = valid_workflow("leaning")
        workflow["variables"] = {"prompt": "prompt:minimax/fox"}
        workflow["steps"][0]["pipeline"]["arguments"]["negative"] = "prompt:scenic"
        client.put("/api/workflows/Leaning", json={"workflow": workflow})

        details = client.get("/api/workflows").json()["details"]
        assert details["Leaning"]["prompt_refs"] == ["minimax/fox", "scenic"]
        assert details["Basic"]["prompt_refs"] == []


def test_sse_resumes_from_last_event_id_header(server):
    """EventSource reconnects send Last-Event-ID; the stream must resume
    after it, not replay from the start."""
    with server(success_script) as client:
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, job["id"], ["succeeded"])

        with client.stream(
            "GET",
            f"/api/jobs/{job['id']}/events",
            headers={"Last-Event-ID": "2"},
        ) as response:
            seqs = [
                json.loads(line[len("data: ") :])["seq"]
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]
        assert seqs and seqs[0] == 3

        # an explicit ?after further along wins over a smaller header
        with client.stream(
            "GET",
            f"/api/jobs/{job['id']}/events?after=4",
            headers={"Last-Event-ID": "2"},
        ) as response:
            seqs = [
                json.loads(line[len("data: ") :])["seq"]
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]
        assert seqs and seqs[0] == 5


def test_sse_after_below_the_start_replays_the_whole_log(server):
    """events[after+1:] with after < -1 slices from the END of the log - a
    client asking for everything from before the beginning must not be
    handed only the tail."""
    with server(success_script) as client:
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, job["id"], ["succeeded"])

        def read_events(url):
            events = []
            with client.stream("GET", url) as response:
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[len("data: ") :]))
            return events

        full = read_events(f"/api/jobs/{job['id']}/events?after=-1")
        assert full[0]["seq"] == 0
        assert len(full) > 1
        # -2 is the value that exposes the slice: events[-1:] on the old
        # code is the last event alone. (A large negative like -99 slices
        # events[-98:], which on a short log is everything - and proves nothing.)
        clamped = read_events(f"/api/jobs/{job['id']}/events?after=-2")
        assert [e["seq"] for e in clamped] == [e["seq"] for e in full]


def test_rerun_endpoint_and_historical_job_surface(tmp_path):
    """The HTTP surface over history: historical detail, empty event
    stream, and rerun by id - including after a restart."""
    from dw.server.jobs import JobManager

    history = str(tmp_path / "jobs.sqlite")

    def make_client():
        manager = JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(success_script),
            history_path=history,
        )
        app = create_app(
            workflow_dir=str(tmp_path),
            output_dir=str(tmp_path / "outputs"),
            job_manager=manager,
        )
        return TestClient(app)

    with make_client() as client:
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, job["id"], ["succeeded"])

    with make_client() as client:  # restarted server
        detail = client.get(f"/api/jobs/{job['id']}").json()
        assert detail["historical"] is True and detail["status"] == "succeeded"

        # a historical job has no event log - the stream closes immediately
        with client.stream("GET", f"/api/jobs/{job['id']}/events") as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            assert list(response.iter_lines()) == []

        rerun = client.post(f"/api/jobs/{job['id']}/rerun")
        assert rerun.status_code == 201
        assert rerun.json()["id"] != job["id"]
        wait_for_status(client, rerun.json()["id"], ["succeeded"])

        assert client.post("/api/jobs/nonexistent/rerun").status_code == 404


def test_inline_base_dir_is_validated(server, tmp_path):
    """base_dir is HTTP-supplied path input - traversal and non-directories
    are refused at submission and validation, per the security rules."""
    with server(success_script) as client:
        for bad in ("../../etc", str(tmp_path / "missing"), "/etc/passwd"):
            response = client.post(
                "/api/jobs", json={"workflow": valid_workflow(), "base_dir": bad}
            )
            assert response.status_code == 400, bad
            response = client.post(
                "/api/validate", json={"workflow": valid_workflow(), "base_dir": bad}
            )
            assert response.status_code == 400, bad
        # nothing reached the worker
        assert client.app.state.job_manager.worker_manager.commands == []

        # a real directory is accepted
        good = client.post(
            "/api/jobs", json={"workflow": valid_workflow(), "base_dir": str(tmp_path)}
        )
        assert good.status_code == 201
        wait_for_status(client, good.json()["id"], ["succeeded"])


def test_job_for_file_escapes_like_wildcards(tmp_path):
    """'_' in a file name must not act as a single-character wildcard and
    attribute the file to a similarly named later job."""
    from dw.server.jobs import JobHistory, Job

    history = JobHistory(str(tmp_path / "jobs.sqlite"))

    def finished(job_id, file_name):
        job = Job({"workflow_name": "w", "arguments": {}})
        job.id = job_id
        job.manifest = [{"step": "s", "files": [f"/out/{file_name}"]}]
        job.status = "succeeded"
        job.finished_at = 1.0 if job_id == "older" else 2.0
        history.record(job)

    finished("older", "test_image-0.0.png")
    finished("newer", "testXimage-0.0.png")

    assert history.job_for_file("test_image-0.0.png")["id"] == "older"
    assert history.job_for_file("testXimage-0.0.png")["id"] == "newer"


def test_terminal_jobs_are_trimmed_from_memory(tmp_path):
    """Finished jobs beyond the replay-grace window leave memory; history
    still serves them."""
    from dw.server.jobs import JobManager, TERMINAL_JOBS_KEPT

    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    ids = []
    for _ in range(TERMINAL_JOBS_KEPT + 5):
        job = manager.submit(workflow=valid_workflow())
        deadline = time.time() + 5
        while job.status != "succeeded" and time.time() < deadline:
            time.sleep(0.01)
        ids.append(job.id)
    manager.shutdown()

    assert len(manager.jobs) == TERMINAL_JOBS_KEPT
    # the oldest are gone from memory but fully served from history
    evicted = manager.get(ids[0])
    assert evicted["historical"] is True and evicted["status"] == "succeeded"


class TestModelManager:
    """The hub cache endpoints: inventory, deletion, and the busy guard."""

    def fake_cache(self, tmp_path, monkeypatch):
        cache = tmp_path / "hub"
        snapshot = cache / "models--acme--tiny" / "snapshots" / "aaaa1111"
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").write_bytes(b"x" * 256)
        monkeypatch.setattr("dw.hub_cache.constants.HF_HUB_CACHE", str(cache))

    def test_models_are_listed_and_deleted(self, server, tmp_path, monkeypatch):
        self.fake_cache(tmp_path, monkeypatch)
        with server(success_script) as client:
            listed = client.get("/api/models").json()
            assert [r["repo_id"] for r in listed["repos"]] == ["acme/tiny"]
            deleted = client.delete("/api/models", params={"repo": "acme/tiny"})
            assert deleted.status_code == 200
            assert deleted.json()["freed"] >= 256
            assert client.get("/api/models").json()["repos"] == []

    def test_deleting_an_unknown_repo_is_404(self, server, tmp_path, monkeypatch):
        self.fake_cache(tmp_path, monkeypatch)
        with server(success_script) as client:
            response = client.delete("/api/models", params={"repo": "acme/other"})
            assert response.status_code == 404

    def test_delete_is_refused_while_a_job_is_active(
        self, server, tmp_path, monkeypatch
    ):
        self.fake_cache(tmp_path, monkeypatch)
        with server(hanging_script) as client:
            job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
            wait_for_status(client, job["id"], ["running"])

            response = client.delete("/api/models", params={"repo": "acme/tiny"})
            assert response.status_code == 409
            # Nothing was deleted out from under the run
            assert len(client.get("/api/models").json()["repos"]) == 1

            client.post(f"/api/jobs/{job['id']}/cancel")
            wait_for_status(client, job["id"], ["cancelled"])


def first_hangs_then_succeeds():
    """A script whose first execute never finishes (until cancelled) and
    whose later executes succeed - lets a test hold the runner busy while
    it rearranges the waiting queue."""
    count = {"n": 0}

    def script(command):
        count["n"] += 1
        if count["n"] == 1:
            yield from hanging_script(command)
        else:
            yield from success_script(command)

    return script


class TestQueueManagement:
    """The waiting queue is visible and reorderable; the running job is not."""

    def submit(self, client):
        return client.post("/api/jobs", json={"workflow": valid_workflow()}).json()

    def test_moved_job_runs_first(self, server):
        with server(first_hangs_then_succeeds()) as client:
            blocker = self.submit(client)
            wait_for_status(client, blocker["id"], ["running"])
            second = self.submit(client)
            third = self.submit(client)

            # Positions are reported while jobs wait
            jobs = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
            assert jobs[second["id"]]["queue_position"] == 0
            assert jobs[third["id"]]["queue_position"] == 1

            response = client.post(
                f"/api/jobs/{third['id']}/move", json={"direction": "front"}
            )
            assert response.json()["queue"] == [third["id"], second["id"]]

            client.post(f"/api/jobs/{blocker['id']}/cancel")
            wait_for_status(client, third["id"], ["succeeded"])
            wait_for_status(client, second["id"], ["succeeded"])

            # The promoted job genuinely ran first
            third_started = client.get(f"/api/jobs/{third['id']}").json()["started_at"]
            second_started = client.get(f"/api/jobs/{second['id']}").json()[
                "started_at"
            ]
            assert third_started < second_started

    def test_running_and_unknown_jobs_do_not_move(self, server):
        with server(first_hangs_then_succeeds()) as client:
            blocker = self.submit(client)
            wait_for_status(client, blocker["id"], ["running"])

            running = client.post(
                f"/api/jobs/{blocker['id']}/move", json={"direction": "up"}
            )
            assert running.status_code == 409
            unknown = client.post("/api/jobs/nope/move", json={"direction": "up"})
            assert unknown.status_code == 404
            bad = client.post(
                f"/api/jobs/{blocker['id']}/move", json={"direction": "sideways"}
            )
            assert bad.status_code == 400

            client.post(f"/api/jobs/{blocker['id']}/cancel")
            wait_for_status(client, blocker["id"], ["cancelled"])

    def test_cancelled_queued_job_leaves_the_queue_order(self, server):
        with server(first_hangs_then_succeeds()) as client:
            blocker = self.submit(client)
            wait_for_status(client, blocker["id"], ["running"])
            second = self.submit(client)
            third = self.submit(client)

            client.post(f"/api/jobs/{second['id']}/cancel")
            jobs = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
            assert jobs[third["id"]]["queue_position"] == 0
            assert "queue_position" not in jobs[second["id"]]

            client.post(f"/api/jobs/{blocker['id']}/cancel")
            wait_for_status(client, third["id"], ["succeeded"])


class TestModelDownloads:
    """The download endpoints wire the DownloadManager to HTTP."""

    def make_manager(self):
        from dw.hub_cache import DownloadManager

        def instant(repo_id, tqdm_class=None):
            tracker = tqdm_class(total=50)
            tracker.update(50)

        class Info:
            siblings = []

        return DownloadManager(download_fn=instant, info_fn=lambda repo_id: Info())

    def test_download_lifecycle_over_http(self, server, tmp_path):
        from dw.server.jobs import JobManager
        from dw.server.app import create_app

        manager = JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(success_script),
            history_path=str(tmp_path / "jobs.sqlite"),
        )
        app = create_app(
            workflow_dir=str(tmp_path),
            output_dir=str(tmp_path / "outputs"),
            job_manager=manager,
            download_manager=self.make_manager(),
        )
        with TestClient(app) as client:
            started = client.post("/api/models/download", json={"repo_id": "acme/tiny"})
            assert started.status_code == 202
            download_id = started.json()["id"]

            deadline = time.time() + 5
            while time.time() < deadline:
                listed = client.get("/api/models/downloads").json()["downloads"]
                entry = next(d for d in listed if d["id"] == download_id)
                if entry["status"] == "completed":
                    break
                time.sleep(0.02)
            assert entry["status"] == "completed"
            assert entry["downloaded"] == 50

            bad = client.post("/api/models/download", json={"repo_id": "no//pe"})
            assert bad.status_code == 400
            unknown = client.post("/api/models/downloads/nope/cancel")
            assert unknown.status_code == 404

    def test_delete_and_update_are_refused_while_a_download_runs(self, tmp_path):
        """A download writing into the hub cache is the same hazard a running
        job is: deleting those files, or letting pip replace package files
        underneath it, corrupts what the download is producing."""
        import threading
        from types import SimpleNamespace
        from dw.hub_cache import DownloadManager
        from dw.server.updater import DiffusersUpdater

        release = threading.Event()
        started = threading.Event()

        def slow_download(repo_id, tqdm_class=None):
            tracker = tqdm_class(total=10)
            tracker.update(1)
            started.set()
            release.wait(timeout=5)

        class Info:
            siblings = []

        downloads = DownloadManager(
            download_fn=slow_download, info_fn=lambda repo_id: Info()
        )
        manager = JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(success_script),
            history_path=str(tmp_path / "jobs.sqlite"),
        )
        pip = SimpleNamespace(returncode=0, stdout="", stderr="")
        app = create_app(
            workflow_dir=str(tmp_path),
            output_dir=str(tmp_path / "outputs"),
            job_manager=manager,
            download_manager=downloads,
            diffusers_updater=DiffusersUpdater(run_fn=lambda: pip),
        )
        with TestClient(app) as client:
            assert (
                client.post(
                    "/api/models/download", json={"repo_id": "acme/tiny"}
                ).status_code
                == 202
            )
            assert started.wait(timeout=5)

            refused_delete = client.delete("/api/models?repo=acme/other")
            assert refused_delete.status_code == 409
            assert "download is in progress" in refused_delete.json()["detail"]

            refused_update = client.post("/api/system/diffusers/update")
            assert refused_update.status_code == 409
            assert "download is in progress" in refused_update.json()["detail"]

            release.set()

    def test_a_download_is_cancellable_from_the_moment_it_is_listed(self, monkeypatch):
        """The entry used to be published under the lock and given its cancel
        event afterwards; a cancel from another thread in between raised
        KeyError. The gap held exactly one call - threading.Event() - so a
        hook on it stands in for that other thread deterministically."""
        import threading
        from dw.hub_cache import DownloadManager

        class Info:
            siblings = []

        downloads = DownloadManager(
            download_fn=lambda repo_id, tqdm_class=None: None,
            info_fn=lambda repo_id: Info(),
        )

        real_event = threading.Event

        def event_that_cancels_whatever_is_listed():
            for entry in downloads.status_list():
                if entry["status"] == "downloading":
                    downloads.cancel(entry["id"])
            return real_event()

        monkeypatch.setattr(threading, "Event", event_that_cancels_whatever_is_listed)

        # On the old ordering this raised KeyError('_cancel') out of start()
        started = downloads.start("acme/tiny")

        deadline = time.time() + 5
        while time.time() < deadline:
            if downloads.status(started["id"])["status"] != "downloading":
                break
            time.sleep(0.02)
        assert downloads.status(started["id"])["status"] == "completed"


class TestTaskDescription:
    """The task schema endpoint the editor's forms consume."""

    def test_describe_task_over_http(self, server):
        with server(success_script) as client:
            description = client.get("/api/tasks/qr_code").json()
            names = [p["name"] for p in description["parameters"]]
            assert "qr_code_contents" in names
            assert client.get("/api/tasks/not_a_task").status_code == 404

    def test_task_typos_surface_in_validation(self, server):
        workflow = {
            "id": "task_typo",
            "steps": [
                {
                    "name": "join",
                    "task": {
                        "command": "concat_videos",
                        "arguments": {"videos": [], "trim_framse": 1},
                    },
                }
            ],
        }
        with server(success_script) as client:
            result = client.post("/api/validate", json={"workflow": workflow}).json()
            assert result["valid"]
            assert any("trim_framse" in warning for warning in result["warnings"])


class TestDiffusersUpdate:
    """The diffusers update endpoints run pip in the background and guard
    the busy worker."""

    def make_client(self, tmp_path, run_fn, manager=None):
        from dw.server.updater import DiffusersUpdater

        manager = manager or JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(success_script),
            history_path=str(tmp_path / "jobs.sqlite"),
        )
        app = create_app(
            workflow_dir=str(tmp_path),
            output_dir=str(tmp_path / "outputs"),
            job_manager=manager,
            diffusers_updater=DiffusersUpdater(run_fn=run_fn),
        )
        return TestClient(app)

    @staticmethod
    def wait_for_update(client, statuses, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = client.get("/api/system/diffusers").json()
            if state["status"] in statuses:
                return state
            time.sleep(0.02)
        raise AssertionError(f"update never reached {statuses}: {state}")

    def test_update_lifecycle(self, tmp_path):
        from types import SimpleNamespace

        pip = SimpleNamespace(returncode=0, stdout="Successfully installed", stderr="")
        with self.make_client(tmp_path, lambda: pip) as client:
            state = client.get("/api/system/diffusers").json()
            assert state["status"] == "idle"
            assert "version" in state and "commit" in state

            started = client.post("/api/system/diffusers/update")
            assert started.status_code == 202
            done = self.wait_for_update(client, ["succeeded", "failed"])
            assert done["status"] == "succeeded"
            assert "Successfully installed" in done["log"]

    def test_failed_update_reports_pip_output(self, tmp_path):
        from types import SimpleNamespace

        pip = SimpleNamespace(returncode=1, stdout="", stderr="No space left")
        with self.make_client(tmp_path, lambda: pip) as client:
            client.post("/api/system/diffusers/update")
            done = self.wait_for_update(client, ["succeeded", "failed"])
            assert done["status"] == "failed"
            assert "exited with code 1" in done["error"]
            assert "No space left" in done["log"]

    def test_second_update_refused_while_running(self, tmp_path):
        import threading
        from types import SimpleNamespace

        release = threading.Event()

        def slow_pip():
            release.wait(timeout=5)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with self.make_client(tmp_path, slow_pip) as client:
            assert client.post("/api/system/diffusers/update").status_code == 202
            refused = client.post("/api/system/diffusers/update")
            assert refused.status_code == 409
            release.set()
            self.wait_for_update(client, ["succeeded"])

    def test_update_refused_while_a_job_runs(self, tmp_path):
        from types import SimpleNamespace

        manager = JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(success_script),
            history_path=str(tmp_path / "jobs.sqlite"),
        )
        manager.is_busy = lambda: True
        pip = SimpleNamespace(returncode=0, stdout="", stderr="")
        with self.make_client(tmp_path, lambda: pip, manager=manager) as client:
            refused = client.post("/api/system/diffusers/update")
            assert refused.status_code == 409
            assert "running or queued" in refused.json()["detail"]


class TestPromptLibrary:
    """CRUD over the prompt directory, mirroring workflow browsing:
    subfolder round trips, schema validation before disk, confinement."""

    def test_save_prompt_roundtrip_and_confinement(self, server, tmp_path):
        with server(success_script) as client:
            prompt = {
                "text": "a red fox at dawn",
                "description": "test prompt",
                "intended_model": "minimax-h3",
                "tags": ["wildlife"],
            }
            response = client.put("/api/prompts/minimax/Fox", json={"prompt": prompt})
            assert response.status_code == 200
            assert (
                client.get("/api/prompts/minimax/Fox").json()["text"]
                == "a red fox at dawn"
            )

            listing = client.get("/api/prompts").json()
            assert "minimax/Fox" in listing["prompts"]
            detail = listing["details"]["minimax/Fox"]
            assert detail["description"] == "test prompt"
            assert detail["intended_model"] == "minimax-h3"
            assert detail["tags"] == ["wildlife"]
            # the text rides along - the editors show it as the tooltip
            # wherever a prompt: reference stands in for it
            assert detail["text"] == "a red fox at dawn"

            # schema-invalid definitions never reach disk
            response = client.put(
                "/api/prompts/Broken", json={"prompt": {"description": "no text"}}
            )
            assert response.status_code == 400
            assert client.get("/api/prompts/Broken").status_code == 404

            # text that is itself a reference is refused - it would be
            # resolved again at run time
            response = client.put(
                "/api/prompts/Sneaky",
                json={"prompt": {"text": "previous_result:gen"}},
            )
            assert response.status_code == 400

            # delete: removes exactly the named prompt, confined the same way
            assert client.delete("/api/prompts/minimax/Fox").status_code == 200
            assert client.get("/api/prompts/minimax/Fox").status_code == 404
            assert client.delete("/api/prompts/minimax/Fox").status_code == 404

            # writes stay confined to the prompt directory
            for evasion in ("/api/prompts/../escape", "/api/prompts/..%2Fescape"):
                response = client.put(evasion, json={"prompt": {"text": "t"}})
                assert response.status_code in (400, 404, 405), evasion
            assert not (tmp_path / "escape.json").exists()

    def test_unreferenceable_names_are_refused(self, server, tmp_path):
        # A save the API accepted but no 'prompt:' reference could ever
        # load would be a trap - the name rule is enforced here too
        with server(success_script) as client:
            for name in ("a/b/c", "My%20Prompt", ".hidden"):
                response = client.put(
                    f"/api/prompts/{name}", json={"prompt": {"text": "t"}}
                )
                assert response.status_code == 400, name
                assert "prompt" in response.json()["detail"].lower()

            # a stray file already on disk that breaks the rule is not listed
            deep = tmp_path / "prompts" / "a" / "b"
            deep.mkdir(parents=True)
            (deep / "c.json").write_text(json.dumps({"text": "orphan"}))
            assert "a/b/c" not in client.get("/api/prompts").json()["prompts"]

    def test_prompt_schema_is_served(self, server):
        with server(success_script) as client:
            schema = client.get("/api/prompt-schema").json()
            assert "text" in schema["properties"]
            assert schema["required"] == ["text"]


class TestEnhance:
    """The enhance endpoint queues an ordinary job whose single saved text
    file is the return channel."""

    def test_presets_are_listed(self, server):
        with server(success_script) as client:
            presets = client.get("/api/enhancers").json()["presets"]
            keys = {p["key"] for p in presets}
            assert "h3" in keys
            for preset in presets:
                assert preset["label"]
                assert preset["default_model"]
                assert isinstance(preset["models"], list)

    def test_enhance_queues_a_job_that_delegates_to_the_builtin(self, server):
        with server(success_script) as client:
            response = client.post(
                "/api/enhance", json={"idea": "a cat in the rain", "preset": "h3"}
            )
            assert response.status_code == 201
            job = response.json()
            assert wait_for_status(client, job["id"], ["succeeded"])

            spec = client.get(f"/api/jobs/{job['id']}").json()
            assert spec["workflow"].startswith("enhance_")

    def test_enhance_workflows_save_their_text(self):
        from dw.server.enhancers import build_enhance_workflow

        for preset in ("h3", "t2i"):
            definition = build_enhance_workflow(preset, "an idea")
            step = definition["steps"][0]
            assert step["result"] == {"content_type": "text/plain", "save": True}

        # ids are unique so output files never collide
        first = build_enhance_workflow("h3", "an idea")["id"]
        second = build_enhance_workflow("h3", "an idea")["id"]
        assert first != second

        # the h3 preset delegates to the builtin enhancer workflow
        step = build_enhance_workflow("h3", "an idea")["steps"][0]
        assert step["workflow"]["path"] == "builtin:h3_context_ir.json"
        assert step["workflow"]["arguments"]["prompt"] == "an idea"

    def test_enhance_workflows_are_schema_valid(self):
        from dw.schema import load_schema, validate_data
        from dw.server.enhancers import build_enhance_workflow

        for preset in ("h3", "t2i"):
            definition = build_enhance_workflow(
                preset, "an idea", model_name="Qwen/Qwen2.5-1.5B-Instruct"
            )
            status, message = validate_data(definition, load_schema("workflow"))
            assert status, message

    def test_an_unknown_preset_is_a_client_error(self, server):
        with server(success_script) as client:
            response = client.post(
                "/api/enhance", json={"idea": "an idea", "preset": "nope"}
            )
            assert response.status_code == 400
            assert "Unknown enhancer preset" in response.json()["detail"]

    def test_an_empty_idea_is_a_client_error(self, server):
        with server(success_script) as client:
            response = client.post("/api/enhance", json={"idea": "  ", "preset": "h3"})
            assert response.status_code == 400
