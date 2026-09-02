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


def test_submit_accepts_a_stored_workflow_name(server, tmp_path):
    """The names /api/workflows hands out are what an agent has in hand, so
    they must be submittable as-is - with or without .json, nested included."""
    with server(success_script) as client:
        nested = tmp_path / "workflows" / "nested"
        nested.mkdir()
        (nested / "Deep.json").write_text(json.dumps(valid_workflow("deep")))

        for name, expected in [
            ("Basic", "basic"),
            ("Basic.json", "basic"),
            ("nested/Deep", "deep"),
        ]:
            response = client.post("/api/jobs", json={"workflow_path": name})
            assert response.status_code == 201, (name, response.json())
            assert response.json()["workflow"] == expected

        # the worker gets the resolved file, not the catalog name
        manager = client.app.state.job_manager
        executes = [
            c for c in manager.worker_manager.commands if c["type"] == "execute"
        ]
        assert executes[0]["workflow_path"].endswith("Basic.json")


def test_submit_accepts_a_real_path(server, tmp_path):
    with server(success_script) as client:
        path = tmp_path / "loose.json"
        path.write_text(json.dumps(valid_workflow("loose")))

        response = client.post("/api/jobs", json={"workflow_path": str(path)})

        assert response.status_code == 201
        assert response.json()["workflow"] == "loose"


def test_submit_rejects_a_traversal_shaped_name(server):
    with server(success_script) as client:
        assert (
            client.post("/api/jobs", json={"workflow_path": "../secret"}).status_code
            == 400
        )
        assert (
            client.post("/api/jobs", json={"workflow_path": "nope"}).status_code == 400
        )
        assert client.app.state.job_manager.worker_manager.commands == []


def test_validate_accepts_a_stored_workflow_name(server, tmp_path):
    with server(success_script) as client:
        result = client.post("/api/validate", json={"workflow_path": "Basic"}).json()
        assert result["valid"] is True and result["error"] is None

        # the stored file is what gets checked, warnings and all
        typo = valid_workflow("typo")
        typo["steps"][0]["pipeline"]["configuration"][
            "component_type"
        ] = "ZImagePipeline"
        typo["steps"][0]["pipeline"]["arguments"]["guidance_scael"] = 3
        (tmp_path / "workflows" / "Typo.json").write_text(json.dumps(typo))

        result = client.post("/api/validate", json={"workflow_path": "Typo"}).json()

        assert result["valid"] is True
        assert any("guidance_scael" in w for w in result["warnings"])


def test_validate_requires_exactly_one_workflow_source(server):
    with server(success_script) as client:
        assert client.post("/api/validate", json={}).status_code == 400
        assert (
            client.post(
                "/api/validate",
                json={"workflow": valid_workflow(), "workflow_path": "Basic"},
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/validate", json={"workflow_path": "../secret"}
            ).status_code
            == 400
        )


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


def test_gallery_paginates_and_groups_by_workflow_folder(server, tmp_path):
    """Outputs nested under a workflow subfolder (dw/workflow.py's
    effective_output_dir) still show up in the gallery, tagged with their
    folder, and a limit/offset page returns exactly that page plus an
    accurate total - so an output directory with more files than any single
    page still has every file reachable."""
    from PIL import Image

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        (outputs / "ltx").mkdir()

        # five files at the root, two nested under 'ltx'
        for i in range(5):
            Image.new("RGB", (2, 2)).save(outputs / f"root-{i}.png")
        for i in range(2):
            Image.new("RGB", (2, 2)).save(outputs / "ltx" / f"nested-{i}.png")

        full = client.get("/api/gallery").json()
        assert full["total"] == 7
        assert set(full["folders"]) == {"", "ltx"}
        names = {f["name"] for f in full["files"]}
        assert "ltx/nested-0.png" in names
        assert "root-0.png" in names
        nested_entry = next(f for f in full["files"] if f["name"] == "ltx/nested-0.png")
        assert nested_entry["folder"] == "ltx"

        # a page smaller than the total returns exactly that many, and the
        # next page picks up where it left off with no overlap
        first_page = client.get("/api/gallery?limit=3&offset=0").json()
        assert len(first_page["files"]) == 3
        assert first_page["total"] == 7
        second_page = client.get("/api/gallery?limit=3&offset=3").json()
        assert len(second_page["files"]) == 3
        first_names = {f["name"] for f in first_page["files"]}
        second_names = {f["name"] for f in second_page["files"]}
        assert first_names.isdisjoint(second_names)

        # filtering by folder narrows both the listing and its total
        ltx_only = client.get("/api/gallery?folder=ltx").json()
        assert ltx_only["total"] == 2
        assert all(f["folder"] == "ltx" for f in ltx_only["files"])

        root_only = client.get("/api/gallery?folder=").json()
        assert root_only["total"] == 5
        assert all(f["folder"] == "" for f in root_only["files"])

        # the nested file's own routes (metadata, delete) work through the
        # slash in its name
        meta = client.get("/api/gallery/ltx/nested-0.png/metadata").json()
        assert meta["name"] == "ltx/nested-0.png"
        assert client.delete("/api/gallery/ltx/nested-0.png").status_code == 200
        assert not (outputs / "ltx" / "nested-0.png").exists()


def test_gallery_thumbnail_is_smaller_than_the_original(server, tmp_path):
    from PIL import Image

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        Image.new("RGB", (1200, 1200), "red").save(outputs / "big.png")

        original = client.get("/outputs/big.png")
        thumbnail = client.get("/api/gallery/big.png/thumbnail")

        assert thumbnail.status_code == 200
        assert thumbnail.headers["content-type"] == "image/jpeg"
        assert len(thumbnail.content) < len(original.content)

        import io
        from PIL import Image as PILImage

        thumb_image = PILImage.open(io.BytesIO(thumbnail.content))
        assert max(thumb_image.size) <= 320

        # a non-image file has no thumbnail rendition
        (outputs / "clip.mp4").write_bytes(b"\x00" * 10)
        assert client.get("/api/gallery/clip.mp4/thumbnail").status_code == 404


def test_gallery_urls_change_when_a_file_is_rewritten(server, tmp_path):
    """A rerun overwrites the same name - the URL must move or the browser
    keeps showing the image it already cached."""
    import os
    from PIL import Image

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        path = outputs / "test_image-gen.0-0.0.png"
        Image.new("RGB", (4, 4), "red").save(path)

        first = client.get("/api/gallery").json()["files"][0]["url"]
        assert first.startswith("/outputs/test_image-gen.0-0.0.png?")
        assert client.get(first).status_code == 200

        # the same file, rewritten - as a second run of the workflow does
        Image.new("RGB", (4, 4), "blue").save(path)
        os.utime(path, (0, os.stat(path).st_mtime + 10))

        second = client.get("/api/gallery").json()["files"][0]["url"]
        assert second != first
        assert client.get(second).status_code == 200


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
            "variable_names": ["prompt"],
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


def test_history_persists_a_finished_jobs_event_tail(tmp_path):
    from dw.server.jobs import JobHistory

    history = JobHistory(tmp_path / "jobs.sqlite")
    job = _finished_job_with_events(
        "job-1", [{"seq": 0, "event": "phase", "phase": "loading"}]
    )

    history.record(job)

    assert history.events_for("job-1") == [
        {"seq": 0, "event": "phase", "phase": "loading"}
    ]


def test_history_keeps_only_the_last_events(tmp_path):
    """A long run emits thousands of progress events. The tail is what
    explains an outcome; the head is step-by-step noise."""
    from dw.server.jobs import JobHistory, MAX_PERSISTED_EVENTS

    history = JobHistory(tmp_path / "jobs.sqlite")
    events = [{"seq": i, "event": "log", "message": f"line {i}"} for i in range(500)]
    history.record(_finished_job_with_events("job-1", events))

    stored = history.events_for("job-1")

    assert len(stored) == MAX_PERSISTED_EVENTS
    assert stored[-1]["seq"] == 499, "the tail, not the head, is what is kept"


def test_events_for_is_empty_for_a_job_recorded_before_this_change(tmp_path):
    """An existing install's sqlite file has rows with no events column
    value. They must read as 'nothing stored', not crash."""
    import sqlite3

    from dw.server.jobs import JobHistory

    db = tmp_path / "jobs.sqlite"
    history = JobHistory(db)
    history.record(_finished_job_with_events("job-1", [{"seq": 0}]))
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE jobs SET events = NULL WHERE id = 'job-1'")

    assert history.events_for("job-1") == []


def test_events_for_is_none_for_an_unknown_job(tmp_path):
    from dw.server.jobs import JobHistory

    assert JobHistory(tmp_path / "jobs.sqlite").events_for("ghost") is None


def test_an_existing_database_without_the_events_column_migrates(tmp_path):
    """Opening a pre-change database must add the column, not fail and not
    lose the rows already in it."""
    import sqlite3

    from dw.server.jobs import JobHistory

    db = tmp_path / "jobs.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("""CREATE TABLE jobs (
                id TEXT PRIMARY KEY, workflow TEXT, status TEXT,
                created_at REAL, started_at REAL, finished_at REAL,
                arguments TEXT, spec TEXT, manifest TEXT, warnings TEXT,
                error TEXT
            )""")
        connection.execute(
            "INSERT INTO jobs VALUES ('old',?,?,?,?,?,?,?,?,?,?)",
            ("w", "complete", 1.0, 1.0, 2.0, "{}", "{}", "[]", "[]", None),
        )

    history = JobHistory(db)

    assert history.get("old")["status"] == "complete"
    assert history.events_for("old") == []


def test_recording_still_writes_every_other_column(tmp_path):
    """The insert names its columns, so widening the table cannot shift a
    value into the wrong one. This pins the columns that would have moved."""
    from dw.server.jobs import JobHistory

    history = JobHistory(tmp_path / "jobs.sqlite")
    history.record(_finished_job_with_events("job-1", [{"seq": 0}]))

    detail = history.get("job-1")

    assert detail["id"] == "job-1"
    assert detail["status"] == "complete"
    assert detail["workflow"] == "w"
    assert detail["error"] is None


def _finished_job_with_events(job_id, events):
    """A minimal stand-in for a finished Job: JobHistory.record reads plain
    attributes, so a real worker run is not needed to test persistence."""

    class FinishedJob:
        id = job_id
        workflow_name = "w"
        status = "complete"
        created_at = 1.0
        started_at = 1.0
        finished_at = 2.0
        manifest = []
        warnings = []
        error = None
        spec = {"arguments": {}, "workflow_path": "w.json"}

    job = FinishedJob()
    job.events = events
    return job


def test_event_log_returns_events_for_a_live_job(server):
    with server(success_script) as client:
        job_id = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()[
            "id"
        ]
        wait_for_status(client, job_id, TERMINAL_STATES)

        body = client.get(f"/api/jobs/{job_id}/event-log").json()

        assert body["id"] == job_id
        assert body["events"], "a completed job should have recorded events"
        assert [event["seq"] for event in body["events"]] == list(
            range(len(body["events"]))
        )
        assert body["last_seq"] == body["events"][-1]["seq"]
        assert body["truncated"] is False
        assert body["note"] is None


def test_event_log_pages_with_after_and_limit(server):
    with server(success_script) as client:
        job_id = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()[
            "id"
        ]
        wait_for_status(client, job_id, TERMINAL_STATES)
        total = client.get(f"/api/jobs/{job_id}/event-log").json()["events"]
        assert len(total) >= 3, "test needs a job with at least three events"

        first = client.get(f"/api/jobs/{job_id}/event-log?limit=2").json()
        assert len(first["events"]) == 2
        assert first["truncated"] is True
        assert first["last_seq"] == 1

        rest = client.get(
            f"/api/jobs/{job_id}/event-log?after={first['last_seq']}"
        ).json()
        assert rest["events"][0]["seq"] == 2
        assert rest["truncated"] is False


def test_event_log_clamps_a_negative_after(server):
    with server(success_script) as client:
        job_id = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()[
            "id"
        ]
        wait_for_status(client, job_id, TERMINAL_STATES)

        body = client.get(f"/api/jobs/{job_id}/event-log?after=-99").json()

        assert body["events"][0]["seq"] == 0


def test_event_log_serves_a_historical_jobs_persisted_events(server):
    """A job recovered from sqlite is a plain dict, but its event tail was
    persisted with it - that is what makes last night's failure explainable."""
    with server(success_script) as client:
        manager = client.app.state.job_manager
        manager.get = lambda job_id: {"id": job_id, "status": "failed"}
        manager.history.events_for = lambda job_id: [
            {"seq": 0, "event": "phase", "phase": "loading"},
            {"seq": 1, "event": "job_status", "status": "failed"},
        ]

        body = client.get("/api/jobs/historical/event-log").json()

        assert [event["seq"] for event in body["events"]] == [0, 1]
        assert body["last_seq"] == 1
        assert body["truncated"] is False
        assert body["note"] is None


def test_event_log_pages_a_historical_jobs_events(server):
    with server(success_script) as client:
        manager = client.app.state.job_manager
        manager.get = lambda job_id: {"id": job_id, "status": "complete"}
        manager.history.events_for = lambda job_id: [
            {"seq": index} for index in range(5)
        ]

        body = client.get("/api/jobs/historical/event-log?after=1&limit=2").json()

        assert [event["seq"] for event in body["events"]] == [2, 3]
        assert body["truncated"] is True


def test_event_log_says_so_when_a_historical_job_kept_no_events(server):
    """A job recorded before events were persisted. Say that, rather than
    returning an empty list that reads as 'nothing happened'."""
    with server(success_script) as client:
        manager = client.app.state.job_manager
        manager.get = lambda job_id: {"id": job_id, "status": "complete"}
        manager.history.events_for = lambda job_id: []

        body = client.get("/api/jobs/historical/event-log").json()

        assert body["events"] == []
        assert "not retained" in body["note"]


def test_event_log_404s_for_an_unknown_job(server):
    with server(success_script) as client:
        response = client.get("/api/jobs/nope/event-log")

        assert response.status_code == 404


def test_event_log_says_so_when_a_historical_jobs_log_was_truncated(server):
    """History keeps only the last MAX_PERSISTED_EVENTS. A tail that fits one
    page would otherwise answer `after=-1` with `truncated: false` and no
    note - reading as the whole log of a job that emitted thousands."""
    with server(success_script) as client:
        manager = client.app.state.job_manager
        manager.get = lambda job_id: {"id": job_id, "status": "failed"}
        manager.history.events_for = lambda job_id: [
            {"seq": index} for index in range(2800, 3000)
        ]

        body = client.get("/api/jobs/historical/event-log").json()

        assert body["events"][0]["seq"] == 2800
        assert body["truncated"] is False, "this page is not itself cut short"
        assert "last 200" in body["note"]
        assert "not retained" not in body["note"], "distinct from the no-log note"


def test_event_log_does_not_claim_truncation_for_a_complete_historical_log(server):
    """A job that genuinely emitted exactly MAX_PERSISTED_EVENTS lost nothing.
    The signal is the first stored seq, not the length of the tail."""
    from dw.server.jobs import MAX_PERSISTED_EVENTS

    with server(success_script) as client:
        manager = client.app.state.job_manager
        manager.get = lambda job_id: {"id": job_id, "status": "complete"}
        manager.history.events_for = lambda job_id: [
            {"seq": index} for index in range(MAX_PERSISTED_EVENTS)
        ]

        body = client.get(
            f"/api/jobs/historical/event-log?limit={MAX_PERSISTED_EVENTS}"
        ).json()

        assert body["note"] is None


def test_a_recorded_job_reads_back_through_the_event_log_route(server):
    """The whole persistence seam end to end: JobHistory.record writes the
    tail, the route reads it back. Both halves are tested in isolation
    elsewhere; this is the join, which is where a lossy tail hides."""
    with server(success_script) as client:
        manager = client.app.state.job_manager
        manager.history.record(
            _finished_job_with_events(
                "recorded",
                [
                    {"seq": 0, "event": "phase", "phase": "loading"},
                    {"seq": 1, "event": "job_status", "status": "failed"},
                ],
            )
        )

        body = client.get("/api/jobs/recorded/event-log").json()

        assert [event["seq"] for event in body["events"]] == [0, 1]
        assert body["events"][0]["phase"] == "loading"
        assert body["status"] == "complete"
        assert body["truncated"] is False
        assert body["note"] is None


def test_a_recorded_job_whose_log_was_dropped_says_so_through_the_route(server):
    """The Finding-4 case with no stand-ins: a long run really recorded, read
    back through the route. The head is gone and the answer has to admit it."""
    from dw.server.jobs import MAX_PERSISTED_EVENTS

    with server(success_script) as client:
        manager = client.app.state.job_manager
        manager.history.record(
            _finished_job_with_events("long", [{"seq": index} for index in range(3000)])
        )

        body = client.get("/api/jobs/long/event-log?limit=1000").json()

        assert len(body["events"]) == MAX_PERSISTED_EVENTS
        assert body["events"][0]["seq"] == 3000 - MAX_PERSISTED_EVENTS
        assert body["truncated"] is False
        assert f"last {MAX_PERSISTED_EVENTS}" in body["note"]


def test_workflow_details_name_their_variables(server):
    """The listing says which knobs a workflow takes, so an agent picking a
    workflow to run knows what to pass without fetching each candidate's
    full definition. Names only - the defaults of every workflow on disk
    are an order of magnitude more payload on a listing the UI reloads."""
    with server(success_script) as client:
        workflow = valid_workflow("knobby")
        workflow["variables"] = {"prompt": "a cat", "steps": 25}
        client.put("/api/workflows/Knobby", json={"workflow": workflow})

        details = client.get("/api/workflows").json()["details"]
        assert details["Knobby"]["variable_names"] == ["prompt", "steps"]
        assert details["Knobby"]["variables"] == 2
