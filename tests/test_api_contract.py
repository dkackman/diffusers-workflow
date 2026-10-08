"""The UI's response contract: the routes the UI reads declare response
models, and the OpenAPI document generated from them is committed where the
UI generates its types from (ui/src/lib/generated/). See docs/ARCHITECTURE.md,
"The UI's response contract"."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_server import (  # noqa: F401
    server,
    success_script,
    hanging_script,
    valid_workflow,
    wait_for_status,
)

REPO = Path(__file__).resolve().parent.parent
DUMP = REPO / "scripts" / "dump_openapi.py"


def _model_in(mode: str):
    """ApiModel as a fresh interpreter builds it with the switch set or not."""
    env = {k: v for k, v in os.environ.items() if k != "DW_STRICT_RESPONSES"}
    if mode == "strict":
        env["DW_STRICT_RESPONSES"] = "1"
    code = "from dw.server.api_models import ApiModel; print(ApiModel.model_config['extra'])"
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_the_suite_runs_strict():
    from dw.server.api_models import STRICT

    assert STRICT, "tests/conftest.py sets DW_STRICT_RESPONSES before dw is imported"


def test_runtime_is_lenient_and_strict_mode_forbids():
    assert _model_in("runtime") == "allow"
    assert _model_in("strict") == "forbid"


def test_a_lenient_model_passes_an_undeclared_key_through():
    # What production does with a key a model has not declared yet: send it
    from pydantic import ConfigDict

    from dw.server.api_models import ApiModel

    class Probe(ApiModel):
        model_config = ConfigDict(extra="allow")
        x: int

    app = FastAPI()

    @app.get("/p", response_model=Probe, response_model_exclude_unset=True)
    def p():
        return {"x": 1, "later": {"y": 2}}

    assert TestClient(app).get("/p").json() == {"x": 1, "later": {"y": 2}}


def _dump(extra_env):
    env = {**os.environ, **extra_env}
    return subprocess.run(
        [sys.executable, str(DUMP), "--stdout"],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_the_document_does_not_depend_on_the_machine():
    assert _dump({"DW_DEVICE": "cpu"}) == _dump({"DW_DEVICE": "mps"})


def test_the_document_carries_no_release_version():
    assert json.loads(_dump({}))["info"]["version"] == "0"


def _dump_module():
    spec = importlib.util.spec_from_file_location("dump_openapi", DUMP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pins_name_fastapi_and_pydantic():
    # The two packages whose upgrade can change the generated document
    assert set(_dump_module().pins()) == {"fastapi", "pydantic"}


def test_the_dump_refuses_to_write_under_versions_it_is_not_pinned_to(
    tmp_path, monkeypatch, capsys
):
    dump = _dump_module()
    target = tmp_path / "openapi.json"
    monkeypatch.setattr(dump, "OPENAPI_PATH", target)
    monkeypatch.setattr(dump, "installed", lambda: {**dump.pins(), "fastapi": "9.9.9"})
    assert dump.main([]) == 2
    assert not target.exists()
    said = capsys.readouterr().out
    assert "fastapi 9.9.9" in said and "constraints-openapi.txt" in said


def test_ci_installs_the_pinned_versions():
    # Every CI job that builds the app strict - the backend's freshness test,
    # e2e's fixture server - must resolve the pins, or a FastAPI release
    # fails PRs that did not touch the contract
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    installs = [
        line
        for line in ci.splitlines()
        if "pip install" in line and "-r requirements.txt" in line
    ]
    assert installs and all("-c constraints-openapi.txt" in line for line in installs)


def test_the_committed_document_is_current():
    stray = _dump_module().unpinned()
    if stray:
        # CI installs the pins, so there a mismatch is a broken install, not
        # a reason to skip; elsewhere the dump would differ for that reason alone
        assert not os.environ.get("CI"), f"CI is not on the pinned versions: {stray}"
        pytest.skip(f"not on the versions the document is pinned to: {stray}")
    committed = (REPO / "ui" / "src" / "lib" / "generated" / "openapi.json").read_text()
    assert committed == _dump({}), (
        "the server's response contract changed: run `python scripts/dump_openapi.py`, "
        "then `cd ui && npm run gen:api`, and commit both"
    )


UI_READ_ROUTES = [
    ("get", "/api/health"),
    ("get", "/api/server"),
    ("get", "/api/memory"),
    ("post", "/api/memory/clear"),
    ("get", "/api/models"),
    ("post", "/api/models/download"),
    ("get", "/api/models/downloads"),
    ("post", "/api/models/downloads/{download_id}/cancel"),
    ("delete", "/api/models"),
    ("get", "/api/system/diffusers"),
    ("post", "/api/system/diffusers/update"),
    ("get", "/api/jobs"),
    ("post", "/api/jobs"),
    ("get", "/api/jobs/{job_id}"),
    ("delete", "/api/jobs/{job_id}/run"),
    ("get", "/api/jobs/{job_id}/workflow"),
    ("post", "/api/jobs/{job_id}/rerun"),
    ("post", "/api/jobs/{job_id}/export"),
    ("post", "/api/jobs/{job_id}/move"),
    ("post", "/api/jobs/{job_id}/cancel"),
    ("post", "/api/enhance"),
    ("post", "/api/validate"),
    ("get", "/api/workspaces"),
    ("post", "/api/workspaces"),
    ("delete", "/api/workspaces/{name}"),
    ("get", "/api/workflows"),
    ("put", "/api/workflows/{name}"),
    ("patch", "/api/workflows/{name}"),
    ("delete", "/api/workflows/{name}"),
    ("get", "/api/prompts"),
    ("put", "/api/prompts/{name}"),
    ("delete", "/api/prompts/{name}"),
    ("get", "/api/enhancers"),
    ("get", "/api/gallery"),
    ("delete", "/api/gallery/{name}"),
    ("get", "/api/gallery/{name}/metadata"),
    ("get", "/api/assets"),
    ("post", "/api/uploads"),
    ("post", "/api/assets/keep"),
    ("delete", "/api/assets/{name}"),
    ("get", "/api/pipelines"),
    ("get", "/api/pipelines/{name}"),
    ("get", "/api/tasks"),
    ("get", "/api/tasks/{command}"),
    ("get", "/api/classes"),
    ("get", "/api/classes/{name}"),
]


def _success_schema(document, method, path):
    responses = document["paths"][path][method]["responses"]
    ok = next(code for code in responses if code.startswith("2"))
    return (
        responses[ok].get("content", {}).get("application/json", {}).get("schema", {})
    )


@pytest.fixture(scope="module")
def document():
    return json.loads(_dump({}))


@pytest.mark.parametrize("method,path", UI_READ_ROUTES)
def test_a_route_the_ui_reads_declares_its_response(document, method, path):
    schema = _success_schema(document, method, path)
    assert "$ref" in schema or schema.get("type") == "array", (
        f"{method.upper()} {path} declares no response model: the UI's type for it "
        "is not generated from the server"
    )


# The payload does not change: keys, absence and int-ness are what they were
# before the routes declared models (dw_mcp and scripts read them too)


def test_memory_with_no_worker_keeps_its_keys(server):
    with server(success_script) as client:
        body = client.get("/api/memory").json()
        card = client.app.state.job_manager.slots[0].ordinal()
    reading = {
        "live": False,
        "info": None,
        "stale": False,
        "reason": "worker_stopped",
        "age_seconds": None,
    }
    # #462 stage C added one reading per card beside the first card's
    assert body == {**reading, "workers": [{"device": card, **reading}]}


def test_health_sends_exactly_its_keys(server):
    with server(success_script) as client:
        body = client.get("/api/health").json()
    assert set(body) == {
        "status",
        "version",
        "worker_alive",
        "current_job",
        "queued",
        "hostname",
        "device",
        "mcp",
        "workers",
    }
    assert type(body["queued"]) is int
    # One worker, not yet started: no process, so no host memory to read
    (worker,) = body["workers"]
    assert set(worker) == {"device", "name", "vram_gb", "current_job", "alive"}
    assert worker["alive"] is False


def test_an_empty_model_cache_counts_bytes_in_integers(server, tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    with server(success_script) as client:
        body = client.get("/api/models").json()
    assert type(body["size_on_disk"]) is int
    assert body["repos"] == [] and body["warnings"] == []


def test_the_dump_leaves_the_home_directory_alone(tmp_path):
    # It builds an app to read its schema; a real JobManager would open (and
    # migrate) ~/.diffusers_helper/jobs.sqlite, the history a running
    # dw.serve owns
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "DIFFUSERS_HELPER_ROOT"}
    env["HOME"] = str(home)
    subprocess.run(
        [sys.executable, str(DUMP), "--stdout"],
        env=env,
        cwd=REPO,
        capture_output=True,
        check=True,
    )
    assert list(home.iterdir()) == []


def test_the_dump_reads_its_own_checkout(tmp_path):
    # In a worktree sharing the main checkout's venv, the editable install
    # points at the main checkout's dw; the contract must come from this one
    fake = tmp_path / "elsewhere" / "dw"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("raise ImportError('another checkout')\n")
    env = {**os.environ, "PYTHONPATH": str(fake.parent)}
    run = subprocess.run(
        [sys.executable, str(DUMP), "--stdout"],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr[-500:]


def test_a_key_the_worker_did_not_report_stays_absent(server):
    # The worker's report names only what its backend measured; a declared
    # field it left out must not arrive as null
    with server(success_script) as client:
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, job["id"], ["succeeded"])
        body = client.get("/api/memory").json()
    assert body["info"] == {"gpu_available": True}


def test_a_job_recorded_before_run_tracking_still_lists_and_opens(tmp_path):
    # A history row from an older server: only id and status were written,
    # so workflow and created_at are null and the newer columns are absent
    import sqlite3

    from dw.server.app import create_app
    from dw.server.jobs import JobManager
    from tests.test_server import ScriptedWorkerManager

    path = str(tmp_path / "old.sqlite")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, workflow TEXT, status TEXT,"
            " created_at REAL, started_at REAL, finished_at REAL, arguments TEXT,"
            " spec TEXT, manifest TEXT, warnings TEXT, error TEXT)"
        )
        connection.execute(
            "INSERT INTO jobs (id, status) VALUES ('old-1', 'succeeded')"
        )
    (tmp_path / "workflows").mkdir()
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=path,
    )
    app = create_app(
        workflow_dir=str(tmp_path / "workflows"),
        output_dir=str(tmp_path / "outputs"),
        job_manager=manager,
    )
    with TestClient(app, base_url="http://localhost") as client:
        listing = client.get("/api/jobs")
        detail = client.get("/api/jobs/old-1")
    assert listing.status_code == 200, listing.text
    [row] = listing.json()["jobs"]
    assert row["workflow"] is None and row["historical"] is True
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["run_id"] is None and body["manifest"] == []
    assert "progress" not in body and "queue_position" not in body


def test_live_and_historical_jobs_keep_their_own_keys(server):
    with server(hanging_script) as client:
        running = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, running["id"], ["running"])
        queued = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        live = client.get(f"/api/jobs/{running['id']}").json()
        listed = {job["id"]: job for job in client.get("/api/jobs").json()["jobs"]}
        client.post(f"/api/jobs/{queued['id']}/cancel")
        client.post(f"/api/jobs/{running['id']}/cancel")
    assert type(queued["queue_position"]) is int
    assert type(listed[queued["id"]]["queue_position"]) is int
    # a live job has progress and no history-only keys
    assert "progress" in live
    assert "spec" not in live and "historical" not in live
    assert "queue_position" not in live


def test_validation_answers_keep_their_keys(server):
    with server(success_script) as client:
        invalid = client.post("/api/validate", json={"workflow": {"id": "x"}}).json()
        valid = client.post(
            "/api/validate",
            json={"workflow": valid_workflow()},
            params={"sizes": False},
        ).json()
    # an invalid answer carries no plan at all, not a null one
    assert invalid["valid"] is False and "plan" not in invalid
    assert invalid["errors"] and {"path", "message"} <= set(invalid["errors"][0])
    assert valid["valid"] is True
    assert type(valid["plan"]["steps"]) is int
    assert {"minutes", "basis", "device", "partial", "unpriced"} <= set(
        valid["plan"]["estimate"]
    )


def test_validate_answers_for_an_uncached_gated_model(server, monkeypatch):
    # model_info's own `gated` is False, "auto" or "manual" - a gated repo
    # the cache does not hold must still get a plan, not a 500
    import dw.plan

    monkeypatch.setattr(dw.plan, "scan_models", lambda cache_dir=None: {"repos": []})
    monkeypatch.setattr(dw.plan, "_model_info", lambda name: (31.4, "manual", False))
    with server(success_script) as client:
        response = client.post("/api/validate", json={"workflow": valid_workflow()})
    assert response.status_code == 200, response.text
    downloads = response.json()["plan"]["downloads_required"]
    assert downloads and downloads[0]["gated"] == "manual"


def test_runtime_sends_a_response_its_model_rejects_rather_than_a_500():
    # Lenient means lenient: a declared field holding a type the model does
    # not expect is logged and sent as the handler built it - on POST
    # /api/jobs the job is already queued by then, and a 500 would invite a
    # retry that queues it twice
    from dw.server.api_models import ApiModel, send_rejected_responses

    class Probe(ApiModel):
        x: int

    app = FastAPI()
    send_rejected_responses(app)

    @app.post(
        "/p", status_code=201, response_model=Probe, response_model_exclude_unset=True
    )
    def p():
        return {"x": "not an int"}

    response = TestClient(app).post("/p")
    assert response.status_code == 201
    assert response.json() == {"x": "not an int"}


def test_the_app_is_lenient_only_outside_strict_mode():
    code = (
        "from fastapi.exceptions import ResponseValidationError\n"
        "import tempfile, os\n"
        "t = tempfile.mkdtemp(); os.environ['DIFFUSERS_HELPER_ROOT'] = t\n"
        "from dw.server.app import create_app\n"
        "app = create_app(workflow_dir=t + '/w', output_dir=t + '/o', prompt_dir=t + '/p')\n"
        "print(ResponseValidationError in app.exception_handlers)\n"
        "app.state.job_manager.shutdown()\n"
    )

    def handled(strict):
        env = {k: v for k, v in os.environ.items() if k != "DW_STRICT_RESPONSES"}
        if strict:
            env["DW_STRICT_RESPONSES"] = "1"
        return subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    assert handled(strict=False) == "True"
    assert handled(strict=True) == "False"


def test_library_listings_keep_their_keys(server, tmp_path):
    (tmp_path / "workflows" / "Broken.json").write_text("{not json")
    with server(success_script) as client:
        spaces = client.get("/api/workspaces").json()
        listing = client.get("/api/workflows").json()
        compact = client.get("/api/workflows", params={"view": "compact"})
        prompts = client.get("/api/prompts").json()
        raw = client.get("/api/workflows/Basic")
    assert set(spaces) == {"workspace_root", "default", "workspaces"}
    # a listing reports each workspace's disk usage; the single-workspace
    # answers do not (usage is sometimes())
    assert set(spaces["workspaces"][0]) == {
        "name",
        "default",
        "root",
        "workflows",
        "assets",
        "outputs",
        "prompts",
        "common_assets",
        "usage",
    }
    assert set(listing) == {
        "workspace",
        "libraries",
        "workflows",
        "details",
        "shadowed",
        "cost_basis",
    }
    # an unreadable file still lists, with no `configures` key at all
    assert "configures" not in listing["details"]["Broken"]
    assert listing["details"]["Basic"]["configures"] == ""
    assert type(listing["details"]["Basic"]["steps"]) is int
    # the agent's compact view is outside the model, and unchanged
    assert compact.status_code == 200
    assert set(prompts) == {"libraries", "prompts", "details", "shadowed"}
    # the raw GET is the file itself, verbatim (docs/ARCHITECTURE.md)
    assert raw.json() == json.loads((tmp_path / "workflows" / "Basic.json").read_text())


def test_gallery_and_asset_answers_keep_their_keys(server, tmp_path):
    from PIL import Image

    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    Image.new("RGB", (4, 4)).save(outputs / "flat.png")
    with server(success_script) as client:
        listing = client.get("/api/gallery").json()
        orphans = client.get("/api/gallery", params={"only_orphans": True})
        metadata = client.get("/api/gallery/flat.png/metadata").json()
        assets = client.get("/api/assets").json()
    assert set(listing) == {
        "files",
        "total",
        "offset",
        "limit",
        "folders",
        "subfolders",
        "workspace",
    }
    [entry] = listing["files"]
    # the flat layout has no run: run_id is '' and version null
    assert entry["run_id"] == "" and entry["version"] is None
    assert type(entry["size"]) is int and "duration_seconds" not in entry
    # the orphan view is a different answer, outside the model
    assert orphans.status_code == 200 and "runs" in orphans.json()
    assert set(metadata) == {
        "name",
        "source",
        "metadata",
        "job",
        "run_id",
        "version",
        "media",
        "findings",
    }
    assert metadata["job"] is None and metadata["findings"] == []
    assert set(assets) == {"workspace", "libraries", "assets", "folders", "shadowed"}


def test_asset_and_prompt_entries_keep_their_keys(tmp_path):
    from dw.server.app import create_app
    from dw.server.jobs import JobManager
    from tests.test_server import ScriptedWorkerManager

    workflows = tmp_path / "workflows"
    workflows.mkdir()
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "iris.png").write_bytes(b"workspace png")
    examples = tmp_path / "examples"
    (examples / "assets").mkdir(parents=True)
    (examples / "assets" / "iris.png").write_bytes(b"examples png")
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(workflows),
    )
    app = create_app(
        workflow_dir=str(workflows),
        output_dir=str(tmp_path / "outputs"),
        prompt_dir=str(tmp_path / "prompts"),
        job_manager=manager,
        asset_dir=str(assets),
        examples_dirs=[str(examples)],
    )
    prompt = {"text": "a fox", "description": "d", "intended_model": "m", "tags": []}
    with TestClient(app, base_url="http://localhost") as client:
        assert (
            client.put("/api/prompts/Fox", json={"prompt": prompt}).status_code == 200
        )
        listing = client.get("/api/assets").json()
        full = client.get("/api/prompts").json()["details"]["Fox"]
        slim = client.get("/api/prompts", params={"include_text": False}).json()
    [asset] = listing["assets"]
    [hidden] = listing["shadowed"]
    common = {"name", "reference", "folder", "kind", "size", "mtime", "origin"}
    # a local client gets no absolute_url; a shadowed copy has no url at all
    assert set(asset) == common | {"writable", "url"}
    assert set(hidden) == common | {"writable", "shadowed_by"}
    card = {"description", "intended_model", "tags", "origin", "writable"}
    assert set(full) == card | {"text"}
    assert set(slim["details"]["Fox"]) == card | {"text_chars"}


def test_a_parameter_with_no_default_says_so_with_null(server):
    # `default` is always present - null for a required parameter - and
    # `required` is what tells the two apart
    with server(success_script) as client:
        task = client.get("/api/tasks/compose_text").json()
        tasks = client.get("/api/tasks").json()
    assert set(task) == {"name", "summary", "accepts_kwargs", "parameters"}
    assert all("default" in parameter for parameter in task["parameters"])
    assert {"commands", "image_processors", "video_processors", "assessment"} <= set(
        tasks
    )


def test_a_non_finite_default_is_named_not_nulled(server):
    # DPMSolverMultistepScheduler's lambda_min_clipped defaults to -inf; JSON
    # has no -inf, and null would claim there is no default at all
    with server(success_script) as client:
        response = client.get(
            "/api/classes/DPMSolverMultistepScheduler", params={"target": "init"}
        )
    assert response.status_code == 200, response.text
    by_name = {p["name"]: p for p in response.json()["parameters"]}
    assert by_name["lambda_min_clipped"]["default"] == "-inf"
