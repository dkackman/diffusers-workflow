"""The notes API: PUT /api/notes over its four target forms, `note` on job
submission, upload and keep, and the note fields every listing and detail
route carries (dw/notes.py holds the stores)."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from dw import notes as N
from dw.runs import new_run_id
from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.workspace import Workspace

from .test_server import ScriptedWorkerManager, valid_workflow, wait_for_status

RUN_ID = "20261012-120000-aabbccdd"
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


@pytest.fixture
def studio(tmp_path):
    """A workspace root: its default workspace has an asset library."""
    return Workspace(tmp_path / "studio", "flag").ensure()


@pytest.fixture
def output_root(studio):
    return studio.outputs


def run_script(output_dir):
    """A worker that opens a fresh run directory under `output_dir` and
    reports it with run_start, the way Workflow.run does."""

    def script(command):
        run_id = new_run_id({"command": len(os.listdir(output_dir))})
        relative = f"server_test/{run_id}"
        run_dir = os.path.join(output_dir, "server_test", run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "a.png"), "wb") as handle:
            handle.write(PNG)
        yield {
            "type": "progress",
            "event": "run_start",
            "run_id": run_id,
            "identity": "server_test",
            "run_dir": relative,
            "version": 1,
        }
        yield {"type": "success", "message": "ok", "run_count": 1, "manifest": []}

    return script


@pytest.fixture
def client(studio, tmp_path):
    manager = JobManager(
        studio.outputs,
        worker_manager=ScriptedWorkerManager(run_script(studio.outputs)),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=studio.workflows,
    )
    app = create_app(
        workflow_dir=studio.workflows,
        output_dir=studio.outputs,
        job_manager=manager,
        prompt_dir=studio.prompts,
        asset_dir=studio.assets,
        workspace=studio.root,
    )
    with TestClient(app, base_url="http://localhost") as test_client:
        yield test_client


@pytest.fixture
def make_run(write_run, output_root):
    """A run of workflow `identity` holding `files`, each listed by a
    manifest step. Returns the run directory."""

    def build(identity, files, run_id=RUN_ID):
        steps = [{"step": "gen", "files": list(files)}]
        return write_run(output_root, identity, run_id, steps, files=files)

    return build


def run_name(run_dir, output_root):
    return os.path.relpath(run_dir, output_root).replace(os.sep, "/")


def asset_entry(client, name):
    assets = {a["name"]: a for a in client.get("/api/assets").json()["assets"]}
    return assets[name]


def snapshot(root):
    """Every file under root with its bytes - what a refused write must not
    change. The job history's SQLite file and its -wal/-shm sidecars are left
    out: the client fixture keeps them beside the studio, and a WAL
    checkpoint rewrites their bytes on the connection's schedule, not the
    request's."""
    found = {}
    for current, _dirs, names in os.walk(root):
        for name in names:
            if name.startswith("jobs.sqlite"):
                continue
            path = os.path.join(current, name)
            with open(path, "rb") as handle:
                found[path] = handle.read()
    return found


# ------------------------------------------------------------- PUT /api/notes


def test_put_file_and_run_note_and_read_back(client, output_root, make_run):
    run_dir = make_run("wf", ["a.png", "final/b.png"])
    run = run_name(run_dir, output_root)
    r = client.put("/api/notes", json={"target": run, "note": "shift 3 vs 5"})
    assert r.status_code == 200
    assert r.json() == {"target": run, "note": "shift 3 vs 5", "note_source": "run"}
    r = client.put(
        "/api/notes", json={"target": f"{run}/final/b.png", "note": "keeper"}
    )
    assert r.status_code == 200
    assert r.json() == {
        "target": f"{run}/final/b.png",
        "note": "keeper",
        "note_source": "file",
    }
    files = {f["label"]: f for f in client.get("/api/gallery").json()["files"]}
    assert files["a.png"]["note"] == "shift 3 vs 5"
    assert files["b.png"]["note"] == "keeper"
    meta = client.get(f"/api/gallery/{run}/final/b.png/metadata").json()
    assert (meta["note"], meta["run_note"], meta["note_source"]) == (
        "keeper",
        "shift 3 vs 5",
        "file",
    )
    meta = client.get(f"/api/gallery/{run}/a.png/metadata").json()
    assert (meta["note"], meta["run_note"], meta["note_source"]) == (
        "shift 3 vs 5",
        "shift 3 vs 5",
        "run",
    )
    assert N.read_notes(run_dir) == {
        "run": "shift 3 vs 5",
        "files": {"final/b.png": "keeper"},
    }


def test_a_file_without_its_own_note_answers_with_the_run_note(
    client, output_root, make_run
):
    run_dir = make_run("wf", ["a.png"])
    run = run_name(run_dir, output_root)
    client.put("/api/notes", json={"target": run, "note": "the run"})
    r = client.put("/api/notes", json={"target": f"{run}/a.png", "note": None})
    assert r.json() == {
        "target": f"{run}/a.png",
        "note": "the run",
        "note_source": "run",
    }


def test_an_output_reference_names_the_same_file(client, output_root, make_run):
    run = run_name(make_run("wf", ["a.png"]), output_root)
    r = client.put("/api/notes", json={"target": f"output:{run}/a.png", "note": "o"})
    assert r.status_code == 200
    assert r.json()["note_source"] == "file"
    files = {f["label"]: f for f in client.get("/api/gallery").json()["files"]}
    assert files["a.png"]["note"] == "o"


def test_a_listing_without_notes_carries_no_note_key(client, output_root, make_run):
    run = run_name(make_run("wf", ["a.png"]), output_root)
    (entry,) = client.get("/api/gallery").json()["files"]
    assert "note" not in entry
    meta = client.get(f"/api/gallery/{run}/a.png/metadata").json()
    assert not {"note", "run_note", "note_source"} & set(meta)


def test_clear_note(client, output_root, make_run):
    run_dir = make_run("wf", ["a.png"])
    run = run_name(run_dir, output_root)
    client.put("/api/notes", json={"target": run, "note": "x"})
    assert os.path.exists(os.path.join(run_dir, N.NOTES_FILE_NAME))
    r = client.put("/api/notes", json={"target": run, "note": ""})
    assert r.status_code == 200
    assert r.json() == {"target": run, "note": None, "note_source": None}
    assert not os.path.exists(os.path.join(run_dir, N.NOTES_FILE_NAME))
    client.put("/api/notes", json={"target": run, "note": "y"})
    r = client.put("/api/notes", json={"target": run, "note": None})
    assert r.json()["note"] is None
    assert not os.path.exists(os.path.join(run_dir, N.NOTES_FILE_NAME))


@pytest.mark.parametrize(
    "target",
    [
        # a gallery file name
        "../etc/passwd",
        "wf/../../x",
        "/abs/path.png",
        f"wf/{RUN_ID}/../../../outside.png",
        "wf/20261012-000000-zzzz/missing.png",
        f"wf/{RUN_ID}/missing.png",
        # a run directory
        f"../{RUN_ID}",
        f"wf/../../{RUN_ID}",
        f"/tmp/{RUN_ID}",
        f"nope/{RUN_ID}",
        # an asset
        "asset:../x.png",
        "asset:../../studio/outputs/wf/a.png",
        "asset:/etc/passwd",
        "asset:missing.png",
        # a workspace
        "workspace:../other",
        "workspace:/abs",
        "workspace:nope",
        "workspace:",
    ],
)
def test_put_note_refuses_traversal_targets(client, studio, make_run, target):
    make_run("wf", ["a.png"])
    before = snapshot(os.path.dirname(studio.root))
    r = client.put("/api/notes", json={"target": target, "note": "x"})
    assert r.status_code in (400, 404), r.json()
    assert snapshot(os.path.dirname(studio.root)) == before


def test_a_flat_layout_file_cannot_hold_a_note(client, output_root):
    with open(os.path.join(output_root, "flat.png"), "wb") as handle:
        handle.write(PNG)
    r = client.put("/api/notes", json={"target": "flat.png", "note": "x"})
    assert r.status_code == 409
    assert "notes need a run directory" in r.json()["detail"]
    assert not os.path.exists(os.path.join(output_root, N.NOTES_FILE_NAME))


def test_note_too_long(client, output_root, make_run):
    run_dir = make_run("wf", ["a.png"])
    run = run_name(run_dir, output_root)
    r = client.put("/api/notes", json={"target": run, "note": "x" * 4001})
    assert r.status_code == 400
    assert not os.path.exists(os.path.join(run_dir, N.NOTES_FILE_NAME))
    r = client.put("/api/notes", json={"target": run, "note": "x" * 4000})
    assert r.status_code == 200


def test_a_write_failure_is_a_500(client, output_root, make_run, monkeypatch):
    run = run_name(make_run("wf", ["a.png"]), output_root)

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("dw.server.routes.media.set_run_note", refuse)
    r = client.put("/api/notes", json={"target": run, "note": "x"})
    assert r.status_code == 500


def test_workspace_note(client, studio):
    r = client.put("/api/notes", json={"target": "workspace:default", "note": "anthem"})
    assert r.status_code == 200
    assert r.json() == {
        "target": "workspace:default",
        "note": "anthem",
        "note_source": None,
    }
    ws = {w["name"]: w for w in client.get("/api/workspaces").json()["workspaces"]}
    assert ws["default"]["note"] == "anthem"
    assert N.read_workspace_note(studio.root) == "anthem"


def test_a_named_workspace_note(client, studio):
    assert client.post("/api/workspaces", json={"name": "ep4"}).status_code == 201
    r = client.put("/api/notes", json={"target": "workspace:ep4", "note": "pilot"})
    assert r.status_code == 200
    ws = {w["name"]: w for w in client.get("/api/workspaces").json()["workspaces"]}
    assert ws["ep4"]["note"] == "pilot"
    assert "note" not in ws["default"]
    assert N.read_workspace_note(os.path.join(studio.root, "ep4")) == "pilot"
    client.put("/api/notes", json={"target": "workspace:ep4", "note": ""})
    ws = {w["name"]: w for w in client.get("/api/workspaces").json()["workspaces"]}
    assert "note" not in ws["ep4"]


def test_an_asset_note_by_reference(client, studio):
    up = client.post(
        "/api/uploads",
        params={"filename": "iris.png", "asset_name": "iris"},
        content=PNG,
    )
    assert up.status_code == 201
    r = client.put(
        "/api/notes", json={"target": "asset:uploads/iris.png", "note": "lead"}
    )
    assert r.status_code == 200
    assert r.json() == {
        "target": "asset:uploads/iris.png",
        "note": "lead",
        "note_source": "file",
    }
    assert asset_entry(client, "uploads/iris.png")["note"] == "lead"
    meta = client.get("/api/gallery/asset:uploads/iris.png/metadata").json()
    assert (meta["note"], meta["note_source"]) == ("lead", "file")
    assert "run_note" not in meta
    uploads = os.path.join(studio.assets, "uploads")
    assert N.read_notes(uploads)["files"] == {"iris.png": "lead"}


# ------------------------------------------------------- upload, keep, delete


def test_asset_note_upload_keep_and_delete(client, studio):
    up = client.post(
        "/api/uploads",
        params={"filename": "iris.png", "note": "cast: iris"},
        content=PNG,
    )
    assert up.status_code == 201
    ref = up.json()["reference"]
    name = ref.split(":", 1)[1]
    assert asset_entry(client, name)["note"] == "cast: iris"

    too_long = client.post(
        "/api/uploads",
        params={"filename": "long.png", "note": "x" * 4001},
        content=PNG,
    )
    assert too_long.status_code == 400
    # refused before a byte was written
    assert sorted(os.listdir(os.path.join(studio.assets, "uploads"))) == sorted(
        [os.path.basename(name), N.NOTES_FILE_NAME]
    )

    assert client.delete(f"/api/assets/{name}").status_code == 200
    assert not os.path.exists(os.path.join(studio.assets, "uploads", N.NOTES_FILE_NAME))


def test_keep_copies_effective_note(client, output_root, make_run):
    run = run_name(make_run("wf", ["a.png", "b.png"]), output_root)
    client.put("/api/notes", json={"target": run, "note": "the run note"})
    client.put("/api/notes", json={"target": f"{run}/b.png", "note": "b's own"})

    kept = client.post("/api/assets/keep", json={"name": f"{run}/a.png"})
    assert kept.status_code == 201
    assert asset_entry(client, "a.png")["note"] == "the run note"

    kept = client.post("/api/assets/keep", json={"name": f"{run}/b.png"})
    assert asset_entry(client, "b.png")["note"] == "b's own"

    kept = client.post(
        "/api/assets/keep",
        json={"name": f"{run}/a.png", "asset_name": "c.png", "note": "given"},
    )
    assert kept.status_code == 201
    assert asset_entry(client, "c.png")["note"] == "given"

    refused = client.post(
        "/api/assets/keep",
        json={"name": f"{run}/a.png", "asset_name": "d.png", "note": "x" * 4001},
    )
    assert refused.status_code == 400
    assert "d.png" not in {
        a["name"] for a in client.get("/api/assets").json()["assets"]
    }


def test_keep_without_any_note_leaves_none(client, output_root, make_run):
    run = run_name(make_run("wf", ["a.png"]), output_root)
    assert client.post("/api/assets/keep", json={"name": f"{run}/a.png"}).status_code
    assert "note" not in asset_entry(client, "a.png")


def test_deleted_asset_leaves_no_note_or_provenance(
    client, studio, output_root, make_run
):
    run = run_name(make_run("wf", ["x.png"]), output_root)
    history = client.app.state.job_manager.history
    history.job_for_file = lambda name, workspace=None: {
        "id": "abc",
        "status": "succeeded",
    }
    kept = client.post(
        "/api/assets/keep",
        json={"name": f"{run}/x.png", "asset_name": "uploads/x.png", "note": "old"},
    )
    assert kept.status_code == 201
    meta = client.get("/api/gallery/asset:uploads/x.png/metadata").json()
    assert meta["job"]["id"] == "abc"
    assert meta["note"] == "old"

    assert client.delete("/api/assets/uploads/x.png").status_code == 200
    uploads = os.path.join(studio.assets, "uploads")
    assert not os.path.exists(os.path.join(uploads, N.NOTES_FILE_NAME))
    assert not os.path.exists(os.path.join(uploads, "manifest.json"))

    up = client.post(
        "/api/uploads",
        params={"filename": "x.png", "asset_name": "x.png"},
        content=PNG,
    )
    assert up.status_code == 201
    assert up.json()["reference"] == "asset:uploads/x.png"
    assert "note" not in asset_entry(client, "uploads/x.png")
    meta = client.get("/api/gallery/asset:uploads/x.png/metadata").json()
    assert meta["job"] is None
    assert meta["run_id"] == ""
    assert "note" not in meta


def test_an_upload_over_an_existing_name_drops_the_old_note(client):
    params = {"filename": "y.png", "asset_name": "y.png", "note": "first"}
    assert client.post("/api/uploads", params=params, content=PNG).status_code == 201
    assert asset_entry(client, "uploads/y.png")["note"] == "first"
    params = {"filename": "y.png", "asset_name": "y.png"}
    assert client.post("/api/uploads", params=params, content=PNG).status_code == 201
    assert "note" not in asset_entry(client, "uploads/y.png")


# --------------------------------------------------------------------- jobs


def test_job_note_reaches_history_run_and_rerun(client, output_root):
    submitted = client.post(
        "/api/jobs", json={"workflow": valid_workflow(), "note": "testing"}
    )
    assert submitted.status_code == 201
    job = submitted.json()
    assert job["note"] == "testing"
    detail = wait_for_status(client, job["id"], ["succeeded"])
    assert detail["note"] == "testing"

    # the run directory the worker reported holds it as the run note
    run_dir = os.path.join(output_root, *detail["run_dir"].split("/"))
    assert N.read_notes(run_dir)["run"] == "testing"
    files = {f["name"]: f for f in client.get("/api/gallery").json()["files"]}
    assert files[f"{detail['run_dir']}/a.png"]["note"] == "testing"

    # the polled list carries it, live and from history
    listed = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
    assert listed[job["id"]]["note"] == "testing"
    manager = client.app.state.job_manager
    assert manager.history.get(job["id"])["note"] == "testing"
    (summary,) = [s for s in manager.history.recent_summaries() if s["id"] == job["id"]]
    assert summary["note"] == "testing"

    # read back from sqlite once the live job is gone
    manager.jobs.pop(job["id"])
    assert client.get(f"/api/jobs/{job['id']}").json()["note"] == "testing"
    listed = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
    assert listed[job["id"]]["note"] == "testing"

    # a rerun carries it forward, and writes it into its own run
    rerun = client.post(f"/api/jobs/{job['id']}/rerun")
    assert rerun.status_code == 201
    assert rerun.json()["note"] == "testing"
    again = wait_for_status(client, rerun.json()["id"], ["succeeded"])
    assert again["note"] == "testing"
    assert again["run_dir"] != detail["run_dir"]
    rerun_dir = os.path.join(output_root, *again["run_dir"].split("/"))
    assert N.read_notes(rerun_dir)["run"] == "testing"


def test_a_job_without_a_note_has_no_note(client, output_root):
    job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
    detail = wait_for_status(client, job["id"], ["succeeded"])
    assert "note" not in detail
    run_dir = os.path.join(output_root, *detail["run_dir"].split("/"))
    assert not os.path.exists(os.path.join(run_dir, N.NOTES_FILE_NAME))
    manager = client.app.state.job_manager
    assert "note" not in manager.history.get(job["id"])


def test_a_job_note_too_long_is_refused_before_queuing(client):
    r = client.post(
        "/api/jobs", json={"workflow": valid_workflow(), "note": "x" * 4001}
    )
    assert r.status_code == 400
    assert client.app.state.job_manager.worker_manager.commands == []


def test_a_run_dir_escaping_the_output_root_gets_no_note(tmp_path, studio):
    """The run_start event is the worker's word: a run_dir that leaves the
    output root writes nothing outside it."""
    from dw.server.job_record import Job
    from dw.server.job_results import record_progress

    outside = tmp_path / "outside"
    outside.mkdir()
    job = Job({"workflow_name": "w", "note": "n"})
    record_progress(
        job,
        {"event": "run_start", "run_id": RUN_ID, "run_dir": "../../outside"},
        studio.outputs,
    )
    assert not (outside / N.NOTES_FILE_NAME).exists()


def test_note_set_during_run_survives_manifest_rewrite(tmp_path):
    from dw.runs import write_manifest

    d = str(tmp_path)
    write_manifest(d, {"status": "running", "steps": []})
    N.set_run_note(d, "mid-run")
    write_manifest(d, {"status": "succeeded", "steps": []})
    assert N.read_notes(d)["run"] == "mid-run"
    with open(os.path.join(d, "manifest.json")) as handle:
        assert "mid-run" not in json.dumps(json.load(handle))


def test_job_detail_and_list_show_the_current_run_note(client, output_root):
    """A run note edited after submission replaces the submitted one on
    the job, live and from history, and a cleared one removes it."""
    job = client.post(
        "/api/jobs", json={"workflow": valid_workflow(), "note": "A"}
    ).json()
    detail = wait_for_status(client, job["id"], ["succeeded"])
    r = client.put("/api/notes", json={"target": detail["run_dir"], "note": "B"})
    assert r.status_code == 200

    def shown():
        listed = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
        return (
            client.get(f"/api/jobs/{job['id']}").json().get("note"),
            listed[job["id"]].get("note"),
        )

    assert shown() == ("B", "B")
    client.app.state.job_manager.jobs.pop(job["id"])  # from sqlite now
    assert shown() == ("B", "B")
    client.put("/api/notes", json={"target": detail["run_dir"], "note": None})
    assert shown() == (None, None)


def test_a_job_whose_run_is_gone_keeps_its_submitted_note(client, output_root):
    import shutil

    job = client.post(
        "/api/jobs", json={"workflow": valid_workflow(), "note": "A"}
    ).json()
    detail = wait_for_status(client, job["id"], ["succeeded"])
    shutil.rmtree(os.path.join(output_root, *detail["run_dir"].split("/")))
    assert client.get(f"/api/jobs/{job['id']}").json()["note"] == "A"


def test_keep_with_an_empty_note_keeps_none(client, output_root, make_run):
    run = run_name(make_run("wf", ["a.png"]), output_root)
    client.put("/api/notes", json={"target": run, "note": "the run note"})
    kept = client.post("/api/assets/keep", json={"name": f"{run}/a.png", "note": ""})
    assert kept.status_code == 201
    assert "note" not in asset_entry(client, "a.png")


def _keep_with_provenance(client, run, asset_name):
    history = client.app.state.job_manager.history
    history.job_for_file = lambda name, workspace=None: {
        "id": "abc",
        "status": "succeeded",
    }
    kept = client.post(
        "/api/assets/keep", json={"name": f"{run}/x.png", "asset_name": asset_name}
    )
    assert kept.status_code == 201
    meta = client.get(f"/api/gallery/asset:{asset_name}/metadata").json()
    assert meta["job"]["id"] == "abc"


def test_an_upload_over_a_kept_name_clears_its_provenance(
    client, output_root, make_run
):
    run = run_name(make_run("wf", ["x.png"]), output_root)
    _keep_with_provenance(client, run, "uploads/x.png")
    up = client.post(
        "/api/uploads", params={"filename": "x.png", "asset_name": "x.png"}, content=PNG
    )
    assert up.status_code == 201
    assert client.get("/api/gallery/asset:uploads/x.png/metadata").json()["job"] is None


def test_an_upload_under_a_fresh_name_leaves_other_provenance(
    client, studio, output_root, make_run
):
    run = run_name(make_run("wf", ["x.png"]), output_root)
    _keep_with_provenance(client, run, "uploads/x.png")
    up = client.post(
        "/api/uploads", params={"filename": "y.png", "asset_name": "y.png"}, content=PNG
    )
    assert up.status_code == 201
    meta = client.get("/api/gallery/asset:uploads/x.png/metadata").json()
    assert meta["job"]["id"] == "abc"

    # a kept record caught mid-write is not the upload's to rewrite
    manifest = os.path.join(studio.assets, "uploads", "manifest.json")
    with open(manifest, "w") as handle:
        handle.write('{"steps": [')
    up = client.post(
        "/api/uploads", params={"filename": "z.png", "asset_name": "z.png"}, content=PNG
    )
    assert up.status_code == 201
    with open(manifest) as handle:
        assert handle.read() == '{"steps": ['


def test_a_kept_asset_whose_job_left_history_reports_it_forgotten(
    client, output_root, make_run
):
    """#918: the id stays, but the stale status is not reported as current."""
    run = run_name(make_run("wf", ["x.png"]), output_root)
    _keep_with_provenance(client, run, "uploads/x.png")
    meta = client.get("/api/gallery/asset:uploads/x.png/metadata").json()
    assert meta["job"] == {"id": "abc", "status": "forgotten"}
