"""The gallery says which files of a run are its deliverable and which are
scratch: each entry's `role`, the listing's `role` filter, and the metadata's
`role` and `inputs` (dw/roles.py decides the roles)."""

import json
import os

from dw.server.outputs import run_roles, split_run_name


def test_recorded_roles_reach_the_listing(gallery_client, output_root, write_run):
    write_run(
        output_root,
        "wf",
        "20261012-000000-aaaaaaaa",
        [
            {"step": "a", "files": ["a.png"], "role": "intermediate"},
            {"step": "b", "files": ["b.png"], "role": "final"},
        ],
        files=["a.png", "b.png"],
    )
    files = {f["label"]: f for f in gallery_client.get("/api/gallery").json()["files"]}
    assert files["a.png"]["role"] == "intermediate"
    assert files["b.png"]["role"] == "final"


def test_older_run_has_no_role(gallery_client, output_root, write_run):
    # A manifest from before roles: b reads a, but nothing is guessed
    realized = {
        "id": "wf",
        "steps": [
            {"name": "a", "task": {"command": "noop", "arguments": {}}},
            {
                "name": "b",
                "task": {
                    "command": "noop",
                    "arguments": {"image": "previous_result:a"},
                },
            },
        ],
    }
    write_run(
        output_root,
        "wf",
        "20261012-000001-bbbbbbbb",
        [
            {"step": "a", "files": ["intermediate/a.png"], "subfolder": "intermediate"},
            {"step": "b", "files": ["b.png"]},
        ],
        realized=realized,
        files=["intermediate/a.png", "b.png"],
    )
    files = {f["label"]: f for f in gallery_client.get("/api/gallery").json()["files"]}
    assert files["a.png"]["role"] is None
    assert files["b.png"]["role"] is None


def test_role_filter(gallery_client, output_root, write_run):
    write_run(
        output_root,
        "wf",
        "20261012-000002-cccccccc",
        [
            {"step": "a", "files": ["a.png"], "role": "intermediate"},
            {"step": "b", "files": ["final/b.png"], "role": "final"},
        ],
        files=["a.png", "final/b.png"],
    )
    finals = gallery_client.get("/api/gallery", params={"role": "final"}).json()[
        "files"
    ]
    assert [f["label"] for f in finals] == ["b.png"]
    both = gallery_client.get(
        "/api/gallery", params={"role": ["final", "intermediate"]}
    ).json()["files"]
    assert len(both) == 2
    assert (
        gallery_client.get("/api/gallery", params={"role": "bogus"}).status_code == 400
    )


def test_flat_file_has_null_role_and_passes_final_filter(gallery_client, output_root):
    os.makedirs(os.path.join(output_root, "loose"), exist_ok=True)
    with open(os.path.join(output_root, "loose", "x.png"), "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
    entries = gallery_client.get("/api/gallery", params={"role": "final"}).json()[
        "files"
    ]
    loose = [f for f in entries if f["label"] == "x.png"]
    assert loose and loose[0].get("role") is None


def test_metadata_carries_role_and_inputs(gallery_client, output_root, write_run):
    write_run(
        output_root,
        "wf",
        "20261012-000003-dddddddd",
        [{"step": "b", "files": ["b.png"], "role": "final"}],
        files=["b.png"],
        inputs=[{"reference": "asset:iris.png", "kind": "asset"}],
    )
    meta = gallery_client.get(
        "/api/gallery/wf/20261012-000003-dddddddd/b.png/metadata"
    ).json()
    assert meta["role"] == "final"
    assert meta["inputs"] == [{"reference": "asset:iris.png", "kind": "asset"}]


def test_split_run_name(tmp_path):
    root = str(tmp_path)
    assert split_run_name(root, "wf/20261012-000003-dddddddd/final/b.png") == (
        os.path.join(root, "wf", "20261012-000003-dddddddd"),
        "final/b.png",
    )
    assert split_run_name(root, "loose/x.png") is None
    assert run_roles(os.path.join(root, "missing")) == {}


def test_metadata_for_a_subfolder_file_of_a_nested_identity_run(
    gallery_client, output_root, write_run
):
    write_run(
        output_root,
        "series/ep1",
        "20261012-000004-eeeeeeee",
        [{"step": "b", "files": ["final/b.png"], "role": "final"}],
        files=["final/b.png"],
        inputs=[{"reference": "output:wf/v1/a.png", "kind": "output"}],
    )
    meta = gallery_client.get(
        "/api/gallery/series/ep1/20261012-000004-eeeeeeee/final/b.png/metadata"
    ).json()
    assert meta["role"] == "final"
    assert meta["inputs"] == [{"reference": "output:wf/v1/a.png", "kind": "output"}]


def _role_less_run_script(output_dir):
    """A worker that writes a two-step run - b reads a's result - whose
    manifest.json records roles, and reports the manifest it held in
    memory, which never carries them."""
    from dw.runs import new_run_id

    def script(command):
        run_id = new_run_id({"command": 1})
        relative = f"wf/{run_id}"
        run_dir = os.path.join(output_dir, "wf", run_id)
        os.makedirs(run_dir, exist_ok=True)
        realized = {
            "id": "wf",
            "steps": [
                {"name": "a", "task": {"command": "noop", "arguments": {}}},
                {
                    "name": "b",
                    "task": {
                        "command": "noop",
                        "arguments": {"image": "previous_result:a"},
                    },
                },
            ],
        }
        steps = [
            {"step": "a", "files": ["a.png"], "role": "intermediate"},
            {"step": "b", "files": ["b.png"], "role": "final"},
        ]
        for name in ("a.png", "b.png"):
            with open(os.path.join(run_dir, name), "wb") as handle:
                handle.write(b"\x89PNG\r\n\x1a\n")
        with open(os.path.join(run_dir, "manifest.json"), "w") as handle:
            json.dump({"run_id": run_id, "status": "succeeded", "steps": steps}, handle)
        with open(os.path.join(run_dir, "workflow.json"), "w") as handle:
            json.dump(realized, handle)
        yield {
            "type": "progress",
            "event": "run_start",
            "run_id": run_id,
            "identity": "wf",
            "run_dir": relative,
            "version": 1,
        }
        manifest = [
            {"step": e["step"], "files": [os.path.join(run_dir, e["files"][0])]}
            for e in steps
        ]
        yield {"type": "success", "message": "ok", "run_count": 1, "manifest": manifest}

    return script


def test_job_detail_entries_carry_the_run_roles(tmp_path):
    """The worker reports a role-less manifest; the job detail fills each
    entry's role from the run directory, live and from history."""
    from fastapi.testclient import TestClient

    from dw.server.app import create_app
    from dw.server.jobs import JobManager
    from dw.workspace import Workspace

    from .test_server import ScriptedWorkerManager, valid_workflow, wait_for_status

    studio = Workspace(tmp_path / "studio", "flag").ensure()
    manager = JobManager(
        studio.outputs,
        worker_manager=ScriptedWorkerManager(_role_less_run_script(studio.outputs)),
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
    with TestClient(app, base_url="http://localhost") as client:
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        detail = wait_for_status(client, job["id"], ["succeeded"])
        assert all("role" not in e for e in manager.jobs[job["id"]].manifest)
        assert {e["step"]: e["role"] for e in detail["manifest"]} == {
            "a": "intermediate",
            "b": "final",
        }
        manager.jobs.pop(job["id"])
        again = client.get(f"/api/jobs/{job['id']}").json()
        assert {e["step"]: e["role"] for e in again["manifest"]} == {
            "a": "intermediate",
            "b": "final",
        }


def test_a_mixed_manifest_reports_only_recorded_roles(tmp_path, write_run):
    run_dir = write_run(
        str(tmp_path),
        "wf",
        "20261012-000010-aaaaaaaa",
        [
            {"step": "a", "files": ["a.png"], "role": "final"},
            {"step": "b", "files": ["b.png"]},
            {"step": "c", "files": ["c.png"], "role": "bogus"},
        ],
    )
    assert run_roles(run_dir) == {"a.png": "final"}


def test_run_inputs_keeps_only_well_formed_entries(tmp_path, write_run):
    from dw.server.outputs import run_inputs

    run_dir = write_run(
        str(tmp_path),
        "wf",
        "20261012-000013-aaaaaaaa",
        [],
        inputs=[
            {"reference": "asset:a.png", "kind": "asset"},
            {"reference": 3, "kind": "asset"},
            {"reference": "x", "kind": "variable"},
            "asset:b.png",
            {"reference": "output:wf/v1/a.png", "kind": "output", "extra": 1},
        ],
    )
    assert run_inputs(run_dir) == [
        {"reference": "asset:a.png", "kind": "asset"},
        {"reference": "output:wf/v1/a.png", "kind": "output"},
    ]


def test_role_cache_is_lru_and_safe_under_threads(tmp_path, write_run, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from dw.server import outputs

    monkeypatch.setattr(outputs, "_ROLE_CACHE", outputs.OrderedDict())
    monkeypatch.setattr(outputs, "_ROLE_CACHE_LIMIT", 4)
    runs = [
        write_run(
            str(tmp_path),
            "wf",
            f"20261012-0001{i:02d}-aaaaaaaa",
            [{"step": "a", "files": ["a.png"], "role": "final"}],
        )
        for i in range(12)
    ]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(run_roles, runs * 20))
    assert all(r == {"a.png": "final"} for r in results)
    assert len(outputs._ROLE_CACHE) == 4

    # the most recently used survives a newcomer; the oldest goes
    outputs._ROLE_CACHE.clear()
    for run in runs[:4]:
        run_roles(run)
    run_roles(runs[0])
    run_roles(runs[4])
    kept = {key[0] for key in outputs._ROLE_CACHE}
    assert runs[0] in kept and runs[1] not in kept


# A file a later run consumed reads as intermediate (#928)

SHOT_RUN = "20261015-000000-aaaaaaaa"
CUT_RUN = "20261015-010000-bbbbbbbb"


def _shot_and_cut(output_root, write_run, reference, **shot_entry):
    write_run(
        output_root,
        "shots",
        SHOT_RUN,
        [{"step": "shot", "files": ["final/shot.png"], "role": "final", **shot_entry}],
        files=["final/shot.png"],
    )
    write_run(
        output_root,
        "assembly",
        CUT_RUN,
        [{"step": "cut", "files": ["final/cut.png"], "role": "final"}],
        files=["final/cut.png"],
        inputs=[{"reference": reference, "kind": reference.split(":", 1)[0]}],
    )


def _roles(client, **params):
    listing = client.get("/api/gallery", params=params).json()["files"]
    return {f["label"]: f.get("role") for f in listing}


def test_a_file_a_later_run_read_lists_as_intermediate(
    gallery_client, output_root, write_run
):
    _shot_and_cut(output_root, write_run, "output:shots/latest/final/shot.png")
    assert _roles(gallery_client) == {"shot.png": "intermediate", "cut.png": "final"}
    assert _roles(gallery_client, role="final") == {"cut.png": "final"}
    assert _roles(gallery_client, role="intermediate") == {"shot.png": "intermediate"}
    meta = gallery_client.get(
        f"/api/gallery/shots/{SHOT_RUN}/final/shot.png/metadata"
    ).json()
    assert meta["role"] == "intermediate"


def test_a_declared_final_stays_final(gallery_client, output_root, write_run):
    _shot_and_cut(
        output_root,
        write_run,
        "output:shots/v1/final/shot.png",
        role_declared=True,
    )
    assert _roles(gallery_client, role="final") == {
        "shot.png": "final",
        "cut.png": "final",
    }


def test_a_pre_roles_file_a_later_run_read_stays_unknown(
    gallery_client, output_root, write_run
):
    write_run(
        output_root,
        "shots",
        SHOT_RUN,
        [{"step": "shot", "files": ["shot.png"]}],
        files=["shot.png"],
    )
    write_run(
        output_root,
        "assembly",
        CUT_RUN,
        [{"step": "cut", "files": ["cut.png"], "role": "final"}],
        files=["cut.png"],
        inputs=[{"reference": f"output:shots/{SHOT_RUN}/shot.png", "kind": "output"}],
    )
    assert _roles(gallery_client)["shot.png"] is None


def test_an_asset_kept_from_a_file_demotes_it(tmp_path, write_run):
    from fastapi.testclient import TestClient

    from dw.runs import record_kept_shots
    from dw.server.app import create_app
    from dw.server.jobs import JobManager
    from dw.workspace import Workspace

    from .test_server import ScriptedWorkerManager, success_script

    studio = Workspace(tmp_path / "studio", "studio").ensure()
    manager = JobManager(
        studio.outputs,
        worker_manager=ScriptedWorkerManager(success_script),
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
    with TestClient(app, base_url="http://localhost") as client:
        name = client.get("/api/gallery").json()["workspace"]
        os.makedirs(studio.assets, exist_ok=True)
        with open(os.path.join(studio.assets, "hero.png"), "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
        record_kept_shots(
            studio.assets,
            "hero.png",
            None,
            {"workspace": name, "source": f"shots/{SHOT_RUN}/final/shot.png"},
        )
        _shot_and_cut(studio.outputs, write_run, "asset:hero.png")
        assert _roles(client)["shot.png"] == "intermediate"
