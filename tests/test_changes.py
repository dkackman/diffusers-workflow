# ruff: noqa: F811
"""Workspace change counters (a row of docs/ARCHITECTURE.md, "workspace
change counters"): every server write bumps its kind, after the write
succeeded, and a failed write bumps nothing. The registry in
dw/server/changes.py is tested alone first; then each write route is called
through the HTTP client and the counters read from
GET /api/workspaces/{name}/changes must show exactly the right kind move,
each by one."""

import threading
import time

import pytest

from fastapi.routing import APIRoute

from dw.server.changes import (
    ASSETS,
    GALLERY,
    JOBS,
    KINDS,
    NO_CHANGE,
    WORKSPACES,
    Change,
    ChangeRoute,
    Changes,
    Declared,
    pending_change,
    writes,
)

from .test_ledger import item, plan
from .test_ledger_job_link import client, root  # noqa: F401 - fixtures (F811: tests take them by name)
from .test_ledger_job_link import finish, submit, view, write_plan
from .test_server import valid_workflow, wait_for_status
from fastapi.testclient import TestClient

RUN_ID = "20261012-120000-aabbccdd"
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
PROMPT = {"prompt": {"text": "a red fox"}}
LORA = {
    "model_name": "acme/lora",
    "base_models": ["acme/base"],
    "description": "A look",
    "use_when": "always",
    "status": "trial",
    "evidence": [{"note": "seen once"}],
}


# ------------------------------------------------------------------ registry


def test_counters_only_grow():
    changes = Changes()
    seen = changes.snapshot("a")["kinds"]["prompts"]
    for _ in range(3):
        changes.bump("a", "prompts")
        now = changes.snapshot("a")["kinds"]["prompts"]
        assert now == seen + 1
        seen = now


def test_two_workspaces_are_independent():
    changes = Changes()
    changes.bump("a", "gallery")
    assert changes.snapshot("a")["kinds"]["gallery"] == 1
    assert changes.snapshot("b")["kinds"]["gallery"] == 0


def test_forget_gives_a_new_epoch_and_zeroed_counters():
    changes = Changes()
    changes.bump("a", "jobs", "assets")
    before = changes.snapshot("a")
    changes.forget("a")
    after = changes.snapshot("a")
    assert after["epoch"] != before["epoch"]
    assert set(after["kinds"].values()) == {0}


def test_bump_all_moves_the_kind_in_every_workspace():
    changes = Changes()
    changes.bump("a", "prompts")
    changes.bump_all("prompts")
    assert changes.snapshot("a")["kinds"]["prompts"] == 2
    assert changes.snapshot("b")["kinds"]["prompts"] == 1
    assert changes.snapshot("b")["kinds"]["jobs"] == 0


def test_bump_workspaces_shows_in_every_snapshot():
    changes = Changes()
    changes.bump_workspaces()
    assert changes.snapshot("a")["workspaces"] == 1
    assert changes.snapshot("b")["workspaces"] == 1


def test_an_unknown_kind_raises():
    changes = Changes()
    with pytest.raises(ValueError):
        changes.bump("a", "nope")
    with pytest.raises(ValueError):
        changes.bump_all("nope")
    with pytest.raises(ValueError):
        changes.bump("a")


def test_no_workspace_is_the_default_workspace():
    changes = Changes()
    changes.bump(None, "requests")
    assert changes.snapshot(None) == changes.snapshot("default")
    assert changes.snapshot("default")["kinds"]["requests"] == 1


# --------------------------------------------------------------------- route


def counters(client, name="default"):
    r = client.get(f"/api/workspaces/{name}/changes")
    assert r.status_code == 200, r.text
    return r.json()


def moved(before, after):
    """What moved between two snapshots: {kind or "workspaces": by}."""
    found = {k: after["kinds"][k] - before["kinds"][k] for k in KINDS}
    found["workspaces"] = after["workspaces"] - before["workspaces"]
    return {k: by for k, by in found.items() if by}


@pytest.fixture
def run(client, write_run):
    """A run of workflow `wf` holding a.png, as the gallery names it."""
    steps = [{"step": "gen", "files": ["a.png"]}]
    write_run(client.root.outputs, "wf", RUN_ID, steps, files=["a.png"])
    return f"wf/{RUN_ID}"


def test_the_route_answers_a_default_workspace(client):
    body = counters(client)
    assert isinstance(body["epoch"], str)
    assert set(body["kinds"]) == set(KINDS)
    assert {"models", "loras"} <= set(body["kinds"])
    assert all(isinstance(v, int) for v in body["kinds"].values())
    assert isinstance(body["workspaces"], int)


def test_an_unknown_workspace_is_404(client):
    assert client.get("/api/workspaces/nope/changes").status_code == 404


@pytest.mark.parametrize("name", ["..%2Fx", "bad name!"])
def test_a_traversal_shaped_workspace_name_is_refused(client, name):
    assert client.get(f"/api/workspaces/{name}/changes").status_code in (400, 404)


def test_a_workspace_made_deleted_and_made_again_has_a_new_epoch(client):
    assert client.post("/api/workspaces", json={"name": "shots"}).status_code < 300
    first = counters(client, "shots")["epoch"]
    r = client.delete("/api/workspaces/shots?acknowledged=true")
    assert r.status_code < 300, r.text
    assert client.post("/api/workspaces", json={"name": "shots"}).status_code < 300
    assert counters(client, "shots")["epoch"] != first


def test_prompts_are_shared_by_every_workspace(client):
    assert client.post("/api/workspaces", json={"name": "shots"}).status_code < 300
    here, there = counters(client), counters(client, "shots")
    assert client.put("/api/prompts/Fox", json=PROMPT).status_code == 200
    assert moved(here, counters(client)) == {"prompts": 1}
    assert moved(there, counters(client, "shots")) == {"prompts": 1}


# ----------------------------------------------------- each write, its kind

REQ = "/api/workspaces/default/requests/cats"
NOTES = "/api/workspaces/default/request-notes"


def _request_revision(client):
    return view(client)["revision"]


def _with_note(client):
    write_plan(client)
    r = client.post(NOTES, json={"request": "cats", "text": "first"})
    assert r.status_code == 200, r.text


def _finished_job(client):
    """A job that ran and wrote its run, its own gallery bump already seen."""
    client.job = finish(client, submit(client, request=None, item_key=None))
    wait_until(lambda: counters(client)["kinds"]["gallery"] >= 1)


def _run_only_in_shots(client, run):
    """`run` copied into workspace `shots`, then deleted from the default."""
    assert client.post("/api/workspaces", json={"name": "shots"}).status_code < 300
    r = client.post("/api/workspaces/shots/copy", json={"runs": [run]})
    assert r.status_code == 200, r.text
    assert client.delete(f"/api/gallery/{run}").status_code == 200


# Each case, keyed by the route's name (its handler), or `<route>:<variant>`
# when one route has several: (setup(client, run), call(client, run) ->
# response, expected move in the default workspace)
CASES = {
    "put_request": (
        lambda c, run: None,
        lambda c, run: c.put(REQ, json={**plan(revision=0), "by": "agent"}),
        {"requests": 1},
    ),
    "add_request_note": (
        lambda c, run: write_plan(c),
        lambda c, run: c.post(NOTES, json={"request": "cats", "text": "hi"}),
        {"requests": 1},
    ),
    "edit_request_note": (
        lambda c, run: _with_note(c),
        lambda c, run: c.put(
            NOTES, json={"request": "cats", "note": "n1", "text": "second"}
        ),
        {"requests": 1},
    ),
    "delete_request_note": (
        lambda c, run: _with_note(c),
        lambda c, run: c.delete(f"{NOTES}?request=cats&note=n1"),
        {"requests": 1},
    ),
    "delete_request": (
        lambda c, run: write_plan(c),
        lambda c, run: c.delete(f"{REQ}?revision={_request_revision(c)}"),
        {"requests": 1},
    ),
    "save_prompt": (
        lambda c, run: None,
        lambda c, run: c.put("/api/prompts/Fox", json=PROMPT),
        {"prompts": 1},
    ),
    "delete_prompt": (
        lambda c, run: c.put("/api/prompts/Fox", json=PROMPT),
        lambda c, run: c.delete("/api/prompts/Fox"),
        {"prompts": 1},
    ),
    "save_workflow": (
        lambda c, run: None,
        lambda c, run: c.put(
            "/api/workflows/mine", json={"workflow": valid_workflow()}
        ),
        {"workflows": 1},
    ),
    "patch_workflow": (
        lambda c, run: c.put(
            "/api/workflows/mine", json={"workflow": valid_workflow()}
        ),
        lambda c, run: c.patch(
            "/api/workflows/mine", json={"variables": {"prompt": "e"}}
        ),
        {"workflows": 1},
    ),
    "delete_workflow": (
        lambda c, run: c.put(
            "/api/workflows/mine", json={"workflow": valid_workflow()}
        ),
        lambda c, run: c.delete("/api/workflows/mine"),
        {"workflows": 1},
    ),
    "upload_media": (
        lambda c, run: None,
        lambda c, run: c.post(
            "/api/uploads", params={"filename": "i.png"}, content=PNG
        ),
        {"assets": 1},
    ),
    "keep_output_as_asset": (
        lambda c, run: None,
        lambda c, run: c.post("/api/assets/keep", json={"name": f"{run}/a.png"}),
        {"assets": 1},
    ),
    "delete_asset": (
        lambda c, run: c.post("/api/assets/keep", json={"name": f"{run}/a.png"}),
        lambda c, run: c.delete("/api/assets/a.png"),
        {"assets": 1},
    ),
    "put_note:asset": (
        lambda c, run: c.post("/api/assets/keep", json={"name": f"{run}/a.png"}),
        lambda c, run: c.put("/api/notes", json={"target": "asset:a.png", "note": "n"}),
        {"assets": 1},
    ),
    "put_star": (
        lambda c, run: None,
        lambda c, run: c.put(
            "/api/stars", json={"name": f"{run}/a.png", "starred": True}
        ),
        {"gallery": 1},
    ),
    "put_note:file": (
        lambda c, run: None,
        lambda c, run: c.put(
            "/api/notes", json={"target": f"{run}/a.png", "note": "n"}
        ),
        {"gallery": 1},
    ),
    "put_note:run": (
        lambda c, run: None,
        lambda c, run: c.put("/api/notes", json={"target": run, "note": "n"}),
        {"gallery": 1},
    ),
    "delete_output": (
        lambda c, run: None,
        lambda c, run: c.delete(f"/api/gallery/{run}/a.png"),
        {"gallery": 1},
    ),
    "add_workspace": (
        lambda c, run: None,
        lambda c, run: c.post("/api/workspaces", json={"name": "shots"}),
        {"workspaces": 1},
    ),
    "remove_workspace": (
        lambda c, run: c.post("/api/workspaces", json={"name": "shots"}),
        lambda c, run: c.delete("/api/workspaces/shots?acknowledged=true"),
        {"workspaces": 1},
    ),
    "put_note:workspace": (
        lambda c, run: None,
        lambda c, run: c.put(
            "/api/notes", json={"target": "workspace:default", "note": "n"}
        ),
        {"workspaces": 1},
    ),
    "delete_job_run": (
        lambda c, run: _finished_job(c),
        lambda c, run: c.delete(f"/api/jobs/{c.job['id']}/run"),
        {"gallery": 1, "jobs": 1},
    ),
    "delete_cached_model": (
        lambda c, run: None,
        lambda c, run: c.delete("/api/models", params={"repo": "acme/tiny"}),
        {"models": 1},
    ),
    "clear_cache": (
        lambda c, run: None,
        lambda c, run: c.delete("/api/caches/inductor"),
        {"models": 1},
    ),
    "save_lora": (
        lambda c, run: None,
        lambda c, run: c.put("/api/loras/mine", json={"entry": LORA}),
        {"loras": 1},
    ),
    "delete_lora": (
        lambda c, run: c.put("/api/loras/mine", json={"entry": LORA}),
        lambda c, run: c.delete("/api/loras/mine"),
        {"loras": 1},
    ),
    "copy_into_workspace": (
        _run_only_in_shots,
        lambda c, run: c.post(
            "/api/workspaces/default/copy", json={"source": "shots", "runs": [run]}
        ),
        {"gallery": 1},
    ),
}

# Routes that declare kinds but have no case above: {route name: why}
UNCASED = {
    "start_download": "reaches the network, and the thread it starts bumps again "
    "when the download ends: test_a_finished_download_moves_models",
    "cancel_download": "needs a download in flight, which ends on a thread that "
    "bumps again: test_a_cancelled_download_moves_models",
    "delete_cache_entry": "needs a quantized entry on disk: the same route "
    "code as clear_cache, declared the same way",
    "update_diffusers": "runs pip: test_a_finished_update_moves_models",
}


WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@pytest.mark.parametrize("case", list(CASES))
def test_each_write_moves_exactly_its_kind(client, run, case, monkeypatch):
    from dw.server.routes import system

    monkeypatch.setattr(system, "delete_model", lambda repo: 0)
    monkeypatch.setattr(system.disk_caches, "get", lambda name: None)
    monkeypatch.setattr(
        system.disk_caches, "clear", lambda name: {"freed": 0, "files": 0}
    )
    setup, call, expected = CASES[case]
    setup(client, run)
    server_wide = set(expected) <= {"models", "loras"}
    if server_wide:
        assert client.post("/api/workspaces", json={"name": "shots"}).status_code < 300
        other = counters(client, "shots")
    before = counters(client)
    response = call(client, run)
    assert 200 <= response.status_code < 300, response.text
    assert moved(before, counters(client)) == expected
    if server_wide:
        # No workspace owns the hub cache or the LoRA catalog: both move
        assert moved(other, counters(client, "shots")) == expected


def _download_manager(changes):
    from dw.hub_cache import DownloadManager

    def instant(repo_id, tqdm_class=None):
        tqdm_class(total=1).update(1)

    class Info:
        siblings = []

    manager = DownloadManager(download_fn=instant, info_fn=lambda repo_id: Info())
    manager.changes = changes
    return manager


def _wait_download_ends(manager, download_id):
    deadline = time.time() + 5
    while time.time() < deadline:
        if manager.status(download_id)["status"] != "downloading":
            return
        time.sleep(0.02)
    raise AssertionError("the download did not end")


def test_a_finished_download_moves_models(tmp_path):
    from dw.server.app import create_app

    manager = _download_manager(None)
    app = create_app(
        workflow_dir=str(tmp_path),
        output_dir=str(tmp_path / "outputs"),
        download_manager=manager,
    )
    with TestClient(app, base_url="http://localhost") as http:
        assert app.state.downloads.changes is app.state.changes
        before = counters(http)["kinds"]["models"]
        started = http.post("/api/models/download", json={"repo_id": "acme/tiny"})
        assert started.status_code == 202
        _wait_download_ends(manager, started.json()["id"])
        # The route's own bump, then the thread's when the download ended
        wait_until(lambda: counters(http)["kinds"]["models"] == before + 2)


def test_a_failed_download_moves_models():
    from dw.hub_cache import DownloadManager

    def broken(repo_id, tqdm_class=None):
        raise RuntimeError("no network")

    changes = Changes()
    manager = DownloadManager(download_fn=broken, info_fn=lambda repo_id: None)
    manager.changes = changes
    entry = manager.start("acme/tiny")
    _wait_download_ends(manager, entry["id"])
    assert manager.status(entry["id"])["status"] == "failed"
    assert changes.snapshot("any")["kinds"]["models"] == 1


def test_a_cancelled_download_moves_models():
    from dw.hub_cache import DownloadManager

    started, release = threading.Event(), threading.Event()

    def slow(repo_id, tqdm_class=None):
        tracker = tqdm_class(total=10)
        started.set()
        release.wait(timeout=5)
        tracker.update(1)  # the tick that raises out of a cancelled download

    changes = Changes()
    manager = DownloadManager(download_fn=slow, info_fn=lambda repo_id: None)
    manager.changes = changes
    entry = manager.start("acme/tiny")
    assert started.wait(5)
    manager.cancel(entry["id"])
    release.set()
    _wait_download_ends(manager, entry["id"])
    assert manager.status(entry["id"])["status"] == "cancelled"
    assert changes.snapshot("any")["kinds"]["models"] == 1


def test_a_finished_update_moves_models():
    from types import SimpleNamespace

    from dw.server.updater import DiffusersUpdater

    changes = Changes()
    ok = SimpleNamespace(returncode=0, stdout="", stderr="")
    updater = DiffusersUpdater(run_fn=lambda commit=None, revert=False: ok)
    updater.changes = changes
    updater.start()
    wait_until(lambda: updater.status()["status"] == "succeeded")
    wait_until(lambda: changes.snapshot("any")["kinds"]["models"] == 1)


def test_a_background_bump_that_raises_is_swallowed():
    from dw.hub_cache import DownloadManager
    from dw.server.updater import DiffusersUpdater

    class Broken:
        def bump_all(self, *kinds):
            raise RuntimeError("boom")

    manager = DownloadManager(download_fn=lambda *a, **k: None, info_fn=lambda r: None)
    manager.changes = Broken()
    manager._finish({"status": "downloading", "cancel_requested": False}, "completed")
    updater = DiffusersUpdater()
    updater.changes = Broken()
    updater._changed()


# ---------------------------------------------------------------------- jobs


@pytest.fixture
def held(client):
    """One job running and held at the worker's gate, so the next ones queue."""
    client.script.gate = threading.Event()
    running = submit(client, request=None, item_key=None)
    wait_for_status(client, running["id"], {"running"})
    try:
        yield client
    finally:
        client.script.gate.set()


def test_a_job_moves_jobs_and_its_outputs_move_gallery(client):
    before = counters(client)
    job = submit(client, request=None, item_key=None)
    finish(client, job)
    wait_until(
        lambda: counters(client)["kinds"]["gallery"] > before["kinds"]["gallery"]
    )
    after = moved(before, counters(client))
    assert after["jobs"] >= 1
    assert after["gallery"] >= 1


def wait_until(check, timeout=5.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError("the counters never moved")


def test_moving_a_queued_job_moves_jobs(held):
    one = submit(held, request=None, item_key=None)
    two = submit(held, request=None, item_key=None)
    before = counters(held)
    r = held.post(f"/api/jobs/{two['id']}/move", json={"direction": "front"})
    assert r.status_code == 200, r.text
    assert moved(before, counters(held)) == {"jobs": 1}
    assert one["id"] != two["id"]


def test_cancelling_a_queued_job_moves_jobs(held):
    queued = submit(held, request=None, item_key=None)
    before = counters(held)
    r = held.post(f"/api/jobs/{queued['id']}/cancel")
    assert r.status_code == 200, r.text
    assert moved(before, counters(held)) == {"jobs": 1}


# ------------------------------------------------------- a failed write: none


def test_a_prompt_save_with_an_invalid_body_moves_nothing(client):
    before = counters(client)
    r = client.put("/api/prompts/Bad", json={"prompt": {"description": "no text"}})
    assert r.status_code == 400
    assert counters(client) == before


def test_a_workflow_save_with_no_definition_moves_nothing(client):
    before = counters(client)
    r = client.put("/api/workflows/mine", json={})
    assert 400 <= r.status_code < 500
    assert counters(client) == before


def test_a_request_put_at_a_stale_revision_moves_nothing(client):
    write_plan(client)
    before = counters(client)
    r = client.put(REQ, json={**plan(revision=0, items=[item("a")]), "by": "agent"})
    assert r.status_code == 409
    assert counters(client) == before


def test_deleting_an_unknown_asset_moves_nothing(client):
    before = counters(client)
    assert client.delete("/api/assets/nope.png").status_code == 404
    assert counters(client) == before


def test_starring_a_missing_file_moves_nothing(client, run):
    before = counters(client)
    r = client.put("/api/stars", json={"name": f"{run}/nope.png", "starred": True})
    assert 400 <= r.status_code < 500
    assert counters(client) == before


# ------------------------------------- which asset library a write moved
# Shared or not is the `LibraryRoot.origin` that `AssetLibrary.find` answered
# (dw/library.py, LibraryPath), never a path comparison of the counter's own.


def _keep(client, run, shared):
    r = client.post("/api/assets/keep", json={"name": f"{run}/a.png", "shared": shared})
    assert 200 <= r.status_code < 300, r.text


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.delete("/api/assets/a.png"),
        lambda c: c.put("/api/notes", json={"target": "asset:a.png", "note": "n"}),
    ],
    ids=["delete", "note"],
)
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "own"])
def test_an_asset_write_moves_every_workspace_only_when_shared(
    client, run, call, shared
):
    assert client.post("/api/workspaces", json={"name": "shots"}).status_code < 300
    _keep(client, run, shared)
    here, there = counters(client), counters(client, "shots")
    response = call(client)
    assert 200 <= response.status_code < 300, response.text
    assert moved(here, counters(client)) == {"assets": 1}
    assert moved(there, counters(client, "shots")) == ({"assets": 1} if shared else {})


@pytest.mark.parametrize("origin, everywhere", [("common", True), ("workspace", False)])
def test_change_in_library_decides_from_the_roots_origin(origin, everywhere):
    from dw.library import LibraryRoot

    changes = Changes()
    ws = type("Ws", (), {"name": "shots", "common_assets": "/elsewhere/assets"})()
    # The root's directory is nowhere near ws.common_assets (a symlinked or
    # differently normalised shared library): only its origin decides
    change = Change(Declared((ASSETS,), False, None, None))
    change.in_library(ws, LibraryRoot("/linked/common", origin, True))
    change.apply(changes, "elsewhere", "delete_asset")
    assert changes.snapshot("shots")["kinds"]["assets"] == 1
    assert changes.snapshot("other")["kinds"]["assets"] == (1 if everywhere else 0)
    assert changes.snapshot("elsewhere")["kinds"]["assets"] == (1 if everywhere else 0)


def test_change_only_refuses_an_undeclared_kind():
    change = Change(Declared((ASSETS, GALLERY), False, None, None))
    with pytest.raises(ValueError):
        change.only(JOBS)
    with pytest.raises(ValueError):
        change.only()
    assert change.only(GALLERY).kinds == (GALLERY,)


def test_a_declaration_names_known_kinds_and_a_reason():
    with pytest.raises(ValueError):
        writes()
    with pytest.raises(ValueError):
        writes("nope")
    with pytest.raises(ValueError):
        NO_CHANGE("")
    assert writes(WORKSPACES)(lambda: None).writes.kinds == (WORKSPACES,)


# ------------------------------------------- every write route declares


def _write_routes(app):
    return [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.methods & WRITE_METHODS
    ]


def test_every_write_route_declares_what_it_changes(client):
    undeclared = sorted(
        route.name
        for route in _write_routes(client.app)
        if getattr(route.endpoint, "writes", None) is None
    )
    assert undeclared == []


def test_every_write_route_is_a_change_route(client):
    plain = sorted(
        route.name
        for route in _write_routes(client.app)
        if not isinstance(route, ChangeRoute)
    )
    assert plain == []


def test_every_declaring_route_has_a_case(client):
    declaring = {
        route.name
        for route in _write_routes(client.app)
        if getattr(route.endpoint, "writes", None) is not None
        and route.endpoint.writes.kinds
        and not route.endpoint.writes.by
    }
    cased = {case.split(":")[0] for case in CASES}
    assert declaring == cased | set(UNCASED)


def test_no_route_bumps_by_hand():
    import pathlib

    routes = pathlib.Path(__file__).parent.parent / "dw" / "server" / "routes"
    banned = (
        "changed(",
        "changed_everywhere(",
        "changed_assets(",
        ".bump(",
        ".bump_all(",
        ".bump_workspaces(",
    )
    found = sorted(
        f"{path.name}: {needle}"
        for path in routes.glob("*.py")
        for needle in banned
        if needle in path.read_text()
    )
    assert found == []


def test_a_copy_of_nothing_moves_nothing(client, run):
    assert client.post("/api/workspaces", json={"name": "shots"}).status_code < 300
    body = {"runs": [run]}
    assert client.post("/api/workspaces/shots/copy", json=body).status_code == 200
    before = counters(client, "shots")
    again = client.post("/api/workspaces/shots/copy", json=body)
    assert again.status_code == 200, again.text
    assert again.json()["already_present"] == [run]
    assert counters(client, "shots") == before


def test_a_handler_that_raises_bumps_nothing():
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.routing import APIRouter
    from fastapi.testclient import TestClient

    done = []
    router = APIRouter(route_class=ChangeRoute)

    @router.post("/refused")
    @writes(GALLERY)
    def refused(change: Change = Depends(pending_change)):
        change.in_workspace("a")
        done.append("written")
        raise HTTPException(status_code=409, detail="no")

    @router.post("/accepted")
    @writes(GALLERY)
    def accepted(change: Change = Depends(pending_change)):
        change.in_workspace("a")
        return {}

    app = FastAPI()
    app.state.changes = Changes()
    app.router.routes.extend(router.routes)
    http = TestClient(app)
    assert http.post("/refused").status_code == 409
    assert done == ["written"]
    assert app.state.changes.snapshot("a")["kinds"]["gallery"] == 0
    assert http.post("/accepted").status_code == 200
    assert app.state.changes.snapshot("a")["kinds"]["gallery"] == 1
