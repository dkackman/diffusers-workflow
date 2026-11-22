# ruff: noqa: F811
"""Workspace change counters (a row of docs/ARCHITECTURE.md, "workspace
change counters"): every server write bumps its kind, after the write
succeeded, and a failed write bumps nothing. The registry in
dw/server/changes.py is tested alone first; then each write route is called
through the HTTP client and the counters read from
GET /api/workspaces/{name}/changes must show exactly the right kind move,
each by one."""

import threading

import pytest

from dw.server.changes import KINDS, Changes

from .test_ledger import item, plan
from .test_ledger_job_link import client, root  # noqa: F401 - fixtures (F811: tests take them by name)
from .test_ledger_job_link import finish, submit, view, write_plan
from .test_server import valid_workflow, wait_for_status

RUN_ID = "20261012-120000-aabbccdd"
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
PROMPT = {"prompt": {"text": "a red fox"}}


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


# Each case: (setup(client, run), call(client, run) -> response, expected move)
CASES = {
    "request-create": (
        lambda c, run: None,
        lambda c, run: c.put(REQ, json={**plan(revision=0), "by": "agent"}),
        {"requests": 1},
    ),
    "request-note-add": (
        lambda c, run: write_plan(c),
        lambda c, run: c.post(NOTES, json={"request": "cats", "text": "hi"}),
        {"requests": 1},
    ),
    "request-note-edit": (
        lambda c, run: _with_note(c),
        lambda c, run: c.put(
            NOTES, json={"request": "cats", "note": "n1", "text": "second"}
        ),
        {"requests": 1},
    ),
    "request-note-delete": (
        lambda c, run: _with_note(c),
        lambda c, run: c.delete(f"{NOTES}?request=cats&note=n1"),
        {"requests": 1},
    ),
    "request-delete": (
        lambda c, run: write_plan(c),
        lambda c, run: c.delete(f"{REQ}?revision={_request_revision(c)}"),
        {"requests": 1},
    ),
    "prompt-save": (
        lambda c, run: None,
        lambda c, run: c.put("/api/prompts/Fox", json=PROMPT),
        {"prompts": 1},
    ),
    "prompt-delete": (
        lambda c, run: c.put("/api/prompts/Fox", json=PROMPT),
        lambda c, run: c.delete("/api/prompts/Fox"),
        {"prompts": 1},
    ),
    "workflow-save": (
        lambda c, run: None,
        lambda c, run: c.put(
            "/api/workflows/mine", json={"workflow": valid_workflow()}
        ),
        {"workflows": 1},
    ),
    "workflow-patch": (
        lambda c, run: c.put(
            "/api/workflows/mine", json={"workflow": valid_workflow()}
        ),
        lambda c, run: c.patch(
            "/api/workflows/mine", json={"variables": {"prompt": "e"}}
        ),
        {"workflows": 1},
    ),
    "workflow-delete": (
        lambda c, run: c.put(
            "/api/workflows/mine", json={"workflow": valid_workflow()}
        ),
        lambda c, run: c.delete("/api/workflows/mine"),
        {"workflows": 1},
    ),
    "asset-upload": (
        lambda c, run: None,
        lambda c, run: c.post(
            "/api/uploads", params={"filename": "i.png"}, content=PNG
        ),
        {"assets": 1},
    ),
    "asset-keep": (
        lambda c, run: None,
        lambda c, run: c.post("/api/assets/keep", json={"name": f"{run}/a.png"}),
        {"assets": 1},
    ),
    "asset-delete": (
        lambda c, run: c.post("/api/assets/keep", json={"name": f"{run}/a.png"}),
        lambda c, run: c.delete("/api/assets/a.png"),
        {"assets": 1},
    ),
    "asset-note": (
        lambda c, run: c.post("/api/assets/keep", json={"name": f"{run}/a.png"}),
        lambda c, run: c.put("/api/notes", json={"target": "asset:a.png", "note": "n"}),
        {"assets": 1},
    ),
    "star": (
        lambda c, run: None,
        lambda c, run: c.put(
            "/api/stars", json={"name": f"{run}/a.png", "starred": True}
        ),
        {"gallery": 1},
    ),
    "file-note": (
        lambda c, run: None,
        lambda c, run: c.put(
            "/api/notes", json={"target": f"{run}/a.png", "note": "n"}
        ),
        {"gallery": 1},
    ),
    "run-note": (
        lambda c, run: None,
        lambda c, run: c.put("/api/notes", json={"target": run, "note": "n"}),
        {"gallery": 1},
    ),
    "output-delete": (
        lambda c, run: None,
        lambda c, run: c.delete(f"/api/gallery/{run}/a.png"),
        {"gallery": 1},
    ),
    "workspace-create": (
        lambda c, run: None,
        lambda c, run: c.post("/api/workspaces", json={"name": "shots"}),
        {"workspaces": 1},
    ),
    "workspace-delete": (
        lambda c, run: c.post("/api/workspaces", json={"name": "shots"}),
        lambda c, run: c.delete("/api/workspaces/shots?acknowledged=true"),
        {"workspaces": 1},
    ),
    "workspace-note": (
        lambda c, run: None,
        lambda c, run: c.put(
            "/api/notes", json={"target": "workspace:default", "note": "n"}
        ),
        {"workspaces": 1},
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_each_write_moves_exactly_its_kind(client, run, case):
    setup, call, expected = CASES[case]
    setup(client, run)
    before = counters(client)
    response = call(client, run)
    assert 200 <= response.status_code < 300, response.text
    assert moved(before, counters(client)) == expected


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


def test_deleting_a_jobs_run_moves_gallery_and_jobs(client):
    job = finish(client, submit(client, request=None, item_key=None))
    wait_until(lambda: counters(client)["kinds"]["gallery"] >= 1)
    before = counters(client)
    r = client.delete(f"/api/jobs/{job['id']}/run")
    assert r.status_code == 200, r.text
    assert moved(before, counters(client)) == {"gallery": 1, "jobs": 1}


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
