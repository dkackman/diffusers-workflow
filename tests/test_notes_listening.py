"""Whether a notes channel is polling a workspace: the server remembers, in
memory, the last list read marked `X-DW-Notes-Channel: 1`."""

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.listening import LISTENING_WINDOW, Listening
from dw.workspace import Workspace

MARK = {"X-DW-Notes-Channel": "1"}


class Clock:
    def __init__(self):
        self.now = 1_800_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def client(tmp_path, clock):
    root = Workspace(tmp_path / "studio", "flag").ensure()
    app = create_app(
        workflow_dir=root.workflows,
        output_dir=root.outputs,
        prompt_dir=root.prompts,
        asset_dir=root.assets,
        workspace=root.root,
    )
    app.state.listening = Listening(clock=clock)
    with TestClient(app, base_url="http://localhost") as c:
        yield c


def state(client, name="default"):
    r = client.get(f"/api/workspaces/{name}/notes-channel")
    assert r.status_code == 200, r.text
    return r.json()


def test_no_poll_means_not_listening(client):
    assert state(client) == {"listening": False, "last_seen": None}


def test_a_marked_list_read_means_listening(client):
    assert (
        client.get("/api/workspaces/default/requests", headers=MARK).status_code == 200
    )
    got = state(client)
    assert got["listening"] is True
    assert got["last_seen"].endswith("Z") or "+00:00" in got["last_seen"]


def test_listening_lapses_after_the_window(client, clock):
    client.get("/api/workspaces/default/requests", headers=MARK)
    seen = state(client)["last_seen"]
    clock.now += LISTENING_WINDOW + 1
    assert state(client) == {"listening": False, "last_seen": seen}


def test_still_listening_inside_the_window(client, clock):
    client.get("/api/workspaces/default/requests", headers=MARK)
    clock.now += LISTENING_WINDOW - 1
    assert state(client)["listening"] is True


def test_an_unmarked_list_read_records_nothing(client):
    client.get("/api/workspaces/default/requests")
    client.get("/api/workspaces/default/requests", headers={"X-DW-Notes-Channel": "0"})
    assert state(client) == {"listening": False, "last_seen": None}


def test_one_workspaces_poll_does_not_mark_another():
    clock = Clock()
    listening = Listening(clock=clock)
    listening.seen("a")
    assert listening.state("a")["listening"] is True
    assert listening.state("b") == {"listening": False, "last_seen": None}


def test_an_unknown_workspace_is_refused_and_not_recorded(client):
    r = client.get("/api/workspaces/nope/requests", headers=MARK)
    assert r.status_code == 404
    assert client.get("/api/workspaces/nope/notes-channel").status_code == 404
    assert client.app.state.listening.state("nope")["last_seen"] is None


# --- a channel that watches the whole server (GET /api/requests, #1529) ---


def put(client, workspace, request_id, state="open"):
    body = {
        "title": request_id,
        "goal": "g",
        "items": [{"key": "a", "title": "A", "depends_on": []}],
        "revision": 0,
        "why": "w",
        "by": "agent",
    }
    r = client.put(f"/api/workspaces/{workspace}/requests/{request_id}", json=body)
    assert r.status_code == 200, r.text
    if state != "open":
        r = client.put(
            f"/api/workspaces/{workspace}/requests/{request_id}",
            json={**body, "revision": 1, "state": state},
        )
        assert r.status_code == 200, r.text


def add_workspace(client, name):
    r = client.post("/api/workspaces", json={"name": name})
    assert r.status_code == 201, r.text


def test_the_server_wide_list_has_every_workspaces_open_requests(client):
    add_workspace(client, "notes")
    put(client, "default", "first")
    put(client, "notes", "turbine")
    put(client, "notes", "shipped", state="done")
    r = client.get("/api/requests")
    assert r.status_code == 200, r.text
    got = {(row["workspace"], row["id"]) for row in r.json()["requests"]}
    assert got == {("default", "first"), ("notes", "turbine")}
    row = next(row for row in r.json()["requests"] if row["id"] == "turbine")
    assert row["state"] == "open" and row["items"] == 1 and "ready" in row
    assert "unreadable" not in r.json()


def test_a_marked_server_wide_read_means_listening_in_every_workspace(client, clock):
    add_workspace(client, "notes")
    assert client.get("/api/requests", headers=MARK).status_code == 200
    assert state(client)["listening"] is True
    assert state(client, "notes")["listening"] is True
    # a workspace made after the poll is covered too: the channel reads it
    # on its next poll
    add_workspace(client, "later")
    assert state(client, "later")["listening"] is True
    clock.now += LISTENING_WINDOW + 1
    assert state(client, "notes")["listening"] is False


def test_an_unmarked_server_wide_read_records_nothing(client):
    client.get("/api/requests")
    assert state(client) == {"listening": False, "last_seen": None}


def test_one_workspaces_broken_ledger_is_named_and_the_others_still_list(
    client, tmp_path
):
    add_workspace(client, "notes")
    put(client, "default", "first")
    (tmp_path / "studio" / "notes" / "requests.json").write_text("not json")
    r = client.get("/api/requests")
    assert r.status_code == 200, r.text
    assert [row["id"] for row in r.json()["requests"]] == ["first"]
    assert set(r.json()["unreadable"]) == {"notes"}


def test_a_workspace_poll_still_marks_only_its_own_after_a_server_wide_one(clock):
    listening = Listening(clock=clock)
    listening.seen_all()
    clock.now += LISTENING_WINDOW + 1
    listening.seen("a")
    assert listening.state("a")["listening"] is True
    assert listening.state("b")["listening"] is False


def test_a_workspace_that_cannot_be_resolved_is_named_not_skipped(client, monkeypatch):
    from fastapi import HTTPException

    from dw.server.routes import requests as routes

    add_workspace(client, "notes")
    put(client, "default", "first")
    real = routes._ledger_root

    def gone(state, name):
        if name == "notes":
            raise HTTPException(status_code=404, detail="no workspace 'notes'")
        return real(state, name)

    monkeypatch.setattr(routes, "_ledger_root", gone)
    r = client.get("/api/requests")
    assert r.status_code == 200, r.text
    assert [row["id"] for row in r.json()["requests"]] == ["first"]
    assert set(r.json()["unreadable"]) == {"notes"}
