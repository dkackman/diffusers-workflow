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
