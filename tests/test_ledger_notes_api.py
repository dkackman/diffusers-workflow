"""The request-notes routes: a user's note on a request or an item, its
reference to a linked job and one of that job's files checked by the
server, and the gallery metadata naming the linked jobs that published a
file."""

import os
import threading

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.workspace import Workspace

from .test_ledger import item, plan
from .test_ledger_job_link import Scripts, finish, submit, view, write_plan
from .test_server import ScriptedWorkerManager, wait_for_status

WS = "default"
NOTES = f"/api/workspaces/{WS}/request-notes"


class ReusingScripts(Scripts):
    """`Scripts`, plus: with `reuse` set to an output file's absolute path,
    the run's manifest also lists that file as a step-cache hit."""

    def __init__(self):
        super().__init__()
        self.reuse = None

    def __call__(self, command):
        for event in super().__call__(command):
            if event.get("type") == "success" and self.reuse:
                event["manifest"] = [
                    {"step": "cached", "files": [self.reuse], "reused": True},
                    *event["manifest"],
                ]
            yield event


@pytest.fixture
def root(tmp_path):
    return Workspace(tmp_path / "studio", "flag").ensure()


@pytest.fixture
def client(root, tmp_path):
    script = ReusingScripts()
    manager = JobManager(
        root.outputs,
        worker_manager=ScriptedWorkerManager(script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=root.workflows,
    )
    app = create_app(
        workflow_dir=root.workflows,
        output_dir=root.outputs,
        job_manager=manager,
        prompt_dir=root.prompts,
        asset_dir=root.assets,
        workspace=root.root,
    )
    with TestClient(app, base_url="http://localhost") as c:
        c.manager = manager
        c.script = script
        c.root = root
        write_plan(c)
        yield c


@pytest.fixture
def take(client):
    """One finished job linked to cats/a, and the name of its file."""
    job = finish(client, submit(client))
    proof = view(client)["items"][0]["attempts"][0]["output"]
    return job["id"], proof["name"]


def add(client, expect=200, **body):
    r = client.post(NOTES, json={"request": "cats", **body})
    assert r.status_code == expect, r.text
    return r.json()


def refused(client, code, expect=422, **body):
    detail = add(client, expect=expect, **body)["detail"]
    assert detail["code"] == code, detail
    return detail


def notes_of(client, key=None):
    request = view(client)
    if key is None:
        return request["notes"]
    return next(i for i in request["items"] if i["key"] == key)["notes"]


# --- adding a note ------------------------------------------------------------


def test_post_a_note_with_a_ref_to_a_linked_job_output(client, take):
    job, name = take
    written = add(
        client, item="a", text="reroll this", ref={"job": job, "output": name}
    )
    note = written["note"]
    assert note["ref"] == {"job": job, "output": name, "kind": "image"}
    assert note["text"] == "reroll this" and note["id"] == "n1"
    assert note["created"] == note["at"]
    read = view(client)
    assert written["revision"] == read["revision"]
    assert read["history"][-1]["by"] == "user"
    [stored] = notes_of(client, "a")
    assert stored["ref"] == {"job": job, "output": name, "kind": "image", "attempt": 1}


def test_post_a_ref_with_only_a_job(client, take):
    job, _ = take
    note = add(client, item="a", text="the whole take", ref={"job": job})["note"]
    assert note["ref"] == {"job": job}


def test_post_a_note_without_a_ref_on_the_request_itself(client):
    note = add(client, text="all of it, warmer")["note"]
    assert "ref" not in note
    assert [n["text"] for n in notes_of(client)] == ["all of it, warmer"]


def test_a_request_note_may_name_a_job_of_any_item(client, take):
    job, name = take
    note = add(client, text="this one", ref={"job": job, "output": name})["note"]
    assert note["ref"]["job"] == job


def test_post_a_ref_to_a_job_not_linked_to_the_item_is_422(client, take):
    job, _ = take
    refused(client, "invalid_note_ref", item="b", text="x", ref={"job": job})
    loose = finish(client, submit(client, request=None, item_key=None))
    refused(client, "invalid_note_ref", item="a", text="x", ref={"job": loose["id"]})
    refused(client, "invalid_note_ref", text="x", ref={"job": loose["id"]})
    refused(client, "invalid_note_ref", item="a", text="x", ref={"job": "nope"})
    assert notes_of(client, "a") == [] and notes_of(client, "b") == []


def test_post_a_ref_to_another_requests_job_is_422(client):
    write_plan(client, request_id="dogs")
    other = finish(client, submit(client, request="dogs", item_key="a"))
    refused(client, "invalid_note_ref", item="a", text="x", ref={"job": other["id"]})


def test_post_a_ref_output_not_in_the_manifest_is_422(client, take):
    job, name = take
    other = os.path.join(os.path.dirname(name), "other.png")
    refused(
        client,
        "invalid_note_ref",
        item="a",
        text="x",
        ref={"job": job, "output": other},
    )
    refused(
        client, "invalid_note_ref", item="a", text="x", ref={"job": job, "output": ""}
    )


def test_post_a_ref_output_absolute_or_dotdot_is_422(client, take, root):
    job, name = take
    folder, run, file = name.split("/")
    for output in (
        os.path.join(root.outputs, name),
        f"/{name}",
        f"{folder}/{run}/../{run}/{file}",
        f"../outputs/{name}",
    ):
        refused(
            client,
            "invalid_note_ref",
            item="a",
            text="x",
            ref={"job": job, "output": output},
        )


def test_the_caller_cannot_set_the_kind_or_the_author(client, take):
    job, name = take
    for body in (
        {"text": "x", "ref": {"job": job, "output": name, "kind": "video"}},
        {"text": "x", "by": "agent"},
        {"text": "x", "ref": {"output": name}},
    ):
        r = client.post(NOTES, json={"request": "cats", "item": "a", **body})
        assert r.status_code == 422, (body, r.text)
    assert notes_of(client, "a") == []


def test_a_live_job_takes_a_ref_without_an_output(client, take):
    _, name = take
    client.script.gate = threading.Event()
    try:
        live = submit(client)
        wait_for_status(client, live["id"], {"running"})
        refused(
            client,
            "invalid_note_ref",
            item="a",
            text="x",
            ref={"job": live["id"], "output": name},
        )
        note = add(client, item="a", text="watching", ref={"job": live["id"]})
        assert note["note"]["ref"] == {"job": live["id"]}
    finally:
        client.script.gate.set()
    finish(client, live)


def test_the_ledgers_refusals_carry_their_codes(client):
    refused(client, "unknown_item", item="zzz", text="x")
    assert client.post(NOTES, json={"request": "dogs", "text": "x"}).status_code == 404
    refused(client, "invalid_request", expect=400, item="a", text="   ")
    bad_id = client.post(NOTES, json={"request": "../x", "text": "x"})
    assert bad_id.status_code == 400, bad_id.text
    write_plan(client, revision=1, items=[item("b")])
    # `a` had no jobs, so it was deleted rather than kept as dropped
    refused(client, "unknown_item", item="a", text="x")


def test_a_note_on_a_dropped_item_is_refused(client, take):
    write_plan(client, revision=1, items=[item("b")])
    refused(client, "item_dropped", item="a", text="x")


# --- editing and removing -----------------------------------------------------


def test_put_and_delete_a_note(client, take):
    job, name = take
    first = add(client, item="a", text="first", ref={"job": job, "output": name})
    note_id = first["note"]["id"]
    r = client.put(NOTES, json={"request": "cats", "note": note_id, "text": "second"})
    assert r.status_code == 200, r.text
    edited = r.json()
    assert edited["note"]["text"] == "second"
    assert edited["note"]["ref"] == first["note"]["ref"]
    assert edited["note"]["created"] == first["note"]["created"]
    assert edited["revision"] == first["revision"] + 1
    assert [n["text"] for n in notes_of(client, "a")] == ["second"]

    r = client.delete(f"{NOTES}?request=cats&note={note_id}")
    assert r.status_code == 200, r.text
    assert r.json() == {"revision": edited["revision"] + 1}
    assert notes_of(client, "a") == []
    again = add(client, item="a", text="third")["note"]
    assert again["id"] != note_id


def test_a_note_edit_cannot_change_its_ref(client, take):
    job, _ = take
    note_id = add(client, item="a", text="x", ref={"job": job})["note"]["id"]
    r = client.put(
        NOTES,
        json={"request": "cats", "note": note_id, "text": "y", "ref": {"job": job}},
    )
    assert r.status_code == 422


def test_an_unknown_note_is_404(client):
    r = client.put(NOTES, json={"request": "cats", "note": "n9", "text": "y"})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "unknown_note"
    r = client.delete(f"{NOTES}?request=cats&note=n9")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "unknown_note"
    r = client.delete(f"{NOTES}?request=dogs&note=n1")
    assert r.json()["detail"]["code"] == "unknown_request"


# --- the plan write -----------------------------------------------------------


def test_a_plan_put_does_not_change_notes(client):
    add(client, item="a", text="on a")
    add(client, text="on the request")
    current = view(client)
    body = plan(
        revision=current["revision"],
        items=[item("a", notes=[]), item("b", ["a"], notes=[])],
    )
    r = client.put(f"/api/workspaces/{WS}/requests/cats", json=body)
    assert r.status_code == 200, r.text
    assert [n["text"] for n in notes_of(client, "a")] == ["on a"]
    assert [n["text"] for n in notes_of(client)] == ["on the request"]


@pytest.mark.parametrize("field,value", [("comment", "warmer"), ("notes", [])])
def test_a_plan_put_carrying_a_note_field_is_refused(client, field, value):
    body = plan(revision=view(client)["revision"], **{field: value})
    r = client.put(f"/api/workspaces/{WS}/requests/cats", json=body)
    assert r.status_code == 422, r.text


# --- the gallery ---------------------------------------------------------------


def test_gallery_metadata_lists_linked_jobs_including_reuse(client, take, root):
    first, name = take
    client.script.reuse = os.path.join(root.outputs, name)
    second = finish(client, submit(client))["id"]
    loose = finish(client, submit(client, request=None, item_key=None))
    assert loose
    r = client.get(f"/api/gallery/{name}/metadata")
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["linked_jobs"] == [
        {"id": second, "request": "cats", "item": "a", "attempt": 2},
        {"id": first, "request": "cats", "item": "a", "attempt": 1},
    ]
    # attribution stays with the job that wrote it
    assert meta["job"]["id"] == first


def test_gallery_metadata_of_an_unlinked_file_lists_none(client):
    finish(client, submit(client, request=None, item_key=None))
    name = client.get("/api/gallery").json()["files"][0]["name"]
    meta = client.get(f"/api/gallery/{name}/metadata").json()
    assert meta["linked_jobs"] == []


def test_no_notes_route_is_marked_query_token_ok(client):
    routes = [
        r for r in client.app.routes if getattr(r, "path", "").endswith("request-notes")
    ]
    assert {m for r in routes for m in r.methods} == {"POST", "PUT", "DELETE"}
    for route in routes:
        assert not getattr(route.endpoint, "query_token_ok", False), route.path
