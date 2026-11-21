"""What a Requests page does through the routes a user's browser reaches
(#1485, #1486): close and reopen, reorder, drop and un-drop, switch or
clear the accepted take, and queue an item's stored workflow - each one a
whole-plan `PUT` written back as read, or a `POST /api/jobs`."""

import json
import os

import pytest

from dw.workspace import REQUESTS_FILE_NAME

from . import test_ledger_job_link as link
from .test_ledger import item
from .test_ledger_job_link import WS, finish, items_of, submit, view
from .test_server import valid_workflow

root, client, planned = link.root, link.client, link.planned

URL = f"/api/workspaces/{WS}/requests"
NOTED = "2026-01-01T00:00:00+00:00"


def put(client, found, why, expect=200, **changes):
    """A user's write: the plan as read, with `changes`."""
    body = {
        "title": found["title"],
        "goal": found["goal"],
        "items": found["items"],
        "revision": found["revision"],
        "why": why,
        **changes,
    }
    r = client.put(f"{URL}/{found['id']}", json=body)
    assert r.status_code == expect, r.text
    return r.json()


# --- close and reopen ------------------------------------------------------


class TestCloseAndReopen:
    def test_a_user_marks_done_abandons_and_reopens(self, planned):
        job = finish(planned, submit(planned))
        found = put(planned, view(planned), "that is the film", state="done")
        assert found["state"] == "done"
        assert found["history"][-1] == {
            "revision": 2,
            "at": found["history"][-1]["at"],
            "by": "user",
            "why": "that is the film",
            "state": "done",
        }
        # closing changes nothing else: the plan, the attempts, the result
        assert found["items"][0]["result"]["job"] == job["id"]
        assert [i["status"] for i in found["items"]] == ["done", "planned"]
        # one closed state to the other, without reopening first
        found = put(planned, found, "never mind", state="abandoned")
        assert found["state"] == "abandoned"
        found = put(planned, found, "one more shot", state="open")
        assert found["state"] == "open"
        assert [h.get("state") for h in found["history"]] == [
            "open",
            "done",
            "abandoned",
            "open",
        ]
        submit(planned, item_key="b")

    def test_a_write_without_a_state_keeps_it(self, planned):
        found = put(planned, view(planned), "done", state="done")
        found = put(planned, found, "retitled", title="Cats, finished")
        assert found["state"] == "done"
        assert "state" not in found["history"][-1]

    def test_a_closed_request_takes_no_job_until_it_is_reopened(self, planned):
        found = put(planned, view(planned), "done", state="done")
        refused = submit(planned, expect=422)
        assert refused["detail"]["code"] == "request_closed"
        put(planned, found, "again", state="open")
        submit(planned)

    def test_the_list_shows_the_state(self, planned):
        put(planned, view(planned), "done", state="done")
        row = planned.get(URL).json()["requests"][0]
        assert row["state"] == "done" and row["ready"] == 0

    def test_reopening_at_the_open_cap_is_refused_by_name(self, planned, monkeypatch):
        from dw import ledger

        found = put(planned, view(planned), "done", state="done")
        monkeypatch.setattr(ledger, "MAX_REQUESTS", 0)
        r = put(planned, found, "again", expect=409, state="open")
        assert r["detail"]["code"] == "ledger_full"
        assert view(planned)["state"] == "done"


# --- the plan's own edits ---------------------------------------------------


def three(client):
    """A request of three items, each with a finished job, one accepted."""
    link.write_plan(
        client,
        items=[
            item("a", workflow="my-shot", arguments={"prompt": "a cat"}),
            item("b", ["a"], estimate_minutes=4),
            item("c"),
        ],
    )
    jobs = {key: finish(client, submit(client, item_key=key)) for key in "abc"}
    found = view(client)
    found["items"][0]["accepted"] = {"job": jobs["a"]["id"]}
    put(client, found, "kept a")
    # a user's note is stored by the notes routes; seed one in the file
    stored = os.path.join(client.root.root, REQUESTS_FILE_NAME)
    doc = json.loads(open(stored).read())
    doc["requests"]["cats"]["items"][2]["notes"] = [
        {"id": "n1", "text": "the best one", "created": NOTED, "at": NOTED}
    ]
    doc["requests"]["cats"]["next_note"] = 2
    with open(stored, "w") as f:
        json.dump(doc, f)
    return view(client), jobs


class TestReorder:
    def test_reordering_keeps_every_items_fields_and_links(self, client):
        before, _ = three(client)
        by_key = {i["key"]: i for i in before["items"]}
        moved = [by_key[k] for k in ("c", "a", "b")]
        after = put(client, {**before, "items": moved}, "c first")
        assert [i["key"] for i in after["items"]] == ["c", "a", "b"]
        assert {i["key"]: i for i in after["items"]} == by_key
        assert view(client)["items"] == after["items"]
        # and the jobs still answer the request and item filters
        for key in "abc":
            listed = client.get(f"/api/jobs?request=cats&item={key}").json()
            assert [j["item"] for j in listed["jobs"]] == [key]


class TestDropAndUndrop:
    def test_an_item_dropped_then_listed_again_is_as_it_was(self, client):
        before, jobs = three(client)
        kept = [i for i in before["items"] if i["key"] != "c"]
        dropped = put(client, {**before, "items": kept}, "no c")
        c = dropped["items"][-1]
        assert (c["key"], c["dropped"], c["status"]) == ("c", True, "dropped")
        # what the item held is still there to bring back
        assert c["notes"][0]["text"] == "the best one"
        assert c["attempts"][0]["job"] == jobs["c"]["id"]
        assert submit(client, item_key="c", expect=422)["detail"]["code"] == (
            "item_dropped"
        )
        # the plan as read, dropped item and all, writes back unchanged
        same = put(client, dropped, "nothing")
        assert same["items"] == dropped["items"]
        # un-drop: the item listed again, not dropped
        back = put(
            client,
            {**same, "items": same["items"][:-1] + [{**c, "dropped": False}]},
            "c after all",
        )
        restored = {i["key"]: i for i in back["items"]}["c"]
        assert restored == {i["key"]: i for i in before["items"]}["c"]
        submit(client, item_key="c")

    def test_an_item_with_no_job_is_gone_when_dropped(self, planned):
        found = view(planned)
        after = put(planned, {**found, "items": found["items"][:1]}, "no b")
        assert [i["key"] for i in after["items"]] == ["a"]

    def test_dropping_an_item_others_depend_on_is_refused(self, client):
        before, _ = three(client)
        r = put(client, {**before, "items": before["items"][1:]}, "no a", expect=400)
        assert "depends on 'a'" in r["detail"]["message"]


class TestAccepted:
    def test_accepted_moves_to_another_succeeded_job_and_clears(self, client):
        before, jobs = three(client)
        second = finish(client, submit(client, item_key="a"))
        found = view(client)
        a = found["items"][0]
        assert a["accepted"]["job"] == jobs["a"]["id"]
        assert a["result"]["job"] == jobs["a"]["id"]  # not the newer one
        a["accepted"] = {"job": second["id"]}
        found = put(client, found, "the second is better")
        a = found["items"][0]
        assert a["accepted"] == {"job": second["id"], "run_id": second["run_id"]}
        assert a["result"]["job"] == second["id"]
        a["accepted"] = None
        found = put(client, found, "undecided")
        a = found["items"][0]
        assert a["accepted"] is None
        assert a["result"]["job"] == second["id"]  # the latest succeeded again
        assert "accepted_missing" not in a

    def test_accepting_a_failed_or_foreign_job_is_refused(self, client):
        before, jobs = three(client)
        client.script.fail_next = True
        failed = finish(client, submit(client, item_key="a"))
        for job in (failed["id"], jobs["b"]["id"], "no-such-job"):
            found = view(client)
            found["items"][0]["accepted"] = {"job": job}
            put(client, found, "this one", expect=400)
        assert items_of(client)["a"]["accepted"]["job"] == jobs["a"]["id"]


# --- running an item's stored workflow --------------------------------------


@pytest.fixture
def runnable(client):
    """A request whose item `a` stores a workflow saved in the workspace and
    the arguments to run it with."""
    path = os.path.join(client.root.workflows, "my-shot.json")
    with open(path, "w") as file:
        json.dump(valid_workflow("my-shot"), file)
    link.write_plan(
        client,
        items=[
            item("a", workflow="my-shot", arguments={"prompt": "a cat"}),
            item("b", ["a"], workflow="my-shot", arguments={"nope": 1}),
        ],
    )
    return client


def run_body(client, key, **extra):
    """What a Run button posts: the item's stored workflow and arguments,
    as read, with the link."""
    stored = items_of(client)[key]
    return {
        "workflow_path": stored["workflow"],
        "arguments": stored["arguments"],
        "request": "cats",
        "item": key,
        **extra,
    }


def quoted(client, body):
    """The plan `POST /api/validate` answers for the run `body` would queue."""
    asked = {k: body[k] for k in ("workflow_path", "arguments")}
    answer = client.post("/api/validate?sizes=false", json=asked).json()
    assert answer["valid"], answer
    return answer["plan"]


class TestRunAnItem:
    def test_the_stored_workflow_runs_linked_through_the_bound_gate(self, runnable):
        assert items_of(runnable)["a"]["ready"] is True
        body = run_body(runnable, "a")
        plan = quoted(runnable, body)
        r = runnable.post("/api/jobs", json={**body, "acknowledged_cost": plan})
        assert r.status_code == 201, r.text
        job = r.json()
        assert job["acknowledged"] == "bound"
        assert (job["request"], job["item"]) == ("cats", "a")
        assert job["arguments"] == {"prompt": "a cat"}
        finish(runnable, job)
        a = items_of(runnable)["a"]
        assert a["status"] == "done" and a["ready"] is False
        assert a["attempts"][0]["job"] == job["id"]

    def test_an_acknowledgement_of_another_plan_is_refused(self, runnable):
        body = run_body(runnable, "a")
        plan = quoted(runnable, {**body, "arguments": {"prompt": "a dog"}})
        r = runnable.post("/api/jobs", json={**body, "acknowledged_cost": plan})
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert detail["reason"] == "fingerprint"
        # the refusal carries what to quote and resend
        again = {**body, "acknowledged_cost": detail["acknowledge"]}
        assert runnable.post("/api/jobs", json=again).status_code == 201
        assert len(items_of(runnable)["a"]["attempts"]) == 1

    def test_stored_arguments_meet_normal_admission(self, runnable):
        # Stored inert: the ledger checked their shape, admission checks
        # them against the workflow, and nothing is queued or linked
        r = runnable.post("/api/jobs", json=run_body(runnable, "b"))
        assert r.status_code == 400, r.text
        assert "nope" in r.json()["detail"]
        assert str(runnable.root.root) not in r.text
        assert items_of(runnable)["b"]["attempts"] == []

    def test_a_stored_workflow_that_is_not_there_is_refused(self, runnable):
        os.remove(os.path.join(runnable.root.workflows, "my-shot.json"))
        r = runnable.post("/api/jobs", json=run_body(runnable, "a"))
        assert r.status_code in (400, 404), r.text
        assert str(runnable.root.root) not in r.text
        assert runnable.manager.jobs == {}

    def test_a_closed_request_refuses_the_run_by_code(self, runnable):
        body = run_body(runnable, "a")
        plan = quoted(runnable, body)
        put(runnable, view(runnable), "done", state="done")
        r = runnable.post("/api/jobs", json={**body, "acknowledged_cost": plan})
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        assert detail["code"] == "request_closed"
        assert "done" in detail["message"] and "open" in detail["message"]
        assert runnable.manager.jobs == {}

    def test_a_dropped_item_refuses_the_run_by_code(self, runnable):
        body = run_body(runnable, "a")
        plan = quoted(runnable, body)
        finish(runnable, submit(runnable))
        found = view(runnable)
        put(runnable, {**found, "items": []}, "start over")
        assert items_of(runnable)["a"]["dropped"] is True
        r = runnable.post("/api/jobs", json={**body, "acknowledged_cost": plan})
        assert r.status_code == 422, r.text
        assert r.json()["detail"]["code"] == "item_dropped"
        assert "dropped" in r.json()["detail"]["message"]
