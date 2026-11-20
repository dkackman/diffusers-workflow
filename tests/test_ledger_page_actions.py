"""What a Requests page does through the routes a user's browser reaches:
close and reopen, each a whole-plan `PUT` written back as read."""

from . import test_ledger_job_link as link
from .test_ledger_job_link import WS, finish, submit, view

root, client, planned = link.root, link.client, link.planned

URL = f"/api/workspaces/{WS}/requests"


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
