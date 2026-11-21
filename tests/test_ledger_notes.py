"""The ledger stores a list of user notes per item and per request, whole-plan
writes keep them, and the request records when the agent last wrote."""

import json

from dw import ledger
from dw.workspace import REQUESTS_FILE_NAME

from .test_ledger import item, plan, refused, write

T = "2026-01-01T10:00:00+00:00"
T1 = "2026-01-01T11:00:00+00:00"
T2 = "2026-01-01T12:00:00+00:00"


def store(tmp_path, request):
    """Write a requests.json by hand, as an older or newer build left it."""
    base = {
        "id": "cats",
        "title": "A cat short",
        "state": "open",
        "revision": 1,
        "prompt": None,
        "goal": "g",
        "items": [{"key": "a", "title": "shot a", "depends_on": []}],
        "history": [{"revision": 1, "at": T, "by": "agent", "why": "first"}],
    }
    base.update(request)
    (tmp_path / REQUESTS_FILE_NAME).write_text(json.dumps({"requests": {"cats": base}}))


def test_a_stored_comment_is_read_as_one_note(tmp_path):
    comment = {"text": "warmer", "at": T}
    store(
        tmp_path,
        {
            "items": [
                {"key": "a", "title": "shot a", "depends_on": [], "comment": comment}
            ]
        },
    )
    got = ledger.get_request(str(tmp_path), "cats")
    assert got["items"][0]["notes"] == [
        {"id": "n1", "text": "warmer", "created": T, "at": T}
    ]
    assert "comment" not in got["items"][0]
    assert "comment" not in got
    assert got["notes"] == []
    assert got["next_note"] == 2


def test_the_request_comment_is_n1_and_items_follow(tmp_path):
    store(
        tmp_path,
        {
            "comment": {"text": "all", "at": T},
            "items": [
                {
                    "key": "a",
                    "title": "a",
                    "depends_on": [],
                    "comment": {"text": "one", "at": T1},
                }
            ],
        },
    )
    got = ledger.get_request(str(tmp_path), "cats")
    assert [n["id"] for n in got["notes"]] == ["n1"]
    assert got["items"][0]["notes"][0]["id"] == "n2"
    assert got["next_note"] == 3


def test_a_ledger_with_no_comment_reads_empty_notes(tmp_path):
    store(tmp_path, {})
    got = ledger.get_request(str(tmp_path), "cats")
    assert got["notes"] == [] and got["items"][0]["notes"] == []
    assert got["next_note"] == 1


def test_the_upgrade_is_idempotent():
    request = {
        "comment": {"text": "x", "at": T},
        "items": [],
        "history": [],
    }
    once = ledger._upgrade(request)
    snapshot = json.dumps(once, sort_keys=True)
    assert json.dumps(ledger._upgrade(once), sort_keys=True) == snapshot


def test_agent_wrote_at_is_filled_from_history_on_read(tmp_path):
    store(
        tmp_path,
        {
            "history": [
                {"revision": 1, "at": T, "by": "agent", "why": "a"},
                {"revision": 2, "at": T1, "by": "agent", "why": "b"},
                {"revision": 3, "at": T2, "by": "user", "why": "c"},
            ]
        },
    )
    assert ledger.get_request(str(tmp_path), "cats")["agent_wrote_at"] == T1


def test_an_agent_write_sets_agent_wrote_at(tmp_path):
    view = write(tmp_path, plan(), by="agent")
    assert view["agent_wrote_at"] == view["history"][-1]["at"]


def test_a_user_write_leaves_agent_wrote_at(tmp_path):
    first = write(tmp_path, plan(), by="agent")
    second = write(tmp_path, plan(revision=1), by="user")
    assert second["agent_wrote_at"] == first["agent_wrote_at"]
    created = write(tmp_path, plan(), request_id="dogs", by="user")
    assert created["agent_wrote_at"] is None


def test_a_plan_write_keeps_stored_notes_whatever_it_sends(tmp_path):
    note = {"id": "n1", "text": "warmer", "created": T, "at": T}
    store(
        tmp_path,
        {
            "notes": [{**note, "id": "n2", "text": "all"}],
            "next_note": 3,
            "items": [{"key": "a", "title": "a", "depends_on": [], "notes": [note]}],
        },
    )
    for revision, by in ((1, "user"), (2, "agent")):
        view = write(
            tmp_path,
            plan(revision=revision, items=[item("a", notes=[])], notes=[]),
            by=by,
        )
        assert view["items"][0]["notes"] == [note]
        assert [n["id"] for n in view["notes"]] == ["n2"]
        assert view["next_note"] == 3


def test_a_plan_write_carrying_comment_is_refused(tmp_path):
    refused(tmp_path, plan(comment="mine"), by="user")
    refused(tmp_path, plan(items=[item("a", comment="mine")]), by="user")


def test_a_stored_request_with_no_history_reads_with_no_agent_wrote_at(tmp_path):
    store(tmp_path, {})
    data = json.loads((tmp_path / REQUESTS_FILE_NAME).read_text())
    del data["requests"]["cats"]["history"]
    (tmp_path / REQUESTS_FILE_NAME).write_text(json.dumps(data))
    assert ledger.get_request(str(tmp_path), "cats")["agent_wrote_at"] is None
    assert "cats" in ledger.read_requests(str(tmp_path))
