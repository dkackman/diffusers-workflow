"""The ledger stores a list of user notes per item and per request, whole-plan
writes keep them, and the request records when the agent last wrote."""

import json

import pytest

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


# --- add, edit and delete one note ---------------------------------------


def _checked(rid, ref):
    return {**ref, "kind": "video"} if "output" in ref else ref


def planned(tmp_path):
    write(tmp_path, plan())
    return str(tmp_path)


def add(root, text="warmer", item="a", refs=None, check=_checked):
    return ledger.add_note(
        root, "cats", item=item, text=text, refs=refs, check_ref=check
    )


def test_add_note_on_an_item_with_a_ref(tmp_path):
    root = planned(tmp_path)
    out = add(root, "lipsync off", refs=[{"job": "j1", "output": "w/r1/final/a.mp4"}])
    assert out["note"]["id"] == "n1"
    assert out["note"]["refs"] == [
        {"item": "a", "job": "j1", "output": "w/r1/final/a.mp4", "kind": "video"}
    ]
    assert out["revision"] == 2
    got = ledger.get_request(root, "cats")
    assert got["history"][-1]["by"] == "user"
    assert got["history"][-1]["why"] == "note added"
    assert got["items"][0]["notes"][0]["refs"][0]["job"] == "j1"


def test_a_request_note_refs_span_items_planned_or_rendered(tmp_path):
    root = planned(tmp_path)
    refs = [{"item": "a", "job": "j1", "output": "w/r1/final/a.mp4"}, {"item": "b"}]
    note = add(root, "takes a and b: warmer", item=None, refs=refs)["note"]
    assert note["refs"] == [
        {"item": "a", "job": "j1", "output": "w/r1/final/a.mp4", "kind": "video"},
        {"item": "b"},
    ]
    # a planned shot with no job: no attempt, not gone
    assert ledger.get_request(root, "cats")["notes"][0]["refs"][1] == {"item": "b"}


def test_an_empty_refs_list_stores_no_refs(tmp_path):
    assert "refs" not in add(planned(tmp_path), refs=[])["note"]


def test_an_item_note_ref_to_another_item_is_refused(tmp_path):
    root = planned(tmp_path)
    refused_note("invalid_note_ref", add, root, refs=[{"item": "b"}])
    assert add(root, refs=[{"item": "a"}])["note"]["refs"] == [{"item": "a"}]


def test_a_request_note_ref_needs_an_item(tmp_path):
    refused_note("invalid_note_ref", add, planned(tmp_path), item=None, refs=[{}])


def test_a_ref_to_an_item_not_in_the_plan_is_refused(tmp_path):
    refused_note(
        "invalid_note_ref", add, planned(tmp_path), item=None, refs=[{"item": "zz"}]
    )


def test_seventeen_refs_are_refused(tmp_path):
    root = planned(tmp_path)
    refs = [{"job": f"j{k}"} for k in range(ledger.MAX_NOTE_REFS + 1)]
    e = refused_note("too_many_refs", add, root, refs=refs)
    assert e.status == 422
    assert ledger.get_request(root, "cats")["items"][0]["notes"] == []
    add(root, refs=refs[: ledger.MAX_NOTE_REFS])


def test_duplicate_refs_collapse_keeping_first_order(tmp_path):
    root = planned(tmp_path)
    refs = [
        {"item": "b"},
        {"item": "a", "job": "j1"},
        {"item": "b"},
        {"item": "a", "job": "j1"},
        {"item": "a", "job": "j1", "output": "o.mp4"},
    ]
    note = add(root, item=None, refs=refs)["note"]
    assert note["refs"] == [
        {"item": "b"},
        {"item": "a", "job": "j1"},
        {"item": "a", "job": "j1", "output": "o.mp4", "kind": "video"},
    ]


def test_an_item_note_ref_defaults_its_item_before_dedupe(tmp_path):
    seen = []

    def spy(rid, ref):
        seen.append(ref)
        return ref

    refs = [{"job": "j1"}, {"item": "a", "job": "j1"}]
    note = add(planned(tmp_path), refs=refs, check=spy)["note"]
    assert note["refs"] == [{"item": "a", "job": "j1"}]
    assert seen == [{"item": "a", "job": "j1"}]


def test_an_output_without_a_job_is_refused(tmp_path):
    refused_note(
        "invalid_note_ref", add, planned(tmp_path), refs=[{"output": "w/r1/a.mp4"}]
    )


def test_an_item_only_ref_to_a_dropped_item_is_refused(tmp_path):
    root = planned(tmp_path)
    write(
        tmp_path,
        plan(revision=1, items=[item("a")]),
        linked_jobs=lambda rid, key: [{"job": "j1"}] if key == "b" else [],
    )
    refused_note("invalid_note_ref", add, root, item=None, refs=[{"item": "b"}])


def test_a_ref_whose_item_is_dropped_later_is_gone(tmp_path):
    root = planned(tmp_path)
    add(root, "both", item=None, refs=[{"item": "a"}, {"item": "b"}])
    before = ledger.get_request(root, "cats")["notes"][0]["refs"]
    assert before == [{"item": "a"}, {"item": "b"}]
    write(
        tmp_path,
        plan(revision=2, items=[item("a")]),
        linked_jobs=lambda rid, key: [{"job": "j1"}] if key == "b" else [],
    )
    after = ledger.get_request(root, "cats")["notes"][0]["refs"]
    assert after == [{"item": "a"}, {"item": "b", "gone": True}]


def test_add_note_without_item_goes_on_the_request(tmp_path):
    root = planned(tmp_path)
    out = add(root, "all colder", item=None)
    got = ledger.get_request(root, "cats")
    assert got["notes"] == [out["note"]]
    assert got["items"][0]["notes"] == []


def test_note_ids_are_never_reused_after_delete(tmp_path):
    root = planned(tmp_path)
    assert add(root)["note"]["id"] == "n1"
    out = ledger.delete_note(root, "cats", "n1")
    assert out["revision"] == 3
    assert ledger.get_request(root, "cats")["history"][-1]["why"] == "note removed"
    assert add(root)["note"]["id"] == "n2"


def test_add_note_sets_created_equal_to_at(tmp_path):
    note = add(planned(tmp_path))["note"]
    assert note["created"] == note["at"]
    assert "refs" not in note


def test_edit_note_with_same_text_keeps_at(tmp_path):
    root = planned(tmp_path)
    note = add(root)["note"]
    out = ledger.edit_note(root, "cats", "n1", "warmer")
    assert out == {"note": note, "revision": 2}
    assert len(ledger.get_request(root, "cats")["history"]) == 2


def test_edit_note_with_new_text_gets_new_at(tmp_path):
    root = planned(tmp_path)
    note = add(root, refs=[{"job": "j1"}])["note"]
    path = tmp_path / REQUESTS_FILE_NAME
    stored = json.loads(path.read_text())
    stored["requests"]["cats"]["items"][0]["notes"][0]["at"] = T
    stored["requests"]["cats"]["items"][0]["notes"][0]["created"] = T
    path.write_text(json.dumps(stored))
    out = ledger.edit_note(root, "cats", "n1", "colder")
    assert out["revision"] == 3
    assert out["note"]["text"] == "colder"
    assert out["note"]["created"] == T and out["note"]["at"] != T
    assert out["note"]["refs"] == note["refs"]
    assert ledger.get_request(root, "cats")["history"][-1]["why"] == "note edited"


def test_edit_and_delete_find_a_request_note(tmp_path):
    root = planned(tmp_path)
    add(root, item=None)
    assert ledger.edit_note(root, "cats", "n1", "x")["note"]["text"] == "x"
    ledger.delete_note(root, "cats", "n1")
    assert ledger.get_request(root, "cats")["notes"] == []


def refused_note(code, call, *args, **kw):
    with pytest.raises(ledger.LedgerError) as e:
        call(*args, **kw)
    assert e.value.code == code, str(e.value)
    return e.value


def test_the_21st_note_on_a_target_is_refused(tmp_path):
    root = planned(tmp_path)
    for k in range(ledger.MAX_NOTES):
        add(root, f"n {k}")
    refused_note("too_many_notes", add, root)
    add(root, item="b")  # another target has its own room


def test_a_note_on_a_dropped_item_is_refused(tmp_path):
    root = planned(tmp_path)
    # a job links to b, so leaving it out drops it rather than deleting it
    write(
        tmp_path,
        plan(revision=1, items=[item("a")]),
        linked_jobs=lambda rid, key: [{"job": "j1"}] if key == "b" else [],
    )
    refused_note("item_dropped", add, root, item="b")


def test_unknown_request_item_and_note(tmp_path):
    root = planned(tmp_path)
    e = refused_note(
        "unknown_request",
        ledger.add_note,
        root,
        "nope",
        item=None,
        text="x",
        refs=None,
        check_ref=_checked,
    )
    assert e.status == 404
    assert refused_note("unknown_item", add, root, item="zz").status == 422
    assert (
        refused_note("unknown_note", ledger.edit_note, root, "cats", "n9", "x").status
        == 404
    )
    refused_note("unknown_note", ledger.delete_note, root, "cats", "n9")
    refused_note("unknown_request", ledger.delete_note, root, "nope", "n1")


def test_blank_or_overlong_text_is_refused(tmp_path):
    root = planned(tmp_path)
    refused_note("invalid_request", add, root, "   ")
    refused_note("invalid_request", add, root, "x" * 4001)
    add(root)
    refused_note("invalid_request", ledger.edit_note, root, "cats", "n1", "")


def test_a_failed_check_ref_stores_nothing(tmp_path):
    root = planned(tmp_path)

    def bad(rid, ref):
        if ref.get("job") == "j9":
            raise ledger.LedgerError("no such job", status=422, code="invalid_note_ref")
        return ref

    refs = [{"job": "j1"}, {"job": "j9"}]
    refused_note("invalid_note_ref", add, root, refs=refs, check=bad)
    got = ledger.get_request(root, "cats")
    assert got["revision"] == 1 and got["items"][0]["notes"] == []
    assert got["next_note"] == 1


def test_a_note_on_a_closed_request_is_allowed(tmp_path):
    root = planned(tmp_path)
    write(tmp_path, plan(revision=1, state="done"))
    assert add(root)["revision"] == 3


def test_check_ref_gets_the_request_and_each_ref(tmp_path):
    root = planned(tmp_path)
    seen = []

    def spy(rid, ref):
        seen.append((rid, ref))
        return ref

    add(root, item=None, refs=[{"item": "a", "job": "j1"}, {"item": "b"}], check=spy)
    assert seen == [("cats", {"item": "a", "job": "j1"}), ("cats", {"item": "b"})]


def test_view_marks_ref_attempt_and_gone(tmp_path):
    root = planned(tmp_path)
    for job in ("j1", "j2", "j9"):
        add(root, job, refs=[{"job": job}])
    add(root, "req-level", item=None, refs=[{"item": "a", "job": "j2"}])
    add(root, "req-gone", item=None, refs=[{"item": "a", "job": "j9"}])
    add(root, "req-moved", item=None, refs=[{"item": "b", "job": "j2"}])
    jobs = [
        {
            "job": "j1",
            "item": "a",
            "status": "succeeded",
            "run_exists": False,
            "created_at": 1,
        },
        {
            "job": "j2",
            "item": "a",
            "status": "succeeded",
            "run_exists": True,
            "created_at": 2,
        },
    ]
    got = ledger.get_request(root, "cats", jobs_for=lambda rid: jobs)
    refs = {n["text"]: n["refs"][0] for n in got["items"][0]["notes"]}
    assert refs["j2"]["attempt"] == 2 and "gone" not in refs["j2"]
    assert refs["j1"]["attempt"] == 1 and refs["j1"]["gone"] is True
    assert refs["j9"]["attempt"] is None and refs["j9"]["gone"] is True
    top = {n["text"]: n["refs"][0] for n in got["notes"]}
    assert top["req-level"]["attempt"] == 2 and "gone" not in top["req-level"]
    assert top["req-gone"]["attempt"] is None and top["req-gone"]["gone"] is True
    # j2 is linked to a, so a ref naming it under b no longer holds
    assert top["req-moved"]["gone"] is True
    stored = json.loads((tmp_path / REQUESTS_FILE_NAME).read_text())["requests"]["cats"]
    assert "attempt" not in stored["items"][0]["notes"][0]["refs"][0]
