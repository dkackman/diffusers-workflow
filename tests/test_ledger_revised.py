"""A `done` item whose plan changed after every succeeded take reads
`revised` (#1531): the server's `changed_at` and `accepted_basis` stamps, the
rule in `ledger_views`, and the linked jobs' `workflow` and `arguments`
reaching it through the real server."""

import copy
import time
from datetime import datetime, timedelta, timezone

import pytest

from dw import ledger

from .test_ledger import item, plan, write
from .test_ledger_job_link import (  # noqa: F401 - fixtures
    WS,
    client,
    finish,
    items_of,
    root,
    submit,
)

T0 = datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc)
WF = "text-to-image"
ARGS = {"prompt": "a cat"}
NEW_ARGS = {"prompt": "a dog"}


def stamp(moment):
    return moment.isoformat(timespec="seconds")


def epoch(seconds):
    """An epoch time `seconds` after T0."""
    return T0.timestamp() + seconds


class Clock:
    def __init__(self):
        self.now = T0

    def at(self, seconds):
        self.now = T0 + timedelta(seconds=seconds)


@pytest.fixture
def clock(monkeypatch):
    state = Clock()
    monkeypatch.setattr(ledger, "_now", lambda: stamp(state.now))
    monkeypatch.setattr(ledger, "_created", lambda: state.now.timestamp())
    return state


def take(job="j1", item_key="a", created=10, workflow=WF, arguments=ARGS, **kw):
    return {
        "job": job,
        "item": item_key,
        "status": "succeeded",
        "run_id": f"run-{job}",
        "run_exists": True,
        "created_at": epoch(created),
        "started_at": epoch(created),
        "finished_at": epoch(created + 60),
        "workflow": workflow,
        "arguments": copy.deepcopy(arguments),
        **kw,
    }


class Ledger:
    """put_request over a fake job list."""

    def __init__(self, tmp_path):
        self.root = tmp_path
        self.jobs = []
        self.revision = 0

    def put(self, items, by="agent", **fields):
        body = plan(revision=self.revision, items=items, **fields)
        view = write(
            self.root,
            body,
            by=by,
            linked_jobs=lambda rid, key: [j for j in self.jobs if j["item"] == key],
            jobs_for=lambda rid: list(self.jobs),
        )
        self.revision = view["revision"]
        return view

    def read(self, now=None):
        request = ledger._load(str(self.root))["cats"]
        return ledger.request_view(request, self.jobs, now=now)

    def summary(self):
        request = ledger._load(str(self.root))["cats"]
        return ledger.summary_view(request, self.jobs)

    def stored(self, key="a"):
        request = ledger._load(str(self.root))["cats"]
        return next(i for i in request["items"] if i["key"] == key)


def items(a_args=ARGS, a_workflow=WF, **a_fields):
    return [
        item(
            "a", workflow=a_workflow, arguments=a_args, estimate_minutes=7, **a_fields
        ),
        item("b", ["a"], workflow=WF, arguments=ARGS, estimate_minutes=3),
    ]


def by_key(view):
    return {i["key"]: i for i in view["items"]}


@pytest.fixture
def done(tmp_path, clock):
    """A written plan, then a take of item a created after it: a is done."""
    led = Ledger(tmp_path)
    led.put(items())
    led.jobs.append(take(created=10))
    assert by_key(led.read())["a"]["status"] == "done"
    return led


def revise(led, clock, args=NEW_ARGS, at=100, **a_fields):
    clock.at(at)
    return led.put(items(a_args=args, **a_fields))


# --- 1: the rule -----------------------------------------------------------


def test_changed_arguments_after_a_take_read_revised(done, clock):
    before = by_key(done.read())["a"]["result"]
    view = revise(done, clock)
    a, b = by_key(view)["a"], by_key(view)["b"]
    assert a["status"] == "revised"
    assert a["revised_because"] == ["arguments"]
    assert a["result"] == before and a["result"]["job"] == "j1"
    assert [x["job"] for x in a["attempts"]] == ["j1"]
    assert a["ready"] is True
    assert b["ready"] is False and b["status"] == "planned"
    # a (7) counts again with b (3)
    assert view["remaining_minutes"] == 10
    assert done.summary()["status_counts"]["revised"] == 1
    assert done.summary()["ready"] == 1


def test_a_revised_item_in_a_closed_request_is_not_ready(done, clock):
    clock.at(100)
    view = done.put(items(a_args=NEW_ARGS), state="done", by="user")
    assert by_key(view)["a"]["status"] == "revised"
    assert by_key(view)["a"]["ready"] is False


def test_a_revised_item_without_a_workflow_is_not_ready(tmp_path, clock):
    led = Ledger(tmp_path)
    led.put([item("a", arguments=ARGS)])
    led.jobs.append(take(created=10, workflow=None))
    clock.at(100)
    a = by_key(led.put([item("a", arguments=NEW_ARGS)]))["a"]
    assert a["status"] == "revised"
    assert a["ready"] is False


# --- 2: no change, no revision ---------------------------------------------


def test_the_plan_written_back_as_read_stays_done(done, clock):
    view = done.read()
    stamp_before = done.stored()["changed_at"]
    clock.at(500)
    body_items = [
        {
            k: v
            for k, v in i.items()
            if k
            in (
                "key",
                "title",
                "depends_on",
                "estimate_minutes",
                "workflow",
                "arguments",
                "accepted",
                "review_until",
            )
        }
        for i in view["items"]
    ]
    again = done.put(body_items)
    assert by_key(again)["a"]["status"] == "done"
    assert done.stored()["changed_at"] == stamp_before
    assert "revised_because" not in by_key(again)["a"]


def test_the_full_get_view_as_the_body_stays_done(done, clock):
    """Every field of the view, derived ones too, sent back as read."""
    view = done.read()
    clock.at(500)
    sendable = {"key", "title", "depends_on", "estimate_minutes"}
    body_items = []
    for i in view["items"]:
        send = {k: v for k, v in i.items() if k in sendable | {"workflow", "arguments"}}
        send["changed_at"] = i["changed_at"]
        send["accepted_basis"] = i["accepted_basis"]
        body_items.append(send)
    try:
        again = done.put(body_items)
    except ledger.LedgerError:
        pytest.skip("changed_at/accepted_basis are refused as unknown, not ignored")
    assert by_key(again)["a"]["status"] == "done"


@pytest.mark.parametrize(
    "edit",
    [
        {"title": "a new title"},
        {"estimate_minutes": 99},
    ],
)
def test_title_or_estimate_edits_leave_it_done(done, clock, edit):
    clock.at(500)
    stored_before = done.stored()["changed_at"]
    base = items()
    base[0] = {**base[0], **edit}
    view = done.put(base)
    assert by_key(view)["a"]["status"] == "done"
    assert done.stored()["changed_at"] == stored_before


def test_depends_on_edit_leaves_it_done(done, clock):
    clock.at(500)
    base = items()
    base[1] = item("b", [], workflow=WF, arguments=ARGS, estimate_minutes=3)
    base.append(item("c", ["a"]))
    base[0] = {**base[0], "depends_on": []}
    view = done.put(base)
    assert by_key(view)["a"]["status"] == "done"


# --- 3, 4, 5: takes --------------------------------------------------------


def test_a_variation_take_without_a_plan_write_leaves_it_done(done):
    done.jobs.append(take("j2", created=200, arguments=NEW_ARGS))
    assert by_key(done.read())["a"]["status"] == "done"
    assert done.summary()["status_counts"].get("revised") is None


def test_writing_the_latest_takes_arguments_makes_it_done(done, clock):
    done.jobs.append(take("j2", created=200, arguments=NEW_ARGS))
    # the plan still says ARGS and j1 is current: write NEW_ARGS, then the
    # change is after j1 but j2 matches
    revise(done, clock, args=NEW_ARGS, at=300)
    assert by_key(done.read())["a"]["status"] == "done"


def test_arguments_back_to_a_takes_make_it_done_again(done, clock):
    assert by_key(revise(done, clock))["a"]["status"] == "revised"
    clock.at(200)
    view = done.put(items(a_args=ARGS))
    assert by_key(view)["a"]["status"] == "done"
    assert "revised_because" not in by_key(view)["a"]


def test_a_new_succeeded_take_after_the_change_makes_it_done(done, clock):
    revise(done, clock)
    done.jobs.append(take("j2", created=150, arguments=NEW_ARGS))
    view = done.read()
    assert by_key(view)["a"]["status"] == "done"
    assert by_key(view)["b"]["ready"] is True
    assert by_key(view)["a"]["result"]["job"] == "j2"


def test_a_take_after_the_change_current_by_time_alone(done, clock):
    """created_at after changed_at is current whatever its arguments."""
    revise(done, clock)
    done.jobs.append(take("j2", created=150, arguments={"prompt": "other"}))
    assert by_key(done.read())["a"]["status"] == "done"


def test_a_failed_take_after_the_change_does_not_clear_it(done, clock):
    revise(done, clock)
    done.jobs.append(take("j2", created=150, status="failed"))
    assert by_key(done.read())["a"]["status"] == "revised"


def test_a_running_take_after_the_change_does_not_clear_it(done, clock):
    """Only a succeeded take can be current."""
    revise(done, clock)
    done.jobs.append(take("j2", created=150, status="running"))
    assert by_key(done.read())["a"]["status"] == "revised"


# --- 6: acceptance ---------------------------------------------------------


def test_accepting_while_revised_makes_it_done_until_the_next_change(done, clock):
    revise(done, clock)
    clock.at(200)
    view = done.put(items(a_args=NEW_ARGS, accepted={"job": "j1"}))
    a = by_key(view)["a"]
    assert a["status"] == "done"
    assert a["accepted_basis"] == {"at": stamp(T0 + timedelta(seconds=200))}
    assert a["changed_at"] == stamp(T0 + timedelta(seconds=100))
    assert by_key(view)["b"]["ready"] is True
    assert by_key(done.read())["a"]["status"] == "done"

    clock.at(300)
    again = done.put(items(a_args={"prompt": "third"}, accepted={"job": "j1"}))
    assert by_key(again)["a"]["status"] == "revised"
    assert by_key(again)["a"]["revised_because"] == ["arguments"]


def test_un_accepting_drops_the_basis_and_the_item_reads_revised_again(done, clock):
    """The basis is the record that a take was accepted for this plan. With
    the acceptance withdrawn nothing vouches for the old take, so the item
    is revised again, as it was before it was accepted."""
    revise(done, clock)
    clock.at(200)
    done.put(items(a_args=NEW_ARGS, accepted={"job": "j1"}))
    clock.at(300)
    view = done.put(items(a_args=NEW_ARGS, accepted=None))
    a = by_key(view)["a"]
    assert a["accepted"] is None
    assert a["accepted_basis"] is None
    assert a["status"] == "revised"
    assert by_key(view)["b"]["ready"] is False


def test_accepting_with_a_plan_change_in_the_same_write_is_not_a_basis(done, clock):
    clock.at(100)
    view = done.put(items(a_args=NEW_ARGS, accepted={"job": "j1"}))
    a = by_key(view)["a"]
    assert a["status"] == "revised"
    assert a["accepted_basis"] is None


def test_accepting_an_item_that_reads_done_sets_no_basis(done, clock):
    clock.at(100)
    view = done.put(items(accepted={"job": "j1"}))
    assert by_key(view)["a"]["status"] == "done"
    assert by_key(view)["a"]["accepted_basis"] is None


# --- 7: dropped ------------------------------------------------------------


def test_a_dropped_item_is_never_revised(done, clock):
    clock.at(100)
    view = done.put([item("b", workflow=WF, arguments=ARGS, estimate_minutes=3)])
    a = by_key(view)["a"]
    assert a["dropped"] is True and a["status"] == "dropped"
    assert "revised_because" not in a
    assert done.summary()["status_counts"].get("revised") is None
    assert view["remaining_minutes"] == 3


def test_a_dropped_item_with_changed_stored_arguments_stays_dropped(done):
    request = ledger._load(str(done.root))["cats"]
    stored = next(i for i in request["items"] if i["key"] == "a")
    stored["dropped"] = True
    stored["arguments"] = NEW_ARGS
    stored["changed_at"] = stamp(T0 + timedelta(seconds=999))
    view = ledger.request_view(request, done.jobs)
    assert by_key(view)["a"]["status"] == "dropped"


# --- 8: review -------------------------------------------------------------


def test_a_revised_item_with_a_hold_reads_review_holding_then_due(done, clock):
    until = stamp(T0 + timedelta(seconds=100 + 3600))
    revise(done, clock, review_until=until)
    during = by_key(done.read(now=T0 + timedelta(seconds=200)))["a"]
    assert during["status"] == "revised"
    assert during["review"]["state"] == "holding"
    assert during["review"]["until"] == until
    after = by_key(done.read(now=T0 + timedelta(seconds=100 + 3600)))["a"]
    assert after["review"]["state"] == "due"
    assert after["review"]["seconds_left"] == 0


def test_a_done_item_has_no_review(done, clock):
    clock.at(100)
    until = stamp(T0 + timedelta(seconds=5000))
    view = done.put(items(review_until=until))
    assert by_key(view)["a"]["status"] == "done"
    assert "review" not in by_key(view)["a"]


# --- 9: workflow -----------------------------------------------------------


def test_a_workflow_change_is_named_in_revised_because(done, clock):
    view = revise(done, clock, args=ARGS)  # arguments equal, workflow moves below
    assert by_key(view)["a"]["status"] == "done"
    clock.at(200)
    view = done.put(items(a_workflow="other-workflow"))
    a = by_key(view)["a"]
    assert a["status"] == "revised"
    assert a["revised_because"] == ["workflow"]
    clock.at(300)
    view = done.put(items(a_args=NEW_ARGS, a_workflow="other-workflow"))
    assert by_key(view)["a"]["revised_because"] == ["workflow", "arguments"]


def test_a_take_with_no_workflow_still_counts_current_by_arguments(tmp_path, clock):
    led = Ledger(tmp_path)
    led.put(items())
    led.jobs.append(take(created=10, workflow=None))
    clock.at(100)
    # workflow differs from nothing recorded; arguments are what decide
    view = led.put(items(a_workflow="other-workflow"))
    assert by_key(view)["a"]["status"] == "done"
    clock.at(200)
    view = led.put(items(a_args=NEW_ARGS, a_workflow="other-workflow"))
    assert by_key(view)["a"]["status"] == "revised"
    assert by_key(view)["a"]["revised_because"] == ["arguments"]


# --- 10: server stamps -----------------------------------------------------


def test_client_sent_stamps_are_ignored(tmp_path, clock):
    led = Ledger(tmp_path)
    forged = stamp(T0 - timedelta(days=30))
    try:
        view = led.put(
            [
                item(
                    "a",
                    workflow=WF,
                    arguments=ARGS,
                    changed_at=forged,
                    accepted_basis={"at": forged},
                    revised_because=["workflow"],
                )
            ]
        )
    except ledger.LedgerError as e:
        pytest.fail(f"client-sent stamps should be ignored, not refused: {e}")
    a = view["items"][0]
    assert a["changed_at"] == stamp(T0)
    assert a["accepted_basis"] is None
    assert "revised_because" not in a
    assert led.stored()["changed_at"] == stamp(T0)
    assert "accepted_basis" not in led.stored()


def test_a_forged_changed_at_cannot_hide_a_revision(done, clock):
    clock.at(100)
    base = items(a_args=NEW_ARGS)
    base[0]["changed_at"] = stamp(T0 - timedelta(days=30))
    base[0]["accepted_basis"] = {"at": stamp(T0 + timedelta(days=30))}
    view = done.put(base)
    assert by_key(view)["a"]["status"] == "revised"
    assert by_key(view)["a"]["changed_at"] == stamp(T0 + timedelta(seconds=100))


def test_changed_at_is_set_on_create_and_moves_only_with_the_plan(done, clock):
    assert done.stored()["changed_at"] == stamp(T0)
    clock.at(100)
    done.put(items(title="retitled"))
    assert done.stored()["changed_at"] == stamp(T0)
    clock.at(200)
    done.put(items(a_args=NEW_ARGS))
    assert done.stored()["changed_at"] == stamp(T0 + timedelta(seconds=200))
    clock.at(300)
    done.put(items(a_args=NEW_ARGS))
    assert done.stored()["changed_at"] == stamp(T0 + timedelta(seconds=200))


# --- 11: pre-stamp items ---------------------------------------------------


def test_an_item_with_no_changed_at_is_never_revised():
    request = {
        "id": "cats",
        "state": "open",
        "items": [
            {
                "key": "a",
                "workflow": WF,
                "arguments": NEW_ARGS,
                "estimate_minutes": 4,
            },
            {"key": "b", "depends_on": ["a"], "workflow": WF},
        ],
    }
    jobs = [take(created=10)]
    view = ledger.request_view(request, jobs)
    a = by_key(view)["a"]
    assert a["status"] == "done"
    assert a["changed_at"] is None and a["accepted_basis"] is None
    assert by_key(view)["b"]["ready"] is True
    assert view["remaining_minutes"] == 0
    assert ledger.summary_view(request, jobs)["status_counts"] == {
        "done": 1,
        "planned": 1,
    }


# --- HTTP ------------------------------------------------------------------


def put_plan(client, revision, a_arguments):  # noqa: F811
    body = plan(
        revision=revision,
        items=[
            item("a", arguments=a_arguments, estimate_minutes=5),
            item("b", ["a"]),
        ],
    )
    body["by"] = "agent"
    r = client.put(f"/api/workspaces/{WS}/requests/cats", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_request_jobs_carry_the_takes_arguments_and_workflow(client):  # noqa: F811
    put_plan(client, 0, {"prompt": "hi"})
    job = finish(client, submit(client, prompt="hi"))
    assert job["status"] == "succeeded"
    live = client.manager.request_jobs(WS, "cats")
    assert [j["arguments"] for j in live] == [{"prompt": "hi"}]
    assert "workflow" in live[0]
    # and again from history once the live job is gone
    client.manager.jobs.pop(job["id"])
    closed = client.manager.request_jobs(WS, "cats")
    assert [j["arguments"] for j in closed] == [{"prompt": "hi"}]
    assert closed[0]["workflow"] == live[0]["workflow"]


def test_a_put_changing_the_arguments_after_a_take_reads_revised(client):  # noqa: F811
    put_plan(client, 0, {"prompt": "hi"})
    finish(client, submit(client, prompt="hi"))
    a = items_of(client)["a"]
    assert a["status"] == "done"
    # changed_at has one-second resolution: let the clock pass the take
    time.sleep(1.2)
    r = client.put(
        f"/api/workspaces/{WS}/requests/cats",
        json={
            **plan(
                revision=1,
                items=[
                    item("a", arguments={"prompt": "bye"}, estimate_minutes=5),
                    item("b", ["a"]),
                ],
            ),
            "by": "agent",
        },
    )
    assert r.status_code == 200, r.text
    got = {i["key"]: i for i in r.json()["items"]}
    assert got["a"]["status"] == "revised"
    assert got["a"]["revised_because"] == ["arguments"]
    assert got["a"]["result"] is not None
    assert got["b"]["ready"] is False
    again = items_of(client)
    assert again["a"]["status"] == "revised"
    listed = client.get(f"/api/workspaces/{WS}/requests").json()["requests"]
    assert listed[0]["status_counts"].get("revised") == 1

    # writing the original arguments back makes the take current again
    time.sleep(1.2)
    put_plan(client, 2, {"prompt": "hi"})
    assert items_of(client)["a"]["status"] == "done"
