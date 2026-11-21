"""A planned item held for review until a deadline: what the ledger stores
(`review_until`, `review_set_at`), the bounds it enforces, and the `review`
the views derive from the item's notes and the clock."""

from datetime import datetime, timedelta, timezone

import pytest

from dw import ledger

from .test_ledger import item, plan, refused, write

T0 = datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc)


def stamp(moment):
    return moment.isoformat(timespec="seconds")


@pytest.fixture
def clock(monkeypatch):
    """The ledger's write clock, set with `clock.at(moment)`."""
    state = {"now": T0}
    monkeypatch.setattr(ledger, "_now", lambda: stamp(state["now"]))

    class Clock:
        def at(self, moment):
            state["now"] = moment

    return Clock()


def held(until, **fields):
    return plan(items=[item("a", review_until=until), item("b", ["a"])], **fields)


def stored(tmp_path, key="a"):
    request = ledger._load(str(tmp_path))["cats"]
    return next(i for i in request["items"] if i["key"] == key)


def read(tmp_path, now, jobs=()):
    request = ledger._load(str(tmp_path))["cats"]
    view = ledger.request_view(request, jobs, now=now)
    return next(i for i in view["items"] if i["key"] == "a")


def note(tmp_path, text="warmer", item_key="a"):
    ledger.add_note(
        str(tmp_path),
        "cats",
        item=item_key,
        text=text,
        refs=[],
        check_ref=lambda request_id, ref: ref,
    )


def hold(tmp_path, clock, seconds=3600):
    clock.at(T0)
    until = stamp(T0 + timedelta(seconds=seconds))
    write(tmp_path, held(until))
    return until


def test_a_future_hold_stamps_the_write_time_and_reads_as_holding(tmp_path, clock):
    until = hold(tmp_path, clock)
    assert stored(tmp_path)["review_until"] == until
    assert stored(tmp_path)["review_set_at"] == stamp(T0)
    got = read(tmp_path, T0 + timedelta(seconds=600))
    assert got["review"] == {"until": until, "state": "holding", "seconds_left": 3000}


def test_an_offset_input_is_stored_in_utc(tmp_path, clock):
    zone = timezone(timedelta(hours=-5))
    local = (T0 + timedelta(hours=1)).astimezone(zone)
    write(tmp_path, held(local.isoformat()))
    assert stored(tmp_path)["review_until"] == stamp(T0 + timedelta(hours=1))


def test_an_unchanged_echo_keeps_review_set_at(tmp_path, clock):
    until = hold(tmp_path, clock)
    clock.at(T0 + timedelta(minutes=10))
    write(tmp_path, held(until, revision=1))
    assert stored(tmp_path)["review_until"] == until
    assert stored(tmp_path)["review_set_at"] == stamp(T0)


def test_a_changed_value_restamps_review_set_at(tmp_path, clock):
    hold(tmp_path, clock)
    clock.at(T0 + timedelta(minutes=10))
    later = stamp(T0 + timedelta(hours=2))
    write(tmp_path, held(later, revision=1))
    assert stored(tmp_path)["review_until"] == later
    assert stored(tmp_path)["review_set_at"] == stamp(T0 + timedelta(minutes=10))


@pytest.mark.parametrize(
    "value, code",
    [
        (5, "invalid_review_until"),
        (True, "invalid_review_until"),
        ("tomorrow", "invalid_review_until"),
        ("2026-11-21T11:00:00", "invalid_review_until"),
        (stamp(T0 + timedelta(seconds=59)), "review_too_soon"),
        (stamp(T0 + timedelta(hours=24, seconds=1)), "review_too_far"),
    ],
)
def test_a_bad_deadline_is_refused(tmp_path, clock, value, code):
    error = refused(tmp_path, held(value), code=code)
    assert error.status == 422
    assert "'a'" in str(error)


def test_the_bounds_are_inclusive(tmp_path, clock):
    write(tmp_path, held(stamp(T0 + timedelta(seconds=60))))
    write(tmp_path, held(stamp(T0 + timedelta(hours=24)), revision=1))
    assert stored(tmp_path)["review_until"] == stamp(T0 + timedelta(hours=24))


def test_null_clears_both_fields_and_the_review(tmp_path, clock):
    hold(tmp_path, clock)
    write(tmp_path, held(None, revision=1))
    assert stored(tmp_path)["review_until"] is None
    assert stored(tmp_path)["review_set_at"] is None
    assert "review" not in read(tmp_path, T0)


def test_an_item_note_after_the_hold_redirects(tmp_path, clock):
    hold(tmp_path, clock)
    clock.at(T0 + timedelta(minutes=5))
    note(tmp_path)
    assert read(tmp_path, T0 + timedelta(minutes=6))["review"]["state"] == "redirected"


def test_a_request_note_after_the_hold_redirects(tmp_path, clock):
    hold(tmp_path, clock)
    clock.at(T0 + timedelta(minutes=5))
    note(tmp_path, item_key=None)
    assert read(tmp_path, T0 + timedelta(minutes=6))["review"]["state"] == "redirected"


def test_a_note_from_before_the_hold_does_not_redirect(tmp_path, clock):
    clock.at(T0)
    write(tmp_path, plan())
    note(tmp_path)
    note(tmp_path, item_key=None)
    clock.at(T0 + timedelta(minutes=5))
    until = stamp(T0 + timedelta(hours=1))
    write(tmp_path, held(until, revision=3))
    assert read(tmp_path, T0 + timedelta(minutes=6))["review"]["state"] == "holding"


def test_deleting_the_redirecting_note_returns_to_holding(tmp_path, clock):
    hold(tmp_path, clock)
    clock.at(T0 + timedelta(minutes=5))
    note(tmp_path)
    ledger.delete_note(str(tmp_path), "cats", "n1")
    assert read(tmp_path, T0 + timedelta(minutes=6))["review"]["state"] == "holding"


def test_the_hold_is_due_at_and_after_the_deadline(tmp_path, clock):
    until = hold(tmp_path, clock)
    at = read(tmp_path, T0 + timedelta(hours=1))["review"]
    assert at == {"until": until, "state": "due", "seconds_left": 0}
    after = read(tmp_path, T0 + timedelta(hours=3))["review"]
    assert after["state"] == "due"
    assert after["seconds_left"] == 0


def test_a_queued_job_ends_the_review(tmp_path, clock):
    hold(tmp_path, clock)
    jobs = [{"job": "j1", "item": "a", "status": "queued"}]
    assert "review" not in read(tmp_path, T0, jobs)


def test_a_dropped_item_has_no_review(tmp_path, clock):
    hold(tmp_path, clock)
    jobs = [{"job": "j1", "item": "a", "status": "failed"}]
    clock.at(T0 + timedelta(minutes=1))
    write(tmp_path, plan(revision=1, items=[item("b")]), linked_jobs=lambda r, k: jobs)
    request = ledger._load(str(tmp_path))["cats"]
    dropped = next(i for i in request["items"] if i["key"] == "a")
    assert dropped["dropped"] is True and dropped["review_until"]
    assert "review" not in read(tmp_path, T0, jobs)


def test_a_done_request_has_no_review(tmp_path, clock):
    hold(tmp_path, clock)
    clock.at(T0 + timedelta(minutes=1))
    write(tmp_path, held(stamp(T0 + timedelta(hours=1)), revision=1, state="done"))
    assert "review" not in read(tmp_path, T0)


def test_an_echoed_past_deadline_stays_redirected(tmp_path, clock):
    """An agent answers a note and echoes the old, now past deadline: the
    item must not become runnable without review."""
    until = hold(tmp_path, clock)
    clock.at(T0 + timedelta(minutes=5))
    note(tmp_path)
    clock.at(T0 + timedelta(hours=2))
    write(tmp_path, held(until, revision=2))
    assert stored(tmp_path)["review_set_at"] == stamp(T0)
    got = read(tmp_path, T0 + timedelta(hours=2))
    assert got["review"]["state"] == "redirected"


def test_a_plan_written_back_unchanged_by_the_user_keeps_the_hold(tmp_path, clock):
    hold(tmp_path, clock)
    view = ledger.get_request(str(tmp_path), "cats")
    clock.at(T0 + timedelta(minutes=30))
    body = {key: view[key] for key in ("title", "goal", "items", "revision")} | {
        "why": "reordered"
    }
    write(tmp_path, body, by="user")
    assert stored(tmp_path)["review_until"] == stamp(T0 + timedelta(hours=1))
    assert stored(tmp_path)["review_set_at"] == stamp(T0)


def test_a_client_sent_review_set_at_and_review_are_ignored(tmp_path, clock):
    until = stamp(T0 + timedelta(hours=1))
    body = plan(
        items=[
            item(
                "a",
                review_until=until,
                review_set_at="2020-01-01T00:00:00+00:00",
                review={"until": until, "state": "due", "seconds_left": 0},
            )
        ]
    )
    write(tmp_path, body)
    assert stored(tmp_path)["review_set_at"] == stamp(T0)
    assert "review" not in stored(tmp_path)


def test_a_hold_without_review_set_at_is_not_redirected(tmp_path, clock):
    hold(tmp_path, clock)
    clock.at(T0 + timedelta(minutes=5))
    note(tmp_path)
    request = ledger._load(str(tmp_path))["cats"]
    del request["items"][0]["review_set_at"]
    got = ledger.request_view(request, now=T0 + timedelta(minutes=6))["items"][0]
    assert got["review"]["state"] == "holding"


def test_a_request_stored_before_the_hold_still_reads(tmp_path, clock):
    from dw.server.api_models_library import LedgerRequest

    write(tmp_path, plan())
    request = ledger._load(str(tmp_path))["cats"]
    for stored_item in request["items"]:
        stored_item.pop("review_until", None)
        stored_item.pop("review_set_at", None)
    view = ledger.request_view(request, now=T0)
    got = LedgerRequest.model_validate(view)
    assert got.items[0].review_until is None
    assert got.items[0].review_set_at is None
    assert "review" not in view["items"][0]
