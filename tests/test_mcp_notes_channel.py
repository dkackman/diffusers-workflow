"""The notes channel's rules (`dw_mcp/ledger.py`): which notes wait, what is
announced and when, and what an event says. The client here answers from
the real ledger (`dw/ledger.py`) on a temp folder, under a clock the test
moves; nothing sleeps."""

import json
import os
import re
from datetime import datetime

import pytest

from dw import ledger as store
from dw.workspace import REQUESTS_FILE_NAME
from dw.notes import MAX_NOTE_LENGTH
from dw_mcp import ledger
from dw_mcp.client import DwApiError

from .notes_helpers import save_note

WORKSPACE = "studio"


class Clock:
    def __init__(self):
        self.minute = 0

    def now(self):
        return f"2026-11-20T14:{self.minute:02d}:00+00:00"

    def tick(self, minutes=1):
        self.minute += minutes


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(store, "_now", clock.now)
    return clock


class LedgerClient:
    """What the channel needs of a DwClient, read from a ledger folder."""

    def __init__(self, root):
        self.root = str(root)
        self.workspace = WORKSPACE
        self.calls = []
        self.headers = []  # (path, headers) of every read
        self.fail = {}  # path suffix -> DwApiError to raise
        self.jobs = []  # the linked jobs every request read is given
        self.now = None  # the aware time a read derives review holds at

    def get_json(self, path, params=None, workspace=None, headers=None):
        self.calls.append(path)
        self.headers.append((path, headers))
        for suffix, error in self.fail.items():
            if path.endswith(suffix):
                raise error
        prefix = f"/api/workspaces/{WORKSPACE}/requests"
        assert path.startswith(prefix), path
        request_id = path[len(prefix) :].lstrip("/")
        if not request_id:
            return {
                "workspace": WORKSPACE,
                **store.listing(self.root, jobs_for=self.linked),
            }
        stored = store._load(self.root).get(request_id)
        if stored is None:
            raise DwApiError(f"No request {request_id!r}", status_code=404)
        return store.request_view(stored, self.linked(request_id), now=self.now)

    def linked(self, request_id):
        return self.jobs

    def reads(self):
        """The requests read one by one since the last ask."""
        prefix = f"/api/workspaces/{WORKSPACE}/requests/"
        read = [p[len(prefix) :] for p in self.calls if p.startswith(prefix)]
        self.calls.clear()
        return read


@pytest.fixture
def client(tmp_path, clock):
    return LedgerClient(tmp_path)


def items(*keys):
    return [{"key": key, "title": f"Shot {key}", "depends_on": []} for key in keys]


def write(client, request_id="lighthouse", by="agent", why="w", **fields):
    """One whole-plan write as `by`, at the request's current revision."""
    current = store.get_request(client.root, request_id)
    body = {
        "title": "Lighthouse short",
        "goal": "three shots",
        "items": current["items"] if current else items("shot1", "shot2", "shot3"),
        "revision": current["revision"] if current else 0,
        "why": why,
    }
    body.update(fields)
    return store.put_request(client.root, request_id, body, by=by)


def note(client, text, item=None, request_id="lighthouse", refs=None):
    """The user saves `text` on an item, or on the request itself (None
    clears it)."""
    why = "note" if item is None else f"note on {item}"
    write(client, request_id, by="user", why=why)
    save_note(client.root, request_id, text, item, refs=refs)


def started(client):
    """A watch whose startup poll is behind it."""
    watch = ledger.NoteWatch()
    assert watch.poll(client) == []
    client.calls.clear()
    return watch


# --- what is announced ------------------------------------------------------


def test_an_idle_ledger_costs_one_list_call_per_poll(client):
    write(client)
    watch = ledger.NoteWatch()
    assert watch.poll(client) == []
    assert client.reads() == ["lighthouse"]
    assert watch.poll(client) == []
    assert client.calls == [f"/api/workspaces/{WORKSPACE}/requests"]


def test_a_note_saved_while_listening_is_one_event(client, clock):
    write(client)
    watch = started(client)
    clock.tick(2)
    note(client, "off lipsync - reroll it and recreate the final piece", "shot3")

    (event,) = watch.poll(client)

    assert event["content"] == (
        'Note from the user on request "lighthouse", item "shot3" (Shot shot3), '
        "saved 2026-11-20 14:02 UTC:\n\n"
        "off lipsync - reroll it and recreate the final piece\n\n"
        f'Read the request with get_request("lighthouse", workspace="{WORKSPACE}") '
        'and act on the note; say what you did in the "why" of your next '
        "set_request."
    )
    assert event["meta"] == {
        "kind": "note",
        "workspace": WORKSPACE,
        "request": "lighthouse",
        "item": "shot3",
        "revision": "2",
        "note": "n1",
        "created": "2026-11-20T14:02:00+00:00",
    }
    assert watch.poll(client) == []


def test_a_request_level_note_names_no_item(client, clock):
    write(client)
    watch = started(client)
    clock.tick()
    note(client, "make the whole thing colder")

    (event,) = watch.poll(client)

    assert event["content"].startswith(
        'Note from the user on request "lighthouse", saved '
    )
    assert "item" not in event["meta"]
    assert event["meta"]["request"] == "lighthouse"


def test_an_edited_note_is_announced_again_and_an_unchanged_one_is_not(client, clock):
    write(client)
    watch = started(client)
    clock.tick()
    note(client, "warmer", "shot1")
    assert len(watch.poll(client)) == 1

    clock.tick()
    note(client, "colder", "shot2")  # shot1's note is untouched by this write
    (second,) = watch.poll(client)
    assert second["meta"]["item"] == "shot2"

    clock.tick()
    note(client, "warmer, and slower", "shot1")
    (edited,) = watch.poll(client)
    assert edited["meta"]["item"] == "shot1"
    assert "warmer, and slower" in edited["content"]


def test_a_note_the_agent_wrote_after_is_never_announced(client, clock):
    write(client)
    watch = started(client)
    clock.tick()
    note(client, "warmer", "shot1")
    clock.tick()
    write(client, why="retook shot1 warmer")

    assert watch.poll(client) == []


def test_several_notes_in_one_poll_are_one_event_each_in_plan_order(client, clock):
    write(client)
    watch = started(client)
    clock.tick()
    note(client, "b", "shot2")
    note(client, "whole", None)
    note(client, "a", "shot1")

    events = watch.poll(client)

    assert [e["meta"].get("item", "") for e in events] == ["", "shot1", "shot2"]


def test_a_closed_request_and_a_dropped_item_stop_being_announced(client, clock):
    write(client)
    write(client, "harbour")
    watch = started(client)
    clock.tick()
    note(client, "on a request about to close", "shot1", request_id="harbour")
    write(client, "harbour", by="user", why="closed", state="abandoned")
    note(client, "on an item about to be dropped", "shot2")
    plan = store.get_request(client.root, "lighthouse")["items"]
    plan = [{**i, "dropped": i["key"] == "shot2"} for i in plan]
    write(client, by="user", why="dropped shot2", items=plan)

    assert watch.poll(client) == []
    # and a closed request is not read at all
    clock.tick()
    assert watch.poll(client) == []
    assert client.reads() == ["lighthouse"]


def test_a_cleared_note_is_not_announced(client, clock):
    write(client)
    watch = started(client)
    clock.tick()
    note(client, "warmer", "shot1")
    note(client, None, "shot1")

    assert watch.poll(client) == []


# --- startup and restart ----------------------------------------------------


def test_startup_announces_what_is_waiting_once_as_one_digest(client, clock):
    write(client)
    clock.tick()
    note(client, "off lipsync:\nreroll it", "shot3")
    note(client, "colder overall")
    write(client, "harbour")
    clock.tick()
    note(client, "answered already", "shot1", request_id="harbour")
    clock.tick()
    write(client, "harbour", why="did it")

    watch = ledger.NoteWatch()
    (event,) = watch.poll(client)

    assert event["meta"] == {
        "kind": "notes_waiting",
        "workspace": WORKSPACE,
        "count": "2",
    }
    content = event["content"]
    assert content.startswith(
        "2 notes from the user were already waiting on dw requests in "
        f'workspace "{WORKSPACE}" when this session connected:'
    )
    assert (
        '- request "lighthouse", item "shot3" (Shot shot3), saved '
        "2026-11-20 14:01 UTC: off lipsync: reroll it" in content
    )
    assert '- request "lighthouse", saved' in content
    assert "answered already" not in content
    assert "confirm with them before spending GPU time" in content
    assert watch.poll(client) == []


def test_a_restart_repeats_the_digest_not_the_single_events(client, clock):
    write(client)
    first = started(client)
    clock.tick()
    note(client, "warmer", "shot1")
    assert first.poll(client)[0]["meta"]["kind"] == "note"

    (again,) = ledger.NoteWatch().poll(client)

    assert again["meta"]["kind"] == "notes_waiting"
    assert again["meta"]["count"] == "1"
    assert again["content"].startswith("1 note from the user was already waiting")


def test_startup_with_nothing_waiting_says_nothing(client):
    write(client)
    assert ledger.NoteWatch().poll(client) == []
    assert ledger.NoteWatch().poll(LedgerClient(client.root + "/nowhere")) == []


def test_the_digest_clips_each_note_and_caps_the_list(client, clock):
    keys = [f"s{n:02d}" for n in range(ledger.DIGEST_MAX_NOTES + 3)]
    write(client, items=items(*keys))
    clock.tick()
    for key in keys:
        note(client, "x" * (ledger.DIGEST_QUOTE_CAP + 50), key)

    (event,) = ledger.NoteWatch().poll(client)

    lines = [line for line in event["content"].splitlines() if line.startswith("- ")]
    assert len(lines) == ledger.DIGEST_MAX_NOTES + 1
    assert lines[-1] == "- and 3 more"
    assert lines[0].endswith("x" * ledger.DIGEST_QUOTE_CAP + " ... (clipped)")
    assert event["meta"]["count"] == str(len(keys))


# --- the event --------------------------------------------------------------


def _note(text, item="shot3", **more):
    return {
        "request": "lighthouse",
        "revision": 7,
        "item": item,
        "note": "n1",
        "title": "Shot 3: lamp room",
        "text": text,
        "created": "2026-11-20T14:02:09+00:00",
        "at": "2026-11-20T14:02:09+00:00",
        "refs": [],
        **more,
    }


@pytest.mark.parametrize(
    "event",
    [
        ledger.note_event(_note("x"), "studio"),
        ledger.note_event(_note("x", item=""), "studio"),
        ledger.waiting_event([_note("x"), _note("y", item="")], "studio"),
    ],
    ids=["item note", "request note", "digest"],
)
def test_meta_keys_are_identifiers_and_values_are_strings(event):
    """The channel contract: a key outside [A-Za-z0-9_] is silently dropped,
    and a value is a string."""
    assert set(event) == {"content", "meta"}
    assert isinstance(event["content"], str)
    for key, value in event["meta"].items():
        assert re.fullmatch(r"[A-Za-z0-9_]+", key), key
        assert isinstance(value, str), key


def test_a_note_is_quoted_verbatim_up_to_the_ledgers_own_cap():
    assert ledger.NOTE_QUOTE_CAP == MAX_NOTE_LENGTH
    text = 'line one\n  "quoted" <tag>\n' + "z" * 100
    whole = ledger.note_event(_note(text), "studio")["content"]
    assert f":\n\n{text}\n\nRead the request" in whole
    assert "Clipped" not in whole

    long = ledger.note_event(_note("q" * (MAX_NOTE_LENGTH + 5)), "studio")["content"]
    assert "q" * MAX_NOTE_LENGTH + "\n\n" in long
    assert "q" * (MAX_NOTE_LENGTH + 1) not in long
    assert (
        f"(Clipped at {MAX_NOTE_LENGTH} characters of {MAX_NOTE_LENGTH + 5}; "
        "get_request has the whole note.)" in long
    )


def test_a_saved_time_is_utc_whatever_offset_the_ledger_wrote():
    event = ledger.note_event(
        {**_note("x"), "at": "2026-11-20T16:02:09+02:00"}, "studio"
    )
    assert "saved 2026-11-20 14:02 UTC" in event["content"]


# --- failures ---------------------------------------------------------------


def test_an_unreachable_list_raises_and_changes_nothing(client, clock):
    write(client)
    watch = started(client)
    clock.tick()
    note(client, "warmer", "shot1")
    client.fail["/requests"] = DwApiError("Cannot reach diffusers-workflow")

    with pytest.raises(DwApiError):
        watch.poll(client)

    client.fail.clear()
    assert [e["meta"]["kind"] for e in watch.poll(client)] == ["note"]


def test_a_failed_first_poll_leaves_the_digest_for_the_first_that_works(client, clock):
    write(client)
    clock.tick()
    note(client, "warmer", "shot1")
    watch = ledger.NoteWatch()
    client.fail["/requests"] = DwApiError("401", status_code=401)
    with pytest.raises(DwApiError):
        watch.poll(client)

    client.fail.clear()
    assert [e["meta"]["kind"] for e in watch.poll(client)] == ["notes_waiting"]


def test_an_unreadable_ledger_is_a_failure_not_an_empty_one(client, tmp_path):
    write(client)
    watch = started(client)
    (tmp_path / "requests.json").write_text("{not json")

    with pytest.raises(DwApiError) as e:
        watch.poll(client)
    assert "could not be read" in str(e.value)
    assert str(tmp_path) not in str(e.value)


def test_one_unreadable_request_is_retried_and_the_others_still_announce(client, clock):
    write(client)
    write(client, "harbour")
    watch = started(client)
    clock.tick()
    note(client, "a", "shot1")
    note(client, "b", "shot1", request_id="harbour")
    client.fail["/requests/harbour"] = DwApiError("timed out")

    (event,) = watch.poll(client)
    assert event["meta"]["request"] == "lighthouse"
    assert watch.fault == "timed out"

    client.fail.clear()
    (late,) = watch.poll(client)
    assert late["meta"]["request"] == "harbour"
    assert watch.fault is None


def test_a_request_deleted_between_the_list_and_the_read_is_no_fault(client, clock):
    write(client)
    watch = started(client)
    clock.tick()
    note(client, "a", "shot1")
    client.fail["/requests/lighthouse"] = DwApiError("gone", status_code=404)

    assert watch.poll(client) == []
    assert watch.fault is None


def test_failed_polls_back_off_to_a_ceiling_and_a_good_one_resets():
    delays = [ledger.retry_delay(10.0, n) for n in range(8)]
    assert delays[:5] == [10.0, 20.0, 40.0, 80.0, 160.0]
    assert delays[5:] == [ledger.MAX_BACKOFF_SECONDS] * 3
    assert ledger.retry_delay(10.0, 10_000) == ledger.MAX_BACKOFF_SECONDS
    # an interval above the ceiling is never shortened by a failure
    assert ledger.retry_delay(900.0, 3) == 900.0


# --- references, edits, the header ------------------------------------------

REF = {
    "item": "shot3",
    "title": "Shot 3: lamp room",
    "takes": 3,
    "job": "j_8f2c",
    "attempt": 2,
    "output": "lighthouse/run_41/final/shot3.mp4",
    "kind": "video",
}


def test_the_list_poll_carries_the_channel_header_and_a_request_read_does_not(
    client, clock
):
    write(client)
    watch = ledger.NoteWatch()
    watch.poll(client)
    by_path = dict(client.headers)
    listing = f"/api/workspaces/{WORKSPACE}/requests"
    assert by_path[listing] == {"X-DW-Notes-Channel": "1"}
    assert by_path[f"{listing}/lighthouse"] is None


def test_an_event_names_each_reference_its_take_and_file():
    planned = {"item": "shot5", "title": "Shot 5: the keeper"}
    event = ledger.note_event(
        _note("too dark", refs=[REF, planned, {**REF, "item": "shot3"}]), "studio"
    )
    first, *rest = event["content"].split("\n\n")
    assert first == (
        'Note from the user on request "lighthouse", item "shot3" (Shot 3: lamp '
        "room), saved 2026-11-20 14:02 UTC, about:\n"
        '- item "shot3" (Shot 3: lamp room), take 2 of 3 (job j_8f2c), file '
        '"lighthouse/run_41/final/shot3.mp4" (video)\n'
        '- item "shot5" (Shot 5: the keeper), not rendered yet\n'
        '- item "shot3" (Shot 3: lamp room), take 2 of 3 (job j_8f2c), file '
        '"lighthouse/run_41/final/shot3.mp4" (video)'
    )
    assert rest[0] == "too dark"
    assert event["meta"]["items"] == "shot3,shot5"
    assert event["meta"]["note"] == "n1"
    assert event["meta"]["created"] == "2026-11-20T14:02:09+00:00"


def test_reference_variants():
    line = ledger._ref_line
    bare = {"item": "shot3", "title": "T", "job": "j1", "attempt": None}
    assert line(bare) == '- item "shot3" (T), job j1'
    assert line({**REF, "output": None}) == (
        '- item "shot3" (Shot 3: lamp room), take 2 of 3 (job j_8f2c)'
    )
    assert line({**REF, "gone": True}).endswith(
        "(video) (that job's run no longer exists)"
    )
    assert line({"item": "s", "title": "T", "gone": True, "dropped": True}) == (
        '- item "s" (T) (dropped from the plan)'
    )
    assert line({**bare, "gone": True}) == (
        '- item "shot3" (T), job j1 (no longer a take of this item)'
    )


def test_an_edited_note_says_when_it_was_saved_and_when_edited():
    event = ledger.note_event(_note("x", at="2026-11-20T15:30:00+00:00"), "studio")
    assert event["content"].startswith(
        'Note from the user on request "lighthouse", item "shot3" (Shot 3: lamp '
        "room), saved 2026-11-20 14:02 UTC, edited 2026-11-20 15:30 UTC:\n\n"
    )
    assert "items" not in event["meta"]


def test_the_digest_puts_a_notes_references_under_it():
    event = ledger.waiting_event([_note("dark", refs=[REF])], "studio")
    assert (
        '- request "lighthouse", item "shot3" (Shot 3: lamp room), saved '
        "2026-11-20 14:02 UTC: dark\n"
        '  - item "shot3" (Shot 3: lamp room), take 2 of 3 (job j_8f2c), '
        'file "lighthouse/run_41/final/shot3.mp4" (video)'
    ) in event["content"]


def test_two_notes_on_one_item_are_announced_separately(client, clock):
    write(client)
    watch = started(client)
    clock.tick()
    write(client, by="user", why="notes")
    path = os.path.join(client.root, REQUESTS_FILE_NAME)
    document = json.load(open(path))
    request = document["requests"]["lighthouse"]
    at = request["history"][-1]["at"]
    shot = next(i for i in request["items"] if i["key"] == "shot1")
    shot["notes"] = [
        {"id": f"n{n}", "text": t, "created": at, "at": at}
        for n, t in ((1, "warmer"), (2, "slower"))
    ]
    json.dump(document, open(path, "w"))

    events = watch.poll(client)

    assert [e["meta"]["note"] for e in events] == ["n1", "n2"]
    assert watch.poll(client) == []


def test_a_reference_is_read_from_the_plan_by_its_own_item(client, clock):
    write(client)
    watch = started(client)
    clock.tick()
    note(client, "lamp is off", "shot1", refs=[{"item": "shot2"}])
    (event,) = watch.poll(client)
    assert '- item "shot2" (Shot shot2), not rendered yet' in event["content"]
    assert event["meta"]["items"] == "shot2"


def test_a_note_edited_to_new_text_is_announced_again_and_the_same_text_is_not(
    client, clock
):
    write(client)
    watch = started(client)
    clock.tick()
    note(client, "warmer", "shot1")
    (first,) = watch.poll(client)
    assert "edited" not in first["content"]

    clock.tick()
    store.edit_note(client.root, "lighthouse", "n1", "warmer")  # same text: no-op
    assert watch.poll(client) == []

    clock.tick()
    store.edit_note(client.root, "lighthouse", "n1", "much warmer")
    (again,) = watch.poll(client)
    assert ", edited 2026-11-20 14:03 UTC" in again["content"]
    assert "saved 2026-11-20 14:01 UTC" in again["content"]


def test_a_queued_jobs_reference_is_not_stale(client, clock):
    write(client)
    watch = started(client)
    client.jobs = [
        {"job": "jq", "item": "shot1", "status": "queued", "run_exists": False}
    ]
    clock.tick()
    note(client, "when it runs, warmer", "shot1", refs=[{"item": "shot1", "job": "jq"}])
    (event,) = watch.poll(client)
    assert '- item "shot1" (Shot shot1), take 1 of 1 (job jq)\n' in event["content"]
    assert "no longer" not in event["content"]


# --- review holds ----------------------------------------------------------


class Mono:
    """The monotonic clock a NoteWatch reads, moved by the test."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def mono():
    return Mono()


def moment(minute):
    return datetime.fromisoformat(f"2026-11-20T14:{minute:02d}:00+00:00")


def held_plan(client, until_minute, request_id="lighthouse", **fields):
    """The plan with shot1 held until 14:<until_minute>, shot2 after it, as
    the agent wrote it at the clock's now. Both items have a workflow."""
    stored = store._load(client.root).get(request_id)
    keys = ("shot1", "shot2")
    plan_items = [
        {
            "key": key,
            "title": f"Shot {key}",
            "depends_on": ["shot1"] if key == "shot2" else [],
            "workflow": "txt2img",
            **({"estimate_minutes": 2.5} if key == "shot1" else {}),
            **(
                {"review_until": f"2026-11-20T14:{until_minute:02d}:00+00:00"}
                if key == "shot1"
                else {}
            ),
        }
        for key in keys
    ]
    return write(
        client,
        request_id,
        by="agent",
        items=plan_items,
        revision=stored["revision"] if stored else 0,
        **fields,
    )


def watching(client, mono):
    watch = ledger.NoteWatch(clock=mono)
    assert watch.poll(client) == []
    client.calls.clear()
    return watch


def test_a_due_ready_hold_is_announced_once(client, clock, mono):
    held_plan(client, 10)
    client.now = moment(0)
    watch = watching(client, mono)
    client.now = moment(11)
    mono.t += 11 * 60

    (event,) = watch.poll(client)

    assert event["content"] == (
        'Review time is up on request "lighthouse" with no note from the user. '
        "Held items now due:\n\n"
        "- shot1: Shot shot1 (about 2.5 min)\n\n"
        'Read the request with get_request("lighthouse", workspace="studio"). '
        "Run only the items still ready and without a new note, each with "
        'run_workflow(request="lighthouse/<key>", acknowledged_cost=<the plan '
        "you quoted>). If a run is refused for a changed plan, or "
        "validate_workflow now quotes a figure well off what you told the user, "
        "do not run it: set a new review_until with the new quote in the why."
    )
    assert event["meta"] == {
        "kind": "review_due",
        "workspace": WORKSPACE,
        "request": "lighthouse",
        "revision": "1",
        "items": "shot1",
        "until": "2026-11-20T14:10:00+00:00",
    }
    assert all(isinstance(v, str) for v in event["meta"].values())
    mono.t += 60
    assert watch.poll(client) == []
    assert watch.poll(client) == []


def test_a_line_without_an_estimate_has_no_bracket():
    due = [
        {
            "request": "r",
            "revision": 3,
            "item": "a",
            "title": "A",
            "until": "2026-11-20T14:10:00+00:00",
            "estimate_minutes": None,
        }
    ]
    assert "- a: A\n\nRead" in ledger.review_event(due, "w")["content"]


def test_nothing_is_announced_while_holding(client, clock, mono):
    held_plan(client, 10)
    client.now = moment(0)
    watch = watching(client, mono)
    client.now = moment(5)
    mono.t += 5 * 60
    assert watch.poll(client) == []


def test_a_redirected_hold_is_not_announced(client, clock, mono):
    held_plan(client, 10)
    client.now = moment(0)
    watch = watching(client, mono)
    clock.tick(3)
    note(client, "warmer first", "shot1")
    client.now = moment(11)
    mono.t += 11 * 60
    (event,) = watch.poll(client)  # the note, and no review_due after it
    assert event["meta"]["kind"] == "note"


def test_a_deadline_passing_rereads_a_request_whose_stamp_did_not_move(
    client, clock, mono
):
    held_plan(client, 10)
    client.now = moment(0)
    watch = watching(client, mono)
    assert watch.poll(client) == []
    assert client.reads() == []  # unchanged and not yet due: the list only
    client.now = moment(10)
    mono.t += 10 * 60 + 1
    (event,) = watch.poll(client)
    assert event["meta"]["kind"] == "review_due"
    assert client.reads() == ["lighthouse"]


def test_due_but_not_ready_says_nothing_until_it_is(client, clock, mono):
    held_plan(client, 10)
    client.jobs = [{"job": "j", "item": "shot1", "status": "running"}]
    client.now = moment(0)
    watch = watching(client, mono)
    client.now = moment(11)
    mono.t += 11 * 60
    assert watch.poll(client) == []
    client.jobs = []  # the running job went away: shot1 is ready again
    (event,) = watch.poll(client)
    assert event["meta"]["items"] == "shot1"


def test_a_new_review_until_is_announced_again(client, clock, mono):
    held_plan(client, 10)
    client.now = moment(0)
    watch = watching(client, mono)
    client.now = moment(11)
    mono.t += 11 * 60
    assert len(watch.poll(client)) == 1
    clock.minute = 12
    held_plan(client, 20)
    client.now = moment(13)
    assert watch.poll(client) == []
    client.now = moment(21)
    mono.t += 9 * 60
    (event,) = watch.poll(client)
    assert event["meta"]["until"] == "2026-11-20T14:20:00+00:00"


def test_the_first_poll_sends_a_stale_event_after_the_digest(client, clock, mono):
    held_plan(client, 10)
    note(client, "from before", "shot2")
    client.now = moment(20)
    digest, event = ledger.NoteWatch(clock=mono).poll(client)
    assert digest["meta"]["kind"] == "notes_waiting"
    assert event["content"].startswith(
        'Review time passed on request "lighthouse" before this session '
        "connected. Held items:\n\n- shot1: Shot shot1 (about 2.5 min)\n\n"
    )
    assert event["content"].endswith(
        "These holds were set before this session, so the user's go-ahead may "
        "be stale. Tell the user what is waiting and confirm before spending "
        "GPU time."
    )
    assert event["meta"]["stale"] == "1"
    assert event["meta"]["kind"] == "review_due"
    assert (
        "stale"
        not in ledger.review_event(
            [
                {
                    "request": "r",
                    "revision": 1,
                    "item": "a",
                    "title": "A",
                    "until": "u",
                    "estimate_minutes": 1,
                }
            ],
            "w",
        )["meta"]
    )


def test_a_note_and_a_due_hold_in_one_poll_send_the_note_first(client, clock, mono):
    held_plan(client, 10)
    write(client, "other")
    client.now = moment(0)
    watch = watching(client, mono)
    clock.tick(2)
    note(client, "look at this", "shot3", request_id="other")
    client.now = moment(11)
    mono.t += 11 * 60
    note_ev, review_ev = watch.poll(client)
    assert note_ev["meta"]["kind"] == "note"
    assert review_ev["meta"]["kind"] == "review_due"


def test_a_closed_request_drops_its_hold_bookkeeping(client, clock, mono):
    held_plan(client, 10)
    client.now = moment(0)
    watch = watching(client, mono)
    stored = store._load(client.root)["lighthouse"]
    write(client, state="done", revision=stored["revision"])
    watch.poll(client)
    assert "lighthouse" not in watch._due_at
    assert "lighthouse" not in watch._due_told
    assert "lighthouse" not in watch._stamps
