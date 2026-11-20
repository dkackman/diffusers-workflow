"""The notes channel's rules (`dw_mcp/ledger.py`): which notes wait, what is
announced and when, and what an event says. The client here answers from
the real ledger (`dw/ledger.py`) on a temp folder, under a clock the test
moves; nothing sleeps."""

import re

import pytest

from dw import ledger as store
from dw.notes import MAX_NOTE_LENGTH
from dw_mcp import ledger
from dw_mcp.client import DwApiError

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
        self.fail = {}  # path suffix -> DwApiError to raise

    def get_json(self, path, params=None, workspace=None):
        self.calls.append(path)
        for suffix, error in self.fail.items():
            if path.endswith(suffix):
                raise error
        prefix = f"/api/workspaces/{WORKSPACE}/requests"
        assert path.startswith(prefix), path
        request_id = path[len(prefix) :].lstrip("/")
        if not request_id:
            return {"workspace": WORKSPACE, **store.listing(self.root)}
        found = store.get_request(self.root, request_id)
        if found is None:
            raise DwApiError(f"No request {request_id!r}", status_code=404)
        return found

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


def note(client, text, item=None, request_id="lighthouse"):
    """The user saves `text` on an item, or on the request itself."""
    current = store.get_request(client.root, request_id)
    if item is None:
        return write(client, request_id, by="user", why="note", comment=text)
    plan = [
        {**entry, "comment": text} if entry["key"] == item else entry
        for entry in current["items"]
    ]
    return write(client, request_id, by="user", why=f"note on {item}", items=plan)


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
    }
    assert watch.poll(client) == []


def test_a_request_level_note_names_no_item(client, clock):
    write(client)
    watch = started(client)
    clock.tick()
    note(client, "make the whole thing colder")

    (event,) = watch.poll(client)

    assert event["content"].startswith(
        'Note from the user on request "lighthouse" (Lighthouse short), saved '
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
    assert '- request "lighthouse" (Lighthouse short), saved' in content
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


def _note(text, item="shot3"):
    return {
        "request": "lighthouse",
        "revision": 7,
        "item": item,
        "title": "Shot 3: lamp room",
        "text": text,
        "at": "2026-11-20T14:02:09+00:00",
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
