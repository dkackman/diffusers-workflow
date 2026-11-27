"""The requests ledger: a workspace's plan of what a person asked for.

One file, `requests.json` at a workspace root, holding
`{"requests": {<id>: Request}}`. A request is the whole plan an agent (or
the user, from the UI) wrote for one ask: a title, a goal, the prompt that
started it (write-once), its state and an ordered list of items, each a
piece of work with a key, a title, the keys it depends on, an estimate and
the workflow and arguments meant to make it. Every write is a whole plan,
revisioned: the caller names the revision it read, and a write against a
newer one is refused rather than merged.

An item's status is derived, never stored, from the jobs linked to it (a
job is submitted with a `request` and an `item`; the callers pass the linked
jobs into the views): `dropped` when it was removed from the plan after jobs
were linked to it (an item with no linked jobs is deleted when a write
leaves it out); else `done` when any linked job succeeded, `running`,
`queued`, `failed` (every linked job failed or was cancelled) or `planned`
(none). Each item also derives `attempts`, `result` and `spent_minutes`, and
the request `spent_minutes` and `remaining_minutes`; this module reads only
the job dicts it is handed, never the job store.

Written with `write_json_atomic` under a file lock beside it
(`.requests.json.lock`, the model_use.py pattern), so two servers on one
workspace read, check the revision and replace the file one at a time; a
per-directory thread lock in front keeps one process's threads from
queueing on the file. Read without a lock: a rename is atomic, so a reader
sees the old file or the new one. Callers resolve and validate the workspace root; this
module joins only its own fixed file name onto it.
"""

import json
import os
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from filelock import FileLock, Timeout

from .notes import NoteTooLong, clean_note
from .runs import write_json_atomic
from .security import (
    MAX_VARIABLE_VALUE_LENGTH,
    validate_ledger_key,
)
from .workspace import REQUESTS_FILE_NAME, REQUESTS_LOCK_NAME
from .ledger_items import MAX_TITLE_LENGTH, _check_dependencies, _item, _key, _text
from .ledger_views import (  # noqa: F401 - COVER_* and the views are read as ledger.*
    COVER_SCAN,
    COVER_SIZE,
    STATES,
    LedgerError,
    _load,
    _unreadable,
    _upgrade,
    ledger_path,
    read_requests,
    unreadable,
    _live_keys,
    item_changes,
    current_prompts,
    is_revised,
    item_view,
    request_view,
    summary_view,
    takes,
)

MAX_WHY_LENGTH = 200
MAX_TEXT_LENGTH = MAX_VARIABLE_VALUE_LENGTH  # prompt, goal
# Caps that never wedge a workspace: only what is still in play counts
# toward the working caps (live items, open requests), and what is finished
# has a far higher cap of its own with a way out - a closed request is
# deleted, a dropped item is purged. MAX_FILE_BYTES bounds the whole file.
MAX_ITEMS = 200  # live items of one request
MAX_DROPPED_ITEMS = 200  # dropped items of one request
MAX_REQUESTS = 100  # open requests of one workspace
MAX_CLOSED_REQUESTS = 1000  # done and abandoned ones
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_HISTORY = 100
# History entries that keep `changed`, newest first: bounds what it adds to the file
MAX_CHANGED_HISTORY = 10
MAX_NOTES = 20  # live notes of one target: an item, or the request itself
MAX_NOTE_REFS = 16  # references in one note
# Seconds a write waits for another server's write to the same file
LOCK_TIMEOUT = 10

AUTHORS = ("agent", "user")

# What an item may carry on a write. `status`, `ready`, `attempts`,
# `result`, `accepted_missing`, `spent_minutes`, `review_set_at`, `review`,
# `revised_because` and the server's stamps `changed_at` and
# `accepted_basis` are derived or server-set and ignored, so a plan read
# with GET can be written back.
_ITEM_FIELDS = frozenset(
    (
        "key",
        "title",
        "depends_on",
        "estimate_minutes",
        "workflow",
        "arguments",
        "dropped",
        "accepted",
        "notes",
        "status",
        "ready",
        "attempts",
        "result",
        "accepted_missing",
        "spent_minutes",
        "review_until",
        "review_set_at",
        "review",
        "changed_at",
        "accepted_basis",
        "revised_because",
    )
)

_locks = {}
_locks_guard = threading.Lock()


def _lock(directory):
    key = os.path.realpath(directory)
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


@contextmanager
def _writing(root):
    """Hold the ledger at `root` for one read-check-replace: this process's
    threads first, then every other process on the file lock. Raises
    LedgerError (503 `ledger_busy`) when another process keeps it past
    `LOCK_TIMEOUT`."""
    with _lock(root):
        held = FileLock(
            os.path.join(root, REQUESTS_LOCK_NAME),
            timeout=LOCK_TIMEOUT,
            thread_local=False,
        )
        try:
            held.acquire()
        except Timeout as e:
            raise LedgerError(
                f"{REQUESTS_FILE_NAME} is being written by another server; try again",
                status=503,
                code="ledger_busy",
            ) from e
        try:
            yield
        finally:
            held.release()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _created():
    """A new request's `created`, on a job's `created_at` clock (`takes`)."""
    return time.time()


# --- views (dw/ledger_views.py derives them) -----------------------------


def _no_jobs(_request_id):
    return []


def list_requests(root, jobs_for=_no_jobs, prompts=None, needs=False):
    """The list view, newest change first. `jobs_for(request_id)` lists the
    jobs linked to a request; `prompts` the `name -> digest` callable rule 2
    reads (None: rule 1 alone; `is_revised`); `needs` adds each row's."""
    rows = [
        summary_view(r, jobs_for(r.get("id")), prompts, needs)
        for r in read_requests(root).values()
    ]
    rows.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
    return rows


def listing(root, jobs_for=_no_jobs, prompts=None, needs=False):
    """What the list route answers: `requests` (`list_requests`), and
    `unreadable` - why - when the file is there but is not a ledger."""
    problem = unreadable(root)
    if problem is not None:
        return {"requests": [], "unreadable": str(problem)}
    return {"requests": list_requests(root, jobs_for, prompts, needs)}


def get_request(root, request_id, jobs_for=_no_jobs, prompts=None):
    """One request's view, or None. Raises LedgerError when the file is
    there but is not a ledger. `prompts` as `list_requests`'."""
    validate_ledger_key(request_id, "Request id")
    request = (_load(root) if root is not None else {}).get(request_id)
    if request is None:
        return None
    return request_view(request, jobs_for(request_id), prompts=prompts)


def link_problem(root, request_id, item_key):
    """The LedgerError that says why a job cannot be linked to
    `request_id`/`item_key` in the workspace at `root`, or None when it can:
    `unknown_request` (no such request, or no root), `request_closed`,
    `unknown_item`, `item_dropped`, or `ledger_unreadable` (409). Both keys are checked as keys first
    (InvalidInputError otherwise). Takes no lock."""
    validate_ledger_key(request_id, "Request")
    validate_ledger_key(item_key, "Item")
    problem = unreadable(root)
    if problem is not None:
        return problem
    request = read_requests(root).get(request_id)
    if request is None:
        return LedgerError(
            f"no request {request_id!r} in this workspace",
            status=422,
            code="unknown_request",
        )
    if request.get("state") != "open":
        return LedgerError(
            f"request {request_id!r} is {request.get('state')}, not open",
            status=422,
            code="request_closed",
        )
    for item in request.get("items", []):
        if item.get("key") == item_key:
            if item.get("dropped"):
                return LedgerError(
                    f"item {item_key!r} of request {request_id!r} was dropped",
                    status=422,
                    code="item_dropped",
                )
            return None
    return LedgerError(
        f"request {request_id!r} has no item {item_key!r}",
        status=422,
        code="unknown_item",
    )


def _no_links(_request_id, _key):
    return []


# --- the write -----------------------------------------------------------


_BODY_FIELDS = frozenset(
    (
        "title",
        "goal",
        "items",
        "revision",
        "why",
        "prompt",
        "state",
        "purge",
        "notes",
        "next_note",
        "agent_wrote_at",
        "close_proposal",
    )
)

# A body that leaves `close_proposal` out keeps the stored one; `None` is
# the agent's withdrawal, so the two are told apart
_LEFT_OUT = object()
CLOSED_STATES = ("done", "abandoned")
MAX_CLOSE_REASON_LENGTH = 200


def _check_body(body):
    """The write's own fields, checked before the file is read: revision,
    title, goal, why, prompt, state and the raw items."""
    if not isinstance(body, dict):
        raise LedgerError("the request body must be an object")
    unknown = set(body) - _BODY_FIELDS
    if unknown:
        raise LedgerError(f"unknown fields {sorted(unknown)}")
    revision = body.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise LedgerError(
            "revision must be the revision you read, or 0 to create the request"
        )
    prompt = body.get("prompt")
    if prompt is not None:
        _text(prompt, "prompt", MAX_TEXT_LENGTH, allow_empty=True)
    state = body.get("state")
    if state is not None and state not in STATES:
        raise LedgerError(f"state must be one of {', '.join(STATES)}")
    raw_items = body.get("items")
    if not isinstance(raw_items, list):
        raise LedgerError("items must be a list")
    if len(raw_items) > MAX_ITEMS + MAX_DROPPED_ITEMS:
        raise LedgerError(f"at most {MAX_ITEMS} items, got {len(raw_items)}")
    purge = body.get("purge") or []
    if not isinstance(purge, list) or len(purge) > MAX_ITEMS + MAX_DROPPED_ITEMS:
        raise LedgerError("purge must be a list of item keys")
    return {
        "purge": [_key(key, "purge key") for key in purge],
        "revision": revision,
        "title": _text(body.get("title"), "title", MAX_TITLE_LENGTH, one_line=True),
        "goal": _text(body.get("goal"), "goal", MAX_TEXT_LENGTH, allow_empty=True),
        "why": _text(body.get("why"), "why", MAX_WHY_LENGTH, one_line=True),
        "prompt": prompt,
        "state": state,
        "items": raw_items,
        "close_proposal": body.get("close_proposal", _LEFT_OUT),
    }


def _check_against(requests, request_id, write, by):
    """The stored request a write replaces (None to create), refusing a
    stale revision, an agent's close, a full ledger and a changed prompt."""
    current = requests.get(request_id)
    current_revision = current.get("revision", 0) if current else 0
    if write["revision"] != current_revision:
        raise LedgerError(
            f"request {request_id!r} is at revision {current_revision}, not "
            f"{write['revision']} - read it again and rewrite the plan against "
            f"revision {current_revision}",
            status=409,
            code="stale_revision",
            current_revision=current_revision,
        )
    _check_close(current, write["state"], by)
    _check_room(requests, current, write["state"], by)
    if (
        current is not None
        and write["prompt"] is not None
        and write["prompt"] != current.get("prompt")
    ):
        raise LedgerError(
            "prompt is write-once: it is the ask that started the request "
            "and was set when it was created",
            status=422,
            code="prompt_is_write_once",
        )
    return current


def _check_close(current, state, by):
    """Refuse (422 `close_needs_user`) an agent write that closes the
    request, or changes how it was closed: only the user closes. Leaving
    `state` out, or sending the stored one, passes, and so does `open`."""
    was = current.get("state") if current else "open"
    if by == "agent" and state in CLOSED_STATES and state != was:
        raise LedgerError(
            "only the user closes a request - propose it with close_proposal "
            "{state, reason} and the user accepts or declines it in the UI",
            status=422,
            code="close_needs_user",
        )


def _check_room(requests, current, state, by="user"):
    """Refuse (409 `ledger_full`) a write that would add an open request
    past MAX_REQUESTS or a closed one past MAX_CLOSED_REQUESTS. Only a
    request entering a side counts - creating, closing, reopening - so one
    already there stays writable at the cap."""
    was = current.get("state") if current else None
    state = state or was or "open"
    is_open = state == "open"
    if was is not None and (was == "open") == is_open:
        return
    held = sum((r.get("state") == "open") == is_open for r in requests.values())
    if is_open and held >= MAX_REQUESTS:
        raise LedgerError(
            f"a workspace holds at most {MAX_REQUESTS} open requests - "
            + (
                "propose a close or delete one"
                if by == "agent"
                else "close one (state done or abandoned) or delete one"
            )
            + " to make room",
            status=409,
            code="ledger_full",
        )
    if not is_open and held >= MAX_CLOSED_REQUESTS:
        raise LedgerError(
            f"a workspace holds at most {MAX_CLOSED_REQUESTS} done or "
            "abandoned requests - delete one to make room",
            status=409,
            code="ledger_full",
        )


def _check_purge(purge, stored, listed):
    for key in purge:
        if key in listed:
            raise LedgerError(
                f"purge names {key!r}, which this plan still lists - leave it "
                "out of items to purge it"
            )
        if key not in stored:
            raise LedgerError(f"purge names {key!r}, which is not an item")


def _merge_items(
    raw_items,
    current,
    request_id,
    linked_jobs,
    by,
    at,
    purge=(),
    prompts=None,
    warnings=None,
):
    """The plan's items in its order, then every stored item it no longer
    lists that jobs link to, as dropped; one no job links to, or named in
    `purge`, is deleted. Live and dropped items are capped apart."""
    stored = {item.get("key"): item for item in (current or {}).get("items", [])}
    live, seen = [], set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise LedgerError("each item must be an object")
        key = _key(raw.get("key"), "Item key")
        unknown = set(raw) - _ITEM_FIELDS
        if unknown:
            raise LedgerError(f"item {key!r}: unknown fields {sorted(unknown)}")
        if key in seen:
            raise LedgerError(f"item {key!r} appears twice")
        seen.add(key)
        if raw.get("dropped") is True:
            continue  # sent back as dropped: removed from the plan, like an omission
        if raw.get("dropped") not in (None, False):
            raise LedgerError(f"item {key!r}: dropped must be true or false")
        previous = stored.get(key) or {}
        live.append(
            _item(
                raw,
                key,
                previous,
                linked_jobs(request_id, key),
                by,
                at,
                prompts,
                warnings,
            )
        )
    listed = {item["key"] for item in live}
    _check_purge(purge, stored, listed)
    dropped = [
        {**item, "dropped": True}
        for key, item in stored.items()
        if key not in listed and key not in purge and linked_jobs(request_id, key)
    ]
    if len(live) > MAX_ITEMS:
        raise LedgerError(
            f"at most {MAX_ITEMS} items, not counting dropped ones; got {len(live)}"
        )
    if len(dropped) > MAX_DROPPED_ITEMS:
        raise LedgerError(
            f"at most {MAX_DROPPED_ITEMS} dropped items, got {len(dropped)} - "
            "name the ones to remove for good in `purge` (their jobs keep "
            "their history, link and files)",
            status=409,
            code="too_many_dropped_items",
        )
    # The live plan only: a dropped item keeps the depends_on it was stored
    # with and no write can edit it, so checking it would hold its
    # dependency in the plan for good
    _check_dependencies(live)
    return live + dropped


def _history(current, write, state, revision, by, at):
    """The history with this write's entry: who, when and why, with the
    goal and the state when it set or changed them."""
    entry = {"revision": revision, "at": at, "by": by, "why": write["why"]}
    if current is None or write["goal"] != current.get("goal"):
        entry["goal"] = write["goal"]
    if current is None or state != current.get("state"):
        entry["state"] = state
    history = list((current or {}).get("history", [])) + [entry]
    if len(history) > MAX_HISTORY:
        history = history[:1] + history[-(MAX_HISTORY - 1) :]
    return history


def _agent_proposal(sent, state, at, revision):
    """The agent's `{state, reason}` as stored, pending; refused (422
    `request_closed`) on a request this write leaves closed."""
    if set(sent) != {"state", "reason"}:
        raise LedgerError(
            "the agent proposes a close as close_proposal {state, reason}"
        )
    if sent["state"] not in CLOSED_STATES:
        raise LedgerError("close_proposal state must be done or abandoned")
    if state != "open":
        raise LedgerError(
            f"the request is already {state} - there is nothing to propose",
            status=422,
            code="request_closed",
        )
    reason = _text(
        sent["reason"],
        "close_proposal reason",
        MAX_CLOSE_REASON_LENGTH,
        one_line=True,
    )
    return {
        "state": sent["state"],
        "reason": reason,
        "by": "agent",
        "at": at,
        "revision": revision,
        "outcome": "pending",
    }


def _user_decline(sent, stored, at):
    """The pending proposal settled as declined by the user, with an
    optional one-line note; refused (422 `no_close_proposal`) when nothing
    is pending."""
    if sent.get("outcome") != "declined" or not set(sent) <= {"outcome", "note"}:
        raise LedgerError(
            "the user settles a close proposal as close_proposal "
            '{outcome: "declined", note} - or closes the request with state'
        )
    if not stored or stored.get("outcome") != "pending":
        raise LedgerError(
            "there is no pending close proposal to decline",
            status=422,
            code="no_close_proposal",
        )
    declined = {**stored, "outcome": "declined", "decided_at": at}
    note = sent.get("note")
    if note is not None and note != "":
        declined["note"] = _text(
            note, "close_proposal note", MAX_CLOSE_REASON_LENGTH, one_line=True
        )
    return declined


def _close_proposal(current, write, state, by, at, revision):
    """`(proposal, event)`: the close proposal this write leaves stored
    (None for none) and the history `close` it records, if any. A body that
    leaves the field out, or sends the stored one back, keeps it; the agent
    proposes (`close_proposed`, replacing any earlier one) or withdraws one
    pending with null (`close_withdrawn`); the user declines one pending
    (`close_declined`), and the user's close accepts one pending
    (`close_accepted`), whatever state the user picks."""
    stored = (current or {}).get("close_proposal")
    sent = write["close_proposal"]
    proposal, event = stored, None
    if sent is _LEFT_OUT or (sent is not None and sent == stored):
        pass
    elif sent is None:
        if by == "agent" and stored and stored.get("outcome") == "pending":
            proposal, event = None, "close_withdrawn"
    elif not isinstance(sent, dict):
        raise LedgerError("close_proposal must be an object or null")
    elif by == "agent":
        proposal = _agent_proposal(sent, state, at, revision)
        event = "close_proposed"
    else:
        proposal, event = _user_decline(sent, stored, at), "close_declined"
    was = (current or {}).get("state", "open")
    if (
        by == "user"
        and state in CLOSED_STATES
        and was == "open"
        and proposal
        and proposal.get("outcome") == "pending"
    ):
        proposal = {**proposal, "outcome": "accepted", "decided_at": at}
        event = "close_accepted"
    return proposal, event


def _record_changes(history, before, after):
    """Put on the newest history entry which items this write changed, and
    drop that from all but the newest MAX_CHANGED_HISTORY entries."""
    changed = item_changes(before, after)
    if changed:
        history[-1]["changed"] = changed
    for entry in [e for e in history if "changed" in e][:-MAX_CHANGED_HISTORY]:
        del entry["changed"]


def put_request(
    root,
    request_id,
    body,
    *,
    by,
    linked_jobs=_no_links,
    jobs_for=_no_jobs,
    prompts=None,
):
    """Write one request's whole plan and return its view.

    `body` holds title, goal, items, revision and why, and optionally
    prompt, state and `purge` (keys of items this plan leaves out,
    removed for good even though jobs link to them) and `close_proposal`.
    Only the user closes a request: an agent's `state` done or abandoned
    is 422 `close_needs_user`, and the agent proposes the close instead
    (`_close_proposal`). `linked_jobs(request_id, key)` lists the jobs linked
    to an item as `{job, status, run_id, created_at, workflow, arguments,
    prompt_digests}`;
    `accepted` must name one of them that succeeded, and the rest say
    whether the stored item reads revised (`_item`). `jobs_for(request_id)` lists every job linked to the request, for
    the returned view's derived fields, and `prompts` the `name -> digest`
    callable the revised rule reads (rule 2), for both.

    Raises LedgerError, or OSError when the file cannot be written.
    """
    if root is None:
        raise LedgerError(
            "this workspace has no root folder, so it cannot hold requests",
            status=409,
            code="no_workspace_root",
        )
    _key(request_id, "Request id")
    if by not in AUTHORS:
        raise LedgerError(f"by must be one of {', '.join(AUTHORS)}")
    write = _check_body(body)

    with _writing(root):
        requests = _load(root)
        current = _check_against(requests, request_id, write, by)
        at = _now()
        revision = write["revision"] + 1
        state = write["state"] or (current or {}).get("state") or "open"
        warnings = []
        # which instance of the id this is: jobs older than it are not takes
        born = {"created": _created()} if current is None else current
        request = {
            "id": request_id,
            "title": write["title"],
            "state": state,
            "revision": revision,
            "prompt": current.get("prompt") if current else write["prompt"],
            "goal": write["goal"],
            "notes": list((current or {}).get("notes") or []),
            "next_note": (current or {}).get("next_note", 1),
            "agent_wrote_at": at
            if by == "agent"
            else (current or {}).get("agent_wrote_at"),
            "items": _merge_items(
                write["items"],
                current,
                request_id,
                lambda rid, key: takes(born, linked_jobs(rid, key)),
                by,
                at,
                write["purge"],
                prompts,
                warnings,
            ),
            "history": _history(current, write, state, revision, by, at),
        }
        if "created" in born:
            request["created"] = born["created"]
        proposal, event = _close_proposal(current, write, state, by, at, revision)
        if proposal is not None:
            request["close_proposal"] = proposal
        if event:
            request["history"][-1]["close"] = event
        if current is not None:  # nothing diffs against a request's first write
            _record_changes(request["history"], current["items"], request["items"])
        requests[request_id] = request
        document = {"requests": requests}
        size = len(json.dumps(document, indent=2).encode("utf-8"))
        if size > MAX_FILE_BYTES:
            raise LedgerError(
                f"{REQUESTS_FILE_NAME} would be {size} bytes, past its "
                f"{MAX_FILE_BYTES}-byte cap - shorten the plan's text or "
                "arguments, purge dropped items or delete a closed request",
                status=413,
                code="ledger_too_large",
            )
        write_json_atomic(ledger_path(root), document)
    view = request_view(request, jobs_for(request_id), prompts=prompts)
    if warnings:
        view["warnings"] = warnings
    return view


def _note_target(requests, request_id, item):
    """The stored request and the note list a note belongs on: the item's
    when `item` is given, else the request's."""
    request = requests.get(request_id)
    if request is None:
        raise LedgerError(
            f"no request {request_id!r} in this workspace",
            status=404,
            code="unknown_request",
        )
    if item is None:
        request["notes"] = request.get("notes") or []
        return request, request["notes"]
    for stored in request["items"]:
        if stored.get("key") == item:
            if stored.get("dropped"):
                raise LedgerError(
                    f"item {item!r} of request {request_id!r} was dropped",
                    status=422,
                    code="item_dropped",
                )
            stored["notes"] = stored.get("notes") or []
            return request, stored["notes"]
    raise LedgerError(
        f"request {request_id!r} has no item {item!r}",
        status=422,
        code="unknown_item",
    )


def _find_note(request, note_id):
    """The note list holding `note_id` and the note, in the request or in
    any of its items."""
    for notes in (request.get("notes"), *(i.get("notes") for i in request["items"])):
        for note in notes or []:
            if note.get("id") == note_id:
                return notes, note
    raise LedgerError(
        f"request {request['id']!r} has no note {note_id!r}",
        status=404,
        code="unknown_note",
    )


def _note_text(text):
    try:
        cleaned = clean_note(text)
    except (TypeError, NoteTooLong) as e:
        raise LedgerError(str(e)) from e
    if cleaned is None:
        raise LedgerError("a note needs text - delete it to remove it")
    return cleaned


def _noted_write(root, requests, request, why):
    """Record a note write by the user: bump the revision, add its history
    entry, check the file's size and save. Returns the new revision."""
    at = _now()
    request["revision"] += 1
    history = (request.get("history") or []) + [
        {"revision": request["revision"], "at": at, "by": "user", "why": why}
    ]
    if len(history) > MAX_HISTORY:
        history = history[:1] + history[-(MAX_HISTORY - 1) :]
    request["history"] = history
    document = {"requests": requests}
    size = len(json.dumps(document, indent=2).encode("utf-8"))
    if size > MAX_FILE_BYTES:
        raise LedgerError(
            f"{REQUESTS_FILE_NAME} would be {size} bytes, past its "
            f"{MAX_FILE_BYTES}-byte cap - shorten the note or delete a closed "
            "request",
            status=413,
            code="ledger_too_large",
        )
    write_json_atomic(ledger_path(root), document)
    return request["revision"]


def _need_root(root):
    if root is None:
        raise LedgerError(
            "this workspace has no root folder, so it cannot hold requests",
            status=409,
            code="no_workspace_root",
        )


def _note_refs(request, item, refs):
    """A new note's references as `{item, job?, output?}`, checked against
    the plan and with duplicates collapsed in first order. An item note's
    references default to and must name its item; each item must be in the
    plan and not dropped; `output` needs `job`. Raises LedgerError
    (`too_many_refs`, `invalid_note_ref`)."""
    refs = list(refs or [])
    if len(refs) > MAX_NOTE_REFS:
        raise LedgerError(
            f"at most {MAX_NOTE_REFS} references fit in one note, not {len(refs)}",
            status=422,
            code="too_many_refs",
        )
    live = _live_keys(request["items"])
    out = []
    for ref in refs:
        given = {k: ref[k] for k in ("item", "job", "output") if ref.get(k) is not None}
        ref = {"item": item, **given}
        where = ref["item"]
        if not isinstance(where, str):
            raise _bad_ref("a request note's reference names its item")
        if item is not None and where != item:
            raise _bad_ref(
                f"a note on item {item!r} references only that item, not {where!r}"
            )
        if where not in live:
            raise _bad_ref(
                f"item {where!r} of request {request['id']!r} is not in the plan "
                "or was dropped"
            )
        if "output" in ref and "job" not in ref:
            raise _bad_ref("a reference names `output` only with its `job`")
        if ref not in out:
            out.append(ref)
    return out


def _bad_ref(why):
    return LedgerError(why, status=422, code="invalid_note_ref")


def add_note(root, request_id, *, item, text, refs, check_ref):
    """Add one user note to an item (or to the request when `item` is None)
    and answer `{note, revision}`. `refs` is a list of `{item?, job?,
    output?}` (see `_note_refs`); `check_ref(request_id, ref)` is called
    once per reference, after its item is filled in, and answers the
    reference to store or raises LedgerError (`invalid_note_ref`); the
    ledger reads no jobs itself. One refused reference refuses the note.

    Raises LedgerError (`unknown_request`, `unknown_item`, `item_dropped`,
    `too_many_notes`, `too_many_refs`, `invalid_note_ref`), or OSError when
    the file cannot be written."""
    _need_root(root)
    _key(request_id, "Request id")
    text = _note_text(text)
    with _writing(root):
        requests = _load(root)
        request, notes = _note_target(requests, request_id, item)
        if len(notes) >= MAX_NOTES:
            raise LedgerError(
                f"at most {MAX_NOTES} notes fit on one request or item - "
                "delete one first",
                status=422,
                code="too_many_notes",
            )
        stored = [check_ref(request_id, ref) for ref in _note_refs(request, item, refs)]
        at = _now()
        note = {
            "id": f"n{request['next_note']}",
            "text": text,
            "created": at,
            "at": at,
            **({"refs": stored} if stored else {}),
        }
        request["next_note"] += 1
        notes.append(note)
        revision = _noted_write(root, requests, request, "note added")
    return {"note": note, "revision": revision}


def edit_note(root, request_id, note_id, text):
    """Change one note's text and answer `{note, revision}`. `at` moves to
    now, `created` and `refs` stay. The same text is a no-op: the note and the
    current revision come back and nothing is written."""
    _need_root(root)
    _key(request_id, "Request id")
    text = _note_text(text)
    with _writing(root):
        requests = _load(root)
        request, _ = _note_target(requests, request_id, None)
        _, note = _find_note(request, note_id)
        if note["text"] == text:
            return {"note": note, "revision": request["revision"]}
        note["text"] = text
        note["at"] = _now()
        revision = _noted_write(root, requests, request, "note edited")
    return {"note": note, "revision": revision}


def delete_note(root, request_id, note_id):
    """Remove one note and answer `{revision}`. Its id is not handed out
    again."""
    _need_root(root)
    _key(request_id, "Request id")
    with _writing(root):
        requests = _load(root)
        request, _ = _note_target(requests, request_id, None)
        notes, note = _find_note(request, note_id)
        notes.remove(note)
        revision = _noted_write(root, requests, request, "note removed")
    return {"revision": revision}


def delete_request(root, request_id, revision, *, jobs_for=_no_jobs):
    """Remove one request from the ledger, at the revision it was read at,
    and answer `{id, deleted, revision, linked_jobs}`.

    Only the ledger entry goes. Every job linked to it keeps its history
    row, its `request` and `item`, and its files, so the job and gallery
    filters still find them; a request written again under the same id
    does not read those jobs as its takes (`takes`). Refused (409
    `request_has_active_jobs`, with their ids as `jobs`) while a linked job
    is queued or running: cancel them or let them finish.

    Raises LedgerError, or OSError when the file cannot be written."""
    if root is None:
        raise LedgerError(
            "this workspace has no root folder, so it holds no requests",
            status=409,
            code="no_workspace_root",
        )
    _key(request_id, "Request id")
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise LedgerError("revision must be the revision you read")
    with _writing(root):
        requests = _load(root)
        current = requests.get(request_id)
        if current is None:
            raise LedgerError(
                f"no request {request_id!r} in this workspace",
                status=404,
                code="unknown_request",
            )
        if revision != current["revision"]:
            raise LedgerError(
                f"request {request_id!r} is at revision {current['revision']}, "
                f"not {revision} - read it again before deleting it",
                status=409,
                code="stale_revision",
                current_revision=current["revision"],
            )
        jobs = takes(current, jobs_for(request_id))
        active = [j["job"] for j in jobs if j.get("status") in ("queued", "running")]
        if active:
            raise LedgerError(
                f"request {request_id!r} has {len(active)} queued or running "
                "job(s) linked to it - cancel them or let them finish first",
                status=409,
                code="request_has_active_jobs",
                jobs=active,
            )
        del requests[request_id]
        write_json_atomic(ledger_path(root), {"requests": requests})
    return {
        "id": request_id,
        "deleted": True,
        "revision": revision,
        "linked_jobs": len(jobs),
    }
