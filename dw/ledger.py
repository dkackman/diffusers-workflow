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
import math
import os
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from filelock import FileLock, Timeout

from .notes import NoteTooLong, clean_note
from .runs import write_json_atomic
from .security import (
    MAX_VARIABLE_VALUE_LENGTH,
    InvalidInputError,
    validate_ledger_key,
    validate_string_input,
    validate_variable_name,
)
from .workspace import REQUESTS_FILE_NAME, REQUESTS_LOCK_NAME
from .ledger_views import (  # noqa: F401 - COVER_* and the views are read as ledger.*
    COVER_SCAN,
    COVER_SIZE,
    _live_keys,
    current_prompts,
    is_revised,
    item_view,
    request_view,
    summary_view,
)

MAX_TITLE_LENGTH = 200
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
MAX_DEPENDS_ON = 16
MAX_ESTIMATE_MINUTES = 10_000
MAX_ARGUMENTS = 64
MAX_WORKFLOW_NAME_LENGTH = 200
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_HISTORY = 100
MAX_NOTES = 20  # live notes of one target: an item, or the request itself
MAX_NOTE_REFS = 16  # references in one note
MIN_REVIEW_SECONDS = 60  # a review hold's shortest span
MAX_REVIEW_HOURS = 24  # and its longest
# Seconds a write waits for another server's write to the same file
LOCK_TIMEOUT = 10

STATES = ("open", "done", "abandoned")
AUTHORS = ("agent", "user")

# A stored workflow name: a catalog or library name such as `my-shot`,
# `models/flux-dev` or `builtin:upscale`. Stored only - nothing resolves it
# here - but held to name characters so a later stage can.
_WORKFLOW_NAME_PATTERN = r"^[\w][\w.:-]*(/[\w][\w.:-]*)*\Z"

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


class LedgerError(ValueError):
    """A write the ledger refuses. `status` is the HTTP status the server
    answers with and `code` a stable name a client can branch on."""

    def __init__(self, message, *, status=400, code="invalid_request", **extra):
        super().__init__(message)
        self.status = status
        self.code = code
        self.extra = extra

    def detail(self):
        return {"code": self.code, "message": str(self), **self.extra}


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


def ledger_path(root):
    return os.path.join(root, REQUESTS_FILE_NAME)


def _unreadable(why):
    return LedgerError(
        f"{REQUESTS_FILE_NAME} could not be read ({why}); fix or remove it "
        "before writing",
        status=409,
        code="ledger_unreadable",
    )


def _request_fault(request_id, request):
    """What is wrong with one stored request's shape, or None: the fields
    the views read without checking."""
    if not isinstance(request, dict):
        return "is not an object"
    if request.get("id") != request_id:
        return "is stored under another id"
    if request.get("state") not in STATES:
        return "has no known state"
    revision = request.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        return "has no revision"
    items = request.get("items")
    if not isinstance(items, list) or not all(
        isinstance(item, dict) and isinstance(item.get("key"), str) for item in items
    ):
        return "has items that are not a list of keyed objects"
    history = request.get("history", [])
    if not isinstance(history, list) or not all(isinstance(h, dict) for h in history):
        return "has a history that is not a list of objects"
    return None


def _load(root):
    """The stored ledger, `{}` when there is no file. Raises LedgerError
    (`ledger_unreadable`) when the file is there but is not a ledger - not
    readable, not JSON, or JSON of another shape - so a write never
    replaces a file it could not read. The message never names the folder."""
    path = ledger_path(root)
    try:
        with open(path) as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        # An OSError's text carries the file's absolute path; its strerror
        # does not
        why = (e.strerror or type(e).__name__) if isinstance(e, OSError) else e
        raise _unreadable(why) from e
    requests = data.get("requests") if isinstance(data, dict) else None
    if not isinstance(requests, dict):
        raise _unreadable('it is not {"requests": {...}}')
    for request_id, request in requests.items():
        fault = _request_fault(request_id, request)
        if fault:
            raise _unreadable(f"request {str(request_id)[:64]!r} {fault}")
        _upgrade(request)
    return requests


def _upgrade(request):
    """Bring a stored request to the notes shape, in place, and return it.
    A stored `comment {text, at}` (the request's, then each item's in plan
    order) becomes one note `{id, text, created, at}` with `created = at`;
    `next_note` counts past the ids handed out; `agent_wrote_at` is filled
    from the latest agent entry in the history when it was never stored.
    Idempotent, and `comment` is never written again."""
    request.setdefault("next_note", 1)
    for owner in (request, *request["items"]):
        comment = owner.pop("comment", None)
        owner.setdefault("notes", [])
        if isinstance(comment, dict) and comment.get("text"):
            at = comment.get("at")
            owner["notes"].append(
                {
                    "id": f"n{request['next_note']}",
                    "text": comment["text"],
                    "created": at,
                    "at": at,
                }
            )
            request["next_note"] += 1
    if "agent_wrote_at" not in request:
        agent = [
            h.get("at") for h in request.get("history") or [] if h.get("by") == "agent"
        ]
        request["agent_wrote_at"] = agent[-1] if agent else None
    return request


def read_requests(root):
    """Every stored request by id, `{}` when there is none or the file is
    unreadable. Takes no lock."""
    if root is None:
        return {}
    try:
        return _load(root)
    except LedgerError:
        return {}


def unreadable(root):
    """The `ledger_unreadable` LedgerError for a file that is there but is
    not a ledger, or None."""
    if root is None:
        return None
    try:
        _load(root)
    except LedgerError as e:
        return e
    return None


# --- views (dw/ledger_views.py derives them) -----------------------------


def _no_jobs(_request_id):
    return []


def list_requests(root, jobs_for=_no_jobs, prompts=None):
    """The list view, newest change first. `jobs_for(request_id)` lists the
    jobs linked to a request; `prompts` is the `name -> digest` callable the
    revised rule reads (`ledger_views.is_revised`), None for rule 1 alone."""
    rows = [
        summary_view(r, jobs_for(r.get("id")), prompts)
        for r in read_requests(root).values()
    ]
    rows.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
    return rows


def listing(root, jobs_for=_no_jobs, prompts=None):
    """What the list route answers: `requests` (`list_requests`), and
    `unreadable` - why - when the file is there but is not a ledger."""
    problem = unreadable(root)
    if problem is not None:
        return {"requests": [], "unreadable": str(problem)}
    return {"requests": list_requests(root, jobs_for, prompts)}


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


# --- validation ----------------------------------------------------------


def _text(value, what, max_length, *, one_line=False, allow_empty=False):
    if not isinstance(value, str):
        raise LedgerError(f"{what} must be text")
    try:
        validate_string_input(value, max_length=max_length, allow_empty=allow_empty)
    except InvalidInputError as e:
        raise LedgerError(f"{what}: {e}") from e
    if one_line and ("\n" in value or "\r" in value):
        raise LedgerError(f"{what} must be one line")
    if not allow_empty and not value.strip():
        raise LedgerError(f"{what} cannot be blank")
    return value


def _key(value, what):
    try:
        return validate_ledger_key(value, what)
    except InvalidInputError as e:
        raise LedgerError(str(e)) from e


def _estimate(value, key):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"item {key!r}: estimate_minutes must be a number or null")
    # The range first: isfinite raises on an integer too large for a float
    if not 0 <= value <= MAX_ESTIMATE_MINUTES or not math.isfinite(value):
        raise LedgerError(
            f"item {key!r}: estimate_minutes must be between 0 and "
            f"{MAX_ESTIMATE_MINUTES}, got {value}"
        )
    return value


def _workflow(value, key):
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise LedgerError(f"item {key!r}: workflow must be a name or null")
    if len(value) > MAX_WORKFLOW_NAME_LENGTH:
        raise LedgerError(
            f"item {key!r}: workflow name too long: {len(value)} > "
            f"{MAX_WORKFLOW_NAME_LENGTH}"
        )
    if not re.match(_WORKFLOW_NAME_PATTERN, value) or ".." in value.split("/"):
        raise LedgerError(
            f"item {key!r}: invalid workflow name {value!r} - a catalog or "
            "library name such as 'my-shot' or 'models/flux-dev'"
        )
    return value


def _arguments(value, key):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LedgerError(f"item {key!r}: arguments must be an object")
    if len(value) > MAX_ARGUMENTS:
        raise LedgerError(
            f"item {key!r}: at most {MAX_ARGUMENTS} arguments, got {len(value)}"
        )
    out = {}
    for name, argument in value.items():
        try:
            validate_variable_name(name)
        except InvalidInputError as e:
            raise LedgerError(f"item {key!r}: {e}") from e
        if isinstance(argument, str):
            _text(
                argument,
                f"item {key!r} argument {name!r}",
                MAX_VARIABLE_VALUE_LENGTH,
                allow_empty=True,
            )
        elif isinstance(argument, float) and not math.isfinite(argument):
            raise LedgerError(f"item {key!r} argument {name!r} must be finite")
        elif argument is not None and not isinstance(argument, (bool, int, float)):
            raise LedgerError(
                f"item {key!r} argument {name!r}: only a string, number, "
                "boolean or null is stored, not "
                f"{'a list' if isinstance(argument, list) else 'an object'}"
            )
        out[name] = argument
    return out


def _accepted(value, key, existing, linked):
    if value is None:
        return None
    if isinstance(value, str):
        value = {"job": value}
    if not isinstance(value, dict) or set(value) - {"job", "run_id"}:
        raise LedgerError(
            f"item {key!r}: accepted must be {{job}} (or a job id) or null"
        )
    if existing and value.get("job") == existing.get("job"):
        return existing
    job = value.get("job")
    for entry in linked:
        if entry.get("job") == job:
            if entry.get("status") != "succeeded":
                raise LedgerError(
                    f"item {key!r}: job {job!r} has not succeeded, so it "
                    "cannot be accepted"
                )
            return {"job": job, "run_id": entry.get("run_id")}
    raise LedgerError(
        f"item {key!r}: job {job!r} is not linked to this request and item, "
        "so it cannot be accepted"
    )


def _check_dependencies(items):
    keys = {item["key"] for item in items}
    graph = {}
    for item in items:
        deps = item["depends_on"]
        for dep in deps:
            if dep == item["key"]:
                raise LedgerError(f"item {dep!r} depends on itself")
            if dep not in keys:
                raise LedgerError(
                    f"item {item['key']!r} depends on {dep!r}, which is not an "
                    "item of this request"
                )
        graph[item["key"]] = deps
    # Iterative three-colour DFS: a grey node met again closes a cycle
    colour = dict.fromkeys(graph, 0)
    for start in graph:
        if colour[start]:
            continue
        stack = [(start, iter(graph[start]))]
        colour[start] = 1
        path = [start]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                colour[node] = 2
                stack.pop()
                path.pop()
            elif colour[child] == 1:
                cycle = path[path.index(child) :] + [child]
                raise LedgerError("depends_on has a cycle: " + " -> ".join(cycle))
            elif colour[child] == 0:
                colour[child] = 1
                stack.append((child, iter(graph[child])))
                path.append(child)


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
    )
)


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
    }


def _check_against(requests, request_id, write):
    """The stored request a write replaces (None to create), refusing a
    stale revision, a full ledger and a changed prompt."""
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
    _check_room(requests, current, write["state"])
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


def _check_room(requests, current, state):
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
            f"a workspace holds at most {MAX_REQUESTS} open requests - close "
            "one (state done or abandoned) or delete one to make room",
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


def _depends_on(raw, key):
    depends_on = raw.get("depends_on") or []
    if not isinstance(depends_on, list):
        raise LedgerError(f"item {key!r}: depends_on must be a list of keys")
    if len(depends_on) > MAX_DEPENDS_ON:
        raise LedgerError(
            f"item {key!r}: depends on at most {MAX_DEPENDS_ON} items, "
            f"got {len(depends_on)}"
        )
    keys = [_key(dep, f"item {key!r} depends_on key") for dep in depends_on]
    return list(dict.fromkeys(keys))


def _review_until(value, key, previous, at):
    """`(review_until, review_set_at)` as stored, in UTC. A value equal to the
    stored one keeps its stamp, so a plan written back unchanged does not
    restart the hold; a new one falls MIN_REVIEW_SECONDS..MAX_REVIEW_HOURS out."""
    if value is None:
        return None, None
    try:
        when = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        when = None
    if when is None or when.utcoffset() is None:
        raise _review_error(key, "invalid_review_until", "an ISO time with an offset")
    when = when.astimezone(timezone.utc).replace(microsecond=0)
    if when.isoformat() == previous.get("review_until"):
        return previous["review_until"], previous.get("review_set_at")
    ahead = (when - datetime.fromisoformat(at)).total_seconds()
    if ahead < MIN_REVIEW_SECONDS:
        bound = f"at least {MIN_REVIEW_SECONDS} seconds from now"
        raise _review_error(key, "review_too_soon", bound)
    if ahead > MAX_REVIEW_HOURS * 3600:
        raise _review_error(key, "review_too_far", f"within {MAX_REVIEW_HOURS} hours")
    return when.isoformat(), at


def _review_error(key, code, must_be):
    return LedgerError(
        f"item {key!r} review_until must be {must_be}", status=422, code=code
    )


def _item(raw, key, previous, linked, by, at, prompts=None):
    """One live item as stored, with the server's stamps: `changed_at`, set
    when the item is created or its `workflow` or `arguments` differ from the
    stored ones and kept otherwise (so a GET->PUT round trip moves nothing);
    and `accepted_basis` (`{at, prompts}`), recorded when the write names an
    `accepted` take while the stored item reads revised and the write
    leaves its plan as it was - the item then reads done until the plan
    moves again, or a prompt it references gets text other than the digest
    `prompts` records (`ledger_views.is_revised`; `prompts` the `name ->
    digest` callable)."""
    until, set_at = _review_until(raw.get("review_until"), key, previous, at)
    workflow = _workflow(raw.get("workflow"), key)
    arguments = _arguments(raw.get("arguments"), key)
    accepted = _accepted(raw.get("accepted"), key, previous.get("accepted"), linked)
    unchanged = bool(previous) and (
        workflow == previous.get("workflow")
        and arguments == (previous.get("arguments") or {})
    )
    changed_at = previous.get("changed_at") if unchanged else at
    basis = previous.get("accepted_basis")
    if accepted is not None and unchanged and is_revised(previous, linked, prompts):
        basis = {"at": at}
        held = current_prompts(previous, prompts)
        if held:
            basis["prompts"] = held
    return {
        "key": key,
        "title": _text(
            raw.get("title"), f"item {key!r} title", MAX_TITLE_LENGTH, one_line=True
        ),
        "depends_on": _depends_on(raw, key),
        "estimate_minutes": _estimate(raw.get("estimate_minutes"), key),
        "workflow": workflow,
        "arguments": arguments,
        "dropped": False,
        "accepted": accepted,
        "notes": list(previous.get("notes") or []),
        "review_until": until,
        "review_set_at": set_at,
        "changed_at": changed_at,
        **({"accepted_basis": basis} if isinstance(basis, dict) else {}),
    }


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
    raw_items, current, request_id, linked_jobs, by, at, purge=(), prompts=None
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
            _item(raw, key, previous, linked_jobs(request_id, key), by, at, prompts)
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
    removed for good even though jobs link to them). `linked_jobs(request_id, key)` lists the jobs linked
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
        current = _check_against(requests, request_id, write)
        at = _now()
        revision = write["revision"] + 1
        state = write["state"] or (current or {}).get("state") or "open"
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
                linked_jobs,
                by,
                at,
                write["purge"],
                prompts,
            ),
            "history": _history(current, write, state, revision, by, at),
        }
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
    return request_view(request, jobs_for(request_id), prompts=prompts)


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
    reads those jobs as its own. Refused (409 `request_has_active_jobs`,
    with their ids as `jobs`) while a linked job is queued or running:
    cancel them or let them finish.

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
        jobs = jobs_for(request_id)
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
