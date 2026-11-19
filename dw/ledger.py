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

Written with `write_json_atomic` under a per-directory lock (the notes.py
pattern), read without one: a rename is atomic, so a reader sees the old
file or the new one. Callers resolve and validate the workspace root; this
module joins only its own fixed file name onto it.
"""

import json
import math
import os
import re
import threading
from datetime import datetime, timezone

from .notes import NoteTooLong, clean_note
from .runs import write_json_atomic
from .security import (
    MAX_VARIABLE_VALUE_LENGTH,
    InvalidInputError,
    validate_ledger_key,
    validate_string_input,
    validate_variable_name,
)
from .workspace import REQUESTS_FILE_NAME

MAX_TITLE_LENGTH = 200
MAX_WHY_LENGTH = 200
MAX_TEXT_LENGTH = MAX_VARIABLE_VALUE_LENGTH  # prompt, goal
MAX_ITEMS = 200
MAX_REQUESTS = 100
MAX_DEPENDS_ON = 16
MAX_ESTIMATE_MINUTES = 10_000
MAX_ARGUMENTS = 64
MAX_WORKFLOW_NAME_LENGTH = 200
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_HISTORY = 100

STATES = ("open", "done", "abandoned")
AUTHORS = ("agent", "user")

# A stored workflow name: a catalog or library name such as `my-shot`,
# `models/flux-dev` or `builtin:upscale`. Stored only - nothing resolves it
# here - but held to name characters so a later stage can.
_WORKFLOW_NAME_PATTERN = r"^[\w][\w.:-]*(/[\w][\w.:-]*)*\Z"

# What an item may carry on a write. `status`, `attempts`, `result`,
# `accepted_missing` and `spent_minutes` are derived and ignored, so a plan
# read with GET can be written back as it came.
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
        "comment",
        "status",
        "attempts",
        "result",
        "accepted_missing",
        "spent_minutes",
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


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ledger_path(root):
    return os.path.join(root, REQUESTS_FILE_NAME)


def _load(root):
    """The stored ledger, `{}` when there is no file. Raises LedgerError
    when the file is there but unreadable, so a write never replaces a
    ledger it could not read."""
    path = ledger_path(root)
    try:
        with open(path) as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise LedgerError(
            f"{REQUESTS_FILE_NAME} could not be read ({e}); fix or remove it "
            "before writing",
            status=409,
            code="ledger_unreadable",
        ) from e
    requests = data.get("requests") if isinstance(data, dict) else None
    if not isinstance(requests, dict):
        return {}
    return {k: v for k, v in requests.items() if isinstance(v, dict)}


def read_requests(root):
    """Every stored request by id, `{}` when there is none or the file is
    unreadable. Takes no lock."""
    if root is None:
        return {}
    try:
        return _load(root)
    except LedgerError:
        return {}


# --- views ---------------------------------------------------------------


def _no_jobs(_request_id):
    return []


def _minutes(job):
    """How long a linked job ran, in minutes to two places, or None when it
    has not both started and finished."""
    started, finished = job.get("started_at"), job.get("finished_at")
    if started is None or finished is None:
        return None
    return round(max(finished - started, 0) / 60, 2)


def _item_status(item, jobs):
    """An item's status, derived from the jobs linked to it: dropped; done
    (any succeeded); running; queued; failed (every linked job failed or was
    cancelled); planned (none)."""
    if item.get("dropped"):
        return "dropped"
    statuses = {job.get("status") for job in jobs}
    if "succeeded" in statuses:
        return "done"
    if "running" in statuses:
        return "running"
    if "queued" in statuses:
        return "queued"
    if statuses:
        return "failed"
    return "planned"


def _result(item, jobs):
    """`(result, accepted_missing)`: the accepted job's run when it is set
    and its run still exists, else the latest-finished succeeded job whose
    run exists, else None."""
    accepted = item.get("accepted")
    missing = False
    if accepted:
        for job in jobs:
            if job.get("job") == accepted.get("job") and job.get("run_exists"):
                return {"job": job["job"], "run_id": job.get("run_id")}, False
        missing = True
    kept = [
        job
        for job in jobs
        if job.get("status") == "succeeded" and job.get("run_exists")
    ]
    if not kept:
        return None, missing
    latest = max(kept, key=lambda job: job.get("finished_at") or 0)
    return {"job": latest["job"], "run_id": latest.get("run_id")}, missing


def _jobs_by_item(jobs):
    grouped = {}
    for job in jobs:
        grouped.setdefault(job.get("item"), []).append(job)
    for group in grouped.values():
        group.sort(key=lambda job: job.get("created_at") or 0)
    return grouped


def item_view(item, jobs):
    """One item as read, with what its linked `jobs` derive: `status`,
    `attempts` (every linked job, oldest first, as `{job, status, note,
    minutes}` - `note` only when set, `minutes` null until it has run),
    `result`, `accepted_missing` (only when true) and `spent_minutes`."""
    attempts = [
        {
            "job": job.get("job"),
            "status": job.get("status"),
            **({"note": job["note"]} if job.get("note") else {}),
            "minutes": _minutes(job),
        }
        for job in jobs
    ]
    result, missing = _result(item, jobs)
    return {
        **item,
        "status": _item_status(item, jobs),
        "attempts": attempts,
        "result": result,
        **({"accepted_missing": True} if missing else {}),
        "spent_minutes": _spent(jobs),
    }


def _spent(jobs):
    return round(sum(_minutes(job) or 0 for job in jobs), 2)


def _remaining(request, grouped):
    """What is left to do: the estimates of the items not done and not
    dropped."""
    total = 0
    for item in request.get("items", []):
        if _item_status(item, grouped.get(item.get("key"), [])) in ("done", "dropped"):
            continue
        total += item.get("estimate_minutes") or 0
    return round(total, 2)


def request_view(request, jobs=()):
    """A stored request as read: every item with its derived fields
    (`item_view`) and the request's `spent_minutes` and `remaining_minutes`.
    `jobs` are the linked jobs (`{job, item, status, run_id, run_exists,
    note, created_at, started_at, finished_at}`)."""
    grouped = _jobs_by_item(jobs)
    view = dict(request)
    view["items"] = [
        item_view(item, grouped.get(item.get("key"), []))
        for item in request.get("items", [])
    ]
    view["spent_minutes"] = _spent(jobs)
    view["remaining_minutes"] = _remaining(request, grouped)
    return view


def summary_view(request, jobs=()):
    """One row of the list: what a request is, without its items."""
    grouped = _jobs_by_item(jobs)
    counts = {}
    for item in request.get("items", []):
        status = _item_status(item, grouped.get(item.get("key"), []))
        counts[status] = counts.get(status, 0) + 1
    history = request.get("history") or []
    return {
        "id": request.get("id"),
        "title": request.get("title"),
        "state": request.get("state"),
        "revision": request.get("revision"),
        "items": len(request.get("items", [])),
        "status_counts": counts,
        "spent_minutes": _spent(jobs),
        "remaining_minutes": _remaining(request, grouped),
        "updated_at": history[-1].get("at") if history else None,
    }


def list_requests(root, jobs_for=_no_jobs):
    """The list view, newest change first. `jobs_for(request_id)` lists the
    jobs linked to a request."""
    rows = [
        summary_view(r, jobs_for(r.get("id"))) for r in read_requests(root).values()
    ]
    rows.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
    return rows


def get_request(root, request_id, jobs_for=_no_jobs):
    """One request's view, or None."""
    validate_ledger_key(request_id, "Request id")
    request = read_requests(root).get(request_id)
    if request is None:
        return None
    return request_view(request, jobs_for(request_id))


def link_problem(root, request_id, item_key):
    """The LedgerError that says why a job cannot be linked to
    `request_id`/`item_key` in the workspace at `root`, or None when it can:
    `unknown_request` (no such request, or no root), `request_closed`,
    `unknown_item`, `item_dropped`. Both keys are checked as keys first
    (InvalidInputError otherwise). Takes no lock."""
    validate_ledger_key(request_id, "Request")
    validate_ledger_key(item_key, "Item")
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
    if not math.isfinite(value) or not 0 <= value <= MAX_ESTIMATE_MINUTES:
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


def _comment(value, key, existing, by, at):
    """The comment to store. A user writes it; an agent's write keeps what
    is there (omitting it, or sending it back unchanged)."""
    if by != "user":
        if value is _MISSING or value == existing:
            return existing
        if (
            isinstance(value, dict)
            and existing
            and value.get("text") == existing.get("text")
        ):
            return existing
        raise LedgerError(
            f"item {key!r}: the comment is the user's - an agent's write omits it"
        )
    if value is _MISSING:
        return existing
    if isinstance(value, dict):
        unknown = set(value) - {"text", "at"}
        if unknown:
            raise LedgerError(
                f"item {key!r}: comment carries unknown fields {sorted(unknown)}"
            )
        value = value.get("text")
    try:
        text = clean_note(value)
    except (NoteTooLong, TypeError) as e:
        raise LedgerError(f"item {key!r} comment: {e}") from e
    if text is None:
        return None
    _text(text, f"item {key!r} comment", len(text) or 1)
    if existing and existing.get("text") == text:
        return existing
    return {"text": text, "at": at}


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


_MISSING = object()


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
    ("title", "goal", "items", "revision", "why", "prompt", "state")
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
    if len(raw_items) > MAX_ITEMS:
        raise LedgerError(f"at most {MAX_ITEMS} items, got {len(raw_items)}")
    return {
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
    if current is None:
        if len(requests) >= MAX_REQUESTS:
            raise LedgerError(
                f"a workspace holds at most {MAX_REQUESTS} requests",
                status=409,
                code="ledger_full",
            )
    elif write["prompt"] is not None and write["prompt"] != current.get("prompt"):
        raise LedgerError(
            "prompt is write-once: it is the ask that started the request "
            "and was set when it was created",
            status=422,
            code="prompt_is_write_once",
        )
    return current


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


def _item(raw, key, previous, linked, by, at):
    """One live item as stored."""
    return {
        "key": key,
        "title": _text(
            raw.get("title"), f"item {key!r} title", MAX_TITLE_LENGTH, one_line=True
        ),
        "depends_on": _depends_on(raw, key),
        "estimate_minutes": _estimate(raw.get("estimate_minutes"), key),
        "workflow": _workflow(raw.get("workflow"), key),
        "arguments": _arguments(raw.get("arguments"), key),
        "dropped": False,
        "accepted": _accepted(
            raw.get("accepted"), key, previous.get("accepted"), linked
        ),
        "comment": _comment(
            raw.get("comment", _MISSING), key, previous.get("comment"), by, at
        ),
    }


def _merge_items(raw_items, current, request_id, linked_jobs, by, at):
    """The plan's items in its order, then every stored item it no longer
    lists that jobs link to, as dropped; one no job links to is deleted."""
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
        live.append(_item(raw, key, previous, linked_jobs(request_id, key), by, at))
    listed = {item["key"] for item in live}
    dropped = [
        {**item, "dropped": True}
        for key, item in stored.items()
        if key not in listed and linked_jobs(request_id, key)
    ]
    items = live + dropped
    if len(items) > MAX_ITEMS:
        raise LedgerError(
            f"at most {MAX_ITEMS} items, counting dropped ones; got {len(items)}"
        )
    _check_dependencies(items)
    return items


def _history(current, write, revision, by, at):
    entry = {"revision": revision, "at": at, "by": by, "why": write["why"]}
    if current is None or write["goal"] != current.get("goal"):
        entry["goal"] = write["goal"]
    history = list((current or {}).get("history", [])) + [entry]
    if len(history) > MAX_HISTORY:
        history = history[:1] + history[-(MAX_HISTORY - 1) :]
    return history


def put_request(
    root, request_id, body, *, by, linked_jobs=_no_links, jobs_for=_no_jobs
):
    """Write one request's whole plan and return its view.

    `body` holds title, goal, items, revision and why, and optionally
    prompt and state. `linked_jobs(request_id, key)` lists the jobs linked
    to an item as `{job, status, run_id}`; `accepted` must name one of them
    that succeeded. `jobs_for(request_id)` lists every job linked to the request, for
    the returned view's derived fields.

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

    with _lock(root):
        requests = _load(root)
        current = _check_against(requests, request_id, write)
        at = _now()
        revision = write["revision"] + 1
        request = {
            "id": request_id,
            "title": write["title"],
            "state": write["state"] or (current or {}).get("state") or "open",
            "revision": revision,
            "prompt": current.get("prompt") if current else write["prompt"],
            "goal": write["goal"],
            "items": _merge_items(
                write["items"], current, request_id, linked_jobs, by, at
            ),
            "history": _history(current, write, revision, by, at),
        }
        requests[request_id] = request
        document = {"requests": requests}
        size = len(json.dumps(document, indent=2).encode("utf-8"))
        if size > MAX_FILE_BYTES:
            raise LedgerError(
                f"{REQUESTS_FILE_NAME} would be {size} bytes, past its "
                f"{MAX_FILE_BYTES}-byte cap - shorten the plan's text "
                "or arguments",
                status=413,
                code="ledger_too_large",
            )
        write_json_atomic(ledger_path(root), document)
    return request_view(request, jobs_for(request_id))
