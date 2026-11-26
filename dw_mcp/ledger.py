"""The requests ledger: the plan an agent keeps for what the user asked.

A request lives in its workspace's `requests.json` on the server
(`dw/ledger.py`), read and written whole through
`/api/workspaces/<name>/requests/<id>`. A write names the revision it was
read at, so two writers cannot silently overwrite each other; the server's
refusal says which revision is current.
"""

import time
from datetime import datetime, timezone

from dw_mcp.client import DwApiError, api_path


def _workspace(client, workspace):
    return workspace or client.workspace


def request_filter(request):
    """A listing's `request` filter as query params: `"<request-id>"`, or
    `"<request-id>/<item-key>"` for one item - the way `run_workflow` names
    a link. {} for none. The server owns the key rule."""
    if request is None or request == "":
        return {}
    parts = request.split("/") if isinstance(request, str) else []
    if len(parts) not in (1, 2):
        raise DwApiError(
            '`request` must be "<request-id>" or "<request-id>/<item-key>" '
            f"(got {request!r})."
        )
    return dict(zip(("request", "item"), parts))


def get_request(client, request=None, workspace=None, headers=None):
    """One request's plan, or the workspace's list when `request` is None.
    `headers` go on the list read only."""
    name = _workspace(client, workspace)
    if request is None:
        extra = {"headers": headers} if headers else {}
        return client.get_json(api_path("api", "workspaces", name, "requests"), **extra)
    return client.get_json(api_path("api", "workspaces", name, "requests", request))


def set_request(
    client,
    request,
    title,
    goal,
    items,
    revision,
    why,
    prompt=None,
    state=None,
    workspace=None,
    purge=None,
    propose_close=...,
):
    """Write a request's whole plan, as the agent. The user's notes are
    never sent: the server keeps what is there. `propose_close` left as
    `...` sends nothing (the stored proposal is kept); `{state, reason}`
    proposes a close, None withdraws a pending one."""
    payload = {
        "title": title,
        "goal": goal,
        "items": items,
        "revision": revision,
        "why": why,
        "by": "agent",
    }
    if prompt is not None:
        payload["prompt"] = prompt
    if state is not None:
        payload["state"] = state
    if purge:
        payload["purge"] = purge
    if propose_close is not ...:
        payload["close_proposal"] = propose_close
    return client.put_json(
        api_path(
            "api", "workspaces", _workspace(client, workspace), "requests", request
        ),
        payload,
    )


def delete_request(client, request, revision, workspace=None):
    """Remove one request from the ledger at the revision read; its linked
    jobs keep their rows, their link and their files."""
    return client.delete_json(
        api_path(
            "api", "workspaces", _workspace(client, workspace), "requests", request
        ),
        params={"revision": revision},
    )


# --- notes waiting on the agent ---------------------------------------------
#
# A user's notes on a request or one of its items are for the
# agent. The rules here decide which notes still wait and what the notes
# channel (`dw-mcp --notes-channel`, served by dw_mcp/server.py) says about
# them. Nothing here imports the MCP SDK or sleeps.

# Marks the channel's request-list poll, so the server can tell a session
# is listening. A hint for the page, never an authority.
CHANNEL_HEADER = {"X-DW-Notes-Channel": "1"}

# Polls between asks for GET /api/requests on a server that answered 404:
# about five minutes at the default interval, so a server updated while a
# channel runs is found again without restarting the session
ROUTE_RETRY_POLLS = 30
# The ledger's own cap on a note (dw/notes.py); a note is quoted whole
# up to it
NOTE_QUOTE_CAP = 4000
# What one note gets in the digest of those waiting at connect
DIGEST_QUOTE_CAP = 300
DIGEST_MAX_NOTES = 20
DEFAULT_POLL_SECONDS = 10.0
MIN_POLL_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 300.0
# After the handshake, before the first event: a client attaches its listener
# for channel notifications once `initialize` has answered, and drops what
# arrives before that
CONNECT_GRACE_SECONDS = 5.0


def _instant(at):
    """An ISO timestamp as an aware datetime, or None. The ledger writes
    every `at` with its offset; one without is read as UTC."""
    try:
        when = datetime.fromisoformat(at)
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def note_waiting(note, request):
    """Whether a note still waits on the agent: on an open request, saved
    after the agent's last write (`agent_wrote_at`; none yet leaves it
    waiting). The ledger stamps whole seconds, so a note saved in the same
    second as the write still waits: better announced twice than missed. The UI's `noteWaiting` (ui/src/lib/request/noteState.ts) is
    the same rule; tests/fixtures/note_waiting_cases.json holds the two
    equal."""
    if not note or request.get("state") != "open":
        return False
    saved = _instant(note.get("at"))
    wrote = _instant(request.get("agent_wrote_at"))
    if saved is None or wrote is None:
        return True
    return saved >= wrote


def _read_refs(note, request):
    """A note's references with what the channel text needs from the plan:
    the referenced item's `title`, its live-ness (`dropped`) and `takes`, the
    count of its attempts. The item is looked up by `ref.item`, not by the
    note's own target."""
    items = {item.get("key"): item for item in request.get("items") or []}
    out = []
    for ref in note.get("refs") or []:
        item = items.get(ref.get("item")) or {}
        out.append(
            {
                **ref,
                "title": item.get("title") or "",
                "dropped": not item or bool(item.get("dropped")),
                "takes": len(item.get("attempts") or []),
            }
        )
    return out


def waiting_notes(request):
    """The notes of one request that wait on the agent: the request's own,
    then each live item's in plan order (notes in stored order), as `{request,
    revision, item, title, note, text, created, at, refs}`. `item` is "" for
    the request's own notes; `note` is the note's id; each ref carries its
    item's `title`, `dropped` and `takes` besides what the view gave."""
    targets = [("", request.get("title"), request.get("notes"))]
    targets += [
        (item.get("key"), item.get("title"), item.get("notes"))
        for item in request.get("items") or []
        if not item.get("dropped")
    ]
    return [
        {
            "request": request.get("id"),
            "revision": request.get("revision"),
            "item": key,
            "note": note.get("id"),
            "title": title,
            "text": note.get("text") or "",
            "created": note.get("created") or note.get("at"),
            "at": note.get("at"),
            "refs": _read_refs(note, request),
        }
        for key, title, notes in targets
        for note in notes or []
        if note_waiting(note, request)
    ]


def _saved(at):
    """When a note was saved, for a sentence: UTC to the minute."""
    when = _instant(at)
    if when is None:
        return str(at)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _target(note):
    """Which plan a note is on: its request, and its item when it has one."""
    if note["item"]:
        title = f" ({note['title']})" if note.get("title") else ""
        return f'request "{note["request"]}", item "{note["item"]}"{title}'
    return f'request "{note["request"]}"'


def _when(note):
    """`saved <created>`, then `, edited <at>` once `at` has moved."""
    created = note.get("created") or note["at"]
    text = f"saved {_saved(created)}"
    if note["at"] != created:
        text += f", edited {_saved(note['at'])}"
    return text


def _ref_line(ref):
    """One reference as a line: the item, then the take and file it names,
    then why it no longer holds when it is `gone`: its item was dropped, its
    job is no longer one of the item's takes (`attempt` null), or that job's
    run is gone."""
    title = f" ({ref['title']})" if ref.get("title") else ""
    line = f'- item "{ref["item"]}"{title}'
    if "job" not in ref:
        if not (ref.get("gone") and ref.get("dropped")):
            line += ", not rendered yet"
    else:
        if ref.get("attempt") is not None:
            line += f", take {ref['attempt']} of {ref.get('takes', 0)}"
            line += f" (job {ref['job']})"
        else:
            line += f", job {ref['job']}"
        if ref.get("output"):
            line += f', file "{ref["output"]}"'
            if ref.get("kind"):
                line += f" ({ref['kind']})"
    if ref.get("gone"):
        if ref.get("dropped"):
            line += " (dropped from the plan)"
        elif "job" in ref and ref.get("attempt") is None:
            line += " (no longer a take of this item)"
        elif "job" in ref:
            line += " (that job's run no longer exists)"
    return line


def _read_call(note, workspace):
    return f'get_request("{note["request"]}", workspace="{workspace}")'


def note_event(note, workspace):
    """The channel event for one note saved while the session is open: its
    text verbatim, what it is on, and what to do. `meta` keys are
    identifiers and its values strings - the channel contract drops any
    other key."""
    text = note["text"]
    clipped = len(text) > NOTE_QUOTE_CAP
    refs = note.get("refs") or []
    head = f"Note from the user on {_target(note)}, {_when(note)}"
    if refs:
        head += ", about:\n" + "\n".join(_ref_line(ref) for ref in refs)
    else:
        head += ":"
    lines = [head, text[:NOTE_QUOTE_CAP]]
    if clipped:
        lines.append(
            f"(Clipped at {NOTE_QUOTE_CAP} characters of {len(text)}; "
            "get_request has the whole note.)"
        )
    lines.append(
        f"Read the request with {_read_call(note, workspace)} and act on the "
        'note; say what you did in the "why" of your next set_request, '
        'naming the note id you answer ("n3: rerolled shot3 as take 3").'
    )
    meta = {
        "kind": "note",
        "workspace": str(workspace),
        "request": str(note["request"]),
        "revision": str(note["revision"]),
        "note": str(note["note"]),
        "created": str(note.get("created") or note["at"]),
    }
    if note["item"]:
        meta["item"] = str(note["item"])
    if refs:
        meta["items"] = ",".join(dict.fromkeys(str(r["item"]) for r in refs))
    return {"content": "\n\n".join(lines), "meta": meta}


def waiting_event(notes, workspace):
    """The one event for the notes already waiting when the channel
    connected. They were written before this session, so the agent is told
    to check with the user before spending on them."""
    count = len(notes)
    lines = []
    for note in notes[:DIGEST_MAX_NOTES]:
        text = " ".join(note["text"].split())
        if len(text) > DIGEST_QUOTE_CAP:
            text = text[:DIGEST_QUOTE_CAP] + " ... (clipped)"
        lines.append(f"- {_target(note)}, {_when(note)}: {text}")
        lines += [f"  {_ref_line(ref)}" for ref in note.get("refs") or []]
    if count > DIGEST_MAX_NOTES:
        lines.append(f"- and {count - DIGEST_MAX_NOTES} more")
    one = count == 1
    head = (
        f"{count} note{'' if one else 's'} from the user "
        f"{'was' if one else 'were'} already waiting on dw requests in "
        f'workspace "{workspace}" when this session connected:'
    )
    tail = (
        "These were saved before this session, so they may be stale. Read "
        f'each request with get_request("<request>", workspace="{workspace}"), '
        "tell the user what is waiting, and confirm with them before spending "
        'GPU time on one; say what you did in the "why" of your next '
        'set_request, naming the note ids you answer ("n3: rerolled shot3").'
    )
    meta = {"kind": "notes_waiting", "workspace": str(workspace), "count": str(count)}
    return {"content": "\n\n".join([head, "\n".join(lines), tail]), "meta": meta}


def due_reviews(request):
    """The held items of one open request whose deadline has passed with no
    note on them and that are ready to run: live items in plan order, as
    `{request, revision, item, title, until, estimate_minutes}`."""
    return [
        {
            "request": request.get("id"),
            "revision": request.get("revision"),
            "item": item.get("key"),
            "title": item.get("title") or "",
            "until": item["review"]["until"],
            "estimate_minutes": item.get("estimate_minutes"),
        }
        for item in request.get("items") or []
        if not item.get("dropped")
        and (item.get("review") or {}).get("state") == "due"
        and item.get("ready")
    ]


def review_event(due, workspace, stale=False):
    """The channel event for one request's due items (`due_reviews`). Fresh,
    it tells the agent to run what is still ready under the cost it quoted;
    `stale` (the hold was set before this session connected) tells it to
    confirm with the user first. `meta` values are strings."""
    request = due[0]["request"]
    if stale:
        head = (
            f'Review time is up on request "{request}", on a hold set before '
            "this session connected. Held items:"
        )
        tail = (
            "These holds were set before this session, so the user's go-ahead "
            "may be stale. Tell the user what is waiting and confirm before "
            "spending GPU time."
        )
    else:
        head = (
            f'Review time is up on request "{request}" with no note from the '
            "user. Held items now due:"
        )
        tail = (
            f'Read the request with get_request("{request}", workspace="{workspace}"). '
            "Run only the items still ready and without a new note, each with "
            f'run_workflow(request="{request}/<key>", acknowledged_cost=<the '
            "plan you quoted>). If a run is refused for a changed plan, or "
            "validate_workflow now quotes a figure well off what you told the "
            "user, do not run it and do not hold again: clear its "
            "review_until and put the new figure in the why."
        )
    lines = []
    for item in due:
        minutes = item.get("estimate_minutes")
        about = f" (about {minutes:g} min)" if minutes else ""
        lines.append(f"- {item['item']}: {item['title']}{about}")
    earliest = min(
        due,
        key=lambda d: _instant(d["until"]) or datetime.max.replace(tzinfo=timezone.utc),
    )
    meta = {
        "kind": "review_due",
        "workspace": str(workspace),
        "request": str(request),
        "revision": str(due[0]["revision"]),
        "items": ",".join(str(d["item"]) for d in due),
        "until": str(earliest["until"]),
    }
    if stale:
        meta["stale"] = "1"
    return {"content": "\n\n".join([head, "\n".join(lines), tail]), "meta": meta}


def _holding_left(request):
    """Seconds until the earliest still-`holding` item of `request` is due,
    or None when none is."""
    left = [
        item["review"]["seconds_left"]
        for item in request.get("items") or []
        if not item.get("dropped")
        and (item.get("review") or {}).get("state") == "holding"
    ]
    return min(left) if left else None


def retry_delay(interval, failures):
    """Seconds until the next poll: the interval, doubled for each failed
    poll in a row, up to MAX_BACKOFF_SECONDS."""
    if failures <= 0:
        return interval
    return min(interval * 2 ** min(failures, 16), max(interval, MAX_BACKOFF_SECONDS))


class NoteWatch:
    """Which waiting notes and due review holds one channel process has yet
    to announce.

    `poll` reads a request list - one call when nothing changed - and each
    open request whose stamp (revision, `updated_at`, count of ready items)
    moved since it was last read, or whose earliest hold has run out on the
    monotonic `clock` since (a deadline passing changes nothing the list
    shows). With `everywhere`, the list is the server's (`/api/requests`:
    every workspace's open requests, one read), so a note reaches the
    session whichever workspace it works in; a server without that route
    is read one workspace at a time, the client's, and asked again every
    ROUTE_RETRY_POLLS polls. Otherwise it is the `workspace` list (the
    client's when None). Requests are keyed by workspace and id, and a
    workspace the read names as unreadable keeps what was known of it. The
    first poll that reaches the server answers with one digest of the notes
    waiting per workspace; after it, each note is announced once, keyed by
    its note id and `at` (an edited note is announced again, one the agent
    has written after never is), and each due item once per
    `(item, review_until)`, after the notes of the same poll. A due hold is
    stale when it was already on the item the first time this watch read a
    request that the first listing held, fresh otherwise (a request that
    first appears in a later listing was created, hold and all, while the
    watch ran).
    """

    def __init__(self, workspace=None, everywhere=False, clock=time.monotonic):
        self.workspace = workspace
        self.everywhere = everywhere
        self.fault = None  # why the last poll could not read part of the ledger
        self.notice = None  # what a person should know of how it is watching
        self._skip_route = 0  # polls left before asking for the route again
        self._clock = clock
        self._started = False
        # each keyed by (workspace, request id)
        self._stamps = {}  # -> (revision, updated_at, ready) last read
        self._announced = {}  # -> {(note id, at)} still waiting
        self._due_at = {}  # -> clock moment its earliest hold runs out
        self._due_told = {}  # -> {(item, review_until)} announced
        self._inherited = {}  # -> {(item, review_until)} held at first read
        self._first_keys = None  # the open requests in the first listing read

    def _rows(self, client):
        """`(workspace, row)` for each open request the list names, and the
        workspaces it could not read (name -> why), whose requests are kept
        as they were rather than taken for closed. Raises DwApiError when no
        list can be read."""
        if self.everywhere and self._skip_route > 0:
            self._skip_route -= 1
        elif self.everywhere:
            try:
                listing = client.get_json(
                    api_path("api", "requests"), headers=CHANNEL_HEADER
                )
            except DwApiError as e:
                if e.status_code != 404:
                    raise
                # a server older than the route: one workspace, asking again
                # now and then in case it is updated under a running channel
                self._skip_route = ROUTE_RETRY_POLLS - 1
                self.notice = (
                    "the server has no GET /api/requests, so only workspace "
                    f'"{_workspace(client, self.workspace)}" is watched; '
                    "update the server to hear every workspace"
                )
            else:
                self.notice = None
                rows = [
                    (row.get("workspace"), row)
                    for row in listing.get("requests") or []
                    if row.get("state") == "open"
                ]
                return rows, dict(listing.get("unreadable") or {})
        name = _workspace(client, self.workspace)
        listing = get_request(client, workspace=name, headers=CHANNEL_HEADER)
        if listing.get("unreadable"):
            raise DwApiError(str(listing["unreadable"]))
        rows = [
            (name, row)
            for row in listing.get("requests") or []
            if row.get("state") == "open"
        ]
        return rows, {}

    def _read_due(self, key, stamp):
        """Whether request `key` must be read again: its stamp moved, or its
        earliest hold has run out."""
        if self._stamps.get(key) != stamp:
            return True
        due_at = self._due_at.get(key)
        return due_at is not None and self._clock() >= due_at

    def poll(self, client):
        """The events to send now. Raises DwApiError when the list cannot
        be read; a single request or workspace that cannot be is left for
        the next poll and named in `fault`."""
        rows, broken = self._rows(client)
        self.fault = "; ".join(f"{n}: {why}" for n, why in broken.items()) or None
        fresh, reviews, open_keys = [], [], set()
        if self._first_keys is None:
            self._first_keys = {(name, row.get("id")) for name, row in rows}
        for name, row in rows:
            key = (name, row.get("id"))
            open_keys.add(key)
            stamp = (row.get("revision"), row.get("updated_at"), row.get("ready"))
            if not self._read_due(key, stamp):
                continue
            try:
                request = get_request(client, key[1], name)
            except DwApiError as e:
                if e.status_code != 404:  # deleted since the list was read
                    self.fault = str(e)
                continue
            waiting = waiting_notes(request)
            told = self._announced.get(key, ())
            fresh += [(name, n) for n in waiting if (n["note"], n["at"]) not in told]
            self._announced[key] = {(n["note"], n["at"]) for n in waiting}
            if key not in self._inherited:
                held = key in self._first_keys
                self._inherited[key] = {
                    (item.get("key"), item["review_until"])
                    for item in request.get("items") or []
                    if held and not item.get("dropped") and item.get("review_until")
                }
            inherited = self._inherited[key]
            due = due_reviews(request)
            seen = self._due_told.setdefault(key, set())
            new = [d for d in due if (d["item"], d["until"]) not in seen]
            seen.update((d["item"], d["until"]) for d in new)
            old = [d for d in new if (d["item"], d["until"]) in inherited]
            if old:
                reviews.append((name, old, True))
            if len(old) < len(new):
                reviews.append((name, [d for d in new if d not in old], False))
            left = _holding_left(request)
            if left is None:
                self._due_at.pop(key, None)
            else:
                self._due_at[key] = self._clock() + left
            self._stamps[key] = stamp
        for key in set(self._stamps) - open_keys:
            if key[0] in broken:  # not read this time, not closed
                continue
            del self._stamps[key]
            del self._announced[key]
            self._due_at.pop(key, None)
            self._due_told.pop(key, None)
            self._inherited.pop(key, None)
        started, self._started = self._started, True
        if not started:
            by_workspace = {}
            for name, note in fresh:
                by_workspace.setdefault(name, []).append(note)
            events = [
                waiting_event(notes, name) for name, notes in by_workspace.items()
            ]
        else:
            events = [note_event(note, name) for name, note in fresh]
        return events + [review_event(d, name, stale=s) for name, d, s in reviews]
