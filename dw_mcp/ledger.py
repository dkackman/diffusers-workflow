"""The requests ledger: the plan an agent keeps for what the user asked.

A request lives in its workspace's `requests.json` on the server
(`dw/ledger.py`), read and written whole through
`/api/workspaces/<name>/requests/<id>`. A write names the revision it was
read at, so two writers cannot silently overwrite each other; the server's
refusal says which revision is current.
"""

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


def get_request(client, request=None, workspace=None):
    """One request's plan, or the workspace's list when `request` is None."""
    name = _workspace(client, workspace)
    if request is None:
        return client.get_json(api_path("api", "workspaces", name, "requests"))
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
):
    """Write a request's whole plan, as the agent. The user's notes are
    never sent: the server keeps what is there."""
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

# The ledger's own cap on a comment (dw/notes.py); a note is quoted whole
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


def waiting_notes(request):
    """The notes of one request that wait on the agent: the request's own,
    then each live item's in plan order (notes in stored order), as `{request,
    revision, item, title, note, text, at, refs}`. `item` is "" for the
    request's own notes; `note` is the note's id."""
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
            "at": note.get("at"),
            "refs": note.get("refs") or [],
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
    title = f" ({note['title']})" if note.get("title") else ""
    if note["item"]:
        return f'request "{note["request"]}", item "{note["item"]}"{title}'
    return f'request "{note["request"]}"{title}'


def _read_call(note, workspace):
    return f'get_request("{note["request"]}", workspace="{workspace}")'


def note_event(note, workspace):
    """The channel event for one note saved while the session is open: its
    text verbatim, what it is on, and what to do. `meta` keys are
    identifiers and its values strings - the channel contract drops any
    other key."""
    text = note["text"]
    clipped = len(text) > NOTE_QUOTE_CAP
    lines = [
        f"Note from the user on {_target(note)}, saved {_saved(note['at'])}:",
        text[:NOTE_QUOTE_CAP],
    ]
    if clipped:
        lines.append(
            f"(Clipped at {NOTE_QUOTE_CAP} characters of {len(text)}; "
            "get_request has the whole note.)"
        )
    lines.append(
        f"Read the request with {_read_call(note, workspace)} and act on the "
        'note; say what you did in the "why" of your next set_request.'
    )
    meta = {
        "kind": "note",
        "workspace": str(workspace),
        "request": str(note["request"]),
        "revision": str(note["revision"]),
    }
    if note["item"]:
        meta["item"] = str(note["item"])
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
        lines.append(f"- {_target(note)}, saved {_saved(note['at'])}: {text}")
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
        "set_request."
    )
    meta = {"kind": "notes_waiting", "workspace": str(workspace), "count": str(count)}
    return {"content": "\n\n".join([head, "\n".join(lines), tail]), "meta": meta}


def retry_delay(interval, failures):
    """Seconds until the next poll: the interval, doubled for each failed
    poll in a row, up to MAX_BACKOFF_SECONDS."""
    if failures <= 0:
        return interval
    return min(interval * 2 ** min(failures, 16), max(interval, MAX_BACKOFF_SECONDS))


class NoteWatch:
    """Which waiting notes one channel process has yet to announce.

    `poll` reads the workspace's request list - one call when nothing
    changed - and each open request whose revision moved since it was last
    read. The first poll that reaches the server answers with one digest of
    everything waiting; after it, each note is announced once, keyed by its
    request, item and `at`, so an edited note (a new `at`) is announced
    again and one the agent has written after never is.
    """

    def __init__(self, workspace=None):
        self.workspace = workspace
        self.fault = None  # why the last poll could not read a request
        self._started = False
        self._stamps = {}  # request id -> (revision, updated_at) last read
        self._announced = {}  # request id -> {(item key, at)} still waiting

    def poll(self, client):
        """The events to send now. Raises DwApiError when the list cannot
        be read; a single request that cannot be is left for the next poll
        and named in `fault`."""
        self.fault = None
        name = _workspace(client, self.workspace)
        listing = get_request(client, workspace=name)
        if listing.get("unreadable"):
            raise DwApiError(str(listing["unreadable"]))
        fresh, open_ids = [], set()
        for row in listing.get("requests") or []:
            if row.get("state") != "open":
                continue
            request_id = row.get("id")
            open_ids.add(request_id)
            stamp = (row.get("revision"), row.get("updated_at"))
            if self._stamps.get(request_id) == stamp:
                continue
            try:
                request = get_request(client, request_id, name)
            except DwApiError as e:
                if e.status_code != 404:  # deleted since the list was read
                    self.fault = str(e)
                continue
            waiting = waiting_notes(request)
            told = self._announced.get(request_id, ())
            fresh += [n for n in waiting if (n["item"], n["note"], n["at"]) not in told]
            self._announced[request_id] = {
                (n["item"], n["note"], n["at"]) for n in waiting
            }
            self._stamps[request_id] = stamp
        for request_id in set(self._stamps) - open_ids:
            del self._stamps[request_id]
            del self._announced[request_id]
        started, self._started = self._started, True
        if not fresh:
            return []
        if not started:
            return [waiting_event(fresh, name)]
        return [note_event(note, name) for note in fresh]
