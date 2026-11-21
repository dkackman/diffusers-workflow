"""The requests ledger routes: a workspace's requests, one request's plan,
and a user's notes for the agent on a request or one of its items.

`dw/ledger.py` owns the model and its rules; these handlers read the
workspace root and the linked jobs and turn a `LedgerError` into the HTTP
error it carries. Registered after the library router.
"""

from fastapi import APIRouter, HTTPException, Query, Request

from ..api_models import LedgerList, LedgerRequest
from ..api_models_library import (
    LedgerDeleted,
    NoteAdd,
    NoteDeleted,
    NoteEdit,
    NotesChannelState,
    NoteWritten,
    OpenRequestList,
    RequestWrite,
)
from ... import ledger
from ...security import SecurityError, validate_path
from ...workspace import DEFAULT_WORKSPACE_NAME, workspace_names
from ..deps import internal_error, workspace_for, workspace_root
from ..outputs import shown_outputs

router = APIRouter()

# --- the requests ledger (dw/ledger.py owns the model and its rules) ---
# Every route is behind the bearer token like the rest of /api/*, and none
# carries `query_token_ok`: nothing a browser loads without headers reads
# the ledger. A default workspace with no root folder reads as holding no
# requests, and a write is refused with 409 `no_workspace_root`.
# `{request_id:path}`: the router decodes a percent-encoded '/', so an id
# like `a/b` (or an empty one) must still match here and meet the ledger's
# id rule (`validate_ledger_key`), not fall through to a bare 404/405.


def _jobs_for(request, name):
    """`rid -> the jobs linked to that request`, for the derived item state."""
    manager = request.app.state.job_manager
    return lambda rid: manager.request_jobs(name, rid)


def _linked_jobs_for(request, name):
    """`(rid, item key) -> [{job, status, run_id}]`: what `accepted` is
    checked against."""
    jobs_for = _jobs_for(request, name)

    def linked(rid, key):
        return [
            {"job": j["job"], "status": j["status"], "run_id": j["run_id"]}
            for j in jobs_for(rid)
            if j["item"] == key
        ]

    return linked


def _ledger_root(state, name):
    """The workspace root whose requests.json a request names, or None for
    a default workspace with no root folder. Raises 400/404/409 as
    `workspace_for` does."""
    if name == DEFAULT_WORKSPACE_NAME:
        return state.default_workspace.root or None
    selected = workspace_for(state, name)
    try:
        return validate_path(
            selected.root, workspace_root(state).root, allow_create=False
        )
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/api/workspaces/{name}/requests",
    response_model=LedgerList,
    response_model_exclude_unset=True,
)
def list_requests(request: Request, name: str):
    """A workspace's requests, newest change first, without their items.
    `unreadable` says why when requests.json is there but is not a ledger."""
    root = _ledger_root(request.app.state, name)
    if request.headers.get("X-DW-Notes-Channel") == "1":
        request.app.state.listening.seen(name)
    return {"workspace": name, **ledger.listing(root, _jobs_for(request, name))}


@router.get(
    "/api/requests",
    response_model=OpenRequestList,
    response_model_exclude_unset=True,
)
def list_open_requests(request: Request):
    """Every workspace's open requests, without their items, newest change
    first: the one read a notes channel makes when it watches the whole
    server. Marked `X-DW-Notes-Channel: 1`, it counts as a poll of every
    workspace. `unreadable` names a workspace whose requests.json is there
    but is not a ledger; the others still list."""
    state = request.app.state
    if request.headers.get("X-DW-Notes-Channel") == "1":
        state.listening.seen_all()
    names = (
        workspace_names(state.workspace_root)
        if state.workspace_root
        else [DEFAULT_WORKSPACE_NAME]
    )
    rows, broken = [], {}
    for name in names:
        try:
            root = _ledger_root(state, name)
        except HTTPException:
            continue  # removed since the root was listed
        problem = ledger.unreadable(root)
        if problem is not None:
            broken[name] = str(problem)
            continue
        jobs_for = _jobs_for(request, name)
        rows += [
            {"workspace": name, **ledger.summary_view(r, jobs_for(r.get("id")))}
            for r in ledger.read_requests(root).values()
            if r.get("state") == "open"
        ]
    rows.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
    return {"requests": rows, **({"unreadable": broken} if broken else {})}


@router.get(
    "/api/workspaces/{name}/notes-channel",
    response_model=NotesChannelState,
)
def notes_channel(request: Request, name: str):
    """Whether a notes channel has polled this workspace's request list in
    the last 30 seconds, so the UI can say whether a session is listening."""
    _ledger_root(request.app.state, name)
    return request.app.state.listening.state(name)


@router.get(
    "/api/workspaces/{name}/requests/{request_id:path}",
    response_model=LedgerRequest,
    response_model_exclude_unset=True,
)
def get_request(request: Request, name: str, request_id: str):
    """One request: its plan, every item with its derived status, and its
    history. 409 `ledger_unreadable` when requests.json is not a ledger."""
    root = _ledger_root(request.app.state, name)
    try:
        found = ledger.get_request(root, request_id, _jobs_for(request, name))
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ledger.LedgerError as e:
        raise HTTPException(status_code=e.status, detail=e.detail())
    if found is None:
        raise HTTPException(
            status_code=404, detail=f"No request {request_id!r} in workspace {name!r}"
        )
    return found


@router.put(
    "/api/workspaces/{name}/requests/{request_id:path}",
    response_model=LedgerRequest,
    response_model_exclude_unset=True,
)
def put_request(request: Request, name: str, request_id: str, body: RequestWrite):
    """Write one request's whole plan. `revision` is the revision it was
    read at, or 0 to create it; a write against any other revision is 409
    `stale_revision`, with `current_revision` in the detail. `prompt` is
    set on create only (a later change is 422 `prompt_is_write_once`). An
    item left out is deleted, or kept as dropped when jobs link to it."""
    root = _ledger_root(request.app.state, name)
    payload = body.model_dump(exclude={"by"}, exclude_unset=True)
    try:
        return ledger.put_request(
            root,
            request_id,
            payload,
            by=body.by,
            linked_jobs=_linked_jobs_for(request, name),
            jobs_for=_jobs_for(request, name),
        )
    except ledger.LedgerError as e:
        raise HTTPException(status_code=e.status, detail=e.detail())
    except OSError:
        raise internal_error(f"Could not write request {request_id!r}")


# --- a user's notes for the agent, on a request or one of its items ---
# Their own path: `/requests/{request_id:path}` would swallow a `/notes`
# suffix. No `by`: only the user writes a note, so every write is the
# user's, and no dw_mcp tool calls these.

NOTES_URL = "/api/workspaces/{name}/request-notes"


def _note_ref_check(request, name):
    """`(request_id, ref) -> the ref to store`, or LedgerError
    `invalid_note_ref`. A ref naming only its item (a planned shot) needs no
    job. Otherwise `ref.job` must be linked to the request's `ref.item`;
    `ref.output`, when given, must be a file the gallery shows from that
    finished job's manifest, and its kind is stored with it."""
    manager = request.app.state.job_manager

    def refused(why):
        return ledger.LedgerError(why, status=422, code="invalid_note_ref")

    def check(request_id, ref):
        item = ref["item"]
        job_id = ref.get("job")
        if job_id is None:
            return {"item": item}
        linked = next(
            (
                job
                for job in manager.request_jobs(name, request_id)
                if job["job"] == job_id and job["item"] == item
            ),
            None,
        )
        if linked is None:
            raise refused(
                f"job {job_id!r} is not linked to item {item!r} of request "
                f"{request_id!r}"
            )
        output = ref.get("output")
        if output is None:
            return {"item": item, "job": job_id}
        if linked["status"] in ("queued", "running"):
            raise refused(
                f"job {job_id!r} has not finished, so it names no file yet - "
                "leave out `output` to note the take as a whole"
            )
        kind = shown_outputs(manager.history.manifest_of(job_id)).get(output)
        if kind is None:
            raise refused(
                f"{output!r} is not a file job {job_id!r} wrote - name one of "
                "its manifest's outputs as the gallery names it"
            )
        return {"item": item, "job": job_id, "output": output, "kind": kind}

    return check


def _note_write(write, what):
    """Run one note write, its LedgerError as the HTTP error it carries."""
    try:
        return write()
    except ledger.LedgerError as e:
        raise HTTPException(status_code=e.status, detail=e.detail())
    except OSError:
        raise internal_error(f"Could not {what}")


@router.post(NOTES_URL, response_model=NoteWritten, response_model_exclude_unset=True)
def add_request_note(request: Request, name: str, body: NoteAdd):
    """Add one of the user's notes for the agent, on an item or (no `item`)
    on the request itself; answers the note and the request's new revision.
    Each of `refs` names an item (the note's own, for an item note; any live
    item, for a request note) and optionally a job linked to it and one file
    of that job's manifest; the server adds the file's `kind`. 422
    `invalid_note_ref` for a ref that does not hold, `too_many_refs`,
    `unknown_item`, `item_dropped` or `too_many_notes`; 404
    `unknown_request`."""
    root = _ledger_root(request.app.state, name)
    return _note_write(
        lambda: ledger.add_note(
            root,
            body.request,
            item=body.item,
            text=body.text,
            refs=[ref.model_dump(exclude_none=True) for ref in body.refs or []],
            check_ref=_note_ref_check(request, name),
        ),
        f"add a note to request {body.request!r}",
    )


@router.put(NOTES_URL, response_model=NoteWritten, response_model_exclude_unset=True)
def edit_request_note(request: Request, name: str, body: NoteEdit):
    """Change one note's text; its `refs` and `created` stay, `at` moves to
    now. The same text writes nothing. 404 `unknown_request` or
    `unknown_note`."""
    root = _ledger_root(request.app.state, name)
    return _note_write(
        lambda: ledger.edit_note(root, body.request, body.note, body.text),
        f"edit a note of request {body.request!r}",
    )


@router.delete(NOTES_URL, response_model=NoteDeleted, response_model_exclude_unset=True)
def delete_request_note(
    request: Request,
    name: str,
    request_id: str = Query(alias="request", description="The request id"),
    note: str = Query(description="The note id"),
):
    """Remove one note; its id is never handed out again. 404
    `unknown_request` or `unknown_note`."""
    root = _ledger_root(request.app.state, name)
    return _note_write(
        lambda: ledger.delete_note(root, request_id, note),
        f"remove a note of request {request_id!r}",
    )


@router.delete(
    "/api/workspaces/{name}/requests/{request_id:path}",
    response_model=LedgerDeleted,
    response_model_exclude_unset=True,
)
def delete_request(
    request: Request,
    name: str,
    request_id: str,
    revision: int = Query(description="The revision the request was read at"),
):
    """Remove one request from the ledger. Its linked jobs keep their
    history rows, their `request`/`item` and their files. 409
    `stale_revision` at another revision, 409 `request_has_active_jobs`
    while a linked job is queued or running, 404 `unknown_request`."""
    root = _ledger_root(request.app.state, name)
    try:
        deleted = ledger.delete_request(
            root, request_id, revision, jobs_for=_jobs_for(request, name)
        )
    except ledger.LedgerError as e:
        raise HTTPException(status_code=e.status, detail=e.detail())
    except OSError:
        raise internal_error(f"Could not delete request {request_id!r}")
    return {"workspace": name, **deleted}
