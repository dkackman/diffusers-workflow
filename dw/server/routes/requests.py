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
from ...prompts import PromptDigests
from ...security import SecurityError, validate_path
from ...workspace import DEFAULT_WORKSPACE_NAME, workspace_names
from ..deps import (
    internal_error,
    server_prompt_library,
    workspace_for,
    workspace_root,
)
from ..changes import REQUESTS, ChangeRoute, pending_change, writes
from ..outputs import shown_outputs

router = APIRouter(route_class=ChangeRoute)

# --- the requests ledger (dw/ledger.py owns the model and its rules) ---
# Every route is behind the bearer token and none carries `query_token_ok`:
# nothing a browser loads without headers reads the ledger. A default
# workspace with no root folder holds no requests; a write is 409
# `no_workspace_root`. `{request_id:path}`: an id like `a/b` (or an empty
# one) must match here and meet `validate_ledger_key`, not be a bare 404/405.


def _jobs_for(request, name, reasons=False):
    """`rid -> its linked jobs`, each with its `reason` only if `reasons`."""
    manager = request.app.state.job_manager
    return lambda rid: [
        j if reasons else {**j, "reason": None} for j in manager.request_jobs(name, rid)
    ]


def _prompts(request):
    """A fresh `name -> sha256 of that prompt's text` memo over the server's
    prompt library, one per view: what the revised rule's prompt clause
    reads (`ledger_views.is_revised`, rule 2)."""
    return PromptDigests(server_prompt_library(request.app.state))


def _linked_jobs_for(request, name):
    """`(rid, item key) -> [{job, status, run_id, created_at, workflow,
    arguments, prompt_digests}]`: what `accepted` is checked against, and
    what says whether the stored item reads revised."""
    jobs_for = _jobs_for(request, name)

    def linked(rid, key):
        return [
            {
                field: j.get(field)
                for field in ("job", "status", "run_id", "created_at", "workflow")
                + ("arguments", "prompt_digests")
            }
            for j in jobs_for(rid)
            if j["item"] == key
        ]

    return linked


def _ledger_root(state, name):
    """The root whose requests.json a request names, None for a default
    workspace with no root. Raises 400/404/409 as `workspace_for` does."""
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
def list_requests(request: Request, name: str, needs: bool = False):
    """A workspace's requests, newest change first, without their items;
    `needs=true` adds each row's `needs`. `unreadable` says why when
    requests.json is there but is not a ledger."""
    root = _ledger_root(request.app.state, name)
    if request.headers.get("X-DW-Notes-Channel") == "1":
        request.app.state.listening.seen(name)
    return {
        "workspace": name,
        **ledger.listing(root, _jobs_for(request, name), _prompts(request), needs),
    }


@router.get(
    "/api/requests",
    response_model=OpenRequestList,
    response_model_exclude_unset=True,
)
def list_open_requests(request: Request):
    """Every workspace's open requests, without items, newest change first:
    the one read a notes channel makes to watch the whole server. Marked
    `X-DW-Notes-Channel: 1`, it polls every workspace. `unreadable` names a
    workspace whose requests.json is not a ledger or could not be resolved;
    the others still list, and a watcher keeps what it knew of the named
    ones rather than taking them for closed."""
    state = request.app.state
    if request.headers.get("X-DW-Notes-Channel") == "1":
        state.listening.seen_all()
    names = (
        workspace_names(state.workspace_root)
        if state.workspace_root
        else [DEFAULT_WORKSPACE_NAME]
    )
    rows, broken = [], {}
    prompts = _prompts(request)
    for name in names:
        try:
            root = _ledger_root(state, name)
        except HTTPException as e:  # removed since the root was listed
            broken[name] = str(e.detail)
            continue
        problem = ledger.unreadable(root)
        if problem is not None:
            broken[name] = str(problem)
            continue
        jobs_for = _jobs_for(request, name)
        rows += [
            {
                "workspace": name,
                **ledger.summary_view(r, jobs_for(r.get("id")), prompts),
            }
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
def get_request(request: Request, name: str, request_id: str, reasons: bool = False):
    """One request: its plan, items with derived status, and its
    history; `reasons=true` adds a failed take's `reason`. 409
    `ledger_unreadable` when requests.json is no ledger."""
    root = _ledger_root(request.app.state, name)
    try:
        found = ledger.get_request(
            root, request_id, _jobs_for(request, name, reasons), _prompts(request)
        )
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
@writes(REQUESTS)
def put_request(
    request: Request,
    name: str,
    request_id: str,
    body: RequestWrite,
    reasons: bool = False,
):
    """Write one request's whole plan. `revision` is the revision it was
    read at, or 0 to create it; any other is 409 `stale_revision` with
    `current_revision`. `prompt` is set on create only (422
    `prompt_is_write_once`). An item left out is deleted, or kept as dropped
    when jobs link to it. `reasons=true` adds a failed take's `reason`."""
    root = _ledger_root(request.app.state, name)
    payload = body.model_dump(exclude={"by"}, exclude_unset=True)
    try:
        written = ledger.put_request(
            root,
            request_id,
            payload,
            by=body.by,
            linked_jobs=_linked_jobs_for(request, name),
            jobs_for=_jobs_for(request, name, reasons),
            prompts=_prompts(request),
        )
    except ledger.LedgerError as e:
        raise HTTPException(status_code=e.status, detail=e.detail())
    except OSError:
        raise internal_error(f"Could not write request {request_id!r}")
    pending_change(request).in_workspace(name)
    return written


# --- a user's notes for the agent, on a request or one of its items ---
# Their own path: `/requests/{request_id:path}` would swallow a `/notes`
# suffix. No `by`: only the user writes a note, so every write is the
# user's, and no dw_mcp tool calls these.

NOTES_URL = "/api/workspaces/{name}/request-notes"


def _note_ref_check(request, name):
    """`(request_id, ref) -> the ref to store`, or LedgerError
    `invalid_note_ref`. A ref naming only its item needs no job. Otherwise
    `ref.job` must be linked to `ref.item`; `ref.output` must be a file the
    gallery shows from that finished job's manifest, its kind stored."""
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


def _note_write(request, name, write, what):
    """Run one note write, a LedgerError as its HTTP error; a write that
    held moves the workspace's `requests` counter."""
    try:
        written = write()
    except ledger.LedgerError as e:
        raise HTTPException(status_code=e.status, detail=e.detail())
    except OSError:
        raise internal_error(f"Could not {what}")
    pending_change(request).in_workspace(name)
    return written


@router.post(NOTES_URL, response_model=NoteWritten, response_model_exclude_unset=True)
@writes(REQUESTS)
def add_request_note(request: Request, name: str, body: NoteAdd):
    """Add one of the user's notes for the agent, on an item or (no `item`)
    on the request; answers the note and the request's new revision. Each
    of `refs` names an item (the note's own, or any live item for a request
    note), optionally a job linked to it and one file of its manifest; the
    server adds the file's `kind`. 422
    `invalid_note_ref` for a ref that does not hold, `too_many_refs`,
    `unknown_item`, `item_dropped` or `too_many_notes`; 404
    `unknown_request`."""
    root = _ledger_root(request.app.state, name)
    return _note_write(
        request,
        name,
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
@writes(REQUESTS)
def edit_request_note(request: Request, name: str, body: NoteEdit):
    """Change one note's text; its `refs` and `created` stay, `at` moves to
    now. The same text writes nothing. 404 `unknown_request` or
    `unknown_note`."""
    root = _ledger_root(request.app.state, name)
    return _note_write(
        request,
        name,
        lambda: ledger.edit_note(root, body.request, body.note, body.text),
        f"edit a note of request {body.request!r}",
    )


@router.delete(NOTES_URL, response_model=NoteDeleted, response_model_exclude_unset=True)
@writes(REQUESTS)
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
        request,
        name,
        lambda: ledger.delete_note(root, request_id, note),
        f"remove a note of request {request_id!r}",
    )


@router.delete(
    "/api/workspaces/{name}/requests/{request_id:path}",
    response_model=LedgerDeleted,
    response_model_exclude_unset=True,
)
@writes(REQUESTS)
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
    pending_change(request).in_workspace(name)
    return {"workspace": name, **deleted}
