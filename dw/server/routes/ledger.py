"""The requests ledger over HTTP: a workspace's `requests.json`.

GET lists a workspace's requests and reads one; PUT writes one request's
whole plan against the revision it was read at (`dw/ledger.py` owns the
model and its rules). Every route is behind the bearer token like the rest
of /api/*, and none carries `query_token_ok`: nothing a browser loads
without headers reads the ledger.

A server started from loose directories has a default workspace with no
root folder: it reads as holding no requests, and a write is refused with
409 `no_workspace_root`.
"""

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ... import ledger
from ...security import SecurityError, validate_path
from ...workspace import DEFAULT_WORKSPACE_NAME
from ..api_models import LedgerList, LedgerRequest
from ..deps import internal_error, workspace_for, workspace_root

router = APIRouter()


class RequestWrite(BaseModel):
    """One request's whole plan."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(description=f"One line, at most {ledger.MAX_TITLE_LENGTH}")
    goal: str = Field(description="What done looks like")
    items: list[dict[str, Any]] = Field(
        description="The ordered plan; each item is key, title, depends_on, "
        "estimate_minutes, workflow, arguments, accepted, comment"
    )
    revision: int = Field(description="The revision read, or 0 to create")
    why: str = Field(description=f"One line, at most {ledger.MAX_WHY_LENGTH}")
    prompt: str | None = Field(default=None, description="Create only")
    state: Literal["open", "done", "abandoned"] | None = None
    by: Literal["agent", "user"] = Field(
        default="user", description="Who wrote this revision; the UI is the user"
    )


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


def _refused(error):
    return HTTPException(status_code=error.status, detail=error.detail())


@router.get(
    "/api/workspaces/{name}/requests",
    response_model=LedgerList,
    response_model_exclude_unset=True,
)
def list_requests(request: Request, name: str):
    """A workspace's requests, newest change first, without their items."""
    root = _ledger_root(request.app.state, name)
    return {"workspace": name, "requests": ledger.list_requests(root)}


@router.get(
    "/api/workspaces/{name}/requests/{request_id}",
    response_model=LedgerRequest,
    response_model_exclude_unset=True,
)
def get_request(request: Request, name: str, request_id: str):
    """One request: its plan, every item with its derived status, and its
    history."""
    root = _ledger_root(request.app.state, name)
    try:
        found = ledger.get_request(root, request_id)
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if found is None:
        raise HTTPException(
            status_code=404, detail=f"No request {request_id!r} in workspace {name!r}"
        )
    return found


@router.put(
    "/api/workspaces/{name}/requests/{request_id}",
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
    payload = body.model_dump(exclude={"by"})
    try:
        return ledger.put_request(root, request_id, payload, by=body.by)
    except ledger.LedgerError as e:
        raise _refused(e)
    except OSError:
        raise internal_error(f"Could not write request {request_id!r}")
