"""FastAPI dependencies and the per-request lookups the routers share.

Everything here reads `app.state` (the `state` argument, or a request's
`request.app.state`) rather than closing over an app, so two apps built in one
process each resolve their own workspaces, search paths and ceiling index.
"""

import json
import logging
import os
from typing import Optional

from fastapi import HTTPException, Request

from ..security import SecurityError
from ..vram_inheritance import build_index
from ..library import ASSETS_KIND, LORAS_KIND, PROMPTS_KIND, library_path
from ..workspace import (
    DEFAULT_WORKSPACE_NAME,
    Workspace,
    _holds_a_workspace,
    named_workspace,
)

logger = logging.getLogger("dw")


def internal_error(message):
    """Log the exception being handled with its traceback and return the
    500 that answers it: the detail is the category, never the message,
    which can carry a path or a value. Raised from inside an except block."""
    logger.exception(message)
    return HTTPException(
        status_code=500, detail="internal error - the server log has the detail"
    )


def asset_library_missing(shared=False):
    """The 409 every asset route answers when there is no library to write
    to or read from - one wording, so upload, keep and delete cannot drift."""
    if shared:
        return HTTPException(
            status_code=409,
            detail="This server has no shared asset library - it was "
            "configured from loose directories rather than a workspace "
            "root, so there is nothing for an asset to be common to",
        )
    return HTTPException(status_code=409, detail="This workspace has no asset library")


def prompt_library_missing():
    """The 409 a prompt save answers when the server has no prompt library."""
    return HTTPException(status_code=409, detail="This server has no prompt library")


def writable_asset_directory(state, ws, shared=False):
    """The directory an asset write lands in: the writable root of the
    workspace's asset path (`LibraryPath.writable_root`), or the shared one
    when `shared`. A shared root that is not a directory yet is off the path
    but is still the workspace's to create, so it is read off the workspace.
    Raises the one 409 when there is nowhere to write."""
    root = library_path(ASSETS_KIND, ws, state.examples_dirs).writable_root(shared)
    directory = root.root if root else None
    if directory is None and shared:
        directory = getattr(ws, "common_assets", None)
    if not directory:
        raise asset_library_missing(shared=shared)
    return directory


def writable_prompt_directory(state):
    """The directory a prompt save lands in, or the one 409."""
    root = server_prompt_library(state).writable_root()
    if root is None:
        raise prompt_library_missing()
    return root.root


def workspace_root(state):
    root = state.workspace_root
    if root is None:
        raise HTTPException(
            status_code=409,
            detail="This server has no workspace root - it was started "
            "with individual directory overrides, so it has one "
            "workspace and cannot create others",
        )
    return root


def workspace_for(state, name):
    """The Workspace a request names.

    No name, or the default name, is the server's own configuration -
    the directories it was started with - so every call that predates
    workspaces keeps working unchanged. A named one resolves under the
    root, and must already exist: creating a workspace by mentioning it
    would turn a typo into a directory. Checked by looking at the one
    candidate directory rather than listing the whole root - this runs
    on every gallery thumbnail request.
    """
    if not name or name == DEFAULT_WORKSPACE_NAME:
        return state.default_workspace
    root = workspace_root(state)
    try:
        # Prompts are shared by reference: a named workspace reads the one
        # library this server was started with, wherever --prompt-dir put it
        selected = named_workspace(root, name, prompts_root=state.prompt_dir)
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not _holds_a_workspace(selected.root):
        raise HTTPException(status_code=404, detail=f"No such workspace: {name}")
    return selected


def selected_workspace(request: Request, workspace: Optional[str] = None) -> Workspace:
    """FastAPI dependency form of workspace_for, reading the name from
    the `?workspace=` query parameter every scoped route already takes -
    used as `ws: Workspace = Depends(selected_workspace)`."""
    return workspace_for(request.app.state, workspace)


def sources_for(state, ws):
    """The workflow search path of one workspace: its own workflows
    first, then the same read-only roots every workspace shares."""
    return library_path("workflows", ws, state.examples_dirs)


def ceiling_index(state, ws):
    """The catalog's VRAM ceilings by pipeline identity, as this
    workspace's search path lists them (`dw/vram_inheritance.py`, #502).
    A file that cannot be read contributes nothing.

    One index per distinct listing, kept on the app (`state.ceiling_indexes`)
    - keyed by every file's path and mtime, so an edited, added or removed
    template rebuilds it and nothing else does."""
    paths = []
    for name, source in sources_for(state, ws).entries()[0].items():
        path = os.path.join(source.root, f"{name}.json")
        try:
            paths.append((name, path, os.path.getmtime(path)))
        except OSError:
            continue
    signature = tuple(paths)
    cached = state.ceiling_indexes.get(ws.name)
    if cached and cached[0] == signature:
        return cached[1]
    catalog = []
    for name, path, _ in paths:
        try:
            with open(path, "r") as file:
                catalog.append((name, json.load(file)))
        except (OSError, ValueError):
            continue
    index = build_index(catalog)
    state.ceiling_indexes[ws.name] = (signature, index)
    return index


def server_prompt_library(state):
    """The prompt search path: the library this server writes to, then
    the read-only ones an --examples-dir tree brought with it. A name in
    an earlier root shadows the same name later, as on the workflow
    search path. Shared by every workspace, so it names none. The server's
    own library stays on the path before it exists - a save creates it -
    while an examples tree's that is not a directory is dropped."""
    return library_path(
        PROMPTS_KIND,
        state.default_workspace,
        state.examples_dirs,
        primary=state.prompt_dir,
    )


def observed_for_name(
    state, name, definition, arguments=None, *, workspace=None, card=None
):
    """One workflow's `observed` block, from the same aggregate the
    listing uses - so the figure a caller reads in the listing and the
    one they read here are the same figure.

    `arguments` narrow it to the bucket the run being planned falls in;
    without them it is the figure the stored defaults give, which is the
    listing's. `workspace` scopes it to one workspace's own writable copy
    (#274); omitted, it is a shared catalog entry's pooled figure (#154).
    `card` ('cuda:1') prices it for that card rather than the server's."""
    costs = getattr(state, "observed_costs", None)
    return (
        costs.observed(name, definition, arguments, workspace=workspace, card=card)
        if costs
        else None
    )


def lora_library_missing():
    """The 409 a LoRA save answers when the server has no LoRA library."""
    return HTTPException(
        status_code=409,
        detail="This server has no LoRA library - it was configured from "
        "loose directories rather than a workspace root",
    )


def server_lora_library(state):
    """The LoRA catalog's search path: the root's loras/ (writable), then
    each examples tree's loras/ (read-only). Shared by every workspace."""
    return library_path(
        LORAS_KIND,
        state.default_workspace,
        state.examples_dirs,
        primary=getattr(state, "lora_dir", None),
    )
