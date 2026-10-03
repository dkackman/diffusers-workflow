"""The LoRA catalog routes (`dw/lora_catalog.py` holds the rules).

`GET /api/loras/recommend` is registered before the greedy
`{name:path}` routes, and `recommend` is a reserved entry name, so the two
cannot shadow each other.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...library import ReadOnlyLibraryError
from ...lora_catalog import (
    RESERVED_NAMES,
    SCHEMA_NAME,
    entry_errors,
    is_repo_id,
    matches,
    query_terms,
    ranked,
    rejection_reasons,
    workflow_bases,
)
from ...lora_hub import search_hub
from ...schema import load_schema
from ...security import InvalidInputError, SecurityError, validate_lora_name, validate_path
from ...workspace import Workspace, forget_workspace_usage
from ..deps import lora_library_missing, selected_workspace, server_lora_library, sources_for

logger = logging.getLogger("dw")
router = APIRouter()


class LoraRequest(BaseModel):
    entry: Dict[str, Any] = Field(description="The catalog entry to save")


def _named(name):
    """A request's entry name, validated; 404 for a name no entry can have."""
    bare = name.removesuffix(".json")
    try:
        return validate_lora_name(bare)
    except InvalidInputError as error:
        raise HTTPException(status_code=404, detail=str(error))


def catalog_entries(state):
    """(entries, roots, library): every valid entry on the search path by
    name, the root each came from, and the path. An unreadable or invalid
    file is logged and skipped - one bad file must not hide the catalog."""
    library = server_lora_library(state).existing()
    winners, _hidden = library.entries()
    entries, roots = {}, {}
    for name, root in winners.items():
        path = library.path_in(root, name)
        if path is None:
            continue
        try:
            with open(path, "r") as file:
                entry = json.load(file)
        except (OSError, ValueError) as error:
            logger.warning(f"Skipping unreadable LoRA entry {name}: {error}")
            continue
        problem = entry_errors(entry) if isinstance(entry, dict) else "not an object"
        if problem:
            logger.warning(f"Skipping invalid LoRA entry {name}: {problem}")
            continue
        entries[name] = entry
        roots[name] = root
    return entries, roots, library


def resolve_model(state, ws, model, partition=None):
    """`model` as `(repo, partition)` pairs: a catalog workflow's bases, or
    a Hub repo id taken as given. A workflow is tried first - a workflow
    name like `models/qwen` has a repo id's shape too."""
    found = sources_for(state, ws).find(model.removesuffix(".json"))
    if found is not None:
        try:
            with open(found[0], "r") as file:
                definition = json.load(file)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=f"Workflow '{model}' cannot be read: {error}")
        if not isinstance(definition, dict):
            raise HTTPException(status_code=400, detail=f"Workflow '{model}' is not a JSON object")
        bases = workflow_bases(definition)
    elif is_repo_id(model):
        bases = [(model, None)]
    else:
        raise HTTPException(
            status_code=400,
            detail=f"'{model}' is neither a workflow on this server nor a Hub repo id (owner/name)",
        )
    if partition:
        bases = [(repo, partition) for repo, _ in bases]
    return bases


def described_bases(bases):
    return [{"repo": repo, "workflow": partition} for repo, partition in bases]


@router.get("/api/lora-schema")
def get_lora_schema():
    """The JSON schema a catalog entry must satisfy - its own path, so an
    entry named 'schema' cannot shadow it."""
    return JSONResponse(load_schema(SCHEMA_NAME))


@router.get("/api/loras")
def list_loras(
    request: Request,
    model: Optional[str] = None,
    workflow: Optional[str] = None,
    status: Optional[str] = None,
    tag: Optional[str] = None,
    ws: Workspace = Depends(selected_workspace),
):
    state = request.app.state
    entries, roots, library = catalog_entries(state)
    body = {"libraries": library.describe()}
    if model:
        bases = resolve_model(state, ws, model, workflow)
        entries = {n: e for n, e in entries.items() if matches(e, bases)}
        body["resolved"] = described_bases(bases)
    if status:
        entries = {n: e for n, e in entries.items() if e.get("status") == status}
    if tag:
        entries = {n: e for n, e in entries.items() if tag in e.get("tags", [])}
    body["loras"] = [
        {**row, "origin": roots[row["name"]].origin, "writable": roots[row["name"]].writable}
        for row in sorted(ranked(entries, []), key=lambda row: row["name"])
    ]
    return body


TRIAL_NOTE = (
    "Catalog rows are LoRAs tried on this base. Hub rows are candidates to "
    "trial, ranked by downloads only - run one beside a no-LoRA render at the "
    "same seed, and save_lora a trial that works (status proven, its job in "
    "evidence)."
)


@router.get("/api/loras/recommend")
def recommend_loras(
    request: Request,
    model: str,
    query: str = Query("", max_length=200),
    limit: int = Query(8, ge=1, le=25),
    ws: Workspace = Depends(selected_workspace),
):
    """Catalog entries for `model` ranked against `query`, then Hub
    candidates for the same exact bases. The Hub is searched only here."""
    state = request.app.state
    bases = resolve_model(state, ws, model)
    entries, _roots, _library = catalog_entries(state)
    fitting = {n: e for n, e in entries.items() if matches(e, bases)}
    terms = query_terms(query)
    catalog = [
        {"source": "catalog", **row}
        for row in ranked(fitting, terms)
        if row.get("status") != "rejected"
    ]
    body = {"resolved": described_bases(bases), "catalog": catalog, "hub": [], "note": TRIAL_NOTE}
    repos = sorted({repo for repo, _ in bases})
    if repos:
        hub, hub_error = search_hub(repos, query, terms, limit, rejection_reasons(fitting))
        body["hub"] = hub
        if hub_error:
            body["hub_error"] = hub_error
    return body


@router.put("/api/loras/{name:path}")
def save_lora(http_request: Request, name: str, request: LoraRequest):
    """Write an entry into the server's LoRA library, schema-checked first.
    Saving over a shipped entry writes a copy that shadows it."""
    state = http_request.app.state
    bare = name.removesuffix(".json")
    try:
        bare = validate_lora_name(bare)
    except InvalidInputError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if bare in RESERVED_NAMES:
        raise HTTPException(status_code=400, detail=f"'{bare}' is reserved by the LoRA routes")
    problem = entry_errors(request.entry)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    root = server_lora_library(state).writable_root()
    if root is None:
        raise lora_library_missing()
    try:
        path = validate_path(os.path.join(root.root, f"{bare}.json"), root.root, allow_create=True)
    except SecurityError as error:
        raise HTTPException(status_code=400, detail=str(error))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(request.entry, file, indent=2)
        file.write("\n")
    logger.info(f"Saved LoRA entry {bare} to {path}")
    return {"name": bare}


@router.delete("/api/loras/{name:path}")
def delete_lora(request: Request, name: str):
    """Remove one of the server's own entries; a shipped one is a 403."""
    state = request.app.state
    library = server_lora_library(state)
    found = library.find(_named(name))
    if found is None:
        raise HTTPException(status_code=404, detail=f"Unknown LoRA entry: {name}")
    path, root = found
    try:
        library.require_writable(root, name)
    except ReadOnlyLibraryError as refusal:
        raise HTTPException(status_code=403, detail=str(refusal))
    os.remove(path)
    forget_workspace_usage()
    return {"name": name.removesuffix(".json"), "deleted": True}


@router.get("/api/loras/{name:path}")
def get_lora(request: Request, name: str):
    found = server_lora_library(request.app.state).find(_named(name))
    if found is None:
        raise HTTPException(status_code=404, detail=f"Unknown LoRA entry: {name}")
    path, root = found
    try:
        with open(path, "r") as file:
            body = json.load(file)
    except (OSError, ValueError) as error:
        logger.warning(f"Unreadable LoRA entry {name}: {error}")
        raise HTTPException(status_code=404, detail=f"LoRA entry {name} is unreadable")
    return JSONResponse(
        body,
        headers={
            "X-Lora-Origin": root.origin,
            "X-Lora-Writable": "true" if root.writable else "false",
        },
    )
