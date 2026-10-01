"""The library routes: workspaces, workflows, stored prompts and the prompt
enhancers - the three listings 3c reshapes together.

Every handler reads its state from `request.app.state`. The greedy
`{name:path}` GETs stay after their `/download` and `/variables` siblings,
since the first route that matches wins.
"""

import copy
import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ...introspection import workflow_argument_warnings
from ...prompts import RESERVED_TEXT_PREFIXES
from ...schema import load_schema, validate_data
from ...security import InvalidInputError, SecurityError, validate_prompt_reference
from ...workflow import Workflow
from ...workflow_sources import (
    EXAMPLES_ORIGIN,
    WORKSPACE_ORIGIN,
    listing,
    workflow_names,
)
from ...workspace import (
    DEFAULT_WORKSPACE_NAME,
    NotAWorkspaceError,
    Workspace,
    create_workspace,
    delete_workspace,
    forget_workspace_usage,
    named_workspace,
    workspace_contents,
    workspace_names,
    workspace_usage,
)
from ..admission import JobRequest, admit_for
from ..catalog import (
    matching_prompts,
    attach_observed,
    prompt_details,
    resolve_prompt_name,
    resolve_readable_workflow,
    resolve_writable_workflow,
    workflow_details,
)
from ..catalog_shape import derive_catalog_metadata, project_listing
from ..deps import (
    observed_for_name,
    prompt_roots,
    selected_workspace,
    sources_for,
    workspace_root,
)
from ..enhancers import build_enhance_workflow, preset_descriptions
from ..http_security import query_token_ok
from ..jobs import TERMINAL_STATES

logger = logging.getLogger("dw")

router = APIRouter()


class WorkspaceRequest(BaseModel):
    name: str = Field(description="Name for the new workspace")


@router.get("/api/workspaces")
def list_workspaces(request: Request):
    """Every workspace on this server, the default first.

    A workspace is a namespace, not a security boundary: the API token
    is all-or-nothing, so anything that can list these can reach all of
    them.
    """
    state = request.app.state
    root = state.workspace_root
    # workspace_names lists the whole root once; everything after the
    # first entry (always the default, see its docstring) is a named
    # workspace to describe individually
    names = workspace_names(root)[1:] if root else []
    listed = [state.default_workspace]
    for name in names:
        listed.append(named_workspace(root, name))
    described = []
    for space in listed:
        entry = space.describe()
        # Roughly how much disk it holds, cached for a minute inside
        # workspace_usage - a listing is a glance, and a job writing
        # into outputs moves the number continuously anyway
        entry["usage"] = workspace_usage(space)
        described.append(entry)
    return {
        "workspace_root": root.root if root else None,
        "default": DEFAULT_WORKSPACE_NAME,
        "workspaces": described,
    }


@router.post("/api/workspaces", status_code=201)
def add_workspace(http_request: Request, request: WorkspaceRequest):
    """Create a workspace: its own workflows, assets and outputs, sharing
    this server's one prompt library."""
    state = http_request.app.state
    root = workspace_root(state)
    try:
        created = create_workspace(root, request.name)
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    forget_workspace_usage()
    logger.info(f"Created workspace {request.name} at {created.root}")
    return created.describe()


@router.delete("/api/workspaces/{name}")
def remove_workspace(request: Request, name: str, acknowledged: bool = False):
    """Delete a workspace and everything in it.

    Answers what it would remove and refuses until `acknowledged=true`:
    this deletes generated work, and a count is what makes it an
    informed choice rather than a surprise. The unacknowledged message
    names only what would be removed - how to proceed is left to the
    caller, since the MCP surface tells its own callers to acknowledge
    through a differently-named parameter (`acknowledged_cost`).
    """
    state = request.app.state
    manager = request.app.state.job_manager
    root = workspace_root(state)
    if name == DEFAULT_WORKSPACE_NAME:
        raise HTTPException(
            status_code=400,
            detail="The default workspace cannot be deleted - it is the "
            "workspace root itself, and holds the shared prompt library",
        )
    if name not in workspace_names(root):
        raise HTTPException(status_code=404, detail=f"No such workspace: {name}")

    contents = workspace_contents(named_workspace(root, name))
    if not acknowledged:
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"Deleting workspace '{name}' removes these "
                f"files permanently.",
                "contents": contents,
            },
        )
    with manager._lock:
        queued = [
            job
            for job in manager.jobs.values()
            if job.status not in TERMINAL_STATES and job.spec.get("workspace") == name
        ]
    if queued:
        raise HTTPException(
            status_code=409,
            detail=f"Workspace '{name}' has {len(queued)} job(s) queued or "
            f"running - cancel them first",
        )
    try:
        delete_workspace(root, name)
    except NotAWorkspaceError as e:
        # The directory holds more than a workspace - refused outright,
        # since what else it holds is not the caller's to acknowledge away
        raise HTTPException(
            status_code=409, detail={"message": str(e), "entries": e.entries}
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    forget_workspace_usage()
    logger.info(f"Deleted workspace {name}")
    return {"name": name, "deleted": True, "contents": contents}


# How much of a long variable default the variables route shows before
# cutting it: enough to recognize a prompt by, far short of carrying one
VARIABLE_VALUE_PREVIEW = 200


@router.get("/api/workflows")
def list_workflows(
    request: Request,
    ws: Workspace = Depends(selected_workspace),
    shape: Optional[str] = None,
    traits: Optional[str] = None,
    configures: Optional[str] = None,
    include_models: bool = False,
    view: Optional[str] = None,
):
    """Every workflow the search path offers, each detail saying which
    source it came from and whether it can be written to. 'workflow_dir'
    stays the writable one - what a save targets.

    `shape`, `traits` (comma-separated, all must match) and `configures`
    narrow the listing; `view=compact` is the agent's view - summaries
    rather than descriptions, templates rather than model configs
    unless `include_models` asks for them. `workflows` always names
    exactly the entries `details` holds.
    """
    state = request.app.state
    sources = sources_for(state, ws)
    found = listing(sources)
    try:
        details = project_listing(
            attach_observed(
                workflow_details(found),
                getattr(state, "observed_costs", None),
                ws.name,
            ),
            shape=shape,
            traits=[t.strip() for t in (traits or "").split(",") if t.strip()],
            configures=configures,
            include_models=include_models,
            view=view,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "workspace": ws.name,
        "workflow_dir": ws.workflows,
        "sources": [source.to_dict() for source in sources],
        "workflows": sorted(details),
        "details": details,
        # What a `cost` is, and so what a null one means. Curated:
        # figures a maintainer measured once on the devices named and
        # wrote into the workflow - nothing derives them from this
        # server's own job history, so null means nobody wrote one
        # down, not that the run is cheap or that this box has never
        # run it (#91). A detail's `observed` block, when present, is the
        # other kind of number: this box's own finished runs of that
        # workflow, derived rather than claimed, and never a substitute
        # for `cost` (#93)
        "cost_basis": "curated",
    }


@router.put("/api/workflows/{name:path}")
def save_workflow(
    http_request: Request,
    name: str,
    request: JobRequest,
    ws: Workspace = Depends(selected_workspace),
):
    """Write a workflow into the writable workflow directory. The
    definition must be schema-valid - the editor validates before saving,
    and a save that silently wrote a broken file would betray both.

    A name that currently resolves to a read-only source (an example, a
    builtin) is not overwritten: the copy lands in the writable source
    and shadows it from then on.
    """
    state = http_request.app.state
    if request.workflow is None:
        raise HTTPException(
            status_code=400,
            detail='Provide the definition as {"workflow": {...}}',
        )
    path, source = resolve_writable_workflow(sources_for(state, ws), name)
    candidate = Workflow(
        copy.deepcopy(request.workflow),
        ws.outputs,
        path,
        ws.workflows,
    )
    try:
        candidate.validate()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(request.workflow, file, indent=2)
        file.write("\n")
    logger.info(f"Saved workflow {name} to {path}")
    # What the catalog will say about it, so the author sees the match
    # it just created. An empty summary is a warning, never a refusal:
    # a workflow with no description still runs, it is just invisible
    # to shape-first discovery
    metadata = derive_catalog_metadata(request.workflow)
    warnings = list(workflow_argument_warnings(request.workflow))
    warnings += candidate.null_variable_argument_warnings()
    if not metadata["summary"]:
        warnings.append(
            "No summary: add a 'description' (its first sentence becomes "
            "the catalog summary) or a 'summary' so the listing can say "
            "what this workflow is for"
        )
    return {
        "name": name,
        "workspace": ws.name,
        "origin": source.origin,
        "warnings": warnings,
        "shape": metadata["shape"],
        "traits": metadata["traits"],
        "summary": metadata["summary"],
    }


@router.delete("/api/workflows/{name:path}")
def delete_workflow(
    request: Request, name: str, ws: Workspace = Depends(selected_workspace)
):
    """Remove a workflow file from the writable workflow directory.

    A read-only source is refused rather than silently ignored: an
    example or a builtin is not the caller's to delete, and saying so
    is more useful than a 404 that reads like the file is missing.
    """
    state = request.app.state
    manager = request.app.state.job_manager
    path, source = resolve_readable_workflow(sources_for(state, ws), name)
    if not source.writable:
        raise HTTPException(
            status_code=403,
            detail=f"'{name}' comes from the read-only {source.origin} "
            f"directory {source.root} and cannot be deleted",
        )
    os.remove(path)
    logger.info(f"Deleted workflow {name} ({path})")
    forget_workspace_usage()
    # This identity's job history goes with it (#274) - otherwise a name
    # reused in this workspace, including by a regression cycle that
    # deletes and recreates the same workflow, would inherit the deleted
    # copy's observed figures and host-memory history
    manager.history.orphan_workflow_history(ws.name, name)
    return {
        "name": name,
        "workspace": ws.name,
        "origin": source.origin,
        "deleted": True,
    }


@router.get("/api/workflows/{name:path}/download")
@query_token_ok
def download_workflow(
    request: Request, name: str, ws: Workspace = Depends(selected_workspace)
):
    """Serve a workflow definition as a forced download."""
    state = request.app.state
    path, _source = resolve_readable_workflow(sources_for(state, ws), name)
    return FileResponse(
        path, filename=os.path.basename(path), media_type="application/json"
    )


# Declared before the catch-all below, which would otherwise swallow
# '<name>/variables' as a workflow called that
@router.get("/api/workflows/{name:path}/variables")
def get_workflow_variables(
    request: Request,
    name: str,
    full: bool = False,
    ws: Workspace = Depends(selected_workspace),
):
    """A workflow's variables and the values they default to.

    The listing says which variables a workflow has; confirming what one
    of them defaults to meant fetching the whole definition, quantization
    blocks and all, to read a single integer. This answers that question
    by itself.

    Long strings - a shot's prompt runs to kilobytes, and a list-driven
    workflow's default list holds several - are cut to their first 200
    characters wherever they sit and named in `truncated`
    (`shots[0].prompt`), so the answer stays small for the numbers and
    names it is usually asked about; `full=true` returns them whole, and
    `GET /api/workflows/{name}` is still the definition itself.
    """
    state = request.app.state
    path, source = resolve_readable_workflow(sources_for(state, ws), name)
    try:
        with open(path, "r") as file:
            definition = json.load(file)
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Could not read workflow: {e}")

    def preview(value, path):
        if isinstance(value, str) and len(value) > VARIABLE_VALUE_PREVIEW:
            truncated.append(path)
            return value[:VARIABLE_VALUE_PREVIEW]
        if isinstance(value, list):
            return [preview(item, f"{path}[{i}]") for i, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: preview(item, f"{path}.{key}") for key, item in value.items()}
        return value

    variables = definition.get("variables") or {}
    values, truncated = {}, []
    for variable, value in variables.items():
        values[variable] = value if full else preview(value, variable)
    answer = {
        "name": name,
        "variables": values,
        "truncated": truncated,
        "seed": definition.get("seed"),
        "origin": source.origin,
    }
    # The rule beside the default it constrains: a consumer reading
    # `num_frames: 124` with no range picked 61 and paid 138 s of
    # loading to be told the rule was 17n + 5 from 124 (#96)
    constraints = definition.get("variable_constraints")
    if isinstance(constraints, dict) and constraints:
        answer["constraints"] = constraints
    # What an entry of each list-driven variable carries, with any rule
    # that reaches one of its fields stated beside that field: a caller
    # reading what a `shots` entry takes reads the bound for
    # `num_frames` there, rather than having to match it to a key of
    # `constraints` that names no top-level variable (#145)
    lists = derive_catalog_metadata(definition).get("lists")
    if lists:
        answer["lists"] = lists
    # What this box's own runs of it actually took, beside the defaults
    # they were run with - derived, never the curated `cost` (#93)
    observed = observed_for_name(
        state, name, definition, workspace=ws.name if source.writable else None
    )
    if observed:
        answer["observed"] = observed
    return answer


@router.get("/api/workflows/{name:path}")
def get_workflow(
    request: Request, name: str, ws: Workspace = Depends(selected_workspace)
):
    state = request.app.state
    path, source = resolve_readable_workflow(sources_for(state, ws), name)
    try:
        with open(path, "r") as file:
            definition = json.load(file)
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Could not read workflow: {e}")
    # Which root it came from and whether a save would land here or
    # copy elsewhere - the editor reads these to offer save-in-place
    # only for a writable source, save-a-copy otherwise
    return JSONResponse(
        definition,
        headers={
            "X-Workflow-Origin": source.origin,
            "X-Workflow-Writable": "true" if source.writable else "false",
        },
    )


class PromptRequest(BaseModel):
    prompt: Dict[str, Any] = Field(description="The prompt definition to save")


@router.get("/api/prompt-schema")
def get_prompt_schema():
    """The JSON schema for stored prompts - the editor's diagnostics.
    Its own path, so a prompt named 'schema' cannot shadow it."""
    return JSONResponse(load_schema("prompt"))


def referenceable(name):
    try:
        validate_prompt_reference(name)
        return True
    except InvalidInputError:
        return False


def _find_prompt(state, name):
    """(path, writable) for the first root on the search path that holds
    this name. 404s when no root does, the way resolve_prompt_name does
    for a name that cannot be referenced at all."""
    for index, root in enumerate(prompt_roots(state)):
        try:
            # allow_create so a name that is simply absent from this root
            # is a miss to carry on from, rather than a 404 raised out of
            # the middle of the search
            path = resolve_prompt_name(root, name, allow_create=True)
        except HTTPException as error:
            # a name no workflow could reference is a miss too, not the
            # 400 a save would get for it
            raise HTTPException(status_code=404, detail=error.detail)
        if os.path.isfile(path):
            return path, index == 0
    raise HTTPException(status_code=404, detail=f"Unknown prompt: {name}")


@router.get("/api/prompts")
def list_prompts(
    request: Request,
    tag: str | None = None,
    intended_model: str | None = None,
    include_text: bool = True,
):
    state = request.app.state
    # A stray file too deep or oddly named can sit in the directory, but
    # no workflow could reference it - listing it would only invite that
    paths = {}
    origins = {}
    roots = prompt_roots(state)
    for index, root in enumerate(roots):
        for name in workflow_names(root):
            if referenceable(name) and name not in paths:
                paths[name] = os.path.join(root, f"{name}.json")
                origins[name] = WORKSPACE_ORIGIN if index == 0 else EXAMPLES_ORIGIN
    details = prompt_details(paths)

    # Narrowing happens after the details are read, since that is where a
    # prompt says what it is for, and it narrows every parallel key at
    # once: a `prompts` list and a `details` map that disagree is worse
    # than no filter at all
    wanted = matching_prompts(details, tag, intended_model)
    if wanted is not None:
        details = {name: detail for name, detail in details.items() if name in wanted}
    # The three parallel keys agree by construction, filter or no filter.
    # `prompt_details` drops a path whose mtime it cannot read - the file
    # went away between the walk and the read - and listing a name that
    # carries no detail only tells a caller to go and get a 404.
    origins = {name: origin for name, origin in origins.items() if name in details}

    # The MCP listing cannot carry 44 prompt bodies - it exceeds a client's
    # result cap and the listing becomes uncallable - but the editors read
    # `text` as the card fallback, so the omission is opt-in and the size
    # is reported in its place
    if not include_text:
        details = {
            name: {
                **{key: value for key, value in detail.items() if key != "text"},
                "text_chars": len(detail.get("text") or ""),
            }
            for name, detail in details.items()
        }

    return {
        # The writable library, unchanged: what a save is written to,
        # and what a client that predates the search path expects
        "prompt_dir": state.prompt_dir,
        "prompt_dirs": roots,
        "prompts": sorted(details),
        "origins": origins,
        "details": details,
    }


@router.put("/api/prompts/{name:path}")
def save_prompt(http_request: Request, name: str, request: PromptRequest):
    """Write a prompt into the prompt directory. Like a workflow save,
    the definition must be schema-valid before it lands on disk."""
    state = http_request.app.state
    status, message = validate_data(request.prompt, load_schema("prompt"))
    if not status:
        raise HTTPException(status_code=400, detail=message)
    if str(request.prompt.get("text", "")).startswith(RESERVED_TEXT_PREFIXES):
        raise HTTPException(
            status_code=400,
            detail="A prompt's text may not itself begin with a reference "
            f"prefix ({', '.join(RESERVED_TEXT_PREFIXES)})",
        )
    path = resolve_prompt_name(state.prompt_dir, name, allow_create=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(request.prompt, file, indent=2)
        file.write("\n")
    logger.info(f"Saved prompt {name} to {path}")
    return {"name": name}


@router.delete("/api/prompts/{name:path}")
def delete_prompt(request: Request, name: str):
    """Remove a prompt file from the prompt directory. A prompt that
    came from a read-only examples library is not this server's to
    delete - the same 403 a read-only workflow answers with."""
    state = request.app.state
    path, writable = _find_prompt(state, name)
    if not writable:
        raise HTTPException(
            status_code=403,
            detail=f"Prompt {name} is read-only: it comes from an examples "
            f"library, not this workspace's prompt directory",
        )
    os.remove(path)
    logger.info(f"Deleted prompt {name} ({path})")
    forget_workspace_usage()
    return {"name": name, "deleted": True}


@router.get("/api/prompts/{name:path}/download")
@query_token_ok
def download_prompt(request: Request, name: str):
    """Serve a stored prompt as a forced download."""
    state = request.app.state
    path, _ = _find_prompt(state, name)
    return FileResponse(
        path, filename=os.path.basename(path), media_type="application/json"
    )


@router.get("/api/prompts/{name:path}")
def get_prompt(request: Request, name: str):
    state = request.app.state
    path, writable = _find_prompt(state, name)
    try:
        with open(path, "r") as file:
            # Which library it came from, the way a workflow carries its
            # source - the editor offers delete only for a prompt this
            # server owns, and save-a-copy for a read-only one
            return JSONResponse(
                json.load(file),
                headers={
                    "X-Prompt-Origin": (
                        WORKSPACE_ORIGIN if writable else EXAMPLES_ORIGIN
                    ),
                    "X-Prompt-Writable": "true" if writable else "false",
                },
            )
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Could not read prompt: {e}")


class EnhanceRequest(BaseModel):
    idea: str = Field(description="The idea to expand into a full prompt")
    preset: str = Field(default="h3", description="Enhancer preset key")
    model_name: Optional[str] = Field(
        default=None, description="LLM repo id; the preset's default when omitted"
    )
    device: Optional[str] = Field(
        default=None,
        description="Device for the language model; defaults to cpu, "
        "keeping VRAM free for generation",
    )


@router.get("/api/enhancers")
def list_enhancers():
    return {"presets": preset_descriptions()}


@router.post("/api/enhance", status_code=201)
def enhance(
    http_request: Request,
    request: EnhanceRequest,
    ws: Workspace = Depends(selected_workspace),
):
    """Queue a prompt enhancement as an ordinary job. The enhanced text
    is the job's single manifest file once it succeeds.

    Scoped like any other job: the caller reads the result back from the
    workspace it asked in, so this has to write there too."""
    state = http_request.app.state
    manager = http_request.app.state.job_manager
    try:
        definition = build_enhance_workflow(
            request.preset,
            request.idea,
            model_name=request.model_name,
            device=request.device,
        )
        # Admitted like any other job - submit records and queues what
        # was admitted, it does not check
        admission = admit_for(
            state,
            ws,
            workflow_path=None,
            workflow=definition,
            arguments={},
            base_dir=None,
            output_dir=ws.outputs,
            workflow_dir=ws.workflows,
        )
        if not admission.ok:
            raise ValueError(admission.message())
        job = manager.submit(
            admitted=admission.workflow,
            workflow=definition,
            arguments={},
            workflow_dir=ws.workflows,
            output_dir=ws.outputs,
            asset_dir=ws.assets,
            workspace=ws.name,
            warnings=admission.warnings,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return manager.describe(job)
