"""FastAPI application exposing the workflow engine.

All state lives in the JobManager; this module is routing, validation and
SSE framing. Everything path-shaped goes through dw.security validators.
Interactive API docs are served at /docs (OpenAPI at /openapi.json).
"""

import os
import base64
import mimetypes
import shutil
import io
import zipfile
import tempfile
import copy
import json
import re
import uuid
import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from urllib.parse import quote, urlparse
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.routing import Match, Route
from starlette.background import BackgroundTask

from ..security import (
    validate_asset_reference,
    validate_path,
    validate_output_path,
    validate_prompt_reference,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    validate_commit_hash,
    InvalidInputError,
    PathTraversalError,
    SecurityError,
    workflows_are_trusted,
)
from ..introspection import (
    describe_class,
    list_classes,
    list_pipelines,
    describe_pipeline,
    list_tasks,
    describe_task,
    workflow_argument_warnings,
)
from ..for_each import entry_field_warnings
from ..schema import (
    load_schema,
    schema_section,
    validate_data,
    format_validation_errors,
    SchemaSectionError,
)
from ..prompts import (
    PROMPT_PREFIX,
    RESERVED_TEXT_PREFIXES,
    resolve_prompt_reference,
)
from ..assets import ASSET_PREFIX, is_asset_reference, resolve_asset_reference
from ..variable_constraints import constraint_errors, constraint_warnings
from .observed_cost import ObservedCosts, declared_drivers
from ..variables import argument_errors
from ..workflow import Workflow, workflow_from_definition, workflow_from_file
from .enhancers import build_enhance_workflow, preset_descriptions
from .exports import export_directory, export_job
from ..result import read_embedded_metadata
from ..media_info import probe_media
from ..media_audio import (
    MAX_INLINE_AUDIO_BYTES,
    NoSoundtrack,
    audio_shape,
    extract_audio,
    media_duration,
    projected_wav_base64_size,
)
from ..media_frames import (
    contact_sheet,
    frames_at,
    resolve_crop_box,
    seam_tiles,
    video_shape,
)
from ..hub_cache import scan_models, delete_model, DownloadManager
from ..host_memory_projection import CEILING_FRACTION, host_memory_warnings
from ..plan import build_plan, gate_warnings, unseeded_cache_warnings
from ..runs import (
    MANIFEST_FILE_NAME,
    OUTPUT_PREFIX,
    REALIZED_FILE_NAME,
    is_output_reference,
    is_run_id,
    record_run_versions,
    resolve_output_reference,
    run_versions,
    recorded_shots,
    split_run_path,
)
from ..workspace import (
    ASSETS_SUBDIR,
    DEFAULT_WORKSPACE_NAME,
    PROMPTS_SUBDIR,
    ConfiguredWorkspace,
    NotAWorkspaceError,
    Workspace,
    _holds_a_workspace,
    create_workspace,
    delete_workspace,
    example_libraries,
    forget_workspace_usage,
    named_workspace,
    workspace_contents,
    workspace_names,
    workspace_usage,
)
from ..workflow_sources import (
    COMMON_ORIGIN,
    EXAMPLES_ORIGIN,
    WORKSPACE_ORIGIN,
    find_workflow,
    listing,
    resolve_in_source,
    resolve_sub_workflow,
    source_for_path,
    workflow_names,
    workflow_sources,
    writable_source,
    SubWorkflowNotFound,
)
from .jobs import (
    ACK_BOOLEAN,
    ACK_BOUND,
    ACK_NONE,
    JobManager,
    MAX_PERSISTED_EVENTS,
    QUEUED,
    RUNNING,
    TERMINAL_STATES,
)
from .netinfo import local_addresses
from .updater import DiffusersUpdater
from .sysinfo import runtime_info
from .catalog_shape import derive_catalog_metadata, project_listing
from . import guides
from .guides import GuideError
from .. import settings

logger = logging.getLogger("dw")

# How long one SSE poll waits for a new event before checking liveness
SSE_POLL_SECONDS = 1.0


class AcknowledgedCost(BaseModel):
    """A cost acknowledgement bound to the plan a validate call answered
    with (#85): the server refuses to queue a run whose plan no longer
    matches it. `minutes` is recorded, never compared."""

    fingerprint: str = Field(description="plan.fingerprint from POST /api/validate")
    minutes: Optional[float] = Field(
        default=None, description="plan.estimate.minutes, recorded on the job"
    )
    downloads: List[str] = Field(
        default_factory=list,
        description="The repos in plan.downloads_required that were acknowledged",
    )


ACKNOWLEDGED_COST_FIELD = Field(
    default=None,
    description="Cost acknowledgement: true (recorded), or an object "
    "{fingerprint, minutes, downloads} bound to the plan validate answered "
    "with - then the run is refused with 409 if its plan changed",
)


class JobRequest(BaseModel):
    workflow_path: Optional[str] = Field(
        default=None, description="Path to a workflow JSON file on the server"
    )
    workflow: Optional[Dict[str, Any]] = Field(
        default=None, description="Inline workflow definition"
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Workflow variable overrides"
    )
    base_dir: Optional[str] = Field(
        default=None,
        description="Directory relative paths in an inline workflow resolve against",
    )
    workspace: Optional[str] = Field(
        default=None,
        description="Which workspace to run or resolve in; the default when omitted",
    )
    acknowledged_cost: Optional[Union[bool, AcknowledgedCost]] = ACKNOWLEDGED_COST_FIELD


# What a run directory holds besides its outputs - the files a run writes
# about itself. A run whose directory holds nothing else is an orphan
# (see _iter_orphan_runs, #170) whatever shape its output would have had.
# job.json is what an export bundle writes, listed defensively.
RUN_BOOKKEEPING_FILES = frozenset({MANIFEST_FILE_NAME, REALIZED_FILE_NAME, "job.json"})

# What each workflow produces and takes, for listing cards - cached by mtime
_workflow_detail_cache = {}


def _prune_detail_cache(cache, directory, names):
    """Forget files a listing no longer names - a long-lived server that
    creates and deletes scratch files would otherwise grow the cache forever.

    `names` are relative names under `directory`.
    """
    live = {os.path.join(directory, f"{name}.json") for name in names}
    for stale in [path for path in cache if path not in live]:
        del cache[stale]


def _prune_missing(cache):
    """Forget cached files that are gone from disk.

    Pruning by what one listing named would be wrong here: the workflow
    cache is shared by every workspace, and a listing only ever sees one
    workspace's search path, so anything cached for another workspace would
    be thrown away and re-parsed on the next switch. Existence is the test
    that holds for all of them at once.
    """
    for stale in [path for path in cache if not os.path.exists(path)]:
        del cache[stale]


def collect_prompt_references(value):
    """Every stored-prompt name a definition references, at any depth - so
    deleting a prompt can warn which workflows would break."""
    references = set()
    if isinstance(value, str):
        if value.startswith(PROMPT_PREFIX):
            references.add(value.removeprefix(PROMPT_PREFIX).strip())
    elif isinstance(value, dict):
        for item in value.values():
            references |= collect_prompt_references(item)
    elif isinstance(value, list):
        for item in value:
            references |= collect_prompt_references(item)
    return references


def _catalog_name_from_root(path, root):
    """The listing name a resolved workflow path has under a root.

    None when the path is not under the root after all - a name that does
    not name an entry is worse than no name for anything that later joins
    on it.
    """
    if root is None:
        return None
    relative = os.path.relpath(path, root)
    if relative.startswith(".."):
        return None
    return os.path.splitext(relative)[0].replace(os.sep, "/")


def catalog_name_for(path, source):
    """The listing name a resolved workflow path has within its source.

    None when the run came from an inline definition, or when the path is
    not under the source root after all - a name that does not name an
    entry is worse than no name for anything that later joins on it.
    """
    if source is None:
        return None
    return _catalog_name_from_root(path, source.root)


def attach_observed(details, observed_costs, workspace_name=None):
    """Fold this box's own history into each detail, as `observed`.

    Separate from `workflow_details` because that cache is keyed on a file's
    mtime and this figure changes when no file has: a job finishing moves
    every number here. A detail carries `cost_drivers` and the defaults they
    take, which is everything the aggregate needs - the file is not read a
    second time.

    A detail's own `writable` says whether its entry is this workspace's own
    copy or a shared catalog one (#274): only the former is scoped to
    `workspace_name`, so two workspaces' saves of the same name do not leak
    into each other's figure, while a template or example still pools every
    workspace's runs of it, matching #154.
    """
    if observed_costs is None or not observed_costs.refresh():
        return details
    for name, detail in details.items():
        drivers = detail.get("cost_drivers") or {}
        # The shape `observed_for` reads: the drivers with their defaults,
        # and the variable names, which is what the no-drivers fallback
        # (default-arguments-only runs) compares a job's arguments against
        surrogate = {
            "cost_drivers": sorted(drivers),
            "variables": {
                **{variable: None for variable in detail.get("variable_names") or []},
                **drivers,
            },
        }
        workspace = workspace_name if detail.get("writable") else None
        observed = observed_costs.observed(
            name, surrogate, fresh=False, workspace=workspace
        )
        if observed:
            detail["observed"] = observed
    return details


def workflow_details(sources_by_name):
    """Per-workflow card metadata: output kinds, step and variable counts,
    and the variable names themselves - enough for an agent to pick a
    workflow and know what to pass it without fetching each candidate, and,
    for a list-driven workflow, what an entry of each list carries. The
    names but not their defaults: across the workflows on disk the defaults
    are an order of magnitude more payload, on a listing the UI reloads.

    Takes the name -> source mapping the search path produced, so each
    entry also says where it came from and whether it can be written to -
    what a client needs to decide between offering save and offering
    save-a-copy.
    """
    details = {}
    for name, source in sources_by_name.items():
        path = os.path.join(source.root, f"{name}.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        cached = _workflow_detail_cache.get(path)
        if cached and cached[0] == mtime:
            # The cached detail is placement-free; the origin and writability
            # are the source's, and a warm cache must still carry them or a
            # second listing loses the fields a client decides save-vs-copy on
            details[name] = {
                **cached[1],
                "origin": source.origin,
                "writable": source.writable,
            }
            continue
        try:
            with open(path, "r") as file:
                definition = json.load(file)
            kinds = sorted(
                {
                    step["result"]["content_type"].split("/")[0]
                    for step in definition.get("steps", [])
                    if isinstance(step.get("result"), dict)
                    and "content_type" in step["result"]
                }
            )
            variables = definition.get("variables", {}) or {}
            metadata = derive_catalog_metadata(definition)
            cost = definition.get("cost")
            detail = {
                "kinds": kinds,
                "steps": len(definition.get("steps", [])),
                "variables": len(variables),
                "variable_names": sorted(variables),
                "description": str(definition.get("description", "") or ""),
                # Empty for a template; a catalog name for a model config, which
                # is what lets a client show the two as different kinds of thing
                "configures": str(definition.get("configures", "") or ""),
                "prompt_refs": sorted(collect_prompt_references(definition)),
                "shape": metadata["shape"],
                "traits": metadata["traits"],
                "summary": metadata["summary"],
                "lists": metadata["lists"],
                # What a variable's value is allowed to be, so the rule is
                # read rather than guessed at (#96)
                "constraints": definition.get("variable_constraints") or {},
                # The variables the author says move this workflow's cost,
                # with what they default to - what buckets this box's own
                # runs into comparable ones (#93). Carried here so an
                # observed figure needs no second read of the file
                "cost_drivers": {
                    name: (definition.get("variables") or {}).get(name)
                    for name in declared_drivers(definition)
                },
                "cost": cost if isinstance(cost, list) and cost else None,
            }
        except Exception:
            detail = {
                "kinds": [],
                "steps": 0,
                "variables": 0,
                "variable_names": [],
                "description": "",
                "prompt_refs": [],
                "shape": "utility",
                "traits": [],
                "summary": "",
                "lists": {},
                "constraints": {},
                "cost_drivers": {},
                "cost": None,
            }
        _workflow_detail_cache[path] = (mtime, detail)
        # Cached by content, not by placement: the same file listed from a
        # different source keeps its parsed detail and gets fresh origins
        details[name] = {
            **detail,
            "origin": source.origin,
            "writable": source.writable,
        }
    _prune_missing(_workflow_detail_cache)
    # A model config names its template as a catalog name. Resolve it here,
    # where the whole listing is in hand, so a badge is a link to a real card
    # rather than a string - and say which name did not resolve. A config
    # also takes its shape and traits from the template: what it makes is
    # the template's business, what it costs is its own. Entries can be the
    # very dict cached above (a cache hit skips the copy at the origin
    # merge), so copy before mutating - otherwise a stale "not found yet"
    # verdict would stick in the cache and outlive the typo once the
    # template it names is added.
    for name, detail in details.items():
        named = detail.get("configures", "")
        if not named:
            continue
        detail = dict(detail)
        template = details.get(named)
        if template is None:
            detail["configures_missing"] = named
            detail["configures"] = ""
        else:
            detail["shape"] = template["shape"]
            detail["traits"] = list(template["traits"])
        details[name] = detail
    return details


def _write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)


def resolve_readable_workflow(sources, name):
    """The path a name has anywhere on the search path, and its source.

    Reads span every root - the workspace's own workflows, any examples
    directory, and the packaged builtins - front to back, so a workspace
    copy shadows the example it came from.
    """
    path, source = find_workflow(sources, name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown workflow: {name}")
    return path, source


def resolve_writable_workflow(sources, name):
    """Where a save goes: always the writable source, whatever the name
    currently resolves to.

    Saving a workflow opened from an example is not an overwrite of that
    example - it is a copy into the user's own library, which is what makes
    the read-only roots safe to browse and edit from.
    """
    source = writable_source(sources)
    if source is None:
        raise HTTPException(
            status_code=409, detail="This server has no writable workflow directory"
        )
    path = resolve_in_source(source, name, allow_create=True)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown workflow: {name}")
    return path, source


def resolve_workflow_reference(workflow_path, sources):
    """A submitted workflow_path, resolved to a file on disk, and the source
    it lives in - the same search path the /api/workflows CRUD routes read
    from, spanning every root rather than confining to one, since a run of
    an example is a read and reads are not confined to the writable root.

    Tried as a stored workflow name first - exactly what /api/workflows
    hands out, with or without .json and nested names included - so an
    agent can run what a listing gave it. A relative or absolute path that
    already names a file under one of the sources resolves the same way:
    os.path.abspath handles a path relative to the server's cwd, and
    source_for_path holds it to that source's containment check.

    Anything that resolves under no source - an unknown name, a traversal
    attempt, or a real file elsewhere on disk - is rejected with 400,
    rather than silently opened: a workflow_path is not a general
    filesystem path.

    Returns (None, None) when workflow_path itself is None - an inline
    workflow submission names no path to resolve.
    """
    if workflow_path is None:
        return None, None
    path, source = find_workflow(sources, workflow_path)
    if path is not None:
        return path, source
    candidate = os.path.abspath(workflow_path)
    source = source_for_path(sources, candidate)
    if source is not None:
        # The containment check re-applied to the path this returns, rather
        # than trusted from source_for_path's answer about it - and applied
        # before anything asks the filesystem about the path, so a
        # workflow_path outside every source cannot be used to find out
        # whether a file exists there
        try:
            confined = validate_path(candidate, source.root, allow_create=False)
        except SecurityError:
            confined = None
        if confined is not None and os.path.isfile(confined):
            return confined, source
    raise HTTPException(
        status_code=400,
        detail=f"workflow_path must name a workflow the server can reach: "
        f"{workflow_path}",
    )


# What each prompt says about itself, for listing cards - cached by mtime
_prompt_detail_cache = {}


def prompt_details(paths):
    """Per-prompt card metadata: description, intended model, tags - and
    the text itself, which the editors show as the tooltip wherever a
    prompt: reference stands in for it.

    Keyed by path rather than by name under one directory: the prompt
    library is a search path now, and two roots can hold the same name.
    """
    details = {}
    for name, path in paths.items():
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        cached = _prompt_detail_cache.get(path)
        if cached and cached[0] == mtime:
            details[name] = cached[1]
            continue
        try:
            with open(path, "r") as file:
                definition = json.load(file)
            detail = {
                "description": str(definition.get("description", "") or ""),
                "intended_model": str(definition.get("intended_model", "") or ""),
                "tags": [str(tag) for tag in definition.get("tags", []) or []],
                "text": str(definition.get("text", "") or ""),
            }
        except Exception:
            detail = {"description": "", "intended_model": "", "tags": [], "text": ""}
        _prompt_detail_cache[path] = (mtime, detail)
        details[name] = detail
    # By existence, not by what this listing named: the cache spans every
    # root on the search path, and one listing shows only the names that
    # were not shadowed
    _prune_missing(_prompt_detail_cache)
    return details


def _matching_prompts(details, tag, intended_model):
    """The prompt names matching the filters, or None when no filter was given.

    Case-insensitive and exact per value: a `tags` entry or the whole
    `intended_model`, never a substring - `minimax-music` must not match
    `minimax-music3`, which is the confusion the one-spelling-per-family
    rule exists to prevent.
    """
    if tag is None and intended_model is None:
        return None
    wanted_tag = tag.lower() if tag is not None else None
    wanted_model = intended_model.lower() if intended_model is not None else None
    matches = set()
    for name, detail in details.items():
        if wanted_tag is not None and wanted_tag not in {
            str(each).lower() for each in detail.get("tags") or []
        }:
            continue
        if (
            wanted_model is not None
            and str(detail.get("intended_model") or "").lower() != wanted_model
        ):
            continue
        matches.add(name)
    return matches


def resolve_prompt_name(prompt_dir, name, allow_create=False):
    """The on-disk path for a prompt name, confined to prompt_dir.

    The name is held to the same rule 'prompt:' references enforce - a save
    the API accepted but no workflow could ever reference would be a trap. A
    save is told what is wrong with the name; a read just misses."""
    bare = name.removesuffix(".json")
    try:
        validate_prompt_reference(bare)
    except InvalidInputError as e:
        status = 400 if allow_create else 404
        raise HTTPException(status_code=status, detail=str(e))
    try:
        return validate_path(
            os.path.join(prompt_dir, f"{bare}.json"),
            prompt_dir,
            allow_create=allow_create,
        )
    except SecurityError as e:
        raise HTTPException(status_code=404, detail=f"Unknown prompt: {e}")


def default_ui_dir():
    """Where the built SPA lives: ui/dist in a checkout (the copy npm just
    built), else the copy packaged into the wheel at dw/server/ui, else None."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(here)), "ui", "dist"),
        os.path.join(here, "ui"),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "index.html")):
            return candidate
    return None


def _historical_log_note(stored):
    """What a restored job's event page has to admit about itself.

    History keeps only the last MAX_PERSISTED_EVENTS of a run, so a page can
    be complete as a page and still be missing the start of the job. The
    first stored event's seq is the direct signal: anything above zero means
    the head was dropped at record time. Length is not the signal - a job
    that emitted exactly MAX_PERSISTED_EVENTS events lost nothing.
    """
    if not stored:
        return "This job kept no event log - events were not retained with job history."
    if stored[0].get("seq", 0) > 0:
        return (
            f"Only the last {MAX_PERSISTED_EVENTS} events of this job were "
            f"retained; everything before seq {stored[0]['seq']} was dropped "
            "when the job was recorded."
        )
    return None


#  Host header values a locally-bound server accepts by default, regardless
# of what --host is configured to - a loopback request always presents one
# of these regardless of the server's own bind address.
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
# Bind addresses that mean "every interface" - a request never carries one
# of these as its Host, so they define no allowlist
WILDCARD_HOSTS = {"0.0.0.0", "::", ""}
# Where the MCP endpoint is mounted when --mcp is given (see the mcp block
# at the bottom of create_app) - the Server page quotes it in the command
# it tells you to run on the other machine
MCP_PATH = "/mcp"


def query_token_ok(fn):
    """Mark a GET endpoint as one a browser loads without being able to set
    headers (EventSource, an <img> tag, an <a download> navigation) - only
    routes carrying this marker accept the bearer token as a ?token= query
    param. Matched by the actual route at request time, not by a path
    suffix, so a resource that merely happens to be named "download" or
    "thumbnail" does not inherit the allowance."""
    fn.query_token_ok = True
    return fn


def _matched_route(request: Request):
    """Resolve the Route (if any) that will handle this request. Runs in
    middleware, before routing has attached anything to request.scope, so
    routes are matched by hand against request.app.router.routes. Skips
    non-Route entries (the SPA static Mount) and routes with no endpoint.

    A HEAD request path-matches a GET-only route as Match.PARTIAL (method
    mismatch) rather than Match.FULL, since this route is declared with
    methods=["GET"] and nothing here adds HEAD to it - but a HEAD request
    is still the same header-less browser load a GET would be, so it is
    treated the same for the query-token allowance."""
    method = request.scope.get("method")
    for route in request.app.router.routes:
        if not isinstance(route, Route) or route.endpoint is None:
            continue
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return route
        if (
            match == Match.PARTIAL
            and method == "HEAD"
            and route.methods
            and "GET" in route.methods
        ):
            return route
    return None


def create_app(
    workflow_dir="./workflows",
    output_dir="./outputs",
    log_level="INFO",
    job_manager=None,
    ui_dir=None,
    download_manager=None,
    diffusers_updater=None,
    prompt_dir="./prompts",
    asset_dir=None,
    examples_dirs=None,
    workspace=None,
    host="127.0.0.1",
    token=None,
    mcp=False,
    port=8765,
):
    """Build the application. A caller (tests) can inject a JobManager.

    `host` is the address the server is bound to (informational here - it
    is added to the Host-header allowlist alongside the loopback names, so
    a deployment bound to one specific non-loopback address still accepts
    its own requests). `token`, if given, is a static bearer token required
    on every /api/* request - see require_bearer_token below.
    """
    manager = job_manager or JobManager(
        output_dir, log_level=log_level, workflow_dir=workflow_dir
    )
    # An injected manager must confine jobs to the same workflow_dir the
    # routes do, or /api/validate and /api/jobs would enforce different
    # boundaries
    if manager.workflow_dir is None:
        manager.workflow_dir = workflow_dir
    elif manager.workflow_dir != workflow_dir:
        raise ValueError(
            "job_manager.workflow_dir must match the app's workflow_dir: "
            f"{manager.workflow_dir!r} != {workflow_dir!r}"
        )

    mcp_asgi = mcp_server = mcp_client = None
    if mcp:
        from .mcp_mount import build_mcp_app

        mcp_asgi, mcp_server, mcp_client = build_mcp_app(
            host=host, port=port, token=token
        )

    @asynccontextmanager
    async def lifespan(app):
        if mcp_server is None:
            yield
        else:
            # the SDK's session manager is the mounted app's own lifespan,
            # which Starlette does not run for a sub-app
            try:
                async with mcp_server.session_manager.run():
                    yield
            finally:
                # the client owns a connection pool; a session manager that
                # fails to start must not leak it
                mcp_client.close()
        manager.shutdown()

    app = FastAPI(
        title="diffusers-workflow",
        description="Declarative diffusers workflows over HTTP: queue a job, "
        "stream its progress, fetch what it saved.",
        lifespan=lifespan,
    )
    app.state.job_manager = manager
    # This box's own job history as a cost, recomputed when the jobs table
    # moves rather than when a file does - a job landing changes every
    # figure and changes no workflow file (#93)
    app.state.observed_costs = ObservedCosts(getattr(manager, "history", None))
    app.state.workflow_dir = workflow_dir
    # The search path: the writable directory first, then read-only roots -
    # any --examples-dir, then the packaged builtins. Reads span all of it,
    # saves only ever reach the front
    app.state.workflow_sources = workflow_sources(workflow_dir, examples_dirs)
    app.state.prompt_dir = prompt_dir
    # The read-only libraries the --examples-dir trees bring with them: an
    # example workflow references the prompts and assets that live beside
    # its tree, not the ones in this workspace. They are searched after the
    # workspace's own and never written to - a save of an example prompt
    # lands in the workspace, the way saving an example workflow does
    _example_libraries = example_libraries(examples_dirs)
    app.state.example_prompt_dirs = _example_libraries[PROMPTS_SUBDIR]
    app.state.example_asset_dirs = _example_libraries[ASSETS_SUBDIR]
    # Where uploads land and 'asset:' references resolve. None when the
    # caller configured no asset library: uploads then fall back to the
    # output directory's uploads/ subfolder, as they did before there was one
    app.state.asset_dir = os.path.abspath(asset_dir) if asset_dir else None
    # The workspace the three directories above default to folders of, for a
    # client that wants to name the root rather than reason about the parts.
    # None when the caller resolved no workspace (a test building an app
    # around three explicit directories)
    app.state.workspace = os.path.abspath(workspace) if workspace else None
    # The root that holds named workspaces. Its own folders are the default
    # workspace - which is what the three directories above already point at,
    # so a server given individual directory overrides simply has one
    # workspace and no others
    app.state.workspace_root = (
        Workspace(app.state.workspace, "flag") if app.state.workspace else None
    )
    # The default workspace itself, as a Workspace: its four folders are the
    # configured directories above, not '<root>/workflows' and friends - a
    # caller can override any one of them individually (--workflow-dir,
    # etc), so they cannot be derived from a root the way a named
    # workspace's folders are
    app.state.default_workspace = ConfiguredWorkspace(
        workflows=app.state.workflow_dir,
        assets=app.state.asset_dir,
        outputs=manager.output_dir,
        prompts=app.state.prompt_dir,
        root=app.state.workspace,
    )
    app.state.mcp_mounted = mcp_asgi is not None
    # A StaticFiles instance per output/asset root, built lazily and reused -
    # a mount is bound to one directory at startup, but a named workspace's
    # root does not exist yet then. Keeping the instance around (rather than
    # building one per request) is what makes /outputs and /inputs answer
    # ETag/If-None-Match with 304 and Range with 206 the way a real mount
    # does, instead of the plain FileResponse this replaced always resending
    # the whole file
    app.state.static_files_by_root = {}

    wildcard_bind = host in WILDCARD_HOSTS
    allowed_hosts = set(LOOPBACK_HOSTS)
    if host and not wildcard_bind:
        allowed_hosts.add(host.lower())

    @app.middleware("http")
    async def reject_foreign_origins(request, call_next):
        """Refuse browser cross-origin requests - a drive-by web page must
        not be able to queue jobs on this server. Requests without an
        Origin header (curl, scripts, same-origin GETs) pass.

        An Origin is accepted when its hostname is a loopback name, the
        configured bind host, or the hostname the request itself was
        addressed to (same-origin). The last clause is what lets a browser
        on another machine use a `--host 0.0.0.0` server by its LAN IP or
        hostname - and it stays safe against DNS rebinding, where the
        attacker's page carries its own Origin while Host is whatever
        resolved: the two differ, so the request is refused. Scheme and
        port are ignored, matching the Host check: a TLS-terminating proxy
        forwards Host unchanged while the browser's Origin is https."""
        origin = request.headers.get("origin")
        if origin:
            origin_host = (urlparse(origin).hostname or "").lower()
            request_host = (request.url.hostname or "").lower()
            # origin_host must be non-empty for the same-origin clause:
            # `Origin: null` (a sandboxed iframe, a file:// page) parses to
            # no hostname and would otherwise match a request whose Host
            # carries none either
            if origin_host not in allowed_hosts and not (
                origin_host and origin_host == request_host
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin requests are not allowed"},
                )
        return await call_next(request)

    # Defense-in-depth for requests that carry no Origin at all (curl,
    # scripts, the MCP client) and so skip the check above entirely: a
    # request that arrived on this port but claims to be addressed to some
    # unrelated public domain is rejected. This does not stop DNS rebinding
    # by itself (the Origin check already does, since a browser's Origin
    # header reflects the real requesting origin regardless of DNS) - it
    # only closes the gap for non-browser clients that never send Origin.
    # A wildcard bind is reached by whatever address the machine has - a LAN
    # IP, a hostname - never by the bind string itself, so there is no
    # allowlist to build; the Host check is skipped for it.
    @app.middleware("http")
    async def reject_foreign_hosts(request, call_next):
        hostname = request.url.hostname
        if (
            not wildcard_bind
            and hostname is not None
            and hostname.lower() not in allowed_hosts
        ):
            return JSONResponse(
                status_code=400,
                content={"detail": "Unrecognized Host header"},
            )
        return await call_next(request)

    @app.middleware("http")
    async def require_bearer_token(request: Request, call_next):
        """Static bearer-token auth (opt-in via --token / DW_API_TOKEN).
        Only /api/* is gated - the UI's own static files and /outputs (an
        <img>/<script> tag cannot attach an Authorization header anyway)
        stay reachable so the page can load far enough to let a user enter
        the token in the first place. EventSource cannot set custom headers
        either, and neither can the <img> tags the gallery grid loads its
        thumbnails through nor the <a download> navigations the download
        buttons make, so those GET routes additionally accept the token as a
        `token` query parameter - a documented trade-off, not a header-auth
        peer."""
        if not token:
            return await call_next(request)
        path = request.url.path
        if not (path.startswith("/api/") or path == "/mcp" or path.startswith("/mcp/")):
            return await call_next(request)
        provided = None
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[len("bearer ") :].strip()
        # GET/HEAD only, and only on a route explicitly marked
        # query_token_ok - matched against the real route (see
        # _matched_route), not by a path suffix, so a resource that
        # happens to be named "download" or "thumbnail" does not inherit
        # the allowance meant for the real routes.
        if provided is None and request.method in ("GET", "HEAD"):
            route = _matched_route(request)
            if route is not None and getattr(route.endpoint, "query_token_ok", False):
                provided = request.query_params.get("token")
        # compared as bytes: compare_digest refuses non-ASCII str
        if provided is None or not secrets.compare_digest(
            provided.encode("utf-8"), token.encode("utf-8")
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid bearer token"},
            )
        return await call_next(request)

    # -------------------------------------------------------- workspace lookup

    def _workspace_root():
        root = app.state.workspace_root
        if root is None:
            raise HTTPException(
                status_code=409,
                detail="This server has no workspace root - it was started "
                "with individual directory overrides, so it has one "
                "workspace and cannot create others",
            )
        return root

    def _workspace_for(name):
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
            return app.state.default_workspace
        root = _workspace_root()
        try:
            selected = named_workspace(root, name)
        except SecurityError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not _holds_a_workspace(selected.root):
            raise HTTPException(status_code=404, detail=f"No such workspace: {name}")
        return selected

    def selected_workspace(workspace: Optional[str] = None) -> Workspace:
        """FastAPI dependency form of _workspace_for, reading the name from
        the `?workspace=` query parameter every scoped route already takes -
        used as `ws: Workspace = Depends(selected_workspace)`."""
        return _workspace_for(workspace)

    def _sources_for(ws):
        """The workflow search path of one workspace: its own workflows
        first, then the same read-only roots every workspace shares."""
        return workflow_sources(ws.workflows, examples_dirs)

    # ------------------------------------------------------------------ jobs

    def _acknowledgement_form(value):
        """none | boolean | bound - classified once, here, so the check and
        the record agree (#85)."""
        if isinstance(value, AcknowledgedCost):
            return ACK_BOUND
        return ACK_BOOLEAN if value is True else ACK_NONE

    def _check_bound_acknowledgement(candidate, arguments, acknowledged, workspace):
        """Refuse with 409 when the run `candidate` + `arguments` will
        execute is not the one `acknowledged` was bound to: a different
        fingerprint, or a download the caller did not acknowledge. The body
        carries the current plan so the agent re-quotes from it without a
        second validate call. A plan that cannot be built is a refusal too -
        never a silent pass (#85).
        """
        record = acknowledged.model_dump()

        def refuse(message, reason, plan):
            raise HTTPException(
                status_code=409,
                detail={
                    "message": message,
                    "reason": reason,
                    "acknowledged": record,
                    "plan": plan,
                },
            )

        try:
            from .. import get_device, get_device_type

            current = build_plan(
                candidate,
                arguments,
                device=get_device_type(get_device()),
                prompt_dir=workspace.prompts,
                lookup_sizes=False,
            )
        except Exception:
            logger.exception("Plan could not be built for a bound acknowledgement")
            refuse(
                "The run could not be planned, so a bound acknowledgement "
                "cannot be checked; acknowledge with true or validate again",
                "unplannable",
                None,
            )
        current["workspace"] = workspace.name
        current["output_dir"] = workspace.outputs
        if current["fingerprint"] != acknowledged.fingerprint:
            refuse(
                "The run's shape changed since it was acknowledged: the "
                "workflow or its arguments differ from what was validated.",
                "fingerprint",
                current,
            )
        missing = [
            entry["repo"]
            for entry in current["downloads_required"]
            if entry.get("repo") and entry["repo"] not in acknowledged.downloads
        ]
        if missing:
            refuse(
                "The run's shape changed since it was acknowledged: it now "
                f"has to download {', '.join(missing)} first",
                "downloads",
                current,
            )

    def _candidate_for(
        workflow_path, workflow, base_dir, output_dir, workflow_dir, arguments
    ):
        """The Workflow a job spec names, built and checked as submit() will
        build and check it - schema first, then the caller's arguments -
        so a bound acknowledgement never turns the caller's 400 into a 409
        telling them to acknowledge with true and find out.

        Raises what submit() raises (ValueError, SecurityError, ...), which
        the routes already answer as 400.
        """
        if workflow_path is not None:
            candidate = workflow_from_file(workflow_path, output_dir, workflow_dir)
        else:
            candidate = workflow_from_definition(
                copy.deepcopy(workflow), output_dir, base_dir, workflow_dir
            )
        candidate.validate()
        problems = argument_errors(candidate.workflow_definition, arguments)
        # A value outside a rule the workflow declares, refused before the
        # job id rather than after the weights are loaded (#96)
        problems += constraint_errors(
            candidate.workflow_definition, arguments, supplied=set(arguments or {})
        )
        if problems:
            raise ValueError(
                "; ".join(
                    f"{problem['path']}: {problem['message']}" for problem in problems
                )
            )
        return candidate

    @app.post("/api/jobs", status_code=201)
    def submit_job(request: JobRequest, ws: Workspace = Depends(selected_workspace)):
        """Queue a workflow. The workspace it runs in comes from the body or,
        for a client that scopes every call the same way, the query string -
        the body wins when both are given."""
        try:
            workspace = _workspace_for(request.workspace or ws.name)
            sources = _sources_for(workspace)
            resolved, source = resolve_workflow_reference(
                request.workflow_path, sources
            )
            # Built unconditionally - both the reference check below and a
            # bound acknowledgement (further down) need the definition a run
            # would actually use, and resolve_workflow_reference already
            # returns (None, None) for an inline definition, which
            # _candidate_for handles the same way _candidate_for always has
            candidate = _candidate_for(
                resolved,
                request.workflow,
                request.base_dir,
                workspace.outputs,
                source.root if source else workspace.workflows,
                request.arguments,
            )
            # The same reference check POST /api/validate makes, because a
            # caller who skipped the free pre-flight should still not get a
            # job id for an argument that cannot resolve. The name half of
            # this check lives in JobManager.submit, where the definition is
            # loaded; this half needs the workspace's search path, which is
            # here - which is why a bad 'asset:' used to queue and die on the
            # first step while a bad variable name was refused outright
            reference_problems = _argument_reference_errors(
                candidate.workflow_definition, request.arguments, workspace
            )
            if reference_problems:
                raise ValueError(
                    "; ".join(
                        f"{problem['path']}: {problem['message']}"
                        for problem in reference_problems
                    )
                )
            # A bound acknowledgement is checked against the plan this
            # request would run - before anything is queued, since a refusal
            # is free here and costs a job id anywhere later (#85)
            form = _acknowledgement_form(request.acknowledged_cost)
            if form == ACK_BOUND:
                _check_bound_acknowledgement(
                    candidate, request.arguments, request.acknowledged_cost, workspace
                )
            job = manager.submit(
                workflow_path=resolved,
                workflow=request.workflow,
                arguments=request.arguments,
                base_dir=request.base_dir,
                # The root this run is confined to: the source the workflow
                # came from, so an example runs where it lives while an
                # inline definition stays held to this workspace's own
                # workflows
                workflow_dir=source.root if source else workspace.workflows,
                # The roots this job runs against, so it stays in its
                # workspace however many others the server serves meanwhile
                output_dir=workspace.outputs,
                asset_dir=workspace.assets,
                workspace=workspace.name,
                # The listing name, when the request came as one - what a
                # later runtime-by-workflow report joins on. Derived from
                # the resolved path rather than echoing what was asked
                # for, so 'Basic', 'Basic.json' and an absolute path
                # inside the source all record the one catalog name
                catalog_name=catalog_name_for(resolved, source),
                acknowledged=form,
                acknowledged_cost=(
                    request.acknowledged_cost.model_dump()
                    if form == ACK_BOUND
                    else None
                ),
            )
        except HTTPException:
            raise
        except Exception as e:
            # workflow_from_file / validate / the security layer all raise for
            # bad requests - every failure here is the client's fault
            raise HTTPException(status_code=400, detail=str(e))
        return manager.describe(job)

    @app.get("/api/jobs")
    def list_jobs(
        workspace: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """All jobs by default - a plain filter, not `selected_workspace`,
        since the jobs list spans every workspace the server holds unless a
        caller asks to narrow it.

        `status` narrows to one state or a comma-separated set of them.
        `limit` keeps the newest N, and `total` always reports how many
        matched before the cut, so a caller can tell a bounded answer from a
        complete one. The default is still every matching job, oldest first -
        what the web UI polls."""
        statuses = [part.strip() for part in status.split(",")] if status else None
        statuses = [part for part in statuses if part] if statuses else None
        if statuses:
            unknown = [
                state
                for state in statuses
                if state not in (QUEUED, RUNNING, *TERMINAL_STATES)
            ]
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown job status {', '.join(unknown)} - one of "
                    f"{', '.join((QUEUED, RUNNING, *TERMINAL_STATES))}",
                )
        jobs = manager.list(workspace=workspace, statuses=statuses)
        total = len(jobs)
        if limit is not None:
            if limit < 0:
                raise HTTPException(
                    status_code=400, detail="limit must not be negative"
                )
            # the newest are the interesting ones, and the list is oldest
            # first - so the cut comes off the front, not the back. max(0, ...)
            # because a limit above what matched is no cut at all: a bare
            # negative start would be read from the end instead, and answer a
            # limit of 12 against 9 matching jobs with the last 3 of them
            jobs = jobs[max(0, len(jobs) - limit) :] if limit else []
        return {"jobs": jobs, "total": total}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        # a historical job is already a detail dict; a live one renders itself
        return job if isinstance(job, dict) else manager.describe(job)

    @app.get("/api/jobs/{job_id}/workflow")
    def get_job_workflow(job_id: str):
        """The workflow this job ran, for the read-only graph on the job page
        and for `get_job_workflow` over MCP.

        `realized: true` means every mutable input is pinned - the copy the
        run itself wrote. `false` means the job predates run tracking (or its
        run directory is gone) and this is the definition as submitted. 404
        when neither is readable - the job itself still is."""
        if manager.get(job_id) is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        realized = manager.realized(job_id)
        definition = realized if realized is not None else manager.definition(job_id)
        if definition is None:
            raise HTTPException(
                status_code=404, detail="No workflow definition for this job"
            )
        return {
            "id": job_id,
            "definition": definition,
            "realized": realized is not None,
            # Which variable a new-seed rerun would draw into, or null when
            # there is none - read from the workflow as written, since the
            # realized copy above has its seed pinned to the integer it used
            "seed_variable": manager.seed_variable(job_id),
        }

    class RerunRequest(BaseModel):
        new_seed: bool = Field(
            default=False,
            description="Draw a fresh seed into the workflow's seed variable. "
            "Without it a rerun repeats the original arguments exactly, which "
            "the step cache serves from the earlier run - the same seed and "
            "inputs would produce the same files.",
        )
        acknowledged_cost: Optional[Union[bool, AcknowledgedCost]] = (
            ACKNOWLEDGED_COST_FIELD
        )

    @app.post("/api/jobs/{job_id}/rerun", status_code=201)
    def rerun_job(job_id: str, body: RerunRequest = RerunRequest()):
        """Queue a fresh job from a previous job's stored spec. Takes
        `acknowledged_cost` as POST /api/jobs does; a bound one is checked
        against the stored spec's plan - the fresh seed of `new_seed` does
        not change a fingerprint."""
        form = _acknowledgement_form(body.acknowledged_cost)
        if form == ACK_BOUND:
            prepared = manager.rerun_spec(job_id)
            if prepared is None:
                raise HTTPException(status_code=404, detail="Unknown job")
            spec, arguments = prepared
            try:
                candidate = _candidate_for(
                    spec.get("workflow_path"),
                    spec.get("workflow"),
                    spec.get("base_dir"),
                    spec.get("output_dir") or manager.output_dir,
                    spec.get("workflow_dir"),
                    arguments,
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
            _check_bound_acknowledgement(
                candidate,
                arguments,
                body.acknowledged_cost,
                _workspace_for(spec.get("workspace")),
            )
        try:
            job = manager.rerun(
                job_id,
                new_seed=body.new_seed,
                acknowledged=form,
                acknowledged_cost=(
                    body.acknowledged_cost.model_dump() if form == ACK_BOUND else None
                ),
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return manager.describe(job)

    @app.post("/api/jobs/{job_id}/export", status_code=201)
    def export_job_route(
        job_id: str,
        overwrite: bool = False,
        ws: Workspace = Depends(selected_workspace),
    ):
        """Gather one finished job into '<workspace>/exports/<job id>/': the
        workflow it ran, the run's manifest, the job row, the media it used
        and the media it made, plus a README. 404 for an unknown job, 409 for
        one still running or for an export that already exists without
        `overwrite`.

        The three JSON files come back inline as well as on disk - the
        directory is on the server, and a client on another machine has no
        other way to read them without fetching the zip."""
        try:
            summary = export_job(
                manager,
                job_id,
                ws.root,
                _asset_roots_for_job(job_id, ws),
                overwrite=overwrite,
            )
        except FileExistsError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except ValueError as e:
            message = str(e)
            if message.startswith("Unknown job"):
                raise HTTPException(status_code=404, detail=message)
            raise HTTPException(status_code=409, detail=message)
        body = summary.as_dict()
        zip_path = f"/exports/{quote(job_id)}.zip"
        body["zip_url"] = _served_url(zip_path, ws)
        absolute_zip_url = _absolute_served_url(zip_path, ws)
        if absolute_zip_url is not None:
            body["absolute_zip_url"] = absolute_zip_url
        # Same rule get_server_info's field states (#353): whether the zip
        # URL above needs a bearer token an MCP-only agent has no way to
        # attach itself, which is what tells the caller whether to fetch it
        # or hand it to the person.
        body["auth_required"] = bool(token)
        for key, name in (
            ("workflow", "workflow.json"),
            ("manifest", "manifest.json"),
            ("job", "job.json"),
        ):
            try:
                with open(os.path.join(summary.directory, name), "r") as file:
                    body[key] = json.load(file)
            except (OSError, ValueError):
                body[key] = None
        return body

    class MoveRequest(BaseModel):
        direction: str = Field(description="up, down, front, or back")

    @app.post("/api/jobs/{job_id}/move")
    def move_job(job_id: str, body: MoveRequest):
        """Reorder a queued job. 409 once it is running or finished -
        only the waiting portion of the queue can be rearranged."""
        if manager.get(job_id) is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        try:
            order = manager.move(job_id, body.direction)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if order is None:
            raise HTTPException(
                status_code=409, detail="Job is not queued - only queued jobs move"
            )
        return {"id": job_id, "queue": order}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        status = manager.cancel(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return {"id": job_id, "status": status}

    @app.get("/api/jobs/{job_id}/events")
    @query_token_ok
    async def job_events(request: Request, job_id: str, after: int = -1):
        """Server-sent events: every progress event from `after` (exclusive)
        until the job reaches a terminal state. Reconnect with the last seen
        seq (or let EventSource send Last-Event-ID) to resume without loss."""
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        if isinstance(job, dict):
            # historical jobs carry no event log - an immediately-closed
            # stream lets clients treat them uniformly
            return StreamingResponse(iter(()), media_type="text/event-stream")

        last_event_id = request.headers.get("last-event-id")
        if last_event_id is not None:
            try:
                after = max(after, int(last_event_id))
            except ValueError:
                pass

        async def stream():
            last_seq = after
            while True:
                events = job.events_after(last_seq)
                for event in events:
                    last_seq = event["seq"]
                    yield f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n"
                if job.status in TERMINAL_STATES and not job.events_after(last_seq):
                    return
                await asyncio.to_thread(job.wait_for_event, last_seq, SSE_POLL_SECONDS)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{job_id}/event-log")
    def job_event_log(job_id: str, after: int = -1, limit: int = 200):
        """Job events as one JSON page rather than a stream, for clients that
        poll instead of holding a connection open (the MCP server). `after` is
        exclusive, matching the SSE route's parameter of the same name."""
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        limit = max(1, min(limit, 1000))
        if isinstance(job, dict):
            # A job restored from sqlite: history persists a bounded tail of
            # its events, so it can still explain itself after a restart
            status = job.get("status")
            stored = manager.history.events_for(job_id) or []
            pending = [event for event in stored if event.get("seq", -1) > after]
            note = _historical_log_note(stored)
        else:
            status = job.status
            pending = job.events_after(after)
            note = None
        page = pending[:limit]
        return {
            "id": job_id,
            "status": status,
            "events": page,
            "last_seq": page[-1]["seq"] if page else max(after, -1),
            # this page is cut short; `note` covers what record time dropped
            "truncated": len(pending) > len(page),
            "note": note,
        }

    # ---------------------------------------------------------- introspection

    @app.get("/api/pipelines")
    def pipelines():
        """Every pipeline class the installed diffusers exports."""
        return {"pipelines": list_pipelines()}

    @app.get("/api/pipelines/{name}")
    def pipeline_description(name: str):
        """A pipeline's __call__ argument schema, for form generation."""
        try:
            return describe_pipeline(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            # A pipeline whose import fails on this install (missing extra
            # dependency) is absent, not a server error
            raise HTTPException(status_code=404, detail=f"Could not load {name}: {e}")

    @app.get("/api/tasks")
    def tasks():
        """Every task command a workflow's task step can name."""
        return list_tasks()

    @app.get("/api/tasks/{command}")
    def get_task(command: str):
        """A task command's argument schema - the registered implementation
        function's real signature, in the same shape as a class description."""
        try:
            return describe_task(command)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/classes")
    def classes(kind: str):
        """Class names of one kind (pipelines, models, schedulers,
        quantization) - the pickers' data source."""
        try:
            return {"kind": kind, "classes": list_classes(kind)}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/classes/{name:path}")
    def class_description(name: str, target: str = "init"):
        """A class's argument schema: target=call reads __call__, init reads
        __init__, load reads from_pretrained plus the curated loading knobs."""
        try:
            return describe_class(name, target=target)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not load {name}: {e}")

    @app.get("/api/schema")
    def workflow_schema(section: Optional[str] = None):
        """The workflow JSON schema, for schema-aware JSON editing.

        `?section=` answers one part of it - `steps`, `pipelines`, `tasks`,
        `result`, `variables` or `configuration` - as
        `{section, sections, elsewhere, schema}`, for a reader that wants
        the shape of a result block and not 36 KB of quantization configs
        (#101). Additive: the no-argument call is the whole schema, as it
        was. An unknown section is a 404 naming the ones that exist."""
        schema = load_schema("workflow")
        if section is None:
            return JSONResponse(schema)
        try:
            return JSONResponse(schema_section(schema, section))
        except SchemaSectionError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # ------------------------------------------------------------ guides

    @app.get("/api/guides")
    def list_guides():
        """The documentation that bears on choosing a capability: each
        guide's name, what it covers, and its section headings. Served by
        the engine rather than read from an MCP client's install, so the
        guides an agent reads are the guides for the engine it drives."""
        return guides.list_guides()

    @app.get("/api/guides/{name}")
    def get_guide(name: str, section: Optional[str] = None):
        """One guide from /api/guides, whole or one section of it. A
        section name is matched loosely - case and punctuation dropped -
        so a heading copied approximately still resolves. An unknown name
        or section is a 404 whose detail lists what exists."""
        try:
            return guides.get_guide(name, section=section)
        except GuideError as e:
            raise HTTPException(status_code=404, detail=str(e))

    def _argument_reference_errors(definition, arguments, ws):
        """The 'asset:', 'prompt:' and 'output:' references that name nothing
        this workspace can reach, in the values a run would actually use -
        the caller's `arguments`, plus every declared `variables` default
        the caller did not override.

        A stored default is exactly as much a promise as a caller's value:
        `validate_workflow(name="templates/ltx2/reference-sheet")` with no
        arguments at all used to answer valid because only `arguments` was
        checked, while the same call with the stored default handed back
        explicitly answered invalid - one run, two verdicts (#166). Reported
        at `variables.<name>` so the message still says whether the caller
        wrote the bad reference or merely didn't override one.

        Resolved through the engine's own resolvers over the roots this
        workspace searches, so validation agrees with what the run would
        find - an asset that exists in another workspace is a miss here for
        the same reason it would be a miss there. Only the reference is
        resolved, never loaded: the point is to answer before any bytes move.
        """

        def over_roots(roots, resolve):
            """Resolve against each root in turn, and on a total miss raise
            the *first* root's error rather than the last.

            The resolvers name the path they searched in their message, and
            the first root is the workspace's own library plus the read-only
            fallbacks the environment pins - which is the path a run would
            report. The last root's message would name an examples directory
            and leave out the workspace, reading as though the library the
            caller works in was never looked in."""
            first = None
            for root in roots:
                try:
                    return resolve(root)
                except Exception as e:
                    first = first or e
            # `roots` is never empty here: the asset branch answers an empty
            # search path itself, and the prompt path always holds the
            # server's own library. Re-raising None would be a TypeError
            raise first

        def _string_leaves(value, path):
            """Every string in `value`, paired with the path it sits at.

            `value` is walked the way a for_each entry is - a list or dict
            of arbitrary nesting - so a reference inside `shots[2].
            references[1].from_file` is found the same as one at the
            argument's own top level."""
            if isinstance(value, str):
                yield path, value
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    yield from _string_leaves(item, f"{path}[{i}]")
            elif isinstance(value, dict):
                for key, item in value.items():
                    yield from _string_leaves(item, f"{path}.{key}")

        if arguments is not None and not isinstance(arguments, dict):
            return []
        supplied = arguments if isinstance(arguments, dict) else {}

        effective = []
        declared = definition.get("variables") if isinstance(definition, dict) else None
        if isinstance(declared, dict):
            for name, value in declared.items():
                if name not in supplied:
                    effective.append((f"variables.{name}", value))
        for name, value in supplied.items():
            effective.append((f"arguments.{name}", value))

        errors = []
        for base_path, value in effective:
            for path, leaf in _string_leaves(value, base_path):
                try:
                    if is_asset_reference(leaf):
                        roots = _resolution_roots(ws)
                        if not roots:
                            # A server configured with no asset library has
                            # no root to fail against: over_roots would
                            # re-raise its "first error", which is None,
                            # and the caller would read a TypeError about
                            # BaseException in place of a verdict
                            name = leaf.removeprefix(ASSET_PREFIX).strip()
                            raise ValueError(
                                f"Unknown asset {name!r}: "
                                "this workspace has no asset library"
                            )
                        over_roots(
                            roots,
                            lambda root: resolve_asset_reference(leaf, asset_dir=root),
                        )
                    elif leaf.startswith(PROMPT_PREFIX):
                        over_roots(
                            _prompt_roots(),
                            lambda root: resolve_prompt_reference(
                                leaf, prompt_dir=root
                            ),
                        )
                    elif is_output_reference(leaf):
                        resolve_output_reference(leaf, root=ws.outputs)
                except Exception as e:
                    # Every resolver here raises with a message written for
                    # the person who wrote the reference - a traversal
                    # refusal from the security layer included
                    errors.append({"path": path, "message": str(e)})
        return errors

    def _probe_command_for(candidate, request, workspace, workflow_dir):
        """The execute-shaped command a cache probe of this validate request
        needs - the same fields _run_job sends, so the worker loads the
        workflow exactly as a job would."""
        command = {
            "arguments": request.arguments,
            "output_dir": workspace.outputs,
            "workflow_dir": workflow_dir,
        }
        if workspace.assets:
            command["asset_dir"] = workspace.assets
        if request.workflow_path is not None:
            command["workflow_path"] = candidate.file_spec
        else:
            command["workflow"] = request.workflow
            command["base_dir"] = os.path.dirname(candidate.file_spec)
        return command

    @app.post("/api/validate")
    def validate_workflow(
        request: JobRequest,
        ws: Workspace = Depends(selected_workspace),
        sizes: bool = Query(
            True,
            description="Ask the hub how large each missing model is; false "
            "skips the network for a faster answer",
        ),
    ):
        """Schema-validate a workflow and check its pipeline arguments
        against real signatures, without queuing anything. Give either an
        inline workflow or a workflow_path - a path on the server or a
        stored workflow name from /api/workflows. The workspace it resolves
        in comes from the body or the query string, body first. A valid
        answer also carries a plan: the fingerprint of the work these
        arguments produce, the step count, the list lengths, the model
        repos not in the cache, and an estimate from the workflow's cost
        block."""
        if (request.workflow is None) == (request.workflow_path is None):
            raise HTTPException(
                status_code=400,
                detail="Provide exactly one of workflow or workflow_path",
            )
        try:
            workspace = _workspace_for(request.workspace or ws.name)
            if request.workflow_path is not None:
                # Built from the file so relative paths inside it resolve
                # against its own directory, exactly as a run would
                sources = _sources_for(workspace)
                resolved, source = resolve_workflow_reference(
                    request.workflow_path, sources
                )
                # Confined to the source it came from, not to the writable
                # root - an example is read where it lives
                source_root = source.root if source else workspace.workflows
                candidate = workflow_from_file(resolved, workspace.outputs, source_root)
                definition = candidate.workflow_definition
                # The listing name the job history is keyed on, so the plan
                # can quote what this box's own runs of it took (#154)
                catalog_name = catalog_name_for(resolved, source)
            else:
                definition = request.workflow
                source_root = workspace.workflows
                # An inline definition has no catalog name, so no history
                catalog_name = None
                candidate = workflow_from_definition(
                    copy.deepcopy(request.workflow),
                    workspace.outputs,
                    request.base_dir,
                    source_root,
                )
        except HTTPException:
            raise
        except SecurityError as e:
            # Messages the security layer writes itself - safe to surface
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            # Anything else could carry internals in its message; the log
            # keeps the detail, the client gets the category
            logger.exception("Workflow could not be constructed for validation")
            raise HTTPException(
                status_code=400,
                detail="Workflow could not be constructed - the server log "
                "has the detail",
            )
        # `arguments` defaults to `{}` on the model (JobRequest is shared
        # with run_workflow, which needs a dict), so an omitted field and an
        # explicit `{}` are otherwise indistinguishable here - and the two
        # mean different things: omitted is "check the document", explicit
        # is "check a run with these arguments" (#364). model_fields_set
        # tells them apart without changing the field's default for every
        # other caller of validate_workflow.
        caller_arguments = (
            request.arguments if "arguments" in request.model_fields_set else None
        )
        try:
            # The caller's list is the one a for_each expands over, so the
            # pre-flight checks the step set that will actually run
            errors = candidate.validation_errors(arguments=caller_arguments)
        except Exception:
            # An error here is not the schema's verdict on the workflow -
            # validation_errors() reports that by returning it. It is the
            # validator itself failing, and its message could carry
            # internals, so the log keeps the detail and the client is told
            # the category, as above
            logger.exception("Workflow could not be validated")
            detail = (
                "The workflow could not be validated - the server log has the detail"
            )
            return {
                "valid": False,
                "error": detail,
                "errors": [{"path": None, "message": detail}],
                "warnings": [],
            }
        if errors:
            return {
                "valid": False,
                "error": format_validation_errors(errors),
                "errors": errors,
                "warnings": [],
            }
        # The arguments a caller is about to run with, checked the way the
        # run itself would check them: an undeclared name, a value that will
        # not coerce, an 'asset:'/'prompt:'/'output:' reference that names
        # nothing in this workspace. Without this the free pre-flight covers
        # every part of a run except the part the caller actually wrote
        argument_problems = argument_errors(definition, request.arguments)
        argument_problems += _argument_reference_errors(
            definition, request.arguments, workspace
        )
        if argument_problems:
            return {
                "valid": False,
                "error": format_validation_errors(argument_problems),
                "errors": argument_problems,
                "warnings": [],
                "checked_arguments": sorted(request.arguments or {}),
            }
        answer = {
            "valid": True,
            "error": None,
            "errors": [],
            "warnings": workflow_argument_warnings(definition, request.arguments)
            # A value a declared constraint will round up - the silent half
            # of #96: the run changed the caller's frame count and only the
            # server's log said so
            + constraint_warnings(definition, request.arguments)
            + entry_field_warnings(definition, request.arguments)
            # Why `plan.cached_steps` is 0 for a workflow with no seed - the
            # cache is off, not empty
            + unseeded_cache_warnings(definition, request.arguments)
            # An adapter whose file name says nothing about which checkpoint
            # partition it was trained for: valid, since the name of a
            # future checkpoint cannot be predicted, but nothing at run time
            # would say it loaded onto the wrong one (#155)
            + candidate.adapter_warnings(request.arguments)
            # A required task argument fed by variable:name where name's
            # default is null - a fine document, but a run left as-is would
            # fail; empty once the caller names any arguments, since that
            # condition is a hard error above instead (#364)
            + candidate.null_variable_argument_warnings(caller_arguments)
            # An argument a sub-workflow step passes to a workflow that
            # declares no variable for it - dropped in silence at run time
            + candidate.sub_workflow_warnings(),
        }
        if request.arguments:
            # Naming what was checked is the difference between 'the stored
            # definition is valid' and 'the values you are about to pass are'
            answer["checked_arguments"] = sorted(request.arguments)
        # What the run will execute for these arguments, fingerprinted so
        # an acknowledgement can be bound to it (#85). Best effort: the
        # verdict above is the schema's and the planner may not change it
        try:
            from .. import get_device, get_device_type

            command = _probe_command_for(candidate, request, workspace, source_root)

            def observed_for_child(path, child_definition, arguments=None):
                """A composed child's own observed figure, keyed by the
                catalog name it resolves to - so a parent with no figure of
                its own can quote what this box's runs of the *child* took
                rather than falling back to unknown (#268).

                `arguments` are the composing step's own overrides - the
                same role `arguments` plays for the top-level `observed`
                callback - so a child whose composing step shifted a
                declared scalar `cost_driver` (#341) is bucketed against
                *that* value rather than always the child's stored
                defaults, which silently answered the default bucket's
                history for every override."""
                base_dir = (
                    os.path.dirname(os.path.abspath(candidate.file_spec))
                    if candidate.file_spec
                    else None
                )
                try:
                    child_path, child_root = resolve_sub_workflow(
                        path, base_dir or ".", candidate.workflow_dir
                    )
                except (SecurityError, OSError, ValueError, SubWorkflowNotFound):
                    return None
                child_name = _catalog_name_from_root(child_path, child_root)
                if not child_name:
                    return None
                # resolve_sub_workflow hands back a bare root string, not a
                # Source, so writability is inferred the way that root was
                # built: the workspace's own workflows/ is the writable one
                # (#274)
                child_workspace = (
                    workspace.name if child_root == workspace.workflows else None
                )
                return _observed_for_name(
                    child_name, child_definition, arguments, workspace=child_workspace
                )

            answer["plan"] = build_plan(
                candidate,
                request.arguments,
                device=get_device_type(get_device()),
                prompt_dir=workspace.prompts,
                lookup_sizes=sizes,
                cache_probe=lambda arguments: manager.probe_cache(
                    {**command, "arguments": arguments}
                ),
                # What this box's own runs of this shape took, which is what
                # the estimate quotes ahead of a curated figure (#154) - the
                # same aggregate the listing reports, asked with the
                # caller's arguments rather than the defaults
                observed=(
                    (
                        lambda arguments: _observed_for_name(
                            catalog_name,
                            definition,
                            arguments,
                            workspace=workspace.name if source.writable else None,
                        )
                    )
                    if catalog_name
                    else None
                ),
                observed_for_child=observed_for_child,
            )
        except Exception:
            logger.exception("Plan could not be built")
            answer["plan"] = None
        if answer["plan"]:
            # cached_steps is 0 both when nothing hit and when the probe ran
            # against the wrong workspace's output root (#184) - echoing
            # what it was actually probed against turns the second case
            # from a silent miss into something a caller can read
            answer["plan"]["workspace"] = workspace.name
            answer["plan"]["output_dir"] = workspace.outputs
            answer["warnings"] += gate_warnings(answer["plan"]["downloads_required"])
            if catalog_name:
                answer["warnings"] += _host_memory_warnings(
                    catalog_name,
                    definition,
                    answer["plan"]["list_entries"],
                    workspace=workspace.name if source.writable else None,
                )
        return answer

    def _host_memory_warnings(name, definition, list_entries, *, workspace=None):
        """Whether this box's own history says the requested list is
        projected to exceed host RAM (#243) - best effort, since a warning
        that 500s the free pre-flight would be worse than skipping it."""
        costs = getattr(app.state, "observed_costs", None)
        if costs is None:
            return []
        try:
            from ..host_memory import host_memory_stats

            rows = costs.rows_for(name, workspace=workspace)
            ceiling_mb = (host_memory_stats().get("total_mb") or 0) * CEILING_FRACTION
            return host_memory_warnings(definition, list_entries, rows, ceiling_mb)
        except Exception:
            logger.debug("host memory projection failed for %s", name, exc_info=True)
            return []

    # ------------------------------------------------------------ workspaces

    class WorkspaceRequest(BaseModel):
        name: str = Field(description="Name for the new workspace")

    @app.get("/api/workspaces")
    def list_workspaces():
        """Every workspace on this server, the default first.

        A workspace is a namespace, not a security boundary: the API token
        is all-or-nothing, so anything that can list these can reach all of
        them.
        """
        root = app.state.workspace_root
        # workspace_names lists the whole root once; everything after the
        # first entry (always the default, see its docstring) is a named
        # workspace to describe individually
        names = workspace_names(root)[1:] if root else []
        listed = [app.state.default_workspace]
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

    @app.post("/api/workspaces", status_code=201)
    def add_workspace(request: WorkspaceRequest):
        """Create a workspace: its own workflows, assets and outputs, sharing
        this server's one prompt library."""
        root = _workspace_root()
        try:
            created = create_workspace(root, request.name)
        except SecurityError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileExistsError as e:
            raise HTTPException(status_code=409, detail=str(e))
        forget_workspace_usage()
        logger.info(f"Created workspace {request.name} at {created.root}")
        return created.describe()

    @app.delete("/api/workspaces/{name}")
    def remove_workspace(name: str, acknowledged: bool = False):
        """Delete a workspace and everything in it.

        Answers what it would remove and refuses until `acknowledged=true`:
        this deletes generated work, and a count is what makes it an
        informed choice rather than a surprise.
        """
        root = _workspace_root()
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
                    "message": f"Deleting workspace '{name}' removes these files "
                    f"permanently. Repeat with acknowledged=true to proceed.",
                    "contents": contents,
                },
            )
        with manager._lock:
            queued = [
                job
                for job in manager.jobs.values()
                if job.status not in TERMINAL_STATES
                and job.spec.get("workspace") == name
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

    # ------------------------------------------------------------- workflows

    # How much of a long variable default the variables route shows before
    # cutting it: enough to recognize a prompt by, far short of carrying one
    VARIABLE_VALUE_PREVIEW = 200

    @app.get("/api/workflows")
    def list_workflows(
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
        sources = _sources_for(ws)
        found = listing(sources)
        try:
            details = project_listing(
                attach_observed(
                    workflow_details(found),
                    getattr(app.state, "observed_costs", None),
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

    @app.put("/api/workflows/{name:path}")
    def save_workflow(
        name: str, request: JobRequest, ws: Workspace = Depends(selected_workspace)
    ):
        """Write a workflow into the writable workflow directory. The
        definition must be schema-valid - the editor validates before saving,
        and a save that silently wrote a broken file would betray both.

        A name that currently resolves to a read-only source (an example, a
        builtin) is not overwritten: the copy lands in the writable source
        and shadows it from then on.
        """
        if request.workflow is None:
            raise HTTPException(
                status_code=400,
                detail='Provide the definition as {"workflow": {...}}',
            )
        path, _source = resolve_writable_workflow(_sources_for(ws), name)
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
            "path": path,
            "warnings": warnings,
            "shape": metadata["shape"],
            "traits": metadata["traits"],
            "summary": metadata["summary"],
        }

    @app.delete("/api/workflows/{name:path}")
    def delete_workflow(name: str, ws: Workspace = Depends(selected_workspace)):
        """Remove a workflow file from the writable workflow directory.

        A read-only source is refused rather than silently ignored: an
        example or a builtin is not the caller's to delete, and saying so
        is more useful than a 404 that reads like the file is missing.
        """
        path, source = resolve_readable_workflow(_sources_for(ws), name)
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
        return {"name": name, "deleted": True}

    @app.get("/api/workflows/{name:path}/download")
    @query_token_ok
    def download_workflow(name: str, ws: Workspace = Depends(selected_workspace)):
        """Serve a workflow definition as a forced download."""
        path, _source = resolve_readable_workflow(_sources_for(ws), name)
        return FileResponse(
            path, filename=os.path.basename(path), media_type="application/json"
        )

    # Declared before the catch-all below, which would otherwise swallow
    # '<name>/variables' as a workflow called that
    @app.get("/api/workflows/{name:path}/variables")
    def get_workflow_variables(
        name: str, full: bool = False, ws: Workspace = Depends(selected_workspace)
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
        path, source = resolve_readable_workflow(_sources_for(ws), name)
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
                return {
                    key: preview(item, f"{path}.{key}") for key, item in value.items()
                }
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
        observed = _observed_for_name(
            name, definition, workspace=ws.name if source.writable else None
        )
        if observed:
            answer["observed"] = observed
        return answer

    def _observed_for_name(name, definition, arguments=None, *, workspace=None):
        """One workflow's `observed` block, from the same aggregate the
        listing uses - so the figure a caller reads in the listing and the
        one they read here are the same figure.

        `arguments` narrow it to the bucket the run being planned falls in;
        without them it is the figure the stored defaults give, which is the
        listing's. `workspace` scopes it to one workspace's own writable copy
        (#274); omitted, it is a shared catalog entry's pooled figure (#154)."""
        costs = getattr(app.state, "observed_costs", None)
        return (
            costs.observed(name, definition, arguments, workspace=workspace)
            if costs
            else None
        )

    @app.get("/api/workflows/{name:path}")
    def get_workflow(name: str, ws: Workspace = Depends(selected_workspace)):
        path, source = resolve_readable_workflow(_sources_for(ws), name)
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

    # --------------------------------------------------------------- prompts

    class PromptRequest(BaseModel):
        prompt: Dict[str, Any] = Field(description="The prompt definition to save")

    @app.get("/api/prompt-schema")
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

    def _prompt_roots():
        """The prompt search path: the library this server writes to, then
        the read-only ones an --examples-dir tree brought with it. A name in
        an earlier root shadows the same name later, as on the workflow
        search path."""
        roots = [app.state.prompt_dir]
        primary = os.path.abspath(app.state.prompt_dir)
        for root in app.state.example_prompt_dirs:
            if os.path.abspath(root) != primary:
                roots.append(root)
        return roots

    def _find_prompt(name):
        """(path, writable) for the first root on the search path that holds
        this name. 404s when no root does, the way resolve_prompt_name does
        for a name that cannot be referenced at all."""
        for index, root in enumerate(_prompt_roots()):
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

    @app.get("/api/prompts")
    def list_prompts(
        tag: str | None = None,
        intended_model: str | None = None,
        include_text: bool = True,
    ):
        # A stray file too deep or oddly named can sit in the directory, but
        # no workflow could reference it - listing it would only invite that
        paths = {}
        origins = {}
        roots = _prompt_roots()
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
        wanted = _matching_prompts(details, tag, intended_model)
        if wanted is not None:
            details = {
                name: detail for name, detail in details.items() if name in wanted
            }
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
            "prompt_dir": app.state.prompt_dir,
            "prompt_dirs": roots,
            "prompts": sorted(details),
            "origins": origins,
            "details": details,
        }

    @app.put("/api/prompts/{name:path}")
    def save_prompt(name: str, request: PromptRequest):
        """Write a prompt into the prompt directory. Like a workflow save,
        the definition must be schema-valid before it lands on disk."""
        status, message = validate_data(request.prompt, load_schema("prompt"))
        if not status:
            raise HTTPException(status_code=400, detail=message)
        if str(request.prompt.get("text", "")).startswith(RESERVED_TEXT_PREFIXES):
            raise HTTPException(
                status_code=400,
                detail="A prompt's text may not itself begin with a reference "
                f"prefix ({', '.join(RESERVED_TEXT_PREFIXES)})",
            )
        path = resolve_prompt_name(app.state.prompt_dir, name, allow_create=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            json.dump(request.prompt, file, indent=2)
            file.write("\n")
        logger.info(f"Saved prompt {name} to {path}")
        return {"name": name, "path": path}

    @app.delete("/api/prompts/{name:path}")
    def delete_prompt(name: str):
        """Remove a prompt file from the prompt directory. A prompt that
        came from a read-only examples library is not this server's to
        delete - the same 403 a read-only workflow answers with."""
        path, writable = _find_prompt(name)
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

    @app.get("/api/prompts/{name:path}/download")
    @query_token_ok
    def download_prompt(name: str):
        """Serve a stored prompt as a forced download."""
        path, _ = _find_prompt(name)
        return FileResponse(
            path, filename=os.path.basename(path), media_type="application/json"
        )

    @app.get("/api/prompts/{name:path}")
    def get_prompt(name: str):
        path, writable = _find_prompt(name)
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

    # ------------------------------------------------------------- enhancers

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

    @app.get("/api/enhancers")
    def list_enhancers():
        return {"presets": preset_descriptions()}

    @app.post("/api/enhance", status_code=201)
    def enhance(request: EnhanceRequest, ws: Workspace = Depends(selected_workspace)):
        """Queue a prompt enhancement as an ordinary job. The enhanced text
        is the job's single manifest file once it succeeds.

        Scoped like any other job: the caller reads the result back from the
        workspace it asked in, so this has to write there too."""
        try:
            definition = build_enhance_workflow(
                request.preset,
                request.idea,
                model_name=request.model_name,
                device=request.device,
            )
            job = manager.submit(
                workflow=definition,
                arguments={},
                workflow_dir=ws.workflows,
                output_dir=ws.outputs,
                asset_dir=ws.assets,
                workspace=ws.name,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return manager.describe(job)

    # --------------------------------------------------------------- gallery

    # Built from the security layer's allowlists so a new format is added
    # exactly once - the gallery had already drifted (.bmp, .mkv, .mov)
    from ..security import (
        ALLOWED_AUDIO_EXTENSIONS,
    )

    MEDIA_KINDS = {
        **{ext: "image" for ext in ALLOWED_IMAGE_EXTENSIONS},
        **{ext: "video" for ext in ALLOWED_VIDEO_EXTENSIONS},
        **{ext: "audio" for ext in ALLOWED_AUDIO_EXTENSIONS},
        # Not in the security allowlists above (nothing loads a .txt back
        # into a pipeline, so it is not a path a run reads), but a
        # text-shape run's deliverable is a real output and belongs in the
        # gallery like any other kind (#238)
        ".txt": "text",
    }

    # The allowlist members that are not already-compressed containers -
    # everything else in MEDIA_KINDS deflates for about nothing, so it is
    # stored instead (see _zip_download)
    RAW_MEDIA_EXTENSIONS = {".bmp", ".wav"}

    # Exposed for tests - the two sets _zip_download's compression policy
    # reads, so a test can assert the relationship without reaching into a
    # closure
    app.state.media_kinds = MEDIA_KINDS
    app.state.raw_media_extensions = RAW_MEDIA_EXTENSIONS

    # Longest side of an on-demand gallery thumbnail, in pixels
    GALLERY_THUMBNAIL_MAX_DIM = 320

    def _strip_output_prefix(name):
        """A gallery name, accepting the way a workflow argument would
        reference it ('output:<name>', #356) as well as the bare form
        every gallery listing reports. `asset:` already gets this courtesy
        on this same endpoint family (`is_asset_reference` below); a caller
        who spelled a name by copying an `output:` reference used to be met
        with a wrong-looking "path does not exist" instead, because the
        prefix was joined straight into the path rather than stripped first.

        Applied once, at the top of every route that takes a gallery
        `name`, so the rest of that route - job lookups, run-path parsing,
        the file it echoes back - sees the same bare name `_output_file`
        resolves, rather than resolving the file correctly while a sibling
        lookup keyed on the untouched string quietly misses.
        """
        if is_output_reference(name):
            return name.removeprefix(OUTPUT_PREFIX).strip()
        return name

    def _output_file(name, root=None):
        """A file inside a workspace's output directory, or a 404 - never
        outside it."""
        root = root or manager.output_dir
        name = _strip_output_prefix(name)
        try:
            path = validate_path(
                os.path.join(root, name),
                root,
                allow_create=False,
            )
        except PathTraversalError:
            # PathTraversalError's own message can embed the resolved
            # *absolute* server path (dw/security.py validate_path, the
            # containment branch) - useful in a log, not in a response a
            # remote caller reads. Still say *why* it was refused, since a
            # caller needs to tell "this name would have escaped the
            # workspace" from "this name is simply wrong" (#310) - the
            # distinction #134 pinned and a later leak fix (#247) collapsed.
            raise HTTPException(
                status_code=404,
                detail=f"Unknown file: {name} - path contains a disallowed pattern",
            )
        except InvalidInputError:
            raise HTTPException(
                status_code=404, detail=f"Unknown file: {name} - path does not exist"
            )
        except SecurityError:
            raise HTTPException(status_code=404, detail=f"Unknown file: {name}")
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Unknown file")
        return path

    def _asset_in(name, roots):
        """The file a bare asset name has in one of these roots, or a 404.

        `_asset_file` with the search path already in hand, for a caller
        resolving many names against the one workspace: each call to
        `resolve_asset_reference` walks the pinned fallbacks on its own, so
        calling it once per root re-walked them all every time - the name is
        validated once here instead, and each root is then just a join and
        an isfile check.
        """
        try:
            validate_asset_reference(name)
        except SecurityError as e:
            raise HTTPException(status_code=404, detail=str(e))
        for root in roots:
            candidate = os.path.join(root, name)
            if not os.path.isfile(candidate):
                continue
            try:
                return validate_path(candidate, root)
            except SecurityError:
                # A symlink under this root can still point outside it -
                # isfile follows the link and says yes, and validate_path
                # is what actually catches the escape. That's a miss for
                # this root, not a 500: fall through to the next one and,
                # on a total miss, the same 404 every other miss gets.
                continue
        if not roots:
            detail = f"Unknown asset {name!r}: this workspace has no asset library"
        else:
            detail = f"Unknown asset {name!r}: not found in {', '.join(roots)}"
        raise HTTPException(status_code=404, detail=detail)

    def _asset_file(reference, ws):
        """The file an 'asset:' reference names in this workspace, or a 404.

        Looked for down the same search path a run resolves 'asset:' in
        (_asset_roots), so what the API can read is what a job would load.
        A miss names every root that was searched, so the caller sees
        their own workspace library among them rather than just the last
        (often an examples directory they never wrote to).
        """
        return _asset_in(
            reference.removeprefix(ASSET_PREFIX).strip(),
            _resolution_roots(ws),
        )

    def _static_files_for(root):
        """The StaticFiles instance bound to one root, built on first use and
        cached on app.state - see the comment where the cache is created."""
        cache = app.state.static_files_by_root
        files = cache.get(root)
        if files is None:
            files = StaticFiles(directory=root)
            cache[root] = files
        return files

    def _common_assets(ws):
        """The library every workspace under this root shares, or None.

        A recurring cast is not the property of the workspace that first
        uploaded it, and a fresh workspace could not see it at all - the
        prompt library has been shared from the start for the same reason.
        """
        return getattr(ws, "common_assets", None)

    def _asset_roots(ws):
        """The asset search path of one workspace: its own library, then the
        one shared by every workspace under this root, then the read-only
        ones an --examples-dir tree brought with it. The same order 'asset:'
        resolves in (dw/assets.asset_search_path), so what the browser lists
        is what a job would load."""
        roots = []
        for root in [ws.assets, _common_assets(ws), *app.state.example_asset_dirs]:
            if not root:
                continue
            root = os.path.abspath(root)
            if root not in roots and os.path.isdir(root):
                roots.append(root)
        return roots

    def _resolution_roots(ws):
        """`_asset_roots(ws)`, falling back to the workspace's own (possibly
        nonexistent) library when the search path is empty.

        A caller resolving a name still needs *somewhere* to fail against:
        with no root at all the 404 would name no directory, leaving the
        caller to guess where it looked. Naming the workspace's own
        directory keeps the failure pointing at the library the caller
        thinks they're working in, even when that library hasn't been
        created yet.

        Never `[None]`: a server configured with no asset library at all has
        nothing to point at either, and `_asset_in` turns the resulting empty
        list into the "no asset library" 404 rather than joining `None`.
        """
        roots = _asset_roots(ws)
        if roots:
            return roots
        return [os.path.abspath(ws.assets)] if ws.assets else []

    def _asset_roots_for_job(job_id, ws):
        """The asset search path a job's own run used, for export: its spec's
        `asset_dir` (or the historical row's), then the read-only example
        libraries an --examples-dir tree brought with it - the same shape
        `_asset_roots` builds for the selected workspace, but rooted at
        wherever the job actually ran rather than at the workspace the
        caller happens to be scoped to now. A job that ran in one workspace
        while the caller exports it scoped to another must still find its
        own 'asset:' files, not the other workspace's.

        Falls back to `_asset_roots(ws)` when the job carries no asset_dir
        of its own - an inline-workflow job, or one recorded before this
        field existed."""
        job = manager.get(job_id)
        if job is None:
            return _asset_roots(ws)
        spec = (job.get("spec") or {}) if isinstance(job, dict) else job.spec
        asset_dir = spec.get("asset_dir")
        if not asset_dir:
            return _asset_roots(ws)
        roots = []
        for root in [asset_dir, _common_assets(ws), *app.state.example_asset_dirs]:
            if not root:
                continue
            root = os.path.abspath(root)
            if root not in roots and os.path.isdir(root):
                roots.append(root)
        return roots

    def _served_url(path, ws, version=None):
        """The URL a served file is reachable at: the default workspace's
        files keep the URL they have always had, a named one carries the
        same selector its API calls do, so one route serves both. 'v=' is
        cache-busting for a name reused by a rerun, not the workspace
        selector, so it always comes last."""
        url = path if ws.is_default else f"{path}?workspace={quote(ws.name)}"
        if version is None:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}v={version}"

    def _absolute_served_url(path, ws, version=None):
        """The same URL, made openable by a client with no other way to
        learn this server's origin (#353) - an MCP-only agent, which is
        never told a request's Host and must not guess one. `None` unless
        an operator has configured `public_url` (or `DW_PUBLIC_URL`):
        deriving an origin from request/forwarded headers would trust
        whatever the caller claims to be, so a caller gets nothing rather
        than a guess."""
        origin = os.environ.get("DW_PUBLIC_URL") or settings.public_url
        if not origin:
            return None
        return f"{origin.rstrip('/')}{_served_url(path, ws, version)}"

    def _iter_gallery_files(root, group_runs=True):
        """Every media file under a directory tree. Yields (relative_name,
        folder, subfolder, kind, path) - relative_name always uses '/' so
        it round-trips through a URL the same way on every platform.

        With group_runs (the gallery's own use, over the output directory):
        recurses into the per-workflow subfolders (dw/workflow.py's
        effective_output_dir writes each run under '<workflow
        identity>/<run id>/', and mirrors a workflow's position under a
        'workflows' tree in the flat layout). The folder a file is grouped
        under is the identity - the run id is dropped, so a workflow run
        fifty times is one folder in the filter, not fifty - and whatever
        followed the run id is the subfolder, the part of the run a step's
        'result.subfolder' put it in ('final', 'intermediate'). A flat-layout
        path has no run id to anchor on, so its subfolder is '' and its
        whole directory is the folder, as it always was.

        Without it (the asset library's use, which has no run ids to strip):
        folder is just the plain relative directory and subfolder is ''."""
        for current, _dirs, names in os.walk(root):
            rel_root = os.path.relpath(current, root)
            directory = "" if rel_root == "." else rel_root.replace(os.sep, "/")
            for name in names:
                extension = os.path.splitext(name)[1].lower()
                kind = MEDIA_KINDS.get(extension)
                if kind is None:
                    continue
                relative_name = name if not directory else f"{directory}/{name}"
                if group_runs:
                    folder, run_id, subfolder = split_run_path(relative_name)
                else:
                    folder, subfolder, run_id = directory, "", ""
                yield (
                    relative_name,
                    folder,
                    subfolder,
                    run_id,
                    kind,
                    os.path.join(current, name),
                )

    def _gallery_entries(root, ws):
        entries = []
        try:
            files = list(_iter_gallery_files(root))
        except OSError:
            files = []
        # One read of each workflow's run ordinals per listing, not per file:
        # a run of fifty files would otherwise re-read the same manifests
        # fifty times
        versions_by_folder = {}

        def _version(folder, run_id):
            if not run_id:
                return None
            if folder not in versions_by_folder:
                versions_by_folder[folder] = run_versions(os.path.join(root, folder))
            return versions_by_folder[folder].get(run_id)

        for relative_name, folder, subfolder, run_id, kind, path in files:
            try:
                stat = os.stat(path)
            except OSError:
                continue
            # File names look like '{workflow}-{step}-{i}.{j}.{k}.ext'; every
            # artifact from one step shares the '{workflow}-{step}' prefix, so
            # the label keeps the full name including the extension rather
            # than truncating at the first dot - otherwise sibling outputs of
            # the same step would show identical, indistinguishable labels,
            # and a step that writes more than one kind of file (e.g. a still
            # plus a video) would lose the extension that tells them apart
            label = os.path.basename(relative_name)
            output_path = f"/outputs/{quote(relative_name)}"
            entry = {
                "name": relative_name,
                "folder": folder,
                "subfolder": subfolder,
                # Which run wrote it, and that run's ordinal among this
                # workflow's runs - the 'v4' a person sees in the grid
                # and an agent says out loud. Two runs write the same
                # basename, so `label` cannot tell them apart and
                # `name` is too long to quote. None under the flat
                # layout, which has no runs to number
                "run_id": run_id,
                "version": _version(folder, run_id),
                # Quoted (slashes kept literal): a name carrying '#', '?'
                # or '%' would otherwise break the src the gallery
                # renders it into. The mtime still rides along for cache
                # busting when a file's content changes without its name
                # changing (e.g. a manual overwrite outside the engine) -
                # normal reruns get a fresh name instead, see
                # dw/result.py's output_file_path
                "url": _served_url(output_path, ws, int(stat.st_mtime)),
                "kind": kind,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "label": label,
            }
            absolute_url = _absolute_served_url(output_path, ws, int(stat.st_mtime))
            if absolute_url is not None:
                entry["absolute_url"] = absolute_url
            entries.append(entry)
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        return entries

    def _iter_orphan_runs(root):
        """Run directories under `root` holding nothing but their own
        bookkeeping (RUN_BOOKKEEPING_FILES) - a run whose output was deleted
        before #134's by-name `delete_output`, or one that failed before
        writing anything. Yields (name, mtime) where `name` is the
        `<identity>/<run id>` string `delete_output` already accepts (#170).

        By what is absent, not by extension: a `text`-shape run writes .txt
        and a `utility`-shape run may write nothing the gallery lists, and
        neither is junk. This call only lists; deciding whether an entry is
        junk stays a human/agent call before `delete_output` is invoked."""
        for current, dirs, _names in os.walk(root):
            if not is_run_id(os.path.basename(current)):
                continue
            # A run directory holds no run directories of its own
            dirs[:] = []
            # A dotfile is not output either: a .DS_Store Finder left behind
            # would otherwise make the run permanently non-orphan
            has_output = any(
                name not in RUN_BOOKKEEPING_FILES and not name.startswith(".")
                for _sub_current, _sub_dirs, sub_names in os.walk(current)
                for name in sub_names
            )
            if has_output:
                continue
            try:
                mtime = os.stat(current).st_mtime
            except OSError:
                continue
            name = os.path.relpath(current, root).replace(os.sep, "/")
            yield (name, mtime)

    def _orphan_entries(root):
        entries = [
            {"name": name, "mtime": mtime} for name, mtime in _iter_orphan_runs(root)
        ]
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        return entries

    @app.get("/api/gallery")
    def gallery(
        limit: int = 200,
        offset: int = 0,
        folder: Optional[str] = None,
        subfolder: Optional[str] = None,
        only_orphans: bool = False,
        version: Optional[int] = None,
        media: bool = False,
        ws: Workspace = Depends(selected_workspace),
    ):
        """A page of media files in the output directory, newest first.
        Stateless by design - the gallery survives server restarts because
        it reads the directory tree, not job history. 'folders' lists every
        distinct workflow folder present (over the whole directory, not just
        this page), for the UI's folder filter - a run id is not a folder of
        its own, so a workflow's runs group together; '' stands for files
        saved directly at the output root, and is itself always a member so
        that folder-less outputs stay selectable once anything is nested.
        'subfolders' is the other axis, over the whole directory the same
        way: the in-run subfolders steps wrote into ('final',
        'intermediate'), '' for files at a run's root. `folder` and
        `subfolder` filter independently and intersect when both are given.
        `version` narrows to the runs holding that ordinal - with `folder`,
        the one run "v4" names; without it, that run of every workflow.

        `only_orphans=true` inverts the whole call: instead of media files,
        it returns run directories holding nothing but their own
        bookkeeping (manifest.json, workflow.json, job.json) as `runs`,
        each `{name, mtime}` - a run that wrote any file at all, a
        text-shape prompt or a utility's side output included, is not
        listed. `folder`/`subfolder` and the `folders`/`subfolders` facets
        do not apply in this mode, since an orphan run has no file to
        carry either. `name` is exactly what `DELETE /api/gallery/{name}`
        accepts, so listing and deleting an orphan is a two-call round
        trip (#170).

        `media=true` adds `duration_seconds` to each audio/video entry,
        probed the same way `get_gallery_metadata` reports it - which two
        takes of the same workflow otherwise have no way to be told apart
        by, since size and mtime are misleading proxies for length (#356).
        Off by default and bounded by `limit`: only the page actually
        returned is probed, not the whole listing, so the cost of asking
        stays proportional to the page size rather than the library size."""
        if only_orphans:
            entries = _orphan_entries(ws.outputs)
            offset = max(0, offset)
            limit = max(0, limit)
            page = entries[offset : offset + limit]
            return {
                "runs": page,
                "total": len(entries),
                "offset": offset,
                "limit": limit,
                "workspace": ws.name,
            }
        entries = _gallery_entries(ws.outputs, ws)
        folders = sorted({e["folder"] for e in entries} | {""})
        subfolders = sorted({e["subfolder"] for e in entries} | {""})
        if folder is not None:
            entries = [e for e in entries if e["folder"] == folder]
        if subfolder is not None:
            entries = [e for e in entries if e["subfolder"] == subfolder]
        if version is not None:
            entries = [e for e in entries if e["version"] == version]
        offset = max(0, offset)
        limit = max(0, limit)
        page = entries[offset : offset + limit]
        if media:
            for entry in page:
                if entry["kind"] not in ("audio", "video"):
                    continue
                probed = probe_media(os.path.join(ws.outputs, entry["name"]))
                if probed is not None:
                    entry["duration_seconds"] = probed.get("duration_seconds")
        return {
            "files": page,
            "total": len(entries),
            "offset": offset,
            "limit": limit,
            "folders": folders,
            "subfolders": subfolders,
            "workspace": ws.name,
        }

    @app.get("/api/gallery/{name:path}/metadata")
    def gallery_metadata(
        name: str,
        envelope: bool = False,
        ws: Workspace = Depends(selected_workspace),
    ):
        """Generation metadata embedded in a saved image ('workflow' inside
        it is the full definition the editor can reopen), plus the job that
        produced the file when history remembers one, plus - for audio and
        video - what the file itself holds: duration, format and level,
        which is how an agent that cannot listen checks a track. Only an
        image embeds 'metadata' this way - it is always null for audio and
        video, since neither format has a slot this writer uses; recover
        the recipe from 'job' (GET /api/jobs/{id}/workflow) when one is
        known, or from nothing when it isn't (a kept asset has no job).

        `envelope=true` adds the soundtrack's level second by second, which
        is what says *where* in a track something is - whether a shot is
        still voiced at its last frame, how deep the hole at a seam goes.
        Opt-in: a ten-minute track is 600 numbers, and the default call has
        to stay small.

        `name` may also be an 'asset:' reference, and then it is the input
        asset of that name that is described rather than an output (#127).
        The numbers here - duration, frame count, fps, sample rate - are
        what decide whether a call will work at all, and for a file the
        caller is about to *consume* they were previously unobtainable:
        the only way to read a wav's length was to run a job that copied it
        into the output directory. `job` is null for an asset (nothing here
        produced it) and `source` says which of the two roots answered."""
        run_id, version = "", None
        name = _strip_output_prefix(name)
        if is_asset_reference(name):
            path = _asset_file(name, ws)
            source, job = "asset", None
        else:
            path = _output_file(name, ws.outputs)
            source = "output"
            try:
                # Scoped to this workspace: two workspaces can each write a
                # file with the same relative name, and an unscoped lookup
                # could attribute this one to the wrong workspace's job
                job = manager.history.job_for_file(name, workspace=ws.name)
            except Exception:
                job = None
            # Which run wrote it, and that run's ordinal - the same 'v4' the
            # listing reports. After "look at version 3" this is the next
            # call, so it confirms the right file was reached rather than
            # sending the caller back to the listing
            folder, run_id, _subfolder = split_run_path(name)
            if run_id:
                version = run_versions(os.path.join(ws.outputs, folder)).get(run_id)
        metadata = read_embedded_metadata(path)
        extension = os.path.splitext(path)[1].lower()
        media = (
            probe_media(path, envelope=envelope)
            if MEDIA_KINDS.get(extension) in ("audio", "video")
            else None
        )
        if media is not None and source == "output":
            # Where each shot of a joined video sits, as the run that wrote
            # it recorded (dw/shots.py) - null for a file not joined from shots
            media["shots"] = recorded_shots(ws.outputs, name)
        return {
            "name": name,
            "source": source,
            "metadata": metadata,
            "job": job,
            "run_id": run_id,
            "version": version,
            "media": media,
        }

    @app.get("/api/gallery/{name:path}/audio")
    def gallery_audio(
        name: str,
        start: Optional[float] = None,
        duration: Optional[float] = None,
        ws: Workspace = Depends(selected_workspace),
    ):
        """The soundtrack of an output or asset, as WAV - a muxed video's
        track, which `get_output_audio` used to refuse outright, or an
        excerpt (`start` + `duration`, seconds) of a track too long to send
        whole (#193). An excerpt names itself in the response headers
        (`X-DW-Excerpt-Start`, `X-DW-Excerpt-Duration`) beside the whole
        track's `X-DW-Duration` - omitted only when a container carries no
        duration in its own header - so a cut is never silent (#204).

        An audio-only file asked for whole is served as its own bytes in its
        own encoding - there is nothing to extract, and a transcode would
        change what the agent hears."""
        name = _strip_output_prefix(name)
        if is_asset_reference(name):
            path = _asset_file(name, ws)
        else:
            path = _output_file(name, ws.outputs)
        extension = os.path.splitext(path)[1].lower()
        kind = MEDIA_KINDS.get(extension)
        if kind not in ("audio", "video"):
            raise HTTPException(status_code=404, detail=f"{name} carries no soundtrack")

        excerpt = start is not None or duration is not None
        if kind == "audio" and not excerpt:
            # The container's own header has the duration - reading it does
            # not decode a single frame, unlike probe_media (which measures
            # level and would pay for a full decode just for one number).
            headers = {}
            duration_seconds = media_duration(path)
            if duration_seconds is not None:
                headers["X-DW-Duration"] = str(duration_seconds)
            if extension == ".wav":
                # mimetypes says audio/x-wav on macOS, audio/vnd.wave from
                # Python 3.14's builtin table on a box with no system mime
                # file; an extract says audio/wav, and a whole WAV must not
                # read as a different kind
                media_type = "audio/wav"
            else:
                media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            return FileResponse(path, media_type=media_type, headers=headers)

        # A track over the cap is refused at the header, not after it has
        # been decoded and shipped: the MCP side would refuse the same bytes
        # for the same reason, having paid for all of them. An excerpt is
        # sized by its own span - `duration`, clipped to what is left of the
        # track after `start` - so a whole-length "excerpt" is not a way
        # around the gate.
        shape = audio_shape(path)
        if shape is not None and shape["duration_seconds"] is not None:
            span = shape["duration_seconds"]
            if excerpt and duration is not None:
                span = max(0.0, min(float(duration), span - float(start or 0.0)))
            projected = projected_wav_base64_size({**shape, "duration_seconds": span})
            if projected > MAX_INLINE_AUDIO_BYTES:
                what = (
                    f"a {span:.1f}s excerpt of {name}"
                    if excerpt
                    else f"{name}'s whole soundtrack"
                )
                advice = (
                    "Ask for a shorter `duration`"
                    if excerpt
                    else "Ask for an excerpt with `start` and `duration` (seconds)"
                )
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{what} would be {projected} bytes base64-encoded as WAV "
                        f"- over the {MAX_INLINE_AUDIO_BYTES} byte limit for an "
                        f"inline clip. {advice}, or download the file."
                    ),
                )

        try:
            data, info = extract_audio(path, start=start, duration=duration)
        except NoSoundtrack:
            raise HTTPException(status_code=404, detail=f"{name} carries no soundtrack")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        headers = {"X-DW-Duration": str(info["of_seconds"])}
        if info["excerpt"]:
            headers["X-DW-Excerpt-Start"] = str(info["start"])
            headers["X-DW-Excerpt-Duration"] = str(info["duration_seconds"])
        return Response(content=data, media_type="audio/wav", headers=headers)

    FRAME_MIN_DIMENSION = 64
    # The most moments one `at` may name: each is a seek, a decode and a
    # PNG encode in the server process, and a contact sheet is the shape
    # for seeing more of a clip at once
    MAX_FRAME_MOMENTS = 32

    @app.get("/api/gallery/{name:path}/frames")
    def gallery_frames(
        name: str,
        at: Optional[str] = None,
        count: Optional[int] = None,
        seams: Optional[str] = None,
        boundaries: Optional[str] = None,
        names: Optional[str] = None,
        max_dimension: int = 512,
        crop: Optional[str] = None,
        ws: Workspace = Depends(selected_workspace),
    ):
        """Frames of a video output or asset, as PNG tiles - the way an
        agent with no video content type sees what a run made (#193).
        Exactly one selector: `at` (a comma list of seconds or "frame:N"),
        `count` (an evenly spaced contact sheet, `frame_grid` without a
        workflow), or `seams` ("true", or a comma list of 1-based seam
        numbers) for the last frame before and first frame after each
        boundary, side by side. `boundaries` is the comma list of frame
        indexes each shot after the first starts at, and `names` the
        shots' names. Without `boundaries`, an output's seams are the shots
        its run's manifest recorded for it (a `concat_videos`,
        `dissolve_videos` or chained step), named as recorded unless `names`
        is given; a file with none recorded still needs `boundaries`.
        Tiles are downscaled to `max_dimension` on their longest side.
        `crop` is `x,y,width,height` in the video's own source pixels
        (`video_shape`'s `width`/`height`) - resolved once and cut from
        every sampled frame before any stamping, fitting or composing, so
        it names the same region whatever `max_dimension` downscales the
        result to."""
        name = _strip_output_prefix(name)
        if is_asset_reference(name):
            path = _asset_file(name, ws)
        else:
            path = _output_file(name, ws.outputs)
        if MEDIA_KINDS.get(os.path.splitext(path)[1].lower()) != "video":
            raise HTTPException(status_code=404, detail=f"{name} is not a video")

        chosen = [
            key
            for key, value in (("at", at), ("count", count), ("seams", seams))
            if value
        ]
        if len(chosen) != 1:
            raise HTTPException(
                status_code=400,
                detail="Pass exactly one of `at`, `count` or `seams`"
                + (f" - got {', '.join(chosen)}" if chosen else ""),
            )
        # A floor on each *sub-tile* of a composite (contact sheet / seam
        # pair) - a caller asking for a small max_dimension still gets a
        # legible grid, which is then fit to max_dimension as a whole below.
        sub_tile_width = max(FRAME_MIN_DIMENSION, int(max_dimension))
        limit = max(1, int(max_dimension))

        try:
            # Computed once and threaded through every selector below: each
            # of frames_at/contact_sheet/seam_tiles would otherwise call
            # video_shape itself, opening the container (and, lacking a
            # header frame count, decoding it whole to count) a second time
            # just to answer the same frame_count/fps/width/height (#193).
            shape = video_shape(path)
            crop_box = (
                resolve_crop_box(
                    [c.strip() for c in crop.split(",")],
                    shape["width"],
                    shape["height"],
                )
                if crop
                else None
            )
            if at:
                moments = [
                    m.strip() if m.strip().startswith("frame:") else float(m)
                    for m in at.split(",")
                    if m.strip()
                ]
                if len(moments) > MAX_FRAME_MOMENTS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"`at` names {len(moments)} moments; the most is "
                        f"{MAX_FRAME_MOMENTS} - ask for a contact sheet (`count`) "
                        "to see more of the clip at once",
                    )
                tiles = frames_at(path, moments, shape=shape, crop_box=crop_box)
            elif count:
                tiles = [
                    contact_sheet(
                        path,
                        count,
                        tile_width=sub_tile_width,
                        shape=shape,
                        crop_box=crop_box,
                    )
                ]
            else:
                recorded = (
                    None
                    if boundaries or is_asset_reference(name)
                    else recorded_shots(ws.outputs, name)
                )
                if recorded:
                    # The file's own seams, from its run's manifest
                    starts = [shot["start_frame"] for shot in recorded[1:]]
                    shot_names = (
                        [n.strip() for n in names.split(",")]
                        if names
                        else [shot["name"] for shot in recorded]
                    )
                elif not boundaries:
                    raise HTTPException(
                        status_code=400,
                        detail="`seams` needs `boundaries`: the frame index each "
                        "shot after the first starts at, comma-separated - this "
                        "file's run recorded no shots for it",
                    )
                else:
                    starts = [int(b) for b in boundaries.split(",") if b.strip()]
                    shot_names = (
                        [n.strip() for n in names.split(",")] if names else None
                    )
                wanted = (
                    None
                    if seams.lower() == "true"
                    else {int(s) for s in seams.split(",") if s.strip()}
                )
                if wanted is not None and not wanted:
                    raise HTTPException(
                        status_code=400,
                        detail="`seams` names no seam - pass `true` for every seam, "
                        "or seam numbers from 1",
                    )
                tiles = seam_tiles(
                    path,
                    starts,
                    names=shot_names,
                    tile_width=sub_tile_width,
                    shape=shape,
                    wanted=wanted,
                    crop_box=crop_box,
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {
            "name": name,
            **shape,
            "tiles": [_encoded_tile(tile, limit) for tile in tiles],
            "crop": (
                [
                    crop_box[0],
                    crop_box[1],
                    crop_box[2] - crop_box[0],
                    crop_box[3] - crop_box[1],
                ]
                if crop_box
                else None
            ),
        }

    def _encoded_tile(tile, limit):
        image = tile["image"]
        longest = max(image.width, image.height)
        if longest > limit:
            scale = limit / longest
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                )
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = {key: value for key, value in tile.items() if key != "image"}
        encoded.update(
            {
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "mime_type": "image/png",
                "width": image.width,
                "height": image.height,
            }
        )
        return encoded

    @app.get("/api/gallery/{name:path}/thumbnail")
    @query_token_ok
    def gallery_thumbnail(
        name: str, request: Request, ws: Workspace = Depends(selected_workspace)
    ):
        """A small JPEG rendition of an image output, for the grid - the
        full-resolution file is only fetched for the detail/lightbox view.
        Generated on demand rather than cached to disk, so it never grows
        the output directory the gallery itself scans."""
        path = _output_file(name, ws.outputs)
        extension = os.path.splitext(path)[1].lower()
        if MEDIA_KINDS.get(extension) != "image":
            raise HTTPException(
                status_code=404, detail="Thumbnails are only generated for images"
            )
        # The file's mtime and size are the validator: the grid re-requests
        # every visible thumbnail on each visit, and a 304 skips the
        # decode/resize/encode; a rerun that overwrites the file changes it
        stat = os.stat(path)
        etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        cache_headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=cache_headers)
        try:
            from PIL import Image

            with Image.open(path) as image:
                # shrink first (JPEGs decode at reduced size via draft), then
                # convert - converting a full-resolution image only to
                # discard most of it is the expensive order
                image.draft(
                    "RGB", (GALLERY_THUMBNAIL_MAX_DIM, GALLERY_THUMBNAIL_MAX_DIM)
                )
                image.thumbnail((GALLERY_THUMBNAIL_MAX_DIM, GALLERY_THUMBNAIL_MAX_DIM))
                image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=80)
        except (OSError, ValueError) as e:
            # what PIL raises for an unreadable or corrupt file
            raise HTTPException(
                status_code=500, detail=f"Could not generate thumbnail: {e}"
            )
        return Response(
            content=buffer.getvalue(), media_type="image/jpeg", headers=cache_headers
        )

    @app.get("/api/gallery/{name:path}/download")
    @query_token_ok
    def download_output(name: str, ws: Workspace = Depends(selected_workspace)):
        """Serve one output file as a forced download rather than an inline view."""
        path = _output_file(name, ws.outputs)
        return FileResponse(path, filename=os.path.basename(name))

    # A generous ceiling rather than a real limit - it exists so a
    # malformed client cannot ask the server to zip the whole directory
    MAX_ARCHIVE_FILES = 1000

    class ArchiveRequest(BaseModel):
        names: list[str] = Field(min_length=1, max_length=MAX_ARCHIVE_FILES)

    def _zip_download(entries, filename):
        """Bundle (arcname, path) pairs into a zip and serve it as a download.

        The archive is a temp file rather than memory - a selection of videos
        does not fit in RAM - unlinked once the response has been sent. The
        three routes that hand back a zip share this so the cleanup contract
        lives in one place: nothing has attached the background unlink while
        the archive is being written, so a failure there has to unlink on the
        way out or leak a half-written file into tmp.
        """
        handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        try:
            with handle:
                with zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as archive:
                    for arcname, path in entries:
                        extension = os.path.splitext(path)[1].lower()
                        # A file in MEDIA_KINDS but not RAW_MEDIA_EXTENSIONS
                        # is an already-compressed container - deflating it
                        # buys about nothing for a full CPU pass the caller
                        # waits through (the response doesn't start until the
                        # temp file is complete), so it is stored instead.
                        # Everything else - .json, .md, .txt, .bmp, .wav, an
                        # unrecognized extension - deflates, including the
                        # export zip's text files. ".txt" is in MEDIA_KINDS
                        # (kind "text", #238) but is plain text, not an
                        # already-compressed container, so it stays out of
                        # this policy the same way .json and .md do
                        kind = MEDIA_KINDS.get(extension)
                        stored = (
                            kind is not None
                            and kind != "text"
                            and extension not in RAW_MEDIA_EXTENSIONS
                        )
                        archive.write(
                            path,
                            arcname=arcname,
                            compress_type=(
                                zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
                            ),
                        )
        except BaseException:
            os.unlink(handle.name)
            raise

        return FileResponse(
            handle.name,
            media_type="application/zip",
            filename=filename,
            background=BackgroundTask(os.unlink, handle.name),
        )

    def _archive_selection(entries, kind):
        """`_zip_download` plus the one tail the two archive routes shared:
        a timestamped `dw-<kind>s-*.zip` name and a log line naming the
        count. Logged after the archive is written, not before, so a write
        that fails partway (a bad path slipping past resolution, a full
        disk) doesn't log a success that didn't happen.
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        response = _zip_download(entries, f"dw-{kind}s-{stamp}.zip")
        logger.info(f"Archived {len(entries)} {kind} files")
        return response

    @app.post("/api/gallery/archive")
    def archive_outputs(
        request: ArchiveRequest, ws: Workspace = Depends(selected_workspace)
    ):
        """Bundle a multi-file gallery selection into one zip. A browser
        cannot zip on its own and throttles a burst of single downloads, so
        the whole selection has to arrive as one file. Written to a temp
        file rather than memory - a selection of videos does not fit in
        RAM - and unlinked once the response has been sent."""
        # Resolved before anything is written, so a bad name in the
        # selection fails the request instead of yielding a partial zip
        # the gallery-relative name is the entry name, so a workflow's output
        # subfolders stay intact inside the download
        names = [_strip_output_prefix(name) for name in request.names]
        paths = [(name, _output_file(name, ws.outputs)) for name in names]

        return _archive_selection(paths, "output")

    # What a run directory holds besides its media: the engine writes them to
    # describe the run, and the gallery - which lists media - never shows them
    RUN_SIDECARS = (MANIFEST_FILE_NAME, REALIZED_FILE_NAME)

    def _prune_empty_run_directory(name, root):
        """Drop the run directory a just-deleted output belonged to, once no
        media is left in it.

        A run writes `manifest.json` and `workflow.json` beside its files, and
        nothing in the gallery addresses either one. Deleting every output of a
        run therefore used to leave the directory behind forever: a consumer
        that removed everything it made still could not put a workspace back
        the way it found it, and nothing it could call would even show the
        residue (#134). Tying the sidecars' lifetime to the outputs they
        describe is what makes "delete what you made" true.

        Only the sidecars may remain - any other leftover file means something
        is still there to describe, and the directory stays.

        Returns:
            The run id swept, or None if nothing was
        """
        identity, run_id, _ = split_run_path(name)
        if not run_id:
            # The flat layout writes no run directory and no sidecars
            return None
        relative = f"{identity}/{run_id}" if identity else run_id
        try:
            run_dir = validate_path(
                os.path.join(root, relative), root, allow_create=False
            )
        except SecurityError:
            return None
        if not os.path.isdir(run_dir):
            return None

        for directory, _subdirectories, files in os.walk(run_dir):
            for file_name in files:
                if directory == run_dir and file_name in RUN_SIDECARS:
                    continue
                return None

        # Pin the siblings' numbers first: a run that predates versions is
        # ranked, and removing one ahead of it would renumber it
        record_run_versions(os.path.dirname(run_dir))
        shutil.rmtree(run_dir, ignore_errors=True)
        # And the identity folders above it, while they are empty - a swept
        # workspace should not keep one directory per workflow it once ran
        parent = os.path.dirname(run_dir)
        while os.path.normpath(parent) != os.path.normpath(root):
            try:
                os.rmdir(parent)
            except OSError:
                break
            parent = os.path.dirname(parent)
        logger.info(f"Swept empty run directory {relative}")
        return run_id

    def _run_directory(name, root):
        """The run directory `<identity>/<run id>` names, or None.

        A run that failed before it wrote anything still has a directory and a
        manifest, and no gallery name addresses it - so the name of the
        directory itself is the only handle there can be (#134).
        """
        parts = [part for part in (name or "").split("/") if part]
        if not parts or not is_run_id(parts[-1]):
            return None
        try:
            path = validate_path(
                os.path.join(root, "/".join(parts)), root, allow_create=False
            )
        except SecurityError:
            return None
        return path if os.path.isdir(path) else None

    @app.delete("/api/gallery/{name:path}")
    def delete_output(name: str, ws: Workspace = Depends(selected_workspace)):
        """Remove one file from the output directory.

        When that was the last media file of its run, the run directory goes
        with it, sidecars included. `name` may also be a run directory
        (`<identity>/<run id>`), which removes the whole run - the only handle
        on a run that failed before it wrote any media (#134).
        """
        name = _strip_output_prefix(name)
        run_dir = _run_directory(name, ws.outputs)
        if run_dir is not None:
            # As in _prune_empty_run_directory: pin the siblings' numbers
            # before one of them goes
            record_run_versions(os.path.dirname(run_dir))
            shutil.rmtree(run_dir, ignore_errors=True)
            parent = os.path.dirname(run_dir)
            while os.path.normpath(parent) != os.path.normpath(ws.outputs):
                try:
                    os.rmdir(parent)
                except OSError:
                    break
                parent = os.path.dirname(parent)
            logger.info(f"Deleted run directory {name}")
            forget_workspace_usage()
            return {
                "name": name,
                "deleted": True,
                "run_swept": os.path.basename(run_dir),
            }

        path = _output_file(name, ws.outputs)
        os.remove(path)
        logger.info(f"Deleted output file {name}")
        swept = _prune_empty_run_directory(name, ws.outputs)
        forget_workspace_usage()
        return {"name": name, "deleted": True, "run_swept": swept}

    # ---------------------------------------------------------------- uploads

    UPLOADS_SUBDIR = "uploads"
    # Audio included: the asset library holds it and workflows read it (an
    # H3 audio reference is built from a .wav), so refusing it here would
    # leave one input kind with no way onto the machine
    ALLOWED_UPLOAD_EXTENSIONS = (
        ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
    )
    MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB - covers a short video clip

    @app.post("/api/uploads", status_code=201)
    async def upload_media(
        request: Request,
        filename: str,
        asset_name: Optional[str] = None,
        shared: bool = False,
        ws: Workspace = Depends(selected_workspace),
    ):
        """Save a browser-picked image, video or audio file into the asset library's
        uploads/ subfolder and hand back the reference a workflow argument
        can carry.

        An upload is input, so it belongs in the asset library rather than
        among generated output, and the reference handed back is
        'asset:uploads/<name>' - portable, and meaningful in a workflow that
        is saved and rerun later. A server with no asset library configured
        keeps the old behavior, writing to the output directory's uploads/
        and returning an absolute path. The body is the raw file bytes: no
        multipart parser dependency needed for a single-file upload.

        `asset_name` stores it under a name of the caller's choosing -
        'cast/priya-voice.wav' rather than the random one a browser upload
        gets - which is what makes a recurring cast's references readable
        in every workflow that carries them. It may name a folder, is
        confined to the library the way `keep_output`'s is, and takes the
        uploaded file's extension when it has none of its own. Without it
        the name stays random, so two uploads of the same file never
        collide.

        `shared` puts it in the library every workspace under this root
        shares rather than in this workspace's own - a recurring cast that
        episode four, in a workspace of its own, still has to reach.
        """
        extension = os.path.splitext(os.path.basename(filename))[1].lower()
        if extension not in ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=400, detail=f"File extension not allowed: {extension}"
            )

        # Refuse an oversized upload from its declared length, before
        # reading a single byte of it
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload too large: {declared} > {MAX_UPLOAD_BYTES}",
            )
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty upload")
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload too large: {len(body)} > {MAX_UPLOAD_BYTES}",
            )

        library = ws.assets or ws.outputs
        if shared:
            library = _common_assets(ws)
            if not library:
                raise HTTPException(
                    status_code=409,
                    detail="This server has no shared asset library - it was "
                    "configured from loose directories rather than a workspace "
                    "root, so there is nothing for an asset to be common to",
                )
        uploads_dir = os.path.join(library, UPLOADS_SUBDIR)
        name = f"{uuid.uuid4().hex}{extension}"
        if asset_name:
            name = asset_name
            if not os.path.splitext(name)[1]:
                name = f"{name}{extension}"
            try:
                # The same check the keep route makes: a name, possibly with
                # folders in it, that cannot climb out of the library
                name = validate_asset_reference(name)
            except SecurityError as e:
                raise HTTPException(status_code=400, detail=str(e))
            if os.path.splitext(name)[1].lower() != extension:
                raise HTTPException(
                    status_code=400,
                    detail=f"asset_name {asset_name!r} does not match the "
                    f"uploaded file's kind ({extension})",
                )
        os.makedirs(uploads_dir, exist_ok=True)
        try:
            dest = validate_output_path(os.path.join(uploads_dir, name), uploads_dir)
        except SecurityError as e:
            raise HTTPException(status_code=400, detail=str(e))
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        # Off the event loop: a 200 MB write would otherwise stall every SSE
        # stream and poll for its duration
        await run_in_threadpool(_write_bytes, dest, body)
        logger.info(f"Saved upload {filename!r} -> {dest}")
        if shared or ws.assets:
            path = f"/inputs/{UPLOADS_SUBDIR}/{quote(name)}"
            result = {
                "path": f"asset:{UPLOADS_SUBDIR}/{name}",
                "url": _served_url(path, ws),
                "shared": shared,
            }
            absolute_url = _absolute_served_url(path, ws)
            if absolute_url is not None:
                result["absolute_url"] = absolute_url
            return result
        path = f"/outputs/{UPLOADS_SUBDIR}/{quote(name)}"
        result = {
            "path": dest,
            "url": _served_url(path, ws),
        }
        absolute_url = _absolute_served_url(path, ws)
        if absolute_url is not None:
            result["absolute_url"] = absolute_url
        return result

    def _asset_origin(ws, root):
        """Which library an asset came from: this workspace's own, the one
        shared by every workspace under the root, or a read-only examples
        tree. A client that cannot tell them apart cannot say why deleting
        one answers 403.

        By directory, never by position in the search path: the workspace's
        own library drops out of `_asset_roots` until it exists, and the
        examples tree that then sits first is still nobody's to write."""
        own = ws.assets
        if own and os.path.abspath(own) == root:
            return WORKSPACE_ORIGIN
        common = _common_assets(ws)
        if common and os.path.abspath(common) == root:
            return COMMON_ORIGIN
        return EXAMPLES_ORIGIN

    @app.get("/api/assets")
    def list_assets(ws: Workspace = Depends(selected_workspace)):
        """The asset library: the input media an 'asset:' reference names.

        Reported by reference rather than by path - 'asset:uploads/x.png' is
        what a workflow argument carries, and a client that only ever sees
        references cannot accidentally write a path that means something
        else on another machine. Empty, not an error, on a server with no
        library configured: nothing is wrong, there is just nowhere for an
        asset to be.
        """
        library = ws.assets
        roots = _asset_roots(ws)
        if not roots:
            return {
                "asset_dir": library,
                "asset_dirs": [],
                "assets": [],
                "folders": [],
                "libraries": [],
                "shadowed": [],
            }

        libraries = [
            {
                "origin": (origin := _asset_origin(ws, root)),
                "dir": root,
                "writable": origin != EXAMPLES_ORIGIN,
            }
            for root in roots
        ]

        assets = []
        shadowed = []
        # Which origin first claimed a name, so a later root's same name can
        # be reported as shadowed rather than silently dropped
        seen = {}
        for root in roots:
            try:
                files = list(_iter_gallery_files(root, group_runs=False))
            except OSError:
                files = []
            origin = _asset_origin(ws, root)
            for relative, folder, _subfolder, _run_id, kind, path in files:
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                # A name in the workspace shadows the same name in an
                # examples library, exactly as 'asset:' resolution does
                if relative in seen:
                    shadowed.append(
                        {
                            "name": relative,
                            "reference": f"asset:{relative}",
                            "folder": folder,
                            "kind": kind,
                            "size": stat.st_size,
                            "mtime": stat.st_mtime,
                            "origin": origin,
                            "shadowed_by": seen[relative],
                        }
                    )
                    continue
                seen[relative] = origin
                asset_path = f"/inputs/{quote(relative)}"
                asset_entry = {
                    "name": relative,
                    "reference": f"asset:{relative}",
                    "folder": folder,
                    "kind": kind,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "origin": origin,
                    # For the editor's own preview - fetchable the same
                    # way an upload's URL is
                    "url": _served_url(asset_path, ws),
                }
                absolute_url = _absolute_served_url(asset_path, ws)
                if absolute_url is not None:
                    asset_entry["absolute_url"] = absolute_url
                assets.append(asset_entry)
        assets.sort(key=lambda entry: entry["mtime"], reverse=True)
        return {
            # The workspace's own library, unchanged: where an upload lands
            "asset_dir": library,
            "asset_dirs": [lib["dir"] for lib in libraries],
            "assets": assets,
            "folders": sorted({entry["folder"] for entry in assets} | {""}),
            "libraries": libraries,
            "shadowed": shadowed,
        }

    class KeepRequest(BaseModel):
        name: str = Field(
            description="The generated file to keep, as the gallery names it"
        )
        asset_name: Optional[str] = Field(
            default=None,
            description="Name to keep it under in the asset library; its own "
            "file name when omitted. May name a folder; the kept file's "
            "extension is assumed when the name has none",
        )
        overwrite: bool = Field(
            default=False, description="Replace an asset already under that name"
        )
        shared: bool = Field(
            default=False,
            description="Keep it in the library every workspace under this "
            "root shares, rather than in this workspace's own",
        )

    @app.post("/api/assets/keep", status_code=201)
    def keep_output_as_asset(
        request: KeepRequest, ws: Workspace = Depends(selected_workspace)
    ):
        """Keep a generated file as an input asset, under a stable name.

        A run's files live under '<workflow>/<run id>/', which is the right
        place for them and the wrong name to build on: 'latest' moves, and a
        pinned run id breaks the moment outputs are pruned. Keeping one
        copies it into the workspace's asset library, where an 'asset:' name
        stays put - which is what turns a generated still or score into an
        input later workflows can rely on.

        Within the workspace, so nothing crosses a namespace, and no bytes
        cross the network: a client that had to download and re-upload a
        multi-gigabyte video to reuse one frame would be paying for the
        round trip twice.
        """
        library = _common_assets(ws) if request.shared else ws.assets
        if not library:
            raise HTTPException(
                status_code=409,
                detail=(
                    "This server has no shared asset library"
                    if request.shared
                    else "This workspace has no asset library"
                ),
            )

        kept_name = _strip_output_prefix(request.name)
        source = _output_file(kept_name, ws.outputs)
        asset_name = request.asset_name or os.path.basename(kept_name)
        # The kept file's own extension when the name carries none, and a
        # refusal when it carries a contradicting one - exactly what the
        # upload route does with its `asset_name`. Without this a kept asset
        # could be written under an extensionless name, which the library
        # listing (which reads by kind) never shows again: the call reported
        # success and the asset was invisible (T014)
        extension = os.path.splitext(os.path.basename(kept_name))[1].lower()
        if not os.path.splitext(asset_name)[1]:
            asset_name = f"{asset_name}{extension}"
        elif os.path.splitext(asset_name)[1].lower() != extension:
            raise HTTPException(
                status_code=400,
                detail=f"asset_name {request.asset_name!r} does not match the "
                f"kept file's kind ({extension or 'no extension'})",
            )
        try:
            asset_name = validate_asset_reference(asset_name)
            destination = validate_path(os.path.join(library, asset_name), library)
        except SecurityError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if os.path.exists(destination) and not request.overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"asset:{asset_name} already exists - pass overwrite=true "
                f"to replace it",
            )

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if os.path.exists(destination):
            os.remove(destination)
        # A hard link first: keeping one frame of a multi-gigabyte render
        # should not cost another copy of it, and both names refer to the
        # same content anyway. Falls back to a copy when the link cannot be
        # made - a different filesystem, or one that has no links
        try:
            os.link(source, destination)
            linked = True
        except OSError:
            shutil.copy2(source, destination)
            linked = False

        logger.info(f"Kept output {request.name} as asset:{asset_name}")
        return {
            "reference": f"asset:{asset_name}",
            "name": asset_name,
            "path": destination,
            "linked": linked,
            "shared": bool(request.shared),
        }

    @app.post("/api/assets/archive")
    def archive_assets(
        request: ArchiveRequest, ws: Workspace = Depends(selected_workspace)
    ):
        """Bundle a multi-file asset selection into one zip - the gallery's
        bulk download, for the input side of it.

        Resolved down the same search path a run resolves 'asset:' in, so a
        selection spanning the workspace's own library, the shared one and
        an examples tree downloads as one archive; the library-relative name
        is the entry name, which is the name the 'asset:' reference carries.
        """
        # The search path depends on the workspace, not on the name, so it is
        # built once rather than per name - each root's isdir check would
        # otherwise repeat once per name in the selection for no reason
        roots = _resolution_roots(ws)
        # Stripped and deduped before resolving, so "iris.png" and
        # "iris.png " (or a name repeated by an eager client) become the one
        # zip entry rather than a collision on write
        names = list(dict.fromkeys(n.strip() for n in request.names))
        # Resolved before anything is written, so a bad name in the
        # selection fails the request instead of yielding a partial zip
        paths = [(name, _asset_in(name, roots)) for name in names]

        return _archive_selection(paths, "asset")

    @app.delete("/api/assets/{name:path}")
    def delete_asset(name: str, ws: Workspace = Depends(selected_workspace)):
        """Permanently remove one file from the asset library.

        Deletes from whichever library on the search path holds it, the
        workspace's own first, so the name deleted is the name 'asset:'
        would have resolved to. An asset a read-only examples tree brought
        with it is not this server's to delete - the same 403 a read-only
        prompt or workflow answers with.

        Not recoverable, and any workflow still carrying that 'asset:'
        reference stops loading. Without this, everything else that writes
        the library (uploads, keep) had no counterpart and a mistake could
        only be cleaned up on the box (T014).
        """
        roots = _asset_roots(ws)
        if not roots:
            raise HTTPException(
                status_code=409, detail="This server has no asset library"
            )
        try:
            relative = validate_asset_reference(name)
        except SecurityError as e:
            raise HTTPException(status_code=400, detail=str(e))

        for root in roots:
            try:
                path = validate_path(os.path.join(root, relative), root)
            except SecurityError:
                continue
            if not os.path.isfile(path):
                continue
            origin = _asset_origin(ws, root)
            if origin == EXAMPLES_ORIGIN:
                raise HTTPException(
                    status_code=403,
                    detail=f"asset:{relative} is read-only: it comes from an "
                    f"examples library, not a library this server writes",
                )
            os.remove(path)
            logger.info(f"Deleted asset:{relative} ({path})")
            forget_workspace_usage()
            return {"name": relative, "deleted": True, "origin": origin}

        raise HTTPException(status_code=404, detail=f"No such asset: {relative}")

    # ----------------------------------------------------------------- models

    @app.get("/api/models")
    def get_models():
        """What the Hugging Face hub cache holds, largest repo first."""
        return scan_models()

    downloads = download_manager or DownloadManager()

    class DownloadRequest(BaseModel):
        repo_id: str = Field(description="Hub repo to download, e.g. org/model")

    @app.post("/api/models/download", status_code=202)
    def start_download(body: DownloadRequest):
        """Start a background snapshot download into the hub cache."""
        try:
            return downloads.start(body.repo_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/models/downloads")
    def list_downloads():
        return {"downloads": downloads.status_list()}

    @app.post("/api/models/downloads/{download_id}/cancel")
    def cancel_download(download_id: str):
        """Request cancellation; takes effect at the next progress tick.
        Partial files stay in the cache and resume on a retry."""
        status = downloads.cancel(download_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Unknown download")
        return status

    @app.delete("/api/models")
    def delete_cached_model(repo: str):
        """Delete every cached revision of one repo from the hub cache.

        Refused while a job is running or queued: the worker may be reading
        exactly the files a delete would remove out from under it."""
        if manager.is_busy():
            raise HTTPException(
                status_code=409,
                detail="A job is running or queued - deleting model files "
                "out from under it would corrupt the run",
            )
        if downloads.is_active():
            raise HTTPException(
                status_code=409,
                detail="A model download is in progress - deleting cache "
                "files while it writes them would corrupt both",
            )
        try:
            freed = delete_model(repo)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        logger.info(f"Deleted {repo} from the hub cache ({freed} bytes)")
        return {"repo_id": repo, "deleted": True, "freed": freed}

    # ------------------------------------------------------ diffusers update

    updater = diffusers_updater or DiffusersUpdater()

    @app.get("/api/system/diffusers")
    def diffusers_state():
        """Installed diffusers version (with its git commit when installed
        from git) and the state of any update."""
        return updater.status()

    class UpdateDiffusersRequest(BaseModel):
        commit: Optional[str] = Field(
            default=None,
            description="Git commit hash to pin the install to (7-40 hex "
            "characters) instead of tracking GitHub HEAD",
        )
        revert: bool = Field(
            default=False,
            description="Pin back to the known-good published release "
            "(pyproject.toml's diffusers floor) instead of installing from "
            "git. Mutually exclusive with commit.",
        )

    @app.post("/api/system/diffusers/update", status_code=202)
    def update_diffusers(body: UpdateDiffusersRequest = UpdateDiffusersRequest()):
        """Upgrade diffusers in the background: GitHub HEAD by default, a
        pinned commit when `commit` is given, or a revert to the last
        known-good published release when `revert` is true.

        Refused while a job is running or queued: pip replacing package
        files under a loaded pipeline is the model-delete hazard in another
        form. On success the idle worker is shut down so the next job
        imports the new version."""
        if body.commit and body.revert:
            raise HTTPException(
                status_code=400,
                detail="commit and revert are mutually exclusive",
            )
        commit = None
        if body.commit:
            try:
                commit = validate_commit_hash(body.commit)
            except InvalidInputError as e:
                raise HTTPException(status_code=400, detail=str(e))
        if manager.is_busy():
            raise HTTPException(
                status_code=409,
                detail="A job is running or queued - updating diffusers "
                "underneath it could corrupt the run",
            )
        if downloads.is_active():
            raise HTTPException(
                status_code=409,
                detail="A model download is in progress - replacing package "
                "files while it runs could corrupt the download",
            )
        try:
            return updater.start(
                on_success=manager.restart_worker_if_idle,
                commit=commit,
                revert=body.revert,
            )
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    # --------------------------------------------------------- memory/health

    @app.get("/api/memory")
    def memory():
        try:
            return manager.memory_status()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Worker unavailable: {e}")

    @app.post("/api/memory/clear")
    def clear_memory():
        """Drop every loaded pipeline and the step cache, freeing VRAM/RAM
        without waiting for the next job to evict one model for another.

        Refused while a job is running or queued (409) rather than blocked -
        the queue is FIFO, so the caller should wait for the job to finish
        and retry instead of this call stalling until it does.

        A server with no worker process resident answers `cleared` with a
        null `info` rather than a 503: the worker is on-demand, so its
        absence means there was nothing loaded to clear."""
        if manager.is_busy():
            raise HTTPException(
                status_code=409,
                detail="A job is running or queued - clearing memory out "
                "from under it would corrupt the run. Wait for it to finish.",
            )
        try:
            info = manager.clear_memory()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=f"Worker unavailable: {e}")
        return {"cleared": True, "info": info}

    @app.get("/api/health")
    def health():
        import socket

        from .. import __version__, get_device, get_device_type

        worker = manager.worker_manager
        return {
            "status": "ok",
            "version": __version__,
            # on-demand subprocess: false on an idle server that hasn't run
            # a job yet (or after a memory clear) is normal, not a fault -
            # it means no model process is currently resident, not that the
            # server is unhealthy (#206)
            "worker_alive": bool(
                worker.worker_active
                and worker.worker_process is not None
                and worker.worker_process.is_alive()
            ),
            "current_job": manager._current_job_id,
            "queued": sum(1 for j in manager.list() if j["status"] == "queued"),
            # which machine answered - the thing a remote client cannot
            # otherwise tell apart from a stale tunnel pointed at nothing
            "hostname": socket.gethostname(),
            "device": get_device_type(get_device()),
            "mcp": bool(app.state.mcp_mounted),
        }

    @app.get("/api/server")
    def server_info(ws: Workspace = Depends(selected_workspace)):
        """How this server is reachable, for the UI's Server page: what it
        is bound to, whether a token is needed, whether MCP is mounted, and
        the addresses another machine could name it by.

        No URL is composed here - the caller pairs an address with `port`
        and `mcp.path` - and the token itself is never reported in any
        form, only whether one is required. An interface enumeration
        failure is not a server failure: `addresses` comes back empty.

        `directories` is scoped to the `?workspace=` a caller names (or the
        session's own pin, via `_scoped`) - a mounted `download_output`
        confines a write to *that* workspace's output tree, so reporting
        the server's own default here regardless of the selector sent a
        caller pinned elsewhere writing into `default` without any error (#389).
        """
        import socket

        from .. import __version__, get_device, get_device_type

        try:
            addresses = local_addresses()
        except Exception:
            logger.debug("Could not enumerate local addresses", exc_info=True)
            addresses = []
        return {
            "hostname": socket.gethostname(),
            "version": __version__,
            "device": get_device_type(get_device()),
            "bind_host": host,
            "port": port,
            "wildcard_bind": wildcard_bind,
            "auth_required": bool(token),
            # The posture a security check has to know it is testing: with
            # this off, a workflow file is untrusted input - no arbitrary
            # imports, no remote code, no location outside the workspace's
            # roots. It is not a secret (the refusals name the flag), and
            # without it the posture could only be inferred from behavior
            # (#120)
            "trust_workflows": workflows_are_trusted(),
            "mcp": {"mounted": bool(app.state.mcp_mounted), "path": MCP_PATH},
            "addresses": addresses,
            # Python/torch/CUDA-driver/other-package versions - the detail
            # neither this route's own `version` field nor `get_health`
            # answers, e.g. whether bitsandbytes is even installed (#222)
            "runtime": runtime_info(),
            "directories": {
                # ws's properties are already absolute (Workspace and
                # ConfiguredWorkspace both resolve at construction). This
                # "workspace" is the root path a mounted download_output
                # confines a write to (dw_mcp/media.py's _remote_root) -
                # None for a default workspace configured from individual
                # directory overrides with no --workspace root, same as
                # before this route was workspace-aware
                "workspace": ws.root,
                "workflows": ws.workflows,
                "assets": ws.assets,
                "outputs": ws.outputs,
                "prompts": ws.prompts,
            },
        }

    # ---------------------------------------------------------------- outputs

    # ------------------------------------------------------------------ mcp

    if mcp_asgi is not None:
        # One route rather than app.mount("/mcp", ...): Starlette's Mount
        # only matches paths *under* its prefix, so a bare POST /mcp - the
        # URL clients are configured with - would fall through to the SPA
        # catch-all below and come back 405. This matches /mcp and
        # anything under it; build_mcp_app's wrapper normalizes the path
        # for the SDK app's single route.
        # Exactly the two spellings require_bearer_token gates - a single
        # "/mcp{path:path}" route would also answer /mcpfoo, which the gate
        # does not cover.
        app.router.routes.append(Route("/mcp", endpoint=mcp_asgi, name="mcp"))
        app.router.routes.append(
            Route("/mcp/{sub_path:path}", endpoint=mcp_asgi, name="mcp_sub")
        )

    # Generated files and input media, served as routes rather than static
    # mounts: a mount is bound to one directory at startup, and a workspace
    # can be created afterwards. Each handler delegates to a StaticFiles
    # instance for the workspace's own root (_static_files_for) rather than
    # a bare FileResponse - a FileResponse never answers 304 (no
    # If-None-Match handling), so every gallery load re-streamed the whole
    # file; going through StaticFiles.get_response restores ETag/
    # If-None-Match 304s, Range/206 and its own 404 handling, the way a real
    # mount always has.
    #
    # Ungated, as the mounts were, and for the same reason: an <img> or
    # <video> tag cannot attach an Authorization header. The auth middleware
    # only gates /api/, so these stay reachable exactly as before.
    #
    # '/inputs', not '/assets': Vite emits the SPA's own bundles under
    # /assets/, and serving the library there shadows them - the page loads
    # and then renders nothing, because its script and stylesheet 404. The
    # name is also the symmetric one, next to /outputs
    @app.get("/outputs/{name:path}")
    async def output_file(
        name: str, request: Request, ws: Workspace = Depends(selected_workspace)
    ):
        """One generated file, from the workspace that made it."""
        files = _static_files_for(ws.outputs)
        return await files.get_response(_strip_output_prefix(name), request.scope)

    @app.get("/inputs/{name:path}")
    async def input_file(
        name: str, request: Request, ws: Workspace = Depends(selected_workspace)
    ):
        """One file from the asset search path, for the editor's preview of
        an uploaded or chosen asset - the workspace's own library first,
        then any read-only examples library, so an example workflow's media
        previews the way an upload does."""
        roots = _asset_roots(ws)
        if not roots:
            raise HTTPException(status_code=404, detail="no asset library")
        for root in roots:
            try:
                candidate = validate_path(os.path.join(root, name), root)
            except SecurityError:
                continue
            if os.path.isfile(candidate):
                files = _static_files_for(root)
                return await files.get_response(name, request.scope)
        # Nothing has it: let the workspace's own library answer, so the
        # 404 (and its headers) come from StaticFiles as they always did
        files = _static_files_for(roots[0])
        return await files.get_response(name, request.scope)

    def _export_download_name(directory, job_id):
        """'<workflow>-v4-<job id>.zip' when the exported manifest says which
        run it was, else '<job id>.zip'. Only the saved file's name: the
        URL and the entries inside keep the job id, so nothing that already
        names an export changes."""
        try:
            with open(os.path.join(directory, MANIFEST_FILE_NAME)) as file:
                manifest = json.load(file)
        except (OSError, ValueError):
            return f"{job_id}.zip"
        if not isinstance(manifest, dict):
            return f"{job_id}.zip"
        version = manifest.get("version")
        identity = (manifest.get("workflow") or {}).get("identity")
        if not isinstance(version, int) or isinstance(version, bool):
            return f"{job_id}.zip"
        if not isinstance(identity, str) or not identity:
            return f"v{version}-{job_id}.zip"
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", identity).strip("-.")
        return f"{slug}-v{version}-{job_id}.zip" if slug else f"v{version}-{job_id}.zip"

    # Ungated for the same reason the two above are: a download link cannot
    # attach an Authorization header either
    @app.get("/exports/{job_id}.zip")
    def export_zip(job_id: str, ws: Workspace = Depends(selected_workspace)):
        """One job's export as a zip, built on request from the directory
        rather than kept as a second copy. Entries are named
        '<job id>/<relative path>', so unzipping anywhere gives the same tree
        the server holds."""
        try:
            directory = export_directory(ws.root, job_id)
        except SecurityError:
            raise HTTPException(status_code=404, detail="No export for this job")
        if not os.path.isdir(directory):
            raise HTTPException(status_code=404, detail="No export for this job")

        entries = []
        for current, _dirs, names in os.walk(directory):
            for name in sorted(names):
                path = os.path.join(current, name)
                entry = os.path.relpath(path, directory).replace(os.sep, "/")
                entries.append((f"{job_id}/{entry}", path))
        return _zip_download(entries, _export_download_name(directory, job_id))

    # ---------------------------------------------------------------- the UI

    resolved_ui = ui_dir or default_ui_dir()
    if resolved_ui:
        # Mounted last so /api and /outputs keep precedence; html=True serves
        # index.html at /, and the SPA routes by hash so no fallback is needed
        app.mount("/", StaticFiles(directory=resolved_ui, html=True), name="ui")

    return app
