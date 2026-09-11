"""FastAPI application exposing the workflow engine.

All state lives in the JobManager; this module is routing, validation and
SSE framing. Everything path-shaped goes through dw.security validators.
Interactive API docs are served at /docs (OpenAPI at /openapi.json).
"""

import os
import shutil
import io
import zipfile
import tempfile
import copy
import json
import uuid
import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from urllib.parse import quote, urlparse
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
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
    SecurityError,
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
from ..schema import load_schema, validate_data, format_validation_errors
from ..prompts import PROMPT_PREFIX, RESERVED_TEXT_PREFIXES
from ..workflow import Workflow, workflow_from_definition, workflow_from_file
from .enhancers import build_enhance_workflow, preset_descriptions
from .exports import export_directory, export_job
from ..result import read_embedded_metadata
from ..media_info import probe_media
from ..hub_cache import scan_models, delete_model, DownloadManager
from ..runs import strip_run_id
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
    source_for_path,
    workflow_names,
    workflow_sources,
    writable_source,
)
from .jobs import JobManager, MAX_PERSISTED_EVENTS, TERMINAL_STATES
from .netinfo import local_addresses
from .updater import DiffusersUpdater
from .catalog_shape import derive_catalog_metadata, project_listing
from . import guides
from .guides import GuideError

logger = logging.getLogger("dw")

# How long one SSE poll waits for a new event before checking liveness
SSE_POLL_SECONDS = 1.0


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


def catalog_name_for(path, source):
    """The listing name a resolved workflow path has within its source.

    None when the run came from an inline definition, or when the path is
    not under the source root after all - a name that does not name an
    entry is worse than no name for anything that later joins on it.
    """
    if source is None:
        return None
    relative = os.path.relpath(path, source.root)
    if relative.startswith(".."):
        return None
    return os.path.splitext(relative)[0].replace(os.sep, "/")


def workflow_details(sources_by_name):
    """Per-workflow card metadata: output kinds, step and variable counts,
    and the variable names themselves - enough for an agent to pick a
    workflow and know what to pass it without fetching each candidate. The
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
            )
        except HTTPException:
            raise
        except Exception as e:
            # workflow_from_file / validate / the security layer all raise for
            # bad requests - every failure here is the client's fault
            raise HTTPException(status_code=400, detail=str(e))
        return manager.describe(job)

    @app.get("/api/jobs")
    def list_jobs(workspace: Optional[str] = None):
        """All jobs by default - a plain filter, not `selected_workspace`,
        since the jobs list spans every workspace the server holds unless a
        caller asks to narrow it."""
        return {"jobs": manager.list(workspace=workspace)}

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

    @app.post("/api/jobs/{job_id}/rerun", status_code=201)
    def rerun_job(job_id: str, body: RerunRequest = RerunRequest()):
        """Queue a fresh job from a previous job's stored spec."""
        try:
            job = manager.rerun(job_id, new_seed=body.new_seed)
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
        body["zip_url"] = _served_url(f"/exports/{quote(job_id)}.zip", ws)
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
    def workflow_schema():
        """The workflow JSON schema, for schema-aware JSON editing."""
        return JSONResponse(load_schema("workflow"))

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

    @app.post("/api/validate")
    def validate_workflow(
        request: JobRequest, ws: Workspace = Depends(selected_workspace)
    ):
        """Schema-validate a workflow and check its pipeline arguments
        against real signatures, without queuing anything. Give either an
        inline workflow or a workflow_path - a path on the server or a
        stored workflow name from /api/workflows. The workspace it resolves
        in comes from the body or the query string, body first."""
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
                candidate = workflow_from_file(
                    resolved,
                    workspace.outputs,
                    # Confined to the source it came from, not to the
                    # writable root - an example is read where it lives
                    source.root if source else workspace.workflows,
                )
                definition = candidate.workflow_definition
            else:
                definition = request.workflow
                candidate = workflow_from_definition(
                    copy.deepcopy(request.workflow),
                    workspace.outputs,
                    request.base_dir,
                    workspace.workflows,
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
        try:
            errors = candidate.validation_errors()
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
        return {
            "valid": True,
            "error": None,
            "errors": [],
            "warnings": workflow_argument_warnings(definition),
        }

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
                workflow_details(found),
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
            raise HTTPException(status_code=400, detail="Provide an inline workflow")
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

        Long defaults - a shot's prompt runs to kilobytes - are cut to their
        first 200 characters and named in `truncated`, so the answer stays
        small for the numbers and names it is usually asked about; `full=true`
        returns them whole, and `GET /api/workflows/{name}` is still the
        definition itself.
        """
        path, source = resolve_readable_workflow(_sources_for(ws), name)
        try:
            with open(path, "r") as file:
                definition = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Could not read workflow: {e}")

        variables = definition.get("variables") or {}
        values, truncated = {}, []
        for variable, value in variables.items():
            if (
                not full
                and isinstance(value, str)
                and len(value) > VARIABLE_VALUE_PREVIEW
            ):
                values[variable] = value[:VARIABLE_VALUE_PREVIEW]
                truncated.append(variable)
            else:
                values[variable] = value
        return {
            "name": name,
            "variables": values,
            "truncated": truncated,
            "seed": definition.get("seed"),
            "origin": source.origin,
        }

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
    def list_prompts():
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
        names = sorted(paths)
        return {
            # The writable library, unchanged: what a save is written to,
            # and what a client that predates the search path expects
            "prompt_dir": app.state.prompt_dir,
            "prompt_dirs": roots,
            "prompts": names,
            "origins": origins,
            "details": prompt_details(paths),
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
        ALLOWED_IMAGE_EXTENSIONS,
        ALLOWED_VIDEO_EXTENSIONS,
    )

    MEDIA_KINDS = {
        **{ext: "image" for ext in ALLOWED_IMAGE_EXTENSIONS},
        **{ext: "video" for ext in ALLOWED_VIDEO_EXTENSIONS},
        **{ext: "audio" for ext in ALLOWED_AUDIO_EXTENSIONS},
    }

    # Longest side of an on-demand gallery thumbnail, in pixels
    GALLERY_THUMBNAIL_MAX_DIM = 320

    def _output_file(name, root=None):
        """A file inside a workspace's output directory, or a 404 - never
        outside it."""
        root = root or manager.output_dir
        try:
            path = validate_path(
                os.path.join(root, name),
                root,
                allow_create=False,
            )
        except SecurityError as e:
            raise HTTPException(status_code=404, detail=f"Unknown file: {e}")
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Unknown file")
        return path

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

    def _iter_gallery_files(root, group_runs=True):
        """Every media file under a directory tree. Yields (relative_name,
        folder, kind, path) - relative_name always uses '/' so it
        round-trips through a URL the same way on every platform.

        With group_runs (the gallery's own use, over the output directory):
        recurses into the per-workflow subfolders (dw/workflow.py's
        effective_output_dir writes each run under '<workflow
        identity>/<run id>/', and mirrors a workflow's position under a
        'workflows' tree in the flat layout), and the folder a file is
        grouped under drops the run id - a workflow run fifty times is one
        folder in the filter, not fifty. Which run a file came from is still
        in its name, and in the manifest beside it.

        Without it (the asset library's use, which has no run ids to strip):
        folder is just the plain relative directory."""
        for current, _dirs, names in os.walk(root):
            rel_root = os.path.relpath(current, root)
            directory = "" if rel_root == "." else rel_root.replace(os.sep, "/")
            for name in names:
                extension = os.path.splitext(name)[1].lower()
                kind = MEDIA_KINDS.get(extension)
                if kind is None:
                    continue
                relative_name = name if not directory else f"{directory}/{name}"
                folder = strip_run_id(relative_name) if group_runs else directory
                yield relative_name, folder, kind, os.path.join(current, name)

    def _gallery_entries(root, ws):
        entries = []
        try:
            files = list(_iter_gallery_files(root))
        except OSError:
            files = []
        for relative_name, folder, kind, path in files:
            try:
                stat = os.stat(path)
            except OSError:
                continue
            # File names look like '{workflow}-{step}.{i}-{j}.{k}.ext'; the
            # part before the first dot is a readable label and embedded
            # metadata carries the precise identity
            label = os.path.basename(relative_name).split(".")[0]
            entries.append(
                {
                    "name": relative_name,
                    "folder": folder,
                    # Quoted (slashes kept literal): a name carrying '#', '?'
                    # or '%' would otherwise break the src the gallery
                    # renders it into. The mtime still rides along for cache
                    # busting when a file's content changes without its name
                    # changing (e.g. a manual overwrite outside the engine) -
                    # normal reruns get a fresh name instead, see
                    # dw/result.py's output_file_path
                    "url": _served_url(
                        f"/outputs/{quote(relative_name)}", ws, int(stat.st_mtime)
                    ),
                    "kind": kind,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "label": label,
                }
            )
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        return entries

    @app.get("/api/gallery")
    def gallery(
        limit: int = 200,
        offset: int = 0,
        folder: Optional[str] = None,
        ws: Workspace = Depends(selected_workspace),
    ):
        """A page of media files in the output directory, newest first.
        Stateless by design - the gallery survives server restarts because
        it reads the directory tree, not job history. 'folders' lists every
        distinct workflow folder present (over the whole directory, not just
        this page), for the UI's folder filter - a run id is not a folder of
        its own, so a workflow's runs group together; '' stands for files
        saved directly at the output root, and is itself always a member so
        that folder-less outputs stay selectable once anything is nested."""
        entries = _gallery_entries(ws.outputs, ws)
        folders = sorted({e["folder"] for e in entries} | {""})
        if folder is not None:
            entries = [e for e in entries if e["folder"] == folder]
        offset = max(0, offset)
        limit = max(0, limit)
        page = entries[offset : offset + limit]
        return {
            "files": page,
            "total": len(entries),
            "offset": offset,
            "limit": limit,
            "folders": folders,
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
        which is how an agent that cannot listen checks a track.

        `envelope=true` adds the soundtrack's level second by second, which
        is what says *where* in a track something is - whether a shot is
        still voiced at its last frame, how deep the hole at a seam goes.
        Opt-in: a ten-minute track is 600 numbers, and the default call has
        to stay small."""
        path = _output_file(name, ws.outputs)
        metadata = read_embedded_metadata(path)
        try:
            # Scoped to this workspace: two workspaces can each write a file
            # with the same relative name, and an unscoped lookup could
            # attribute this one to the wrong workspace's job
            job = manager.history.job_for_file(name, workspace=ws.name)
        except Exception:
            job = None
        extension = os.path.splitext(path)[1].lower()
        media = (
            probe_media(path, envelope=envelope)
            if MEDIA_KINDS.get(extension) in ("audio", "video")
            else None
        )
        return {"name": name, "metadata": metadata, "job": job, "media": media}

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
        paths = [(name, _output_file(name, ws.outputs)) for name in request.names]

        handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        try:
            with handle:
                with zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as archive:
                    for name, path in paths:
                        # the gallery-relative name keeps a workflow's output
                        # subfolders intact inside the download
                        archive.write(path, arcname=name)
        except BaseException:
            os.unlink(handle.name)
            raise

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.info(f"Archived {len(paths)} output files")
        return FileResponse(
            handle.name,
            media_type="application/zip",
            filename=f"dw-outputs-{stamp}.zip",
            background=BackgroundTask(os.unlink, handle.name),
        )

    @app.delete("/api/gallery/{name:path}")
    def delete_output(name: str, ws: Workspace = Depends(selected_workspace)):
        """Remove one file from the output directory."""
        path = _output_file(name, ws.outputs)
        os.remove(path)
        logger.info(f"Deleted output file {name}")
        return {"name": name, "deleted": True}

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
            return {
                "path": f"asset:{UPLOADS_SUBDIR}/{name}",
                "url": _served_url(f"/inputs/{UPLOADS_SUBDIR}/{quote(name)}", ws),
                "shared": shared,
            }
        return {
            "path": dest,
            "url": _served_url(f"/outputs/{UPLOADS_SUBDIR}/{quote(name)}", ws),
        }

    def _asset_origin(ws, index, root):
        """Which library an asset came from: this workspace's own, the one
        shared by every workspace under the root, or a read-only examples
        tree. A client that cannot tell them apart cannot say why deleting
        one answers 403."""
        if index == 0:
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
            return {"asset_dir": library, "asset_dirs": [], "assets": [], "folders": []}

        assets = []
        seen = set()
        for index, root in enumerate(roots):
            try:
                files = list(_iter_gallery_files(root, group_runs=False))
            except OSError:
                files = []
            for relative, folder, kind, path in files:
                # A name in the workspace shadows the same name in an
                # examples library, exactly as 'asset:' resolution does
                if relative in seen:
                    continue
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                seen.add(relative)
                assets.append(
                    {
                        "name": relative,
                        "reference": f"asset:{relative}",
                        "folder": folder,
                        "kind": kind,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "origin": _asset_origin(ws, index, root),
                        # For the editor's own preview - fetchable the same
                        # way an upload's URL is
                        "url": _served_url(f"/inputs/{quote(relative)}", ws),
                    }
                )
        assets.sort(key=lambda entry: entry["mtime"], reverse=True)
        return {
            # The workspace's own library, unchanged: where an upload lands
            "asset_dir": library,
            "asset_dirs": roots,
            "assets": assets,
            "folders": sorted({entry["folder"] for entry in assets} | {""}),
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

        source = _output_file(request.name, ws.outputs)
        asset_name = request.asset_name or os.path.basename(request.name)
        # The kept file's own extension when the name carries none, and a
        # refusal when it carries a contradicting one - exactly what the
        # upload route does with its `asset_name`. Without this a kept asset
        # could be written under an extensionless name, which the library
        # listing (which reads by kind) never shows again: the call reported
        # success and the asset was invisible (T014)
        extension = os.path.splitext(os.path.basename(request.name))[1].lower()
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

        for index, root in enumerate(roots):
            try:
                path = validate_path(os.path.join(root, relative), root)
            except SecurityError:
                continue
            if not os.path.isfile(path):
                continue
            origin = _asset_origin(ws, index, root)
            if origin == EXAMPLES_ORIGIN:
                raise HTTPException(
                    status_code=403,
                    detail=f"asset:{relative} is read-only: it comes from an "
                    f"examples library, not a library this server writes",
                )
            os.remove(path)
            logger.info(f"Deleted asset:{relative} ({path})")
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

    @app.get("/api/health")
    def health():
        import socket

        from .. import __version__, get_device, get_device_type

        worker = manager.worker_manager
        return {
            "status": "ok",
            "version": __version__,
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
    def server_info():
        """How this server is reachable, for the UI's Server page: what it
        is bound to, whether a token is needed, whether MCP is mounted, and
        the addresses another machine could name it by.

        No URL is composed here - the caller pairs an address with `port`
        and `mcp.path` - and the token itself is never reported in any
        form, only whether one is required. An interface enumeration
        failure is not a server failure: `addresses` comes back empty.
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
            "mcp": {"mounted": bool(app.state.mcp_mounted), "path": MCP_PATH},
            "addresses": addresses,
            "directories": {
                # The workspace the three below default to folders of; an
                # individually overridden folder still reports its own path
                "workspace": app.state.workspace,
                "workflows": os.path.abspath(app.state.workflow_dir),
                "assets": app.state.asset_dir,
                "outputs": os.path.abspath(manager.output_dir),
                "prompts": (
                    os.path.abspath(app.state.prompt_dir)
                    if app.state.prompt_dir
                    else None
                ),
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
        return await files.get_response(name, request.scope)

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
            raise HTTPException(status_code=404, detail="No asset library")
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

        handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        try:
            with handle:
                with zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as archive:
                    for current, _dirs, names in os.walk(directory):
                        for name in sorted(names):
                            path = os.path.join(current, name)
                            entry = os.path.relpath(path, directory).replace(
                                os.sep, "/"
                            )
                            archive.write(path, f"{job_id}/{entry}")
        except BaseException:
            # Nothing is going to attach the background unlink now, so the
            # half-written archive has to go here
            os.unlink(handle.name)
            raise

        return FileResponse(
            handle.name,
            media_type="application/zip",
            filename=f"{job_id}.zip",
            # The archive is a temp file, not a second permanent copy - it
            # goes as soon as the response has been sent
            background=BackgroundTask(os.unlink, handle.name),
        )

    # ---------------------------------------------------------------- the UI

    resolved_ui = ui_dir or default_ui_dir()
    if resolved_ui:
        # Mounted last so /api and /outputs keep precedence; html=True serves
        # index.html at /, and the SPA routes by hash so no fallback is needed
        app.mount("/", StaticFiles(directory=resolved_ui, html=True), name="ui")

    return app
