"""FastAPI application exposing the workflow engine.

All state lives in the JobManager; this module is routing, validation and
SSE framing. Everything path-shaped goes through dw.security validators.
Interactive API docs are served at /docs (OpenAPI at /openapi.json).
"""

import os
import io
import copy
import json
import uuid
import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from urllib.parse import quote, urlparse
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..security import (
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
from ..schema import load_schema, validate_data
from ..prompts import PROMPT_PREFIX, RESERVED_TEXT_PREFIXES
from ..workflow import Workflow, workflow_from_definition, workflow_from_file
from .enhancers import build_enhance_workflow, preset_descriptions
from ..result import read_embedded_metadata
from ..hub_cache import scan_models, delete_model, DownloadManager
from .jobs import JobManager, MAX_PERSISTED_EVENTS, TERMINAL_STATES
from .updater import DiffusersUpdater

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


def workflow_names(workflow_dir):
    """Workflow names under workflow_dir, as relative paths without .json."""
    names = []
    if not os.path.isdir(workflow_dir):
        return names
    for root, _dirs, files in os.walk(workflow_dir):
        for file_name in files:
            if file_name.endswith(".json"):
                relative = os.path.relpath(os.path.join(root, file_name), workflow_dir)
                names.append(relative[: -len(".json")])
    return sorted(names)


# What each workflow produces and takes, for listing cards - cached by mtime
_workflow_detail_cache = {}


def _prune_detail_cache(cache, directory, names):
    """Forget files a listing no longer names - a long-lived server that
    creates and deletes scratch files would otherwise grow the cache forever."""
    live = {os.path.join(directory, f"{name}.json") for name in names}
    for stale in [path for path in cache if path not in live]:
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


def workflow_details(workflow_dir, names):
    """Per-workflow card metadata: output kinds, step and variable counts,
    and the variable names themselves - enough for an agent to pick a
    workflow and know what to pass it without fetching each candidate. The
    names but not their defaults: across the workflows on disk the defaults
    are an order of magnitude more payload, on a listing the UI reloads."""
    details = {}
    for name in names:
        path = os.path.join(workflow_dir, f"{name}.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        cached = _workflow_detail_cache.get(path)
        if cached and cached[0] == mtime:
            details[name] = cached[1]
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
            detail = {
                "kinds": kinds,
                "steps": len(definition.get("steps", [])),
                "variables": len(variables),
                "variable_names": sorted(variables),
                "description": str(definition.get("description", "") or ""),
                "prompt_refs": sorted(collect_prompt_references(definition)),
            }
        except Exception:
            detail = {
                "kinds": [],
                "steps": 0,
                "variables": 0,
                "variable_names": [],
                "description": "",
                "prompt_refs": [],
            }
        _workflow_detail_cache[path] = (mtime, detail)
        details[name] = detail
    _prune_detail_cache(_workflow_detail_cache, workflow_dir, names)
    return details


def _write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)


def resolve_workflow_name(workflow_dir, name, allow_create=False):
    """The on-disk path for a workflow name, confined to workflow_dir."""
    if not name.endswith(".json"):
        name = f"{name}.json"
    try:
        return validate_path(
            os.path.join(workflow_dir, name), workflow_dir, allow_create=allow_create
        )
    except SecurityError as e:
        raise HTTPException(status_code=404, detail=f"Unknown workflow: {e}")


def resolve_workflow_reference(workflow_dir, workflow_path):
    """A submitted workflow_path, resolved to a file on disk, confined to
    workflow_dir - the same confinement the /api/workflows CRUD routes
    already enforce via resolve_workflow_name.

    Tried as a stored workflow name - exactly what /api/workflows hands
    out, with or without .json and nested names included - so an agent can
    run what a listing gave it. A relative or absolute path that already
    names a file under workflow_dir resolves the same way: os.path.join
    discards workflow_dir in favor of an absolute second argument, so an
    absolute path under workflow_dir reaches the same containment check.
    Anything that does not resolve under workflow_dir - an unknown name, a
    traversal attempt, or a real file elsewhere on disk - is rejected with
    400, rather than silently opened: a workflow_path is not a general
    filesystem path.
    """
    if workflow_path is None:
        return workflow_path
    try:
        return resolve_workflow_name(workflow_dir, workflow_path)
    except HTTPException:
        pass
    # Not a stored name. A path relative to the server's cwd - the shape the
    # Workflow page submits when --workflow-dir is itself relative, e.g.
    # './workflows/x.json' against './workflows' - would double the directory
    # if joined onto workflow_dir, so it is resolved from the cwd and then
    # held to the same containment check.
    try:
        return validate_path(os.path.abspath(workflow_path), workflow_dir)
    except SecurityError:
        raise HTTPException(
            status_code=400,
            detail=f"workflow_path must name a workflow under the workflow "
            f"directory: {workflow_path}",
        )


# What each prompt says about itself, for listing cards - cached by mtime
_prompt_detail_cache = {}


def prompt_details(prompt_dir, names):
    """Per-prompt card metadata: description, intended model, tags - and
    the text itself, which the editors show as the tooltip wherever a
    prompt: reference stands in for it."""
    details = {}
    for name in names:
        path = os.path.join(prompt_dir, f"{name}.json")
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
    _prune_detail_cache(_prompt_detail_cache, prompt_dir, names)
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
# Routes a browser loads without being able to set headers (EventSource, an
# <img> tag, an <a download> navigation); these alone accept the bearer
# token as a ?token= query param
QUERY_TOKEN_ROUTE_SUFFIXES = ("/events", "/thumbnail", "/download")


def create_app(
    workflow_dir="./workflows",
    output_dir="./outputs",
    log_level="INFO",
    job_manager=None,
    ui_dir=None,
    download_manager=None,
    diffusers_updater=None,
    prompt_dir="./prompts",
    host="127.0.0.1",
    token=None,
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

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.shutdown()

    app = FastAPI(
        title="diffusers-workflow",
        description="Declarative diffusers workflows over HTTP: queue a job, "
        "stream its progress, fetch what it saved.",
        lifespan=lifespan,
    )
    app.state.job_manager = manager
    app.state.workflow_dir = workflow_dir
    app.state.prompt_dir = prompt_dir

    @app.middleware("http")
    async def reject_foreign_origins(request, call_next):
        """Refuse browser cross-origin requests - a drive-by web page must
        not be able to queue jobs on a localhost GPU server. Requests
        without an Origin header (curl, scripts, same-origin GETs) pass."""
        origin = request.headers.get("origin")
        if origin:
            origin_host = urlparse(origin).hostname
            if origin_host not in LOOPBACK_HOSTS:
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
    wildcard_bind = host in WILDCARD_HOSTS
    allowed_hosts = set(LOOPBACK_HOSTS)
    if host:
        allowed_hosts.add(host.lower())

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
        if not path.startswith("/api/"):
            return await call_next(request)
        provided = None
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[len("bearer ") :].strip()
        # GET only: every header-less browser load is a GET, and this keeps
        # the query form off state-changing routes that happen to share a
        # suffix (POST /api/models/download)
        if (
            provided is None
            and request.method == "GET"
            and path.endswith(QUERY_TOKEN_ROUTE_SUFFIXES)
        ):
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

    # ------------------------------------------------------------------ jobs

    @app.post("/api/jobs", status_code=201)
    def submit_job(request: JobRequest):
        try:
            job = manager.submit(
                workflow_path=resolve_workflow_reference(
                    app.state.workflow_dir, request.workflow_path
                ),
                workflow=request.workflow,
                arguments=request.arguments,
                base_dir=request.base_dir,
            )
        except HTTPException:
            raise
        except Exception as e:
            # workflow_from_file / validate / the security layer all raise for
            # bad requests - every failure here is the client's fault
            raise HTTPException(status_code=400, detail=str(e))
        return manager.describe(job)

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": manager.list()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        # a historical job is already a detail dict; a live one renders itself
        return job if isinstance(job, dict) else manager.describe(job)

    @app.post("/api/jobs/{job_id}/rerun", status_code=201)
    def rerun_job(job_id: str):
        """Queue a fresh job from a previous job's stored spec."""
        try:
            job = manager.rerun(job_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return manager.describe(job)

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

    @app.post("/api/validate")
    def validate_workflow(request: JobRequest):
        """Schema-validate a workflow and check its pipeline arguments
        against real signatures, without queuing anything. Give either an
        inline workflow or a workflow_path - a path on the server or a
        stored workflow name from /api/workflows."""
        if (request.workflow is None) == (request.workflow_path is None):
            raise HTTPException(
                status_code=400,
                detail="Provide exactly one of workflow or workflow_path",
            )
        try:
            if request.workflow_path is not None:
                # Built from the file so relative paths inside it resolve
                # against its own directory, exactly as a run would
                candidate = workflow_from_file(
                    resolve_workflow_reference(
                        app.state.workflow_dir, request.workflow_path
                    ),
                    manager.output_dir,
                    app.state.workflow_dir,
                )
                definition = candidate.workflow_definition
            else:
                definition = request.workflow
                candidate = workflow_from_definition(
                    copy.deepcopy(request.workflow),
                    manager.output_dir,
                    request.base_dir,
                    app.state.workflow_dir,
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
            candidate.validate()
        except Exception as e:
            return {"valid": False, "error": str(e), "warnings": []}
        return {
            "valid": True,
            "error": None,
            "warnings": workflow_argument_warnings(definition),
        }

    # ------------------------------------------------------------- workflows

    @app.get("/api/workflows")
    def list_workflows():
        names = workflow_names(app.state.workflow_dir)
        return {
            "workflow_dir": app.state.workflow_dir,
            "workflows": names,
            "details": workflow_details(app.state.workflow_dir, names),
        }

    @app.put("/api/workflows/{name:path}")
    def save_workflow(name: str, request: JobRequest):
        """Write a workflow into the workflow directory. The definition must
        be schema-valid - the editor validates before saving, and a save that
        silently wrote a broken file would betray both."""
        if request.workflow is None:
            raise HTTPException(status_code=400, detail="Provide an inline workflow")
        path = resolve_workflow_name(app.state.workflow_dir, name, allow_create=True)
        candidate = Workflow(
            copy.deepcopy(request.workflow),
            manager.output_dir,
            path,
            app.state.workflow_dir,
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
        return {
            "name": name,
            "path": path,
            "warnings": workflow_argument_warnings(request.workflow),
        }

    @app.delete("/api/workflows/{name:path}")
    def delete_workflow(name: str):
        """Remove a workflow file from the workflow directory."""
        path = resolve_workflow_name(app.state.workflow_dir, name)
        os.remove(path)
        logger.info(f"Deleted workflow {name} ({path})")
        return {"name": name, "deleted": True}

    @app.get("/api/workflows/{name:path}/download")
    def download_workflow(name: str):
        """Serve a workflow definition as a forced download."""
        path = resolve_workflow_name(app.state.workflow_dir, name)
        return FileResponse(
            path, filename=os.path.basename(path), media_type="application/json"
        )

    @app.get("/api/workflows/{name:path}")
    def get_workflow(name: str):
        path = resolve_workflow_name(app.state.workflow_dir, name)
        try:
            with open(path, "r") as file:
                return JSONResponse(json.load(file))
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Could not read workflow: {e}")

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

    @app.get("/api/prompts")
    def list_prompts():
        # A stray file too deep or oddly named can sit in the directory, but
        # no workflow could reference it - listing it would only invite that
        names = [n for n in workflow_names(app.state.prompt_dir) if referenceable(n)]
        return {
            "prompt_dir": app.state.prompt_dir,
            "prompts": names,
            "details": prompt_details(app.state.prompt_dir, names),
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
        """Remove a prompt file from the prompt directory."""
        path = resolve_prompt_name(app.state.prompt_dir, name)
        os.remove(path)
        logger.info(f"Deleted prompt {name} ({path})")
        return {"name": name, "deleted": True}

    @app.get("/api/prompts/{name:path}/download")
    def download_prompt(name: str):
        """Serve a stored prompt as a forced download."""
        path = resolve_prompt_name(app.state.prompt_dir, name)
        return FileResponse(
            path, filename=os.path.basename(path), media_type="application/json"
        )

    @app.get("/api/prompts/{name:path}")
    def get_prompt(name: str):
        path = resolve_prompt_name(app.state.prompt_dir, name)
        try:
            with open(path, "r") as file:
                return JSONResponse(json.load(file))
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
    def enhance(request: EnhanceRequest):
        """Queue a prompt enhancement as an ordinary job. The enhanced text
        is the job's single manifest file once it succeeds."""
        try:
            definition = build_enhance_workflow(
                request.preset,
                request.idea,
                model_name=request.model_name,
                device=request.device,
            )
            job = manager.submit(workflow=definition, arguments={})
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

    def _output_file(name):
        """A file inside the output directory, or a 404 - never outside it."""
        try:
            path = validate_path(
                os.path.join(manager.output_dir, name),
                manager.output_dir,
                allow_create=False,
            )
        except SecurityError as e:
            raise HTTPException(status_code=404, detail=f"Unknown file: {e}")
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Unknown file")
        return path

    def _iter_gallery_files():
        """Every media file under the output directory, recursing into the
        per-workflow subfolders (dw/workflow.py's effective_output_dir mirrors
        a workflow's position under a 'workflows' tree into the output dir).
        Yields (relative_name, folder, kind, path) - relative_name always
        uses '/' so it round-trips through a URL the same way on every
        platform."""
        for root, _dirs, names in os.walk(manager.output_dir):
            rel_root = os.path.relpath(root, manager.output_dir)
            folder = "" if rel_root == "." else rel_root.replace(os.sep, "/")
            for name in names:
                extension = os.path.splitext(name)[1].lower()
                kind = MEDIA_KINDS.get(extension)
                if kind is None:
                    continue
                relative_name = name if not folder else f"{folder}/{name}"
                yield relative_name, folder, kind, os.path.join(root, name)

    def _gallery_entries():
        entries = []
        try:
            files = list(_iter_gallery_files())
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
                    "url": f"/outputs/{quote(relative_name)}?v={int(stat.st_mtime)}",
                    "kind": kind,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "label": label,
                }
            )
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        return entries

    @app.get("/api/gallery")
    def gallery(limit: int = 200, offset: int = 0, folder: Optional[str] = None):
        """A page of media files in the output directory, newest first.
        Stateless by design - the gallery survives server restarts because
        it reads the directory tree, not job history. 'folders' lists every
        distinct workflow subfolder present (over the whole directory, not
        just this page), for the UI's folder filter - '' stands for files
        saved directly at the output root, and is itself always a member so
        that folder-less outputs stay selectable once anything is nested."""
        entries = _gallery_entries()
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
        }

    @app.get("/api/gallery/{name:path}/metadata")
    def gallery_metadata(name: str):
        """Generation metadata embedded in a saved image ('workflow' inside
        it is the full definition the editor can reopen), plus the job that
        produced the file when history remembers one."""
        path = _output_file(name)
        metadata = read_embedded_metadata(path)
        try:
            job = manager.history.job_for_file(name)
        except Exception:
            job = None
        return {"name": name, "metadata": metadata, "job": job}

    @app.get("/api/gallery/{name:path}/thumbnail")
    def gallery_thumbnail(name: str, request: Request):
        """A small JPEG rendition of an image output, for the grid - the
        full-resolution file is only fetched for the detail/lightbox view.
        Generated on demand rather than cached to disk, so it never grows
        the output directory the gallery itself scans."""
        path = _output_file(name)
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
    def download_output(name: str):
        """Serve one output file as a forced download rather than an inline view."""
        path = _output_file(name)
        return FileResponse(path, filename=os.path.basename(name))

    @app.delete("/api/gallery/{name:path}")
    def delete_output(name: str):
        """Remove one file from the output directory."""
        path = _output_file(name)
        os.remove(path)
        logger.info(f"Deleted output file {name}")
        return {"name": name, "deleted": True}

    # ---------------------------------------------------------------- uploads

    UPLOADS_SUBDIR = "uploads"
    ALLOWED_UPLOAD_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
    MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB - covers a short video clip

    @app.post("/api/uploads", status_code=201)
    async def upload_media(request: Request, filename: str):
        """Save a browser-picked image or video into the output directory's
        uploads/ subfolder and hand back its path - the same string shape a
        workflow's 'image'/'video' arguments already accept (a plain path,
        resolved absolute so it works regardless of the workflow file's own
        directory). The body is the raw file bytes: no multipart parser
        dependency needed for a single-file upload.
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

        uploads_dir = os.path.join(manager.output_dir, UPLOADS_SUBDIR)
        os.makedirs(uploads_dir, exist_ok=True)
        name = f"{uuid.uuid4().hex}{extension}"
        try:
            dest = validate_output_path(os.path.join(uploads_dir, name), uploads_dir)
        except SecurityError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Off the event loop: a 200 MB write would otherwise stall every SSE
        # stream and poll for its duration
        await run_in_threadpool(_write_bytes, dest, body)
        logger.info(f"Saved upload {filename!r} -> {dest}")
        return {
            "path": dest,
            "url": f"/outputs/{UPLOADS_SUBDIR}/{quote(name)}",
        }

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
        from .. import __version__

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
        }

    # ---------------------------------------------------------------- outputs

    app.mount("/outputs", StaticFiles(directory=manager.output_dir), name="outputs")

    # ---------------------------------------------------------------- the UI

    resolved_ui = ui_dir or default_ui_dir()
    if resolved_ui:
        # Mounted last so /api and /outputs keep precedence; html=True serves
        # index.html at /, and the SPA routes by hash so no fallback is needed
        app.mount("/", StaticFiles(directory=resolved_ui, html=True), name="ui")

    return app
