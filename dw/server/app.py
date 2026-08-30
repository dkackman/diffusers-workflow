"""FastAPI application exposing the workflow engine.

All state lives in the JobManager; this module is routing, validation and
SSE framing. Everything path-shaped goes through dw.security validators.
Interactive API docs are served at /docs (OpenAPI at /openapi.json).
"""

import os
import copy
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..security import validate_path, SecurityError
from ..introspection import (
    describe_class,
    list_classes,
    list_pipelines,
    describe_pipeline,
    list_tasks,
    workflow_argument_warnings,
)
from ..schema import load_schema
from ..workflow import Workflow
from ..result import read_embedded_metadata
from .jobs import JobManager, TERMINAL_STATES

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


def workflow_details(workflow_dir, names):
    """Per-workflow card metadata: output kinds and variable count."""
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
            detail = {
                "kinds": kinds,
                "variables": len(definition.get("variables", {})),
            }
        except Exception:
            detail = {"kinds": [], "variables": 0}
        _workflow_detail_cache[path] = (mtime, detail)
        details[name] = detail
    return details


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


def default_ui_dir():
    """The built SPA (ui/dist) when running from a checkout, else None."""
    candidate = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "ui",
        "dist",
    )
    return candidate if os.path.isfile(os.path.join(candidate, "index.html")) else None


def create_app(
    workflow_dir="./examples",
    output_dir="./outputs",
    log_level="INFO",
    job_manager=None,
    ui_dir=None,
):
    """Build the application. A caller (tests) can inject a JobManager."""
    manager = job_manager or JobManager(output_dir, log_level=log_level)

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

    @app.middleware("http")
    async def reject_foreign_origins(request, call_next):
        """Refuse browser cross-origin requests - a drive-by web page must
        not be able to queue jobs on a localhost GPU server. Requests
        without an Origin header (curl, scripts, same-origin GETs) pass."""
        origin = request.headers.get("origin")
        if origin:
            host = urlparse(origin).hostname
            if host not in ("localhost", "127.0.0.1", "::1"):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin requests are not allowed"},
                )
        return await call_next(request)

    # ------------------------------------------------------------------ jobs

    @app.post("/api/jobs", status_code=201)
    def submit_job(request: JobRequest):
        try:
            job = manager.submit(
                workflow_path=request.workflow_path,
                workflow=request.workflow,
                arguments=request.arguments,
                base_dir=request.base_dir,
            )
        except (ValueError, SecurityError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # workflow_from_file / validate raise plain Exceptions for bad
            # files and schema failures - those are the client's fault too
            raise HTTPException(status_code=400, detail=str(e))
        return job.detail()

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": manager.list()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        # a historical job is already a detail dict; a live one renders itself
        return job if isinstance(job, dict) else job.detail()

    @app.post("/api/jobs/{job_id}/rerun", status_code=201)
    def rerun_job(job_id: str):
        """Queue a fresh job from a previous job's stored spec."""
        try:
            job = manager.rerun(job_id)
        except (ValueError, SecurityError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return job.detail()

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
        against real signatures, without queuing anything."""
        if request.workflow is None:
            raise HTTPException(status_code=400, detail="Provide an inline workflow")
        candidate = Workflow(
            copy.deepcopy(request.workflow),
            manager.output_dir,
            os.path.join(request.base_dir or os.getcwd(), "__inline__.json"),
        )
        try:
            candidate.validate()
        except Exception as e:
            return {"valid": False, "error": str(e), "warnings": []}
        return {
            "valid": True,
            "error": None,
            "warnings": workflow_argument_warnings(request.workflow),
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
        candidate = Workflow(copy.deepcopy(request.workflow), manager.output_dir, path)
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

    @app.get("/api/workflows/{name:path}")
    def get_workflow(name: str):
        path = resolve_workflow_name(app.state.workflow_dir, name)
        try:
            with open(path, "r") as file:
                return JSONResponse(json.load(file))
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Could not read workflow: {e}")

    # --------------------------------------------------------------- gallery

    MEDIA_KINDS = {
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".webp": "image",
        ".gif": "image",
        ".mp4": "video",
        ".webm": "video",
        ".wav": "audio",
        ".mp3": "audio",
        ".flac": "audio",
        ".ogg": "audio",
    }

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

    @app.get("/api/gallery")
    def gallery(limit: int = 200):
        """Media files in the output directory, newest first. Stateless by
        design - the gallery survives server restarts because it reads the
        directory, not job history."""
        entries = []
        try:
            names = os.listdir(manager.output_dir)
        except OSError:
            names = []
        for name in names:
            extension = os.path.splitext(name)[1].lower()
            kind = MEDIA_KINDS.get(extension)
            if kind is None:
                continue
            path = os.path.join(manager.output_dir, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            entries.append(
                {
                    "name": name,
                    "url": f"/outputs/{name}",
                    "kind": kind,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    # File names look like '{workflow}-{step}.{i}-{j}.{k}.ext';
                    # the part before the first dot is a readable label and
                    # embedded metadata carries the precise identity
                    "label": name.split(".")[0],
                }
            )
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        return {"files": entries[: max(0, limit)], "total": len(entries)}

    @app.get("/api/gallery/{name}/metadata")
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

    @app.delete("/api/gallery/{name}")
    def delete_output(name: str):
        """Remove one file from the output directory."""
        path = _output_file(name)
        os.remove(path)
        logger.info(f"Deleted output file {name}")
        return {"name": name, "deleted": True}

    # --------------------------------------------------------- memory/health

    @app.get("/api/memory")
    def memory():
        try:
            return manager.memory_status()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Worker unavailable: {e}")

    @app.get("/api/health")
    def health():
        worker = manager.worker_manager
        return {
            "status": "ok",
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
