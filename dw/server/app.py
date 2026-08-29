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
    list_pipelines,
    describe_pipeline,
    list_tasks,
    workflow_argument_warnings,
)
from ..workflow import Workflow
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


def resolve_workflow_name(workflow_dir, name):
    """The on-disk path for a workflow name, confined to workflow_dir."""
    if not name.endswith(".json"):
        name = f"{name}.json"
    try:
        return validate_path(
            os.path.join(workflow_dir, name), workflow_dir, allow_create=False
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
        return {
            "workflow_dir": app.state.workflow_dir,
            "workflows": workflow_names(app.state.workflow_dir),
        }

    @app.get("/api/workflows/{name:path}")
    def get_workflow(name: str):
        path = resolve_workflow_name(app.state.workflow_dir, name)
        try:
            with open(path, "r") as file:
                return JSONResponse(json.load(file))
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Could not read workflow: {e}")

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
