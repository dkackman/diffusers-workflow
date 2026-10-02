"""The server's own routes: what diffusers and the engine offer (pipelines,
tasks, classes, the workflow schema, the guides), the model cache and its
downloads, the diffusers updater, memory, health, and how this server is
reachable.

State comes from `request.app.state`, never a closure: `downloads` and
`updater` are per app, so two apps in one process keep their own.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..api_models import (
    ClassList,
    PipelineDescription,
    PipelineNames,
    TaskList,
    DiffusersStatus,
    HealthInfo,
    MemoryCleared,
    MemoryInfo,
    ModelCache,
    ModelDeleted,
    ModelDownload,
    ModelDownloads,
    ServerInfo,
)
from ...hub_cache import delete_model, scan_models
from ...introspection import (
    describe_class,
    describe_pipeline,
    describe_task,
    list_classes,
    list_pipelines,
    list_tasks,
)
from ...schema import SchemaSectionError, load_schema, schema_section
from ...security import InvalidInputError, validate_commit_hash
from ...trust import workflows_are_trusted
from ...workspace import Workspace
from .. import guides
from ..deps import selected_workspace
from ..guides import GuideError
from ..netinfo import local_addresses
from ..sysinfo import runtime_info

logger = logging.getLogger("dw")

router = APIRouter()

# Where the MCP endpoint is mounted when --mcp is given (see `_mount_mcp`
# in dw/server/app.py) - the Server page quotes it in the command it tells you to
# run on the other machine
MCP_PATH = "/mcp"


@router.get(
    "/api/pipelines", response_model=PipelineNames, response_model_exclude_unset=True
)
def pipelines():
    """Every pipeline class the installed diffusers exports."""
    return {"pipelines": list_pipelines()}


@router.get(
    "/api/pipelines/{name}",
    response_model=PipelineDescription,
    response_model_exclude_unset=True,
)
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


@router.get("/api/tasks", response_model=TaskList, response_model_exclude_unset=True)
def tasks():
    """Every task command a workflow's task step can name."""
    return list_tasks()


@router.get(
    "/api/tasks/{command}",
    response_model=PipelineDescription,
    response_model_exclude_unset=True,
)
def get_task(command: str):
    """A task command's argument schema - the registered implementation
    function's real signature, in the same shape as a class description."""
    try:
        return describe_task(command)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/api/classes", response_model=ClassList, response_model_exclude_unset=True)
def classes(kind: str):
    """Class names of one kind (pipelines, models, schedulers,
    quantization) - the pickers' data source."""
    try:
        return {"kind": kind, "classes": list_classes(kind)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/api/classes/{name:path}",
    response_model=PipelineDescription,
    response_model_exclude_unset=True,
)
def class_description(name: str, target: str = "init"):
    """A class's argument schema: target=call reads __call__, init reads
    __init__, load reads from_pretrained plus the curated loading knobs."""
    try:
        return describe_class(name, target=target)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load {name}: {e}")


@router.get("/api/schema")
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


@router.get("/api/guides")
def list_guides():
    """The documentation that bears on choosing a capability: each
    guide's name, what it covers, and its section headings. Served by
    the engine rather than read from an MCP client's install, so the
    guides an agent reads are the guides for the engine it drives."""
    return guides.list_guides()


@router.get("/api/guides/{name}")
def get_guide(name: str, section: Optional[str] = None):
    """One guide from /api/guides, whole or one section of it. A
    section name is matched loosely - case and punctuation dropped -
    so a heading copied approximately still resolves, and also reaches
    a `###` subsection not listed at the top level, by its own heading
    or by a term inside it. An unknown name or section is a 404 whose
    detail lists what exists."""
    try:
        return guides.get_guide(name, section=section)
    except GuideError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/api/models", response_model=ModelCache, response_model_exclude_unset=True)
def get_models():
    """What the Hugging Face hub cache holds, largest repo first."""
    return scan_models()


class DownloadRequest(BaseModel):
    repo_id: str = Field(description="Hub repo to download, e.g. org/model")


@router.post(
    "/api/models/download",
    status_code=202,
    response_model=ModelDownload,
    response_model_exclude_unset=True,
)
def start_download(request: Request, body: DownloadRequest):
    """Start a background snapshot download into the hub cache."""
    state = request.app.state
    try:
        return state.downloads.start(body.repo_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/api/models/downloads",
    response_model=ModelDownloads,
    response_model_exclude_unset=True,
)
def list_downloads(request: Request):
    state = request.app.state
    return {"downloads": state.downloads.status_list()}


@router.post(
    "/api/models/downloads/{download_id}/cancel",
    response_model=ModelDownload,
    response_model_exclude_unset=True,
)
def cancel_download(request: Request, download_id: str):
    """Request cancellation; takes effect at the next progress tick.
    Partial files stay in the cache and resume on a retry."""
    state = request.app.state
    status = state.downloads.cancel(download_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown download")
    return status


@router.delete(
    "/api/models", response_model=ModelDeleted, response_model_exclude_unset=True
)
def delete_cached_model(request: Request, repo: str):
    """Delete every cached revision of one repo from the hub cache.

    Refused while a job is running or queued: the worker may be reading
    exactly the files a delete would remove out from under it."""
    state = request.app.state
    manager = request.app.state.job_manager
    if manager.is_busy():
        raise HTTPException(
            status_code=409,
            detail="A job is running or queued - deleting model files "
            "out from under it would corrupt the run",
        )
    if state.downloads.is_active():
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


@router.get(
    "/api/system/diffusers",
    response_model=DiffusersStatus,
    response_model_exclude_unset=True,
)
def diffusers_state(request: Request):
    """Installed diffusers version (with its git commit when installed
    from git) and the state of any update."""
    state = request.app.state
    return state.updater.status()


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


@router.post(
    "/api/system/diffusers/update",
    status_code=202,
    response_model=DiffusersStatus,
    response_model_exclude_unset=True,
)
def update_diffusers(
    request: Request, body: UpdateDiffusersRequest = UpdateDiffusersRequest()
):
    """Upgrade diffusers in the background: GitHub HEAD by default, a
    pinned commit when `commit` is given, or a revert to the last
    known-good published release when `revert` is true.

    Refused while a job is running or queued: pip replacing package
    files under a loaded pipeline is the model-delete hazard in another
    form. On success the idle worker is shut down so the next job
    imports the new version."""
    state = request.app.state
    manager = request.app.state.job_manager
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
    if state.downloads.is_active():
        raise HTTPException(
            status_code=409,
            detail="A model download is in progress - replacing package "
            "files while it runs could corrupt the download",
        )
    try:
        return state.updater.start(
            on_success=manager.restart_worker_if_idle,
            commit=commit,
            revert=body.revert,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


# --------------------------------------------------------- memory/health


@router.get("/api/memory", response_model=MemoryInfo, response_model_exclude_unset=True)
def memory(request: Request):
    manager = request.app.state.job_manager
    try:
        return manager.memory_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Worker unavailable: {e}")


@router.post(
    "/api/memory/clear", response_model=MemoryCleared, response_model_exclude_unset=True
)
def clear_memory(request: Request):
    """Drop every loaded pipeline and the step cache, freeing VRAM/RAM
    without waiting for the next job to evict one model for another.

    Refused while a job is running or queued (409) rather than blocked -
    the queue is FIFO, so the caller should wait for the job to finish
    and retry instead of this call stalling until it does.

    A server with no worker process resident answers `cleared` with a
    null `info` rather than a 503: the worker is on-demand, so its
    absence means there was nothing loaded to clear."""
    manager = request.app.state.job_manager
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


@router.get("/api/health", response_model=HealthInfo, response_model_exclude_unset=True)
def health(request: Request):
    state = request.app.state
    manager = request.app.state.job_manager
    import socket

    from ... import __version__, get_device, get_device_type

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
        "mcp": bool(state.mcp_mounted),
    }


@router.get("/api/server", response_model=ServerInfo, response_model_exclude_unset=True)
def server_info(request: Request, ws: Workspace = Depends(selected_workspace)):
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
    state = request.app.state
    import socket

    from ... import __version__, get_device, get_device_type

    try:
        addresses = local_addresses()
    except Exception:
        logger.debug("Could not enumerate local addresses", exc_info=True)
        addresses = []
    return {
        "hostname": socket.gethostname(),
        "version": __version__,
        "device": get_device_type(get_device()),
        "bind_host": state.bind_host,
        "port": state.bind_port,
        "wildcard_bind": state.wildcard_bind,
        "auth_required": bool(state.api_token),
        # The posture a security check has to know it is testing: with
        # this off, a workflow file is untrusted input - no arbitrary
        # imports, no remote code, no location outside the workspace's
        # roots. It is not a secret (the refusals name the flag), and
        # without it the posture could only be inferred from behavior
        # (#120)
        "trust_workflows": workflows_are_trusted(),
        "mcp": {"mounted": bool(state.mcp_mounted), "path": MCP_PATH},
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
