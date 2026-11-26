"""The server's own routes: what diffusers and the engine offer (pipelines,
tasks, classes, the workflow schema, the guides), the model cache and its
downloads, the diffusers updater, memory, health, how this server is
reachable, and the preferences store (below).

State comes from `request.app.state`, never a closure: `downloads` and
`updater` are per app, so two apps in one process keep their own.
"""

import json
import logging
from contextlib import contextmanager
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..api_models import (
    ClassList,
    PipelineDescription,
    PipelineNames,
    TaskList,
    DiffusersStatus,
    DwUpdateState,
    DiskCacheCleared,
    DiskCacheEntryDeleted,
    DiskCacheList,
    HealthInfo,
    MemoryCleared,
    MemoryStatus,
    ModelCache,
    ModelDeleted,
    ModelDownload,
    ModelDownloads,
    ServerInfo,
)
from ... import disk_caches, model_use, preferences
from ...hub_cache import delete_model, scan_models
from ...introspection import (
    describe_class,
    describe_pipeline,
    describe_task,
    list_classes,
    list_pipelines,
    list_tasks,
)
from ...log_setup import log_storage_stats
from ...schema import SchemaSectionError, load_schema, schema_section
from ...security import InvalidInputError, validate_commit_hash
from ... import __version__
from ...trust import workflows_are_trusted
from ...workflow import workflow_from_definition
from ...workspace import Workspace
from .. import guides
from ..api_models_library import (
    DefaultTierSet,
    DefaultTierWrite,
    PreferenceReset,
    PreferenceSet,
    PreferencesRead,
)
from ..changes import MODELS, NO_CHANGE, WORKFLOWS, ChangeRoute, writes
from ..deps import internal_error, selected_workspace
from ..preconditions import check_if_match, content_etag_of
from ..guides import GuideError
from ..jobs import JobsActive, WorkerBusy
from ..netinfo import local_addresses
from ..self_update import install_command, latest_version
from ..sysinfo import runtime_info

logger = logging.getLogger("dw")

router = APIRouter(route_class=ChangeRoute)

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
    """What the Hugging Face hub cache holds, largest repo first. Each row
    carries `last_job_use`, when a dw job that named the repo last started
    (dw/model_use.py, #1445), null when none has - always set, so the field
    is never dropped as unset. `last_accessed` stays the files' atime."""
    cache = scan_models()
    used = model_use.read()
    for repo in cache.get("repos") or []:
        repo["last_job_use"] = used.get(repo.get("repo_id"))
    return cache


class DownloadRequest(BaseModel):
    repo_id: str = Field(description="Hub repo to download, e.g. org/model")


@router.post(
    "/api/models/download",
    status_code=202,
    response_model=ModelDownload,
    response_model_exclude_unset=True,
)
@writes(MODELS)
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
@writes(MODELS)
def cancel_download(request: Request, download_id: str):
    """Request cancellation; `cancel_requested` shows until it is terminal.
    Partial files stay in the cache and resume on a retry."""
    state = request.app.state
    status = state.downloads.cancel(download_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown download")
    return status


@router.delete(
    "/api/models", response_model=ModelDeleted, response_model_exclude_unset=True
)
@writes(MODELS)
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
    # A later re-download must not inherit the deleted copy's date (#1445)
    model_use.forget(repo)
    logger.info(f"Deleted {repo} from the hub cache ({freed} bytes)")
    return {"repo_id": repo, "deleted": True, "freed": freed}


# ------------------------------------------------------ dw's disk caches


@router.get(
    "/api/caches", response_model=DiskCacheList, response_model_exclude_unset=True
)
def list_caches():
    """dw's own disk caches (dw/disk_caches.py): each one's root, size, and
    whether it can be cleared from here."""
    return {"caches": disk_caches.scan_all()}


@router.delete(
    "/api/caches/{name}",
    response_model=DiskCacheCleared,
    response_model_exclude_unset=True,
)
@writes(MODELS)
def clear_cache(request: Request, name: str):
    """Empty one cache's root, keeping the root and its marker. Refused
    while anything may be writing into it (`_hold_writers_off`)."""
    try:
        disk_caches.get(name)
    except disk_caches.UnknownCache as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        with _hold_writers_off(request, name):
            result = disk_caches.clear(name)
    except disk_caches.NotClearable as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"name": name, "cleared": True, **result}


@router.delete(
    "/api/caches/{name}/entries/{entry}",
    response_model=DiskCacheEntryDeleted,
    response_model_exclude_unset=True,
)
@writes(MODELS)
def delete_cache_entry(request: Request, name: str, entry: str):
    """Delete one entry of a cache that has entries (quantized). The id is
    checked to be 64 hex before any path is built from it."""
    try:
        disk_caches.entry_cache(name, entry)
    except disk_caches.UnknownCache as e:
        raise HTTPException(status_code=404, detail=str(e))
    except disk_caches.NoEntries as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        with _hold_writers_off(request, name):
            result = disk_caches.delete_entry(name, entry)
    except disk_caches.UnknownEntry as e:
        raise HTTPException(status_code=404, detail=str(e))
    except disk_caches.NotClearable as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"name": name, "entry": entry, "deleted": True, "freed": result["freed"]}


# Caches that live beside the hub cache (under HF_HOME): changed only when no
# download is writing there either, the rule DELETE /api/models follows
_DOWNLOAD_SENSITIVE_CACHES = frozenset({"quantized"})


@contextmanager
def _hold_writers_off(request, name):
    """Hold off whatever may write into cache `name` for the block: 409 while
    a job is running or queued - a compile, autotune or quantized save in
    progress writes into the root, and a delete mid-write could leave a
    truncated artifact a later lookup trusts - and, for the quantized cache,
    while a model download is active. A job submitted during the block waits
    in the queue until it ends: the busy check and the delete are not two
    steps a submit can fall between."""
    state = request.app.state
    if name in _DOWNLOAD_SENSITIVE_CACHES and state.downloads.is_active():
        raise HTTPException(
            status_code=409,
            detail="A model download is in progress - the quantized cache "
            "can't be changed until it finishes",
        )
    try:
        with state.job_manager.hold_dispatch():
            yield
    except JobsActive:
        raise HTTPException(
            status_code=409,
            detail="A job is running or queued - clearing a cache it may be "
            "writing would leave it a truncated artifact",
        )


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
@writes(MODELS)
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
    if state.dw_updater.status()["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="A dw update is running - two pip runs in one "
            "environment could corrupt it",
        )
    try:
        return state.updater.start(
            on_success=manager.restart_worker_if_idle,
            commit=commit,
            revert=body.revert,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ------------------------------------------------------------ dw update


def _dw_state(request):
    state = request.app.state
    return {
        "version": __version__,
        "latest": latest_version(),
        "managed": bool(state.managed),
        "command": install_command(),
        **state.dw_updater.status(),
    }


@router.get("/api/system/dw", response_model=DwUpdateState)
def dw_state(request: Request):
    """The installed dw version, the newest release, whether this server can
    update itself, and the state of any update."""
    return _dw_state(request)


@router.post("/api/system/dw/update", status_code=202, response_model=DwUpdateState)
@writes(MODELS)
def update_dw(request: Request):
    """Update dw in the background on an installed copy, then restart the
    server through its service manager.

    Refused on an unmanaged server (its `command` is the way), and while a
    job is running or queued or a download is active: pip replacing package
    files under a loaded pipeline is the diffusers update's hazard, and the
    restart that follows would end the job."""
    state = request.app.state
    if not state.managed:
        raise HTTPException(
            status_code=409,
            detail="This server is not managed by a service, so it cannot "
            f"restart itself - update it by running: {install_command()}",
        )
    if state.job_manager.is_busy():
        raise HTTPException(
            status_code=409,
            detail="A job is running or queued - updating dw would restart "
            "the server underneath it",
        )
    if state.downloads.is_active():
        raise HTTPException(
            status_code=409,
            detail="A model download is in progress - the restart that "
            "follows the update would cut it off",
        )
    if state.updater.status()["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="A diffusers update is running - two pip runs in one "
            "environment could corrupt it",
        )
    try:
        state.dw_updater.start()
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _dw_state(request)


# --------------------------------------------------------- memory/health


DEVICE_QUERY = Query(
    None,
    max_length=32,
    description="The card to ask about, as `cuda:1`; every card when omitted.",
)


@router.get(
    "/api/memory", response_model=MemoryStatus, response_model_exclude_unset=True
)
def memory(request: Request, device: Optional[str] = DEVICE_QUERY):
    """Memory per card: `device`'s reading, or the first card's at the top
    level with `workers` holding one reading per card."""
    manager = request.app.state.job_manager
    try:
        return manager.memory_status(device=device)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Worker unavailable: {e}")


@router.post(
    "/api/memory/clear", response_model=MemoryCleared, response_model_exclude_unset=True
)
@NO_CHANGE("memory is live state on the status poll")
def clear_memory(request: Request, device: Optional[str] = DEVICE_QUERY):
    """Drop every loaded pipeline and the step cache on each idle card -
    `device`'s alone when named - freeing VRAM/RAM without waiting for
    the next job to evict one model for another.

    A card running a job is left alone, and `workers` says so. Refused
    (409) rather than blocked when no card asked about is idle - `device`'s
    is running a job, even with another card idle - so the caller waits
    for the job to finish and retries instead of this call stalling.

    A card with no worker process resident answers `cleared` with a
    null `info` rather than a 503: the worker is on-demand, so its
    absence means there was nothing loaded to clear."""
    manager = request.app.state.job_manager
    try:
        return manager.clear_memory(device=device)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except WorkerBusy as e:
        raise HTTPException(
            status_code=409,
            detail=f"{e} - clearing memory out from under it would corrupt "
            "the run. Wait for it to finish.",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Worker unavailable: {e}")


@router.get("/api/health", response_model=HealthInfo, response_model_exclude_unset=True)
def health(request: Request):
    state = request.app.state
    manager = request.app.state.job_manager
    import socket

    from ... import __version__, get_device, get_device_type
    from ...acceleration_warnings import available_attention_backends

    workers = manager.workers()
    device = get_device_type(get_device())
    return {
        "status": "ok",
        "version": __version__,
        # on-demand subprocess: false on an idle server that hasn't run
        # a job yet is normal (a memory clear keeps a live worker, #1240), not a fault -
        # it means no model process is currently resident, not that the
        # server is unhealthy (#206). Any worker of the pool (#462)
        "worker_alive": any(worker["alive"] for worker in workers),
        # The longest-running job, as before the pool; `workers` names each
        "current_job": manager.running_job_id(),
        "workers": workers,
        "queued": manager.queued_count(),
        # which machine answered - the thing a remote client cannot
        # otherwise tell apart from a stale tunnel pointed at nothing
        "hostname": socket.gethostname(),
        "device": device,
        "mcp": bool(state.mcp_mounted),
        "attention_backends": available_attention_backends(device),
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
        # What the history and the log take on disk (#1457). Read-only, and
        # never a failure of this route: an unreadable figure is null
        "storage": _storage(request.app.state),
    }


def _storage(state):
    history = getattr(getattr(state, "job_manager", None), "history", None)
    job_history = None
    if history is not None:
        try:
            job_history = history.storage_stats()
        except Exception:
            logger.debug("Could not read job-history storage", exc_info=True)
    try:
        logs = log_storage_stats()
    except Exception:
        logger.debug("Could not read log storage", exc_info=True)
        logs = None
    return {"job_history": job_history, "logs": logs}


# --- The preferences store over HTTP: GET, PUT and DELETE /api/preferences.
#
# The store is `dw/preferences.py`'s: the shipped ladders, with this server's
# overlay (the settings directory's preferences.json) replacing whole keys.
# These routes are the overlay's only writer. A PUT is refused with 422 when
# its ladder is the wrong shape, names a workflow the catalog does not hold,
# or gives a target arguments its workflow refuses; whether a target fits
# this host is not a refusal here - that is validate's, which names every skip.
# `If-Match` is opt-in, as on the note routes, and a stale one is a 412.
#
# Resolutions answered here use the host's device and smallest card, without
# the gated-access and download probe admission adds, so a read makes no Hub
# calls. A job already queued keeps the target admission pinned into it.

# A stale If-Match: 412 Precondition Failed (the note routes predate this and
# answer 409)
STALE = 412

# The inline workflow targets are opened from: no steps of its own, only a
# place in the workspace for resolve_sub_workflow_reference to search from
_OPENER = {
    "id": "preferences",
    "steps": [
        {
            "name": "noop",
            "pipeline": {"configuration": {}},
        }
    ],
}


def _key_bytes(store, key):
    ladders = store.get("preferences") or {}
    if key not in ladders:
        return b"null"
    return json.dumps(
        {"source": store["sources"].get(key), "ladder": ladders[key]},
        sort_keys=True,
    ).encode("utf-8")


def _key_etag(store, key):
    return content_etag_of(_key_bytes(store, key))


def _tier_etag(store):
    return content_etag_of(store["default_tier"].encode("utf-8"))


def _host(request):
    host = preferences.default_host()
    manager = getattr(request.app.state, "job_manager", None)
    smallest = getattr(manager, "smallest_card_gb", None)
    if callable(smallest):
        try:
            card = smallest()
        except Exception:  # noqa: BLE001 - the host's own figure stands
            card = None
        if card:
            host.smallest_card_gb = card
    return host


def _opener(ws: Workspace):
    """`open_target(name)` the way a workflow in `ws` opens a preference
    target, opening each name once."""
    workflow = workflow_from_definition(
        json.loads(json.dumps(_OPENER)), ws.outputs, None, ws.workflows
    )
    return workflow._open_preference_target


def _entry(request, ws, store, key, resolve):
    entry = {
        "source": store["sources"][key],
        "etag": _key_etag(store, key),
        "ladder": store["preferences"][key],
    }
    if resolve:
        entry["resolutions"] = preferences.resolutions(
            key, store, _host(request), _opener(ws)
        )
    return entry


def _unknown(store, key):
    known = ", ".join(sorted(store.get("preferences") or {})) or "<none>"
    return HTTPException(
        status_code=404,
        detail=f"Unknown preference '{key}'; known keys: {known}",
    )


def _key_expectation(request, key):
    if request.headers.get("if-match") is None:
        return None
    label = f"Preference '{key}'"
    return lambda store: check_if_match(
        request, _key_bytes(store, key), label, status_code=STALE
    )


def _checked_key(key):
    try:
        return preferences.check_key(key)
    except preferences.PreferenceError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get(
    "/api/preferences",
    response_model=PreferencesRead,
    response_model_exclude_unset=True,
)
def get_preferences(
    request: Request,
    response: Response,
    key: Optional[str] = Query(None, description="One key only"),
    resolve: bool = Query(
        False, description="Add this host's pick per tier to each key"
    ),
    ws: Workspace = Depends(selected_workspace),
):
    """The merged ladders - shipped, with this server's overlay replacing
    whole keys - each with its `source` and etag, and `default_tier`."""
    store = preferences.load_preferences()
    if key is not None:
        _checked_key(key)
        if key not in store["preferences"]:
            raise _unknown(store, key)
        response.headers["ETag"] = _key_etag(store, key)
    keys = [key] if key is not None else sorted(store["preferences"])
    return {
        "default_tier": store["default_tier"],
        "etag": _tier_etag(store),
        "preferences": {k: _entry(request, ws, store, k, resolve) for k in keys},
    }


@router.put(
    "/api/preferences/{key}",
    response_model=PreferenceSet,
    response_model_exclude_unset=True,
)
@writes(WORKFLOWS, shared=True)
def put_preference(
    request: Request,
    key: str,
    ladder: dict[str, Any] = Body(
        description="Tier -> an ordered list of targets: a workflow name, or "
        "{workflow, arguments?, devices?}"
    ),
    ws: Workspace = Depends(selected_workspace),
):
    """Replace a key's ladder in this server's overlay (a new key adds one).

    Refused with 422 when the ladder is the wrong shape, names a workflow
    not in the catalog, or gives a target arguments it refuses; with 412
    when `If-Match` no longer names the key as it stands. The answer carries
    this host's pick per tier, so a ladder nothing on this host fits shows
    validate's refusal now."""
    _checked_key(key)
    try:
        preferences.check_ladder(key, ladder, _opener(ws))
    except preferences.PreferenceError as e:
        raise HTTPException(status_code=422, detail=str(e))
    expect = _key_expectation(request, key)

    def change(merged, document):
        if expect:
            expect(merged)
        document["preferences"][key] = ladder

    try:
        store = preferences.update_overlay(change)
    except OSError:
        raise internal_error(f"Could not write preference '{key}'")
    return {"key": key, **_entry(request, ws, store, key, True)}


@router.delete(
    "/api/preferences/{key}",
    response_model=PreferenceReset,
    response_model_exclude_unset=True,
)
@writes(WORKFLOWS, shared=True)
def delete_preference(
    request: Request,
    key: str,
    ws: Workspace = Depends(selected_workspace),
):
    """Reset a key to what ships: drop this server's ladder for it. A key
    only the server had is gone after; one it never overrode is unchanged."""
    _checked_key(key)
    expect = _key_expectation(request, key)

    def change(merged, document):
        if key not in merged["preferences"]:
            raise _unknown(merged, key)
        if expect:
            expect(merged)
        document["preferences"].pop(key, None)

    try:
        store = preferences.update_overlay(change)
    except OSError:
        raise internal_error(f"Could not reset preference '{key}'")
    if key not in store["preferences"]:
        return {"key": key, "removed": True}
    return {"key": key, "removed": False, **_entry(request, ws, store, key, True)}


@router.put("/api/preferences", response_model=DefaultTierSet)
@writes(WORKFLOWS, shared=True)
def put_default_tier(request: Request, body: DefaultTierWrite = Body()):
    """Set the tier a `preference:` reference without `tier` resolves at.
    `If-Match` names the etag GET /api/preferences answers beside it."""
    tier = body.default_tier
    expecting = request.headers.get("if-match") is not None

    def change(merged, document):
        if expecting:
            check_if_match(
                request,
                merged["default_tier"].encode("utf-8"),
                "The default tier",
                status_code=STALE,
            )
        document["default_tier"] = tier

    try:
        store = preferences.update_overlay(change)
    except OSError:
        raise internal_error("Could not write the default tier")
    return {"default_tier": store["default_tier"], "etag": _tier_etag(store)}
