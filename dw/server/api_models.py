"""Response models for the routes the web UI reads - the server half of the
UI's response contract. The UI's types are generated from the OpenAPI
document these produce (scripts/dump_openapi.py -> ui/src/lib/generated/),
so a change here that the UI depends on fails its type check.

Lenient at runtime: a key a model has not declared is still sent, so a
handler that grows a field never turns into a 500 for dw_mcp or a script.
Strict under DW_STRICT_RESPONSES=1 - the test suite, the e2e fixture server
and the OpenAPI dump - so an undeclared key fails a test, and the generated
types carry no index signature that would let a removed field type-check.

Routes declare `response_model_exclude_unset=True`: a key the handler did
not emit stays absent rather than arriving as null.
"""

import logging
import os

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .admission import AcknowledgedCost
from .catalog_shape import SHAPES, TRAITS

STRICT = os.environ.get("DW_STRICT_RESPONSES") == "1"

logger = logging.getLogger(__name__)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid" if STRICT else "allow")


def send_rejected_responses(app) -> None:
    """Make runtime leniency cover a declared field of the wrong type too,
    not just an undeclared key: the response is logged and sent as the
    handler built it, with the route's own status. Some routes have acted
    by then - POST /api/jobs has queued the job - and a 500 would invite a
    retry that does it twice. Installed by create_app outside strict mode."""
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import ResponseValidationError
    from fastapi.responses import JSONResponse

    async def send_as_built(request, exc: ResponseValidationError):
        route = request.scope.get("route")
        logger.error(
            "Response for %s %s does not match its model: %s",
            request.method,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            jsonable_encoder(exc.body),
            status_code=getattr(route, "status_code", None) or 200,
        )

    app.add_exception_handler(ResponseValidationError, send_as_built)


def sometimes(description: str | None = None) -> Any:
    """A key the handler emits only in some states, and never as null when
    it does: typed as its value, defaulting to an unvalidated None that
    `response_model_exclude_unset` keeps out of the payload. The generated
    type is then `key?: T`, not `key?: T | null`."""
    return Field(default=None, description=description)


# ----------------------------------------------------------------- system


class HealthInfo(ApiModel):
    status: str
    version: str
    worker_alive: bool = Field(
        description="The worker is on-demand: false on an idle server that has not "
        "run a job yet, or after a memory clear, is normal - no model process is "
        "resident, not a fault."
    )
    current_job: str | None
    queued: int
    hostname: str = Field(description="Which machine answered.")
    device: str
    mcp: bool


class ServerAddress(ApiModel):
    address: str
    family: str
    interface: str | None


class McpMount(ApiModel):
    mounted: bool
    path: str


class ServerRuntime(ApiModel):
    python_version: str
    torch_version: str | None
    cuda_version: str | None
    driver_version: str | None
    packages: dict[str, str | None] = Field(
        description="Installed version of each package the engine leans on; null "
        "when it is not installed."
    )


class ServerDirectories(ApiModel):
    workspace: str | None = Field(
        description="The workspace the folders below are folders of, when the "
        "server resolved one; an individually overridden folder still reports its "
        "own path."
    )
    workflows: str
    assets: str | None
    outputs: str
    prompts: str | None


class ServerInfo(ApiModel):
    hostname: str
    version: str
    device: str
    bind_host: str
    port: int
    wildcard_bind: bool
    auth_required: bool
    trust_workflows: bool = Field(
        description="With this off, a workflow file is untrusted input: no arbitrary "
        "imports, no remote code, no location outside the workspace's roots."
    )
    mcp: McpMount
    addresses: list[ServerAddress]
    runtime: ServerRuntime
    directories: ServerDirectories


class MemoryDetail(ApiModel):
    """The worker's own memory report. Open even in strict mode: its keys
    vary by backend, and the UI reads only the ones declared here."""

    model_config = ConfigDict(extra="allow")

    run_count: int = sometimes()
    gpu_available: bool = sometimes()
    gpu_device_name: str | None = None
    gpu_memory_allocated_mb: int | float = sometimes()
    gpu_memory_reserved_mb: int | float = sometimes()
    gpu_memory_free_mb: int | float = sometimes()
    gpu_memory_total_mb: int | float = sometimes()


class MemoryInfo(ApiModel):
    live: bool = Field(description="Measured now, rather than the last reading.")
    info: MemoryDetail | None
    stale: bool = Field(
        description="`info` is an earlier reading: compare only `live` readings."
    )
    reason: str | None = Field(description="Why the reading is not a live one.")
    age_seconds: int | float | None


class MemoryCleared(ApiModel):
    cleared: bool
    info: MemoryDetail | None


class ModelRevision(ApiModel):
    commit_hash: str
    size_on_disk: int
    refs: list[str]
    last_modified: int | float | None


class ModelRepo(ApiModel):
    repo_id: str
    repo_type: str
    size_on_disk: int
    nb_files: int
    last_accessed: int | float | None
    last_modified: int | float | None
    revisions: list[ModelRevision]


class ModelCache(ApiModel):
    cache_dir: str
    size_on_disk: int
    repos: list[ModelRepo]
    warnings: list[str]
    disk_free: int | None
    disk_total: int | None


class ModelDownload(ApiModel):
    id: str
    repo_id: str
    status: Literal["downloading", "completed", "cancelled", "failed"]
    downloaded: int
    total: int | None
    error: str | None
    started_at: int | float
    finished_at: int | float | None


class ModelDownloads(ApiModel):
    downloads: list[ModelDownload] = Field(description="Newest first.")


class ModelDeleted(ApiModel):
    repo_id: str
    deleted: bool
    freed: int = Field(description="Bytes freed.")


class DiffusersInstall(ApiModel):
    version: str | None
    commit: str | None = Field(description="The git commit, for a git install.")


class DiffusersStatus(ApiModel):
    status: Literal["idle", "running", "succeeded", "failed"]
    error: str | None
    log: str | None
    started_at: int | float | None
    finished_at: int | float | None
    requested_commit: str | None
    revert: bool
    before: DiffusersInstall | None = Field(
        description="What was installed when the update started."
    )
    version: str | None
    commit: str | None


# ------------------------------------------------------------------- jobs

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
OutputKind = Literal["image", "video", "audio", "text"]


class JobSummary(ApiModel):
    id: str
    workflow: str | None = Field(
        description="Null only on a row recorded before the workflow was stored."
    )
    workflow_name: str | None = Field(description="The catalog name it was run by.")
    status: JobStatus
    created_at: int | float | None
    started_at: int | float | None
    finished_at: int | float | None
    workspace: str = Field(
        description="The workspace this job ran in - 'default' for the default one."
    )
    run_id: str | None = Field(
        description="The run this job opened - null until it opens one, and for a "
        "job recorded before runs were tracked."
    )
    run_version: int | None = Field(
        description="That run's ordinal among the workflow's runs - the `v4` the "
        "gallery shows for its files. Null until the run opens, and for older rows."
    )
    acknowledged: Literal["none", "boolean", "bound"] = Field(
        description="Which form of cost acknowledgement queued the job: none (the "
        "web UI and any caller that sent nothing), a bare boolean, or one bound to "
        "the plan a validate answered with."
    )
    historical: bool = sometimes("Read from job history rather than a live job.")
    queue_position: int = sometimes("Index in the waiting queue; only while queued.")


class ManifestEntry(ApiModel):
    """One step's saved files. Open even in strict mode: the worker adds
    per-step detail (`selected`, `shots`, ...) the UI does not read."""

    model_config = ConfigDict(extra="allow")

    step: str
    files: list[str]
    subfolder: str = sometimes(
        "The in-run subfolder the step's `result.subfolder` chose - "
        "`final`, `intermediate`, any relative path - `''` when it chose none. "
        "Absent only on a job recorded before the field existed."
    )
    reused: bool = sometimes(
        "The step was served from the step cache: these files are an "
        "earlier run's, republished, and nothing was generated for them this time."
    )
    parent_step: str = sometimes("The composing step a rolled-up entry came from.")


class JobProgress(ApiModel):
    step: str | None
    parent_step: str | None = Field(
        description="The step of the queued workflow the one above is running "
        "inside, for a composed run; null when they are the same thing."
    )
    step_index: int | None
    total_steps: int | None
    phase: str | None
    phase_detail: Any
    seconds_in_phase: int | float | None
    seconds_since_event: int | float
    denoise_step: int | None = Field(
        description="Null until the denoise loop starts; a number that stops moving "
        "is a stuck one."
    )
    denoise_total_steps: int | None


class JobDetail(JobSummary):
    arguments: dict[str, Any]
    warnings: list[str]
    manifest: list[ManifestEntry] | None = Field(
        description="Null on a history row recorded before manifests, or by a run "
        "that wrote none."
    )
    error: str | None
    traceback: str | None
    event_count: int
    run_dir: str | None
    acknowledged_cost: AcknowledgedCost | None = Field(
        description="The plan the caller bound its acknowledgement to, when it did."
    )
    progress: JobProgress | None = Field(
        default=None,
        description="Where a running (or failed) job had got to; live jobs only.",
    )
    spec: dict[str, Any] = sometimes("The submitted spec; history rows only.")
    output_kinds: dict[str, OutputKind | None] = sometimes(
        "Each output file's kind; null for a kind the gallery does not "
        "show. On GET /api/jobs/{id} only."
    )


class JobList(ApiModel):
    jobs: list[JobSummary] = Field(description="Oldest first.")
    total: int = Field(description="How many matched before `limit` cut the list.")


class RunDeleted(ApiModel):
    job_id: str
    run_dir: str
    deleted: bool
    run_swept: str = Field(description="The deleted run's directory name.")


class JobWorkflow(ApiModel):
    id: str
    definition: dict[str, Any]
    realized: bool = Field(
        description="Every mutable input is pinned - the copy the run itself wrote."
    )
    seed_variable: str | None = Field(
        description="The variable a new-seed rerun would draw into, null when the "
        "workflow has none - the cue for whether to offer that at all."
    )


class ExportedFile(ApiModel):
    path: str
    bytes: int


class JobExport(ApiModel):
    job_id: str
    directory: str
    files: list[ExportedFile]
    total_bytes: int
    missing: list[str]
    zip_url: str
    absolute_zip_url: str = sometimes()
    auth_required: bool
    workflow: Any = Field(description="workflow.json as exported; null if unreadable.")
    manifest: Any = Field(description="manifest.json as exported; null if unreadable.")
    job: Any = Field(description="job.json as exported; null if unreadable.")


class JobMoved(ApiModel):
    id: str
    queue: list[str]


class JobCancelled(ApiModel):
    id: str
    status: JobStatus


# ------------------------------------------------------------- validation


class ValidationFinding(ApiModel):
    """One violation with its JSON path. Open even in strict mode: a
    finding carries its own extra keys after these two."""

    model_config = ConfigDict(extra="allow")

    path: str | None
    message: str


class ElidedStep(ApiModel):
    step: str | None
    reason: str
    overridden_by: str = sometimes("The supplied variable that made it unread.")


class RequiredDownload(ApiModel):
    repo: str | None
    url: str = sometimes("A from_single_file URL, which has no repo.")
    gb: int | float | None
    gated: bool | Literal["auto", "manual"] | None = Field(
        description="The hub's own `gated` field: false, or how access is granted."
    )
    access_blocked: bool | None


class PlanEstimate(ApiModel):
    minutes: int | float | None
    basis: Literal[
        "per_entry", "catalog", "derived", "other_device", "unknown", "observed"
    ]
    device: str
    measured_on: str | None
    partial: bool
    unpriced: list[str] = Field(
        description="What contributed nothing to `minutes` when `partial` is true - "
        "the workflow's own id when its own steps went unpriced, else the path of "
        "each composed child with no cost block. Empty when `partial` is false."
    )
    runs: int | None
    cached_minutes: int | float | None
    tempered: bool = sometimes()
    observed_minutes: int | float = sometimes()
    curated_minutes: int | float = sometimes()
    low_confidence: bool = sometimes()


class Plan(ApiModel):
    fingerprint: str
    steps: int
    elided_steps: list[ElidedStep] = Field(
        description="The steps that will not run because nothing reads their result "
        "and they save no file - already excluded from `steps`."
    )
    list_entries: dict[str, int]
    cached_steps: int | None = Field(
        description="How many steps the worker's step cache would serve; null when "
        "the worker was busy or did not answer."
    )
    downloads_required: list[RequiredDownload]
    estimate: PlanEstimate
    workspace: str
    output_dir: str | None


class ValidationResult(ApiModel):
    valid: bool
    error: str | None
    errors: list[ValidationFinding] = Field(
        description="Every schema violation with its JSON path; empty when valid."
    )
    warnings: list[str]
    checked_arguments: list[str] = sometimes()
    plan: Plan | None = Field(
        default=None,
        description="What the run will execute for the definition validated - on a "
        "valid answer; null when the server could not build it, absent from an "
        "invalid answer.",
    )


# ---------------------------------------------------------------- library

Origin = Literal["workspace", "common", "examples", "builtin"]
Shape = Literal[SHAPES]
Trait = Literal[TRAITS]


class LibraryRoot(ApiModel):
    root: str
    origin: Origin
    writable: bool


class ShadowedEntry(ApiModel):
    name: str
    origin: Origin
    shadowed_by: Origin


class DiskUsage(ApiModel):
    files: int
    bytes: int


class WorkspaceInfo(ApiModel):
    name: str
    default: bool
    root: str | None
    workflows: str
    assets: str | None
    outputs: str
    prompts: str | None
    common_assets: str | None
    usage: DiskUsage = sometimes("Roughly how much disk it holds; listings only.")


class WorkspaceList(ApiModel):
    workspace_root: str | None
    default: str
    workspaces: list[WorkspaceInfo]


class WorkspaceDeleted(ApiModel):
    name: str
    deleted: bool
    contents: dict[str, DiskUsage] = Field(description="What it held, per folder.")


class WorkflowCost(ApiModel):
    """A measured run, as the workflow's own `cost` block declares it."""

    model_config = ConfigDict(extra="allow")

    device: str
    name: str = sometimes()
    vram_gb: int | float
    minutes: int | float


class WorkflowCard(ApiModel):
    """One workflow's entry in the full listing (the agent's
    `view=compact` answer is a different, smaller shape outside this
    model)."""

    kinds: list[str]
    steps: int
    variables: int
    variable_names: list[str]
    description: str
    configures: str = sometimes(
        "For a model config: the template it is a tuned instance of; '' for a "
        "template. Absent when the file could not be read."
    )
    configures_missing: str = sometimes("A `configures` that names no workflow.")
    prompt_refs: list[str]
    shape: Shape = Field(description="What the workflow makes, derived by the server.")
    traits: list[Trait] = Field(
        description="Sorted, independent facts about how the output is made or what "
        "it needs."
    )
    summary: str = Field(
        description="The description's first sentence, clipped - what a card shows."
    )
    lists: dict[str, Any]
    constraints: dict[str, Any]
    cost_drivers: dict[str, Any]
    cost: list[WorkflowCost] | None = Field(
        description="Measured runs, one per device the maintainer measured on. Null "
        "means unknown - never derived."
    )
    observed: dict[str, Any] = sometimes("This box's own history for it.")
    origin: Origin
    writable: bool = Field(
        description="False for a read-only source: offer save-a-copy, not delete."
    )


class WorkflowList(ApiModel):
    workspace: str
    libraries: list[LibraryRoot] = Field(
        description="The search path in order; the writable workspace root is where "
        "a save lands, whatever library a workflow was read from."
    )
    workflows: list[str]
    details: dict[str, WorkflowCard]
    shadowed: list[ShadowedEntry]
    cost_basis: str


class WorkflowSaved(ApiModel):
    name: str
    workspace: str
    origin: Origin
    warnings: list[str]
    shape: Shape
    traits: list[Trait]
    summary: str


class WorkflowDeleted(ApiModel):
    name: str
    workspace: str
    origin: Origin
    deleted: bool


class PromptCard(ApiModel):
    description: str
    intended_model: str
    tags: list[str]
    text: str = sometimes("Absent when the listing was asked for without text.")
    text_chars: int = sometimes("The text's length, in place of `text`.")
    origin: Origin = Field(
        description="Which library the prompt came from, and whether a save can "
        "reach it."
    )
    writable: bool


class PromptList(ApiModel):
    libraries: list[LibraryRoot]
    prompts: list[str]
    details: dict[str, PromptCard]
    shadowed: list[ShadowedEntry]


class PromptSaved(ApiModel):
    name: str


class Deleted(ApiModel):
    name: str
    deleted: bool


class EnhancerPreset(ApiModel):
    key: str
    label: str
    default_model: str
    models: list[str]
    intended_models: list[str]
    placeholder: str


class EnhancerPresets(ApiModel):
    presets: list[EnhancerPreset]
