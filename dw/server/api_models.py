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

import os

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .admission import AcknowledgedCost

STRICT = os.environ.get("DW_STRICT_RESPONSES") == "1"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid" if STRICT else "allow")


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
    gated: bool | None
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
