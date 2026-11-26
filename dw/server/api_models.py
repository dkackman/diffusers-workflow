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

from typing import Any, Literal

from pydantic import ConfigDict, Field

from .admission import AcknowledgedCost
from .api_models_base import (
    STRICT,
    ApiModel,
    JobStatus,
    OutputKind,
    send_rejected_responses,
    sometimes,
)
from .api_models_library import (
    GalleryFile,
    GalleryList,
    OutputDeleted,
    Origin,
    Shape,
    Trait,
    LibraryRoot,
    ShadowedEntry,
    DiskUsage,
    WorkspaceInfo,
    WorkspaceList,
    WorkspaceDeleted,
    CopiedItems,
    WorkspaceCopied,
    WorkflowCost,
    WorkflowCard,
    BuiltinWorkflows,
    WorkflowList,
    WorkflowSaved,
    WorkflowDeleted,
    PromptCard,
    PromptList,
    PromptSaved,
    Deleted,
    EnhancerPreset,
    EnhancerPresets,
    LoraScale,
    LoraEvidence,
    LoraRow,
    ResolvedBase,
    LoraList,
    AssetOrigin,
    AssetKind,
    AssetLibraryRoot,
    AssetFile,
    ShadowedAsset,
    AssetList,
    Uploaded,
    Kept,
    AssetDeleted,
    LedgerAccepted,
    LedgerNoteRef,
    LedgerNote,
    LedgerAttempt,
    LedgerResult,
    LedgerItem,
    LedgerHistoryEntry,
    LedgerRequest,
    LedgerSummary,
    LedgerList,
    NoteWritten,
    NoteDeleted,
    NotesChannelState,
    GalleryLinkedJob,
    StarSet,
    NoteSet,
    NoteRead,
)

__all__ = [  # re-exported from api_models_library/_base: import them from here
    "GalleryFile",
    "GalleryList",
    "OutputDeleted",
    "STRICT",
    "ApiModel",
    "JobStatus",
    "send_rejected_responses",
    "sometimes",
    "Origin",
    "Shape",
    "Trait",
    "LibraryRoot",
    "ShadowedEntry",
    "DiskUsage",
    "WorkspaceInfo",
    "WorkspaceList",
    "WorkspaceDeleted",
    "CopiedItems",
    "WorkspaceCopied",
    "WorkflowCost",
    "WorkflowCard",
    "BuiltinWorkflows",
    "WorkflowList",
    "WorkflowSaved",
    "WorkflowDeleted",
    "PromptCard",
    "PromptList",
    "PromptSaved",
    "Deleted",
    "EnhancerPreset",
    "EnhancerPresets",
    "LoraScale",
    "LoraEvidence",
    "LoraRow",
    "ResolvedBase",
    "LoraList",
    "AssetOrigin",
    "AssetKind",
    "AssetLibraryRoot",
    "AssetFile",
    "ShadowedAsset",
    "AssetList",
    "Uploaded",
    "Kept",
    "AssetDeleted",
    "LedgerAccepted",
    "LedgerNoteRef",
    "LedgerNote",
    "LedgerAttempt",
    "LedgerResult",
    "LedgerItem",
    "LedgerHistoryEntry",
    "LedgerRequest",
    "LedgerSummary",
    "LedgerList",
    "NoteWritten",
    "NoteDeleted",
    "NotesChannelState",
    "GalleryLinkedJob",
    "StarSet",
    "NoteSet",
    "NoteRead",
]


# ----------------------------------------------------------------- system


class WorkerInfo(ApiModel):
    device: str = Field(description="The card this worker is pinned to.")
    name: str | None = Field(description="The card as a job record's device names it.")
    vram_gb: float | None = Field(
        description="What the card holds, in the GB a catalog cost entry is written in."
    )
    current_job: str | None
    alive: bool
    host_memory_rss_mb: float = sometimes(
        "The worker process's resident host memory; absent while it is not running."
    )
    pid: int = sometimes("The worker process's pid; absent while it is not running.")
    workspace: str = sometimes("The workspace of `current_job`; absent while idle.")
    oom_score_adj: int = sometimes(
        "The kernel oom_score_adj read from /proc at call time: 1000 for a pool's "
        "latest-dispatch worker, else its spawn value; absent where unreadable."
    )


class HealthInfo(ApiModel):
    status: str
    version: str
    worker_alive: bool = Field(
        description="Whether any worker is alive. Workers are on-demand: false on an "
        "idle server that has not run a job yet is normal - no model process "
        "is resident, not a fault. A memory clear keeps a live worker alive "
        "(same pid); it frees models, it does not restart the process."
    )
    current_job: str | None = Field(
        description="The longest-running job; `workers` names every running one."
    )
    workers: list[WorkerInfo] = Field(
        description="One entry per card the server runs a worker on (--devices), "
        "each running its own job: one job per GPU."
    )
    queued: int
    hostname: str = Field(description="Which machine answered.")
    device: str
    mcp: bool
    attention_backends: list[str] = Field(
        default_factory=list,
        description="Attention backends this server's device can run, for the UI's "
        "dropdown; 'sparse' only where comfy-kitchen's compiled kernels loaded",
    )


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


class JobHistoryStorage(ApiModel):
    path: str = Field(description="The job-history SQLite database.")
    size_bytes: int | None = Field(
        description="The database file plus its WAL/SHM sidecars."
    )
    rows: int | None = Field(
        description="Every row in the history, open jobs included: a real "
        "count, not the job listing's capped `total`."
    )
    oldest: float | None = Field(
        description="The oldest row's created_at (epoch seconds)."
    )
    newest: float | None = Field(
        description="The newest row's created_at (epoch seconds)."
    )


class LogStorage(ApiModel):
    path: str = Field(description="The active log file.")
    size_bytes: int | None = Field(description="The active log file's size.")
    backups: int | None = Field(description="Rotated backups on disk (at most 7).")
    total_bytes: int | None = Field(description="The active log plus its backups.")


class ServerStorage(ApiModel):
    """What the server's own records take on disk. Read-only: a figure that
    can't be read is null, never an error on this route."""

    job_history: JobHistoryStorage | None = Field(
        description="Null when the history database is not open."
    )
    logs: LogStorage | None = Field(description="Null when no log file is configured.")


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
    storage: ServerStorage


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
    device: str = sometimes("The card this reading is of, as `cuda:1`.")


class CardMemory(MemoryInfo):
    device: str = Field(description="The card this reading is of, as `cuda:1`.")


class MemoryStatus(MemoryInfo):
    workers: list[CardMemory] = sometimes(
        "One reading per card; the top level repeats the first card's."
    )


class CardCleared(ApiModel):
    device: str
    cleared: bool = Field(description="False for a card left alone: it runs a job.")
    info: MemoryDetail | None = sometimes()
    reason: str = sometimes("Why the card was not cleared: `job_running`.")
    job: str = sometimes("The job the card is running.")


class MemoryCleared(ApiModel):
    cleared: bool
    info: MemoryDetail | None
    device: str = sometimes("The card cleared, when the request named one.")
    workers: list[CardCleared] = sometimes(
        "Each card's outcome; a card running a job is left alone."
    )


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
    last_accessed: int | float | None = Field(
        description="The newest access time (atime) of the repo's files - a "
        "fallback for last_job_use: copies and backups bump it, and a "
        "noatime or relatime disk may never move it."
    )
    last_modified: int | float | None
    revisions: list[ModelRevision]
    last_job_use: int | float | None = Field(
        description="Epoch seconds a dw job that named this repo last "
        "started (dw/model_use.py); null when no dw job has used it since "
        "tracking began - then last_accessed is the only date there is.",
    )


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
    cancel_requested: bool = Field(
        default=False,
        description="A cancel was requested and the download has not yet "
        "stopped: it is stopping, not still pulling.",
    )


class ModelDownloads(ApiModel):
    downloads: list[ModelDownload] = Field(description="Newest first.")


class ModelDeleted(ApiModel):
    repo_id: str
    deleted: bool
    freed: int = Field(description="Bytes freed.")


class DiskCacheEntry(ApiModel):
    id: str = Field(description="The entry's 64-hex key: its directory name.")
    bytes: int
    files: int
    unreadable: bool = Field(
        description="True when its entry.json is missing or unreadable; the "
        "fields below are then null, and it can still be deleted."
    )
    model: str | None
    subfolder: str | None
    component: str | None
    created: float | None = Field(description="Epoch seconds.")
    last_hit: float | None = Field(
        description="Epoch seconds of the last load that read it; null if none has."
    )


class DiskCacheInfo(ApiModel):
    name: str
    root: str
    enabled: bool
    bytes: int
    files: int
    last_used: float | None = Field(
        description="The newest file's mtime (epoch seconds); null when empty."
    )
    clearable: bool
    not_clearable_reason: str | None = None
    cost_of_clearing: str
    disk_free: int | None
    disk_total: int | None
    low_disk: bool | None = Field(
        default=None,
        description="For a cache with a free-space floor (quantized): the "
        "volume is under it, so new entries are not saved.",
    )
    entries: list[DiskCacheEntry] | None = Field(
        default=None, description="For a cache with entries (quantized)."
    )


class DiskCacheList(ApiModel):
    caches: list[DiskCacheInfo]


class DiskCacheCleared(ApiModel):
    name: str
    cleared: bool
    freed: int = Field(description="Bytes freed.")
    files: int


class DiskCacheEntryDeleted(ApiModel):
    name: str
    entry: str
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


class DwUpdateState(ApiModel):
    version: str = Field(description="The installed dw version.")
    latest: str | None = Field(
        description="The newest release on PyPI; null when it could not be read."
    )
    managed: bool = Field(
        description="True when a service manager restarts this server, which is "
        "what lets it update itself."
    )
    command: str = Field(
        description="The one-line install command for this OS: how an unmanaged "
        "copy updates."
    )
    status: Literal["idle", "running", "succeeded", "failed"]
    error: str | None
    log: str | None = Field(description="The last lines of the update's output.")


# ------------------------------------------------------------------- jobs


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
    device: str | None = Field(
        default=None,
        description="The card the job ran on, as ordinal then name - "
        "'cuda:1 NVIDIA GeForce RTX 3090'. Null until it starts, and for older rows.",
    )
    acknowledged: Literal["none", "boolean", "bound"] = Field(
        description="Which form of cost acknowledgement queued the job: none (the "
        "web UI and any caller that sent nothing), a bare boolean, or one bound to "
        "the plan a validate answered with."
    )
    historical: bool = sometimes("Read from job history rather than a live job.")
    queue_position: int = sometimes("Index in the waiting queue; only while queued.")
    waiting_for: str = sometimes(
        "Why dispatch is holding this queued job - 'waiting for host RAM: "
        "needs ~N GB, ~M GB free' (a card is free), or 'queued behind N running "
        "job(s) on every card (...)' when every card is busy; only while queued. "
        "Each hold also emits one queue_wait event (reason host_ram or gpu_busy)."
    )
    note: str = sometimes("The note it was submitted with; also its run's note.")
    rerun_of: str = sometimes(
        "The job this one reran; absent for a job submitted fresh."
    )
    request: str = sometimes(
        "The requests-ledger request this job serves; absent when unlinked."
    )
    item: str = sometimes(
        "The item of `request` this job serves; absent when unlinked."
    )
    overrides: dict = sometimes(
        "The variables a rerun changed from the job it reran; absent when none were."
    )
    cancel_requested: bool = sometimes(
        "True when a cancel was asked of this running job and it has not stopped "
        "yet - a model load or a task has no checkpoint; only while running."
    )
    reason: str = sometimes(
        "Why the job ended: the first line of its `error`, clipped to 160 "
        "characters with a trailing '…'. Only on a failed or cancelled job that "
        "has an error - a plain user cancel has none. The full error, traceback "
        "and all, is on the job's detail."
    )
    restart_batch: str = sometimes(
        "The `server_instance` of the server whose restart cancelled this job - "
        "the key a restart batch is grouped by. Only on a restart tombstone (a "
        "cancelled job whose error starts 'server restart:'); any row of the "
        "batch is a valid id for POST /api/jobs/{id}/rerun-restart-batch."
    )


class StepTiming(ApiModel):
    """Where one pipeline step's time went, on the worker's monotonic clock
    (dw/pipeline_processors/timing.py). Every `_s` is seconds to 0.1; a field
    whose mark never happened is null, never guessed. Open, like the entry."""

    model_config = ConfigDict(extra="allow")

    load_s: float | None = Field(
        description="Loading or fetching the step's pipeline, cached or not."
    )
    load_cached: bool | None = Field(
        description="The pipeline came from the worker's cache; false if any "
        "part was loaded fresh, null when the step loaded nothing."
    )
    encode_s: float | None = Field(
        description="Call start to the first denoiser forward: prompt "
        "encoding and preparation."
    )
    denoise_s: float | None = Field(
        description="First denoiser forward to the last pass's end; the whole "
        "call when the pipeline reports no passes."
    )
    passes: int | None = Field(
        description="Denoise passes the pipeline ran, one per scheduler timestep; can differ from num_inference_steps."
    )
    first_pass_s: float | None = Field(
        description="The first pass, which carries any compile warm-up."
    )
    pass_s: float | None = Field(
        description="Median of passes 2..n, wall time at step boundaries; "
        "null with fewer than two passes."
    )
    compiled: bool | None = Field(
        description="The denoiser runs through torch.compile; null when unknown."
    )
    decode_s: float | None = Field(
        description="Last pass's end to the call's return: VAE decode and "
        "post-processing."
    )
    save_s: float | None = Field(description="Writing the step's files.")
    segments: int = Field(
        description="Pipeline calls summed into this timing: chain segments "
        "or iterations; 1 for a plain step."
    )
    cache: dict[str, Any] = sometimes("Skip counts, on a step that set a `cache`.")


class ShotMeasure(ApiModel):
    """One measure a chain's check computed for a segment's kept attempt
    (#1474), against the ceiling in force."""

    value: float = Field(
        description="The measure; for `words`, the heard words' similarity to the line."
    )
    ceiling: float | None = Field(
        description="The bound the check held it to - the workflow's own when it "
        "set one - or null where the check sets none (measured, never failed)."
    )
    bound: Literal["max", "min"] = Field(
        description="`max`: the value must stay at or under the ceiling; `min`: "
        "at or over it. `texture_ratio` is bounded both ways, so `failed`, not "
        "value against ceiling, is the verdict."
    )
    warn_only: bool = Field(
        description="A failure is warned and recorded but never re-rolled."
    )
    failed: bool = Field(description="The check failed this measure.")


class ShotRecord(ApiModel):
    """One shot of a joined video: where an input landed in it, and for a
    chain's segment what its check recorded (`dw/shots.py` lists every key).
    Open even in strict mode: a join may add detail of its own."""

    model_config = ConfigDict(extra="allow")

    name: str
    start_frame: int
    num_frames: int
    start_sample: int | None = Field(
        default=None, description="Null when the video has no track the join built."
    )
    num_samples: int | None = Field(
        default=None, description="Null when the video has no track the join built."
    )
    planned_segments: int = sometimes(
        "On a chain's shot: the segment count its plan asked for."
    )
    segment_file: str = sometimes(
        "On a chain's shot under save_segments and keep_segments: its segment's file."
    )
    resumed: bool = sometimes(
        "On a chain's shot: the segment was copied from an earlier run (resume_from)."
    )
    attempts: int = sometimes("On a chain's shot: renders its check tried.")
    check_failed: list[str] = sometimes(
        "On a checked chain shot whose every attempt failed: the measures the "
        "kept one failed."
    )
    audio_dbfs: float = sometimes(
        "On a chain's shot with generated audio: its kept audio's level."
    )
    measures: dict[str, ShotMeasure] = sometimes(
        "On a chain's shot with the check on: every measure the kept attempt "
        "computed, by name. Absent with the check off, on a resumed segment, "
        "and on a job recorded before the field existed."
    )


class ManifestEntry(ApiModel):
    """One step's saved files. Open even in strict mode: the worker adds
    per-step detail (`selected`, ...) beyond the keys typed here."""

    model_config = ConfigDict(extra="allow")

    step: str
    files: list[str]
    shots: list[ShotRecord] | None = sometimes(
        "Where each input landed in the step's joined video (dw/shots.py); "
        "null or absent on a step that joined none."
    )
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
    role: Literal["final", "intermediate"] = sometimes(
        "The step's role in its run, as the run's manifest.json records it. "
        "Absent while the run is going, or for a run from before roles."
    )
    timing: StepTiming = sometimes(
        "Where the step's time went. Only on a pipeline step that ran: a "
        "reused step, a task or a composed step has none."
    )


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
    host_memory_job_peak_committed_mb: int | float | None = Field(
        default=None,
        description="The job's peak committed host memory in MB (anonymous + "
        "shared + device-mapped pages), the figure the host-RAM gate projects "
        "from; null until a reading lands.",
    )
    preferences: dict[str, Any] = sometimes(
        "What each 'preference:<key>' sub-workflow step resolved to, keyed "
        "'<path>@<tier>': {target, arguments, skipped: [{workflow, reason}], "
        "source}. A rerun replays it; only on a job that resolved one."
    )
    output_kinds: dict[str, OutputKind | None] = sometimes(
        "Each output file's kind; null for a kind the gallery does not "
        "show. On GET /api/jobs/{id} only."
    )


class JobList(ApiModel):
    jobs: list[JobSummary] = Field(description="Oldest first.")
    total: int = Field(description="How many matched before `limit` cut the list.")
    prompts: dict[str, str] = sometimes(
        "Job id -> the `prompt` argument it was submitted with, with `prompts=true`."
    )


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
    note: str = sometimes("Present when `realized` is false: why, and what was folded.")


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


class RestartBatchQueued(ApiModel):
    id: str = Field(description="The new job's id.")
    rerun_of: str = Field(description="The tombstone it reruns.")
    workflow: str | None
    workspace: str


class RestartBatchCandidate(ApiModel):
    id: str = Field(description="The tombstone's id.")
    workflow: str | None
    workflow_name: str | None = Field(None, description="The catalog name it ran by.")
    workspace: str
    estimate: "PlanEstimate | None" = Field(
        description="The plan's estimate of the rerun; null when none was built."
    )


class RestartBatchSkipped(ApiModel):
    id: str
    reason: str = Field(
        description="Why it was not queued: `already rerun as <id>`, a deleted "
        "workspace, or what admission refused."
    )


class RestartBatchSize(ApiModel):
    server_instance: str = Field(description="The dead server whose batch this is.")
    size: int = Field(description="Every job that restart cancelled.")
    pending: int = Field(
        description="Those not yet rerun and not in a deleted workspace."
    )


class RestartBatchRerun(ApiModel):
    server_instance: str = Field(description="The dead server whose batch this is.")
    queued: list[RestartBatchQueued] = Field(
        description="What was queued, oldest original first; empty in a preview."
    )
    would_queue: list[RestartBatchCandidate] = sometimes(
        "A preview only: what `acknowledged_cost: true` would queue, oldest first."
    )
    skipped: list[RestartBatchSkipped]


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
    filename: str = sometimes(
        "A single weight file within `repo`, for a task that fetches one file "
        "rather than the whole repo."
    )
    revision: str | None = sometimes("The commit the file is read at; null means main.")
    components: list[str] = sometimes(
        "A modular step's own `model_name` when its `load_components` moves "
        "some components to another repo: the subfolders still fetched from "
        "`repo`, counting only those not cached."
    )
    gb: int | float | None
    gated: bool | Literal["auto", "manual"] | None = Field(
        description="The hub's own `gated` field: false, or how access is granted."
    )
    access_blocked: bool | None
    not_found: bool = sometimes(
        "True when the hub answered that the repo does not exist (or is "
        "private to this token); validation then fails on it."
    )


class PlanEstimate(ApiModel):
    minutes: int | float | None
    basis: Literal[
        "per_entry",
        "catalog",
        "derived",
        "other_device",
        "unknown",
        "observed",
        "inherited",
    ]
    device: str
    measured_on: str | None
    partial: bool
    unpriced: list[str] = Field(
        description="What contributed nothing to `minutes` when `partial` is true - "
        "the workflow's own id when its own steps went unpriced, else the path of "
        "each composed child with no cost block or observed history. When `minutes` is null "
        "it still names the composed children that left it unpriced (`partial` is then "
        "false); empty when nothing composed went unpriced."
    )
    runs: int | None
    cached_minutes: int | float | None
    tempered: bool = sometimes()
    observed_minutes: int | float = sometimes()
    curated_minutes: int | float = sometimes()
    low_confidence: bool = sometimes()
    inherited_from: str = sometimes(
        "With basis `inherited`: the catalog template whose runs of the same "
        "pipeline (component_type, model_name, workflow) priced this workflow."
    )
    differs: list[str] = sometimes(
        "With basis `inherited`: what this workflow sets differently from "
        "`inherited_from` among offload, quantization, frame count, attention "
        "backend, cache, LoRA and reference count; empty "
        "when nothing does. Priced from declared task rates (`scaled` starts "
        "`task rates:`): the steps whose input length was not known, so were "
        "priced at the longest input resolved or at the fixed seconds alone. "
        "Otherwise: the cost drivers with no rule to scale "
        "by that moved off the default the figure was measured at, with "
        "`low_confidence` true."
    )
    scaled: str = sometimes(
        "With basis `derived` (or `other_device`, or a tempered `observed`): "
        "what the curated figure was scaled along - steps, canvas or frames "
        "moved off the variable defaults it was measured at (#866) - or, for a "
        "task-only workflow priced from its commands' declared rates, "
        "`task rates: ` and each step's fixed seconds plus seconds per input "
        "second over its input media's length (#1619). With basis "
        "`inherited`: why the template's minutes were re-priced "
        "linearly (more pipeline steps or frames than the template run measured) "
        "- an approximation, with `low_confidence` true."
    )
    withheld: str = sometimes(
        "With basis `unknown` and `inherited_from` set: why the template's runs "
        "could not be re-priced for this workflow (its size cannot be compared "
        "to the template's)."
    )
    unpriced_reason: str = sometimes(
        "With basis `unknown` and no `withheld`: why a task-only workflow has no "
        "price - `no declared rate: ` and the commands with no declared task rate "
        "to price it from (no pipeline to inherit one from, no cost block and no "
        "run history of its own)."
    )
    per_segment: bool = sometimes(
        "True when an `observed` figure is a matched chain's, read per segment "
        "from this box's history."
    )
    priced_for: str = sometimes(
        "The card the run would be dispatched to, whose history the figure is - "
        "as `cuda:1 NVIDIA GeForce RTX 3090`."
    )


class PlanHostMemory(ApiModel):
    projected_mb: int = Field(
        description="The host RAM, in MB, this run is projected to hold at its peak."
    )
    basis: Literal["observed", "fit", "declared", "inherited"] = Field(
        description="`inherited`: an inline workflow with no runs of its own, read "
        "from the largest peak of this machine's runs of the catalog template that "
        "loads the same pipeline. `observed`: the largest job peak of this machine's successful "
        "runs with the same pipelines, quantization and offload; `fit`: projected "
        "from this workflow's own runs for the requested list length; `declared`: "
        "the workflow's own `host_ram_gb`, larger than anything observed."
    )
    runs: int = Field(description="How many runs the figure is read from.")
    inherited_from: str = sometimes("With basis `inherited`: the template's name.")


class Plan(ApiModel):
    fingerprint: str
    steps: int
    elided_steps: list[ElidedStep] = Field(
        description="The steps that will not run because nothing reads their result "
        "and they save no file - already excluded from `steps`."
    )
    list_entries: dict[str, int]
    cached_steps: int | None = Field(
        description="How many steps the step cache of the card the run would be "
        "dispatched to would serve; null when that worker was busy or did not answer."
    )
    downloads_required: list[RequiredDownload]
    estimate: PlanEstimate
    workspace: str
    host_memory: PlanHostMemory = sometimes(
        "The host RAM this run is projected to hold at its peak, from this "
        "machine's own history (#869); absent when there is no figure."
    )


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


# ---------------------------------------------------------------- gallery


class JobRef(ApiModel):
    model_config = ConfigDict(extra="allow")

    id: str
    status: str


class LevelFinding(ApiModel):
    """A probe threshold crossing (dw/assessment_rules.finding)."""

    rule: str
    severity: str
    at: Any
    value: Any
    threshold: Any
    says: str


class GalleryInput(ApiModel):
    reference: str
    kind: Literal["asset", "output"]


class GalleryMetadata(ApiModel):
    name: str
    source: Literal["output", "asset"]
    metadata: dict[str, Any] | None = Field(
        description="What the file embeds: the workflow and arguments that made it."
    )
    job: JobRef | None
    run_id: str
    version: int | None
    media: dict[str, Any] | None = Field(
        description="The probe of an audio or video file, or an image's kind, width, height "
        "and mode (plus `alpha` coverage when it has transparency data), or an .otioz's timeline (tracks, clips with durations, members); "
        "null for another file."
    )
    findings: list[LevelFinding]
    role: Literal["final", "intermediate"] | None = sometimes(
        "The file's role in its run; null when the run does not say."
    )
    inputs: list[GalleryInput] | None = sometimes(
        "The `asset:` and `output:` references the run read; null for a run "
        "that recorded none."
    )
    note: str = sometimes("The effective note: the file's own, else its run's.")
    run_note: str = sometimes("The note of the run that wrote the file.")
    note_source: Literal["file", "run"] = sometimes("Whose note `note` is.")
    starred: bool = sometimes("True when the file is starred; absent otherwise.")
    linked_jobs: list[GalleryLinkedJob] = Field(
        description="Every job linked to a request whose manifest lists the "
        "file, a step-cache reuse included, newest first; empty for an asset."
    )


# ---------------------------------------------------------- introspection


class PipelineParameter(ApiModel):
    name: str
    required: bool
    default: Any = Field(description="Null when there is none; see `required`.")
    annotation: str | None
    doc_type: str | None = sometimes("The type the docstring names.")
    description: str = sometimes()
    domain: Any = sometimes("The values a task argument may take.")
    range: str = sometimes("The domain's bounds, in words.")
    choices: list[str] = sometimes("The values a string task argument accepts.")


class PipelineDescription(ApiModel):
    name: str
    summary: str
    accepts_kwargs: bool
    parameters: list[PipelineParameter]
    compatibles: list[str] = sometimes("Scheduler classes this one can swap with.")


class PipelineNames(ApiModel):
    pipelines: list[str]


class TaskList(ApiModel):
    commands: list[str]
    image_processors: list[str]
    video_processors: list[str]
    assessment: list[str]


class ClassList(ApiModel):
    kind: str
    classes: list[str]


RestartBatchCandidate.model_rebuild()  # reads PlanEstimate, declared after it
