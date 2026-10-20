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
from .api_models_base import STRICT, ApiModel, send_rejected_responses, sometimes
from .api_models_library import (
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
)

# The library's models live in api_models_library, and the base in
# api_models_base; every route and test imports them from here
__all__ = [
    "STRICT",
    "ApiModel",
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
    oom_score_adj: int = sometimes(
        "The worker's kernel oom_score_adj, read from /proc at call time (1000 for "
        "the latest dispatch's worker in a pool, else the value it was spawned "
        "with); absent where unreadable."
    )


class HealthInfo(ApiModel):
    status: str
    version: str
    worker_alive: bool = Field(
        description="Whether any worker is alive. Workers are on-demand: false on an "
        "idle server that has not run a job yet, or after a memory clear, is normal - "
        "no model process is resident, not a fault."
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


# ------------------------------------------------------------------- jobs

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
OutputKind = Literal["image", "video", "audio", "text"]
# An asset library entry may also be a .cube colour lookup table
AssetKind = Literal["image", "video", "audio", "text", "lut"]


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
        "Why dispatch is holding this queued job though a card is free - "
        "'waiting for host RAM: needs ~N GB, ~M GB free'; only while queued and "
        "held. It starts when a running job frees the RAM."
    )
    note: str = sometimes("The note it was submitted with; also its run's note.")
    rerun_of: str = sometimes(
        "The job this one reran; absent for a job submitted fresh."
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
        "`inherited_from` among offload, quantization and frame count; empty "
        "when nothing does. Otherwise: the cost drivers with no rule to scale "
        "by that moved off the default the figure was measured at, with "
        "`low_confidence` true."
    )
    scaled: str = sometimes(
        "With basis `derived` (or `other_device`, or a tempered `observed`): "
        "what the curated figure was scaled along - steps, canvas or frames "
        "moved off the variable defaults it was measured at (#866). With basis "
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
        "With basis `unknown` and no `withheld`: why a workflow of task steps "
        "alone has no price - it has no pipeline to inherit one from, and its "
        "length follows its input media (#948)."
    )
    priced_for: str = sometimes(
        "The card the run would be dispatched to, whose history the figure is - "
        "as `cuda:1 NVIDIA GeForce RTX 3090`."
    )


class PlanHostMemory(ApiModel):
    projected_mb: int = Field(
        description="The host RAM, in MB, this run is projected to hold at its peak."
    )
    basis: Literal["observed", "fit"] = Field(
        description="`observed`: the largest job peak of this machine's successful "
        "runs with the same pipelines, quantization and offload; `fit`: projected "
        "from this workflow's own runs for the requested list length."
    )
    runs: int = Field(description="How many runs the figure is read from.")


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


# --------------------------------------------------------- gallery, assets

# Assets have no packaged (builtin) library
AssetOrigin = Literal["workspace", "common", "examples"]


class GalleryFile(ApiModel):
    name: str
    folder: str
    subfolder: str = Field(
        description="What followed the run id in the file's path - the `final` / "
        "`intermediate` a step's `result.subfolder` chose, `''` for none."
    )
    run_id: str = Field(
        description="The run that wrote the file, `''` under the flat layout."
    )
    version: int | None = Field(
        description="That run's ordinal among the workflow's runs - what the grid "
        "shows as `v4`. Never renumbered, so a deleted sibling leaves a gap. Null "
        "when there is no run."
    )
    url: str
    absolute_url: str = sometimes()
    kind: OutputKind
    size: int
    mtime: int | float
    label: str
    duration_seconds: int | float | None = sometimes(
        "An audio or video file's length, with `media=true`."
    )
    role: Literal["final", "intermediate"] | None = sometimes(
        "`final` for a deliverable, `intermediate` for a file a later step or "
        "run consumed; null under the flat layout or when no manifest lists the file."
    )
    note: str = sometimes("The effective note: the file's own, else its run's.")
    starred: bool = sometimes("True when the file is starred; absent otherwise.")


class GalleryList(ApiModel):
    files: list[GalleryFile] = Field(description="Newest first.")
    total: int
    offset: int
    limit: int
    folders: list[str]
    subfolders: list[str]
    workspace: str


class OutputDeleted(ApiModel):
    name: str
    deleted: bool
    run_swept: str | None = Field(
        description="The run directory the delete emptied and removed, if any."
    )


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
        "and mode; null for another file."
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


class AssetLibraryRoot(ApiModel):
    root: str
    origin: AssetOrigin
    writable: bool


class AssetFile(ApiModel):
    name: str
    reference: str
    folder: str
    kind: AssetKind
    size: int
    mtime: int | float
    origin: AssetOrigin = Field(
        description="Which library it came from: this workspace's own, the `common` "
        "one every workspace shares, or a read-only examples tree - why a delete "
        "can answer 403."
    )
    writable: bool
    url: str
    absolute_url: str = sometimes()
    used_by: list[str] = sometimes(
        "This workspace's own workflows that name it in an `asset:` reference, "
        "when any do"
    )
    note: str = sometimes("The asset's note, when it has one.")


class ShadowedAsset(ApiModel):
    name: str
    reference: str
    folder: str
    kind: AssetKind
    size: int
    mtime: int | float
    origin: AssetOrigin
    writable: bool
    shadowed_by: AssetOrigin


class AssetList(ApiModel):
    workspace: str
    libraries: list[AssetLibraryRoot]
    assets: list[AssetFile]
    folders: list[str]
    shadowed: list[ShadowedAsset]
    total: int = sometimes("Assets matching the prefix, before `limit` cut them")
    shadowed_total: int = sometimes("Shadowed names matching, before `limit`")


class Uploaded(ApiModel):
    reference: str
    workspace: str
    url: str
    absolute_url: str = sometimes()
    shared: bool


class Kept(ApiModel):
    reference: str
    name: str
    workspace: str
    linked: bool
    shared: bool


class StarSet(ApiModel):
    name: str
    starred: bool


class NoteSet(ApiModel):
    target: str
    note: str | None = Field(description="The effective note after the write.")
    note_source: Literal["file", "run"] | None = Field(
        description="Whose note `note` is; null for a workspace or no note."
    )


class AssetDeleted(ApiModel):
    name: str
    workspace: str
    reference: str
    deleted: bool
    origin: AssetOrigin


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
