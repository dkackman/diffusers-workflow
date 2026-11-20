"""Response models for the library routes - workspaces, workflows, prompts,
enhancer presets, LoRAs, the asset library and the requests ledger. Part of
the UI's response contract that `dw/server/api_models.py` describes; it
re-exports every name here.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .api_models_base import ApiModel, JobStatus, sometimes
from ..ledger import MAX_TITLE_LENGTH, MAX_WHY_LENGTH
from ..lora_catalog import STATUS_ORDER
from .catalog_shape import SHAPES, TRAITS

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
    note: str = sometimes("The workspace's note, when it has one; listings only.")


class WorkspaceList(ApiModel):
    workspace_root: str | None
    default: str
    workspaces: list[WorkspaceInfo]


class WorkspaceDeleted(ApiModel):
    name: str
    deleted: bool
    contents: dict[str, DiskUsage] = Field(description="What it held, per folder.")


class CopiedItems(ApiModel):
    runs: list[str] = Field(description="`<workflow>/<run id>` of each run copied.")
    assets: list[str] = Field(
        description="Library-relative path of each asset file copied."
    )


class WorkspaceCopied(ApiModel):
    """What a copy between workspaces did (`POST /api/workspaces/{name}/copy`)."""

    source: str
    destination: str
    copied: CopiedItems
    already_present: list[str] = Field(
        description="Runs the destination already held (`<workflow>/<run id>`) "
        "and assets it held with identical bytes (library-relative path) - "
        "skipped."
    )
    renumbered: dict[str, dict[str, int]] = Field(
        description="Copied runs whose `vN` changed, keyed `<workflow>/<run id>`: "
        "`{from, to}`."
    )
    warnings: dict[str, Any] = Field(
        description="`unresolved_output_refs`, `absolute_paths` and `no_jobs`, "
        "each only when it applies."
    )
    bytes_linked: int
    bytes_copied: int


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


class BuiltinWorkflows(ApiModel):
    workflows: list[str] = Field(
        description="The packaged fragments a sub-workflow step names as "
        "`builtin:<name>.json`, each as what follows the prefix."
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


# The LoRA catalog listing (GET /api/loras). A row is a catalog entry
# (dw/lora_catalog_schema.json) as stored, plus where it came from. Every
# field the schema allows is declared, so the strict suite fails on an entry
# key the model does not know rather than the route dropping it; numbers are
# `int | float` so a stored `1` goes out as `1`, not `1.0`.


class LoraScale(ApiModel):
    default: int | float
    range: list[int | float] = sometimes("`[low, high]`, when the entry states one.")


class LoraEvidence(ApiModel):
    note: str
    job: str = sometimes()
    issue: int = sometimes()


class LoraRow(ApiModel):
    name: str = Field(description="The entry's catalog name: its path under loras/.")
    origin: Origin = Field(description="Which library root the entry came from.")
    writable: bool
    model_name: str = Field(description="The LoRA's Hub repo (owner/name).")
    base_models: list[str] = Field(description="The base repos it was made for.")
    description: str
    use_when: str
    status: Literal[tuple(STATUS_ORDER)]
    weight_name: str = sometimes(
        "The .safetensors file in the repo, when it has several."
    )
    revision: str = sometimes("The commit or branch the entry pins.")
    workflow: str | list[str] | None = sometimes(
        "The base's workflow partition(s) the LoRA fits; null or absent for any."
    )
    trigger: str = sometimes("Words the prompt needs for the LoRA to act.")
    scale: LoraScale = sometimes("The scale the entry states; absent on some entries.")
    stacks_with: list[str] = sometimes()
    evidence: list[LoraEvidence] = sometimes()
    license: str = sometimes()
    tags: list[str] = sometimes()


class ResolvedBase(ApiModel):
    repo: str
    workflow: str | None


class LoraList(ApiModel):
    libraries: list[LibraryRoot]
    resolved: list[ResolvedBase] = sometimes(
        "The bases `model` resolved to; only when `model` was given."
    )
    loras: list[LoraRow] = Field(
        description="Matching entries, rejected ones included, sorted by name."
    )


# An asset library entry may also be a .cube colour lookup table
AssetKind = Literal["image", "video", "audio", "text", "lut"]


# Assets have no packaged (builtin) library
AssetOrigin = Literal["workspace", "common", "examples"]


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


class LedgerAccepted(ApiModel):
    job: str
    run_id: str | None


class LedgerComment(ApiModel):
    text: str
    at: str


class LedgerAttempt(ApiModel):
    job: str
    status: JobStatus
    note: str = sometimes("The job's note.")
    minutes: int | float | None = Field(
        description="Run time; null until the job has started."
    )


class LedgerResult(ApiModel):
    job: str
    run_id: str | None


class LedgerItem(ApiModel):
    key: str
    title: str
    depends_on: list[str]
    estimate_minutes: int | float | None
    workflow: str | None
    arguments: dict[str, Any]
    dropped: bool
    accepted: LedgerAccepted | None
    comment: LedgerComment | None = Field(
        description="The user's; an agent never writes it."
    )
    status: Literal["dropped", "done", "running", "queued", "failed", "planned"] = (
        Field(
            description="Derived from the item's linked jobs, first that holds: "
            "dropped, done (a result exists), running, queued, failed (every "
            "linked job failed or was cancelled), planned."
        )
    )
    attempts: list["LedgerAttempt"] = Field(
        description="Every linked job, oldest first."
    )
    result: "LedgerResult | None" = Field(
        description="The accepted job's run, else the latest succeeded job's "
        "run whose directory still exists; null when neither."
    )
    accepted_missing: bool = sometimes(
        "True when an accepted job's run directory is gone; `result` then "
        "falls back to the latest succeeded attempt."
    )
    spent_minutes: int | float = Field(
        description="Minutes the item's linked jobs ran, summed."
    )


class LedgerHistoryEntry(ApiModel):
    revision: int
    at: str
    by: Literal["agent", "user"]
    why: str
    goal: str | None = sometimes("The goal, when this revision changed it.")


class LedgerRequest(ApiModel):
    id: str
    title: str
    state: Literal["open", "done", "abandoned"]
    revision: int
    prompt: str | None = Field(description="The ask that started it; write-once.")
    goal: str
    items: list[LedgerItem]
    history: list[LedgerHistoryEntry]
    spent_minutes: int | float = Field(description="Minutes every linked job ran.")
    remaining_minutes: int | float = Field(
        description="Estimates of the items not done and not dropped."
    )


class LedgerSummary(ApiModel):
    id: str
    title: str
    state: Literal["open", "done", "abandoned"]
    revision: int
    items: int
    status_counts: dict[str, int]
    spent_minutes: int | float
    remaining_minutes: int | float
    updated_at: str | None


class LedgerList(ApiModel):
    workspace: str
    requests: list[LedgerSummary]
    unreadable: str = sometimes(
        "Why requests.json could not be read, when it is there but is not a "
        "ledger; `requests` is then empty and every write is 409 "
        "`ledger_unreadable` until the file is fixed or removed."
    )


class LedgerDeleted(ApiModel):
    id: str
    workspace: str
    deleted: bool
    revision: int = Field(description="The revision that was deleted.")
    linked_jobs: int = Field(
        description="Jobs linked to it; each keeps its row, link and files."
    )


class RequestWrite(BaseModel):
    """One request's whole plan."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(description=f"One line, at most {MAX_TITLE_LENGTH}")
    goal: str = Field(description="What done looks like")
    items: list[dict[str, Any]] = Field(
        description="The ordered plan; each item is key, title, depends_on, "
        "estimate_minutes, workflow, arguments, accepted, comment"
    )
    revision: int = Field(description="The revision read, or 0 to create")
    why: str = Field(description=f"One line, at most {MAX_WHY_LENGTH}")
    prompt: str | None = Field(default=None, description="Create only")
    state: Literal["open", "done", "abandoned"] | None = None
    purge: list[str] | None = Field(
        default=None,
        description="Keys of stored items this plan leaves out, removed for "
        "good instead of kept as dropped; their jobs keep their link and files",
    )
    by: Literal["agent", "user"] = Field(
        default="user", description="Who wrote this revision; the UI is the user"
    )


class AssetDeleted(ApiModel):
    name: str
    workspace: str
    reference: str
    deleted: bool
    origin: AssetOrigin
