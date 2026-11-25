"""Response models for the library routes: workspaces, workflows, prompts, enhancer
presets, LoRAs, the asset library, the requests ledger, and one gallery file's marks
and request links. Part of the UI response contract `api_models.py` re-exports."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .api_models_base import ApiModel, JobStatus, OutputKind, sometimes
from ..ledger import MAX_TITLE_LENGTH, MAX_WHY_LENGTH
from ..lora_catalog import STATUS_ORDER
from .catalog_shape import SHAPES, TRAITS

Origin = Literal["workspace", "common", "examples", "builtin"]
Shape = Literal[SHAPES]
Trait = Literal[TRAITS]
InputKind = Literal["image", "video", "audio", "file"]


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


class Dated(ApiModel):
    mtime: float = sometimes("Epoch seconds its file last changed; times=true only.")
    created: float = sometimes("Birth time, if the filesystem has it; times=true only.")


class WorkspaceInfo(Dated):
    mtime: float = sometimes(
        "The newest mtime of its folders and their direct entries; times=true only."
    )
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
        description="Runs the destination already held (`<workflow>/<run id>`) and "
        "assets it held with identical bytes (library-relative path) - skipped."
    )
    renumbered: dict[str, dict[str, int]] = Field(
        description="Runs whose `vN` changed, `<workflow>/<run id>` to `{from, to}`."
    )
    warnings: dict[str, Any] = Field(
        description="`unresolved_output_refs`, `absolute_paths`, `no_jobs`: where each applies."
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


class WorkflowCard(Dated):
    """One workflow in the full listing; `view=compact` answers a smaller shape."""

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
    inputs: dict[str, InputKind] = sometimes(
        "Each variable feeding a media argument, and its kind; inputs=true only."
    )
    shape: Shape = Field(description="What the workflow makes, derived by the server.")
    traits: list[Trait] = Field(
        description="Sorted facts about how the output is made or what it needs."
    )
    summary: str = Field(
        description="The description's first sentence, clipped - what a card shows."
    )
    lists: dict[str, Any]
    constraints: dict[str, Any]
    cost_drivers: dict[str, Any]
    cost: list[WorkflowCost] | None = Field(
        description="Measured runs, one per device the maintainer measured on; null "
        "means unknown, never derived."
    )
    observed: dict[str, Any] = sometimes("This box's own history for it.")
    origin: Origin
    writable: bool = Field(
        description="False for a read-only source: offer save-a-copy, not delete."
    )


class BuiltinWorkflows(ApiModel):
    workflows: list[str] = Field(
        description="Packaged fragments, each as a step names it after `builtin:`."
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


class PromptCard(Dated):
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


# The LoRA catalog listing (GET /api/loras): a row is a stored catalog entry
# (dw/lora_catalog_schema.json) plus where it came from. Every field the schema allows
# is declared, so the strict suite fails on a key the model lacks rather than the route
# dropping it; numbers are `int | float`, so a stored `1` goes out as `1`, not `1.0`.


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
    duration_seconds: int | float | None = sometimes("A/V length with `media=true`.")


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


class LedgerNoteRef(ApiModel):
    """What a note is about: one item, planned or rendered, and maybe one of
    its linked jobs and one of that job's files."""

    item: str = Field(description="The item key.")
    job: str = sometimes("A job linked to the item.")
    output: str = sometimes(
        "One file of the job's manifest, relative to the workspace's outputs."
    )
    kind: OutputKind = sometimes("The file's kind, set by the server.")
    attempt: int | None = sometimes(
        "On a read, when the ref names a job: the job's 1-based place among "
        "the item's attempts; null when the job is no longer linked to it."
    )
    gone: bool = sometimes(
        "On a read: true when the item is dropped or out of the plan, or the "
        "job is no longer linked or its run is gone."
    )


class LedgerNote(ApiModel):
    """One note the user wrote for the agent; only request-notes routes write one."""

    id: str = Field(description="Unique within the request; never reused.")
    text: str
    created: str = Field(description="When the note was first saved.")
    at: str = Field(description="When its text last changed.")
    refs: list[LedgerNoteRef] = sometimes(
        "The items, takes or files the note is about."
    )


class LedgerOutput(ApiModel):
    """The one file that stands for a job: what a list shows as its proof."""

    name: str = Field(
        description="The file as the job's manifest and the gallery name it, "
        "relative to the workspace's outputs: `/api/gallery/{name}/thumbnail`."
    )
    kind: OutputKind = Field(description="A thumbnail exists for image and video.")


class LedgerAttempt(ApiModel):
    job: str
    status: JobStatus
    note: str = sometimes("The job's note.")
    reason: str = sometimes(
        "Only with reasons=true: why a failed or cancelled job ended, "
        "the first line of its error."
    )
    minutes: int | float | None = Field(
        description="Run time; null until the job has started."
    )
    output: LedgerOutput | None = Field(
        description="The job's first final image or video, else its first "
        "sound or text; null when it wrote none or its run is gone."
    )


class LedgerResult(ApiModel):
    job: str
    run_id: str | None
    output: LedgerOutput | None


class LedgerReview(ApiModel):
    until: str = Field(description="The deadline of the hold, an ISO time in UTC.")
    state: Literal["holding", "redirected", "due"] = Field(
        description="redirected when a note was left after the hold was set, "
        "else due once the deadline has passed, else holding."
    )
    seconds_left: int = Field(
        description="Whole seconds to the deadline at read time, 0 once passed."
    )


class LedgerAcceptedBasis(ApiModel):
    at: str = Field(description="When the acceptance was recorded, an ISO time.")
    prompts: dict[str, str] = sometimes(
        "The sha256 of each referenced prompt's text when the acceptance was "
        "recorded: it holds until that text changes again."
    )


class LedgerItem(ApiModel):
    key: str
    title: str
    depends_on: list[str]
    estimate_minutes: int | float | None
    workflow: str | None
    arguments: dict[str, Any]
    dropped: bool
    accepted: LedgerAccepted | None
    notes: list[LedgerNote] = Field(description="The user's notes on the item.")
    status: Literal[
        "dropped", "done", "revised", "running", "queued", "failed", "planned"
    ] = Field(
        description="Derived from the item's linked jobs, first that holds: dropped, "
        "done (a result exists) - or revised when its workflow or arguments changed "
        "after every succeeded take and no take was accepted since - running, "
        "queued, failed (every linked job failed or was cancelled), planned."
    )
    revised_because: list[str] = sometimes(
        "What moved on a revised item: `workflow`, `arguments`, `prompt:<name>` (a "
        "referenced prompt's text changed). Absent unless the item is revised."
    )
    changed_at: str | None = Field(
        description="When the item was created or its workflow or arguments last "
        "changed, stamped by the server; null on an item stored before the stamp."
    )
    accepted_basis: "LedgerAcceptedBasis | None" = Field(
        description="When a take was last accepted while the item read "
        "revised, stamped by the server; the item reads done until its plan "
        "changes again. Null when none was."
    )
    ready: bool = Field(
        description="A job could be queued for it now: planned or revised, "
        "with a stored workflow and every dependency done, in an open request."
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
    review_until: str | None = Field(
        description="When the user's review of the item ends, an ISO time in "
        "UTC; null when it is not held."
    )
    review_set_at: str | None = Field(
        description="When the current deadline was set, stamped by the server; "
        "null when it is not held."
    )
    review: LedgerReview = sometimes(
        "The hold as read now; absent unless the item is planned or revised, its request "
        "open and a deadline set."
    )


class LedgerHistoryEntry(ApiModel):
    revision: int
    at: str
    by: Literal["agent", "user"]
    why: str
    goal: str | None = sometimes("The goal, when this revision changed it.")
    state: Literal["open", "done", "abandoned"] = sometimes(
        "The state, when this revision set or changed it: who closed or "
        "reopened the request, when and why."
    )


class LedgerRequest(ApiModel):
    id: str
    title: str
    state: Literal["open", "done", "abandoned"]
    revision: int
    prompt: str | None = Field(description="The ask that started it; write-once.")
    goal: str
    notes: list[LedgerNote] = Field(
        description="The user's notes on the whole request."
    )
    next_note: int = Field(description="The number of the next note id.")
    agent_wrote_at: str | None = Field(
        description="When the agent last wrote the plan; null when it never has."
    )
    items: list[LedgerItem]
    history: list[LedgerHistoryEntry]
    spent_minutes: int | float = Field(description="Minutes every linked job ran.")
    remaining_minutes: int | float = Field(
        description="Estimates of the items not done and not dropped."
    )
    warnings: list[str] | None = Field(
        default=None,
        description="On a write only: what it did not do, e.g. an `accepted` "
        "sent with a plan edit, which is not recorded.",
    )


class LedgerSummary(ApiModel):
    id: str
    title: str
    state: Literal["open", "done", "abandoned"]
    revision: int
    items: int
    status_counts: dict[str, int]
    ready: int = Field(description="Items a job could be queued for now.")
    cover: list[LedgerOutput] = Field(
        description="Up to three outputs of the latest results, newest first."
    )
    spent_minutes: int | float
    remaining_minutes: int | float
    updated_at: str | None
    needs: dict[str, int] = sometimes(
        "Only with needs=true: how many items want the user, by reason - held, ready, "
        "failed (can run again), revised, choose, accepted_missing - and notes_waiting."
    )


class LedgerList(ApiModel):
    workspace: str
    requests: list[LedgerSummary]
    unreadable: str = sometimes(
        "Why requests.json could not be read, when it is there but is not a "
        "ledger; `requests` is then empty and every write is 409 "
        "`ledger_unreadable` until the file is fixed or removed."
    )


class OpenRequestSummary(LedgerSummary):
    workspace: str = Field(description="The workspace the request is in.")


class OpenRequestList(ApiModel):
    requests: list[OpenRequestSummary] = Field(
        description="Every workspace's open requests, newest change first."
    )
    unreadable: dict[str, str] = sometimes(
        "Workspace name -> why its requests.json could not be read; its "
        "requests are left out."
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
        "estimate_minutes, workflow, arguments, accepted, review_until (an ISO "
        "time with a UTC offset, 60 seconds to 24 hours ahead, holding the item "
        "for the user's review; null clears it); an item's `notes` "
        "are ignored (the request-notes routes write them)"
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


class NoteRefWrite(BaseModel):
    """One reference of a new note. `kind` is the server's to set."""

    model_config = ConfigDict(extra="forbid")

    item: str | None = Field(
        default=None,
        description="The item key; left out on an item's note, that item. "
        "Alone, it names a planned shot that may have no job yet",
    )
    job: str | None = Field(default=None, description="A job linked to that item")
    output: str | None = Field(
        default=None,
        description="One file of that finished job's manifest, as the gallery "
        "names it (relative to the workspace's outputs); left out, the note "
        "is about the take as a whole",
    )


class NoteAdd(BaseModel):
    """One new note for the agent, from the user."""

    model_config = ConfigDict(extra="forbid")

    request: str = Field(description="The request id")
    item: str | None = Field(
        default=None, description="The item key; left out, the request's own notes"
    )
    text: str = Field(description="The note, at most 4,000 characters")
    refs: list[NoteRefWrite] | None = Field(
        default=None,
        description="The items, takes or files the note is about, at most 16; "
        "an item's note references only that item",
    )


class NoteEdit(BaseModel):
    """A note's new text. Its references never change."""

    model_config = ConfigDict(extra="forbid")

    request: str = Field(description="The request id")
    note: str = Field(description="The note id")
    text: str = Field(description="The note, at most 4,000 characters")


class NoteWritten(ApiModel):
    note: LedgerNote
    revision: int = Field(description="The request's revision after the write.")


class NotesChannelState(ApiModel):
    listening: bool = Field(
        description="True while a notes channel polled this workspace's "
        "request list in the last 30 seconds."
    )
    last_seen: str | None = Field(
        description="ISO UTC time of the last such poll, null if none since "
        "the server started."
    )


class ChangeCounts(ApiModel):
    """One counter per kind of data a page lists; each only grows."""

    requests: int = Field(description="Request ledger writes: requests and notes.")
    prompts: int = Field(description="Prompt library saves and deletes.")
    workflows: int = Field(description="Workflow saves, patches and deletes.")
    assets: int = Field(description="Asset uploads, keeps, deletes and notes.")
    gallery: int = Field(
        description="Outputs landing from a finished job, and star, note or "
        "delete of an output."
    )
    jobs: int = Field(
        description="Job queue, start, finish, cancel and move, and a run deleted."
    )
    models: int = Field(
        description="Model downloads, cancels and deletes, cache clears, and "
        "a diffusers update. Server-wide."
    )
    loras: int = Field(description="LoRA catalog saves and deletes. Server-wide.")


class WorkspaceChanges(ApiModel):
    epoch: str = Field(
        description="Random id of this workspace's counters: new on a server "
        "restart and when the workspace is created again. A different epoch "
        "means everything changed."
    )
    workspaces: int = Field(
        description="Server-wide counter of workspace create, delete and note writes."
    )
    kinds: ChangeCounts = Field(description="This workspace's counters, by kind.")


class NoteDeleted(ApiModel):
    revision: int = Field(description="The request's revision after the write.")


class AssetDeleted(ApiModel):
    name: str
    workspace: str
    reference: str
    deleted: bool
    origin: AssetOrigin


# ------------------------------------- a gallery file's marks and links


class GalleryLinkedJob(ApiModel):
    """A job linked to a request item that published the file - wrote it or
    republished it from the step cache."""

    id: str
    request: str
    item: str | None
    attempt: int | None = Field(
        description="The job's 1-based place among the item's attempts; null "
        "when the request no longer lists it."
    )


class StarSet(ApiModel):
    name: str
    starred: bool


class NoteSet(ApiModel):
    target: str
    note: str | None = Field(description="The effective note after the write.")
    note_source: Literal["file", "run"] | None = Field(
        description="Whose note `note` is; null for a workspace or no note."
    )


class NoteRead(ApiModel):
    target: str
    note: str | None = Field(
        description="The target's own note, never the run's a file falls back to."
    )
    etag: str = Field(
        description="What `If-Match` on PUT /api/notes names to replace only this note."
    )


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
    duration_seconds: int | float | None = sometimes("A/V length with `media=true`.")
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
    prompts: dict[str, str] = sometimes("Run id -> prompt, with `prompts=true`.")


class OutputDeleted(ApiModel):
    name: str
    deleted: bool
    run_swept: str | None = Field(
        description="The run directory the delete emptied and removed, if any."
    )
