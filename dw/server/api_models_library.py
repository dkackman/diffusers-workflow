"""Response models for the library routes - workspaces, workflows, prompts
and enhancer presets. Part of the UI's response contract that
`dw/server/api_models.py` describes; it re-exports every name here.
"""

from typing import Any, Literal

from pydantic import ConfigDict, Field

from .api_models_base import ApiModel, sometimes
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
