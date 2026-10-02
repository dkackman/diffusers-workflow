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

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STRICT = os.environ.get("DW_STRICT_RESPONSES") == "1"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid" if STRICT else "allow")


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

    run_count: int | None = None
    gpu_available: bool | None = None
    gpu_device_name: str | None = None
    gpu_memory_allocated_mb: int | float | None = None
    gpu_memory_reserved_mb: int | float | None = None
    gpu_memory_free_mb: int | float | None = None
    gpu_memory_total_mb: int | float | None = None


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
