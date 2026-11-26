"""The system routes' response models: GET /api/health and GET /api/server
(the server's address, runtime, directories and storage).

`dw/server/api_models.py` re-exports every name here; import from it, not
from this module.
"""

from pydantic import Field

from .api_models_base import ApiModel, sometimes


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
