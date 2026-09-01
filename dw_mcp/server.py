"""Assemble the MCP tool surface over a DwClient.

The only module that imports the MCP SDK. Every tool body is a one-line
call into a handler, so the handlers stay testable without a session and
this file stays a description of the surface rather than logic.
"""

import functools

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, ToolAnnotations

from dw_mcp import authoring, catalog, diagnose, media, models
from dw_mcp.client import DwApiError

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITES = ToolAnnotations(read_only_hint=False, open_world_hint=False)
OVERWRITES = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)
DELETES = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)


def _anticipated(fn):
    """Let a DwApiError's message reach the model.

    A DwApiError is a failure the handlers saw coming and wrote a message
    for. Anything but a ToolError escaping a tool is treated by the SDK as a
    crash: the message is replaced with "Error executing tool <name>" and a
    traceback is logged. Re-raising as ToolError keeps the text.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except DwApiError as e:
            raise ToolError(str(e)) from e

    return wrapper


def build_server(client):
    """An MCP server whose tools all run against `client`."""
    server = MCPServer(
        "diffusers-workflow",
        instructions=(
            "Author, run and diagnose diffusers-workflow jobs against a "
            "running dw.serve. Validate a workflow before running it - "
            "validation is free, a run occupies the GPU for minutes."
        ),
    )

    def tool(fn, annotations):
        server.add_tool(_anticipated(fn), name=fn.__name__, annotations=annotations)

    # ------------------------------------------------------------- catalog

    def list_workflows() -> dict:
        """List the workflows stored on the server, with their descriptions,
        output kinds and variables."""
        return catalog.list_workflows(client)

    def get_workflow(name: str) -> dict:
        """Get one stored workflow's full JSON definition."""
        return catalog.get_workflow(client, name)

    def get_schema() -> dict:
        """Get the JSON schema every workflow definition must satisfy."""
        return catalog.get_schema(client)

    def list_pipelines() -> dict:
        """List every diffusers pipeline class this installation provides."""
        return catalog.list_pipelines(client)

    def get_pipeline_signature(name: str) -> dict:
        """Get a pipeline's real call arguments. Check this before proposing
        pipeline arguments - a plausible-looking argument that the pipeline
        does not accept is the most common workflow bug."""
        return catalog.get_pipeline_signature(client, name)

    def list_classes(kind: str) -> dict:
        """List class names of one kind: pipelines, models, schedulers, or
        quantization."""
        return catalog.list_classes(client, kind)

    def get_class(name: str, target: str = "init") -> dict:
        """Get a class's argument schema. target: init, call, or load."""
        return catalog.get_class(client, name, target=target)

    def list_tasks() -> dict:
        """List every task command a workflow's task step can name."""
        return catalog.list_tasks(client)

    def get_task(command: str) -> dict:
        """Get a task command's argument schema."""
        return catalog.get_task(client, command)

    def list_models() -> dict:
        """List what the Hugging Face model cache holds, largest first."""
        return catalog.list_models(client)

    def get_memory() -> dict:
        """Get the worker's VRAM and RAM statistics. Check this first when a
        job fails with an out-of-memory error."""
        return catalog.get_memory(client)

    def get_health() -> dict:
        """Check that the server is alive."""
        return catalog.get_health(client)

    def list_jobs() -> dict:
        """List queued, running and recent jobs."""
        return catalog.list_jobs(client)

    def list_gallery(limit: int = 50) -> dict:
        """List generated output files, newest first."""
        return catalog.list_gallery(client, limit=limit)

    def get_gallery_metadata(name: str) -> dict:
        """Get the metadata embedded in a generated file: the exact workflow
        and arguments that produced it. Use this to reproduce a bad result."""
        return catalog.get_gallery_metadata(client, name)

    for fn in (
        list_workflows,
        get_workflow,
        get_schema,
        list_pipelines,
        get_pipeline_signature,
        list_classes,
        get_class,
        list_tasks,
        get_task,
        list_models,
        get_memory,
        get_health,
        list_jobs,
        list_gallery,
        get_gallery_metadata,
    ):
        tool(fn, READ_ONLY)

    # --------------------------------------------------------------- media

    def get_output_image(name: str, max_dimension: int = 768) -> ImageContent:
        """Look at a generated image. Use this to judge output quality - it
        is the only way to see what a workflow actually produced. The image
        is downscaled to `max_dimension` on its longest side."""
        result = media.get_output_image(client, name, max_dimension=max_dimension)
        return ImageContent(
            type="image", data=result["data"], mime_type=result["mime_type"]
        )

    tool(get_output_image, READ_ONLY)

    # ----------------------------------------------------------- authoring

    def validate_workflow(
        workflow: dict | None = None, name: str | None = None
    ) -> dict:
        """Check a workflow against the schema and against real pipeline
        signatures. Free and instant - always run this before run_workflow.
        Give exactly one of `workflow` or `name` - `name` being a stored
        workflow as `list_workflows` reports it."""
        return authoring.validate_workflow(client, workflow=workflow, name=name)

    def save_workflow(name: str, workflow: dict) -> dict:
        """Save a workflow to the server, overwriting any existing workflow
        of that name. Validate it first."""
        return authoring.save_workflow(client, name, workflow)

    def delete_workflow(name: str) -> dict:
        """Permanently delete a stored workflow."""
        return authoring.delete_workflow(client, name)

    tool(validate_workflow, READ_ONLY)
    tool(save_workflow, OVERWRITES)
    tool(delete_workflow, DELETES)

    # ------------------------------------------------------------ diagnose

    def run_workflow(
        workflow_path: str | None = None,
        inline_workflow: dict | None = None,
        arguments: dict | None = None,
        acknowledged_cost: bool = False,
    ) -> dict:
        """Queue a workflow for generation. THIS COSTS GPU TIME: a run
        occupies the machine for minutes and the engine runs one job at a
        time. Tell the user what will run and get their go-ahead, then pass
        acknowledged_cost=true. Returns as soon as the job is queued; poll
        get_job_events for progress. Give exactly one of `workflow_path` -
        a catalog name from `list_workflows` or a path on the server - or
        `inline_workflow`."""
        return diagnose.run_workflow(
            client,
            workflow_path=workflow_path,
            inline_workflow=inline_workflow,
            arguments=arguments,
            acknowledged_cost=acknowledged_cost,
        )

    def get_job(job_id: str) -> dict:
        """Get a job's status, warnings, output manifest, error and
        traceback."""
        return diagnose.get_job(client, job_id)

    def get_job_events(job_id: str, after: int = -1, limit: int = 200) -> dict:
        """Get a page of a job's progress events - phase transitions, memory
        readings and log lines. `after` is exclusive: pass back the previous
        call's `last_seq` to continue."""
        return diagnose.get_job_events(client, job_id, after=after, limit=limit)

    def cancel_job(job_id: str) -> dict:
        """Ask a queued or running job to stop."""
        return diagnose.cancel_job(client, job_id)

    def rerun_job(job_id: str, acknowledged_cost: bool = False) -> dict:
        """Queue a fresh job from a previous job's stored specification. THIS
        COSTS GPU TIME: a rerun is a run - it occupies the machine for
        minutes and the engine runs one job at a time. Tell the user what
        will run and get their go-ahead, then pass acknowledged_cost=true."""
        return diagnose.rerun_job(client, job_id, acknowledged_cost=acknowledged_cost)

    def move_job(job_id: str, direction: str) -> dict:
        """Reorder a queued job: up, down, front, or back."""
        return diagnose.move_job(client, job_id, direction)

    tool(get_job, READ_ONLY)
    tool(get_job_events, READ_ONLY)
    for fn in (run_workflow, cancel_job, rerun_job, move_job):
        tool(fn, WRITES)

    # -------------------------------------------------------------- models

    def download_model(repo_id: str, acknowledged_cost: bool = False) -> dict:
        """Fetch a model repo into the Hugging Face cache. THIS COSTS DISK
        AND BANDWIDTH: a model repo is commonly tens of gigabytes. Check
        list_models first - it may already be cached. Tell the user what you
        are about to fetch and get their go-ahead, then pass
        acknowledged_cost=true. Returns as soon as the download starts; poll
        list_downloads for progress."""
        return models.download_model(
            client, repo_id, acknowledged_cost=acknowledged_cost
        )

    def list_downloads() -> dict:
        """List model downloads the server is running or recently ran."""
        return models.list_downloads(client)

    def cancel_download(download_id: str) -> dict:
        """Ask a running model download to stop. Partial files stay in the
        cache and resume if it is retried."""
        return models.cancel_download(client, download_id)

    def delete_model(repo: str, acknowledged_cost: bool = False) -> dict:
        """Delete every cached revision of one model repo. THIS IS NOT
        RECOVERABLE: getting the model back means downloading it again. Tell
        the user which repo and how much it frees, get their go-ahead, then
        pass acknowledged_cost=true. Refused while a job or download is
        active."""
        return models.delete_model(client, repo, acknowledged_cost=acknowledged_cost)

    def get_diffusers_state() -> dict:
        """Get the installed diffusers version and any update in flight."""
        return models.get_diffusers_state(client)

    def update_diffusers(acknowledged_cost: bool = False) -> dict:
        """Upgrade diffusers to GitHub HEAD. THIS CAN BREAK THE INSTALL: it
        installs an untagged development build that workflows running today
        may not survive, and this tool cannot undo it. Report the current
        version, explain why the update is worth it, get the user's
        go-ahead, then pass acknowledged_cost=true. Refused while a job is
        running or queued."""
        return models.update_diffusers(client, acknowledged_cost=acknowledged_cost)

    tool(list_downloads, READ_ONLY)
    tool(get_diffusers_state, READ_ONLY)
    for fn in (download_model, cancel_download, update_diffusers):
        tool(fn, WRITES)
    tool(delete_model, DELETES)

    return server
