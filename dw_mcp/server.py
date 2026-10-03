"""Assemble the MCP tool surface over a DwClient.

With the `tools_*` modules, the only code that imports the MCP SDK. The
tools are methods of the classes there, each body a one-line call into a
handler, so the handlers stay testable without a session; this file holds
the server's instructions, the annotation constants and the registration
order, which is the order of the tool listing an agent reads.
"""

import functools
import inspect
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from dw_mcp.client import DwApiError
from dw_mcp.tools_authoring import (
    AuthoringTools,
    LoraTools,
    PromptTools,
    WorkspaceTools,
)
from dw_mcp.tools_catalog import CatalogTools, ModelTools
from dw_mcp.tools_jobs import JobTools
from dw_mcp.tools_media import AssetTools, MediaTools

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
READ_ONLY_OPEN = ToolAnnotations(read_only_hint=True, open_world_hint=True)
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


INSTRUCTIONS = (
    "Generate images, video and audio on a real GPU: author, run "
    "and diagnose diffusers-workflow jobs against a running "
    "dw.serve. A workflow is a JSON document of named steps, each a "
    "diffusers pipeline or a utility task.\n"
    "\n"
    "Start from `list_workflows(shape=...)` and run what the catalog "
    "already holds, with `arguments` overriding its variables. "
    "Shapes: image, image-set, image-edit, shot, sequence, audio, "
    "text, utility. Traits: has-audio, chained, image-conditioned, "
    "identity-referenced, needs-input-media, composes-workflows. An "
    "open-ended request names a subject, not a shape - decide the "
    "deliverable's shape first. `list_guides` indexes the docs by "
    "section and `list_tasks` is what a shape is composed from; "
    "author new JSON only when neither the catalog nor a "
    "composition covers the request.\n"
    "\n"
    "`get_server_info` reports the accelerator and this "
    "session's workspace; a CUDA-only choice is unavailable on an "
    "mps or cpu server.\n"
    "\n"
    'The loop: `get_guide("workflows", section="Authoring a '
    'workflow from an agent")` before writing or repairing JSON -> '
    "`validate_workflow` (free; repeat until clean) -> quote its "
    "`plan.estimate` and get the user's go-ahead -> `run_workflow` "
    "-> `wait_for_job` -> `get_job` -> `get_output_image`, "
    "`get_output_frames`, `get_output_audio` to judge the result "
    "against the ask. Tools that spend GPU time or disk, or "
    "delete, refuse until "
    "`acknowledged_cost` is set. A workflow you wrote has no "
    "cost: quote the `models/` entry for its pipeline "
    "(`list_workflows(include_models=true)`) times the number of "
    "images. Validate warns (never refuses) past the VRAM ceiling "
    "inherited from the catalog template with the same pipeline; "
    "your offload or quantization may differ.\n"
    "\n"
    "Arguments carry references rather than literals: `variable:`, "
    "`previous_result:`, `prompt:` (the stored prompt library), "
    "`asset:` (input media on the server - `upload_asset`, "
    "`keep_output`) and `output:` (an earlier run's file). The "
    "guide's References section defines each; a local path means "
    "nothing to the server. "
    "`use_workspace` picks this session's workspace."
)


def build_server(client):
    """An MCP server whose tools all run against `client`."""
    server = MCPServer("diffusers-workflow", instructions=INSTRUCTIONS)

    def tool(fn, annotations):
        # The SDK ships fn.__doc__ verbatim. Python 3.13+ strips a docstring's
        # common indentation at compile time, 3.12 does not - so without this
        # every continuation line reaches the agent with eight leading spaces,
        # about 1_100 tokens of resident surface on the interpreter most
        # servers run. cleandoc makes the description the same on both.
        server.add_tool(
            _anticipated(fn),
            name=fn.__name__,
            description=inspect.cleandoc(fn.__doc__ or ""),
            annotations=annotations,
        )

    # Registration order is the tool listing's order. A method's __doc__ is
    # read-only, so the one interpolated description (the wait cap) is
    # formatted on the class before any method is bound here.
    cat, mod = CatalogTools(client), ModelTools(client)
    med, asset = MediaTools(client), AssetTools(client)
    wks, aut, prm = WorkspaceTools(client), AuthoringTools(client), PromptTools(client)
    lor = LoraTools(client)
    job = JobTools(client)

    for fn in (
        cat.list_guides,
        cat.get_guide,
        cat.list_workflows,
        cat.get_workflow,
        cat.get_schema,
        cat.list_pipelines,
        cat.get_pipeline_signature,
        cat.list_classes,
        cat.get_class,
        cat.list_tasks,
        cat.get_task,
        cat.list_models,
        cat.get_memory,
        cat.get_health,
        cat.get_server_info,
        cat.list_jobs,
        cat.list_gallery,
        cat.get_gallery_metadata,
    ):
        tool(fn, READ_ONLY)
    tool(cat.clear_memory, WRITES)

    for fn in (
        med.get_output_image,
        med.get_output_audio,
        med.get_output_frames,
        med.get_output_text,
        med.assess_output,
    ):
        tool(fn, READ_ONLY)
    tool(med.download_output, OVERWRITES)
    tool(med.delete_output, DELETES)

    tool(asset.list_assets, READ_ONLY)
    tool(asset.upload_asset, WRITES)
    tool(asset.keep_output, WRITES)
    tool(asset.delete_asset, DELETES)

    tool(wks.list_workspaces, READ_ONLY)
    tool(wks.use_workspace, WRITES)
    tool(wks.create_workspace, WRITES)
    tool(wks.delete_workspace, DELETES)

    tool(aut.validate_workflow, READ_ONLY)
    tool(aut.save_workflow, OVERWRITES)
    tool(aut.delete_workflow, DELETES)

    for fn in (
        prm.list_prompts,
        prm.get_prompt,
        prm.get_prompt_schema,
        prm.list_enhancers,
    ):
        tool(fn, READ_ONLY)
    tool(prm.save_prompt, OVERWRITES)
    tool(prm.delete_prompt, DELETES)
    tool(prm.enhance_prompt, WRITES)

    tool(lor.list_loras, READ_ONLY)
    tool(lor.save_lora, OVERWRITES)
    tool(lor.recommend_loras, READ_ONLY_OPEN)

    for fn in (job.get_job, job.get_job_workflow, job.get_job_events, job.wait_for_job):
        tool(fn, READ_ONLY)
    for fn in (
        job.run_workflow,
        job.cancel_job,
        job.rerun_job,
        job.move_job,
        job.export_job,
    ):
        tool(fn, WRITES)

    tool(mod.list_downloads, READ_ONLY)
    tool(mod.get_diffusers_state, READ_ONLY)
    for fn in (mod.download_model, mod.cancel_download, mod.update_diffusers):
        tool(fn, WRITES)
    tool(mod.delete_model, DELETES)

    return server
