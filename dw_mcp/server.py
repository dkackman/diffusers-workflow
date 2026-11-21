"""Assemble the MCP tool surface over a DwClient.

With the `tools_*` modules, the only code that imports the MCP SDK. The
tools are methods of the classes there, each body a one-line call into a
handler, so the handlers stay testable without a session; this file holds
the server's instructions, the annotation constants and the registration
order, which is the order of the tool listing an agent reads.

It also serves the notes channel (`serve_notes_channel`): a second, tool-less
stdio server that tells an open Claude Code session about the notes a user
saves on a request. What it announces is decided in `dw_mcp/ledger.py`.
"""

import functools
import inspect
import sys

import anyio
import httpx
from mcp.server.connection import Connection
from mcp.server.lowlevel import Server
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.runner import serve_connection
from mcp.server.stdio import stdio_server
from mcp.shared.jsonrpc_dispatcher import JSONRPCDispatcher
from mcp.types import ToolAnnotations

from dw_mcp.client import DwApiError
from dw_mcp.ledger import CONNECT_GRACE_SECONDS, NoteWatch, retry_delay
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
    "and diagnose diffusers-workflow jobs on dw.serve. "
    "A workflow is JSON steps: each a "
    "diffusers pipeline or a utility task.\n"
    "\n"
    "Start from `list_workflows(shape=...)`: run what the catalog "
    "holds, with `arguments` for its variables. "
    "Shapes: image, image-set, image-edit, shot, sequence, audio, "
    "text, utility. Traits: has-audio, chained, image-conditioned, "
    "identity-referenced, needs-input-media, composes-workflows. An "
    "open-ended ask names a subject: decide the shape first. "
    "`list_guides` indexes the docs, "
    "`list_tasks` what a shape is composed from; author new JSON "
    "only when neither covers it.\n"
    "\n"
    "`get_server_info` reports the accelerator and the "
    "session workspace (`use_workspace` picks it); CUDA-only "
    "choices fail on mps or cpu. One job runs per GPU; "
    "`get_health` lists each card's.\n"
    "\n"
    '`get_guide("workflows", section="Authoring a '
    'workflow from an agent")` before writing or fixing JSON -> '
    "`validate_workflow` (free; repeat until clean) -> quote its "
    "`plan.estimate`, get the user's go-ahead -> `run_workflow` "
    "-> `wait_for_job` -> `get_job` -> `get_output_image`, "
    "`get_output_frames`, `get_output_audio` to judge it "
    "against the ask. Give each variation a `note` "
    "saying what it tests. A 3+ job ask gets a request "
    "(`set_request`): "
    '`run_workflow(request="<id>/<item>")` links each job; the '
    "item's `accepted` names the keeper, else `set_note` "
    "marks it. Tools "
    "that spend GPU time or disk or delete need `acknowledged_cost`. "
    "A workflow you wrote has no "
    "cost: quote its pipeline's `models/` entry "
    "(`list_workflows(include_models=true)`) times the number of "
    "images. Validate warns (never refuses) past the VRAM ceiling "
    "of the same-pipeline catalog template - your offload or "
    "quantization may differ - but refuses templates past the card.\n"
    "\n"
    "Arguments carry references: `variable:`, "
    "`previous_result:`, `prompt:` (saved prompts), "
    "`asset:` (server input media: `upload_asset`, "
    "`keep_output`) and `output:` (a prior run's file). The "
    "guide's References section defines each; a local path is not "
    "on the server."
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
        med.compare_outputs,
    ):
        tool(fn, READ_ONLY)
    tool(med.download_output, OVERWRITES)
    tool(med.delete_output, DELETES)

    tool(asset.list_assets, READ_ONLY)
    tool(asset.upload_asset, WRITES)
    tool(asset.keep_output, WRITES)
    tool(asset.set_note, WRITES)
    tool(asset.delete_asset, DELETES)

    tool(wks.list_workspaces, READ_ONLY)
    tool(wks.use_workspace, WRITES)
    tool(wks.create_workspace, WRITES)
    tool(wks.copy_to_workspace, WRITES)
    tool(wks.delete_workspace, DELETES)
    tool(wks.get_request, READ_ONLY)
    tool(wks.set_request, WRITES)
    tool(wks.delete_request, DELETES)

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


# --- the notes channel ------------------------------------------------------
#
# A Claude Code channel (code.claude.com/docs/en/channels-reference, a
# research preview): a stdio MCP server that declares the experimental
# capability below and sends the notification below; Claude Code shows each
# one to the model as <channel source="dw-notes" ...>content</channel>.

NOTES_CHANNEL_NAME = "dw-notes"
CHANNEL_CAPABILITY = "claude/channel"
CHANNEL_NOTIFICATION = "notifications/claude/channel"

NOTES_INSTRUCTIONS = (
    'Events from this channel (<channel source="dw-notes" ...>) are notes '
    "the user saved on a dw request, or on one of its items, in the dw web "
    "UI. A note is the user's own instruction for that request or item. "
    '`kind="note"` is one saved just now: act on it. '
    '`kind="notes_waiting"` lists notes saved before this session '
    "connected: say what is waiting and confirm before spending GPU time on "
    "them.\n"
    "\n"
    "Before acting, read the current plan with the dw tools' "
    "`get_request(request, workspace)` - the note may have been edited or "
    "cleared since. Act through the normal dw tools under their usual cost "
    "gate, then record what you did in the `why` of `set_request`. That "
    "write is the answer: the user is watching the request page and sees "
    "nothing typed in this session, so do not reply here instead of updating "
    "the plan. This channel is one-way and has no tools of its own."
)


def build_notes_channel():
    """The channel's server and what it answers `initialize` with: no
    tools, the channel capability, and its own instructions."""
    server = Server(NOTES_CHANNEL_NAME, instructions=NOTES_INSTRUCTIONS)
    options = server.create_initialization_options(
        experimental_capabilities={CHANNEL_CAPABILITY: {}}
    )
    return server, options


def _say(message):
    # stdout is the MCP transport: everything a person reads goes to stderr
    print(f"dw-mcp: notes channel: {message}", file=sys.stderr, flush=True)


async def _announce(connection, client, watch, interval, sleep, say):
    """Poll and send until cancelled, from a moment after the handshake (see
    CONNECT_GRACE_SECONDS). A poll that fails is reported once per distinct
    reason and retried on a growing delay; it never ends the loop."""
    await connection.initialized.wait()
    await sleep(min(interval, CONNECT_GRACE_SECONDS))
    failures, reported = 0, None
    while True:
        try:
            events = await anyio.to_thread.run_sync(
                watch.poll, client, abandon_on_cancel=True
            )
            fault = watch.fault
        except (DwApiError, httpx.HTTPError) as e:
            events, fault = [], str(e)
        except Exception as e:
            # Whatever answers at that URL, one reply this code did not
            # expect must not end a channel the session cannot restart
            events, fault = [], f"{type(e).__name__}: {e}"
        for event in events:
            await connection.notify(CHANNEL_NOTIFICATION, event)
        if fault is None and reported is not None:
            say("reading the ledger again")
        elif fault is not None and fault != reported:
            say(f"could not read the ledger, will keep trying: {fault}")
        failures, reported = (failures + 1 if fault else 0), fault
        await sleep(retry_delay(interval, failures))


async def serve_notes_channel(client, interval, workspace=None, sleep=anyio.sleep):
    """Serve the notes channel over stdio until the client closes stdin.

    The connection is served handshake-only (`serve_connection` over a loop
    `Connection`), not through `Server.run`: Claude Code does not register a
    server that negotiates protocol revision 2026-07-28 as a channel, and
    `Server.run` would accept that revision's `server/discover` probe. Here
    the probe is answered "method not found" and the client falls back to
    `initialize`. It also hands this function the `Connection`, which is
    what a notification outside any request is sent on.
    """
    server, options = build_notes_channel()
    # No workspace named: every one, so a session that moves with
    # use_workspace still hears its notes
    watch = NoteWatch(workspace, everywhere=workspace is None)
    async with stdio_server() as (read_stream, write_stream):
        dispatcher = JSONRPCDispatcher(
            read_stream, write_stream, inline_methods=frozenset({"initialize"})
        )
        connection = Connection.for_loop(dispatcher)
        async with server.lifespan(server) as state, anyio.create_task_group() as tg:
            tg.start_soon(_announce, connection, client, watch, interval, sleep, _say)
            await serve_connection(
                server,
                dispatcher,
                connection=connection,
                lifespan_state=state,
                init_options=options,
            )
            tg.cancel_scope.cancel()


def run_notes_channel(client, interval, workspace=None):
    """`serve_notes_channel`, blocking. A client that went away without
    closing stdin (a write to a closed pipe) ends it as quietly as one that
    did."""
    gone = (BrokenPipeError, anyio.BrokenResourceError, anyio.ClosedResourceError)
    try:
        anyio.run(serve_notes_channel, client, interval, workspace)
    except* gone:
        _say("the client went away")
