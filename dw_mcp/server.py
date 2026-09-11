"""Assemble the MCP tool surface over a DwClient.

The only module that imports the MCP SDK. Every tool body is a one-line
call into a handler, so the handlers stay testable without a session and
this file stays a description of the surface rather than logic.
"""

import functools
from typing import Literal, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent, ToolAnnotations

from dw_mcp import (
    assets,
    authoring,
    catalog,
    diagnose,
    exports,
    guides,
    media,
    models,
    prompts,
    workspaces,
)
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
            "Generate images and video on a real GPU: author, run and "
            "diagnose diffusers-workflow jobs against a running dw.serve. "
            "A workflow is a JSON document of named steps, each a "
            "diffusers pipeline or a utility task; the engine runs one job "
            "at a time.\n"
            "\n"
            "Start from `list_workflows(shape=...)`: the server keeps a "
            "large catalog, and its compact listing carries each "
            "workflow's summary, shape, traits, cost and variable names - "
            "run what is already there, with `arguments` overriding its "
            "variables, rather than authoring a new workflow for a "
            "request an existing one covers. Shapes: image, image-set, "
            "image-edit, shot, sequence, audio, text, utility. Traits: "
            "has-audio, chained, image-conditioned, identity-referenced, "
            "needs-input-media, composes-workflows.\n"
            "\n"
            "When a request is open-ended - a subject rather than a shape "
            '("a lego movie trailer set in the marvel universe") - no '
            "catalog entry will name it, because entries are written in "
            "shapes: a single image, an image set, one shot, a multi-shot "
            "cut sequence, video with speech. Decide which shape the "
            "deliverable is first, then call `list_workflows` with it; "
            "`list_guides` indexes the documentation by section so a shape "
            "can be looked up rather than guessed at, and `list_tasks` is "
            "what a shape is composed from when no single workflow covers "
            "it. Author new JSON only once neither does, and say what it "
            "will cost before spending it.\n"
            "\n"
            "The engine that answers is one machine: `get_server_info` "
            "reports its accelerator, its directories and which workspace "
            "this session works in, and what a workflow can ask for "
            "follows from that - a CUDA-only choice is not available on an "
            "mps or cpu server.\n"
            "\n"
            "The loop for anything that generates: `get_guide` "
            '("workflows", section "Authoring a workflow from an agent") '
            "before writing or repairing any JSON, since the reference "
            "conventions below are engine-specific and a draft that guesses "
            "them validates and then fails at run time -> `validate_workflow` "
            "(free, catches schema errors and arguments the pipeline does "
            "not accept; repeat until it is clean, since fixing one layer "
            "exposes the next) -> `run_workflow` -> `wait_for_job` rather than a "
            "polling loop -> `get_job` for the manifest -> "
            "`get_output_image` to actually look at what was made and say "
            "whether it answers the request. Tools that cost GPU minutes, "
            "disk or unrecoverable deletion refuse until "
            "`acknowledged_cost=true`: tell the user what it will cost (a "
            "workflow you wrote or copied has no `cost`; quote the figure "
            "from the `models/` entry that loads the same pipeline, found "
            "with `list_workflows(include_models=true)`, times the number "
            "of images), get their go-ahead, then call again.\n"
            "\n"
            "Workflow arguments carry references rather than literals, "
            "which is what makes multi-stage work composable: "
            '"variable:name" (an override), '
            '"previous_result:step" (an earlier step in the same run), '
            '"prompt:name" or "prompt:folder/name" (the stored prompt '
            "library - `list_prompts`, `get_prompt_schema`), "
            '"asset:name.ext" (input media on the server - `list_assets`, '
            "`upload_asset` to push a local file, `keep_output` to promote "
            "a generated file into a stable input), and "
            '"output:workflow/run-id/file.png" (a file an earlier run '
            'wrote, with "latest" in the run-id position picking the '
            "newest run holding it). Prefer an asset: or output: reference "
            "over a filesystem path: a path on this machine usually means "
            "nothing to the server.\n"
            "\n"
            "Each run writes its own directory, "
            "<workflow>/<run id>/, with a manifest beside its files; "
            "`list_gallery` and a job's manifest name files the way "
            "`get_output_image`, `download_output` and `keep_output` "
            "expect them.\n"
            "\n"
            "The server can hold several workspaces - separate workflows, "
            "assets and outputs, one shared prompt library: "
            "`list_workspaces` shows them and `use_workspace` picks one "
            "for the rest of the session, which is how to keep your work "
            "out of another agent's namespace."
        ),
    )

    def tool(fn, annotations):
        server.add_tool(_anticipated(fn), name=fn.__name__, annotations=annotations)

    # ------------------------------------------------------------- catalog

    def list_workflows(
        shape: Optional[str] = None,
        traits: Optional[str] = None,
        configures: Optional[str] = None,
        include_models: bool = False,
    ) -> dict:
        """List the workflows stored on the server, compactly. Decide the
        deliverable's shape first and pass it: one of image, image-set,
        image-edit, shot, sequence, audio, text, utility. `traits` narrows
        further (comma-separated, all must match): has-audio, chained,
        image-conditioned, identity-referenced, needs-input-media,
        composes-workflows. Each entry carries a one-line `summary`, its
        `shape` and `traits` (what it needs supplied), `cost` (measured
        runs per device; null means unknown - call `get_memory` and say
        so), output kinds and variable names. Templates only by default;
        `configures=<template>` lists the checkpoint configs tuned for
        one, `include_models=true` lists them all. `get_workflow` has the
        full description and definition."""
        return catalog.list_workflows(
            client,
            shape=shape,
            traits=[t.strip() for t in (traits or "").split(",") if t.strip()],
            configures=configures,
            include_models=include_models,
        )

    def get_workflow(name: str, variables_only: bool = False) -> dict:
        """Get one stored workflow's full JSON definition, by a name from
        `list_workflows`. Read one before editing it, and to learn the
        idioms this installation actually uses. Pass
        `variables_only=true` when the question is only what a variable
        defaults to - it answers with the variables and their values and
        nothing else, which is a fraction of the definition; long defaults
        (a shot's prompt) come back cut to 200 characters with the cut ones
        named in `truncated`."""
        return catalog.get_workflow(client, name, variables_only=variables_only)

    def get_schema() -> dict:
        """Get the JSON schema every workflow definition must satisfy - the
        authority on a workflow's structure: steps, pipelines, tasks,
        results, variables. Read it before authoring one from scratch, and
        note that schema validation runs before variable substitution, so a
        variable's default has to be the type its use expects (25, not
        "25")."""
        return catalog.get_schema(client)

    def list_pipelines() -> dict:
        """List every diffusers pipeline class this installation provides.
        These are the names a step's `component_type` can take - and the
        list is this installation's, so a pipeline from a newer diffusers
        will not be here until it is updated."""
        return catalog.list_pipelines(client)

    def get_pipeline_signature(name: str) -> dict:
        """Get a pipeline's real call arguments. Check this before proposing
        pipeline arguments - a plausible-looking argument that the pipeline
        does not accept is the most common workflow bug."""
        return catalog.get_pipeline_signature(client, name)

    def list_classes(
        kind: Literal["pipelines", "models", "schedulers", "quantization"],
    ) -> dict:
        """List class names of one kind: pipelines, models, schedulers, or
        quantization. These are the names a workflow's `component_type`,
        `scheduler_type` or `config_type` can take."""
        return catalog.list_classes(client, kind)

    def get_class(name: str, target: Literal["init", "call", "load"] = "init") -> dict:
        """Get a class's argument schema, from whichever entry point a
        workflow reaches it by: `init` reads the constructor (quantization
        configs, schedulers, models built from named arguments), `call`
        reads __call__ (a pipeline's `arguments`), `load` reads
        from_pretrained plus the curated loading knobs, which is what a
        component's `from_pretrained_arguments` can carry."""
        return catalog.get_class(client, name, target=target)

    def list_tasks() -> dict:
        """List every task command a workflow's task step can name - the
        non-pipeline work: upscaling, face restoration, ControlNet
        preprocessors, captioning, frame interpolation, video and audio
        handling. Check here before assuming something needs a pipeline."""
        return catalog.list_tasks(client)

    def get_task(command: str) -> dict:
        """Get a task command's argument schema, read from its real
        implementation signature. The counterpart of
        `get_pipeline_signature` for a task step."""
        return catalog.get_task(client, command)

    def list_models() -> dict:
        """List what the Hugging Face model cache holds, largest first."""
        return catalog.list_models(client)

    def get_memory() -> dict:
        """Get the worker's VRAM and RAM statistics. Check this first when a
        job fails with an out-of-memory error."""
        return catalog.get_memory(client)

    def get_health() -> dict:
        """Check that the server is alive, and see what answered: its
        version and accelerator, whether the worker process is up, the job
        running now and how many are queued."""
        return catalog.get_health(client)

    def get_server_info() -> dict:
        """Get what this installation can do and where it keeps things: the
        accelerator a run will use (`device` - cuda, mps or cpu), the dw
        version, and the workflow, output and prompt directories. Check the
        device before authoring: a CUDA-only choice - bitsandbytes
        quantization, torch.compile, flash attention - is not available on
        an mps or cpu server, and `directories` is what a path passed to
        run_workflow or download_output is relative to. If this session
        works in a named workspace, `directories` are scoped to that
        workspace."""
        return workspaces.server_info(client)

    def list_jobs() -> dict:
        """List queued, running and recent jobs, with their status and queue
        position. The ids here are what `get_job`, `wait_for_job`,
        `get_job_events`, `cancel_job`, `rerun_job` and `move_job` take -
        including jobs from before this session, so a run someone started in
        the browser can be picked up here. In a named workspace this lists
        that workspace's jobs; in the default workspace it lists every job
        the server holds, whichever workspace ran it."""
        return catalog.list_jobs(client)

    def list_gallery(limit: int = 50) -> dict:
        """List generated output files, newest first. A name is
        <workflow>/<run id>/<file> - the form `get_output_image`,
        `get_output_text`, `download_output`, `keep_output` and
        `delete_output` all take, and the form an "output:" reference in a
        later workflow is built from. Each entry also carries a ready-made
        `url` for viewing the file over HTTP, already scoped to the right
        workspace; use it as given rather than composing one from the
        name."""
        return catalog.list_gallery(client, limit=limit)

    def get_gallery_metadata(name: str, envelope: bool = False) -> dict:
        """Get the metadata embedded in a generated file: the exact
        workflow, arguments and seed that produced it. Use this to
        reproduce a result, or to see what a run that went wrong actually
        ran - it is the definition, not a summary, so it can be edited and
        re-run. For audio and video the `media` block carries duration,
        sample rate, channels, fps, size and level - the checks an agent
        that cannot listen makes on a deliverable. `envelope=true` adds
        that level second by second (`media.envelope.rms_dbfs` /
        `peak_dbfs`, one entry per second), which is what says *where* in a
        track something is: whether a shot is still sounding at its last
        frame, how deep the hole at a seam goes, where a score goes quiet.
        Leave it off unless you are asking a question about a position in
        the track - a long track is a long list."""
        return catalog.get_gallery_metadata(client, name, envelope=envelope)

    def list_guides() -> dict:
        """List the documentation the engine serves: each guide's
        name, what it covers, and its section headings. Read this when a
        request is open-ended enough that no catalog entry obviously
        answers it - a request names a subject ("a lego movie trailer"),
        while the catalog and these guides are written in shapes
        (multi-shot video, cuts, a consistent cast, narration over
        B-roll), and the section headings are where the two get matched
        up. Cheaper than guessing: reading a section costs a fraction of
        one wrong run."""
        return guides.list_guides(client)

    def get_guide(name: str, section: str | None = None) -> dict:
        """Get one guide from `list_guides`, whole or one section of it.
        Prefer a section - a guide runs to thousands of lines, and the
        headings in the listing are there so the right part can be asked
        for by name. A section name is matched loosely, so a heading
        copied approximately still resolves."""
        return guides.get_guide(client, name, section=section)

    for fn in (
        list_guides,
        get_guide,
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
        get_server_info,
        list_jobs,
        list_gallery,
        get_gallery_metadata,
    ):
        tool(fn, READ_ONLY)

    # --------------------------------------------------------------- media

    def get_output_image(
        name: str, max_dimension: int = 768
    ) -> list[ImageContent | TextContent]:
        """Look at a generated image, named as `list_gallery` or a job's
        manifest reports it. Use this to judge output quality - it is the
        only way to see what a workflow actually produced, and a run that
        succeeded can still have made the wrong picture. Images only: a
        video or audio output is refused, so inspect those with
        `get_gallery_metadata` or hand the user the file. The image is
        downscaled to `max_dimension` on its longest side; the second part
        of the result reports the size it went in and came out at, so a
        downscale is never silent."""
        result = media.get_output_image(client, name, max_dimension=max_dimension)
        image = ImageContent(
            type="image", data=result["data"], mime_type=result["mime_type"]
        )
        telemetry = TextContent(
            type="text",
            text=(
                f"name: {result['name']}\n"
                f"original_size: {result['original_size']}\n"
                f"returned_size: {result['returned_size']}\n"
                f"bytes: {result['bytes']}"
            ),
        )
        return [image, telemetry]

    def get_output_text(name: str, max_characters: int = 20000) -> dict:
        """Read a text output - a prompt enhancement, or any step whose
        result is text/plain or JSON. Truncated to `max_characters`, and
        the reply says how long the file really was."""
        return media.get_output_text(client, name, max_characters=max_characters)

    def delete_output(name: str) -> dict:
        """Permanently remove one generated file from the output directory.
        Not recoverable: rerunning the job that made it is the only way
        back, and any "output:" reference pointing at it stops resolving.
        Prefer `keep_output` first if it is worth keeping."""
        return media.delete_output(client, name)

    def download_output(
        name: str, destination: str | None = None, overwrite: bool = False
    ) -> dict:
        """Save one output file to disk on the
        machine running the MCP server - for the stdio `dw-mcp` that is
        your own machine; for a
        `dw.serve --mcp` endpoint it is the GPU box, and this tool is not
        the way to get a file to where you are (use get_output_image /
        get_output_text for inline content, or the `url` that
        `list_gallery` reports for each entry, which already carries the
        workspace selector - do not build an /outputs URL by hand). This is
        also NOT how a generated file becomes an input for a later
        workflow: use `keep_output`, which links it inside the workspace
        under an "asset:" name, rather than writing into the server's asset
        directory behind the API's back. Unlike the inline tools, this works
        for any file type, streams the body straight to disk rather than
        buffering it, and returns no content to the conversation - only
        where it was saved. `destination` may be a
        full path, a directory, or omitted to save into the current
        working directory under the output's own name; a '..' path segment
        in it is refused. An existing file at the resolved path is left
        alone unless `overwrite=True`."""
        return media.download_output(
            client, name, destination=destination, overwrite=overwrite
        )

    tool(get_output_image, READ_ONLY)
    tool(get_output_text, READ_ONLY)
    tool(download_output, OVERWRITES)
    tool(delete_output, DELETES)

    # ---------------------------------------------------------------- assets

    def list_assets() -> dict:
        """List the input media on the server, each with the "asset:"
        reference a workflow argument carries. Look here before asking for
        a file: what a workflow needs may already be there."""
        return assets.list_assets(client)

    def upload_asset(file_path: str, asset_name: str | None = None) -> dict:
        """Put a local image, video or audio file into the server's asset
        library and get back the "asset:" reference to use in a workflow.
        The file is read from the machine this MCP server runs on and
        pushed to the engine, so it is how an input reaches a dw.serve
        running somewhere else. Accepts the usual image, video and audio
        extensions, up to 200MB. Reference the result rather than a path: a
        path on this machine means nothing to the server. Pass `asset_name`
        to store it under a readable name ("cast/priya-voice.wav", folders
        allowed, the file's extension assumed) - without one the stored
        name is random, and a set of related inputs cannot be told apart in
        the workflows that carry them."""
        return assets.upload_asset(client, file_path, asset_name=asset_name)

    def keep_output(
        name: str, asset_name: str | None = None, overwrite: bool = False
    ) -> dict:
        """Keep a generated file as an input asset under a stable "asset:"
        name, so later workflows can rely on it - a run's own name moves
        ("latest") or breaks when outputs are pruned. This is the step
        between a render you liked and the next stage that conditions on
        it. `name` is a gallery name; `asset_name` defaults to the file's
        own. The copy happens on the server, inside the workspace: nothing
        is downloaded or re-uploaded."""
        return assets.keep_output(
            client, name, asset_name=asset_name, overwrite=overwrite
        )

    tool(list_assets, READ_ONLY)
    tool(upload_asset, WRITES)
    tool(keep_output, WRITES)

    # ------------------------------------------------------------ workspaces

    def list_workspaces() -> dict:
        """List the server's workspaces and say which one this session is
        working in. Each has its own workflows, assets and outputs; the
        stored prompt library is shared by all of them."""
        return workspaces.list_workspaces(client)

    def use_workspace(name: str) -> dict:
        """Work in a different workspace for the rest of this session - every
        later call reads and writes there. Use this to keep your work out of
        another agent's namespace, rather than sharing the default one."""
        return workspaces.use_workspace(client, name)

    def create_workspace(name: str, use: bool = False) -> dict:
        """Create a workspace on the server. It gets its own workflows,
        assets and outputs and shares the one prompt library. The name is a
        single path segment and cannot be one of the reserved folder names
        (workflows, prompts, assets, outputs). Pass use=true to switch this
        session to it as well; otherwise the session stays where it was and
        the result says so."""
        return workspaces.create_workspace(client, name, use=use)

    def delete_workspace(name: str, acknowledged_cost: bool = False) -> dict:
        """Permanently delete a workspace and every workflow, asset and
        generated file in it. Refuses without acknowledged_cost=True, and
        reports what it would remove instead."""
        return workspaces.delete_workspace(
            client, name, acknowledged_cost=acknowledged_cost
        )

    tool(list_workspaces, READ_ONLY)
    tool(use_workspace, WRITES)
    tool(create_workspace, WRITES)
    tool(delete_workspace, DELETES)

    # ----------------------------------------------------------- authoring

    def validate_workflow(
        workflow: dict | None = None,
        name: str | None = None,
        workspace: str | None = None,
    ) -> dict:
        """Check a workflow against the schema and against real pipeline
        signatures. Free and instant - always run this before run_workflow.
        Give exactly one of `workflow` or `name` - `name` being a stored
        workflow as `list_workflows` reports it. Every schema error comes
        back at once, each with its JSON path. `workspace` names the
        workspace for this one call without switching the session to it -
        use it to pin a job whose `output:` or `asset:` references live in a
        workspace other than the session's."""
        return authoring.validate_workflow(
            client, workflow=workflow, name=name, workspace=workspace
        )

    def save_workflow(name: str, workflow: dict) -> dict:
        """Save a workflow to the server's writable workflow directory,
        overwriting any existing workflow of that name there. Validate it
        first. A name that currently resolves to a read-only source (an
        examples directory) is not overwritten - the copy lands in the
        writable directory and shadows it from then on, which is how an
        example gets adapted without being damaged. `name` may include
        folders."""
        return authoring.save_workflow(client, name, workflow)

    def delete_workflow(name: str) -> dict:
        """Permanently delete a stored workflow from this workspace. A
        workflow from a read-only examples directory is refused rather than
        deleted - `list_workflows` reports which those are as
        `writable: false`."""
        return authoring.delete_workflow(client, name)

    tool(validate_workflow, READ_ONLY)
    tool(save_workflow, OVERWRITES)
    tool(delete_workflow, DELETES)

    # ------------------------------------------------------------- prompts

    def list_prompts() -> dict:
        """List the stored prompts, with their text and descriptions. A
        workflow argument reaches one of these by writing
        "prompt:name" or "prompt:folder/name"."""
        return prompts.list_prompts(client)

    def get_prompt(name: str) -> dict:
        """Get one stored prompt's full definition - its text, description,
        intended model and tags - by a name from `list_prompts`. The prompt
        library is shared by every workspace on the server."""
        return prompts.get_prompt(client, name)

    def get_prompt_schema() -> dict:
        """Get the JSON schema every stored prompt must satisfy. Check this
        before writing one, as you would get_schema before a workflow."""
        return prompts.get_prompt_schema(client)

    def save_prompt(name: str, prompt: dict) -> dict:
        """Save a prompt to the library, overwriting any prompt of that
        name. Its `text` may not itself begin with a reference prefix
        (variable:, previous_result:, constant:, asset:, output:, prompt:)
        - the server refuses that to prevent a reference resolving twice.
        The library is shared by every workspace on this server."""
        return prompts.save_prompt(client, name, prompt)

    def delete_prompt(name: str) -> dict:
        """Permanently delete a stored prompt. A workflow that still
        references it by "prompt:name" will fail to load."""
        return prompts.delete_prompt(client, name)

    def list_enhancers() -> dict:
        """List the enhancer presets `enhance_prompt` accepts - one per
        target model family. Call this before enhance_prompt rather than
        guessing a preset name."""
        return prompts.list_enhancers(client)

    def enhance_prompt(
        idea: str,
        preset: str = "h3",
        model_name: str | None = None,
        device: str | None = None,
        acknowledged_cost: bool = False,
    ) -> dict:
        """Expand a short idea into a full prompt with a language model.
        THIS COSTS TIME ON THE ENGINE: it queues a real job, and the engine
        runs one at a time, so a generation waiting behind it is delayed.
        Tell the user what will be enhanced and get their go-ahead, then
        pass acknowledged_cost=true. Returns as soon as the job is queued;
        the enhanced text is the text file in its finished manifest."""
        return prompts.enhance_prompt(
            client,
            idea,
            preset=preset,
            model_name=model_name,
            device=device,
            acknowledged_cost=acknowledged_cost,
        )

    for fn in (list_prompts, get_prompt, get_prompt_schema, list_enhancers):
        tool(fn, READ_ONLY)
    tool(save_prompt, OVERWRITES)
    tool(delete_prompt, DELETES)
    tool(enhance_prompt, WRITES)

    # ------------------------------------------------------------ diagnose

    def run_workflow(
        workflow_path: str | None = None,
        inline_workflow: dict | None = None,
        arguments: dict | None = None,
        acknowledged_cost: bool = False,
        workspace: str | None = None,
    ) -> dict:
        """Queue a workflow for generation. THIS COSTS GPU TIME: a run
        occupies the machine for minutes and the engine runs one job at a
        time. Tell the user what will run and get their go-ahead, then pass
        acknowledged_cost=true. Returns as soon as the job is queued - a
        generation outlasts any tool-call timeout - so follow it with
        `wait_for_job`, then `get_job` for the manifest. Give exactly one of
        `workflow_path` - a catalog name from `list_workflows`, with or
        without .json, or a path on the server - or `inline_workflow`, a
        full definition for a request nothing stored covers. `arguments`
        overrides the workflow's variables by name, which is how one stored
        workflow serves many requests without being edited or copied.
        `workspace` names the workspace for this one call without switching
        the session to it - use it to pin a job whose `output:` or `asset:`
        references live in a workspace other than the session's."""
        return diagnose.run_workflow(
            client,
            workflow_path=workflow_path,
            inline_workflow=inline_workflow,
            arguments=arguments,
            acknowledged_cost=acknowledged_cost,
            workspace=workspace,
        )

    def get_job(job_id: str) -> dict:
        """Get a job's status, argument warnings, output manifest, error and
        traceback. The manifest names each step's files the way
        `get_output_image`, `download_output` and `keep_output` take them; a
        step served from the step cache is marked `reused` and reports the
        earlier run's files. When a job failed, the error and traceback here
        are what to read before changing anything."""
        return diagnose.get_job(client, job_id)

    def get_job_workflow(job_id: str) -> dict:
        """Get the workflow a job actually ran. When `realized` is true every
        mutable input is pinned - the caller's arguments folded into the
        variables, the seed the run used, stored prompt text inlined, and any
        `output:.../latest/...` rewritten to the run it resolved to - so the
        definition reproduces that run however the library changes. When it is
        false the job predates run tracking and this is the definition as
        submitted. After a long inline run worth keeping, this then
        `save_workflow` is how it gets a name."""
        return diagnose.get_job_workflow(client, job_id)

    def get_job_events(job_id: str, after: int = -1, limit: int = 200) -> dict:
        """Get a page of a job's progress events - phase transitions, memory
        readings and log lines. `after` is exclusive: pass back the previous
        call's `last_seq` to continue."""
        return diagnose.get_job_events(client, job_id, after=after, limit=limit)

    def wait_for_job(job_id: str, timeout_seconds: int = 20) -> dict:
        """Block until a job finishes, instead of polling get_job or
        get_job_events by hand. Returns as soon as the job's status is
        succeeded, failed or cancelled, or - if timeout_seconds elapses
        first - returns its current status with still_running: true so you
        can call again. Does not queue anything, so no acknowledged_cost.
        timeout_seconds is capped well under a generation's real runtime;
        call it repeatedly for a long job. Returns a slim job - status,
        warnings, error, and the manifest once finished - without the
        arguments; get_job has those."""
        return diagnose.wait_for_job(client, job_id, timeout_seconds=timeout_seconds)

    def cancel_job(job_id: str) -> dict:
        """Ask a queued or running job to stop. Cooperative: a running job
        stops at the next step or denoise-step boundary, not instantly.
        Deliberately not gated - it ends a cost rather than starting one."""
        return diagnose.cancel_job(client, job_id)

    def rerun_job(
        job_id: str, acknowledged_cost: bool = False, new_seed: bool = False
    ) -> dict:
        """Queue a fresh job from a previous job's stored specification. THIS
        COSTS GPU TIME: a rerun is a run - it occupies the machine for
        minutes and the engine runs one job at a time. Tell the user what
        will run and get their go-ahead, then pass acknowledged_cost=true.

        Pass new_seed=true for a different image: a workflow that pins its
        seed reruns to the same pixels, and the step cache serves that whole
        run from the earlier one's files (marked `reused`) in a fraction of a
        second rather than generating anything."""
        return diagnose.rerun_job(
            client,
            job_id,
            acknowledged_cost=acknowledged_cost,
            new_seed=new_seed,
        )

    def move_job(
        job_id: str, direction: Literal["up", "down", "front", "back"]
    ) -> dict:
        """Reorder a queued job. Only a job still waiting can move; the one
        already running cannot."""
        return diagnose.move_job(client, job_id, direction)

    def export_job(job_id: str, overwrite: bool = False) -> dict:
        """Gather one finished job into a directory on the server: the
        realized workflow, the run's manifest, the job row, a README, and
        copies of every asset it used, every earlier run's file it read and
        every file it made. The export copies every output and input file
        rather than linking them, so a video job's export costs its size
        again on the server's disk; `total_bytes` in the result reports
        what was copied. Returns the directory, a zip URL, the file list
        with sizes and the total. The three JSON files are in the zip, not
        repeated here - get_job_workflow and get_job serve them individually.
        THE DIRECTORY IS ON THE MACHINE RUNNING THE SERVER, not on yours. To
        give the user the files, fetch the zip URL and unpack it into
        exports/ under the session's working directory - it is the user's
        deliverable, not a temp file; the archive already unpacks into one
        folder named after the job id, so do not create that folder first.
        Refuses a job that is still running; refuses an existing export
        unless overwrite=true."""
        return exports.export_job(client, job_id, overwrite=overwrite)

    tool(get_job, READ_ONLY)
    tool(get_job_workflow, READ_ONLY)
    tool(get_job_events, READ_ONLY)
    tool(wait_for_job, READ_ONLY)
    for fn in (run_workflow, cancel_job, rerun_job, move_job, export_job):
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
