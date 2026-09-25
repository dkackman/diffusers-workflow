"""Assemble the MCP tool surface over a DwClient.

The only module that imports the MCP SDK. Every tool body is a one-line
call into a handler, so the handlers stay testable without a session and
this file stays a description of the surface rather than logic.
"""

import functools
import inspect
from typing import Literal, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import AudioContent, ImageContent, TextContent, ToolAnnotations

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
            "Generate images, video and audio on a real GPU: author, run "
            "and diagnose diffusers-workflow jobs against a running "
            "dw.serve. A workflow is a JSON document of named steps, each a "
            "diffusers pipeline or a utility task; the engine runs one job "
            "at a time.\n"
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
            "`get_server_info` reports the accelerator, directories and this "
            "session's workspace; a CUDA-only choice is unavailable on an "
            "mps or cpu server.\n"
            "\n"
            'The loop: `get_guide("workflows", section="Authoring a '
            'workflow from an agent")` before writing or repairing JSON -> '
            "`validate_workflow` (free; repeat until clean) -> quote its "
            "`plan.estimate` and get the user's go-ahead -> `run_workflow` "
            "-> `wait_for_job` -> `get_job` -> `get_output_image`, "
            "`get_output_frames`, `get_output_audio` to look at and listen "
            "to the result and judge it against the request. Tools that "
            "spend GPU time or disk, or delete for good, refuse until "
            "`acknowledged_cost` is set. A workflow you wrote has no "
            "measured cost: quote the `models/` entry that loads the same "
            "pipeline (`list_workflows(include_models=true)`) times the "
            "number of images.\n"
            "\n"
            "Arguments carry references rather than literals: `variable:`, "
            "`previous_result:`, `prompt:` (the stored prompt library), "
            "`asset:` (input media on the server - `upload_asset`, "
            "`keep_output`) and `output:` (an earlier run's file). The "
            "guide's References section defines each; prefer them to a "
            "local path, which means nothing to the server. "
            "`use_workspace` picks the workspace this session works in."
        ),
    )

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
        `shape` and `traits` (what it needs supplied), `cost` (curated:
        measured once by a maintainer on the devices named, never derived
        from this server's history - null means nobody wrote one down, not
        that the run is cheap. It is also the mark of an
        entry that has been run through on a real device: one without a
        `cost` has only been authored, and its first run is the one that
        finds what the description could not verify. `observed_minutes` and
        `observed_runs`, when present, are this box's *own* finished runs
        of that workflow - the cold median, model load included, and how
        many runs are behind it. Prefer it when quoting a price for this
        machine, fall back to `cost`, and say "unknown" only when neither
        is there; say which one you used - a maintainer's card and this
        box's average are different claims), output kinds and variable names. `lists`, present for a
        list-driven workflow, names per list variable the fields an entry
        takes, the steps run over it and the default's length; there
        `cost[].per_entry`, when present, is the measured cost of one
        entry (`{variable, minutes, entries}`), so a run over a
        different-length list can be priced from it. `constraints`, present
        for a workflow that bounds a variable, is the rule each bounded one
        has to satisfy, terse - pass an
        `arguments` value outside it and `validate_workflow` refuses it for
        free, instead of the run failing after the weights are loaded.
        Templates only by default; `configures=<template>` lists the checkpoint configs
        tuned for one, `include_models=true` lists them all. `get_workflow`
        has the full description and definition."""
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
        come back cut to 200 characters with the cut ones named in
        `truncated`, reaching into a list default too - a shot's prompt
        is named `shots[0].prompt`. `observed`, when this box has run the
        workflow, is the whole derived-cost block: `cold_minutes` /
        `cold_runs` (model load included) and `warm_minutes` /
        `warm_runs` (model already resident), the `drivers` the figure is
        for, `since`, and `unclassified_runs` when a run's event tail was
        trimmed too far to tell which it was. Step-cache-only runs are
        excluded. A variable the workflow bounds is
        reported under `constraints` beside its default - the range and the
        step it has to land on - so a frame count is read rather than
        guessed at."""
        return catalog.get_workflow(client, name, variables_only=variables_only)

    def get_schema(section: str | None = None) -> dict:
        """Get the JSON schema every workflow definition must satisfy - the
        authority on a workflow's structure: steps, pipelines, tasks,
        results, variables. Read it before authoring one from scratch, and
        note that schema validation runs before variable substitution, so a
        variable's default has to be the type its use expects (25, not
        "25").

        Ask for the part you need: `section` takes `steps`, `pipelines`,
        `tasks`, `result`, `variables` or `configuration` and answers that
        fragment - a tenth the size - with `elsewhere` naming the section
        that holds each definition it still references. The whole schema is
        the no-argument call."""
        return catalog.get_schema(client, section=section)

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
        job fails with an out-of-memory error.

        `gpu_*` is the card, `host_memory_*` the machine:
        `host_memory_rss_mb` is what the worker process holds and
        `host_memory_peak_rss_mb` the most it has ever held, beside the
        machine's `host_memory_total_mb` / `host_memory_available_mb`. Read
        both - a workflow that offloads (`offload: "sequential"`,
        `group_offload`) keeps its weights in host memory by design, so the
        card can sit near-empty through a generation and VRAM alone will not
        show what a run is holding or failing to release. A host field is
        absent, rather than null, on a platform that cannot measure it.

        `host_pinned_reserved_mb` / `host_pinned_allocated_mb`, when
        present, are torch's pinned-host cache and are part of
        `host_memory_rss_mb` - see the `acceleration` guide, section
        "Reading Memory While Offloading", for what that means for a worker
        that has released every model and still holds gigabytes.

        `live: true` means `info` was measured now and is the worker's own
        memory - only these readings are comparable with each other.
        `live: false` means it was not: `info: null` (with `stale: false`)
        means nothing has been measured because nothing is resident, and a
        populated `info` is a cached earlier reading - `reason` says why
        (`job_running`, `worker_stopped`, `worker_busy`, `worker_unreachable`)
        and `age_seconds` how old it is. A cached reading is not this
        moment's: one taken while a job is loading a model understates what
        is resident by however much has loaded since, so ask again when the
        server is idle rather than comparing it against a live figure.

        `info.step_cache` is the step cache's own accounting: `entries`,
        `retained_bytes` against `max_retained_bytes`."""
        return catalog.get_memory(client)

    def clear_memory() -> dict:
        """Drop every loaded pipeline and the step cache, freeing VRAM/RAM
        immediately instead of waiting for the next job to evict one model
        for another. Also drops the step cache, so a seeded workflow that
        would otherwise reuse cached results regenerates on its next run.

        Refused with a 409 while a job is running or queued - the queue is
        FIFO, so wait for it to finish and retry rather than expecting this
        call to block until it does. On an idle server with no model process
        resident there is nothing loaded to clear, so it succeeds with a null
        `info` rather than failing."""
        return catalog.clear_memory(client)

    def get_health() -> dict:
        """Check that the server is alive, and see what answered: its
        version and accelerator, whether a model process is currently
        resident, the job running now and how many are queued.

        `worker_alive: false` on an otherwise healthy server (`status: ok`)
        is the normal idle state, not a fault - the worker is an on-demand
        subprocess that has not started yet because no job has run since
        the server started or the last memory clear, and it starts with the
        next job."""
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
        workspace. `trust_workflows` reports the posture a submitted
        workflow is read under: false - the default - means the file is
        untrusted input, so an out-of-ecosystem import, remote code, and a
        media location outside the workspace's roots are all refused."""
        return workspaces.server_info(client)

    def list_jobs(
        limit: int = 20, status: str | None = None, workspace: str | None = None
    ) -> dict:
        """List queued, running and recent jobs, newest first, with their
        status and queue position. The ids here are what `get_job`,
        `wait_for_job`, `get_job_events`, `cancel_job`, `rerun_job` and
        `move_job` take - including jobs from before this session, so a run
        someone started in the browser can be picked up here.

        `limit` is the newest N (20 by default); `total` reports how many
        matched, so a truncated answer says so rather than looking
        complete. `status` narrows to one state or a comma-separated set of
        them - queued, running, succeeded, failed, cancelled. `workspace`
        lists one workspace's jobs; without it, a named workspace lists its
        own and the default workspace lists every job the server holds,
        whichever workspace ran it. Each job carries `acknowledged` - `none`,
        `boolean` or `bound` - which form of cost acknowledgement queued it."""
        return catalog.list_jobs(
            client, limit=limit, status=status, workspace=workspace
        )

    def list_gallery(
        limit: int = 50,
        subfolder: str | None = None,
        only_orphans: bool = False,
        workspace: str | None = None,
        folder: str | None = None,
        version: int | None = None,
        media: bool = False,
    ) -> dict:
        """List generated output files, newest first. A name is
        <workflow>/<run id>/<file>, where <file> may itself sit in a
        subfolder the step chose (`final/episode.mp4`) - the form
        `get_output_image`, `get_output_text`, `download_output`,
        `keep_output` and `delete_output` all take, and the form an
        "output:" reference in a later workflow is built from. Each entry
        carries `folder` (the workflow) and `subfolder` (the part of the run:
        by convention `final` is the deliverable and `intermediate` the
        scratch work, '' when the step chose none); `subfolder=` filters on
        the latter, so `subfolder="final"` is "what did these runs
        deliver". Each entry's `url` is already scoped to its workspace;
        use it as given rather than composing one from the name.

        Entries also carry `run_id` and `version`, the run's stable ordinal
        (the web UI shows `v5`) - quote the version to a person. `folder=`
        plus `version=` lists that run; "output:<folder>/v5/<file>" names
        it. Other tools take `name`.

        `only_orphans=True` inverts the call: instead of files, it returns
        run directories holding nothing but their own bookkeeping
        (manifest.json, workflow.json, job.json) as `runs`, each
        `{name, mtime}` - a run whose output was deleted before
        `delete_output` could remove it by name, or one that failed before
        writing anything. `subfolder` does not apply in this mode. `name` is
        exactly what `delete_output` accepts, so clearing the backlog is
        list, then delete each name. A run that wrote any file at
        all - a text-shape prompt, a utility's side output - is not listed;
        this call only lists, so deciding whether a listed entry is actually
        junk before calling `delete_output` on it is still yours to make.

        `workspace` names the workspace for this one call without
        switching the session to it - the same pin `run_workflow`
        takes, so a job run into another workspace is reachable from
        here without leaving this one.

        `media=True` adds `duration_seconds` to audio/video entries - two
        takes sharing a basename are told apart by length, not size or
        mtime."""
        return catalog.list_gallery(
            client,
            limit=limit,
            subfolder=subfolder,
            only_orphans=only_orphans,
            workspace=workspace,
            folder=folder,
            version=version,
            media=media,
        )

    def get_gallery_metadata(
        name: str, envelope: bool = False, workspace: str | None = None
    ) -> dict:
        """Get the metadata embedded in a generated file: the exact
        workflow, arguments and seed that produced it - the definition,
        not a summary, so a result can be reproduced or a failed run's
        definition edited and re-run. Only an image embeds it; for audio
        and video `metadata` is null and `next` names
        `get_job_workflow(job_id)` when known, else a kept asset has no
        provenance. `media` itself carries duration, sample rate,
        channels, fps, size and level - the checks an agent that cannot
        listen makes on a deliverable; `media.shots` places a joined
        video's shots.
        `envelope=true` adds that level second by second
        (`media.envelope.rms_dbfs` / `peak_dbfs`), which says *where* in a
        track something is: a shot's last frame, a seam's hole, where a
        score goes quiet. Leave it off unless it's about position - a long
        track is a long list.

        `media.peak_dbfs` is what the job's `audio_no_headroom` (-0.5 dBFS,
        pre-encode) and `audio_clipped` (0.0 dBFS, post-encode) warnings
        read - see `normalize_audio` under "Video Processing" in the tasks
        guide. A mux emits only the second.

        `name` may be an "asset:" reference instead of a gallery name, and
        then it describes that input asset - how many frames a shot is,
        whether two shots share an fps, whether a score reaches the length
        of the cut it will lie under. Check before running: frame counts
        and rates are arguments the caller supplies, and a wrong one is a
        failed job or, worse, silence padded onto the end of a track.

        `workspace` pins this call to another workspace."""
        return catalog.get_gallery_metadata(
            client, name, envelope=envelope, workspace=workspace
        )

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
        """Get one guide from `list_guides`, or one section of it. Name the
        section - a guide runs to thousands of lines, and the headings in
        the listing are there so the right part can be asked for by name. A
        section name is matched loosely, so a heading copied approximately
        still resolves. Called without one, the answer is the guide's index
        (its opening and first section, with `sections` and `withheld`
        naming the rest), not the whole file."""
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
    tool(clear_memory, WRITES)

    # --------------------------------------------------------------- media

    def get_output_image(
        name: str,
        max_dimension: int = 768,
        workspace: str | None = None,
        crop: list[int] | None = None,
    ) -> list[ImageContent | TextContent]:
        """Look at a generated image, named as `list_gallery` or a job's
        manifest reports it. Use this to judge output quality - a run that
        succeeded can still have made the wrong picture. Downscaled to
        `max_dimension` on its longest side; the second part reports the
        before/after size, so a downscale is never silent.
        `crop` is `[x, y, width, height]` in the original's pixels,
        cut before the downscale.

        `workspace` pins this call to another workspace."""
        result = media.get_output_image(
            client, name, max_dimension=max_dimension, workspace=workspace, crop=crop
        )
        image = ImageContent(
            type="image", data=result["data"], mime_type=result["mime_type"]
        )
        telemetry = TextContent(
            type="text",
            text=(
                f"name: {result['name']}\n"
                f"original_size: {result['original_size']}\n"
                + (f"crop: {result['crop']}\n" if result["crop"] else "")
                + f"returned_size: {result['returned_size']}\n"
                f"bytes: {result['bytes']}"
            ),
        )
        return [image, telemetry]

    def get_output_audio(
        name: str,
        start: float | None = None,
        duration: float | None = None,
        workspace: str | None = None,
    ) -> list[AudioContent | TextContent]:
        """Listen to a generated soundtrack, named as `list_gallery` or a
        job's manifest reports it - an audio output, or a video's muxed
        track: own encoding when served whole, WAV when extracted or
        excerpted. No downscale exists for audio - a whole clip too
        large is refused; ask for a part with `start`/`duration` in
        seconds, per `get_gallery_metadata`'s envelope. The text part
        says what was cut. To *see* a video, `get_output_frames`. A
        text-only client confirms the *words* an output speaks by
        transcribing it instead: WORKFLOW_GUIDE's "The loop", step 6, in
        `get_guide("workflows", section="Authoring a workflow from an
        agent")`.

        `workspace` pins this call to another workspace."""
        result = media.get_output_audio(
            client, name, start=start, duration=duration, workspace=workspace
        )
        audio = AudioContent(
            type="audio", data=result["data"], mime_type=result["mime_type"]
        )
        lines = [f"name: {result['name']}", f"bytes: {result['bytes']}"]
        if result["duration_seconds"] is not None:
            lines.append(f"duration_seconds: {result['duration_seconds']}")
        if result["excerpt"]:
            e = result["excerpt"]
            lines.append(f"excerpt: {e['duration']}s from {e['start']}s of {e['of']}s")
        telemetry = TextContent(type="text", text="\n".join(lines))
        return [audio, telemetry]

    def get_output_frames(
        name: str,
        at: list[str | float] | None = None,
        seams: bool | list[int] | None = None,
        count: int | None = None,
        boundaries: list[int] | None = None,
        names: list[str] | None = None,
        max_dimension: int = 512,
        hear: float | None = None,
        workspace: str | None = None,
        crop: list[int] | None = None,
    ) -> list[ImageContent | AudioContent | TextContent]:
        """See a generated video as frames - no video content type exists
        over MCP. One selector: `count` (contact sheet), `at` (seconds or
        "frame:N"), or `seams` (true, or seam numbers from 1) for each
        join's frame pair, at a joined output's `media.shots`; else
        `boundaries` (each later shot's first frame) and `names`. Over budget, tiles shrink together.
        `hear=N` adds N seconds of soundtrack around each `at`.
        `crop` is `[x, y, width, height]` in the video's own source
        pixels, cut from every frame before any downscale, like
        `get_output_image`'s.

        `workspace` pins this call to another workspace."""
        result = media.get_output_frames(
            client,
            name,
            at=at,
            seams=seams,
            count=count,
            boundaries=boundaries,
            names=names,
            max_dimension=max_dimension,
            hear=hear,
            workspace=workspace,
            crop=crop,
        )
        parts = []
        for tile in result["tiles"]:
            parts.append(
                ImageContent(
                    type="image", data=tile["data"], mime_type=tile["mime_type"]
                )
            )
            if "audio" in tile:
                parts.append(
                    AudioContent(
                        type="audio",
                        data=tile["audio"]["data"],
                        mime_type=tile["audio"]["mime_type"],
                    )
                )
        lines = [
            f"name: {result['name']}",
            f"frame_count: {result['frame_count']}  fps: {result['fps']}",
        ]
        if result.get("crop"):
            lines.append(f"crop: {result['crop']}")
        fps = result["fps"]
        for tile in result["tiles"]:
            if tile.get("frames"):
                # a contact sheet: every cell, so each one can be located
                cells = ", ".join(
                    f"{frame} ({frame / fps:.2f}s)" if fps else str(frame)
                    for frame in tile["frames"]
                )
                where = f"frames: {cells}"
            else:
                where = f"frame {tile['frame']} @ {tile['seconds']:.2f}s"
            if tile.get("difference") is not None:
                where += f"  difference: {tile['difference']}"
            if tile.get("audio_error"):
                where += f"  hear: {tile['audio_error']}"
            lines.append(
                f"- {tile['label']}  {where}  [{tile['width']}x{tile['height']}]"
            )
        if result["downscaled_to"]:
            lines.append(
                f"downscaled_to: {result['downscaled_to']} (every tile, to fit the inline budget)"
            )
        if result.get("hear"):
            lines.append(f"hear: {result['hear']}s around each moment")
        if result.get("audio_truncated"):
            lines.append(
                "audio_truncated: some tiles' audio was skipped to stay within "
                "the response size budget"
            )
        parts.append(TextContent(type="text", text="\n".join(lines)))
        return parts

    def get_output_text(
        name: str, max_characters: int = 20000, workspace: str | None = None
    ) -> dict:
        """Read a text output - a prompt enhancement, or any step whose
        result is text/plain or JSON. Truncated to `max_characters`, and
        the reply says how long the file really was.

        `workspace` names the workspace for this one call without
        switching the session to it - the same pin `run_workflow`
        takes, so a job run into another workspace is reachable from
        here without leaving this one."""
        return media.get_output_text(
            client, name, max_characters=max_characters, workspace=workspace
        )

    def assess_output(
        name: str,
        probe: str | None = None,
        detail: bool = False,
        workspace: str | None = None,
    ) -> dict:
        """Measure a finished cut - seams, shot levels, sync - on the
        server, without queueing. Findings are places to look, not
        verdicts: check each with get_output_frames/get_output_audio.
        `probe` (analyze_shots, analyze_seams, analyze_sync_drift) returns
        one probe's full body; `detail` adds every probe's. Takes `asset:`."""
        return media.assess_output(
            client, name, probe=probe, detail=detail, workspace=workspace
        )

    def delete_output(
        name: str | None = None,
        workspace: str | None = None,
        job_id: str | None = None,
    ) -> dict:
        """Permanently remove one generated file from the output directory.
        Not recoverable (rerun the job to get it back), and any "output:"
        reference to it stops resolving; prefer `keep_output` if it is
        worth keeping. When it was the last media file of its run, the run
        directory goes with it, sidecars included. `name` may also be a run
        directory ("<workflow>/<run id>", the first two parts of a gallery
        name), which removes the whole run - the only handle on a run that
        failed before writing any media - or give `job_id` instead: the run
        that job wrote is removed whole, and the reply adds `job_id` and
        the resolved `run_dir`. Exactly one of the two; a job with no run
        directory, or unknown, is an error.

        `workspace` pins this call to another workspace without switching
        the session; a `job_id` delete with no `workspace` goes to
        the workspace the job ran in."""
        return media.delete_output(client, name, workspace=workspace, job_id=job_id)

    def download_output(
        name: str,
        destination: str | None = None,
        overwrite: bool = False,
        workspace: str | None = None,
    ) -> dict:
        """Save one output file to disk on the
        machine running the MCP server - for the stdio `dw-mcp` that is
        your own machine; for a
        `dw.serve --mcp` endpoint it is the GPU box, and this tool is not
        the way to get a file to where you are (use get_output_image /
        get_output_text for inline content, or the `url` that
        `list_gallery` reports for each entry, which already carries the
        workspace selector - do not build an /outputs URL by hand). This is
        also not how a generated file becomes an input for a later
        workflow: use `keep_output`, which links it inside the workspace
        under an "asset:" name, rather than writing into the server's asset
        directory behind the API's back. Unlike the inline tools, this works
        for any file type, streams the body straight to disk rather than
        buffering it, and returns no content to the conversation - only
        where it was saved. `destination` may be a
        full path or a directory; a '..' path segment in it is refused. An
        existing file at the resolved path is left alone unless
        `overwrite=True`. On the stdio `dw-mcp`, omitting `destination`
        saves into the current working directory under the output's own
        name. On a `dw.serve --mcp` endpoint the save happens on the server, and destination is required there -
        an omitted one is refused rather than dropped loose in the
        workspace root, where nothing can find or delete it later; use
        the `url` list_gallery reports, get_output_image/get_output_audio/
        get_output_frames for inline content, or keep_output to make it a
        named asset instead.

        `workspace` names the workspace for this one call without
        switching the session to it - the same pin `run_workflow`
        takes, so a job run into another workspace is reachable from
        here without leaving this one."""
        return media.download_output(
            client,
            name,
            destination=destination,
            overwrite=overwrite,
            workspace=workspace,
        )

    tool(get_output_image, READ_ONLY)
    tool(get_output_audio, READ_ONLY)
    tool(get_output_frames, READ_ONLY)
    tool(get_output_text, READ_ONLY)
    tool(assess_output, READ_ONLY)
    tool(download_output, OVERWRITES)
    tool(delete_output, DELETES)

    # ---------------------------------------------------------------- assets

    def list_assets(detail: bool = False) -> dict:
        """List the input media on the server, each with the "asset:"
        reference a workflow argument carries. Look here before asking for
        a file: what a workflow needs may already be there. Entries carry
        name, reference, kind, size and origin only - for one asset's
        duration, frame count, fps, sample rate or channels, pass its
        reference to `get_gallery_metadata`, which reads inputs as well as
        outputs. Pass detail=true for each entry's folder, mtime and url
        too, needed before naming a shared library's writable/read-only
        roots or opening the file's preview URL."""
        return assets.list_assets(client, detail=detail)

    def upload_asset(
        file_path: str | None = None,
        content: str | None = None,
        asset_name: str | None = None,
        shared: bool = False,
    ) -> dict:
        """Put an image, video or audio file into the server's asset
        library and get back the "asset:" reference to use in a workflow.
        Pass exactly one of `file_path` or `content`.

        `file_path` is read from the machine this MCP server runs on and
        pushed to the engine, so it is how an input reaches a dw.serve
        running somewhere else. When this MCP surface is served by
        dw.serve itself, "this machine" is the engine's own box, so
        `file_path` is confined to the directories it works in - a file
        that exists only on your own machine cannot be named this way.

        `content` is for exactly that case: the file's bytes, base64-encoded,
        sent inline in the call rather than read off any disk. Use it for a
        voice sample or small image that lives only on the machine you are
        running on, against a remote `dw.serve --mcp` endpoint with no
        filesystem in common with you. Capped at 4MB, well under
        `file_path`'s 200MB, because these bytes ride in the call itself.
        `asset_name` is required with `content`, since there is no file to
        take a name or extension from.

        Accepts the usual image, video and audio extensions. Reference the
        result rather than a path: a path on this machine means nothing to
        the server. Pass `asset_name` to store it under a readable name
        ("cast/priya-voice.wav", folders allowed) - without one (when using
        `file_path`) the stored name is random, and a set of related inputs
        cannot be told apart in the workflows that carry them. Pass
        `shared=true` to put it in the library every workspace shares
        rather than this session's own - where a recurring cast belongs,
        since a workspace's own assets are invisible from the next
        workspace."""
        return assets.upload_asset(
            client,
            file_path=file_path,
            content=content,
            asset_name=asset_name,
            shared=shared,
        )

    def keep_output(
        name: str,
        asset_name: str | None = None,
        overwrite: bool = False,
        shared: bool = False,
        workspace: str | None = None,
    ) -> dict:
        """Keep a generated file as an input asset under a stable "asset:"
        name, so later workflows can rely on it - a run's own name moves
        ("latest") or breaks when outputs are pruned. This is the step
        between a render you liked and the next stage that conditions on
        it. `name` is a gallery name; `asset_name` defaults to the file's
        own. The copy happens on the server, inside the workspace: nothing
        is downloaded or re-uploaded. Pass `shared=true` to keep it in the
        library every workspace shares instead - where something a later
        piece in its own workspace has to reach belongs.

        `workspace` names the workspace for this one call without
        switching the session to it - the same pin `run_workflow`
        takes, so a job run into another workspace is reachable from
        here without leaving this one."""
        return assets.keep_output(
            client,
            name,
            asset_name=asset_name,
            overwrite=overwrite,
            shared=shared,
            workspace=workspace,
        )

    def delete_asset(name: str) -> dict:
        """Permanently remove one file from the asset library, by the name
        `list_assets` reports (without the "asset:" prefix). Not
        recoverable, and any workflow still carrying that reference stops
        loading. Deletes from whichever library holds it - this
        workspace's own before the shared one, the order an "asset:"
        reference resolves in; one from a read-only examples library is
        refused."""
        return assets.delete_asset(client, name)

    tool(list_assets, READ_ONLY)
    tool(upload_asset, WRITES)
    tool(keep_output, WRITES)
    tool(delete_asset, DELETES)

    # ------------------------------------------------------------ workspaces

    def list_workspaces(detail: bool = False) -> dict:
        """List the server's workspaces and say which one this session is
        working in. Each has its own workflows, assets and outputs; the
        stored prompt library is shared by all of them, and so is the
        shared asset library that `upload_asset(shared=true)` and
        `keep_output(shared=true)` write into - which is how a recurring
        cast stays reachable from the workspace the next piece is made
        in. Entries carry name, default and usage (files/bytes) only; pass
        detail=true for each entry's full folder paths (workflows, assets,
        outputs, prompts, common_assets)."""
        return workspaces.list_workspaces(client, detail=detail)

    def use_workspace(name: str) -> dict:
        """Work in a different workspace for the rest of this session - every
        later call reads and writes there. Use this to keep your work out of
        another agent's namespace, rather than sharing the default one."""
        return workspaces.use_workspace(client, name)

    def create_workspace(name: str, use: bool = False) -> dict:
        """Create a workspace on the server. It gets its own workflows,
        assets and outputs and shares the one prompt library. The name is a
        single path segment and cannot be one of the reserved folder names
        (workflows, prompts, assets, outputs, exports, common). Pass use=true to switch this
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
        workflow: dict | str | None = None,
        name: str | None = None,
        inline_workflow: dict | str | None = None,
        workflow_path: str | None = None,
        workspace: str | None = None,
        arguments: dict | None = None,
    ) -> dict:
        """Check a workflow against the schema and against real pipeline
        signatures. Free and instant - run it before run_workflow.
        Give exactly one of `workflow` or `name` (a stored workflow as
        `list_workflows` reports it); `run_workflow`'s `inline_workflow`
        and `workflow_path` spellings are accepted here too. `workflow` may
        be a JSON-encoded string. Every error comes back at once, each with
        its JSON path. `workspace` pins this one call to another workspace
        without switching the session.

        A valid answer carries `plan`: what will execute for these
        arguments - `estimate.minutes` and its `basis` (`observed`,
        `per_entry`, `catalog`, `derived`, `other_device` or `unknown` -
        how to quote each is WORKFLOW_GUIDE's "The loop", step 4), each
        `downloads_required` entry as its own cost line, and
        `steps`/`list_entries` for how many members the list produced.
        `plan` is null when it could not be built; the verdict stands.

        Pass the same `arguments` you will pass to `run_workflow` and they
        are checked too: an undeclared name, a value that will not coerce,
        and an `asset:`/`prompt:`/`output:` reference naming nothing this
        workspace can reach, each at `arguments.<name>`.
        `checked_arguments` says whether your values or only the stored
        defaults were checked. A value outside a bound the workflow
        declares is an error here rather than a failed run; one the workflow rounds up
        comes back as a warning naming what it becomes.

        Also checked: an unwritable `result.subfolder` or `file_base_name`,
        after `for_each` expansion; and each sub-workflow step - an
        unreachable `workflow.path`, the composed workflow in turn, a
        composition cycle, and (as a warning) an argument passed down that
        it declares no variable for."""
        return authoring.validate_workflow(
            client,
            workflow=workflow,
            name=name,
            inline_workflow=inline_workflow,
            workflow_path=workflow_path,
            workspace=workspace,
            arguments=arguments,
        )

    def save_workflow(
        name: str,
        workflow: dict | str | None = None,
        patch: dict | str | None = None,
    ) -> dict:
        """Save a workflow to the server's writable workflow directory,
        overwriting any existing workflow of that name there. Validate it
        first. A name that currently resolves to a read-only source (an
        examples directory) is not overwritten - the copy lands in the
        writable directory and shadows it from then on, which is how an
        example gets adapted without being damaged. `name` may include
        folders.

        Give exactly one of `workflow` (the full document) or `patch` for a
        small, targeted edit: a JSON Merge Patch (RFC 7396) merged onto the
        currently stored definition, so bumping one argument means sending
        just that argument rather than the whole document -
        `{"variables": {"num_images_per_prompt": 4}}` rather than the whole
        workflow. A patch key set to `null` deletes that key from the
        stored document. A list is replaced whole, never merged - a merge
        patch has no notion of list position, so changing one `shots` entry
        still means sending the whole `shots` list. Either may also be
        given as a JSON-encoded string, which is parsed before saving; a
        string that fails to parse is reported as invalid JSON rather than
        as a type mismatch.

        A workflow stored for reuse should mark each saving step's
        `result.subfolder` - `final` for the step whose output the user will
        be shown, `intermediate` for the rest - so a later consumer can tell
        the deliverable from the scratch files without knowing the workflow."""
        return authoring.save_workflow(client, name, workflow=workflow, patch=patch)

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

    def list_prompts(
        tag: Optional[str] = None,
        intended_model: Optional[str] = None,
        include_text: bool = False,
    ) -> dict:
        """List the stored prompts - the worked examples a workflow reaches
        by writing "prompt:name" or "prompt:folder/name". Each entry carries
        its `description`, `intended_model`, `tags` and the size of its text;
        `get_prompt` returns the text itself. This is where the caption a
        model was trained on is already written out, so read the exemplar
        for the family you are about to run rather than inventing the
        format: `intended_model` narrows to one family, as the listing
        reports it, and `tag` to one label. `include_text=true` returns every body, which for the whole
        library is more than a client will accept - filter first."""
        return prompts.list_prompts(
            client,
            tag=tag,
            intended_model=intended_model,
            include_text=include_text,
        )

    def get_prompt(name: str) -> dict:
        """Get one stored prompt's full definition - its text, description,
        intended model and tags - by a name from `list_prompts`. The prompt
        library is shared by every workspace on the server."""
        return prompts.get_prompt(client, name)

    def get_prompt_schema() -> dict:
        """Get the JSON schema every stored prompt must satisfy. Check this
        before writing one, as you would get_schema before a workflow."""
        return prompts.get_prompt_schema(client)

    def save_prompt(name: str, prompt: dict | str) -> dict:
        """Save a prompt to the library, overwriting any prompt of that
        name. Its `text` may not itself begin with a reference prefix
        (variable:, previous_result:, constant:, asset:, output:, prompt:)
        - the server refuses that to prevent a reference resolving twice.
        The library is shared by every workspace on this server. `prompt`
        may also be a JSON-encoded string; a parse failure is reported as
        invalid JSON, not a type mismatch."""
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
        This costs time on the engine: it queues a real job, and the engine
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
        inline_workflow: dict | str | None = None,
        workflow: dict | str | None = None,
        name: str | None = None,
        arguments: dict | None = None,
        acknowledged_cost: bool | dict = False,
        workspace: str | None = None,
        wait_seconds: int = 0,
    ) -> dict:
        """Queue a workflow for generation. This costs GPU time: a run
        occupies the machine for minutes and the engine runs one job at a
        time. Tell the user what will run and get their go-ahead, then pass
        acknowledged_cost as below. Returns as soon as the job is queued;
        follow it with `wait_for_job`, then `get_job` for the manifest - or
        fold that first wait in with `wait_seconds` above 0, which waits on
        the job exactly as `wait_for_job(job_id,
        timeout_seconds=wait_seconds)` would ({cap}s cap per call) and adds
        its fields to the result (`still_running`, `waited_seconds`,
        `timeout_*`, the slim `job`). If the cap covers the job's
        runtime one call is enough; on `still_running: true` call
        `wait_for_job` again. Give exactly one of `workflow_path` - a
        catalog name from `list_workflows`, with or without .json, or a
        path on the server - or `inline_workflow`, a full definition
        nothing stored covers; `validate_workflow` calls these `name` and
        `workflow`, and both tools accept both spellings.
        `inline_workflow`/`workflow` may also be a JSON-encoded string; a
        parse failure is reported as invalid JSON, not a type mismatch.
        `arguments` overrides the workflow's variables by name. `workspace` pins this
        call to another workspace without switching the session (where its
        `output:`/`asset:` references live).

        Bind the acknowledgement to what you quoted: pass
        {"fingerprint": plan.fingerprint, "minutes": plan.estimate.minutes,
        "downloads": [...the non-null repos in plan.downloads_required]} from
        the validate plan; the server refuses with 409, naming the new
        plan, if the run's shape changed since. Bare true is for a plan
        that was null."""
        return diagnose.run_workflow(
            client,
            workflow_path=workflow_path,
            inline_workflow=inline_workflow,
            workflow=workflow,
            name=name,
            arguments=arguments,
            acknowledged_cost=acknowledged_cost,
            workspace=workspace,
            wait_seconds=wait_seconds,
        )

    # The cap is a number a caller paces against, so the description
    # states it (as wait_for_job's does, below). replace rather than
    # format: the docstring spells out a literal {fingerprint, ...} dict.
    if run_workflow.__doc__:  # absent under python -OO
        run_workflow.__doc__ = run_workflow.__doc__.replace(
            "{cap}", str(diagnose.MAX_WAIT_SECONDS)
        )

    def get_job(job_id: str) -> dict:
        """Get a job's status, argument warnings, output manifest, error and
        traceback. The manifest names each step's files the way
        `get_output_image`, `download_output` and `keep_output` take them,
        and each entry's `subfolder` says what kind of output the step
        declared - by convention `final` is the deliverable, `intermediate`
        the scratch work, and '' a step that said nothing. A step served
        from the step cache is marked `reused` and reports the earlier run's
        files. When a job failed, the error and traceback here are what to
        read before changing anything. `acknowledged` says which form of
        cost acknowledgement queued the job (`none`, `boolean`, `bound`) and
        `acknowledged_cost` is the bound `{fingerprint, minutes, downloads}`
        when there was one."""
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

    def get_job_events(
        job_id: str,
        after: int = -1,
        limit: int = 200,
        kinds: list[str] | None = None,
    ) -> dict:
        """Get a page of a job's progress events - phase transitions, denoise
        steps, memory readings and log lines. `after` is exclusive: pass back
        the previous call's `last_seq` to continue. Each event's `at` is
        seconds since the job started, so where a step's time went is the
        difference between two events. For 'is it still moving?' the
        `progress` block on get_job/wait_for_job is cheaper than a page of
        events. `kinds` (e.g. `["log", "warning"]`) restricts the page to
        events whose `event` or `kind` is one of them, so `["phase_stall"]`
        or `["audio_clipped"]` selects one warning type - `memory` events
        otherwise dominate the payload.

        A `kind: "phase_stall"` entry is a watchdog notice, not progress - it
        fires every ~30s a phase goes quiet, not evidence of a hang by
        itself. It carries `seconds_since_last_progress` and
        `seconds_since_phase_start`; some models are silent for minutes
        normally - check the model's guide before treating one as a fault."""
        return diagnose.get_job_events(
            client, job_id, after=after, limit=limit, kinds=kinds
        )

    def wait_for_job(job_id: str, timeout_seconds: int = 20) -> dict:
        """Block until a job finishes, instead of polling get_job or
        get_job_events by hand: returns as soon as its status is succeeded,
        failed or cancelled, or with still_running: true when
        timeout_seconds elapses first, so you can call again. Queues
        nothing, so no acknowledged_cost.

        One call blocks for at most {cap} seconds, whatever timeout_seconds
        asks for - this deployment's cap, set for the tool-call budget the
        client holds open; a larger value is clamped, not honoured, so
        budget one call per {cap}s of the job, and one call is enough when
        {cap} covers its runtime. Every reply says which happened:
        waited_seconds, timeout_requested_seconds, timeout_applied_seconds
        and timeout_capped.

        Returns a slim job - status, warnings, error, the manifest once
        finished - without the arguments (get_job has those). A running job
        also carries `progress`: step, phase, and
        `denoise_step`/`denoise_total_steps`, null until the denoise loop
        starts. Tell a slow run from a stuck one by whether
        `denoise_step` has moved since a poll minutes ago, not by silence:
        a video reference's lead-in can run many minutes emitting nothing,
        and denoise gaps are uneven under a transformer block cache - both
        normal. If you're also reading get_job_events, a `phase_stall`
        entry there is the same silence being narrated, not a fault or a
        sign of progress - it repeats every ~30s the phase stays quiet, so
        neither seeing one nor watching its event_count climb tells you
        anything `denoise_step` doesn't already say better. Full diagnosis,
        and why `denoise_total_steps` can read one less than asked, in
        WORKFLOW_GUIDE's "The loop", step 5."""
        return diagnose.wait_for_job(client, job_id, timeout_seconds=timeout_seconds)

    # The cap is a number a caller paces against, so the description states
    # it rather than saying "well under a generation's runtime".
    if wait_for_job.__doc__:  # absent under python -OO
        wait_for_job.__doc__ = wait_for_job.__doc__.format(
            cap=diagnose.MAX_WAIT_SECONDS
        )

    def cancel_job(job_id: str) -> dict:
        """Ask a queued or running job to stop. Cooperative: a running job
        stops at the next step or denoise-step boundary, not instantly.
        Deliberately not gated - it ends a cost rather than starting one."""
        return diagnose.cancel_job(client, job_id)

    def rerun_job(
        job_id: str, acknowledged_cost: bool | dict = False, new_seed: bool = False
    ) -> dict:
        """Queue a fresh job from a previous job's stored specification. This
        costs GPU time: a rerun is a run - it occupies the machine for
        minutes and the engine runs one job at a time. Tell the user what
        will run and get their go-ahead, then pass acknowledged_cost.

        Pass new_seed=true for a different image: a workflow that pins its
        seed reruns to the same pixels, and the step cache serves that whole
        run from the earlier one's files (marked `reused`) in a fraction of a
        second rather than generating anything.

        `acknowledged_cost` takes the same bound form as run_workflow; a
        fresh seed never changes the fingerprint, so the original plan still
        binds a new_seed rerun."""
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
        THE DIRECTORY IS ON THE MACHINE RUNNING THE SERVER, not on yours.

        `auth_required` says whether opening the zip needs this server's
        bearer token, a token you cannot attach to someone else's browser
        or tooling. When it is false, fetch open_url yourself and unpack
        it into exports/ under the session's working directory - it is
        the user's deliverable, not a temp file; the archive already
        unpacks into one folder named after the job id, so do not create that folder first.
        When it is true, do NOT fetch it: hand open_url to the person and let them open it
        (`next` says whether it is already absolute or needs the server's
        address told to them). Individual results stay reachable inline
        via get_output_image/get_output_audio/get_output_frames either
        way. Refuses a job that is still running; refuses an existing
        export unless overwrite=true."""
        return exports.export_job(client, job_id, overwrite=overwrite)

    tool(get_job, READ_ONLY)
    tool(get_job_workflow, READ_ONLY)
    tool(get_job_events, READ_ONLY)
    tool(wait_for_job, READ_ONLY)
    for fn in (run_workflow, cancel_job, rerun_job, move_job, export_job):
        tool(fn, WRITES)

    # -------------------------------------------------------------- models

    def download_model(repo_id: str, acknowledged_cost: bool = False) -> dict:
        """Fetch a model repo into the Hugging Face cache. This costs disk
        and bandwidth: a model repo is commonly tens of gigabytes. Check
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
        """Delete every cached revision of one model repo. This is not
        recoverable: getting the model back means downloading it again. Tell
        the user which repo and how much it frees, get their go-ahead, then
        pass acknowledged_cost=true. Refused while a job or download is
        active."""
        return models.delete_model(client, repo, acknowledged_cost=acknowledged_cost)

    def get_diffusers_state() -> dict:
        """Get the installed diffusers version and any update in flight."""
        return models.get_diffusers_state(client)

    def update_diffusers(acknowledged_cost: bool = False) -> dict:
        """Upgrade diffusers to GitHub HEAD. This can break the install: it
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
