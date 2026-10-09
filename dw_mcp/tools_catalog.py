"""Tool classes for the catalog, engine, server and model groups.

A tool is a method; its docstring is the description an agent reads, so
it is product surface. `server.build_server` registers the bound methods.
This module and `server` are the only ones that import the MCP SDK."""

from typing import Literal, Optional

from dw_mcp import catalog, guides, models, workspaces


class CatalogTools:
    """The workflow, engine, server and gallery-read tools."""

    def __init__(self, client):
        self.client = client

    def list_workflows(
        self,
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
            self.client,
            shape=shape,
            traits=[t.strip() for t in (traits or "").split(",") if t.strip()],
            configures=configures,
            include_models=include_models,
        )

    def get_workflow(self, name: str, variables_only: bool = False) -> dict:
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
        return catalog.get_workflow(self.client, name, variables_only=variables_only)

    def get_schema(self, section: str | None = None) -> dict:
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
        return catalog.get_schema(self.client, section=section)

    def list_pipelines(
        self,
    ) -> dict:
        """List every diffusers pipeline class this installation provides.
        These are the names a step's `component_type` can take - and the
        list is this installation's, so a pipeline from a newer diffusers
        will not be here until it is updated."""
        return catalog.list_pipelines(self.client)

    def get_pipeline_signature(self, name: str) -> dict:
        """Get a pipeline's real call arguments. Check this before proposing
        pipeline arguments - a plausible-looking argument that the pipeline
        does not accept is the most common workflow bug."""
        return catalog.get_pipeline_signature(self.client, name)

    def list_classes(
        self,
        kind: Literal["pipelines", "models", "schedulers", "quantization"],
    ) -> dict:
        """List class names of one kind: pipelines, models, schedulers, or
        quantization. These are the names a workflow's `component_type`,
        `scheduler_type` or `config_type` can take."""
        return catalog.list_classes(self.client, kind)

    def get_class(
        self, name: str, target: Literal["init", "call", "load"] = "init"
    ) -> dict:
        """Get a class's argument schema, from whichever entry point a
        workflow reaches it by: `init` reads the constructor (quantization
        configs, schedulers, models built from named arguments), `call`
        reads __call__ (a pipeline's `arguments`), `load` reads
        from_pretrained plus the curated loading knobs, which is what a
        component's `from_pretrained_arguments` can carry."""
        return catalog.get_class(self.client, name, target=target)

    def list_tasks(
        self,
    ) -> dict:
        """List every task command a workflow's task step can name - the
        non-pipeline work: upscaling, face restoration, ControlNet
        preprocessors, captioning, frame interpolation, video and audio
        handling. Check here before assuming something needs a pipeline."""
        return catalog.list_tasks(self.client)

    def get_task(self, command: str) -> dict:
        """Get a task command's argument schema, read from its real
        implementation signature. The counterpart of
        `get_pipeline_signature` for a task step."""
        return catalog.get_task(self.client, command)

    def list_models(
        self,
    ) -> dict:
        """List what the Hugging Face model cache holds, largest first."""
        return catalog.list_models(self.client)

    def get_memory(self, device: str | None = None) -> dict:
        """Get VRAM and RAM statistics. Check this first when a job fails
        with an out-of-memory error. `workers` has one reading per card (the
        top level repeats the first's); `device` ('cuda:1') asks for one.

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
        and `age_seconds` how old it is. A cached reading taken while a job
        loads a model understates what is resident, so ask again when the
        card is idle rather than comparing it against a live figure.

        `info.step_cache` is the step cache's own accounting: `entries`,
        `retained_bytes` against `max_retained_bytes`."""
        return catalog.get_memory(self.client, device)

    def clear_memory(self, device: str | None = None) -> dict:
        """Drop every loaded pipeline and the step cache on each idle card
        (`device`'s alone when named), so a seeded workflow regenerates on
        its next run. A card running a job is skipped (`workers` says
        which); refused with a 409 when no card asked about is idle - wait
        and retry. No model process resident succeeds with a null
        `info`."""
        return catalog.clear_memory(self.client, device)

    def get_health(
        self,
    ) -> dict:
        """Check that the server is alive, and see what answered: its
        version and accelerator and how many jobs are queued. The engine
        runs one job per GPU: `workers` lists each card (`device`, `name`,
        `vram_gb`, `alive`) with its `current_job`; the top-level
        `current_job` is the longest-running.

        `worker_alive: false` on an otherwise healthy server (`status: ok`)
        is the normal idle state, not a fault - workers are on-demand
        subprocesses that start with the next job, so none is resident
        before the first run or after a memory clear."""
        return catalog.get_health(self.client)

    def get_server_info(
        self,
    ) -> dict:
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
        return workspaces.server_info(self.client)

    def list_jobs(
        self, limit: int = 20, status: str | None = None, workspace: str | None = None
    ) -> dict:
        """List queued, running and recent jobs, newest first, with their
        status and queue position. The ids here are what `get_job`,
        `wait_for_job`, `get_job_events`, `cancel_job`, `rerun_job` and
        `move_job` take - including jobs started before this session or in
        the browser.

        `limit` is the newest N (20 by default); `total` reports how many
        matched, so truncation shows. `status` narrows to one state or a
        comma-separated set of them - queued, running, succeeded, failed,
        cancelled. `workspace` lists one workspace's jobs; without it, a
        named workspace lists its own and the default lists every job on
        the server. Each job carries `device`, the card it ran or runs on
        (e.g. `cuda:1 NVIDIA GeForce RTX 3090`, null before multi-GPU), and
        `acknowledged` (`none`, `boolean` or `bound`), the form of cost
        acknowledgement that queued it."""
        return catalog.list_jobs(
            self.client, limit=limit, status=status, workspace=workspace
        )

    def list_gallery(
        self,
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
        `output:` reference in a later workflow is built from. Each entry
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
            self.client,
            limit=limit,
            subfolder=subfolder,
            only_orphans=only_orphans,
            workspace=workspace,
            folder=folder,
            version=version,
            media=media,
        )

    def get_gallery_metadata(
        self, name: str, envelope: bool = False, workspace: str | None = None
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

        `findings` lists each level problem the server measured - full
        scale, near silence - with its threshold and the fix.

        `name` may be an `asset:` reference instead of a gallery name, and
        then it describes that input asset - how many frames a shot is,
        whether two shots share an fps, whether a score reaches the length
        of the cut it will lie under. Check before running: frame counts
        and rates are arguments the caller supplies, and a wrong one is a
        failed job or, worse, silence padded onto the end of a track.

        `workspace` pins this call to another workspace."""
        return catalog.get_gallery_metadata(
            self.client, name, envelope=envelope, workspace=workspace
        )

    def list_guides(
        self,
    ) -> dict:
        """List the documentation the engine serves: each guide's
        name, what it covers, and its section headings. Read this when a
        request is open-ended enough that no catalog entry obviously
        answers it - a request names a subject ("a lego movie trailer"),
        while the catalog and these guides are written in shapes
        (multi-shot video, cuts, a consistent cast, narration over
        B-roll), and the section headings are where the two get matched
        up. Cheaper than guessing: reading a section costs a fraction of
        one wrong run."""
        return guides.list_guides(self.client)

    def get_guide(self, name: str, section: str | None = None) -> dict:
        """Get one guide from `list_guides`, or one section of it. Name the
        section - a guide runs to thousands of lines, and the headings in
        the listing are there so the right part can be asked for by name. A
        section name is matched loosely, so a heading copied approximately
        still resolves, including a `###` subsection not in `sections`
        (e.g. "for_each"). Called without one, the answer is the guide's index
        (its opening and first section, with `sections` and `withheld`
        naming the rest), not the whole file."""
        return guides.get_guide(self.client, name, section=section)


class ModelTools:
    """Model downloads, deletion and the diffusers update."""

    def __init__(self, client):
        self.client = client

    def download_model(self, repo_id: str, acknowledged_cost: bool = False) -> dict:
        """Fetch a model repo into the Hugging Face cache. This costs disk
        and bandwidth: a model repo is commonly tens of gigabytes. Check
        list_models first - it may already be cached. Tell the user what you
        are about to fetch and get their go-ahead, then pass
        acknowledged_cost=true. Returns as soon as the download starts; poll
        list_downloads for progress."""
        return models.download_model(
            self.client, repo_id, acknowledged_cost=acknowledged_cost
        )

    def list_downloads(
        self,
    ) -> dict:
        """List model downloads the server is running or recently ran."""
        return models.list_downloads(self.client)

    def cancel_download(self, download_id: str) -> dict:
        """Ask a running model download to stop. Partial files stay in the
        cache and resume if it is retried."""
        return models.cancel_download(self.client, download_id)

    def delete_model(self, repo: str, acknowledged_cost: bool = False) -> dict:
        """Delete every cached revision of one model repo. This is not
        recoverable: getting the model back means downloading it again. Tell
        the user which repo and how much it frees, get their go-ahead, then
        pass acknowledged_cost=true. Refused while a job or download is
        active."""
        return models.delete_model(
            self.client, repo, acknowledged_cost=acknowledged_cost
        )

    def get_diffusers_state(
        self,
    ) -> dict:
        """Get the installed diffusers version and any update in flight."""
        return models.get_diffusers_state(self.client)

    def update_diffusers(self, acknowledged_cost: bool = False) -> dict:
        """Upgrade diffusers to GitHub HEAD. This can break the install: it
        installs an untagged development build that workflows running today
        may not survive, and this tool cannot undo it. Report the current
        version, explain why the update is worth it, get the user's
        go-ahead, then pass acknowledged_cost=true. Refused while a job is
        running or queued."""
        return models.update_diffusers(self.client, acknowledged_cost=acknowledged_cost)
