"""Read-only tools: everything an agent can look at without spending GPU
time. Each is a pass-through - the API's shapes are already the ones the
web UI consumes, and reshaping them here would only add a second thing to
keep in sync."""

from dw_mcp.client import api_path


def list_workflows(
    client, shape=None, traits=None, configures=None, include_models=False
):
    """Workflow names the server can reach, in the compact view: summary,
    shape, traits, cost, output kinds and variable names per workflow -
    what choosing one needs and nothing that reading one needs. Templates
    only unless `include_models` or `configures` asks for the model configs
    of one template. `get_workflow` has the full definition.

    Called with no filter at all, each entry is cut to its summary and
    shape (`view: "summary"`), because the whole catalog in full detail is
    ~6.8k tokens for a question that is really "which shape do I want"
    (#101). Pass `shape` - which the instructions ask for anyway - and the
    entries come back whole.

    `cost_basis` in the answer says what a `cost` is: `curated` means a
    maintainer measured it once, on the devices the entry names, and wrote
    it into the workflow. Nothing derives one from this server's own job
    history, so `cost: null` means nobody wrote a figure down - not that
    the run is cheap, and not that this box has never run it.

    `observed_minutes` / `observed_runs`, when present, are the other kind
    of number: what *this* box's own finished runs of that workflow actually
    took, cold (model load included, so comparable to a curated `cost`), and
    how many runs stand behind the figure. Derived, never a substitute for
    `cost` - a maintainer's claim on a named card and this machine's last
    week are different things. `get_workflow(variables_only=true)` carries
    the whole block: the cold/warm split, the argument values the figure is
    for, and `since`."""
    params = {"view": "compact"}
    if shape:
        params["shape"] = shape
    if traits:
        params["traits"] = (
            ",".join(traits) if isinstance(traits, (list, tuple)) else traits
        )
    if configures:
        params["configures"] = configures
    if include_models:
        params["include_models"] = "true"
    answer = client.get_json("/api/workflows", params=params)
    if not (shape or traits or configures or include_models):
        return _summarised(answer)
    return answer


# What a summarised entry keeps: enough to choose a shape to ask about,
# nothing that reading one needs
SUMMARY_FIELDS = ("summary", "shape")


def _summarised(answer):
    """The unfiltered listing, each entry cut to its summary and shape.

    The whole catalog in full compact detail is ~6.8k tokens - variables,
    traits, cost and list fields for every workflow on the server, when
    the question the call answers is "which of these is the shape I
    want" (#101). Asking with a `shape` still gets the full entries, and
    the note says so, so nothing is unreachable.
    """
    details = answer.get("details")
    if not isinstance(details, dict):
        return answer
    summarised = {
        name: {key: detail.get(key) for key in SUMMARY_FIELDS if key in detail}
        for name, detail in details.items()
        if isinstance(detail, dict)
    }
    return {
        **answer,
        "details": summarised,
        "view": "summary",
        "note": (
            "Summarised because no `shape` was given: each entry is its "
            "summary and shape only. Call list_workflows(shape=...) for "
            "the full entries - variables, traits, cost and list fields - "
            "of the shape you want."
        ),
    }


def get_workflow(client, name, variables_only=False):
    """One workflow's full JSON definition.

    With variables_only=True, just its variables and what they default to -
    confirming that a stored workflow's audio_bleed_ms is 1800 otherwise
    means pulling the whole definition, quantization blocks and all, to
    read one integer (2026-09-11). Long defaults come back cut to their
    first 200 characters, with the names of the cut ones in `truncated`,
    including strings inside a list default, named like `shots[0].prompt`.
    """
    if variables_only:
        return client.get_json(api_path("api", "workflows", name, "variables"))
    return client.get_json(api_path("api", "workflows", name))


def get_schema(client, section=None):
    """The workflow JSON schema every definition is validated against.

    Whole it is ~8.6k tokens, so name the part you need: `section` takes
    `steps`, `pipelines`, `tasks`, `result`, `variables` or
    `configuration` and answers that fragment plus `elsewhere`, which says
    which section holds each definition it still references."""
    params = {"section": section} if section else None
    return client.get_json("/api/schema", params=params)


def list_pipelines(client):
    """Every diffusers pipeline class this install exports."""
    return client.get_json("/api/pipelines")


def get_pipeline_signature(client, name):
    """A pipeline's real __call__ arguments - check before proposing a fix."""
    return client.get_json(api_path("api", "pipelines", name))


def list_classes(client, kind):
    """Class names of one kind: pipelines, models, schedulers, quantization."""
    return client.get_json("/api/classes", params={"kind": kind})


def get_class(client, name, target="init"):
    """A class's argument schema. target: init, call, or load."""
    return client.get_json(api_path("api", "classes", name), params={"target": target})


def list_tasks(client):
    """Every task command a workflow's task step can name."""
    return client.get_json("/api/tasks")


def get_task(client, command):
    """A task command's argument schema."""
    return client.get_json(api_path("api", "tasks", command))


def list_models(client):
    """What the Hugging Face hub cache holds, largest repo first."""
    return client.get_json("/api/models")


def get_memory(client):
    """Worker VRAM/RAM stats - the first thing to check on an OOM."""
    return client.get_json("/api/memory")


def clear_memory(client):
    """Drop every loaded pipeline and the step cache, freeing VRAM/RAM
    immediately rather than waiting for the next job to evict one model
    for another. Refused with a 409 while a job is running or queued -
    the queue is FIFO, so retry once it finishes rather than expecting
    this call to wait for it."""
    return client.post_json("/api/memory/clear")


def get_health(client):
    """Server liveness, plus what answered: version, device, whether a
    model process is currently resident, the job running now and how many
    are queued. `worker_alive: false` is the normal idle state on a server
    that hasn't run a job yet - not a degraded server."""
    return client.get_json("/api/health")


def get_server_info(client):
    """What this installation can do and where it keeps things: the
    accelerator, the dw version, the workflow/output/prompt directories,
    and how the server is reached."""
    return client.get_json("/api/server")


def list_jobs(client, limit=20, status=None, workspace=None):
    """The live queue plus recent history, newest first and bounded.

    Bounded because the unbounded answer was a dead tool: a server with a
    few months of history spilled 176 entries past the client's tool-result
    limit, and the call failed before a single id could be read
    (2026-09-11). Newest first for the same reason the limit exists - the
    job worth looking at is almost always the last one.

    `total` is what matched before the cut, so a caller can tell a bounded
    answer from a complete one; raise `limit` or narrow with `status` /
    `workspace` to see the rest."""
    params = {}
    if limit is not None:
        params["limit"] = limit
    if status:
        params["status"] = (
            ",".join(status) if isinstance(status, (list, tuple)) else status
        )
    if workspace:
        params["workspace"] = workspace
    body = client.get_json("/api/jobs", params=params or None)
    # The API answers oldest first - the order the web UI's list renders in.
    # An agent reads the top of a tool result, so the newest job belongs there
    jobs = list(reversed(body.get("jobs") or []))
    total = body.get("total", len(jobs))
    answer = {"jobs": jobs, "returned": len(jobs), "total": total}
    if total > len(jobs):
        answer["truncated"] = True
        answer["next"] = (
            f"{total - len(jobs)} older jobs were not listed - raise `limit`, "
            "or narrow with `status` or `workspace`."
        )
    return answer


def list_gallery(
    client,
    limit=50,
    subfolder=None,
    only_orphans=False,
    workspace=None,
    folder=None,
    version=None,
    media=False,
):
    """Generated media in the output directory, newest first. `subfolder`
    narrows to one in-run subfolder ('final', 'intermediate', '' for files
    at a run's root); None means every file. `folder` narrows to one
    workflow and `version` to one run's ordinal, so the two together list
    the run a person calls "v4".

    `only_orphans=True` inverts the call: instead of files, it returns run
    directories holding nothing but their own bookkeeping (manifest.json,
    workflow.json, job.json) as `runs`, each `{name, mtime}` - a run whose
    output was deleted before `delete_output` could remove it by name, or
    one that failed before writing anything. A run that wrote any file at
    all, a text-shape prompt or a utility's side output included, is not
    listed. `subfolder` does not apply in this mode. `name` is exactly what
    `delete_output` accepts, so clearing one is list, then delete (#170).

    Each file entry also carries `label`, a bare display basename for a UI
    grid - it is not a valid reference on its own (two runs can write the
    same basename) and is not accepted by `get_gallery_metadata` or
    `delete_output`. Pass `name` to those, not `label`.

    `version` is that run's ordinal among the workflow's runs, and `run_id`
    the run it came from. The version is what to quote to a person - the web
    UI labels the same file `v5` - and is stable: it is assigned when the
    run opens and a deleted sibling leaves a gap rather than renumbering
    what is left - as does a run that failed, or reused every step from
    the cache, and so wrote nothing to list. Null under the flat output
    layout, which has no runs.

    `media=True` adds `duration_seconds` to each audio/video entry in the
    page returned, probed the way `get_gallery_metadata` measures a file -
    enough to pick between two takes without one metadata call per
    candidate. Off by default; a plain call carries no `duration_seconds`."""
    params = {"limit": limit}
    if subfolder is not None:
        params["subfolder"] = subfolder
    if folder is not None:
        params["folder"] = folder
    if version is not None:
        params["version"] = version
    if only_orphans:
        params["only_orphans"] = "true"
    if media:
        params["media"] = "true"
    return client.get_json("/api/gallery", params=params, workspace=workspace)


def get_gallery_metadata(client, name, envelope=False, workspace=None):
    """Metadata embedded in a saved file: the full workflow that made it -
    the exact workflow, arguments and seed, so a result can be reproduced
    or a failed run's definition edited and re-run - plus the job that
    produced it when history remembers one, plus for audio and video what
    the file holds - duration, sample rate, channels, fps, size, peak and
    mean level in dBFS.

    Only an image (PNG/JPEG/WebP) carries embedded metadata; `metadata` is
    always null for audio and video, since neither format has a slot this
    writer uses. `job` is the fallback recipe when one is known - `next`
    then names `get_job_workflow(job_id)`, which reads the run's realized
    workflow instead. A kept asset (`source: "asset"`) has no job at all,
    so nothing on the server remembers which run made it; `next` says so
    rather than pretending a lookup exists.

    `name` is a gallery name - the `name` field `list_gallery` reports, not
    its `label` (a display-only basename that is not a valid reference) -
    or an 'asset:' reference to read an input asset the same way (#127); the
    same numbers, `job` null, and `source` saying which of the two answered.

    With envelope=True the soundtrack's level is reported second by second
    as well, which is what locates something in a track rather than only
    measuring the whole of it. Opt-in: it is one number per second per
    measure, and the default answer has to stay small."""
    body = client.get_json(
        api_path("api", "gallery", name, "metadata"),
        params={"envelope": "true"} if envelope else None,
        workspace=workspace,
    )
    media = body.get("media")
    job = body.get("job")
    hints = []
    if body.get("metadata") is None:
        if job:
            hints.append(
                "metadata is null because only an image (PNG/JPEG/WebP) "
                "carries it embedded - this file's job is known, and "
                f'get_job_workflow(job_id="{job["id"]}") returns the exact '
                "workflow, arguments and seed that produced it."
            )
        elif body.get("source") == "asset":
            hints.append(
                "metadata is null and this is a kept asset, which carries "
                "no provenance - nothing on the server remembers which job, "
                "if any, produced the file it was kept from."
            )
    if media and body.get("source") == "asset":
        hints.append(
            "These are the numbers a workflow's arguments have to match "
            "before the run, not after: frame_count and fps decide a cut's "
            "'total_frames', sample_rate decides what its audio is mixed "
            "at, and duration_seconds says whether a score reaches the "
            "length of the film it goes under - a score shorter than the "
            "cut is padded with digital silence rather than refused, so "
            "make a longer bed with the 'loop_audio' task instead."
        )
    elif media and media.get("kind") == "audio":
        hints.append(
            "Check duration_seconds against what was asked for: a Music 3 "
            "track that lands within 0.2 s of its audio_duration ceiling was "
            "cut off, one well short of it finished naturally. peak_dbfs is "
            "the level normalize_audio would be given, and the range has two "
            "ends: mean_dbfs below -40 on a track that should be full is a "
            "near-silent render, and peak_dbfs at or above 0 is a deliverable "
            "at or over full scale - a decoded lossy file overshoots by up to "
            "a couple dB legitimately (0.59-1.56 dB measured on Music 3 "
            "mp3s), but a figure of +1 or more is a mix with no headroom, and "
            "'normalize_audio' (peak_dbfs: -3) before the saving step is what "
            "fixes it."
        )
    elif media and media.get("kind") == "video":
        hints.append(
            "peak_dbfs is the level normalize_audio would be given, and the "
            "range has two ends: mean_dbfs below -40 on a track that should "
            "be full is a near-silent render, and peak_dbfs at or above 0 is "
            "a deliverable at or over full scale - 'normalize_audio' "
            "(peak_dbfs: -3) before the saving step is what fixes it."
        )
    if media and media.get("shots"):
        hints.append(
            f"This is a cut of {len(media['shots'])} shots, and whole-file "
            f'numbers cannot see inside a join: assess_output(name="{name}") '
            "measures each seam, the shots' levels and sync, and says where "
            "to look."
        )
    if hints:
        body["next"] = " ".join(hints)
    return body
