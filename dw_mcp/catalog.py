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
    of one template. `get_workflow` has the full definition."""
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
    return client.get_json("/api/workflows", params=params)


def get_workflow(client, name, variables_only=False):
    """One workflow's full JSON definition.

    With variables_only=True, just its variables and what they default to -
    confirming that a stored workflow's audio_bleed_ms is 1800 otherwise
    means pulling the whole definition, quantization blocks and all, to
    read one integer (2026-09-11). Long defaults come back cut to their
    first 200 characters, with the names of the cut ones in `truncated`.
    """
    if variables_only:
        return client.get_json(api_path("api", "workflows", name, "variables"))
    return client.get_json(api_path("api", "workflows", name))


def get_schema(client):
    """The workflow JSON schema every definition is validated against."""
    return client.get_json("/api/schema")


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


def get_health(client):
    """Server liveness, plus what answered: version, device, worker
    liveness, the job running now and how many are queued."""
    return client.get_json("/api/health")


def get_server_info(client):
    """What this installation can do and where it keeps things: the
    accelerator, the dw version, the workflow/output/prompt directories,
    and how the server is reached."""
    return client.get_json("/api/server")


def list_jobs(client):
    """The live queue plus recent history, oldest first."""
    return client.get_json("/api/jobs")


def list_gallery(client, limit=50):
    """Generated media in the output directory, newest first."""
    return client.get_json("/api/gallery", params={"limit": limit})


def get_gallery_metadata(client, name, envelope=False):
    """Metadata embedded in a saved file: the full workflow that made it,
    plus the job that produced it when history remembers one, plus for
    audio and video what the file holds - duration, sample rate, channels,
    fps, size, peak and mean level in dBFS.

    With envelope=True the soundtrack's level is reported second by second
    as well, which is what locates something in a track rather than only
    measuring the whole of it. Opt-in: it is one number per second per
    measure, and the default answer has to stay small."""
    body = client.get_json(
        api_path("api", "gallery", name, "metadata"),
        params={"envelope": "true"} if envelope else None,
    )
    media = body.get("media")
    if media and media.get("kind") in ("audio", "video"):
        body["next"] = (
            "Check duration_seconds against what was asked for: a Music 3 "
            "track that lands within 0.2 s of its audio_duration ceiling was "
            "cut off, one well short of it finished naturally. peak_dbfs is "
            "the level normalize_audio would be given; mean_dbfs below -40 "
            "on a track that should be full is a near-silent render."
        )
    return body
