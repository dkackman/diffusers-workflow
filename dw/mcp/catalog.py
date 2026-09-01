"""Read-only tools: everything an agent can look at without spending GPU
time. Each is a pass-through - the API's shapes are already the ones the
web UI consumes, and reshaping them here would only add a second thing to
keep in sync."""

from dw.mcp.client import path_segment


def list_workflows(client):
    """Workflow names in the server's workflow directory, with details."""
    return client.get_json("/api/workflows")


def get_workflow(client, name):
    """One workflow's full JSON definition."""
    return client.get_json(f"/api/workflows/{path_segment(name)}")


def get_schema(client):
    """The workflow JSON schema every definition is validated against."""
    return client.get_json("/api/schema")


def list_pipelines(client):
    """Every diffusers pipeline class this install exports."""
    return client.get_json("/api/pipelines")


def get_pipeline_signature(client, name):
    """A pipeline's real __call__ arguments - check before proposing a fix."""
    return client.get_json(f"/api/pipelines/{path_segment(name)}")


def list_classes(client, kind):
    """Class names of one kind: pipelines, models, schedulers, quantization."""
    return client.get_json("/api/classes", params={"kind": kind})


def get_class(client, name, target="init"):
    """A class's argument schema. target: init, call, or load."""
    return client.get_json(
        f"/api/classes/{path_segment(name)}", params={"target": target}
    )


def list_tasks(client):
    """Every task command a workflow's task step can name."""
    return client.get_json("/api/tasks")


def get_task(client, command):
    """A task command's argument schema."""
    return client.get_json(f"/api/tasks/{path_segment(command)}")


def list_models(client):
    """What the Hugging Face hub cache holds, largest repo first."""
    return client.get_json("/api/models")


def get_memory(client):
    """Worker VRAM/RAM stats - the first thing to check on an OOM."""
    return client.get_json("/api/memory")


def get_health(client):
    """Server liveness."""
    return client.get_json("/api/health")


def list_jobs(client):
    """The live queue plus recent history, oldest first."""
    return client.get_json("/api/jobs")


def list_gallery(client, limit=50):
    """Generated media in the output directory, newest first."""
    return client.get_json("/api/gallery", params={"limit": limit})


def get_gallery_metadata(client, name):
    """Metadata embedded in a saved file: the full workflow that made it,
    plus the job that produced it when history remembers one."""
    return client.get_json(f"/api/gallery/{path_segment(name)}/metadata")
