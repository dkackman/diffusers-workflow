"""Writing workflows. Validation is free and comes first; saving overwrites,
so it is annotated destructive at the tool layer.

Path confinement is the server's job (dw/security.py already refuses
traversal and anything outside the workflow directory). Nothing here
re-implements it - a second, subtly different check is how the two drift.
"""

from dw_mcp.client import DwApiError, api_path


def validate_workflow(client, workflow=None, name=None, workspace=None, arguments=None):
    """Schema- and signature-check a workflow without queuing anything. This
    is free (no GPU work) and should be called before any run or save. Give
    either an inline definition or the name of a stored one, as
    `list_workflows` reports it.

    `arguments` is the same dict `run_workflow` takes, checked against the
    variables the workflow declares and against this workspace's libraries.
    Without it the answer is about the stored definition and its stock
    defaults - which is everything except the part the caller wrote
    (2026-09-11)."""
    if (workflow is None) == (name is None):
        raise DwApiError(
            "Provide exactly one of `workflow` (an inline definition) or "
            "`name` (a stored workflow)."
        )
    params = {"workspace": workspace} if workspace else None
    payload = {"workflow_path": name} if workflow is None else {"workflow": workflow}
    if arguments:
        payload["arguments"] = arguments
    # The server resolves a name against its own workflow directory, so
    # validation sees the same base directory a run would
    return client.post_json("/api/validate", payload, params=params)


def save_workflow(client, name, workflow):
    """Write a workflow into the server's writable workflow directory,
    overwriting any file already under that name there. A name that resolves
    to one of the server's read-only sources (an examples directory) is not
    overwritten: the copy lands in the writable directory and shadows it from
    then on. The server validates before writing."""
    return client.put_json(api_path("api", "workflows", name), {"workflow": workflow})


def delete_workflow(client, name):
    """Remove a workflow from the server's workflow directory."""
    return client.delete_json(api_path("api", "workflows", name))
