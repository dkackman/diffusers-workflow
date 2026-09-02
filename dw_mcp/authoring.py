"""Writing workflows. Validation is free and comes first; saving overwrites,
so it is annotated destructive at the tool layer.

Path confinement is the server's job (dw/security.py already refuses
traversal and anything outside the workflow directory). Nothing here
re-implements it - a second, subtly different check is how the two drift.
"""

from dw_mcp.client import DwApiError, api_path


def validate_workflow(client, workflow=None, name=None):
    """Schema- and signature-check a workflow without queuing anything. This
    is free (no GPU work) and should be called before any run or save. Give
    either an inline definition or the name of a stored one, as
    `list_workflows` reports it."""
    if (workflow is None) == (name is None):
        raise DwApiError(
            "Provide exactly one of `workflow` (an inline definition) or "
            "`name` (a stored workflow)."
        )
    if workflow is None:
        # The server resolves the name against its own workflow directory,
        # so validation sees the same base directory a run would
        return client.post_json("/api/validate", {"workflow_path": name})
    return client.post_json("/api/validate", {"workflow": workflow})


def save_workflow(client, name, workflow):
    """Write a workflow into the server's workflow directory, overwriting any
    file already under that name. The server validates before writing."""
    return client.put_json(api_path("api", "workflows", name), {"workflow": workflow})


def delete_workflow(client, name):
    """Remove a workflow from the server's workflow directory."""
    return client.delete_json(api_path("api", "workflows", name))
