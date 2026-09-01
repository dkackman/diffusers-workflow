"""Writing workflows. Validation is free and comes first; saving overwrites,
so it is annotated destructive at the tool layer.

Path confinement is the server's job (dw/security.py already refuses
traversal and anything outside the workflow directory). Nothing here
re-implements it - a second, subtly different check is how the two drift.
"""

from urllib.parse import quote

from dw.mcp.catalog import get_workflow
from dw.mcp.client import DwApiError


def validate_workflow(client, workflow=None, name=None):
    """Schema- and signature-check a workflow without queuing anything. This
    is free (no GPU work) and should be called before any run or save. Give
    either an inline definition or the name of a stored one."""
    if (workflow is None) == (name is None):
        raise DwApiError(
            "Provide exactly one of `workflow` (an inline definition) or "
            "`name` (a stored workflow)."
        )
    if workflow is None:
        # /api/validate only accepts an inline definition, so resolve first
        workflow = get_workflow(client, name)
    return client.post_json("/api/validate", {"workflow": workflow})


def save_workflow(client, name, workflow):
    """Write a workflow into the server's workflow directory, overwriting any
    file already under that name. The server validates before writing."""
    return client.put_json(
        f"/api/workflows/{_path_segment(name)}", {"workflow": workflow}
    )


def delete_workflow(client, name):
    """Remove a workflow from the server's workflow directory."""
    return client.delete_json(f"/api/workflows/{_path_segment(name)}")


def _path_segment(name):
    # Quote '/' too (safe=""), so a name like "../escape" reaches the server
    # as the literal segment it is rather than being collapsed by ordinary
    # URL dot-segment normalization before the server ever sees it - the
    # server's own path-traversal check is what has to refuse it.
    return quote(name, safe="")
