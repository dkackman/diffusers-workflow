"""The requests ledger: the plan an agent keeps for what the user asked.

A request lives in its workspace's `requests.json` on the server
(`dw/ledger.py`), read and written whole through
`/api/workspaces/<name>/requests/<id>`. A write names the revision it was
read at, so two writers cannot silently overwrite each other; the server's
refusal says which revision is current.
"""

from dw_mcp.client import api_path


def _workspace(client, workspace):
    return workspace or client.workspace


def get_request(client, request=None, workspace=None):
    """One request's plan, or the workspace's list when `request` is None."""
    name = _workspace(client, workspace)
    if request is None:
        return client.get_json(api_path("api", "workspaces", name, "requests"))
    return client.get_json(api_path("api", "workspaces", name, "requests", request))


def set_request(
    client,
    request,
    title,
    goal,
    items,
    revision,
    why,
    prompt=None,
    state=None,
    workspace=None,
):
    """Write a request's whole plan, as the agent."""
    payload = {
        "title": title,
        "goal": goal,
        "items": items,
        "revision": revision,
        "why": why,
        "by": "agent",
    }
    if prompt is not None:
        payload["prompt"] = prompt
    if state is not None:
        payload["state"] = state
    return client.put_json(
        api_path(
            "api", "workspaces", _workspace(client, workspace), "requests", request
        ),
        payload,
    )


def delete_request(client, request, revision, workspace=None):
    """Remove one request from the ledger at the revision read; its linked
    jobs keep their rows, their link and their files."""
    return client.delete_json(
        api_path(
            "api", "workspaces", _workspace(client, workspace), "requests", request
        ),
        params={"revision": revision},
    )
