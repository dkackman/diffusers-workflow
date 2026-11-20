"""The requests ledger: the plan an agent keeps for what the user asked.

A request lives in its workspace's `requests.json` on the server
(`dw/ledger.py`), read and written whole through
`/api/workspaces/<name>/requests/<id>`. A write names the revision it was
read at, so two writers cannot silently overwrite each other; the server's
refusal says which revision is current.
"""

from dw_mcp.client import DwApiError, api_path


def _workspace(client, workspace):
    return workspace or client.workspace


def request_filter(request):
    """A listing's `request` filter as query params: `"<request-id>"`, or
    `"<request-id>/<item-key>"` for one item - the way `run_workflow` names
    a link. {} for none. The server owns the key rule."""
    if request is None or request == "":
        return {}
    parts = request.split("/") if isinstance(request, str) else []
    if len(parts) not in (1, 2):
        raise DwApiError(
            '`request` must be "<request-id>" or "<request-id>/<item-key>" '
            f"(got {request!r})."
        )
    return dict(zip(("request", "item"), parts))


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
    purge=None,
):
    """Write a request's whole plan, as the agent. The user's `comment` is
    never sent: the server keeps what is there."""
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
    if purge:
        payload["purge"] = purge
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
