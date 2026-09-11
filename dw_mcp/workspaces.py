"""Which of the server's workspaces this session works in.

A server holds several - each with its own workflows, assets and outputs,
all sharing one prompt library - and several agents can work against one
server without sharing a namespace. The selection is a session default
rather than a parameter on every tool: switching is then one visible call in
the transcript instead of an argument that can be forgotten on the one call
where it mattered.
"""

from dw_mcp.client import DEFAULT_WORKSPACE, DwApiError, api_path
from dw_mcp import catalog


def list_workspaces(client):
    """The workspaces on the server, and which one this session is using."""
    result = client.get_json("/api/workspaces")
    return {**result, "current": client.workspace}


def use_workspace(client, name):
    """Work in a different workspace for the rest of this session.

    Checked against the server before it takes effect: a typo that silently
    scoped every later call to a workspace that does not exist would fail
    one call at a time, far from its cause.
    """
    name = name or DEFAULT_WORKSPACE
    listing = client.get_json("/api/workspaces")
    entries = listing.get("workspaces")
    # Membership is checked when the server said what it has. A listing that
    # does not say is not grounds to refuse: the request reached the server,
    # and a name it does not know will be refused by the next call that uses
    # it - with the name in the message, which is what matters
    known = (
        [entry["name"] for entry in entries if isinstance(entry, dict)]
        if isinstance(entries, list)
        else None
    )
    if known and name not in known:
        raise DwApiError(
            f"No workspace named '{name}' on this server. It has: "
            f"{', '.join(known)}. Create one with create_workspace."
        )
    client.workspace = name
    return {"current": name, "workspaces": known or [name]}


def create_workspace(client, name, use=False):
    """Make a new workspace on the server. It gets its own workflows, assets
    and outputs, and shares the server's one prompt library. Creating it
    does not switch to it unless `use` is true - the natural
    create-then-run sequence otherwise runs in the workspace the session
    was already in, and the result says which that is."""
    body = client.post_json("/api/workspaces", {"name": name})
    if use:
        use_workspace(client, name)
    return {
        **body,
        "current": client.workspace,
        "next": (
            f"This session now works in '{client.workspace}'."
            if use
            else f"This session still works in '{client.workspace}' - call "
            f"use_workspace('{name}') or pass workspace='{name}' to "
            f"run_workflow before running anything meant for it."
        ),
    }


def delete_workspace(client, name, acknowledged_cost=False):
    """Delete a workspace and every workflow, asset and generated file in it.

    Refuses without `acknowledged_cost=True`, and reports what it would
    remove instead - a count of files is what makes this an informed choice
    rather than a surprise.
    """
    if not acknowledged_cost:
        # Unacknowledged, the server answers 409 with what it would remove -
        # which the client turns into the error text below, so the count
        # reaches the caller rather than a bare refusal
        try:
            client.delete_json(api_path("api", "workspaces", name))
        except DwApiError as e:
            raise DwApiError(
                f"{e} Call delete_workspace again with acknowledged_cost=True "
                f"to proceed."
            )
        raise DwApiError(
            f"Deleting workspace '{name}' removes everything in it. Call "
            f"again with acknowledged_cost=True to proceed."
        )
    result = client.delete_json(
        api_path("api", "workspaces", name), params={"acknowledged": "true"}
    )
    if client.workspace == name:
        client.workspace = DEFAULT_WORKSPACE
    return {**result, "current": client.workspace}


def server_info(client):
    """What this installation can do and where it keeps things: the
    accelerator, version, directories, and which workspace this session works
    in. When the session is in a named workspace, directories are scoped to
    that workspace rather than the server's default.
    """
    info = catalog.get_server_info(client)

    # /api/server describes the server's default workspace. A session in a
    # named one is told where *its* folders are, so a path it is handed
    # back is relative to the right place. The root's other keys (the
    # workspace root itself) stay
    if client.workspace != DEFAULT_WORKSPACE:
        listing = client.get_json("/api/workspaces")
        for workspace in listing.get("workspaces") or []:
            if (
                isinstance(workspace, dict)
                and workspace.get("name") == client.workspace
            ):
                info["directories"] = {
                    **(info.get("directories") or {}),
                    **{
                        key: workspace.get(key)
                        for key in ("workflows", "assets", "outputs", "prompts")
                    },
                }
                break

    info["workspace"] = client.workspace
    return info
