"""Which of the server's workspaces this session works in.

A server holds several - each with its own workflows, assets and outputs,
all sharing one prompt library - and several agents can work against one
server without sharing a namespace. The selection is a session default
rather than a parameter on every tool: switching is then one visible call in
the transcript instead of an argument that can be forgotten on the one call
where it mattered.
"""

import logging

from dw_mcp.client import DEFAULT_WORKSPACE, DwApiError, api_path
from dw_mcp import catalog

logger = logging.getLogger(__name__)


# What a compact workspace entry keeps: enough to choose one or judge its
# size, nothing that only naming its folders needs
WORKSPACE_SUMMARY_FIELDS = ("name", "default", "usage")

# dw.serve --mcp builds one DwClient for every connected agent (#298) - the
# pin is server-global there, not per-session, and that is a deliberate scope
# decision (this server is single-user) rather than a bug to fix with session
# tracking. The one thing owed to a caller is honesty: when the mounted
# client's pin is about to move away from a workspace that was not the
# default, warn - it may be this same caller switching again, or it may be
# another connected agent's work about to lose its isolation
_CONCURRENT_PIN_WARNING = (
    "This server's MCP mount shares workspace state across every connected "
    "client (dw.serve --mcp has no per-session pin). The pin was "
    "'{previous}' and is now '{name}' - if another agent is also connected, "
    "its calls saw the workspace change too, and its next switch will "
    "change what this session sees. Pass workspace='{name}' on calls that "
    "accept it if you need isolation from other clients."
)


def _switch(client, name):
    """Set client.workspace to `name`, returning a warning string when the
    mounted (shared) client is about to overwrite a pin already in use for
    something other than the default - see `_CONCURRENT_PIN_WARNING`."""
    previous = client.workspace
    warning = None
    if client.mounted and previous != DEFAULT_WORKSPACE and previous != name:
        warning = _CONCURRENT_PIN_WARNING.format(previous=previous, name=name)
        logger.warning(
            "mcp workspace pin changed from %r to %r on the shared mounted "
            "client - another connected agent may be affected",
            previous,
            name,
        )
    client.workspace = name
    return warning


def list_workspaces(client, detail=False):
    """The workspaces on the server, and which one this session is using.

    Compact by default - each entry cut to `name`, `default` and `usage`
    (files/bytes) - because a server that has accumulated dozens of
    workspaces across episodes and regression runs answers in the tens of
    KB otherwise for a question that is usually "which workspaces exist and
    how big are they" (#249). Pass `detail=True` for each entry's full
    folder paths (`root`, `workflows`, `assets`, `outputs`, `prompts`,
    `common_assets`), needed before naming one to `use_workspace` or a
    remote read.
    """
    result = client.get_json("/api/workspaces")
    if not detail:
        entries = result.get("workspaces")
        if isinstance(entries, list):
            result = {
                **result,
                "workspaces": [
                    {
                        key: entry.get(key)
                        for key in WORKSPACE_SUMMARY_FIELDS
                        if key in entry
                    }
                    for entry in entries
                    if isinstance(entry, dict)
                ],
                "note": (
                    "Compact: each entry is name/default/usage only. Call "
                    "list_workspaces(detail=True) for full folder paths."
                ),
            }
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
    warning = _switch(client, name)
    result = {"current": name, "workspaces": known or [name]}
    if warning:
        result["warning"] = warning
    return result


def create_workspace(client, name, use=False):
    """Make a new workspace on the server. It gets its own workflows, assets
    and outputs, and shares the server's one prompt library. Creating it
    does not switch to it unless `use` is true - the natural
    create-then-run sequence otherwise runs in the workspace the session
    was already in, and the result says which that is."""
    body = client.post_json("/api/workspaces", {"name": name})
    warning = _switch(client, name) if use else None
    result = {
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
    if warning:
        result["warning"] = warning
    return result


def delete_workspace(client, name, acknowledged_cost=False):
    """Delete a workspace and every workflow, asset and generated file in it.

    Refuses without `acknowledged_cost=True`, and reports what it would
    remove instead - a count of files is what makes this an informed choice
    rather than a surprise.
    """
    if not acknowledged_cost:
        # Unacknowledged, the server answers 409 with what it would remove -
        # which the client turns into the error text below, so the count
        # reaches the caller rather than a bare refusal. A name that fails
        # for another reason (404 not found, 400 for the default workspace)
        # never reaches that 409, and acknowledging cannot fix it - the
        # acknowledgement sentence belongs only on the 409
        try:
            client.delete_json(api_path("api", "workspaces", name))
        except DwApiError as e:
            if e.status_code == 409:
                raise DwApiError(
                    f"{e} Call delete_workspace again with "
                    f"acknowledged_cost=True to proceed."
                )
            raise
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
