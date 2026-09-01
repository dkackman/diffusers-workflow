"""The model cache and the installed diffusers version.

Where the rest of the surface spends the GPU, this one spends the disk and
the network - and, for a delete, spends them in the direction that cannot be
undone by waiting. Three tools here commit the machine to something a user
would want to have been asked about first, so each carries its own gate and
its own reason: a download is measured in tens of gigabytes, a delete throws
away weights that have to be fetched again, and an update replaces the
library every pipeline in the install is built on.

The refusals deliberately do not share one message. A single "this costs
something, confirm it" would be wrong for each of them in a different way,
and a gate the user learns to wave through is not a gate.
"""

from dw_mcp.client import DwApiError, path_segment

DOWNLOAD_REFUSAL = (
    "Downloading a model pulls tens of gigabytes over the network and into "
    "the Hugging Face cache, and it keeps running in the background once "
    "started. Tell the user which repo you are about to fetch and roughly "
    "what it will cost them in disk and bandwidth, get their go-ahead, then "
    "call again with acknowledged_cost=true. `list_models` shows what is "
    "already cached - check there first, the model may not need fetching."
)

DELETE_REFUSAL = (
    "Deleting a cached model removes every downloaded revision of it. "
    "Nothing about this is recoverable locally: getting it back means "
    "downloading it again. Tell the user exactly which repo you are about "
    "to delete and how much it frees, get their go-ahead, then call again "
    "with acknowledged_cost=true."
)

UPDATE_REFUSAL = (
    "Updating diffusers replaces the installed library with GitHub HEAD - "
    "an untagged development build that can break workflows which currently "
    "run, and there is no undo through this tool. Tell the user what is "
    "installed now (`get_diffusers_state`), why the update is worth it, and "
    "get their go-ahead, then call again with acknowledged_cost=true."
)


def download_model(client, repo_id, acknowledged_cost=False):
    """Start fetching a repo into the Hugging Face cache. Returns as soon as
    the download is started - it does not wait for it to finish."""
    if not acknowledged_cost:
        raise DwApiError(DOWNLOAD_REFUSAL)
    download = client.post_json("/api/models/download", {"repo_id": repo_id})
    return {
        **download,
        "next": "Poll list_downloads for progress. The download continues in "
        "the background; cancel_download stops it.",
    }


def list_downloads(client):
    """Every download the server is running or has recently run."""
    return client.get_json("/api/models/downloads")


def cancel_download(client, download_id):
    """Ask a running download to stop. Not gated: this ends a cost rather
    than starting one, and partial files stay in the cache and resume on a
    retry, so it is cheap to get wrong in the safe direction."""
    return client.post_json(f"/api/models/downloads/{path_segment(download_id)}/cancel")


def delete_model(client, repo, acknowledged_cost=False):
    """Delete every cached revision of one repo from the hub cache.

    The server refuses this while a job or a download is active - it would
    be pulling files out from under a reader - and that refusal carries the
    reason, so it reaches the caller unflattened."""
    if not acknowledged_cost:
        raise DwApiError(DELETE_REFUSAL)
    return client.delete_json("/api/models", params={"repo": repo})


def get_diffusers_state(client):
    """The installed diffusers version, its git commit when installed from
    git, and the state of any update in flight."""
    return client.get_json("/api/system/diffusers")


def update_diffusers(client, acknowledged_cost=False):
    """Upgrade diffusers to GitHub HEAD in the background."""
    if not acknowledged_cost:
        raise DwApiError(UPDATE_REFUSAL)
    update = client.post_json("/api/system/diffusers/update")
    return {
        **update,
        "next": "Poll get_diffusers_state for the outcome. The idle worker is "
        "shut down on success so the next job imports the new version.",
    }
