"""Output and asset resolution: where a workspace's asset library is searched
and how a served file is named as a URL.

Only the part the job, system and library routers reach lives here so far;
the rest of the output/asset cluster joins it with the gallery, media, assets
and files routers. Functions take the app's `state` explicitly.
"""

import os
from urllib.parse import quote

from .. import settings


def common_assets(ws):
    """The library every workspace under this root shares, or None.

    A recurring cast is not the property of the workspace that first
    uploaded it, and a fresh workspace could not see it at all - the
    prompt library has been shared from the start for the same reason.
    """
    return getattr(ws, "common_assets", None)


def asset_roots(state, ws):
    """The asset search path of one workspace: its own library, then the
    one shared by every workspace under this root, then the read-only
    ones an --examples-dir tree brought with it. The same order 'asset:'
    resolves in (dw/assets.asset_search_path), so what the browser lists
    is what a job would load."""
    roots = []
    for root in [ws.assets, common_assets(ws), *state.example_asset_dirs]:
        if not root:
            continue
        root = os.path.abspath(root)
        if root not in roots and os.path.isdir(root):
            roots.append(root)
    return roots


def resolution_roots(state, ws):
    """`asset_roots(state, ws)`, falling back to the workspace's own (possibly
    nonexistent) library when the search path is empty.

    A caller resolving a name still needs *somewhere* to fail against:
    with no root at all the 404 would name no directory, leaving the
    caller to guess where it looked. Naming the workspace's own
    directory keeps the failure pointing at the library the caller
    thinks they're working in, even when that library hasn't been
    created yet.

    Never `[None]`: a server configured with no asset library at all has
    nothing to point at either, and `_asset_in` turns the resulting empty
    list into the "no asset library" 404 rather than joining `None`.
    """
    roots = asset_roots(state, ws)
    if roots:
        return roots
    return [os.path.abspath(ws.assets)] if ws.assets else []


def asset_roots_for_job(state, job_id, ws):
    """The asset search path a job's own run used, for export: its spec's
    `asset_dir` (or the historical row's), then the read-only example
    libraries an --examples-dir tree brought with it - the same shape
    `asset_roots` builds for the selected workspace, but rooted at
    wherever the job actually ran rather than at the workspace the
    caller happens to be scoped to now. A job that ran in one workspace
    while the caller exports it scoped to another must still find its
    own 'asset:' files, not the other workspace's.

    Falls back to `asset_roots(state, ws)` when the job carries no asset_dir
    of its own - an inline-workflow job, or one recorded before this
    field existed."""
    job = state.job_manager.get(job_id)
    if job is None:
        return asset_roots(state, ws)
    spec = (job.get("spec") or {}) if isinstance(job, dict) else job.spec
    asset_dir = spec.get("asset_dir")
    if not asset_dir:
        return asset_roots(state, ws)
    roots = []
    for root in [asset_dir, common_assets(ws), *state.example_asset_dirs]:
        if not root:
            continue
        root = os.path.abspath(root)
        if root not in roots and os.path.isdir(root):
            roots.append(root)
    return roots


def served_url(path, ws, version=None):
    """The URL a served file is reachable at: the default workspace's
    files keep the URL they have always had, a named one carries the
    same selector its API calls do, so one route serves both. 'v=' is
    cache-busting for a name reused by a rerun, not the workspace
    selector, so it always comes last."""
    url = path if ws.is_default else f"{path}?workspace={quote(ws.name)}"
    if version is None:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={version}"


def absolute_served_url(path, ws, version=None):
    """The same URL, made openable by a client with no other way to
    learn this server's origin (#353) - an MCP-only agent, which is
    never told a request's Host and must not guess one. `None` unless
    an operator has configured `public_url` (or `DW_PUBLIC_URL`):
    deriving an origin from request/forwarded headers would trust
    whatever the caller claims to be, so a caller gets nothing rather
    than a guess."""
    origin = os.environ.get("DW_PUBLIC_URL") or settings.public_url
    if not origin:
        return None
    return f"{origin.rstrip('/')}{served_url(path, ws, version)}"
