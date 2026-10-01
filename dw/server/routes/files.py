"""The routes that serve files: generated outputs, input assets and an export
zip.

Not in `ROUTERS`: `create_app` adds these after the `/mcp` routes and before
the UI mount, which is where they have always sat. All three are ungated -
an `<img>`, `<video>` or download link cannot attach an Authorization header -
so the auth middleware, which covers `/api/` and `/mcp` only, leaves them be.

Served as routes rather than static mounts: a mount is bound to one directory
at startup, and a workspace can be created afterwards. Each handler delegates
to a StaticFiles instance for the workspace's own root (`static_files_for`)
rather than a bare FileResponse, which never answers 304 - that keeps ETag/
If-None-Match 304s, Range/206 and StaticFiles' own 404 handling, the way a
real mount has them.

'/inputs', not '/assets': Vite emits the SPA's own bundles under /assets/,
and serving the library there shadows them - the page loads and then renders
nothing. The name is also the symmetric one, next to /outputs.
"""

import json
import os
import re

from fastapi import APIRouter, Depends, HTTPException, Request

from ...assets import is_asset_reference
from ...runs import MANIFEST_FILE_NAME
from ...security import SecurityError
from ...workspace import Workspace
from ..deps import selected_workspace
from ..exports import export_directory
from ..http_security import ACTIVE_DOCUMENT_TYPES
from ..outputs import (
    asset_file,
    workspace_asset_library,
    static_files_for,
    strip_output_prefix,
    zip_download,
)

router = APIRouter()


def _sandbox_active_content(response):
    """Serve a document type under `Content-Security-Policy: sandbox`.

    /outputs and /inputs share the UI's origin and need no token, so an
    .html, .xhtml, .xml or .svg file served as-is is a page whose script
    reads the token the UI keeps in localStorage. Validation refuses a
    workflow writing one (dw/content_types.py), but a planted file or a
    kept asset never passes through there. sandbox gives the document an
    opaque origin and no script, and still lets an image or a .txt show
    in the tab, which an attachment disposition would not. Set on the
    Response StaticFiles built, so its ETag/304 and Range/206 stand"""
    media_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type in ACTIVE_DOCUMENT_TYPES:
        response.headers["Content-Security-Policy"] = "sandbox"
    return response


@router.get("/outputs/{name:path}")
async def output_file(
    name: str, request: Request, ws: Workspace = Depends(selected_workspace)
):
    """One generated file, from the workspace that made it - or, by an
    'asset:' reference, one file from its asset library (#445): every
    other route in this family (`get_gallery_metadata`, `/frames`,
    `/audio`, `/assess`) already accepts one, and this route answering a
    bare StaticFiles 404 for the same name gave no hint why."""
    name = strip_output_prefix(name)
    if is_asset_reference(name):
        # This route is outside the token gate (the auth middleware
        # covers /api/ and /mcp only), so a miss must not carry
        # _asset_file's detail, which names every root searched by its
        # absolute server path. Keep the hint #445 added, without them.
        try:
            path = asset_file(request.app.state, name, ws)
        except HTTPException as e:
            if e.status_code != 404:
                raise
            raise HTTPException(
                status_code=404,
                detail=f"Unknown asset {name!r}: not in this workspace's "
                "asset library (list_assets shows what is)",
            ) from None
        files = static_files_for(request.app.state, os.path.dirname(path))
        response = await files.get_response(os.path.basename(path), request.scope)
    else:
        files = static_files_for(request.app.state, ws.outputs)
        response = await files.get_response(name, request.scope)
    return _sandbox_active_content(response)


@router.get("/inputs/{name:path}")
async def input_file(
    name: str, request: Request, ws: Workspace = Depends(selected_workspace)
):
    """One file from the asset search path, for the editor's preview of
    an uploaded or chosen asset - the workspace's own library first,
    then any read-only examples library, so an example workflow's media
    previews the way an upload does."""
    library = workspace_asset_library(request.app.state, ws)
    if not library.roots():
        raise HTTPException(status_code=404, detail="no asset library")
    found = library.find(name)
    if found:
        files = static_files_for(request.app.state, found[1].root)
        return _sandbox_active_content(await files.get_response(name, request.scope))
    # Nothing has it: let the workspace's own library answer, so the
    # 404 (and its headers) come from StaticFiles as they always did
    files = static_files_for(request.app.state, library.roots()[0].root)
    return await files.get_response(name, request.scope)


def _export_download_name(directory, job_id):
    """'<workflow>-v4-<job id>.zip' when the exported manifest says which
    run it was, else '<job id>.zip'. Only the saved file's name: the
    URL and the entries inside keep the job id, so nothing that already
    names an export changes."""
    try:
        with open(os.path.join(directory, MANIFEST_FILE_NAME)) as file:
            manifest = json.load(file)
    except (OSError, ValueError):
        return f"{job_id}.zip"
    if not isinstance(manifest, dict):
        return f"{job_id}.zip"
    version = manifest.get("version")
    identity = (manifest.get("workflow") or {}).get("identity")
    if not isinstance(version, int) or isinstance(version, bool):
        return f"{job_id}.zip"
    if not isinstance(identity, str) or not identity:
        return f"v{version}-{job_id}.zip"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", identity).strip("-.")
    return f"{slug}-v{version}-{job_id}.zip" if slug else f"v{version}-{job_id}.zip"


# Ungated for the same reason the two above are: a download link cannot
# attach an Authorization header either
@router.get("/exports/{job_id}.zip")
def export_zip(job_id: str, ws: Workspace = Depends(selected_workspace)):
    """One job's export as a zip, built on request from the directory
    rather than kept as a second copy. Entries are named
    '<job id>/<relative path>', so unzipping anywhere gives the same tree
    the server holds."""
    try:
        directory = export_directory(ws.root, job_id)
    except SecurityError:
        raise HTTPException(status_code=404, detail="No export for this job")
    if not os.path.isdir(directory):
        raise HTTPException(status_code=404, detail="No export for this job")

    entries = []
    for current, _dirs, names in os.walk(directory):
        for name in sorted(names):
            path = os.path.join(current, name)
            entry = os.path.relpath(path, directory).replace(os.sep, "/")
            entries.append((f"{job_id}/{entry}", path))
    return zip_download(entries, _export_download_name(directory, job_id))
