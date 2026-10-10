"""The asset library routes: browser uploads, the library listing, keeping a
generated file as an asset, the bulk archive and delete.

Handlers read the server's state from `request.app.state`. The greedy
`DELETE /api/assets/{name:path}` is registered after its fixed-path siblings.
"""

import logging
import os
import shutil
import uuid
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..api_models import (
    AssetDeleted,
    AssetList,
    Kept,
    Uploaded,
)
from ...references import ASSET, make_ref
from ...runs import record_kept_shots, recorded_shots
from ...security import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_LUT_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    SecurityError,
    validate_asset_reference,
    validate_output_path,
    validate_path,
)
from ...library import ReadOnlyLibraryError, shadowed_listing
from ...workspace import Workspace, forget_workspace_usage
from ..deps import (
    asset_library_missing,
    selected_workspace,
    writable_asset_directory,
)
from ..outputs import (
    ArchiveRequest,
    absolute_served_url,
    archive_selection,
    asset_in,
    workspace_asset_library,
    iter_gallery_files,
    job_provenance,
    resolution_library,
    resolve_output_file,
    served_url,
    strip_output_prefix,
)

logger = logging.getLogger("dw")

router = APIRouter()


def _write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)


UPLOADS_SUBDIR = "uploads"
# Audio included: the asset library holds it and workflows read it (an
# H3 audio reference is built from a .wav), so refusing it here would
# leave one input kind with no way onto the machine. A .cube is a 3D colour
# lookup table apply_lut reads (#603) - the one non-media kind, parsed
# strictly when a step reads it, never at upload
ALLOWED_UPLOAD_EXTENSIONS = (
    ALLOWED_IMAGE_EXTENSIONS
    | ALLOWED_VIDEO_EXTENSIONS
    | ALLOWED_AUDIO_EXTENSIONS
    | ALLOWED_LUT_EXTENSIONS
)
# The library holds LUTs too, so the listing shows them (#757)
LUT_KINDS = {ext: "lut" for ext in ALLOWED_LUT_EXTENSIONS}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB - covers a short video clip


@router.post(
    "/api/uploads",
    status_code=201,
    response_model=Uploaded,
    response_model_exclude_unset=True,
)
async def upload_media(
    request: Request,
    filename: str,
    asset_name: Optional[str] = None,
    shared: bool = False,
    ws: Workspace = Depends(selected_workspace),
):
    """Save a browser-picked image, video or audio file into the asset library's
    uploads/ subfolder and hand back the reference a workflow argument
    can carry.

    An upload is input, so it belongs in the asset library rather than
    among generated output, and the reference handed back is
    'asset:uploads/<name>' - portable, and meaningful in a workflow that
    is saved and rerun later. A server with no asset library configured
    answers 409. The body is the raw file bytes: no multipart parser
    dependency needed for a single-file upload.

    `asset_name` stores it under a name of the caller's choosing -
    'cast/priya-voice.wav' rather than the random one a browser upload
    gets - which is what makes a recurring cast's references readable
    in every workflow that carries them. It may name a folder, is
    confined to the library the way `keep_output`'s is, and takes the
    uploaded file's extension when it has none of its own. Without it
    the name stays random, so two uploads of the same file never
    collide.

    `shared` puts it in the library every workspace under this root
    shares rather than in this workspace's own - a recurring cast that
    episode four, in a workspace of its own, still has to reach.
    """
    extension = os.path.splitext(os.path.basename(filename))[1].lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"File extension not allowed: {extension}"
        )

    # Refuse an oversized upload from its declared length, before
    # reading a single byte of it
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large: {declared} > {MAX_UPLOAD_BYTES}",
        )
    # Counted as it arrives: a chunked body sends no Content-Length, and
    # request.body() would hold all of it before any size check
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload too large: more than {MAX_UPLOAD_BYTES} bytes",
            )
    if not body:
        raise HTTPException(status_code=400, detail="Empty upload")

    # An upload is input and goes to an asset library; with none there is
    # nowhere for it to be, the same 409 keep and delete answer (it used to
    # land among the outputs)
    library = writable_asset_directory(request.app.state, ws, shared)
    uploads_dir = os.path.join(library, UPLOADS_SUBDIR)
    name = f"{uuid.uuid4().hex}{extension}"
    if asset_name:
        name = asset_name
        if not os.path.splitext(name)[1]:
            name = f"{name}{extension}"
        try:
            # The same check the keep route makes: a name, possibly with
            # folders in it, that cannot climb out of the library
            name = validate_asset_reference(name)
        except SecurityError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if os.path.splitext(name)[1].lower() != extension:
            raise HTTPException(
                status_code=400,
                detail=f"asset_name {asset_name!r} does not match the "
                f"uploaded file's kind ({extension})",
            )
    os.makedirs(uploads_dir, exist_ok=True)
    try:
        dest = validate_output_path(os.path.join(uploads_dir, name), uploads_dir)
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Off the event loop: a 200 MB write would otherwise stall every SSE
    # stream and poll for its duration
    await run_in_threadpool(_write_bytes, dest, body)
    logger.info(f"Saved upload {filename!r} -> {dest}")
    path = f"/inputs/{UPLOADS_SUBDIR}/{quote(name)}"
    result = {
        "reference": make_ref(ASSET, f"{UPLOADS_SUBDIR}/{name}"),
        "workspace": ws.name,
        "url": served_url(path, ws),
        "shared": shared,
    }
    absolute_url = absolute_served_url(path, ws)
    if absolute_url is not None:
        result["absolute_url"] = absolute_url
    return result


@router.get("/api/assets", response_model=AssetList, response_model_exclude_unset=True)
def list_assets(
    request: Request,
    ws: Workspace = Depends(selected_workspace),
    limit: Optional[int] = Query(
        default=None,
        ge=1,
        description="Newest this many assets (by mtime); all when omitted",
    ),
    prefix: Optional[str] = Query(
        default=None,
        description="Only assets whose name starts with this, e.g. a folder "
        "such as 'cast/'",
    ),
):
    """The asset library: the input media an 'asset:' reference names.

    Reported by reference rather than by path - 'asset:uploads/x.png' is
    what a workflow argument carries, and a client that only ever sees
    references cannot accidentally write a path that means something
    else on another machine. Empty, not an error, on a server with no
    library configured: nothing is wrong, there is just nowhere for an
    asset to be.

    `prefix` keeps the assets (and shadowed names) whose name starts with
    it; `limit` then keeps the newest that many. `total` is what matched
    before the cut, so a caller can tell a bounded answer from a complete one.
    """
    library = workspace_asset_library(request.app.state, ws)
    if not library.roots():
        return {
            "workspace": ws.name,
            "libraries": [],
            "assets": [],
            "folders": [],
            "shadowed": [],
        }

    # What the media walk found under each root, kept beside the names it
    # hands to `entries` - which decides the winner of each name and what it
    # hid, exactly as 'asset:' resolution does
    found = {}

    def media_names(root):
        try:
            files = list(
                iter_gallery_files(root, group_runs=False, extra_kinds=LUT_KINDS)
            )
        except OSError:
            return []
        names = []
        for relative, folder, _subfolder, _run_id, kind, path in files:
            try:
                stat = os.stat(path)
            except OSError:
                continue
            found[(root, relative)] = (folder, kind, stat)
            names.append(relative)
        return names

    winners, hidden = library.entries(media_names)

    def summary(name, root):
        folder, kind, stat = found[(root.root, name)]
        return {
            "name": name,
            "reference": make_ref(ASSET, name),
            "folder": folder,
            "kind": kind,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "origin": root.origin,
            "writable": root.writable,
        }

    assets = []
    for name, root in winners.items():
        asset_path = f"/inputs/{quote(name)}"
        # `url` is for the editor's own preview - fetchable the same way
        # an upload's URL is
        asset_entry = {**summary(name, root), "url": served_url(asset_path, ws)}
        absolute_url = absolute_served_url(asset_path, ws)
        if absolute_url is not None:
            asset_entry["absolute_url"] = absolute_url
        assets.append(asset_entry)
    assets.sort(key=lambda entry: entry["mtime"], reverse=True)
    if prefix:
        assets = [entry for entry in assets if entry["name"].startswith(prefix)]
    # The one producer of the field's name/origin/shadowed_by, plus the
    # media facts an asset entry carries
    shadowed = [
        {**summary(entry["name"], root), "shadowed_by": entry["shadowed_by"]}
        for entry, (_name, root, _winner) in zip(shadowed_listing(hidden), hidden)
    ]
    if prefix:
        shadowed = [entry for entry in shadowed if entry["name"].startswith(prefix)]
    total, shadowed_total = len(assets), len(shadowed)
    folders = sorted({entry["folder"] for entry in assets} | {""})
    if limit is not None:
        assets, shadowed = assets[:limit], shadowed[:limit]
    return {
        "workspace": ws.name,
        "libraries": library.describe(),
        "assets": assets,
        "folders": folders,
        "shadowed": shadowed,
        "total": total,
        "shadowed_total": shadowed_total,
    }


class KeepRequest(BaseModel):
    name: str = Field(description="The generated file to keep, as the gallery names it")
    asset_name: Optional[str] = Field(
        default=None,
        description="Name to keep it under in the asset library; its own "
        "file name when omitted. May name a folder; the kept file's "
        "extension is assumed when the name has none",
    )
    overwrite: bool = Field(
        default=False, description="Replace an asset already under that name"
    )
    shared: bool = Field(
        default=False,
        description="Keep it in the library every workspace under this "
        "root shares, rather than in this workspace's own",
    )


@router.post(
    "/api/assets/keep",
    status_code=201,
    response_model=Kept,
    response_model_exclude_unset=True,
)
def keep_output_as_asset(
    request: Request, body: KeepRequest, ws: Workspace = Depends(selected_workspace)
):
    """Keep a generated file as an input asset, under a stable name.

    A run's files live under '<workflow>/<run id>/', which is the right
    place for them and the wrong name to build on: 'latest' moves, and a
    pinned run id breaks the moment outputs are pruned. Keeping one
    copies it into the workspace's asset library, where an 'asset:' name
    stays put - which is what turns a generated still or score into an
    input later workflows can rely on.

    Within the workspace, so nothing crosses a namespace, and no bytes
    cross the network: a client that had to download and re-upload a
    multi-gigabyte video to reuse one frame would be paying for the
    round trip twice.
    """
    library = writable_asset_directory(request.app.state, ws, body.shared)

    kept_name = strip_output_prefix(body.name)
    source = resolve_output_file(request.app.state, kept_name, ws.outputs)
    asset_name = body.asset_name or os.path.basename(kept_name)
    # The kept file's own extension when the name carries none, and a
    # refusal when it carries a contradicting one - exactly what the
    # upload route does with its `asset_name`. Without this a kept asset
    # could be written under an extensionless name, which the library
    # listing (which reads by kind) never shows again: the call reported
    # success and the asset was invisible (T014)
    extension = os.path.splitext(os.path.basename(kept_name))[1].lower()
    if not os.path.splitext(asset_name)[1]:
        asset_name = f"{asset_name}{extension}"
    elif os.path.splitext(asset_name)[1].lower() != extension:
        raise HTTPException(
            status_code=400,
            detail=f"asset_name {body.asset_name!r} does not match the "
            f"kept file's kind ({extension or 'no extension'})",
        )
    try:
        asset_name = validate_asset_reference(asset_name)
        destination = validate_path(os.path.join(library, asset_name), library)
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if os.path.exists(destination) and not body.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"{make_ref(ASSET, asset_name)} already exists - pass "
            f"overwrite=true to replace it",
        )

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination):
        os.remove(destination)
    # A hard link first: keeping one frame of a multi-gigabyte render
    # should not cost another copy of it, and both names refer to the
    # same content anyway. Falls back to a copy when the link cannot be
    # made - a different filesystem, or one that has no links
    try:
        os.link(source, destination)
        linked = True
    except OSError:
        shutil.copy2(source, destination)
        linked = False

    # The source run's shot boundaries - carrying bytes without them left
    # a kept multi-shot cut looking like one shot to every probe, with no
    # sign anything was missing (#393). Alongside them, the job and run
    # that wrote the file: keep_output is the one place the server still
    # knows which job produced it, since a plain asset has none (#556)
    job, source_run_id, source_version = job_provenance(
        request.app.state, kept_name, ws
    )
    provenance = (
        {
            "job": job,
            "run_id": source_run_id,
            "version": source_version,
            "workspace": ws.name,
            "source": kept_name,
        }
        if job
        else None
    )
    record_kept_shots(
        os.path.dirname(destination),
        os.path.basename(destination),
        recorded_shots(ws.outputs, kept_name),
        provenance,
    )

    logger.info(f"Kept output {body.name} as {make_ref(ASSET, asset_name)}")
    return {
        "reference": make_ref(ASSET, asset_name),
        "name": asset_name,
        "workspace": ws.name,
        "linked": linked,
        "shared": bool(body.shared),
    }


@router.post("/api/assets/archive")
def archive_assets(
    request: Request, body: ArchiveRequest, ws: Workspace = Depends(selected_workspace)
):
    """Bundle a multi-file asset selection into one zip - the gallery's
    bulk download, for the input side of it.

    Resolved down the same search path a run resolves 'asset:' in, so a
    selection spanning the workspace's own library, the shared one and
    an examples tree downloads as one archive; the library-relative name
    is the entry name, which is the name the 'asset:' reference carries.
    """
    # The search path depends on the workspace, not on the name, so it is
    # built once rather than per name - each root's isdir check would
    # otherwise repeat once per name in the selection for no reason
    library = resolution_library(request.app.state, ws)
    # Stripped and deduped before resolving, so "iris.png" and
    # "iris.png " (or a name repeated by an eager client) become the one
    # zip entry rather than a collision on write
    names = list(dict.fromkeys(n.strip() for n in body.names))
    # Resolved before anything is written, so a bad name in the
    # selection fails the request instead of yielding a partial zip
    paths = [(name, asset_in(name, library)) for name in names]

    return archive_selection(paths, "asset")


@router.delete(
    "/api/assets/{name:path}",
    response_model=AssetDeleted,
    response_model_exclude_unset=True,
)
def delete_asset(
    request: Request, name: str, ws: Workspace = Depends(selected_workspace)
):
    """Permanently remove one file from the asset library.

    Deletes from whichever library on the search path holds it, the
    workspace's own first, so the name deleted is the name 'asset:'
    would have resolved to. An asset a read-only examples tree brought
    with it is not this server's to delete - the same 403 a read-only
    prompt or workflow answers with.

    Not recoverable, and any workflow still carrying that 'asset:'
    reference stops loading. Without this, everything else that writes
    the library (uploads, keep) had no counterpart and a mistake could
    only be cleaned up on the box (T014).
    """
    library = workspace_asset_library(request.app.state, ws)
    if not library.roots():
        raise asset_library_missing()
    try:
        relative = validate_asset_reference(name)
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))

    found = library.find(relative)
    if found:
        path, root = found
        try:
            library.require_writable(root, relative)
        except ReadOnlyLibraryError as refusal:
            raise HTTPException(status_code=403, detail=str(refusal))
        os.remove(path)
        logger.info(f"Deleted {make_ref(ASSET, relative)} ({path})")
        forget_workspace_usage()
        return {
            "name": relative,
            "workspace": ws.name,
            "reference": make_ref(ASSET, relative),
            "deleted": True,
            "origin": root.origin,
        }

    raise HTTPException(status_code=404, detail=f"No such asset: {relative}")
