"""Output and asset resolution: where a workspace's asset library is searched,
how a gallery or asset name becomes a file, how a served file is named as a
URL, and the zip download the archive and export routes share.

The gallery, media, assets and files routers all reach it. Functions take the
app's `state` explicitly, so two apps in one process resolve independently.
"""

import logging
import os
import tempfile
import zipfile
from datetime import datetime
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .. import settings
from ..assets import ASSET_PREFIX
from ..library import (
    ASSETS_KIND,
    WORKSPACE_ORIGIN,
    LibraryPath,
    LibraryRoot,
    library_path,
)
from ..runs import OUTPUT_PREFIX, is_output_reference, run_versions, split_run_path
from ..security import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    InvalidInputError,
    PathTraversalError,
    SecurityError,
    contained,
    validate_asset_reference,
    validate_path,
)

logger = logging.getLogger("dw")

# Built from the security layer's allowlists so a new format is added
# exactly once - the gallery had already drifted (.bmp, .mkv, .mov)
MEDIA_KINDS = {
    **{ext: "image" for ext in ALLOWED_IMAGE_EXTENSIONS},
    **{ext: "video" for ext in ALLOWED_VIDEO_EXTENSIONS},
    **{ext: "audio" for ext in ALLOWED_AUDIO_EXTENSIONS},
    # Not in the security allowlists above (nothing loads a .txt back
    # into a pipeline, so it is not a path a run reads), but a
    # text-shape run's deliverable is a real output and belongs in the
    # gallery like any other kind (#238)
    ".txt": "text",
}

# The allowlist members that are not already-compressed containers -
# everything else in MEDIA_KINDS deflates for about nothing, so it is
# stored instead (see zip_download)
RAW_MEDIA_EXTENSIONS = {".bmp", ".wav"}

# A generous ceiling rather than a real limit - it exists so a
# malformed client cannot ask the server to zip the whole directory
MAX_ARCHIVE_FILES = 1000


class ArchiveRequest(BaseModel):
    names: list[str] = Field(min_length=1, max_length=MAX_ARCHIVE_FILES)


def common_assets(ws):
    """The library every workspace under this root shares, or None.

    A recurring cast is not the property of the workspace that first
    uploaded it, and a fresh workspace could not see it at all - the
    prompt library has been shared from the start for the same reason.
    """
    return getattr(ws, "common_assets", None)


def asset_library(state, ws, primary=None):
    """The asset search path of one workspace: its own library, then the
    one shared by every workspace under this root, then the read-only
    ones an --examples-dir tree brought with it. The same path the worker
    resolves 'asset:' over (`library_path_from_env`), so what the browser
    lists is what a job would load.

    `primary` names the front when it is not the workspace's own - the
    directory a job actually ran against. Roots that are not directories
    yet are left off: there is nothing there to list or serve.
    """
    return library_path(
        ASSETS_KIND, ws, state.examples_dirs, primary=primary
    ).existing()


def resolution_library(state, ws):
    """`asset_library(state, ws)`, falling back to the workspace's own
    (possibly nonexistent) library when the search path is empty.

    A caller resolving a name still needs *somewhere* to fail against:
    with no root at all the 404 would name no directory, leaving the
    caller to guess where it looked. Naming the workspace's own
    directory keeps the failure pointing at the library the caller
    thinks they're working in, even when that library hasn't been
    created yet.

    A server configured with no asset library at all has nothing to point
    at either, and `asset_in` turns the resulting empty path into the "no
    asset library" 404.
    """
    library = asset_library(state, ws)
    if library.roots() or not ws.assets:
        return library
    return LibraryPath(ASSETS_KIND, [LibraryRoot(ws.assets, WORKSPACE_ORIGIN, True)])


def asset_library_for_job(state, job_id, ws):
    """The asset search path a job's own run used, for export: its spec's
    `asset_dir` (or the historical row's), then the shared and read-only
    example libraries - the same path `asset_library` builds for the
    selected workspace, but fronted by wherever the job actually ran
    rather than by the workspace the caller happens to be scoped to now. A
    job that ran in one workspace while the caller exports it scoped to
    another must still find its own 'asset:' files, not the other
    workspace's.

    Falls back to `asset_library(state, ws)` when the job carries no
    asset_dir of its own - an inline-workflow job, or one recorded before
    this field existed."""
    job = state.job_manager.get(job_id)
    if job is None:
        return asset_library(state, ws)
    spec = (job.get("spec") or {}) if isinstance(job, dict) else job.spec
    return asset_library(state, ws, primary=spec.get("asset_dir"))


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


def strip_output_prefix(name):
    """A gallery name, accepting the way a workflow argument would
    reference it ('output:<name>', #356) as well as the bare form
    every gallery listing reports. `asset:` already gets this courtesy
    on this same endpoint family (`is_asset_reference` below); a caller
    who spelled a name by copying an `output:` reference used to be met
    with a wrong-looking "path does not exist" instead, because the
    prefix was joined straight into the path rather than stripped first.

    Applied once, at the top of every route that takes a gallery
    `name`, so the rest of that route - job lookups, run-path parsing,
    the file it echoes back - sees the same bare name `_output_file`
    resolves, rather than resolving the file correctly while a sibling
    lookup keyed on the untouched string quietly misses.
    """
    if is_output_reference(name):
        return name.removeprefix(OUTPUT_PREFIX).strip()
    return name


def resolve_output_file(state, name, root=None):
    """A file inside a workspace's output directory, or a 404 - never
    outside it."""
    root = root or state.job_manager.output_dir
    name = strip_output_prefix(name)
    try:
        path = validate_path(
            os.path.join(root, name),
            root,
            allow_create=False,
        )
    except PathTraversalError:
        # PathTraversalError's own message can embed the resolved
        # *absolute* server path (dw/security.py validate_path, the
        # containment branch) - useful in a log, not in a response a
        # remote caller reads. Still say *why* it was refused, since a
        # caller needs to tell "this name would have escaped the
        # workspace" from "this name is simply wrong" (#310) - the
        # distinction #134 pinned and a later leak fix (#247) collapsed.
        raise HTTPException(
            status_code=404,
            detail=f"Unknown file: {name} - path contains a disallowed pattern",
        )
    except InvalidInputError:
        raise HTTPException(
            status_code=404, detail=f"Unknown file: {name} - path does not exist"
        )
    except SecurityError:
        raise HTTPException(status_code=404, detail=f"Unknown file: {name}")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Unknown file")
    return path


def asset_in(name, library):
    """The file a bare asset name has on this search path, or a 404.

    `asset_file` with the path already in hand, for a caller resolving many
    names against the one workspace: the path is built once rather than
    once per name, and `find` confines each candidate to its own root - a
    symlink that leaves it is a miss for that root, not a 500, and on a
    total miss the same 404 every other miss gets.
    """
    try:
        validate_asset_reference(name)
    except SecurityError as e:
        raise HTTPException(status_code=404, detail=str(e))
    found = library.find(name)
    if found:
        return found[0]
    roots = [root.root for root in library.roots()]
    if not roots:
        detail = f"Unknown asset {name!r}: this workspace has no asset library"
    else:
        detail = f"Unknown asset {name!r}: not found in {', '.join(roots)}"
    raise HTTPException(status_code=404, detail=detail)


def asset_file(state, reference, ws):
    """The file an 'asset:' reference names in this workspace, or a 404.

    Looked for down the same search path a run resolves 'asset:' in
    (`asset_library`), so what the API can read is what a job would load.
    A miss names every root that was searched, so the caller sees
    their own workspace library among them rather than just the last
    (often an examples directory they never wrote to).
    """
    return asset_in(
        reference.removeprefix(ASSET_PREFIX).strip(),
        resolution_library(state, ws),
    )


def job_provenance(state, name, ws):
    """The job that wrote an output file, and which run and version -
    `gallery_metadata`'s output branch and `keep_output_as_asset` (#556)
    both need this, the latter so it can carry it into the kept asset's
    sidecar before the run directory it came from is pruned."""
    try:
        # Scoped to this workspace: two workspaces can each write a
        # file with the same relative name, and an unscoped lookup
        # could attribute this one to the wrong workspace's job
        job = state.job_manager.history.job_for_file(name, workspace=ws.name)
    except Exception:
        job = None
    run_id, version = "", None
    folder, run_id, _subfolder = split_run_path(name)
    if run_id:
        try:
            identity_dir = validate_path(os.path.join(ws.outputs, folder), ws.outputs)
        except SecurityError:
            identity_dir = None
        if identity_dir:
            version = run_versions(identity_dir).get(run_id)
    return job, run_id, version


def static_files_for(state, root):
    """The StaticFiles instance bound to one root, built on first use and
    cached on `state.static_files_by_root` (one cache per app)."""
    cache = state.static_files_by_root
    files = cache.get(root)
    if files is None:
        files = StaticFiles(directory=root)
        cache[root] = files
    return files


def iter_gallery_files(root, group_runs=True):
    """Every media file under a directory tree. Yields (relative_name,
    folder, subfolder, kind, path) - relative_name always uses '/' so
    it round-trips through a URL the same way on every platform.

    With group_runs (the gallery's own use, over the output directory):
    recurses into the per-workflow subfolders (dw/workflow.py's
    effective_output_dir writes each run under '<workflow
    identity>/<run id>/', and mirrors a workflow's position under a
    'workflows' tree in the flat layout). The folder a file is grouped
    under is the identity - the run id is dropped, so a workflow run
    fifty times is one folder in the filter, not fifty - and whatever
    followed the run id is the subfolder, the part of the run a step's
    'result.subfolder' put it in ('final', 'intermediate'). A flat-layout
    path has no run id to anchor on, so its subfolder is '' and its
    whole directory is the folder, as it always was.

    Without it (the asset library's use, which has no run ids to strip):
    folder is just the plain relative directory and subfolder is ''.

    A file symlink resolving outside root is skipped: os.walk lists it
    among the names, and the entry would carry the target's size and
    mtime. A linked directory is never descended (os.walk's default)."""
    for current, _dirs, names in os.walk(root):
        rel_root = os.path.relpath(current, root)
        directory = "" if rel_root == "." else rel_root.replace(os.sep, "/")
        for name in names:
            extension = os.path.splitext(name)[1].lower()
            kind = MEDIA_KINDS.get(extension)
            if kind is None:
                continue
            path = os.path.join(current, name)
            if not contained(path, root):
                continue
            relative_name = name if not directory else f"{directory}/{name}"
            if group_runs:
                folder, run_id, subfolder = split_run_path(relative_name)
            else:
                folder, subfolder, run_id = directory, "", ""
            yield (
                relative_name,
                folder,
                subfolder,
                run_id,
                kind,
                path,
            )


def zip_download(entries, filename):
    """Bundle (arcname, path) pairs into a zip and serve it as a download.

    The archive is a temp file rather than memory - a selection of videos
    does not fit in RAM - unlinked once the response has been sent. The
    three routes that hand back a zip share this so the cleanup contract
    lives in one place: nothing has attached the background unlink while
    the archive is being written, so a failure there has to unlink on the
    way out or leak a half-written file into tmp.
    """
    handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        with handle:
            with zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as archive:
                for arcname, path in entries:
                    # ZipFile.write follows a symlink and archives the
                    # target's bytes; nothing the server writes is one
                    if os.path.islink(path):
                        continue
                    extension = os.path.splitext(path)[1].lower()
                    # A file in MEDIA_KINDS but not RAW_MEDIA_EXTENSIONS
                    # is an already-compressed container - deflating it
                    # buys about nothing for a full CPU pass the caller
                    # waits through (the response doesn't start until the
                    # temp file is complete), so it is stored instead.
                    # Everything else - .json, .md, .txt, .bmp, .wav, an
                    # unrecognized extension - deflates, including the
                    # export zip's text files. ".txt" is in MEDIA_KINDS
                    # (kind "text", #238) but is plain text, not an
                    # already-compressed container, so it stays out of
                    # this policy the same way .json and .md do
                    kind = MEDIA_KINDS.get(extension)
                    stored = (
                        kind is not None
                        and kind != "text"
                        and extension not in RAW_MEDIA_EXTENSIONS
                    )
                    archive.write(
                        path,
                        arcname=arcname,
                        compress_type=(
                            zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
                        ),
                    )
    except BaseException:
        os.unlink(handle.name)
        raise

    return FileResponse(
        handle.name,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(os.unlink, handle.name),
    )


def archive_selection(entries, kind):
    """`_zip_download` plus the one tail the two archive routes shared:
    a timestamped `dw-<kind>s-*.zip` name and a log line naming the
    count. Logged after the archive is written, not before, so a write
    that fails partway (a bad path slipping past resolution, a full
    disk) doesn't log a success that didn't happen.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    response = zip_download(entries, f"dw-{kind}s-{stamp}.zip")
    logger.info(f"Archived {len(entries)} {kind} files")
    return response
