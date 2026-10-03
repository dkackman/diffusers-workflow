"""The gallery routes: list a workspace's outputs, bundle a selection into a
zip, and delete one output or run directory.

Registered after the media router, so the `/metadata`, `/thumbnail` and
`/download` GETs under `/api/gallery/{name:path}` are matched before the
greedy `DELETE /api/gallery/{name:path}` sits in the table.
"""

import logging
import os
import shutil
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ..api_models import (
    GalleryList,
    OutputDeleted,
)
from ...media import probe_media
from ...runs import (
    MANIFEST_FILE_NAME,
    REALIZED_FILE_NAME,
    is_run_id,
    record_run_versions,
    run_versions,
    split_run_path,
)
from ...security import SecurityError, validate_path
from ...workspace import Workspace, forget_workspace_usage
from ..deps import selected_workspace
from ..outputs import (
    ArchiveRequest,
    delete_run_directory,
    remove_empty_identity_folders,
    absolute_served_url,
    archive_selection,
    iter_gallery_files,
    resolve_output_file,
    served_url,
    strip_output_prefix,
)

logger = logging.getLogger("dw")

router = APIRouter()

# What a run directory holds besides its outputs - the files a run writes
# about itself. A run whose directory holds nothing else is an orphan
# (see _iter_orphan_runs, #170) whatever shape its output would have had.
# job.json is what an export bundle writes, listed defensively.
RUN_BOOKKEEPING_FILES = frozenset({MANIFEST_FILE_NAME, REALIZED_FILE_NAME, "job.json"})


def _gallery_entries(root, ws):
    entries = []
    try:
        files = list(iter_gallery_files(root))
    except OSError:
        files = []
    # One read of each workflow's run ordinals per listing, not per file:
    # a run of fifty files would otherwise re-read the same manifests
    # fifty times
    versions_by_folder = {}

    def _version(folder, run_id):
        if not run_id:
            return None
        if folder not in versions_by_folder:
            versions_by_folder[folder] = run_versions(os.path.join(root, folder))
        return versions_by_folder[folder].get(run_id)

    for relative_name, folder, subfolder, run_id, kind, path in files:
        try:
            stat = os.stat(path)
        except OSError:
            continue
        # File names look like '{workflow}-{step}-{i}.{j}.{k}.ext'; every
        # artifact from one step shares the '{workflow}-{step}' prefix, so
        # the label keeps the full name including the extension rather
        # than truncating at the first dot - otherwise sibling outputs of
        # the same step would show identical, indistinguishable labels,
        # and a step that writes more than one kind of file (e.g. a still
        # plus a video) would lose the extension that tells them apart
        label = os.path.basename(relative_name)
        output_path = f"/outputs/{quote(relative_name)}"
        entry = {
            "name": relative_name,
            "folder": folder,
            "subfolder": subfolder,
            # Which run wrote it, and that run's ordinal among this
            # workflow's runs - the 'v4' a person sees in the grid
            # and an agent says out loud. Two runs write the same
            # basename, so `label` cannot tell them apart and
            # `name` is too long to quote. None under the flat
            # layout, which has no runs to number
            "run_id": run_id,
            "version": _version(folder, run_id),
            # Quoted (slashes kept literal): a name carrying '#', '?'
            # or '%' would otherwise break the src the gallery
            # renders it into. The mtime still rides along for cache
            # busting when a file's content changes without its name
            # changing (e.g. a manual overwrite outside the engine) -
            # normal reruns get a fresh name instead, see
            # dw/writers.py's output_file_path
            "url": served_url(output_path, ws, int(stat.st_mtime)),
            "kind": kind,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "label": label,
        }
        absolute_url = absolute_served_url(output_path, ws, int(stat.st_mtime))
        if absolute_url is not None:
            entry["absolute_url"] = absolute_url
        entries.append(entry)
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries


def _iter_orphan_runs(root):
    """Run directories under `root` holding nothing but their own
    bookkeeping (RUN_BOOKKEEPING_FILES) - a run whose output was deleted
    before #134's by-name `delete_output`, or one that failed before
    writing anything. Yields (name, mtime) where `name` is the
    `<identity>/<run id>` string `delete_output` already accepts (#170).

    By what is absent, not by extension: a `text`-shape run writes .txt
    and a `utility`-shape run may write nothing the gallery lists, and
    neither is junk. This call only lists; deciding whether an entry is
    junk stays a human/agent call before `delete_output` is invoked."""
    for current, dirs, _names in os.walk(root):
        if not is_run_id(os.path.basename(current)):
            continue
        # A run directory holds no run directories of its own
        dirs[:] = []
        # A dotfile is not output either: a .DS_Store Finder left behind
        # would otherwise make the run permanently non-orphan
        has_output = any(
            name not in RUN_BOOKKEEPING_FILES and not name.startswith(".")
            for _sub_current, _sub_dirs, sub_names in os.walk(current)
            for name in sub_names
        )
        if has_output:
            continue
        try:
            mtime = os.stat(current).st_mtime
        except OSError:
            continue
        name = os.path.relpath(current, root).replace(os.sep, "/")
        yield (name, mtime)


def _orphan_entries(root):
    entries = [
        {"name": name, "mtime": mtime} for name, mtime in _iter_orphan_runs(root)
    ]
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries


@router.get(
    "/api/gallery", response_model=GalleryList, response_model_exclude_unset=True
)
def gallery(
    limit: int = 200,
    offset: int = 0,
    folder: Optional[str] = None,
    subfolder: Optional[str] = None,
    only_orphans: bool = False,
    version: Optional[int] = None,
    media: bool = False,
    ws: Workspace = Depends(selected_workspace),
):
    """A page of media files in the output directory, newest first.
    Stateless by design - the gallery survives server restarts because
    it reads the directory tree, not job history. 'folders' lists every
    distinct workflow folder present (over the whole directory, not just
    this page), for the UI's folder filter - a run id is not a folder of
    its own, so a workflow's runs group together; '' stands for files
    saved directly at the output root, and is itself always a member so
    that folder-less outputs stay selectable once anything is nested.
    'subfolders' is the other axis, over the whole directory the same
    way: the in-run subfolders steps wrote into ('final',
    'intermediate'), '' for files at a run's root. `folder` and
    `subfolder` filter independently and intersect when both are given.
    `version` narrows to the runs holding that ordinal - with `folder`,
    the one run "v4" names; without it, that run of every workflow.

    `only_orphans=true` inverts the whole call: instead of media files,
    it returns run directories holding nothing but their own
    bookkeeping (manifest.json, workflow.json, job.json) as `runs`,
    each `{name, mtime}` - a run that wrote any file at all, a
    text-shape prompt or a utility's side output included, is not
    listed. `folder`/`subfolder` and the `folders`/`subfolders` facets
    do not apply in this mode, since an orphan run has no file to
    carry either. `name` is exactly what `DELETE /api/gallery/{name}`
    accepts, so listing and deleting an orphan is a two-call round
    trip (#170).

    `media=true` adds `duration_seconds` to each audio/video entry,
    probed the same way `get_gallery_metadata` reports it - which two
    takes of the same workflow otherwise have no way to be told apart
    by, since size and mtime are misleading proxies for length (#356).
    Off by default and bounded by `limit`: only the page actually
    returned is probed, not the whole listing, so the cost of asking
    stays proportional to the page size rather than the library size."""
    if only_orphans:
        entries = _orphan_entries(ws.outputs)
        offset = max(0, offset)
        limit = max(0, limit)
        page = entries[offset : offset + limit]
        # A different answer from the file listing GalleryList declares;
        # it goes out as built, outside that model
        return JSONResponse(
            {
                "runs": page,
                "total": len(entries),
                "offset": offset,
                "limit": limit,
                "workspace": ws.name,
            }
        )
    entries = _gallery_entries(ws.outputs, ws)
    folders = sorted({e["folder"] for e in entries} | {""})
    subfolders = sorted({e["subfolder"] for e in entries} | {""})
    if folder is not None:
        entries = [e for e in entries if e["folder"] == folder]
    if subfolder is not None:
        entries = [e for e in entries if e["subfolder"] == subfolder]
    if version is not None:
        entries = [e for e in entries if e["version"] == version]
    offset = max(0, offset)
    limit = max(0, limit)
    page = entries[offset : offset + limit]
    if media:
        for entry in page:
            if entry["kind"] not in ("audio", "video"):
                continue
            probed = probe_media(os.path.join(ws.outputs, entry["name"]))
            if probed is not None:
                entry["duration_seconds"] = probed.get("duration_seconds")
    return {
        "files": page,
        "total": len(entries),
        "offset": offset,
        "limit": limit,
        "folders": folders,
        "subfolders": subfolders,
        "workspace": ws.name,
    }


@router.post("/api/gallery/archive")
def archive_outputs(
    request: Request, body: ArchiveRequest, ws: Workspace = Depends(selected_workspace)
):
    """Bundle a multi-file gallery selection into one zip. A browser
    cannot zip on its own and throttles a burst of single downloads, so
    the whole selection has to arrive as one file. Written to a temp
    file rather than memory - a selection of videos does not fit in
    RAM - and unlinked once the response has been sent."""
    # Resolved before anything is written, so a bad name in the
    # selection fails the request instead of yielding a partial zip
    # the gallery-relative name is the entry name, so a workflow's output
    # subfolders stay intact inside the download
    state = request.app.state
    names = [strip_output_prefix(name) for name in body.names]
    paths = [(name, resolve_output_file(state, name, ws.outputs)) for name in names]

    return archive_selection(paths, "output")


# What a run directory holds besides its media: the engine writes them to
# describe the run, and the gallery - which lists media - never shows them
RUN_SIDECARS = (MANIFEST_FILE_NAME, REALIZED_FILE_NAME)


def _prune_empty_run_directory(name, root):
    """Drop the run directory a just-deleted output belonged to, once no
    media is left in it.

    A run writes `manifest.json` and `workflow.json` beside its files, and
    nothing in the gallery addresses either one. Deleting every output of a
    run therefore used to leave the directory behind forever: a consumer
    that removed everything it made still could not put a workspace back
    the way it found it, and nothing it could call would even show the
    residue (#134). Tying the sidecars' lifetime to the outputs they
    describe is what makes "delete what you made" true.

    Only the sidecars may remain - any other leftover file means something
    is still there to describe, and the directory stays.

    Returns:
        The run id swept, or None if nothing was
    """
    identity, run_id, _ = split_run_path(name)
    if not run_id:
        # The flat layout writes no run directory and no sidecars
        return None
    relative = f"{identity}/{run_id}" if identity else run_id
    try:
        run_dir = validate_path(os.path.join(root, relative), root, allow_create=False)
    except SecurityError:
        return None
    if not os.path.isdir(run_dir):
        return None

    for directory, _subdirectories, files in os.walk(run_dir):
        for file_name in files:
            if directory == run_dir and file_name in RUN_SIDECARS:
                continue
            return None

    # Pin the siblings' numbers first: a run that predates versions is
    # ranked, and removing one ahead of it would renumber it
    record_run_versions(os.path.dirname(run_dir))
    shutil.rmtree(run_dir, ignore_errors=True)
    # And the identity folders above it, while they are empty - a swept
    # workspace should not keep one directory per workflow it once ran
    remove_empty_identity_folders(run_dir, root)
    logger.info(f"Swept empty run directory {relative}")
    return run_id


def _run_directory(name, root):
    """The run directory `<identity>/<run id>` names, or None.

    A run that failed before it wrote anything still has a directory and a
    manifest, and no gallery name addresses it - so the name of the
    directory itself is the only handle there can be (#134).
    """
    parts = [part for part in (name or "").split("/") if part]
    if not parts or not is_run_id(parts[-1]):
        return None
    try:
        path = validate_path(
            os.path.join(root, "/".join(parts)), root, allow_create=False
        )
    except SecurityError:
        return None
    return path if os.path.isdir(path) else None


@router.delete(
    "/api/gallery/{name:path}",
    response_model=OutputDeleted,
    response_model_exclude_unset=True,
)
def delete_output(
    request: Request, name: str, ws: Workspace = Depends(selected_workspace)
):
    """Remove one file from the output directory.

    When that was the last media file of its run, the run directory goes
    with it, sidecars included. `name` may also be a run directory
    (`<identity>/<run id>`), which removes the whole run - the only handle
    on a run that failed before it wrote any media (#134).
    """
    name = strip_output_prefix(name)
    run_dir = _run_directory(name, ws.outputs)
    if run_dir is not None:
        return {
            "name": name,
            "deleted": True,
            "run_swept": delete_run_directory(run_dir, ws.outputs),
        }

    path = resolve_output_file(request.app.state, name, ws.outputs)
    os.remove(path)
    logger.info(f"Deleted output file {name}")
    swept = _prune_empty_run_directory(name, ws.outputs)
    forget_workspace_usage()
    return {"name": name, "deleted": True, "run_swept": swept}
