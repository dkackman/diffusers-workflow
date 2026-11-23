"""The gallery routes: list a workspace's outputs, bundle a selection into a
zip, delete one output or run directory, and the two marks a person puts on
one: `PUT /api/notes` (the note on an output, a run, an asset or a workspace;
dw/notes.py holds the stores) and `PUT /api/stars`.

Registered after the media router, so the `/metadata`, `/thumbnail` and
`/download` GETs under `/api/gallery/{name:path}` are matched before the
greedy `DELETE /api/gallery/{name:path}` sits in the table.
"""

import logging
import os
import shutil
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..api_models import (
    GalleryList,
    NoteSet,
    OutputDeleted,
    StarSet,
)
from ...assets import is_asset_reference
from ...library import ReadOnlyLibraryError
from ...media import probe_media
from ...notes import (
    MAX_NOTE_LENGTH,
    NOTES_FILE_NAME,
    NoteTooLong,
    clean_note,
    drop_file_note,
    effective_note,
    read_notes,
    set_file_note,
    set_file_starred,
    set_run_note,
    set_workspace_note,
)
from ...references import ASSET, WORKSPACE, is_ref, ref_name
from ...roles import ROLES
from ...runs import (
    MANIFEST_FILE_NAME,
    REALIZED_FILE_NAME,
    is_run_id,
    record_run_versions,
    split_run_path,
)
from ...security import (
    InvalidInputError,
    SecurityError,
    validate_asset_reference,
    validate_ledger_key,
    validate_path,
)
from ...workspace import DEFAULT_WORKSPACE_NAME, Workspace, forget_workspace_usage
from ..changes import (
    ASSETS,
    GALLERY,
    NO_CHANGE,
    WORKSPACES,
    Change,
    ChangeRoute,
    pending_change,
    writes,
)
from ..deps import (
    asset_library_missing,
    internal_error,
    selected_workspace,
    workspace_for,
    workspace_root,
)
from ..outputs import (
    ArchiveRequest,
    delete_run_directory,
    remove_empty_identity_folders,
    absolute_served_url,
    archive_selection,
    iter_gallery_files,
    resolve_output_file,
    run_directory,
    consumed_in,
    output_annotations,
    served_url,
    split_run_name,
    strip_output_prefix,
    validated_run_split,
    workspace_asset_library,
)

logger = logging.getLogger("dw")

router = APIRouter(route_class=ChangeRoute)

# What a run directory holds besides its outputs - the files a run writes
# about itself. A run whose directory holds nothing else is an orphan
# (see _iter_orphan_runs, #170) whatever shape its output would have had.
# job.json is what an export bundle writes, listed defensively.
RUN_BOOKKEEPING_FILES = frozenset(
    {MANIFEST_FILE_NAME, REALIZED_FILE_NAME, "job.json", NOTES_FILE_NAME}
)


def _gallery_entries(root, ws, consumed=frozenset()):
    entries = []
    try:
        files = list(iter_gallery_files(root))
    except OSError:
        files = []
    # The per-listing cache output_annotations reads each run's notes,
    # stars and versions through, once rather than once per file
    cache = {}

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
            "version": None,
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
        # None under the flat layout, and for a file no manifest lists
        split = split_run_name(root, relative_name) if run_id else None
        entry["role"] = None
        if split:
            annotated = output_annotations(
                split[0], split[1], relative_name in consumed, cache
            )
            # The listing does not report whose note it is, nor the run's
            annotated.pop("note_source", None)
            annotated.pop("run_note", None)
            entry.update(annotated)
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


def _check_gallery_filters(role, request_id, item):
    """Refuse a gallery query's bad filters: an unknown role is a 400; an
    `item` without a `request`, or either as a malformed ledger key, a 422."""
    if role:
        unknown = sorted(set(role) - set(ROLES))
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown role {unknown}; roles are {list(ROLES)}",
            )
    if item and not request_id:
        raise HTTPException(status_code=422, detail="item needs request")
    for value, what in ((request_id, "request"), (item, "item")):
        if value:
            try:
                validate_ledger_key(value, what)
            except InvalidInputError as e:
                raise HTTPException(status_code=422, detail=str(e))


@router.get(
    "/api/gallery", response_model=GalleryList, response_model_exclude_unset=True
)
def gallery(
    request: Request,
    limit: int = 200,
    offset: int = 0,
    folder: Optional[str] = None,
    subfolder: Optional[str] = None,
    only_orphans: bool = False,
    version: Optional[int] = None,
    media: bool = False,
    role: Optional[list[str]] = Query(None),
    request_id: Optional[str] = Query(
        default=None,
        alias="request",
        description="Only the files of runs whose job is linked to this "
        "requests-ledger request in this workspace",
    ),
    item: Optional[str] = Query(
        default=None,
        description="With `request`: only the files of runs whose job is "
        "linked to this item of it",
    ),
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
    stays proportional to the page size rather than the library size.

    Each entry carries `role`, `final` or `intermediate` (null for a
    flat-layout file or one no manifest lists); a file a later run read
    lists as intermediate unless its step declared it final
    (dw/roles.py rule 4). A run that recorded no role takes the role its
    file's subfolder names, then rule 4; with neither it is null. `role`
    is a repeatable filter; a null role counts as final. Omitted, every file is listed."""
    _check_gallery_filters(role, request_id, item)
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
    # Read once for the listing: which files a later run read (#928)
    entries = _gallery_entries(ws.outputs, ws, consumed_in(request.app.state, ws))
    folders = sorted({e["folder"] for e in entries} | {""})
    subfolders = sorted({e["subfolder"] for e in entries} | {""})
    if folder is not None:
        entries = [e for e in entries if e["folder"] == folder]
    if subfolder is not None:
        entries = [e for e in entries if e["subfolder"] == subfolder]
    if version is not None:
        entries = [e for e in entries if e["version"] == version]
    if role:
        wanted = set(role)
        entries = [e for e in entries if (e.get("role") or "final") in wanted]
    if request_id:
        # A run belongs to the request its job was linked to; the history's
        # run_id column is the map from one to the other
        runs = request.app.state.job_manager.history.run_ids_for_request(
            ws.name, request_id, item
        )
        entries = [e for e in entries if e["run_id"] in runs]
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
@NO_CHANGE("builds a download or a view; writes nothing a page lists")
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
RUN_SIDECARS = (MANIFEST_FILE_NAME, REALIZED_FILE_NAME, NOTES_FILE_NAME)


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


@router.delete(
    "/api/gallery/{name:path}",
    response_model=OutputDeleted,
    response_model_exclude_unset=True,
)
@writes(GALLERY)
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
    run_dir = run_directory(name, ws.outputs)
    if run_dir is not None:
        swept = delete_run_directory(run_dir, ws.outputs)
        return {"name": name, "deleted": True, "run_swept": swept}

    path = resolve_output_file(request.app.state, name, ws.outputs)
    os.remove(path)
    logger.info(f"Deleted output file {name}")
    split = split_run_name(ws.outputs, name)
    if split is not None:
        # The note describes the file; it goes with it. Best effort: the file
        # is already gone, so a note that cannot be rewritten is not a failure
        try:
            note_dir = validate_path(split[0], ws.outputs, allow_create=False)
            drop_file_note(note_dir, split[1])
        except SecurityError:
            pass
        except OSError as e:
            logger.warning(f"Could not drop the note for {name}: {e}")
    swept = _prune_empty_run_directory(name, ws.outputs)
    forget_workspace_usage()
    return {"name": name, "deleted": True, "run_swept": swept}


class NoteRequest(BaseModel):
    target: str = Field(
        description="What the note is on: a gallery file name, a run directory "
        "`<identity>/<run id>`, `asset:<name>` or `workspace:<name>`"
    )
    note: Optional[str] = Field(
        default=None,
        description=f"At most {MAX_NOTE_LENGTH} characters; null or '' clears it",
    )


def _workspace_note_target(state, name):
    """The root a `workspace:<name>` target writes workspace.json into."""
    if not name:
        raise HTTPException(
            status_code=400, detail="Name a workspace: workspace:<name>"
        )
    if name == DEFAULT_WORKSPACE_NAME:
        root = state.default_workspace.root
        if not root:
            raise HTTPException(
                status_code=409,
                detail="This server has no workspace root, so the default "
                "workspace has nowhere to keep a note",
            )
        return root
    # 400 for a name a workspace cannot take, 404 for one that is not there
    selected = workspace_for(state, name)
    try:
        return validate_path(
            selected.root, workspace_root(state).root, allow_create=False
        )
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _asset_note_target(state, ws, name):
    """(directory, file name, library root) of the asset an `asset:<name>`
    target names - resolved the way the asset delete resolves it, read-only
    roots refused. The root says which library holds it (`Change.in_library`)."""
    library = workspace_asset_library(state, ws)
    if not library.roots():
        raise asset_library_missing()
    try:
        relative = validate_asset_reference(name)
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    found = library.find(relative)
    if not found:
        raise HTTPException(status_code=404, detail=f"No such asset: {relative}")
    path, root = found
    try:
        library.require_writable(root, relative)
    except ReadOnlyLibraryError as refusal:
        raise HTTPException(status_code=403, detail=str(refusal))
    return os.path.dirname(path), os.path.basename(path), root


def resolve_note_target(state, ws, target):
    """(directory, name, kind, root) for a note target, kind one of
    "workspace", "run", "file" and "asset" (a file in its library directory,
    `root` being the `LibraryRoot` that holds it; None for any other). Every
    directory is a validator's return value against a non-None root, or the
    server's own configured default root. Raises the route's 400/403/404/409.
    """
    if is_ref(WORKSPACE, target):
        name = ref_name(WORKSPACE, target).strip()
        return _workspace_note_target(state, name), None, "workspace", None
    if is_asset_reference(target):
        name = ref_name(ASSET, target).strip()
        directory, file_name, root = _asset_note_target(state, ws, name)
        return directory, file_name, "asset", root
    name = strip_output_prefix(target)
    run_dir = run_directory(name, ws.outputs)
    if run_dir is not None:
        return run_dir, None, "run", None
    path = resolve_output_file(state, name, ws.outputs)
    split = validated_run_split(ws.outputs, path)
    if split is None:
        raise HTTPException(
            status_code=409,
            detail=f"{name} is not in a run directory, and notes need a run directory",
        )
    return split[0], split[1], "file", None


@router.put("/api/notes", response_model=NoteSet, response_model_exclude_unset=True)
@writes(ASSETS, GALLERY, WORKSPACES)
def put_note(
    request: Request,
    body: NoteRequest,
    ws: Workspace = Depends(selected_workspace),
    change: Change = Depends(pending_change),
):
    """Set or clear the note on a run, a file, an asset or a workspace.

    `target` is a gallery file name (that file's own note), a run directory
    `<identity>/<run id>` (the run note every file of the run falls back to),
    `asset:<name>` or `workspace:<name>` ('default' for the root's own). The
    answer is the target's effective note after the write: a file with no
    note of its own answers with its run's, and `note_source` says which.
    """
    try:
        note = clean_note(body.note)
    except (NoteTooLong, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    directory, name, kind, root = resolve_note_target(
        request.app.state, ws, body.target
    )
    try:
        if kind == "workspace":
            stored, source = set_workspace_note(directory, note), None
        elif kind == "run":
            stored, source = set_run_note(directory, note), "run"
        else:
            set_file_note(directory, name, note)
            stored, source = effective_note(read_notes(directory), name)
    except OSError:
        raise internal_error(f"Could not write the note on {body.target!r}")
    if kind == "workspace":
        # The sidebar lists workspace notes: the server-wide counter
        change.only(WORKSPACES)
    elif kind == "asset":
        change.only(ASSETS).in_library(ws, root)
    else:
        change.only(GALLERY)
    return {
        "target": body.target,
        "note": stored,
        "note_source": source if stored else None,
    }


class StarRequest(BaseModel):
    name: str = Field(description="A gallery file name")
    starred: bool


@router.put("/api/stars", response_model=StarSet)
@writes(GALLERY)
def put_star(
    request: Request, body: StarRequest, ws: Workspace = Depends(selected_workspace)
):
    """Star or unstar one output file - the "keep" mark a person culls a run
    with. It lives in the run's notes.json beside the notes, so it is
    resolved exactly as a file note is, and only a file in a run directory
    can carry one."""
    if is_ref(WORKSPACE, body.name) or is_asset_reference(body.name):
        raise HTTPException(
            status_code=400, detail="Only an output file can be starred"
        )
    directory, name, kind, _root = resolve_note_target(request.app.state, ws, body.name)
    if kind != "file":
        raise HTTPException(
            status_code=400, detail="Only an output file can be starred"
        )
    try:
        starred = set_file_starred(directory, name, body.starred)
    except OSError:
        raise internal_error(f"Could not write the star on {body.name!r}")
    return {"name": body.name, "starred": starred}
