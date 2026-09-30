"""FastAPI application exposing the workflow engine.

All state lives in the JobManager; this module is routing, validation and
SSE framing. Everything path-shaped goes through dw.security validators.
Interactive API docs are served at /docs (OpenAPI at /openapi.json).
"""

import os
import base64
import mimetypes
import shutil
import io
import zipfile
import tempfile
import json
import re
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from urllib.parse import quote
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from filelock import FileLock
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.routing import Route
from starlette.background import BackgroundTask

from ..references import ASSET, make_ref
from ..security import (
    MAX_DECODE_PIXELS,
    contained,
    validate_asset_reference,
    validate_path,
    validate_output_path,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    InvalidInputError,
    PathTraversalError,
    SecurityError,
)
from ..assets import (
    ASSET_PREFIX,
    is_asset_reference,
)
from .observed_cost import ObservedCosts
from .exports import export_directory
from .assess import assess, unknown_probe
from ..result import read_embedded_metadata
from ..media_info import probe_media
from ..media_audio import (
    MAX_INLINE_AUDIO_BYTES,
    NoSoundtrack,
    audio_shape,
    extract_audio,
    media_duration,
    projected_wav_base64_size,
)
from ..media_frames import (
    contact_sheet,
    frames_at,
    resolve_crop_box,
    seam_tiles,
    video_shape,
)
from ..hub_cache import DownloadManager
from ..runs import (
    MANIFEST_FILE_NAME,
    OUTPUT_PREFIX,
    REALIZED_FILE_NAME,
    run_lock_path,
    is_output_reference,
    is_run_id,
    kept_provenance,
    record_kept_shots,
    record_run_versions,
    run_versions,
    recorded_shots,
    shots_beside,
    split_run_path,
)
from ..workspace import (
    ASSETS_SUBDIR,
    PROMPTS_SUBDIR,
    ConfiguredWorkspace,
    Workspace,
    example_libraries,
    forget_workspace_usage,
)
from ..workflow_sources import (
    COMMON_ORIGIN,
    EXAMPLES_ORIGIN,
    WORKSPACE_ORIGIN,
)
from .deps import selected_workspace
from .outputs import (
    absolute_served_url,
    asset_roots,
    common_assets,
    resolution_roots,
    served_url,
)
from .routes import include_routers
from .http_security import (
    ACTIVE_DOCUMENT_TYPES,
    install_middleware,
    query_token_ok,
)
from .jobs import (
    JobManager,
)
from .netinfo import LOOPBACK_HOSTS, WILDCARD_HOSTS
from .updater import DiffusersUpdater

logger = logging.getLogger("dw")

# How long one SSE poll waits for a new event before checking liveness
SSE_POLL_SECONDS = 1.0


# What a run directory holds besides its outputs - the files a run writes
# about itself. A run whose directory holds nothing else is an orphan
# (see _iter_orphan_runs, #170) whatever shape its output would have had.
# job.json is what an export bundle writes, listed defensively.
RUN_BOOKKEEPING_FILES = frozenset({MANIFEST_FILE_NAME, REALIZED_FILE_NAME, "job.json"})


def _write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)


def default_ui_dir():
    """Where the built SPA lives: ui/dist in a checkout (the copy npm just
    built), else the copy packaged into the wheel at dw/server/ui, else None."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(here)), "ui", "dist"),
        os.path.join(here, "ui"),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "index.html")):
            return candidate
    return None


def create_app(
    workflow_dir="./workflows",
    output_dir="./outputs",
    log_level="INFO",
    job_manager=None,
    ui_dir=None,
    download_manager=None,
    diffusers_updater=None,
    prompt_dir="./prompts",
    asset_dir=None,
    examples_dirs=None,
    workspace=None,
    host="127.0.0.1",
    token=None,
    mcp=False,
    port=8765,
):
    """Build the application. A caller (tests) can inject a JobManager.

    `host` is the address the server is bound to (informational here - it
    is added to the Host-header allowlist alongside the loopback names, so
    a deployment bound to one specific non-loopback address still accepts
    its own requests). `token`, if given, is a static bearer token required
    on every /api/* request - see require_bearer_token below.
    """
    manager = job_manager or JobManager(
        output_dir, log_level=log_level, workflow_dir=workflow_dir
    )
    # An injected manager must confine jobs to the same workflow_dir the
    # routes do, or /api/validate and /api/jobs would enforce different
    # boundaries
    if manager.workflow_dir is None:
        manager.workflow_dir = workflow_dir
    elif manager.workflow_dir != workflow_dir:
        raise ValueError(
            "job_manager.workflow_dir must match the app's workflow_dir: "
            f"{manager.workflow_dir!r} != {workflow_dir!r}"
        )

    mcp_asgi = mcp_server = mcp_client = None
    if mcp:
        from .mcp_mount import build_mcp_app

        mcp_asgi, mcp_server, mcp_client = build_mcp_app(
            host=host, port=port, token=token
        )

    @asynccontextmanager
    async def lifespan(app):
        if mcp_server is None:
            yield
        else:
            # the SDK's session manager is the mounted app's own lifespan,
            # which Starlette does not run for a sub-app
            try:
                async with mcp_server.session_manager.run():
                    yield
            finally:
                # the client owns a connection pool; a session manager that
                # fails to start must not leak it
                mcp_client.close()
        manager.shutdown()

    app = FastAPI(
        title="diffusers-workflow",
        description="Declarative diffusers workflows over HTTP: queue a job, "
        "stream its progress, fetch what it saved.",
        lifespan=lifespan,
    )
    app.state.job_manager = manager
    # This box's own job history as a cost, recomputed when the jobs table
    # moves rather than when a file does - a job landing changes every
    # figure and changes no workflow file (#93)
    app.state.observed_costs = ObservedCosts(getattr(manager, "history", None))
    app.state.workflow_dir = workflow_dir
    app.state.prompt_dir = prompt_dir
    # The read-only libraries the --examples-dir trees bring with them: an
    # example workflow references the prompts and assets that live beside
    # its tree, not the ones in this workspace. They are searched after the
    # workspace's own and never written to - a save of an example prompt
    # lands in the workspace, the way saving an example workflow does
    _example_libraries = example_libraries(examples_dirs)
    app.state.example_prompt_dirs = _example_libraries[PROMPTS_SUBDIR]
    app.state.example_asset_dirs = _example_libraries[ASSETS_SUBDIR]
    # Where uploads land and 'asset:' references resolve. None when the
    # caller configured no asset library: uploads then fall back to the
    # output directory's uploads/ subfolder, as they did before there was one
    app.state.asset_dir = os.path.abspath(asset_dir) if asset_dir else None
    # The workspace the three directories above default to folders of, for a
    # client that wants to name the root rather than reason about the parts.
    # None when the caller resolved no workspace (a test building an app
    # around three explicit directories)
    app.state.workspace = os.path.abspath(workspace) if workspace else None
    # The root that holds named workspaces. Its own folders are the default
    # workspace - which is what the three directories above already point at,
    # so a server given individual directory overrides simply has one
    # workspace and no others
    app.state.workspace_root = (
        Workspace(app.state.workspace, "flag") if app.state.workspace else None
    )
    # The default workspace itself, as a Workspace: its four folders are the
    # configured directories above, not '<root>/workflows' and friends - a
    # caller can override any one of them individually (--workflow-dir,
    # etc), so they cannot be derived from a root the way a named
    # workspace's folders are
    app.state.default_workspace = ConfiguredWorkspace(
        workflows=app.state.workflow_dir,
        assets=app.state.asset_dir,
        outputs=manager.output_dir,
        prompts=app.state.prompt_dir,
        root=app.state.workspace,
    )
    app.state.mcp_mounted = mcp_asgi is not None
    # A StaticFiles instance per output/asset root, built lazily and reused -
    # a mount is bound to one directory at startup, but a named workspace's
    # root does not exist yet then. Keeping the instance around (rather than
    # building one per request) is what makes /outputs and /inputs answer
    # ETag/If-None-Match with 304 and Range with 206 the way a real mount
    # does, instead of the plain FileResponse this replaced always resending
    # the whole file
    app.state.static_files_by_root = {}

    wildcard_bind = host in WILDCARD_HOSTS
    allowed_hosts = set(LOOPBACK_HOSTS)
    if host and not wildcard_bind:
        allowed_hosts.add(host.lower())
    # What the middlewares read at request time. `downloads` and `updater` are
    # stored at their construction below, `ceiling_indexes` beside its use
    app.state.api_token = token
    app.state.bind_host = host
    app.state.bind_port = port
    app.state.wildcard_bind = wildcard_bind
    app.state.allowed_hosts = allowed_hosts
    app.state.examples_dirs = examples_dirs
    app.state.downloads = download_manager or DownloadManager()
    app.state.updater = diffusers_updater or DiffusersUpdater()
    # One index per distinct listing, per app (see deps.ceiling_index)
    app.state.ceiling_indexes = {}
    install_middleware(app)
    include_routers(app)

    # --------------------------------------------------------------- gallery

    # Built from the security layer's allowlists so a new format is added
    # exactly once - the gallery had already drifted (.bmp, .mkv, .mov)
    from ..security import (
        ALLOWED_AUDIO_EXTENSIONS,
    )

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
    # stored instead (see _zip_download)
    RAW_MEDIA_EXTENSIONS = {".bmp", ".wav"}

    # Exposed for tests - the two sets _zip_download's compression policy
    # reads, so a test can assert the relationship without reaching into a
    # closure
    app.state.media_kinds = MEDIA_KINDS
    app.state.raw_media_extensions = RAW_MEDIA_EXTENSIONS

    # Longest side of an on-demand gallery thumbnail, in pixels
    GALLERY_THUMBNAIL_MAX_DIM = 320

    def _strip_output_prefix(name):
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

    def _output_file(name, root=None):
        """A file inside a workspace's output directory, or a 404 - never
        outside it."""
        root = root or manager.output_dir
        name = _strip_output_prefix(name)
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

    def _asset_in(name, roots):
        """The file a bare asset name has in one of these roots, or a 404.

        `_asset_file` with the search path already in hand, for a caller
        resolving many names against the one workspace: each call to
        `resolve_asset_reference` walks the pinned fallbacks on its own, so
        calling it once per root re-walked them all every time - the name is
        validated once here instead, and each root is then just a join and
        an isfile check.
        """
        try:
            validate_asset_reference(name)
        except SecurityError as e:
            raise HTTPException(status_code=404, detail=str(e))
        for root in roots:
            candidate = os.path.join(root, name)
            if not os.path.isfile(candidate):
                continue
            try:
                return validate_path(candidate, root)
            except SecurityError:
                # A symlink under this root can still point outside it -
                # isfile follows the link and says yes, and validate_path
                # is what actually catches the escape. That's a miss for
                # this root, not a 500: fall through to the next one and,
                # on a total miss, the same 404 every other miss gets.
                continue
        if not roots:
            detail = f"Unknown asset {name!r}: this workspace has no asset library"
        else:
            detail = f"Unknown asset {name!r}: not found in {', '.join(roots)}"
        raise HTTPException(status_code=404, detail=detail)

    def _asset_file(reference, ws):
        """The file an 'asset:' reference names in this workspace, or a 404.

        Looked for down the same search path a run resolves 'asset:' in
        (asset_roots), so what the API can read is what a job would load.
        A miss names every root that was searched, so the caller sees
        their own workspace library among them rather than just the last
        (often an examples directory they never wrote to).
        """
        return _asset_in(
            reference.removeprefix(ASSET_PREFIX).strip(),
            resolution_roots(app.state, ws),
        )

    def _job_provenance(name, ws):
        """The job that wrote an output file, and which run and version -
        `gallery_metadata`'s output branch and `keep_output_as_asset` (#556)
        both need this, the latter so it can carry it into the kept asset's
        sidecar before the run directory it came from is pruned."""
        try:
            # Scoped to this workspace: two workspaces can each write a
            # file with the same relative name, and an unscoped lookup
            # could attribute this one to the wrong workspace's job
            job = manager.history.job_for_file(name, workspace=ws.name)
        except Exception:
            job = None
        run_id, version = "", None
        folder, run_id, _subfolder = split_run_path(name)
        if run_id:
            try:
                identity_dir = validate_path(
                    os.path.join(ws.outputs, folder), ws.outputs
                )
            except SecurityError:
                identity_dir = None
            if identity_dir:
                version = run_versions(identity_dir).get(run_id)
        return job, run_id, version

    def _static_files_for(root):
        """The StaticFiles instance bound to one root, built on first use and
        cached on app.state - see the comment where the cache is created."""
        cache = app.state.static_files_by_root
        files = cache.get(root)
        if files is None:
            files = StaticFiles(directory=root)
            cache[root] = files
        return files

    def _iter_gallery_files(root, group_runs=True):
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

    def _gallery_entries(root, ws):
        entries = []
        try:
            files = list(_iter_gallery_files(root))
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
                # dw/result.py's output_file_path
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

    @app.get("/api/gallery")
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
            return {
                "runs": page,
                "total": len(entries),
                "offset": offset,
                "limit": limit,
                "workspace": ws.name,
            }
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

    @app.get("/api/gallery/{name:path}/metadata")
    def gallery_metadata(
        name: str,
        envelope: bool = False,
        ws: Workspace = Depends(selected_workspace),
    ):
        """Generation metadata embedded in a saved image ('workflow' inside
        it is the full definition the editor can reopen), plus the job that
        produced the file when history remembers one, plus - for audio and
        video - what the file itself holds: duration, format and level,
        which is how an agent that cannot listen checks a track. Only an
        image embeds 'metadata' this way - it is always null for audio and
        video, since neither format has a slot this writer uses; recover
        the recipe from 'job' (GET /api/jobs/{id}/workflow) when one is
        known, or from nothing when it isn't (a kept asset has no job).

        `envelope=true` adds the soundtrack's level second by second, which
        is what says *where* in a track something is - whether a shot is
        still voiced at its last frame, how deep the hole at a seam goes.
        Opt-in: a ten-minute track is 600 numbers, and the default call has
        to stay small.

        `name` may also be an 'asset:' reference, and then it is the input
        asset of that name that is described rather than an output (#127).
        The numbers here - duration, frame count, fps, sample rate - are
        what decide whether a call will work at all, and for a file the
        caller is about to *consume* they were previously unobtainable:
        the only way to read a wav's length was to run a job that copied it
        into the output directory. `job` is null for an asset with no
        recorded provenance, and `source` says which of the two roots
        answered.

        A kept asset (`keep_output`) is not always provenance-blind: when
        the file it was kept from still had a job in history, `keep_output`
        recorded `{job, run_id, version}` beside it, and this route reads it
        back the same way it reads a kept file's shots (#556). `run_id`/
        `version` are that source run's own - `job.id` is still the one to
        pass `get_job_workflow`, since a pruned run leaves no `workflow.json`
        of its own to reopen."""
        run_id, version = "", None
        name = _strip_output_prefix(name)
        if is_asset_reference(name):
            path = _asset_file(name, ws)
            source, job = "asset", None
            kept = kept_provenance(path)
            if kept:
                job = kept.get("job")
                run_id = kept.get("run_id") or ""
                version = kept.get("version")
        else:
            path = _output_file(name, ws.outputs)
            source = "output"
            # Which run wrote it, and that run's ordinal - the same 'v4' the
            # listing reports. After "look at version 3" this is the next
            # call, so it confirms the right file was reached rather than
            # sending the caller back to the listing
            job, run_id, version = _job_provenance(name, ws)
        metadata = read_embedded_metadata(path)
        extension = os.path.splitext(path)[1].lower()
        media = (
            probe_media(path, envelope=envelope)
            if MEDIA_KINDS.get(extension) in ("audio", "video")
            else None
        )
        if media is not None and source == "output":
            # Where each shot of a joined video sits, as the run that wrote
            # it recorded (dw/shots.py) - null for a file not joined from shots
            media["shots"] = recorded_shots(ws.outputs, name)
        elif media is not None and source == "asset":
            # keep_output carries the source run's shots into a sidecar
            # manifest beside the asset (#393); a file kept before that fix,
            # or never joined from shots, has none
            media["shots"] = shots_beside(path)
        return {
            "name": name,
            "source": source,
            "metadata": metadata,
            "job": job,
            "run_id": run_id,
            "version": version,
            "media": media,
        }

    @app.get("/api/gallery/{name:path}/assess")
    def gallery_assess(
        name: str,
        probe: Optional[str] = None,
        detail: bool = False,
        ws: Workspace = Depends(selected_workspace),
    ):
        """Measure a finished cut and say where to look (#388): every
        assessment probe that applies to the file, run here in the server
        process on one decode - a sync route, so it runs beside a GPU job
        rather than queueing behind it. Findings are places to look, not
        verdicts; nothing acts on one (dw/assessment_rules.py).

        The default answer merges the probes' `findings`, `rules_applied`
        and `rules_skipped`, and names each probe the file cannot feed in
        `not_applicable` (a still, no soundtrack, no recorded shots);
        `detail=true` adds each probe's full answer under `probes`.
        `probe` names one - analyze_shots, analyze_seams or
        analyze_sync_drift - and answers with its full body. It is checked
        before the name is resolved. `name` may be an `asset:` reference,
        and then the shots are the ones keep_output carried beside it."""
        rejected = unknown_probe(probe)
        if rejected:
            raise HTTPException(status_code=400, detail=rejected)
        name = _strip_output_prefix(name)
        if is_asset_reference(name):
            path = _asset_file(name, ws)
            source, shots = "asset", shots_beside(path)
        else:
            path = _output_file(name, ws.outputs)
            source, shots = "output", recorded_shots(ws.outputs, name)
        kind = MEDIA_KINDS.get(os.path.splitext(path)[1].lower())
        try:
            body = assess(path, kind, shots, probe=probe, detail=detail)
        except (ValueError, OSError) as e:
            raise HTTPException(
                status_code=422, detail=f"{name} could not be read: {e}"
            )
        return {"name": name, "source": source, "kind": kind, **body}

    @app.get("/api/gallery/{name:path}/audio")
    def gallery_audio(
        name: str,
        start: Optional[float] = None,
        duration: Optional[float] = None,
        ws: Workspace = Depends(selected_workspace),
    ):
        """The soundtrack of an output or asset, as WAV - a muxed video's
        track, which `get_output_audio` used to refuse outright, or an
        excerpt (`start` + `duration`, seconds) of a track too long to send
        whole (#193). An excerpt names itself in the response headers
        (`X-DW-Excerpt-Start`, `X-DW-Excerpt-Duration`) beside the whole
        track's `X-DW-Duration` - omitted only when a container carries no
        duration in its own header - so a cut is never silent (#204).

        An audio-only file asked for whole is served as its own bytes in its
        own encoding - there is nothing to extract, and a transcode would
        change what the agent hears."""
        name = _strip_output_prefix(name)
        if is_asset_reference(name):
            path = _asset_file(name, ws)
        else:
            path = _output_file(name, ws.outputs)
        extension = os.path.splitext(path)[1].lower()
        kind = MEDIA_KINDS.get(extension)
        if kind not in ("audio", "video"):
            raise HTTPException(status_code=404, detail=f"{name} carries no soundtrack")

        excerpt = start is not None or duration is not None
        if kind == "audio" and not excerpt:
            # The container's own header has the duration - reading it does
            # not decode a single frame, unlike probe_media (which measures
            # level and would pay for a full decode just for one number).
            headers = {}
            duration_seconds = media_duration(path)
            if duration_seconds is not None:
                headers["X-DW-Duration"] = str(duration_seconds)
            if extension == ".wav":
                # mimetypes says audio/x-wav on macOS, audio/vnd.wave from
                # Python 3.14's builtin table on a box with no system mime
                # file; an extract says audio/wav, and a whole WAV must not
                # read as a different kind
                media_type = "audio/wav"
            else:
                media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            return FileResponse(path, media_type=media_type, headers=headers)

        # A track over the cap is refused at the header, not after it has
        # been decoded and shipped: the MCP side would refuse the same bytes
        # for the same reason, having paid for all of them. An excerpt is
        # sized by its own span - `duration`, clipped to what is left of the
        # track after `start` - so a whole-length "excerpt" is not a way
        # around the gate.
        shape = audio_shape(path)
        if shape is not None and shape["duration_seconds"] is not None:
            span = shape["duration_seconds"]
            if excerpt and duration is not None:
                span = max(0.0, min(float(duration), span - float(start or 0.0)))
            projected = projected_wav_base64_size({**shape, "duration_seconds": span})
            if projected > MAX_INLINE_AUDIO_BYTES:
                what = (
                    f"a {span:.1f}s excerpt of {name}"
                    if excerpt
                    else f"{name}'s whole soundtrack"
                )
                advice = (
                    "Ask for a shorter `duration`"
                    if excerpt
                    else "Ask for an excerpt with `start` and `duration` (seconds)"
                )
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{what} would be {projected} bytes base64-encoded as WAV "
                        f"- over the {MAX_INLINE_AUDIO_BYTES} byte limit for an "
                        f"inline clip. {advice}, or download the file."
                    ),
                )

        try:
            data, info = extract_audio(path, start=start, duration=duration)
        except NoSoundtrack:
            raise HTTPException(status_code=404, detail=f"{name} carries no soundtrack")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        headers = {"X-DW-Duration": str(info["of_seconds"])}
        if info["excerpt"]:
            headers["X-DW-Excerpt-Start"] = str(info["start"])
            headers["X-DW-Excerpt-Duration"] = str(info["duration_seconds"])
        return Response(content=data, media_type="audio/wav", headers=headers)

    FRAME_MIN_DIMENSION = 64
    # The most moments one `at` may name: each is a seek, a decode and a
    # PNG encode in the server process, and a contact sheet is the shape
    # for seeing more of a clip at once
    MAX_FRAME_MOMENTS = 32

    @app.get("/api/gallery/{name:path}/frames")
    def gallery_frames(
        name: str,
        at: Optional[str] = None,
        count: Optional[int] = None,
        seams: Optional[str] = None,
        boundaries: Optional[str] = None,
        names: Optional[str] = None,
        max_dimension: int = 512,
        crop: Optional[str] = None,
        ws: Workspace = Depends(selected_workspace),
    ):
        """Frames of a video output or asset, as PNG tiles - the way an
        agent with no video content type sees what a run made (#193).
        Exactly one selector: `at` (a comma list of seconds or "frame:N"),
        `count` (an evenly spaced contact sheet, `frame_grid` without a
        workflow), or `seams` ("true", or a comma list of 1-based seam
        numbers) for the last frame before and first frame after each
        boundary, side by side. `boundaries` is the comma list of frame
        indexes each shot after the first starts at, and `names` the
        shots' names. Without `boundaries`, an output's seams are the shots
        its run's manifest recorded for it (a `concat_videos`,
        `dissolve_videos` or chained step), named as recorded unless `names`
        is given; a linked asset (`keep_output(shared=true)`) uses the same
        shots `get_gallery_metadata`'s `media.shots` reports for it, from the
        sidecar manifest kept beside it. A file with none recorded still
        needs `boundaries`.
        Tiles are downscaled to `max_dimension` on their longest side.
        `crop` is `x,y,width,height` in the video's own source pixels
        (`video_shape`'s `width`/`height`) - resolved once and cut from
        every sampled frame before any stamping, fitting or composing, so
        it names the same region whatever `max_dimension` downscales the
        result to."""
        name = _strip_output_prefix(name)
        if is_asset_reference(name):
            path = _asset_file(name, ws)
        else:
            path = _output_file(name, ws.outputs)
        if MEDIA_KINDS.get(os.path.splitext(path)[1].lower()) != "video":
            raise HTTPException(status_code=404, detail=f"{name} is not a video")

        chosen = [
            key
            for key, value in (("at", at), ("count", count), ("seams", seams))
            if value
        ]
        if len(chosen) != 1:
            raise HTTPException(
                status_code=400,
                detail="Pass exactly one of `at`, `count` or `seams`"
                + (f" - got {', '.join(chosen)}" if chosen else ""),
            )
        # A floor on each *sub-tile* of a composite (contact sheet / seam
        # pair) - a caller asking for a small max_dimension still gets a
        # legible grid, which is then fit to max_dimension as a whole below.
        sub_tile_width = max(FRAME_MIN_DIMENSION, int(max_dimension))
        limit = max(1, int(max_dimension))

        try:
            # Computed once and threaded through every selector below: each
            # of frames_at/contact_sheet/seam_tiles would otherwise call
            # video_shape itself, opening the container (and, lacking a
            # header frame count, decoding it whole to count) a second time
            # just to answer the same frame_count/fps/width/height (#193).
            shape = video_shape(path)
            crop_box = (
                resolve_crop_box(
                    [c.strip() for c in crop.split(",")],
                    shape["width"],
                    shape["height"],
                )
                if crop
                else None
            )
            if at:
                moments = [
                    m.strip() if m.strip().startswith("frame:") else float(m)
                    for m in at.split(",")
                    if m.strip()
                ]
                if len(moments) > MAX_FRAME_MOMENTS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"`at` names {len(moments)} moments; the most is "
                        f"{MAX_FRAME_MOMENTS} - ask for a contact sheet (`count`) "
                        "to see more of the clip at once",
                    )
                tiles = frames_at(path, moments, shape=shape, crop_box=crop_box)
            elif count:
                tiles = [
                    contact_sheet(
                        path,
                        count,
                        tile_width=sub_tile_width,
                        shape=shape,
                        crop_box=crop_box,
                    )
                ]
            else:
                recorded = (
                    None
                    if boundaries
                    else shots_beside(path)
                    if is_asset_reference(name)
                    else recorded_shots(ws.outputs, name)
                )
                if recorded:
                    # The file's own seams, from its run's manifest (or, for
                    # a linked asset, the sidecar `record_kept_shots` wrote
                    # beside it)
                    starts = [shot["start_frame"] for shot in recorded[1:]]
                    shot_names = (
                        [n.strip() for n in names.split(",")]
                        if names
                        else [shot["name"] for shot in recorded]
                    )
                elif not boundaries:
                    raise HTTPException(
                        status_code=400,
                        detail="`seams` needs `boundaries`: the frame index each "
                        "shot after the first starts at - this file's run "
                        "recorded no shots for it",
                    )
                else:
                    starts = [int(b) for b in boundaries.split(",") if b.strip()]
                    shot_names = (
                        [n.strip() for n in names.split(",")] if names else None
                    )
                wanted = (
                    None
                    if seams.lower() == "true"
                    else {int(s) for s in seams.split(",") if s.strip()}
                )
                if wanted is not None and not wanted:
                    raise HTTPException(
                        status_code=400,
                        detail="`seams` names no seam - pass `true` for every seam, "
                        "or seam numbers from 1",
                    )
                tiles = seam_tiles(
                    path,
                    starts,
                    names=shot_names,
                    tile_width=sub_tile_width,
                    shape=shape,
                    wanted=wanted,
                    crop_box=crop_box,
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {
            "name": name,
            **shape,
            "tiles": [_encoded_tile(tile, limit) for tile in tiles],
            "crop": (
                [
                    crop_box[0],
                    crop_box[1],
                    crop_box[2] - crop_box[0],
                    crop_box[3] - crop_box[1],
                ]
                if crop_box
                else None
            ),
        }

    def _encoded_tile(tile, limit):
        image = tile["image"]
        longest = max(image.width, image.height)
        if longest > limit:
            scale = limit / longest
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                )
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = {key: value for key, value in tile.items() if key != "image"}
        encoded.update(
            {
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "mime_type": "image/png",
                "width": image.width,
                "height": image.height,
            }
        )
        return encoded

    @app.get("/api/gallery/{name:path}/thumbnail")
    @query_token_ok
    def gallery_thumbnail(
        name: str, request: Request, ws: Workspace = Depends(selected_workspace)
    ):
        """A small JPEG rendition of an image output, for the grid - the
        full-resolution file is only fetched for the detail/lightbox view.
        Generated on demand rather than cached to disk, so it never grows
        the output directory the gallery itself scans."""
        path = _output_file(name, ws.outputs)
        extension = os.path.splitext(path)[1].lower()
        if MEDIA_KINDS.get(extension) != "image":
            raise HTTPException(
                status_code=404, detail="Thumbnails are only generated for images"
            )
        # The file's mtime and size are the validator: the grid re-requests
        # every visible thumbnail on each visit, and a 304 skips the
        # decode/resize/encode; a rerun that overwrites the file changes it
        stat = os.stat(path)
        etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        cache_headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=cache_headers)
        try:
            from PIL import Image

            with Image.open(path) as image:
                if image.width * image.height > MAX_DECODE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{name} is {image.width}x{image.height}, more "
                        f"than the {MAX_DECODE_PIXELS:,} pixels a thumbnail "
                        "is decoded from",
                    )
                # shrink first (JPEGs decode at reduced size via draft), then
                # convert - converting a full-resolution image only to
                # discard most of it is the expensive order
                image.draft(
                    "RGB", (GALLERY_THUMBNAIL_MAX_DIM, GALLERY_THUMBNAIL_MAX_DIM)
                )
                image.thumbnail((GALLERY_THUMBNAIL_MAX_DIM, GALLERY_THUMBNAIL_MAX_DIM))
                image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=80)
        except Image.DecompressionBombError as e:
            # Pillow's own refusal, on open, of a header past twice its limit
            raise HTTPException(status_code=413, detail=str(e))
        except (OSError, ValueError) as e:
            # what PIL raises for an unreadable or corrupt file
            raise HTTPException(
                status_code=500, detail=f"Could not generate thumbnail: {e}"
            )
        return Response(
            content=buffer.getvalue(), media_type="image/jpeg", headers=cache_headers
        )

    @app.get("/api/gallery/{name:path}/download")
    @query_token_ok
    def download_output(name: str, ws: Workspace = Depends(selected_workspace)):
        """Serve one output file as a forced download rather than an inline view."""
        path = _output_file(name, ws.outputs)
        return FileResponse(path, filename=os.path.basename(name))

    # A generous ceiling rather than a real limit - it exists so a
    # malformed client cannot ask the server to zip the whole directory
    MAX_ARCHIVE_FILES = 1000

    class ArchiveRequest(BaseModel):
        names: list[str] = Field(min_length=1, max_length=MAX_ARCHIVE_FILES)

    def _zip_download(entries, filename):
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

    def _archive_selection(entries, kind):
        """`_zip_download` plus the one tail the two archive routes shared:
        a timestamped `dw-<kind>s-*.zip` name and a log line naming the
        count. Logged after the archive is written, not before, so a write
        that fails partway (a bad path slipping past resolution, a full
        disk) doesn't log a success that didn't happen.
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        response = _zip_download(entries, f"dw-{kind}s-{stamp}.zip")
        logger.info(f"Archived {len(entries)} {kind} files")
        return response

    @app.post("/api/gallery/archive")
    def archive_outputs(
        request: ArchiveRequest, ws: Workspace = Depends(selected_workspace)
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
        names = [_strip_output_prefix(name) for name in request.names]
        paths = [(name, _output_file(name, ws.outputs)) for name in names]

        return _archive_selection(paths, "output")

    # What a run directory holds besides its media: the engine writes them to
    # describe the run, and the gallery - which lists media - never shows them
    RUN_SIDECARS = (MANIFEST_FILE_NAME, REALIZED_FILE_NAME)

    def _remove_empty_identity_folders(run_dir, root):
        """Remove the folders above a deleted run, up to `root`, while empty.

        Under the run lock open_run takes: it creates the identity folder
        and then claims a run inside it, and removing the folder between
        the two would fail that run on a path that no longer exists.
        """
        identity_dir = os.path.dirname(run_dir)
        with FileLock(run_lock_path(identity_dir)):
            parent = identity_dir
            while os.path.normpath(parent) != os.path.normpath(root):
                try:
                    os.rmdir(parent)
                except OSError:
                    break
                parent = os.path.dirname(parent)

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
            run_dir = validate_path(
                os.path.join(root, relative), root, allow_create=False
            )
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
        _remove_empty_identity_folders(run_dir, root)
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

    @app.delete("/api/gallery/{name:path}")
    def delete_output(name: str, ws: Workspace = Depends(selected_workspace)):
        """Remove one file from the output directory.

        When that was the last media file of its run, the run directory goes
        with it, sidecars included. `name` may also be a run directory
        (`<identity>/<run id>`), which removes the whole run - the only handle
        on a run that failed before it wrote any media (#134).
        """
        name = _strip_output_prefix(name)
        run_dir = _run_directory(name, ws.outputs)
        if run_dir is not None:
            # As in _prune_empty_run_directory: pin the siblings' numbers
            # before one of them goes
            record_run_versions(os.path.dirname(run_dir))
            shutil.rmtree(run_dir, ignore_errors=True)
            _remove_empty_identity_folders(run_dir, ws.outputs)
            logger.info(f"Deleted run directory {name}")
            forget_workspace_usage()
            return {
                "name": name,
                "deleted": True,
                "run_swept": os.path.basename(run_dir),
            }

        path = _output_file(name, ws.outputs)
        os.remove(path)
        logger.info(f"Deleted output file {name}")
        swept = _prune_empty_run_directory(name, ws.outputs)
        forget_workspace_usage()
        return {"name": name, "deleted": True, "run_swept": swept}

    # ---------------------------------------------------------------- uploads

    UPLOADS_SUBDIR = "uploads"
    # Audio included: the asset library holds it and workflows read it (an
    # H3 audio reference is built from a .wav), so refusing it here would
    # leave one input kind with no way onto the machine
    ALLOWED_UPLOAD_EXTENSIONS = (
        ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
    )
    MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB - covers a short video clip

    @app.post("/api/uploads", status_code=201)
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
        keeps the old behavior, writing to the output directory's uploads/
        and returning an absolute path. The body is the raw file bytes: no
        multipart parser dependency needed for a single-file upload.

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
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty upload")
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload too large: {len(body)} > {MAX_UPLOAD_BYTES}",
            )

        library = ws.assets or ws.outputs
        if shared:
            library = common_assets(ws)
            if not library:
                raise HTTPException(
                    status_code=409,
                    detail="This server has no shared asset library - it was "
                    "configured from loose directories rather than a workspace "
                    "root, so there is nothing for an asset to be common to",
                )
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
        if shared or ws.assets:
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
        path = f"/outputs/{UPLOADS_SUBDIR}/{quote(name)}"
        result = {
            "workspace": ws.name,
            "url": served_url(path, ws),
        }
        absolute_url = absolute_served_url(path, ws)
        if absolute_url is not None:
            result["absolute_url"] = absolute_url
        return result

    def _asset_origin(ws, root):
        """Which library an asset came from: this workspace's own, the one
        shared by every workspace under the root, or a read-only examples
        tree. A client that cannot tell them apart cannot say why deleting
        one answers 403.

        By directory, never by position in the search path: the workspace's
        own library drops out of `asset_roots` until it exists, and the
        examples tree that then sits first is still nobody's to write."""
        own = ws.assets
        if own and os.path.abspath(own) == root:
            return WORKSPACE_ORIGIN
        common = common_assets(ws)
        if common and os.path.abspath(common) == root:
            return COMMON_ORIGIN
        return EXAMPLES_ORIGIN

    @app.get("/api/assets")
    def list_assets(ws: Workspace = Depends(selected_workspace)):
        """The asset library: the input media an 'asset:' reference names.

        Reported by reference rather than by path - 'asset:uploads/x.png' is
        what a workflow argument carries, and a client that only ever sees
        references cannot accidentally write a path that means something
        else on another machine. Empty, not an error, on a server with no
        library configured: nothing is wrong, there is just nowhere for an
        asset to be.
        """
        library = ws.assets
        roots = asset_roots(app.state, ws)
        if not roots:
            return {
                "asset_dir": library,
                "asset_dirs": [],
                "assets": [],
                "folders": [],
                "libraries": [],
                "shadowed": [],
            }

        libraries = [
            {
                "origin": (origin := _asset_origin(ws, root)),
                "dir": root,
                "writable": origin != EXAMPLES_ORIGIN,
            }
            for root in roots
        ]

        assets = []
        shadowed = []
        # Which origin first claimed a name, so a later root's same name can
        # be reported as shadowed rather than silently dropped
        seen = {}
        for root in roots:
            try:
                files = list(_iter_gallery_files(root, group_runs=False))
            except OSError:
                files = []
            origin = _asset_origin(ws, root)
            for relative, folder, _subfolder, _run_id, kind, path in files:
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                # A name in the workspace shadows the same name in an
                # examples library, exactly as 'asset:' resolution does
                if relative in seen:
                    shadowed.append(
                        {
                            "name": relative,
                            "reference": make_ref(ASSET, relative),
                            "folder": folder,
                            "kind": kind,
                            "size": stat.st_size,
                            "mtime": stat.st_mtime,
                            "origin": origin,
                            "shadowed_by": seen[relative],
                        }
                    )
                    continue
                seen[relative] = origin
                asset_path = f"/inputs/{quote(relative)}"
                asset_entry = {
                    "name": relative,
                    "reference": make_ref(ASSET, relative),
                    "folder": folder,
                    "kind": kind,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "origin": origin,
                    # For the editor's own preview - fetchable the same
                    # way an upload's URL is
                    "url": served_url(asset_path, ws),
                }
                absolute_url = absolute_served_url(asset_path, ws)
                if absolute_url is not None:
                    asset_entry["absolute_url"] = absolute_url
                assets.append(asset_entry)
        assets.sort(key=lambda entry: entry["mtime"], reverse=True)
        return {
            # The workspace's own library, unchanged: where an upload lands
            "asset_dir": library,
            "asset_dirs": [lib["dir"] for lib in libraries],
            "assets": assets,
            "folders": sorted({entry["folder"] for entry in assets} | {""}),
            "libraries": libraries,
            "shadowed": shadowed,
        }

    class KeepRequest(BaseModel):
        name: str = Field(
            description="The generated file to keep, as the gallery names it"
        )
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

    @app.post("/api/assets/keep", status_code=201)
    def keep_output_as_asset(
        request: KeepRequest, ws: Workspace = Depends(selected_workspace)
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
        library = common_assets(ws) if request.shared else ws.assets
        if not library:
            raise HTTPException(
                status_code=409,
                detail=(
                    "This server has no shared asset library"
                    if request.shared
                    else "This workspace has no asset library"
                ),
            )

        kept_name = _strip_output_prefix(request.name)
        source = _output_file(kept_name, ws.outputs)
        asset_name = request.asset_name or os.path.basename(kept_name)
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
                detail=f"asset_name {request.asset_name!r} does not match the "
                f"kept file's kind ({extension or 'no extension'})",
            )
        try:
            asset_name = validate_asset_reference(asset_name)
            destination = validate_path(os.path.join(library, asset_name), library)
        except SecurityError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if os.path.exists(destination) and not request.overwrite:
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
        job, source_run_id, source_version = _job_provenance(kept_name, ws)
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

        logger.info(f"Kept output {request.name} as asset:{asset_name}")
        return {
            "reference": make_ref(ASSET, asset_name),
            "name": asset_name,
            "workspace": ws.name,
            "linked": linked,
            "shared": bool(request.shared),
        }

    @app.post("/api/assets/archive")
    def archive_assets(
        request: ArchiveRequest, ws: Workspace = Depends(selected_workspace)
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
        roots = resolution_roots(app.state, ws)
        # Stripped and deduped before resolving, so "iris.png" and
        # "iris.png " (or a name repeated by an eager client) become the one
        # zip entry rather than a collision on write
        names = list(dict.fromkeys(n.strip() for n in request.names))
        # Resolved before anything is written, so a bad name in the
        # selection fails the request instead of yielding a partial zip
        paths = [(name, _asset_in(name, roots)) for name in names]

        return _archive_selection(paths, "asset")

    @app.delete("/api/assets/{name:path}")
    def delete_asset(name: str, ws: Workspace = Depends(selected_workspace)):
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
        roots = asset_roots(app.state, ws)
        if not roots:
            raise HTTPException(
                status_code=409, detail="This server has no asset library"
            )
        try:
            relative = validate_asset_reference(name)
        except SecurityError as e:
            raise HTTPException(status_code=400, detail=str(e))

        for root in roots:
            try:
                path = validate_path(os.path.join(root, relative), root)
            except SecurityError:
                continue
            if not os.path.isfile(path):
                continue
            origin = _asset_origin(ws, root)
            if origin == EXAMPLES_ORIGIN:
                raise HTTPException(
                    status_code=403,
                    detail=f"{make_ref(ASSET, relative)} is read-only: it comes "
                    f"from an examples library, not a library this server writes",
                )
            os.remove(path)
            logger.info(f"Deleted asset:{relative} ({path})")
            forget_workspace_usage()
            return {
                "name": relative,
                "workspace": ws.name,
                "reference": make_ref(ASSET, relative),
                "deleted": True,
                "origin": origin,
            }

        raise HTTPException(status_code=404, detail=f"No such asset: {relative}")

    # ---------------------------------------------------------------- outputs

    # ------------------------------------------------------------------ mcp

    if mcp_asgi is not None:
        # One route rather than app.mount("/mcp", ...): Starlette's Mount
        # only matches paths *under* its prefix, so a bare POST /mcp - the
        # URL clients are configured with - would fall through to the SPA
        # catch-all below and come back 405. This matches /mcp and
        # anything under it; build_mcp_app's wrapper normalizes the path
        # for the SDK app's single route.
        # Exactly the two spellings require_bearer_token gates - a single
        # "/mcp{path:path}" route would also answer /mcpfoo, which the gate
        # does not cover.
        app.router.routes.append(Route("/mcp", endpoint=mcp_asgi, name="mcp"))
        app.router.routes.append(
            Route("/mcp/{sub_path:path}", endpoint=mcp_asgi, name="mcp_sub")
        )

    # Generated files and input media, served as routes rather than static
    # mounts: a mount is bound to one directory at startup, and a workspace
    # can be created afterwards. Each handler delegates to a StaticFiles
    # instance for the workspace's own root (_static_files_for) rather than
    # a bare FileResponse - a FileResponse never answers 304 (no
    # If-None-Match handling), so every gallery load re-streamed the whole
    # file; going through StaticFiles.get_response restores ETag/
    # If-None-Match 304s, Range/206 and its own 404 handling, the way a real
    # mount always has.
    #
    # Ungated, as the mounts were, and for the same reason: an <img> or
    # <video> tag cannot attach an Authorization header. The auth middleware
    # only gates /api/, so these stay reachable exactly as before.
    #
    # '/inputs', not '/assets': Vite emits the SPA's own bundles under
    # /assets/, and serving the library there shadows them - the page loads
    # and then renders nothing, because its script and stylesheet 404. The
    # name is also the symmetric one, next to /outputs
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
        media_type = (
            response.headers.get("content-type", "").split(";")[0].strip().lower()
        )
        if media_type in ACTIVE_DOCUMENT_TYPES:
            response.headers["Content-Security-Policy"] = "sandbox"
        return response

    @app.get("/outputs/{name:path}")
    async def output_file(
        name: str, request: Request, ws: Workspace = Depends(selected_workspace)
    ):
        """One generated file, from the workspace that made it - or, by an
        'asset:' reference, one file from its asset library (#445): every
        other route in this family (`get_gallery_metadata`, `/frames`,
        `/audio`, `/assess`) already accepts one, and this route answering a
        bare StaticFiles 404 for the same name gave no hint why."""
        name = _strip_output_prefix(name)
        if is_asset_reference(name):
            # This route is outside the token gate (the auth middleware
            # covers /api/ and /mcp only), so a miss must not carry
            # _asset_file's detail, which names every root searched by its
            # absolute server path. Keep the hint #445 added, without them.
            try:
                path = _asset_file(name, ws)
            except HTTPException as e:
                if e.status_code != 404:
                    raise
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown asset {name!r}: not in this workspace's "
                    "asset library (list_assets shows what is)",
                ) from None
            files = _static_files_for(os.path.dirname(path))
            response = await files.get_response(os.path.basename(path), request.scope)
        else:
            files = _static_files_for(ws.outputs)
            response = await files.get_response(name, request.scope)
        return _sandbox_active_content(response)

    @app.get("/inputs/{name:path}")
    async def input_file(
        name: str, request: Request, ws: Workspace = Depends(selected_workspace)
    ):
        """One file from the asset search path, for the editor's preview of
        an uploaded or chosen asset - the workspace's own library first,
        then any read-only examples library, so an example workflow's media
        previews the way an upload does."""
        roots = asset_roots(app.state, ws)
        if not roots:
            raise HTTPException(status_code=404, detail="no asset library")
        for root in roots:
            try:
                candidate = validate_path(os.path.join(root, name), root)
            except SecurityError:
                continue
            if os.path.isfile(candidate):
                files = _static_files_for(root)
                return _sandbox_active_content(
                    await files.get_response(name, request.scope)
                )
        # Nothing has it: let the workspace's own library answer, so the
        # 404 (and its headers) come from StaticFiles as they always did
        files = _static_files_for(roots[0])
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
    @app.get("/exports/{job_id}.zip")
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
        return _zip_download(entries, _export_download_name(directory, job_id))

    # ---------------------------------------------------------------- the UI

    resolved_ui = ui_dir or default_ui_dir()
    if resolved_ui:
        # Mounted last so /api and /outputs keep precedence; html=True serves
        # index.html at /, and the SPA routes by hash so no fallback is needed
        app.mount("/", StaticFiles(directory=resolved_ui, html=True), name="ui")

    return app
