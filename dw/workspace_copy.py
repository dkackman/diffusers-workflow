"""Copying runs from one workspace to another under the same root (#853).

A run is self-contained on disk - `latest`, `vN`, `output:` resolution and
the gallery all read the run folder and its manifests, never the jobs
database - so copying the folder into the destination's outputs is the whole
of making it resolvable there. What this module adds is the care around that:

- Confinement. Both ends are workspaces under one root, named and validated
  (`named_workspace`); every path is checked with `validate_path` against
  its own side's outputs folder; a source that is, passes through or holds a
  symlink is refused, and nothing here ever creates one - output references
  refuse linked run folders (tests/test_security_symlinks.py).
- All or nothing. Every entry is checked before anything is written. A run
  folder already in the destination is the same run (same run id), so it is
  `already_present` and skipped, which makes a repeated copy idempotent; a
  run whose job is still running is a conflict and refuses the request.
- Files. Media is hard-linked, falling back to `copy2` across devices, as
  `keep_output` does - dw never edits media in place. JSON sidecars
  (manifest, workflow.json, notes.json, job.json, any `.json`) are always
  copied: they *are* edited in place (`record_kept_shots`,
  `write_realized_workflow`, the notes), and a link would carry an edit in
  one workspace into the other.
- Versions. A copied run keeps its `vN` unless the destination's identity
  already uses that number; then every run this request copies into that
  identity is renumbered after the destination's highest, in run-id order,
  under the identity's run lock so it cannot race `open_run`. The version is
  the only manifest field a copy edits; pinned `output:` refs name run ids,
  so renumbering breaks none of them.
- Warnings, never refusals: `output:` inputs that do not resolve in the
  destination, manifest paths that still point into the source workspace
  (reported relative to its root - no reply carries an absolute server
  path), and that a copied run has no job row (it appears in the gallery,
  not in the jobs list).

Copying assets is stage B of #853.
"""

import logging
import os
import re
import shutil

from filelock import FileLock

from .runs import (
    MANIFEST_FILE_NAME,
    RUN_VERSION_KEY,
    _read_manifest,
    _run_ids,
    is_run_id,
    resolve_output_reference,
    run_id_sort_key,
    run_lock_path,
    run_versions,
    write_json_atomic,
)
from .security import InvalidInputError, PathTraversalError, validate_path
from .workspace import ConfiguredWorkspace, forget_workspace_usage

logger = logging.getLogger("dw")

# One segment of a run entry: the segment rule `output:` names follow
# (security.OUTPUT_REFERENCE_PATTERN), so whatever can be copied can be named
_SEGMENT = re.compile(r"^[\w][\w.@-]*\Z")
_MAX_SEGMENTS = 7

# The sidecars a copy writes fresh rather than links: every JSON file
SIDECAR_SUFFIX = ".json"

NO_JOBS_NOTE = (
    "Copied runs appear in list_gallery and resolve through output: "
    "references, but not in list_jobs - a copy carries no job record."
)


class CopyRefused(Exception):
    """The request cannot be carried out as asked; nothing was written.

    `status` is the HTTP status the route answers with: 404 for something
    that is not there, 409 for a conflict with what is."""

    def __init__(self, message, status=409, conflicts=None):
        super().__init__(message)
        self.status = status
        self.conflicts = conflicts or []


def _run_entry(entry):
    """A `runs` entry as its segments: `<identity>/<run id>` or a bare
    `<identity>`. An `output:` prefix is accepted, as the gallery quotes it.

    Raises:
        PathTraversalError: For an absolute path or a `..`
        InvalidInputError: For anything else that is not a run name
    """
    if not isinstance(entry, str) or not entry.strip():
        raise InvalidInputError("A runs entry must be a non-empty string")
    name = entry.strip()
    if name.startswith("output:"):
        name = name[len("output:") :]
    if name.startswith(("/", "\\")) or os.path.isabs(name):
        raise PathTraversalError(f"'{entry}' is an absolute path - name a run")
    parts = name.strip("/").split("/")
    if any(part in ("..", ".") for part in parts):
        raise PathTraversalError(f"'{entry}' leaves the outputs folder")
    if len(parts) > _MAX_SEGMENTS or not all(_SEGMENT.match(p) for p in parts):
        raise InvalidInputError(
            f"'{entry}' is not a run - name one as '<workflow>/<run id>', as "
            f"the gallery lists it, or a bare '<workflow>' for all its runs"
        )
    return parts


def _symlink_on_the_way(base, path):
    """The first symlink between `base` (exclusive) and `path` (inclusive),
    or None. validate_path resolves links, so a link that stays inside the
    base passes it; this refuses those too."""
    relative = os.path.relpath(path, base)
    current = base
    for part in relative.split(os.sep):
        current = os.path.join(current, part)
        if os.path.islink(current):
            return current
    return None


def _symlink_inside(directory):
    """The first symlink anywhere under a run folder, or None."""
    for current, dirs, files in os.walk(directory):
        for name in dirs + files:
            path = os.path.join(current, name)
            if os.path.islink(path):
                return path
    return None


def _source_runs(source, entry):
    """[(identity, run id)] that one entry names in the source.

    Raises:
        PathTraversalError: When the entry escapes, or reaches a symlink
        CopyRefused (404): When there is no such run or workflow
    """
    parts = _run_entry(entry)
    outputs = os.path.realpath(source.outputs)
    if is_run_id(parts[-1]) and len(parts) > 1:
        identity, run_ids = "/".join(parts[:-1]), [parts[-1]]
    else:
        identity, run_ids = "/".join(parts), None
    identity_dir = validate_path(os.path.join(outputs, *identity.split("/")), outputs)
    if run_ids is None:
        run_ids = _run_ids(identity_dir) if os.path.isdir(identity_dir) else []
        if not run_ids:
            raise CopyRefused(
                f"No runs of '{identity}' in workspace '{source.name}'", status=404
            )
    found = []
    for run_id in run_ids:
        run_dir = os.path.join(outputs, *identity.split("/"), run_id)
        link = _symlink_on_the_way(outputs, run_dir)
        if link:
            raise PathTraversalError(
                f"'{entry}' passes through a symlink - a copy never follows one"
            )
        run_dir = validate_path(run_dir, outputs)
        if not os.path.isdir(run_dir):
            raise CopyRefused(
                f"No run '{identity}/{run_id}' in workspace '{source.name}'",
                status=404,
            )
        if _symlink_inside(run_dir):
            raise PathTraversalError(
                f"Run '{identity}/{run_id}' holds a symlink - a copy never "
                f"follows or creates one"
            )
        found.append((identity, run_id))
    return found


def _copy_file(source, target, totals):
    """Hard-link a media file, falling back to a copy; copy a sidecar."""
    size = os.path.getsize(source)
    if not source.endswith(SIDECAR_SUFFIX):
        try:
            os.link(source, target)
            totals["bytes_linked"] += size
            return
        except OSError:
            pass  # another device, or a filesystem without hard links
    shutil.copy2(source, target)
    totals["bytes_copied"] += size


def _copy_tree(source_dir, target_dir, totals):
    """Recreate a run folder file by file - no symlinks, ever."""
    os.makedirs(target_dir)
    for current, dirs, files in os.walk(source_dir):
        relative = os.path.relpath(current, source_dir)
        here = (
            target_dir if relative == os.curdir else os.path.join(target_dir, relative)
        )
        for name in dirs:
            os.makedirs(os.path.join(here, name), exist_ok=True)
        for name in files:
            _copy_file(os.path.join(current, name), os.path.join(here, name), totals)


def _place_run(source_dir, identity_dir, run_id, totals):
    """Copy one run folder in under a name no listing reads, then rename it
    into place - a reader never sees half a run."""
    staging = os.path.join(identity_dir, f".copy-{run_id}")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    try:
        _copy_tree(source_dir, staging, totals)
        os.rename(staging, os.path.join(identity_dir, run_id))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _set_version(run_dir, version):
    """Record a copied run's ordinal in its own (copied, never linked)
    manifest. A run with no readable manifest is left alone."""
    manifest = _read_manifest(run_dir)
    if manifest is None or manifest.get(RUN_VERSION_KEY) == version:
        return
    manifest[RUN_VERSION_KEY] = version
    write_json_atomic(os.path.join(run_dir, MANIFEST_FILE_NAME), manifest)


def _copy_identity(source, destination, identity, run_ids, totals):
    """Copy one identity's runs under its run lock and settle their
    versions. Returns {"<identity>/<run>": {"from", "to"}} for each run whose
    number changed."""
    source_identity = os.path.join(
        os.path.realpath(source.outputs), *identity.split("/")
    )
    outputs = os.path.realpath(destination.outputs)
    identity_dir = validate_path(os.path.join(outputs, *identity.split("/")), outputs)
    source_versions = run_versions(source_identity)
    run_ids = sorted(run_ids, key=run_id_sort_key)
    renumbered = {}
    with FileLock(run_lock_path(identity_dir)):
        os.makedirs(identity_dir, exist_ok=True)
        held = run_versions(identity_dir)
        wanted = [source_versions.get(run_id) for run_id in run_ids]
        collides = any(v is not None and v in held.values() for v in wanted)
        next_number = max(held.values(), default=0) + 1
        for run_id, version in zip(run_ids, wanted):
            _place_run(
                os.path.join(source_identity, run_id), identity_dir, run_id, totals
            )
            if version is None:
                continue
            new = version
            if collides:
                new, next_number = next_number, next_number + 1
            # Pinned even when unchanged: a source run that recorded no
            # number was ranked, and its rank in the destination is not
            # the one it had
            _set_version(os.path.join(identity_dir, run_id), new)
            if new != version:
                renumbered[f"{identity}/{run_id}"] = {"from": version, "to": new}
    return renumbered


def _absolute_paths(manifest, source_root):
    """Paths in a manifest that still point into the source workspace,
    relative to its root: a file a step cache hit republished from an
    earlier run, and the workflow file the run was started from."""
    found = []

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and os.path.isabs(value):
            try:
                inside = (
                    os.path.commonpath([os.path.realpath(value), source_root])
                    == source_root
                )
            except ValueError:
                inside = False
            if inside:
                relative = os.path.relpath(os.path.realpath(value), source_root)
                found.append(relative.replace(os.sep, "/"))

    visit({key: manifest.get(key) for key in ("workflow", "steps")})
    return sorted(set(found))


def _unresolved_inputs(manifest, outputs):
    """The `output:` inputs a copied run's manifest records that name
    nothing in the destination."""
    unresolved = []
    for entry in manifest.get("inputs") or []:
        if not isinstance(entry, dict) or entry.get("kind") != "output":
            continue
        reference = entry.get("reference")
        if not isinstance(reference, str):
            continue
        try:
            resolve_output_reference(reference, outputs)
        except (ValueError, OSError, InvalidInputError, PathTraversalError):
            unresolved.append(reference)
    return unresolved


def copy_runs(source, destination, runs, active_runs=()):
    """Copy runs from one workspace into another.

    Args:
        source, destination: Workspaces under one root, already resolved
            and known to exist
        runs: `<identity>/<run id>` or bare `<identity>` entries
        active_runs: `<identity>/<run id>` of the source's runs whose job
            is still queued or running - their manifest is not final

    Returns:
        {copied: {runs, assets}, already_present, renumbered, warnings,
        bytes_linked, bytes_copied}

    Raises:
        InvalidInputError, PathTraversalError: A malformed or escaping entry
        CopyRefused: Anything else that stops the copy; nothing was written
    """
    for side in (source, destination):
        if isinstance(side, ConfiguredWorkspace):
            raise CopyRefused(
                "This server has no workspace root - it was started with "
                "individual directory overrides, so there is no other "
                "workspace to copy to or from"
            )
    if os.path.realpath(source.root) == os.path.realpath(destination.root):
        raise InvalidInputError(
            f"Source and destination are both '{source.name}' - name another "
            f"workspace to copy into"
        )
    if not runs:
        raise InvalidInputError("Name at least one run to copy")
    os.makedirs(destination.outputs, exist_ok=True)
    if not os.access(destination.outputs, os.W_OK):
        raise CopyRefused(f"Workspace '{destination.name}' is not writable")

    named = []
    for entry in runs:
        for found in _source_runs(source, entry):
            if found not in named:
                named.append(found)

    outputs = os.path.realpath(destination.outputs)
    active = set(active_runs)
    conflicts = []
    already_present = []
    to_copy = {}
    for identity, run_id in named:
        name = f"{identity}/{run_id}"
        if name in active:
            conflicts.append(name)
            continue
        target = validate_path(
            os.path.join(outputs, *identity.split("/"), run_id), outputs
        )
        if os.path.lexists(target):
            already_present.append(name)
        else:
            to_copy.setdefault(identity, []).append(run_id)
    if conflicts:
        raise CopyRefused(
            "These runs' jobs are still queued or running, so their manifests "
            "are not final - wait for them, then copy: " + ", ".join(conflicts),
            conflicts=conflicts,
        )

    totals = {"bytes_linked": 0, "bytes_copied": 0}
    renumbered = {}
    copied = []
    for identity, run_ids in to_copy.items():
        renumbered.update(
            _copy_identity(source, destination, identity, run_ids, totals)
        )
        copied.extend(f"{identity}/{run_id}" for run_id in run_ids)
    forget_workspace_usage()
    logger.info(
        f"Copied {len(copied)} run(s) from workspace {source.name} to "
        f"{destination.name}"
    )

    warnings = {}
    source_root = os.path.realpath(source.root)
    unresolved = {}
    absolute = {}
    for name in copied:
        run_dir = os.path.join(outputs, *name.split("/"))
        manifest = _read_manifest(run_dir) or {}
        refs = _unresolved_inputs(manifest, outputs)
        if refs:
            unresolved[name] = refs
        paths = _absolute_paths(manifest, source_root)
        if paths:
            absolute[name] = paths
    if unresolved:
        warnings["unresolved_output_refs"] = unresolved
    if absolute:
        warnings["absolute_paths"] = absolute
    if copied:
        warnings["no_jobs"] = NO_JOBS_NOTE

    return {
        "source": source.name,
        "destination": destination.name,
        "copied": {"runs": copied, "assets": []},
        "already_present": already_present,
        "renumbered": renumbered,
        "warnings": warnings,
        **totals,
    }
