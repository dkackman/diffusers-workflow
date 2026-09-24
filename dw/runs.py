"""A run: the directory one execution of a workflow writes into, and the
manifest it leaves behind.

Output used to be laid out by where the workflow file sits - the subfolder
mirrored its position under the nearest directory literally named 'workflows',
so the *shape of a checkout* was the grouping key, and a workflow moved out of
that tree silently flattened. A run directory replaces that with the
workflow's own identity plus one directory per execution:

    <output_dir>/<identity>/<run id>/
        <the files the run wrote>
        manifest.json

Everything one execution produced - intermediates, finals, and the record of
what made them - lands in one place, prunable and addressable as a unit, and
a rerun can no longer interleave its files with an earlier one's.

The old flat-ish layout stays available: DW_OUTPUT_LAYOUT=flat, an
'output_layout' setting of "flat", or --output-layout flat on dw.run and
dw.serve, for a caller whose scripts glob the output directory.
"""

import contextvars
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone

logger = logging.getLogger("dw")

RUN_LAYOUT = "run"
FLAT_LAYOUT = "flat"
LAYOUTS = (RUN_LAYOUT, FLAT_LAYOUT)

# Set by an entry point, and inherited by a spawned worker the way
# DW_PROMPT_DIR and DW_ASSET_DIR are
OUTPUT_LAYOUT_ENV_VAR = "DW_OUTPUT_LAYOUT"

MANIFEST_FILE_NAME = "manifest.json"

# The workflow a run actually ran, written beside its manifest. Named
# 'workflow.json' rather than something run-specific because the directory
# already says which run it is, and 'python -m dw.run workflow.json' from
# inside it is the whole reproduction story
REALIZED_FILE_NAME = "workflow.json"

# The prefix marking a value as a reference to a file an earlier run wrote.
# Like 'asset:', it stands for a path - what a previous run made is an input
# like any other, and multi-stage work is what a workflow engine is for
OUTPUT_PREFIX = "output:"

# The segment that means "the newest run of this workflow that has the
# file", so a workflow can name the stage before it without being edited
# after every run - see _resolve_segments for why it is not simply the
# newest run directory
LATEST = "latest"

# 'v4' in the run-id position of an 'output:' reference: the run whose
# ordinal is 4 - the number the gallery shows and an agent quotes
_VERSION_SELECTOR = re.compile(r"^v([1-9][0-9]*)$")


def version_selector(segment):
    """The ordinal a 'v<N>' segment names, or None for any other segment."""
    match = _VERSION_SELECTOR.match(segment)
    return int(match.group(1)) if match else None


# What a run id looks like: a UTC timestamp and a short digest of the spec.
# The pattern is not only documentation - the gallery reads it to group a
# workflow's runs under one folder rather than listing every run separately
# The trailing counter appears only when two runs of the same spec start in
# the same second - see run_directory
RUN_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}(-\d+)?$")

# Characters allowed in a path segment derived from a workflow's name or file
_UNSAFE_SEGMENT_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]+")

# The synthetic file name workflow_from_definition gives an inline workflow -
# it carries a directory, not an identity
INLINE_FILE_NAME = "__inline__"


# The output root of the run in progress, so an 'output:' reference resolves
# against the directory this run was told to write to rather than guessing
# one. Set by Workflow.run; a reference realized outside any run falls back
# to the workspace's outputs
_active_output_root = contextvars.ContextVar("dw_output_root", default=None)


def activate_output_root(root):
    """Make an output root the active one; returns a token for deactivate."""
    return _active_output_root.set(root)


def deactivate_output_root(token):
    _active_output_root.reset(token)


def output_root():
    """The output directory 'output:' references resolve against."""
    active = _active_output_root.get()
    if active:
        return active

    from .workspace import resolve_workspace

    return resolve_workspace().outputs


def is_output_reference(value):
    """Whether a value references a file an earlier run wrote."""
    return isinstance(value, str) and value.startswith(OUTPUT_PREFIX)


def _runs_newest_first(directory):
    """The run directories inside a workflow's output folder, newest first.

    Run ids start with a UTC timestamp, so sort order is age order - no stat
    calls, and no dependence on mtimes that a copy would have rewritten
    anyway. Empty when the directory holds no runs, or is not there.
    """
    try:
        return sorted(
            (
                name
                for name in os.listdir(directory)
                if is_run_id(name) and os.path.isdir(os.path.join(directory, name))
            ),
            key=run_id_sort_key,
            reverse=True,
        )
    except OSError:
        return []


def _resolve_segments(directory, parts, reference, root):
    """Build the path a name stands for, expanding 'latest' or 'v<N>' where
    it names a run.

    'latest' means the newest run *that has the file*, not the newest run
    directory: a run that failed part way, or one whose every step was a
    cache hit, leaves a directory holding only its manifest, and a
    second-stage workflow pointed at that would find nothing where the
    stage before it plainly produced something. So the runs are tried
    newest first and the first one holding the rest of the name wins.

    'v<N>' means the run whose recorded ordinal is N - the 'v4' the gallery
    shows - so the number quoted to a person is also a name a workflow can
    take. Unlike 'latest' it picks exactly one run: a v4 that did not write
    the file is an error, not a reason to try v3.

    Only a segment standing where run directories are is a run selector. A
    'latest' or 'v4' segment in a directory that holds no runs is a name
    like any other, so a workflow or a file called either stays reachable.

    Returns the path, or None when runs were found and none of them holds
    the file.
    """
    if not parts:
        return directory
    part, rest = parts[0], parts[1:]
    if part == LATEST:
        runs = _runs_newest_first(directory)
        if runs:
            for run in runs:
                candidate = _resolve_segments(
                    os.path.join(directory, run), rest, reference, root
                )
                if candidate and os.path.isfile(candidate):
                    return candidate
            return None
        if not os.path.exists(os.path.join(directory, part)):
            raise ValueError(
                f"No runs yet under {os.path.relpath(directory, root)} - "
                f"'{reference}' names the newest run of a workflow that "
                f"has not produced one"
            )
    wanted = version_selector(part)
    if wanted is not None and _runs_newest_first(directory):
        versions = run_versions(directory)
        matching = [run for run, version in versions.items() if version == wanted]
        if not matching:
            held = ", ".join(f"v{v}" for v in sorted(set(versions.values())))
            raise ValueError(
                f"No run v{wanted} under {os.path.relpath(directory, root)} - "
                f"'{reference}' names a run by its version, and the runs "
                f"there are {held}"
            )
        # Normally one; two only where history predating versions could
        # not be ranked beneath the first recorded number. Newest first,
        # as 'latest' would try them
        for run in sorted(matching, key=run_id_sort_key, reverse=True):
            candidate = _resolve_segments(
                os.path.join(directory, run), rest, reference, root
            )
            if candidate and os.path.isfile(candidate):
                return candidate
        return None
    return _resolve_segments(os.path.join(directory, part), rest, reference, root)


def resolve_output_reference(reference, root=None):
    """Resolve an 'output:' reference to the file it names.

    The name is a path under the output directory - '<workflow>/<run
    id>/<file>' - and the run id may be written as 'latest', which resolves
    to the newest run of that workflow that holds the file, or as 'v<N>',
    the run whose version is N. 'latest' is what
    lets a second-stage workflow name the first stage's product without
    being edited after every run, and without breaking when the newest run
    failed or reused cached files and so wrote none of its own.

    Args:
        reference: The 'output:...' string
        root: The output directory to resolve against; defaults to the run
            in progress, else the workspace's outputs

    Returns:
        The validated absolute path of the file

    Raises:
        InvalidInputError: If the name is not a valid output name
        PathTraversalError: If the name escapes the output directory
        ValueError: If no such run or file exists
    """
    from .security import validate_output_reference, validate_path

    name = validate_output_reference(reference.removeprefix(OUTPUT_PREFIX).strip())
    root = root or output_root()

    resolved = _resolve_segments(root, name.split("/"), reference, root)
    if resolved is None:
        raise ValueError(
            f"Output '{name}' not found under {root} - no run of that workflow "
            f"holds the file. A run that failed, or reused every step from the "
            f"cache, leaves only its manifest behind"
        )
    # Containment is checked once, on the whole path, after 'latest' has
    # been expanded - so what is validated is the real directory it named.
    # The segments themselves were pattern-checked, so this guards symlinks
    resolved = validate_path(resolved, root)

    if not os.path.isfile(resolved):
        raise ValueError(
            f"Output '{name}' not found under {root} - an 'output:' reference "
            f"names a file an earlier run wrote, like "
            f"'output:ltx2/Gyre/latest/Gyre-still.0-0.0.png'"
        )
    logger.debug(f"Resolved {reference} to {resolved}")
    return resolved


def fetch_output(reference, root=None):
    """The path an 'output:' reference names, for whatever loads paths."""
    return resolve_output_reference(reference, root)


def output_layout():
    """Whether runs get their own directory ('run') or write into the output
    directory the way they did before ('flat').

    Read at call time so a worker subprocess and a test see the current
    value, the same as every other directory question.
    """
    from_environment = os.environ.get(OUTPUT_LAYOUT_ENV_VAR)
    if from_environment in LAYOUTS:
        return from_environment

    from .settings import load_settings

    from_settings = load_settings().output_layout
    return from_settings if from_settings in LAYOUTS else RUN_LAYOUT


def set_output_layout(layout):
    """Pin the layout for this process and anything it spawns."""
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown output layout: {layout}")
    os.environ[OUTPUT_LAYOUT_ENV_VAR] = layout
    return layout


def _safe_segment(text):
    cleaned = _UNSAFE_SEGMENT_CHARACTERS.sub("_", str(text)).strip("._")
    return cleaned or "workflow"


def workflow_identity(file_spec, workflow_id=None):
    """What names this workflow's outputs, as a relative path.

    A workflow's position under a 'workflows' tree still reads as its
    identity when it has one - 'workflows/ltx2/Gyre.json' is 'ltx2/Gyre' -
    because that is the organization a user already chose. Outside such a
    tree the file's own name is the identity, and an inline definition,
    which has no file, is named by its workflow id.

    The result is always a relative path of safe segments: it is joined onto
    the output directory, and nothing about it is allowed to leave.
    """
    name = None
    subfolder = ""
    if file_spec:
        base = os.path.basename(file_spec)
        stem = os.path.splitext(base)[0]
        if stem and stem != INLINE_FILE_NAME:
            name = stem
        directory = os.path.dirname(os.path.abspath(file_spec))
        parts = os.path.normpath(directory).split(os.sep)
        try:
            # The last 'workflows' segment wins, matching the packaged
            # dw/workflows tree when a checkout has a top-level one too
            index = len(parts) - 1 - parts[::-1].index("workflows")
        except ValueError:
            index = None
        if index is not None and index + 1 < len(parts):
            subfolder = os.path.join(*(_safe_segment(p) for p in parts[index + 1 :]))

    name = _safe_segment(name or workflow_id or "workflow")
    return os.path.join(subfolder, name) if subfolder else name


def new_run_id(spec=None, now=None):
    """An identifier for one execution: a UTC timestamp, then eight hex
    digits of the spec that produced it.

    The timestamp is what sorts and what a person reads; the digest is what
    tells two runs of the same second apart and makes a rerun of an edited
    workflow visibly different from a rerun of the same one. A server job
    could have used its job id, but a CLI run has none, and one scheme
    everywhere is what lets anything reading the directory tree - the
    gallery, a future history rebuild - understand both.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    try:
        material = json.dumps(spec, sort_keys=True, default=str)
    except (TypeError, ValueError):
        material = repr(spec)
    digest = hashlib.sha256(material.encode("utf-8", "replace")).hexdigest()[:8]
    return f"{stamp}-{digest}"


def is_run_id(segment):
    """Whether a path segment is a run id this module generated."""
    return bool(RUN_ID_PATTERN.match(segment or ""))


def split_run_path(relative_path):
    """The three parts of a run-relative path: (identity, run id, subfolder).

    'ltx2/Gyre/20260905-181530-a1b2c3d4/final/still.png' ->
    ('ltx2/Gyre', '20260905-181530-a1b2c3d4', 'final'). The run id is
    found wherever it sits, not only as the last directory - a step's
    'subfolder' puts segments after it. A path with no run id in it (the
    flat layout) has its whole directory as identity and nothing else,
    which is what it was before subfolders existed.

    Only the first segment matching RUN_ID_PATTERN counts. A workflow
    *file* named in that shape would produce a matching identity segment;
    that is unsupported rather than impossible.
    """
    parts = [part for part in (relative_path or "").split("/") if part]
    directory = parts[:-1]
    for position, segment in enumerate(directory):
        if is_run_id(segment):
            return (
                "/".join(directory[:position]),
                segment,
                "/".join(directory[position + 1 :]),
            )
    return "/".join(directory), "", ""


def strip_run_id(relative_path):
    """The workflow identity a run-relative path belongs to.

    'ltx2/Gyre/20260905-181530-a1b2c3d4/still-0.png' -> 'ltx2/Gyre', and
    the same with a subfolder after the run id. A path with no run id in it
    comes back with its own directory unchanged, which is what a
    flat-layout output does.
    """
    return split_run_path(relative_path)[0]


# The key a run's ordinal is recorded under in its manifest. It is assigned
# once, when the run directory is opened, and never recomputed - which is the
# whole point: a number quoted in conversation has to still mean the same run
# after a sibling is deleted. Deleting a middle run leaves a gap, and so does
# a run that wrote no media (it failed, or every step was reused from the
# cache): it took a number and has nothing in the gallery to show under it
RUN_VERSION_KEY = "version"

# The length of a run id before any '-N' counter a same-second rerun takes
_RUN_ID_BASE_LENGTH = len("20260101-000000-00000000")

# Recorded ordinals by manifest path, keyed on the manifest's stat so an
# edited or replaced manifest is read again. A recorded number never changes,
# so this is what keeps a gallery listing from parsing every manifest under
# the output root on every call
_recorded_versions = {}


def run_id_sort_key(run_id):
    """Order run ids oldest first, with a rerun's '-N' counter compared as a
    number - lexically '-10' would sort before '-2'."""
    base, counter = run_id[:_RUN_ID_BASE_LENGTH], run_id[_RUN_ID_BASE_LENGTH + 1 :]
    return (base, int(counter) if counter.isdigit() else 1)


def _read_manifest(run_dir):
    """A run's manifest as a dict, or None when it is missing or unreadable."""
    try:
        with open(os.path.join(run_dir, MANIFEST_FILE_NAME)) as file:
            manifest = json.load(file)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _valid_version(version):
    # bool is an int subclass, and True is not version 1
    if isinstance(version, bool) or not isinstance(version, int):
        return None
    return version if version > 0 else None


def _recorded_version(run_dir):
    """The ordinal a run recorded for itself, or None.

    None covers every way the number can be missing: a run made before this
    field existed, one killed before its manifest landed, and one whose
    manifest cannot be parsed. All three are ranked rather than trusted.
    """
    path = os.path.join(run_dir, MANIFEST_FILE_NAME)
    try:
        stat = os.stat(path)
    except OSError:
        _recorded_versions.pop(path, None)
        return None
    signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    cached = _recorded_versions.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    manifest = _read_manifest(run_dir)
    version = _valid_version(manifest.get(RUN_VERSION_KEY)) if manifest else None
    _recorded_versions[path] = (signature, version)
    return version


def _run_ids(identity_dir):
    """Every run directory under one workflow identity, oldest first.

    Run ids sort by their UTC timestamp, so this order is chronological
    to the second - the same property `latest` relies on. Within one second
    the spec digest decides, which is arbitrary but stable; nothing here
    needs finer ordering than that.
    """
    try:
        entries = os.listdir(identity_dir)
    except OSError:
        return []
    return sorted(
        (
            name
            for name in entries
            if is_run_id(name) and os.path.isdir(os.path.join(identity_dir, name))
        ),
        key=run_id_sort_key,
    )


def _ranked_versions(identity_dir):
    """({run id: version}, {run id: recorded version or None})."""
    run_ids = _run_ids(identity_dir)
    recorded = {
        run_id: _recorded_version(os.path.join(identity_dir, run_id))
        for run_id in run_ids
    }
    # Room beneath the lowest recorded number for the unrecorded runs that
    # come before it. Where there is not enough room the sequence starts at
    # 1 and the recorded numbers stand: a duplicate is better than
    # renumbering a run someone has already been told the number of
    leading = 0
    for run_id in run_ids:
        if recorded[run_id] is not None:
            break
        leading += 1
    next_number = 1
    if leading < len(run_ids):
        next_number = max(1, recorded[run_ids[leading]] - leading)
    versions = {}
    for run_id in run_ids:
        if recorded[run_id] is not None:
            versions[run_id] = recorded[run_id]
            # Never backwards: two runs of one second can sort in the
            # opposite order to their numbers, and an unrecorded run after
            # them must not take a number the higher one already holds
            next_number = max(next_number, recorded[run_id] + 1)
        else:
            versions[run_id] = next_number
            next_number += 1
    return versions, recorded


def run_versions(identity_dir):
    """Every run of one workflow mapped to its ordinal: {run id: version}.

    A run that recorded a version keeps it verbatim - that is what makes the
    number survive a sibling being deleted. A run that recorded none (made
    before the field existed, or killed before its manifest landed) is
    ranked into the sequence around it: the unrecorded runs *older* than
    every recorded one take the numbers just beneath the lowest recorded
    one, so history that predates the field lands where it belongs, and an
    unrecorded run anywhere later continues from the highest number before
    it. Ordering is by run id, which is chronological.

    Read only. A ranked number is only as stable as its neighbours until
    `record_run_versions` writes it down.
    """
    return _ranked_versions(identity_dir)[0]


def record_run_versions(identity_dir):
    """Write each ranked number into the manifest of a run that has one but
    records no version, and return every run's ordinal.

    A ranked number moves when an older unrecorded sibling is deleted, so
    runs made before the field existed are pinned the first time anything
    writes under their workflow: a new run opening, or a run directory
    being deleted. The listing never writes. A run with no manifest at all
    is left alone - writing one would invent a record of a run nobody
    recorded - and stays ranked.

    Best effort: a manifest that cannot be rewritten keeps its ranked number.
    """
    versions, recorded = _ranked_versions(identity_dir)
    for run_id, version in versions.items():
        if recorded[run_id] is not None:
            continue
        run_dir = os.path.join(identity_dir, run_id)
        manifest = _read_manifest(run_dir)
        if manifest is None:
            continue
        manifest[RUN_VERSION_KEY] = version
        path = os.path.join(run_dir, MANIFEST_FILE_NAME)
        partial = f"{path}.partial"
        try:
            with open(partial, "w") as file:
                json.dump(manifest, file, indent=2, default=str)
            os.replace(partial, path)
        except OSError as e:
            logger.warning(f"Could not record version {version} in {path}: {e}")
    return versions


def assign_run_version(output_dir, identity):
    """The ordinal the run about to open under `identity` takes.

    One past the highest ordinal any sibling holds - not one past the newest
    run's, because run ids are chronological only across seconds: two runs
    started in the same second are ordered by their spec digest, so the last
    id is not reliably the highest number. Three quick reruns are exactly
    that case.

    Pins the ranked numbers of older runs on the way (`record_run_versions`),
    so history that predates the field stops moving once a new run joins it.
    Sharing that ranking rather than deriving the maximum separately is what
    keeps the number assigned here and the number the gallery reports from
    drifting apart.

    Best effort, like everything else that writes a run's bookkeeping: a
    directory that cannot be read yields 1 rather than failing the run.
    """
    versions = record_run_versions(os.path.join(output_dir, identity))
    return max(versions.values(), default=0) + 1


def run_directory(output_dir, file_spec, workflow_id, run_id):
    """Where one execution writes: <output_dir>/<identity>/<run id>.

    One execution gets one directory, so a run id already taken - two runs
    of the same spec started in the same second, which is what a quick
    rerun is - takes a counter rather than writing into the earlier run's
    directory and burying its manifest.
    """
    base = os.path.join(output_dir, workflow_identity(file_spec, workflow_id), run_id)
    candidate = base
    counter = 1
    while os.path.exists(candidate):
        counter += 1
        candidate = f"{base}-{counter}"
    return candidate


def write_manifest(run_dir, manifest):
    """Record what a run did, beside what it made.

    A server run is already in jobs.sqlite, but a CLI run has never been
    recorded anywhere, and history that lives only in a database cannot
    survive the directory being moved to another machine. Never fatal: a
    run that produced its files has succeeded whether or not this lands.
    """
    path = os.path.join(run_dir, MANIFEST_FILE_NAME)
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(path, "w") as file:
            json.dump(manifest, file, indent=2, default=str)
    except OSError as e:
        logger.warning(f"Could not write {path}: {e}")
        return None
    return path


def write_realized_workflow(run_dir, realized):
    """Leave the workflow that produced a run beside what it produced.

    The submitted definition says what was asked for; this says what ran -
    arguments folded in, the seed pinned, stored prompts inlined,
    'output:latest' resolved. Best effort, exactly like write_manifest: a run
    that produced its files has succeeded whether or not this lands.
    """
    path = os.path.join(run_dir, REALIZED_FILE_NAME)
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(path, "w") as file:
            json.dump(realized, file, indent=2, default=str)
    except (OSError, TypeError, ValueError) as e:
        logger.warning(f"Could not write {path}: {e}")
        return None
    return path


def manifest_relative_files(files, run_dir):
    """A run's file paths as the manifest records them: relative to the run
    directory, so the directory can be moved or copied and still describe
    itself. A file from an earlier run - what a step cache hit republishes -
    is outside this directory and stays absolute.
    """
    recorded = []
    for path in files or []:
        try:
            relative = os.path.relpath(path, run_dir)
        except ValueError:  # different drive on Windows
            recorded.append(path)
            continue
        recorded.append(
            path if relative.startswith(os.pardir) else relative.replace(os.sep, "/")
        )
    return recorded


def recorded_shots(output_root, relative_path):
    """The shot boundaries the run's manifest records for one of its files.

    None when the file is not in a run directory (the flat layout), its run
    has no readable manifest, or no step recorded shots for it - a file that
    was not joined from shots, or one written before shots were recorded.
    """
    from .shots import shots_for_file

    folder, run_id, _subfolder = split_run_path(relative_path)
    if not run_id:
        return None
    run_dir = os.path.join(output_root, folder, run_id)
    manifest = _read_manifest(run_dir)
    if manifest is None:
        return None
    prefix = f"{folder}/{run_id}/" if folder else f"{run_id}/"
    own = relative_path[len(prefix) :] if relative_path.startswith(prefix) else None
    if own is None:
        return None
    for entry in manifest.get("steps") or []:
        if not isinstance(entry, dict) or entry.get("reused"):
            continue
        files = entry.get("files") or []
        if own in files:
            shots = shots_for_file(entry.get("shots"), own, files)
            if shots:
                return shots
    return None


# How far up from a file its run's manifest can sit: the run directory, a
# subfolder (`final/`) and the subfolder's own nesting, which
# SUBFOLDER_PATTERN caps well below this
MANIFEST_SEARCH_DEPTH = 8


def shots_beside(path):
    """The shots a run's manifest records for a file named by its absolute path.

    `recorded_shots` needs the output root and a gallery name; a task that
    was handed a resolved `output:` path has neither, only the file. The run
    directory is the nearest parent holding a manifest, and the file's name
    inside it is what the manifest records. None when no manifest is found
    within MANIFEST_SEARCH_DEPTH parents, or it records no shots for the
    file.
    """
    from .shots import shots_for_file

    path = os.path.abspath(path)
    run_dir = os.path.dirname(path)
    for _ in range(MANIFEST_SEARCH_DEPTH):
        if os.path.isfile(os.path.join(run_dir, MANIFEST_FILE_NAME)):
            break
        parent = os.path.dirname(run_dir)
        if parent == run_dir:
            return None
        run_dir = parent
    else:
        return None
    manifest = _read_manifest(run_dir)
    if manifest is None:
        return None
    own = os.path.relpath(path, run_dir).replace(os.sep, "/")
    for entry in manifest.get("steps") or []:
        if not isinstance(entry, dict) or entry.get("reused"):
            continue
        files = entry.get("files") or []
        if own in files:
            shots = shots_for_file(entry.get("shots"), own, files)
            if shots:
                return shots
    return None
