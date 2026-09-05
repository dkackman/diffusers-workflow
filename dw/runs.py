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

import hashlib
import json
import logging
import os
import re
from datetime import datetime

logger = logging.getLogger("dw")

RUN_LAYOUT = "run"
FLAT_LAYOUT = "flat"
LAYOUTS = (RUN_LAYOUT, FLAT_LAYOUT)

# Set by an entry point, and inherited by a spawned worker the way
# DW_PROMPT_DIR and DW_ASSET_DIR are
OUTPUT_LAYOUT_ENV_VAR = "DW_OUTPUT_LAYOUT"

MANIFEST_FILE_NAME = "manifest.json"

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
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    try:
        material = json.dumps(spec, sort_keys=True, default=str)
    except (TypeError, ValueError):
        material = repr(spec)
    digest = hashlib.sha256(material.encode("utf-8", "replace")).hexdigest()[:8]
    return f"{stamp}-{digest}"


def is_run_id(segment):
    """Whether a path segment is a run id this module generated."""
    return bool(RUN_ID_PATTERN.match(segment or ""))


def strip_run_id(relative_path):
    """The workflow identity a run-relative path belongs to.

    'ltx2/Gyre/20260905-181530-a1b2c3d4/still-0.png' -> 'ltx2/Gyre'. A path
    with no run id in it comes back with its own directory unchanged, which
    is what a flat-layout output does.
    """
    parts = [part for part in (relative_path or "").split("/") if part]
    directory = parts[:-1]
    if directory and is_run_id(directory[-1]):
        directory = directory[:-1]
    return "/".join(directory)


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
