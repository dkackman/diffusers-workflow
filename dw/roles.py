"""The role of each file a run wrote - final or intermediate - and the
inputs it read.

The gallery mixes a run's deliverable with the scratch files that went
into it. The 'final'/'intermediate' result subfolder convention separates
them for the catalog templates, but the engine treats no folder name
specially and a hand-built workflow rarely sets one. A role is decided per
step, first rule wins:

1. the step's `result.role` ('final' or 'intermediate');
2. its `result.subfolder`, when that is or starts with one of those names;
3. the data flow: a step a later step consumes (a previous_result
   reference) is intermediate when that consumer writes media or saves
   nothing at all, and final otherwise - a consumer that writes only text
   or JSON (a transcript, check_script, plan_cuts) reads a deliverable, it
   does not make one out of it.

A composed child's rolled-up manifest entries carry `parent_step` and take
that step's role. The engine records roles when a run ends, from the
`role_basis` it took as the run opened; the server reads only what was
recorded, so a run that predates roles has none - unknown, not guessed. Inputs
are the asset: and output: references the expanded steps carry - a run's
'made from' list.

Those rules see one run. A project built from separate workflows - shots,
narration, a score, then an assembly that cuts them together - leaves every
run's deliverable recorded final, though for the project they are
ingredients. So a listing applies one more rule when it reads, without
rewriting a manifest (`consumed_outputs`, `listed_role`):

4. a file a later run in the same output root read is intermediate - through
   an output: reference (a run id, 'v<N>' or 'latest', resolved to the file
   that run read: its inputs are pinned as it opened, and an unpinned
   'latest' is the newest run that started before it), or through an asset:
   reference to an asset keep_output kept from that file. A role rule 1
   declared is recorded as such (`role_declared`) and never demoted; an entry
   from before that flag is demotable, and a file with no recorded role
   stays unknown.

This module names no model.
"""

import json
import os

from .references import (
    ASSET,
    OUTPUT,
    is_ref,
    iter_previous_result_references,
    make_ref,
    ref_name,
)
from .runs import (
    MANIFEST_FILE_NAME,
    REALIZED_FILE_NAME,
    _read_manifest,
    iter_run_directories,
    kept_provenance,
    resolve_output_reference,
)
from .security import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    SecurityError,
)
from .step_cache import reference_resolves_to

FINAL = "final"
INTERMEDIATE = "intermediate"
INPUT = "input"
ROLES = (FINAL, INTERMEDIATE)
ROLE_KEY = "role"
# On a manifest entry whose role the step's `result.role` declared (rule 1),
# the one role a later run's reading it does not demote
ROLE_DECLARED_KEY = "role_declared"

_MEDIA_EXTENSIONS = frozenset(
    ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
)


def _subfolder_role(subfolder):
    """Rule 2: the role a subfolder's head names, or None."""
    if isinstance(subfolder, str):
        head = subfolder.split("/", 1)[0]
        if head in ROLES:
            return head
    return None


def _stated_role(step):
    """Rule 1: the role a step's `result.role` names, or None."""
    result = step.get("result")
    role = result.get(ROLE_KEY) if isinstance(result, dict) else None
    return role if role in ROLES else None


def _declared_role(step):
    """Rules 1 and 2: the role a step's result block states, or None."""
    result = step.get("result")
    if not isinstance(result, dict):
        return None
    return _stated_role(step) or _subfolder_role(result.get("subfolder"))


def _writes_only_text(files):
    """Whether a consumer saved files and none of them is media."""
    if not files:
        return False
    return not any(
        str(name).lower().endswith(tuple(_MEDIA_EXTENSIONS)) for name in files
    )


def _files_by_step(manifest_steps):
    files = {}
    for entry in manifest_steps or []:
        if not isinstance(entry, dict):
            continue
        # A child's entry is its composing step's output in the parent
        name = entry.get("parent_step") or entry.get("step")
        files.setdefault(name, []).extend(entry.get("files") or [])
    return files


def role_basis(steps):
    """What the rules read from expanded steps: each step's name, its
    declared role (rules 1 and 2), whether rule 1 declared it, and the
    previous_result references it carries. A run rewrites its steps' references in place as they resolve,
    so the engine takes this as the run opens, before any step runs."""
    basis = []
    for step in steps or []:
        if not isinstance(step, dict) or "name" not in step:
            continue
        consumes = [
            reference
            for _path, reference, _via in iter_previous_result_references(
                step, descend_into_from=True
            )
            if isinstance(reference, str)
        ]
        basis.append(
            {
                "name": step["name"],
                "declared": _declared_role(step),
                "stated": _stated_role(step) is not None,
                "consumes": consumes,
            }
        )
    return basis


def step_roles(steps, manifest_steps, basis=None):
    """Each step's role, by name - see the module docstring for the rules.
    `basis` is `role_basis(steps)` taken earlier, in place of `steps`."""
    basis = role_basis(steps) if basis is None else basis
    files = _files_by_step(manifest_steps)
    names = [b["name"] for b in basis]
    roles = {b["name"]: b["declared"] for b in basis if b["declared"]}
    for consumer in basis:
        if _writes_only_text(files.get(consumer["name"])):
            continue
        for reference in consumer["consumes"]:
            target = max(
                (n for n in names if reference_resolves_to(reference, n)),
                key=len,
                default=None,
            )
            if target is not None and target not in roles:
                roles[target] = INTERMEDIATE
    for name in list(names) + list(files):
        roles.setdefault(name, FINAL)
    return roles


def annotate_roles(steps, manifest_steps, basis=None):
    """Copies of `manifest_steps`, each with its step's `role`, and
    `role_declared` where rule 1 decided it."""
    basis = role_basis(steps) if basis is None else basis
    roles = step_roles(None, manifest_steps, basis)
    stated = {b["name"] for b in basis if b.get("stated")}
    annotated = []
    for entry in manifest_steps or []:
        if not isinstance(entry, dict):
            annotated.append(entry)
            continue
        name = entry.get("parent_step") or entry.get("step")
        entry = {**entry, ROLE_KEY: roles.get(name, FINAL)}
        if name in stated:
            entry[ROLE_DECLARED_KEY] = True
        annotated.append(entry)
    return annotated


def entry_role(entry):
    """A manifest entry's recorded role, or None when it predates roles."""
    role = entry.get(ROLE_KEY) if isinstance(entry, dict) else None
    return role if role in ROLES else None


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def run_inputs(steps):
    """The asset: and output: references the steps read, in first-use order."""
    kinds = ((ASSET, "asset"), (OUTPUT, "output"))
    seen = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        for value in _walk_strings(step):
            for prefix, kind in kinds:
                if is_ref(prefix, value) and value not in (
                    s["reference"] for s in seen
                ):
                    seen.append({"reference": value, "kind": kind})
    return seen


def listed_role(role, declared, consumed):
    """Rule 4: the role a listing shows for a file whose run recorded
    `role` - intermediate once a later run `consumed` it, unless rule 1
    `declared` the role. An unknown role stays unknown."""
    if role is None or declared or not consumed:
        return role
    return INTERMEDIATE


# What each run read, by manifest path, keyed on the manifest's stat: a
# listing asks of every run on every call, and a manifest rarely changes
_READS = {}
_KINDS = {"asset": ASSET, "output": OUTPUT}


def _recorded_reads(run_dir):
    """The (kind, reference) pairs a run read: its manifest's pinned
    `inputs`, or for a run from before them the references its realized
    workflow carries."""
    path = os.path.join(run_dir, MANIFEST_FILE_NAME)
    try:
        stat = os.stat(path)
    except OSError:
        return ()
    signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    cached = _READS.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    inputs = (_read_manifest(run_dir) or {}).get("inputs")
    if not isinstance(inputs, list):
        try:
            with open(os.path.join(run_dir, REALIZED_FILE_NAME)) as handle:
                inputs = run_inputs([json.load(handle)])
        except (OSError, ValueError):
            inputs = []
    reads = tuple(
        (entry["kind"], entry["reference"])
        for entry in inputs
        if isinstance(entry, dict)
        and isinstance(entry.get("reference"), str)
        and is_ref(_KINDS.get(entry.get("kind"), ""), entry["reference"])
    )
    _READS[path] = (signature, reads)
    return reads


def _relative(path, root):
    return os.path.relpath(os.path.realpath(path), root).replace(os.sep, "/")


def consumed_outputs(output_root, find_asset=None, workspace=None):
    """Rule 4's evidence: the files under `output_root` (relative, '/'
    separated) that a run under it read. `find_asset` maps an asset name to
    its file, or None; an asset counts when keep_output recorded it was kept
    from a file of `workspace`. One call answers a whole listing."""
    root = os.path.realpath(output_root)
    consumed = set()
    kept = {}
    for run_dir in iter_run_directories(output_root):
        run_id = os.path.basename(run_dir)
        for kind, reference in _recorded_reads(run_dir):
            try:
                if kind == "output":
                    path = resolve_output_reference(reference, root, before=run_id)
                elif find_asset is None:
                    continue
                else:
                    if reference not in kept:
                        kept[reference] = _kept_source(
                            reference, find_asset, workspace, root
                        )
                    path = kept[reference]
                if path and not _relative(path, root).startswith(".."):
                    consumed.add(_relative(path, root))
            except (OSError, ValueError, SecurityError):
                # a reference that no longer names a file consumes nothing
                continue
    return consumed


def _kept_source(reference, find_asset, workspace, root):
    """The output file an asset was kept from, when it was kept from
    `workspace`, else None."""
    path = find_asset(ref_name(ASSET, reference).strip())
    provenance = kept_provenance(path) if path else None
    if not isinstance(provenance, dict):
        return None
    source = provenance.get("source")
    if provenance.get("workspace") != workspace or not isinstance(source, str):
        return None
    return resolve_output_reference(make_ref(OUTPUT, source), root)
