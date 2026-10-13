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
that step's role. The engine records roles when a run ends; the server
infers them with the same function for a run that predates this. Inputs
are the asset: and output: references the expanded steps carry - a run's
'made from' list. This module names no model.
"""

from .references import ASSET, OUTPUT, is_ref, iter_previous_result_references
from .security import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
)
from .step_cache import reference_resolves_to

FINAL = "final"
INTERMEDIATE = "intermediate"
INPUT = "input"
ROLES = (FINAL, INTERMEDIATE)
ROLE_KEY = "role"

_MEDIA_EXTENSIONS = frozenset(
    ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
)


def _declared_role(step):
    """Rules 1 and 2: the role a step's result block states, or None."""
    result = step.get("result")
    if not isinstance(result, dict):
        return None
    role = result.get(ROLE_KEY)
    if role in ROLES:
        return role
    subfolder = result.get("subfolder")
    if isinstance(subfolder, str):
        head = subfolder.split("/", 1)[0]
        if head in ROLES:
            return head
    return None


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


def step_roles(steps, manifest_steps):
    """Each step's role, by name - see the module docstring for the rules."""
    steps = [s for s in steps or [] if isinstance(s, dict) and "name" in s]
    files = _files_by_step(manifest_steps)
    names = [s["name"] for s in steps]
    roles = {}
    for step in steps:
        declared = _declared_role(step)
        if declared:
            roles[step["name"]] = declared
    for consumer in steps:
        if _writes_only_text(files.get(consumer["name"])):
            continue
        for _path, reference, _via in iter_previous_result_references(
            consumer, descend_into_from=True
        ):
            if not isinstance(reference, str):
                continue
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


def annotate_roles(steps, manifest_steps):
    """Copies of `manifest_steps`, each with its step's `role`."""
    roles = step_roles(steps, manifest_steps)
    annotated = []
    for entry in manifest_steps or []:
        if not isinstance(entry, dict):
            annotated.append(entry)
            continue
        name = entry.get("parent_step") or entry.get("step")
        annotated.append({**entry, ROLE_KEY: roles.get(name, FINAL)})
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
