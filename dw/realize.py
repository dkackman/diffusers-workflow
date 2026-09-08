"""Realizing a workflow: a copy of a definition with every mutable input
pinned, so the file left beside a run's manifest reproduces that run however
the library, the catalog or the output tree change afterwards.

What "mutable" means here is precisely the set of things that can differ
between two runs of the same file: the arguments a caller passed, the seed a
seedless workflow drew, the text a stored prompt held at the time, and which
run 'output:.../latest/...' picked. Everything else - 'asset:', 'constant:',
'previous_result:', 'builtin:' and a sub-workflow's path - is a name whose
meaning is pinned by something already recorded (the asset library, the
manifest's dw_version, the file itself), so it is kept as written and, for a
local sub-workflow, digested into the manifest instead.

Two rules hold this module together. It never mutates its input: the caller
hands it the definition the run is about to work from. And it never fails a
run: a reference that will not resolve is left exactly as written, so the
engine raises its own error at the point it would have raised anyway.
"""

import copy
import hashlib
import logging
import os

from .prompts import PROMPT_PREFIX, fetch_prompt
from .runs import (
    LATEST,
    OUTPUT_PREFIX,
    is_output_reference,
    output_root as default_output_root,
    resolve_output_reference,
)
from .security import SecurityError, validate_workflow_path
from .variables import set_variables

logger = logging.getLogger("dw")

BUILTIN_PREFIX = "builtin:"
VARIABLE_PREFIX = "variable:"


def realize_workflow(
    definition,
    arguments,
    seed,
    base_dir=None,
    prompt_dir=None,
    output_root=None,
    workflow_dir=None,
):
    """A copy of `definition` with every mutable input pinned.

    Args:
        definition: The workflow as loaded, before Workflow.run's deep copy.
            Never mutated.
        arguments: The run's argument dict, folded into the variable defaults
            exactly as `set_variables` folds them for the run itself.
        seed: The seed the run resolved - an integer, never None, because
            `Workflow.run` draws one when the workflow names none.
        base_dir: The workflow file's directory, anchoring prompt discovery
            and a sub-workflow's relative path.
        prompt_dir: The prompt library, for `prompt:` inlining.
        output_root: The output directory `output:` names resolve against.
        workflow_dir: The root a sub-workflow path is confined to, as
            `Workflow` confines it; None for an unconfined CLI run.

    Returns:
        (realized, annotations) - the pinned copy, and
        {"prompts": [name, ...], "sub_workflows": {path: sha256 or None}}
        for the manifest to carry, since the schema has nowhere to put them.
    """
    annotations = {"prompts": [], "sub_workflows": {}}
    realized = copy.deepcopy(definition)

    variables = realized.get("variables")
    if isinstance(variables, dict):
        # Exactly what the run computed: set_variables coerces each value to
        # the type of the declared default and rejects an undeclared name
        set_variables(arguments or {}, variables)

    realized["seed"] = seed
    # A definition can point its top-level seed at a declared variable
    # ('"seed": "variable:seed_arg"') rather than an integer, so the run's
    # resolved seed can also be read wherever else the workflow names that
    # variable. Pinning the top-level field alone would leave the variable's
    # own default whatever it was written as (typically none) - and a rerun
    # of this realized copy with no seed argument would put that null
    # default back over the pinned integer everywhere but the top level.
    definition_seed = definition.get("seed")
    if isinstance(definition_seed, str) and definition_seed.startswith(VARIABLE_PREFIX):
        seed_variable = definition_seed.removeprefix(VARIABLE_PREFIX)
        if isinstance(variables, dict) and seed_variable in variables:
            variables[seed_variable] = seed
    realized = _pin(realized, annotations, base_dir, prompt_dir, output_root)
    _record_sub_workflows(realized.get("steps"), annotations, base_dir, workflow_dir)
    return realized, annotations


def strings_with_prefix(tree, prefix):
    """Every string in a nested dict/list tree that starts with `prefix`, in
    first-seen order, deduplicated."""
    found = []

    def collect(value):
        if value.startswith(prefix) and value not in found:
            found.append(value)
        return value

    _map_strings(tree, collect)
    return found


def _map_strings(value, transform):
    """Rebuild `value`, replacing every string it contains with
    `transform(string)`. One recursive walk over dicts, lists and strings -
    the shape `referenced_result_names` in dw/step_cache.py walks - shared by
    `_pin` (which rewrites matching references) and `strings_with_prefix`
    (which only collects them), so a reference is found wherever it sits: a
    pipeline argument, a task argument, a sub-workflow's argument map, an
    element of a list.
    """
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, dict):
        return {key: _map_strings(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_strings(item, transform) for item in value]
    return value


def _pin(value, annotations, base_dir, prompt_dir, output_root):
    """Rebuild a value with prompt and output references pinned."""

    def transform(string):
        if string.startswith(PROMPT_PREFIX):
            return _inline_prompt(string, annotations, prompt_dir, base_dir)
        if is_output_reference(string):
            return _pin_output(string, output_root)
        return string

    return _map_strings(value, transform)


def _inline_prompt(reference, annotations, prompt_dir, base_dir):
    """The stored text, and the name recorded for the manifest.

    A stored prompt's text may not itself begin with a reference prefix (an
    engine rule `fetch_prompt` enforces), so inlining cannot introduce a
    second resolution.
    """
    try:
        text = fetch_prompt(reference, prompt_dir, base_dir)
    except (SecurityError, OSError, ValueError) as e:
        logger.warning(f"Realization kept {reference} as written: {e}")
        return reference
    name = reference.removeprefix(PROMPT_PREFIX).strip()
    if name not in annotations["prompts"]:
        annotations["prompts"].append(name)
    return text


def _pin_output(reference, output_root):
    """'output:<identity>/latest/<file>' rewritten to the run it resolved to.

    An explicit run id is already pinned, so it is returned untouched without
    touching the disk - realizing must not fail on a reference the run has
    not reached yet.
    """
    name = reference.removeprefix(OUTPUT_PREFIX).strip()
    if LATEST not in name.split("/"):
        return reference
    root = output_root or default_output_root()
    try:
        resolved = resolve_output_reference(reference, root)
        relative = os.path.relpath(resolved, root).replace(os.sep, "/")
    except (SecurityError, OSError, ValueError) as e:
        logger.warning(f"Realization kept {reference} as written: {e}")
        return reference
    return f"{OUTPUT_PREFIX}{relative}"


def _record_sub_workflows(steps, annotations, base_dir, workflow_dir):
    """Digest every sub-workflow a step names by local path.

    The schema's 'workflow' step takes a path, not a definition, so the
    realized file keeps the path and the manifest records what the file held.
    A builtin is packaged with the engine and pinned by the manifest's
    dw_version, so it is not digested.
    """

    def scan(value):
        if isinstance(value, dict):
            reference = value.get("workflow")
            if isinstance(reference, dict):
                path = reference.get("path")
                if isinstance(path, str) and not path.startswith(BUILTIN_PREFIX):
                    annotations["sub_workflows"][path] = _digest(
                        path, base_dir, workflow_dir
                    )
            for item in value.values():
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)

    for step in steps or []:
        scan(step)


def _digest(path, base_dir, workflow_dir):
    """The SHA-256 of a sub-workflow file, or None when it cannot be read.

    Resolved the way `Workflow.create_step_action` resolves it - relative to
    the referencing file's directory, then through `validate_workflow_path`
    confined to `workflow_dir` - so a path this run could not have loaded is
    not one realization reads either.
    """
    try:
        candidate = (
            path
            if os.path.isabs(path)
            else os.path.normpath(os.path.join(base_dir or ".", path))
        )
        validated = validate_workflow_path(candidate, workflow_dir)
        with open(validated, "rb") as file:
            return hashlib.sha256(file.read()).hexdigest()
    except (SecurityError, OSError, ValueError) as e:
        logger.debug(f"No digest for sub-workflow {path}: {e}")
        return None
