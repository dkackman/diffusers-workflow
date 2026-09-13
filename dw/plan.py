"""The plan a validate call answers with: what a run of a workflow with a
caller's arguments will actually execute, what it will have to download
first, and what the workflow's own cost block says it will take - with a
fingerprint over the work, so an acknowledgement can be bound to it and a
run whose shape changed after consent refused (#85, stage 2).

Everything here is derived from the same resolvers the run uses -
`realize_workflow` folds the arguments and inlines the prompts, and
`Workflow.expanded_definition` substitutes and expands `for_each` - so the
plan describes the run and not an approximation of it. Nothing here knows a
model: every minute comes from a `cost` block and every repo name from a
`from_pretrained_arguments`.
"""

import copy
import hashlib
import json
import logging
import os

from huggingface_hub import model_info

from .hub_cache import scan_models
from .realize import (
    BUILTIN_PREFIX,
    VARIABLE_PREFIX,
    read_sub_workflow,
    realize_workflow,
)
from .security import validate_url
from .workflow import Workflow

logger = logging.getLogger("dw")

FINGERPRINT_PREFIX = "sha256:"
# Top-level keys that document a workflow rather than shape its work
DOCUMENTATION_KEYS = ("cost", "description", "summary", "configures")
FOR_EACH_KEY = "for_each"
SIZE_LOOKUP_TIMEOUT = 5.0
GIB = 1024**3


def build_plan(
    candidate,
    arguments,
    *,
    device,
    prompt_dir=None,
    cache_dir=None,
    lookup_sizes=True,
    cache_probe=None,
):
    """What a run of `candidate` with `arguments` will execute and cost.

    Args:
        candidate: The Workflow the route built - it carries the file spec
            (so base_dir), the output root and the confinement a run has.
        arguments: The caller's arguments, already past `argument_errors`;
            an undeclared name or an uncoercible value raises here.
        device: The backend that is serving - 'cuda', 'mps' or 'cpu'.
        prompt_dir: The prompt library, for inlining.
        cache_dir: The hub cache to check downloads against; None for the
            default.
        lookup_sizes: Whether to ask the hub how large a missing repo is.
        cache_probe: Stage 2's step-cache probe; unused, `cached_steps` is
            always None until then.
    """
    definition = candidate.workflow_definition
    base_dir = (
        os.path.dirname(os.path.abspath(candidate.file_spec))
        if candidate.file_spec
        else None
    )
    realized, _ = realize_workflow(
        definition,
        arguments,
        seed=0,
        base_dir=base_dir,
        prompt_dir=prompt_dir,
        output_root=candidate.output_dir,
        workflow_dir=candidate.workflow_dir,
        pin_outputs=False,
    )
    # Arguments are already folded into the realized variables, so the
    # expansion takes none; it substitutes and expands exactly as the run
    expanded = Workflow(
        realized, candidate.output_dir, candidate.file_spec, candidate.workflow_dir
    ).expanded_definition()
    entries = list_entries(definition, realized)
    return {
        "fingerprint": fingerprint(expanded, definition),
        "steps": len(expanded.get("steps") or []),
        "list_entries": entries,
        "cached_steps": None,
        "downloads_required": [],
        "estimate": None,
    }


def list_entries(definition, realized):
    """{variable: length} for every `for_each` that names a list variable,
    read from the folded variables - a literal list is not an argument and
    is not listed."""
    variables = realized.get("variables") or {}
    entries = {}
    for step in definition.get("steps") or []:
        if not isinstance(step, dict):
            continue
        reference = step.get(FOR_EACH_KEY)
        if isinstance(reference, str) and reference.startswith(VARIABLE_PREFIX):
            name = reference.removeprefix(VARIABLE_PREFIX)
            value = variables.get(name)
            if isinstance(value, list):
                entries[name] = len(value)
    return entries


def fingerprint(expanded, definition):
    """SHA-256 over the expanded definition with everything that is not
    work removed: the seed wherever it sits, and the documentation keys.

    `definition` is the workflow as written, consulted for whether the
    top-level seed named a variable - if it did, that variable's folded
    value is the seed too and is blanked at its source.
    """
    doc = copy.deepcopy(expanded)
    doc.pop("seed", None)
    for key in DOCUMENTATION_KEYS:
        doc.pop(key, None)
    written_seed = definition.get("seed")
    if isinstance(written_seed, str) and written_seed.startswith(VARIABLE_PREFIX):
        name = written_seed.removeprefix(VARIABLE_PREFIX)
        variables = doc.get("variables")
        if isinstance(variables, dict) and name in variables:
            variables[name] = None
    for step in doc.get("steps") or []:
        if isinstance(step, dict):
            step.pop("seed", None)
            pipeline = step.get("pipeline")
            if isinstance(pipeline, dict):
                pipeline.pop("seed", None)
    # default=repr: a realized 'constant:' can be any Python value, and the
    # fingerprint only needs it to be stable, not round-trippable
    serialized = json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=repr
    )
    return FINGERPRINT_PREFIX + hashlib.sha256(serialized.encode("utf-8")).hexdigest()
