"""How many references of each kind a pipeline's request may carry.

A pipeline that conditions on reference media - MiniMax-H3's `ref2va` is the
one in the catalog - bounds what it accepts: so many images, so many videos,
so many audio clips, so many in total, and for H3 an audio reference may never
be the only one. Those bounds are enforced by the pipeline itself, which means
they are enforced *after* the checkpoint is loaded: a caller who ran the free
`validate_workflow`, was quoted eight minutes and acknowledged the cost found
out minutes in, from a failed job, what a millisecond of arithmetic could have
told them (#136).

The numbers are not written here. They live on the diffusers block that
enforces them, as its constructor's defaults, and this module reads them off
that block - so a diffusers release that raises a limit raises it here too,
and the engine holds no model knowledge but the name of the class that
declares the limits (see REFERENCE_LIMIT_BLOCKS). A family diffusers has no
such block for is simply not checked: this pass only ever refuses a request
the pipeline itself would refuse.
"""

import importlib
import inspect
import logging

from .for_each import MEMBER_SEPARATOR, render_path

logger = logging.getLogger("dw")

# The module a family's reference classes live in -> the block whose
# __init__ defaults declare that family's limits. A pointer, not a number:
# what a limit *is* stays diffusers', which is the only place it can stay
# correct across a release
REFERENCE_LIMIT_BLOCKS = {
    "diffusers.modular_pipelines.minimax_h3": (
        "diffusers.modular_pipelines.minimax_h3.before_encoder",
        "MiniMaxH3Ref2VASetupStep",
    ),
}

# Families where an audio reference may not stand alone - H3 conditions a
# soundtrack on a picture, so audio by itself has nothing to speak over
AUDIO_NEEDS_A_PICTURE = frozenset(REFERENCE_LIMIT_BLOCKS)

REFERENCE_TYPE_KEY = "reference_type"

# Values substitution resolves before this pass runs; one still spelled out
# is another pass's complaint, not this one's
_UNRESOLVED_PREFIXES = ("variable:", "item:", "previous_result:", "gather:")


def _family(module_name):
    """The REFERENCE_LIMIT_BLOCKS key a class's module belongs to, or None."""
    for family in REFERENCE_LIMIT_BLOCKS:
        if module_name == family or module_name.startswith(family + "."):
            return family
    return None


def _limits(family):
    """The (model name, per-kind limits, total) a family declares.

    Read from the block's constructor signature rather than from an instance:
    constructing one is cheap but not free, and a default is exactly what the
    signature holds. The model name is the block's own, so even the family's
    name in the message is diffusers'.
    """
    module_path, class_name = REFERENCE_LIMIT_BLOCKS[family]
    try:
        block = getattr(importlib.import_module(module_path), class_name)
        parameters = inspect.signature(block.__init__).parameters
    except Exception:
        # A diffusers that renamed or dropped the block - checking nothing is
        # the right failure here, since the pipeline still enforces its own
        logger.debug(f"No reference limits available from {family}", exc_info=True)
        return None, None, None

    per_kind = {}
    total = None
    for name, parameter in parameters.items():
        if parameter.default is inspect.Parameter.empty:
            continue
        if name == "max_references":
            total = parameter.default
        elif name.startswith("max_") and name.endswith("s"):
            per_kind[name[len("max_") : -1]] = parameter.default
    model_name = getattr(block, "model_name", family.rsplit(".", 1)[-1])
    return model_name, (per_kind or None), total


def _reference_class(value):
    """The class a reference entry's '*_type' names, or None.

    Anything that does not resolve is left alone: realize_args reports a type
    it cannot load, with a better message than this pass could give.
    """
    if not isinstance(value, dict):
        return None
    name = value.get(REFERENCE_TYPE_KEY)
    if not isinstance(name, str) or name.startswith(_UNRESOLVED_PREFIXES):
        return None
    module_name, _, class_name = name.rpartition(".")
    if not module_name:
        return None
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except Exception:
        return None


def _kinds(entries):
    """(family, [kind, ...]) for a list of reference entries, or None.

    A family is the one whose limits this pass knows how to read; an entry
    whose class carries no 'kind', or a list mixing two families, is not a
    reference set this pass understands.
    """
    if not isinstance(entries, list) or not entries:
        return None
    found = None
    kinds = []
    for entry in entries:
        reference = _reference_class(entry)
        kind = getattr(reference, "kind", None)
        if not isinstance(kind, str):
            return None
        family = _family(getattr(reference, "__module__", ""))
        if family is None:
            return None
        if found is None:
            found = family
        elif found != family:
            return None
        kinds.append(kind)
    return (found, kinds) if found else None


def _errors_for(kinds, family):
    """Every limit a set of reference kinds breaks, as messages."""
    model_name, per_kind, total = _limits(family)
    if per_kind is None and total is None:
        return []

    messages = []
    for kind, limit in sorted((per_kind or {}).items()):
        count = kinds.count(kind)
        if count > limit:
            messages.append(
                f"{model_name} accepts at most {limit} "
                f"{kind} reference{'s' if limit != 1 else ''}, got {count}."
            )
    if total is not None and len(kinds) > total:
        messages.append(
            f"{model_name} accepts at most {total} references in total, "
            f"got {len(kinds)}."
        )
    if family in AUDIO_NEEDS_A_PICTURE and set(kinds) == {"audio"}:
        messages.append(
            "An audio reference has to be paired with at least one image or "
            "video reference and cannot be used on its own."
        )
    return messages


def reference_limit_errors(workflow_definition, source_indices=None):
    """Every reference list a pipeline would refuse, as [{path, message}].

    The definition handed here has already been substituted and expanded, so a
    `for_each` member's own references are checked as they will run;
    `source_indices` maps each expanded step back to the step the author
    wrote, and the member is named in the message - the same convention
    subfolder_errors uses.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        pipeline = step.get("pipeline")
        arguments = pipeline.get("arguments") if isinstance(pipeline, dict) else None
        if not isinstance(arguments, dict):
            continue
        source = (
            source_indices[index]
            if source_indices is not None and index < len(source_indices)
            else index
        )
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        for key, value in arguments.items():
            found = _kinds(value)
            if found is None:
                continue
            family, kinds = found
            for message in _errors_for(kinds, family):
                errors.append(
                    {
                        "path": render_path(
                            ("steps", source, "pipeline", "arguments", key)
                        ),
                        "message": f"{message.rstrip('.')}{where}.",
                    }
                )
    return errors
