"""What a pipeline's own call refuses about a literal `num_frames`, `width` or
`height`, checked before the load (#1361).

A step that loads MiniMax-H3 with `num_frames: 25` passed `validate_workflow`
whenever the workflow declared no `variable_constraints` - which is how an
inline workflow an agent writes is usually spelled - and the refusal arrived
from diffusers after the checkpoint had loaded. `variable_constraints`
(dw/variable_constraints.py) is still how a template refuses a value before
substitution; this pass covers the literal a workflow wrote with no rule.

The numbers are not written here, as in dw/reference_limits.py (#136): they
are properties of the diffusers pipeline class that enforces them, read off
that class with no components loaded, where each falls back to the released
checkpoint's value. The engine names the class (PIPELINE_LIMIT_CLASSES),
never a number. The one exception is the +5 of the 17n + 5 frame grid, which
is `h3_rules.RENDER_GRID` - already the owner of that arithmetic, and pinned
to the VAE by tests/test_h3_guides.py - so it is read from there rather than
copied a second time. A diffusers that renames or drops the class or a
property checks nothing, with a debug log: the pipeline still refuses at run
time, as it did before this pass existed.

Only what the run would refuse is refused. An off-grid count the pipeline
rounds up and accepts - 130 runs as 141 - is left alone, and a value whose
written argument is a `variable:` (or `item:` field) the workflow constrains
is the declared constraint's to refuse, so it is reported once, where
constraint_errors puts it. A windowed refine - a step with both
`refine_windows` and `refine_strength` - or the experimental windowed
generation (`experimental_hybrid_windows`, behind its flag) runs a take of
any length, since dw's layout step lifts `max_duration` for that call
(`h3_rules.lifts_duration_ceiling`, the rule both ask), so its `num_frames`
keeps the floor and loses the ceiling.
"""

import importlib
import logging

from . import references
from .for_each import render_path
from .pipeline_processors.h3_rules import (
    RENDER_GRID,
    is_h3_step,
    lifts_duration_ceiling,
)
from .variable_constraints import aligned, aligned_down, declared_constraints

logger = logging.getLogger("dw")

# The module and class whose properties declare H3's frame and canvas
# limits. A pointer, not a number: what a limit *is* stays diffusers'
PIPELINE_LIMIT_CLASSES = {
    "minimax_h3": (
        "diffusers.modular_pipelines.minimax_h3.modular_pipeline",
        "MiniMaxH3ModularPipeline",
    ),
}

# The properties read off the class
LIMIT_PROPERTIES = (
    "fps",
    "min_duration",
    "max_duration",
    "vae_frames_per_chunk",
    "canvas_multiple",
)

FRAMES_ARGUMENT = "num_frames"
CANVAS_ARGUMENTS = ("width", "height")


def _limits(family="minimax_h3"):
    """The family's limits as {property: value}, or None when they cannot be
    read.

    Read without constructing the pipeline: `object.__new__` gives the class's
    property getters a bare `self`, on which every component is absent and
    each property answers with the released model's value - the read
    tests/test_variable_constraints.py makes for the template rule.
    """
    module_path, class_name = PIPELINE_LIMIT_CLASSES[family]
    try:
        cls = getattr(importlib.import_module(module_path), class_name)
        bare = object.__new__(cls)
        limits = {name: getattr(bare, name) for name in LIMIT_PROPERTIES}
    except Exception:
        logger.debug(f"No pipeline limits available from {class_name}", exc_info=True)
        return None
    if limits["vae_frames_per_chunk"] != RENDER_GRID["modulus"]:
        # The grid's remainder is only meaningful on the chunk it was pinned to
        logger.debug(
            f"{class_name}.vae_frames_per_chunk is {limits['vae_frames_per_chunk']}, "
            f"not the {RENDER_GRID['modulus']} RENDER_GRID is pinned to"
        )
        return None
    limits["class_name"] = class_name
    return limits


def _grid(limits):
    return {
        "modulus": limits["vae_frames_per_chunk"],
        "remainder": RENDER_GRID["remainder"],
    }


def accepted_frames(limits):
    """The smallest and largest on-grid `num_frames` inside the duration
    window - (124, 345) on today's diffusers."""
    grid = _grid(limits)
    fps = limits["fps"]
    low = aligned(int(-(-limits["min_duration"] * fps // 1)), grid)
    high = aligned_down(int(limits["max_duration"] * fps // 1), grid)
    return low, high


def _literal_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def frames_problem(value, limits, lifted=False):
    """Why a literal `num_frames` is outside the pipeline's duration window,
    or None. `lifted` is a windowed refine (`h3_rules.lifts_duration_ceiling`),
    whose take the layout step lays out past `max_duration`: only the floor is
    asked of it."""
    grid = _grid(limits)
    rounded = aligned(value, grid)
    fps = limits["fps"]
    seconds = rounded / fps
    if limits["min_duration"] <= seconds and (
        lifted or seconds <= limits["max_duration"]
    ):
        return None
    low, high = accepted_frames(limits)
    nearest = low if seconds < limits["min_duration"] else high
    return (
        f"MiniMax-H3 generates {limits['min_duration']:g} to "
        f"{limits['max_duration']:g} seconds at {fps:g} fps "
        f"({limits['class_name']}): num_frames {value} rounds up to {rounded} "
        f"on the {grid['modulus']}n + {grid['remainder']} grid, "
        f"{seconds:.2f} s, so the pipeline would refuse it after the load. "
        f"The accepted counts are {low} to {high} on that grid - use {nearest}"
    )


def canvas_problem(key, value, limits):
    """Why a literal `width` or `height` is off the pipeline's canvas
    multiple, or None."""
    multiple = limits["canvas_multiple"]
    if value % multiple == 0:
        return None
    below = value // multiple * multiple
    return (
        f"MiniMax-H3's {key} is a multiple of {multiple} "
        f"({limits['class_name']}.canvas_multiple), and {value} is not - "
        f"use {below} or {below + multiple}"
    )


def _constrained_names(definition):
    """Variable and `for_each` entry field names a declared constraint owns."""
    return set(declared_constraints(definition or {}))


def _owned_by_constraint(written, source, key, constrained):
    """Whether the written step's argument `key` is a `variable:` or `item:`
    reference whose name a declared constraint covers."""
    if not constrained or not isinstance(written, dict):
        return False
    steps = written.get("steps")
    if not isinstance(steps, list) or not 0 <= source < len(steps):
        return False
    step = steps[source]
    pipeline = step.get("pipeline") if isinstance(step, dict) else None
    arguments = pipeline.get("arguments") if isinstance(pipeline, dict) else None
    value = arguments.get(key) if isinstance(arguments, dict) else None
    for kind in (references.VARIABLE, references.ITEM):
        if isinstance(value, str) and references.is_ref(kind, value):
            return references.ref_name(kind, value) in constrained
    return False


def pipeline_limit_errors(workflow_definition, source_indices=None, written=None):
    """Every literal H3 `num_frames`, `width` or `height` the pipeline would
    refuse, as [{path, message}].

    `workflow_definition` is substituted and expanded, so a `for_each`
    member's own values are checked as they will run; `source_indices` maps
    each expanded step back to the step the author wrote, and the member is
    named in the message. `written` is the definition before substitution,
    read for which arguments a declared constraint already owns.
    """
    steps = [
        found
        for found in references.iter_steps(
            workflow_definition.get("steps"), source_indices, "pipeline"
        )
        if is_h3_step(found[2])
    ]
    if not steps:
        return []
    limits = _limits()
    if limits is None:
        return []
    constrained = _constrained_names(written)
    errors = []
    for _, _, pipeline, source, where in steps:
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict):
            continue
        for key in (FRAMES_ARGUMENT, *CANVAS_ARGUMENTS):
            value = arguments.get(key)
            if not _literal_int(value):
                continue
            if _owned_by_constraint(written, source, key, constrained):
                continue
            if key == FRAMES_ARGUMENT:
                message = frames_problem(
                    value, limits, lifts_duration_ceiling(arguments)
                )
            else:
                message = canvas_problem(key, value, limits)
            if message is None:
                continue
            errors.append(
                {
                    "path": render_path(
                        ("steps", source, "pipeline", "arguments", key)
                    ),
                    "message": f"{message.rstrip('.')}{where}.",
                }
            )
    return errors
