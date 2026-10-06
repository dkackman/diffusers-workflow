"""A `hold_audio` argument, checked before the run when it can be.

`hold_audio` is the MiniMax-H3 soundtrack a step generates its video to
(dw/pipeline_processors/h3_blocks.py). Only an H3 core-denoise pipeline has the
blocks that take it, and only audio can be held - so both are refused here,
where the answer is knowable without loading anything, rather than after a
checkpoint load:

- on a step whose pipeline is not a modular pipeline, or whose
  `from_pretrained` `workflow` is not one of H3's three (`t2va`, `fl2va`,
  `ref2va`). A modular pipeline loaded with no `workflow` is left to the
  run-time check, which asks the loaded block graph itself;
- a literal path, or an `asset:`/`output:` reference, whose name carries no
  audio extension (a bare word like `hello` among them), any value that is not
  a string or a mapping (a number), and any `{"media_type": ...}` reference -
  those load an image or a video.

A `previous_result:` names no file yet and is left to the run-time check, the
same as `dw/video_extensions.py` leaves it.

`refine_strength`, the refine that re-denoises an upscaled take from a low sigma,
rides on the same blocks and is checked beside it: refused on a step that is not
H3, when it is not a number in (0, 1), when the step passes no `latents` to refine
or no `hold_audio` to keep its soundtrack, and when `num_inference_steps` is
below 2 - the refine schedule is that many points, so one point is no step.
"""

from . import references
from .adapter_compatibility import FROM_PRETRAINED_KEY, H3_WORKFLOWS, WORKFLOW_KEY
from .argument_media import is_media_reference
from .for_each import MEMBER_SEPARATOR, render_path
from .pipeline_processors.h3_blocks import HOLD_AUDIO_INPUT, REFINE_STRENGTH_INPUT
from .security import ALLOWED_AUDIO_EXTENSIONS

MODULAR_PIPELINE = "ModularPipeline"


def _not_audio(value):
    """Why this `hold_audio` value is not audio, or None - including None for
    anything not yet knowable."""
    if is_media_reference(value):
        # fetch_media loads it as an image or a video before the call
        return (
            f"hold_audio holds a soundtrack, and this is a "
            f"'{value.get('media_type')}' reference - name the audio file "
            f"directly, or with 'asset:', 'output:' or 'previous_result:'"
        )
    if isinstance(value, dict):
        return None
    if not isinstance(value, str):
        # A number or a list is no file and no reference - the call refuses it
        return (
            f"hold_audio holds a soundtrack, and {value!r} is not one - name an "
            f"audio file directly, or with 'asset:', 'output:' or "
            f"'previous_result:'"
        )
    if value.startswith(("http://", "https://")):
        return None
    if references.is_ref(references.UNRESOLVED, value):
        return None
    if references.is_ref((references.CONSTANT, references.PROMPT), value):
        return None
    # A name with no extension - a bare word like 'hello' - is refused at the
    # call too, which checks the extension before it opens anything
    _, dot, ext = value.rpartition(".")
    if dot and ext and "/" not in ext and f".{ext.lower()}" in ALLOWED_AUDIO_EXTENSIONS:
        return None
    return (
        f"hold_audio holds a soundtrack, and '{value}' is not an audio file "
        f"({', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))})"
    )


def _not_h3(pipeline, argument=HOLD_AUDIO_INPUT):
    """Why this step's pipeline cannot take `argument` - one of the H3 block
    arguments - or None when it can or cannot be told before the load."""
    configuration = pipeline.get("configuration")
    component_type = (
        configuration.get("component_type") if isinstance(configuration, dict) else None
    )
    from_pretrained = pipeline.get(FROM_PRETRAINED_KEY)
    workflow = (
        from_pretrained.get(WORKFLOW_KEY) if isinstance(from_pretrained, dict) else None
    )
    holds = (
        f"{argument} is a MiniMax-H3 argument, taken by its "
        f"{', '.join(sorted(H3_WORKFLOWS))} workflows"
    )
    if (
        isinstance(component_type, str)
        and not references.is_ref(references.UNRESOLVED, component_type)
        and component_type.rsplit(".", 1)[-1] != MODULAR_PIPELINE
    ):
        return f"{holds}, and this step loads {component_type}"
    if isinstance(workflow, str) and workflow not in H3_WORKFLOWS:
        return f"{holds}, and this step loads the '{workflow}' workflow"
    return None


def hold_audio_errors(workflow_definition, source_indices=None):
    """Every `hold_audio` argument refused before the run, as [{path, message}].

    Walks the substituted, expanded definition, the convention
    `video_extension_errors` follows: `source_indices` maps an expanded step
    back to the one the author wrote, and a path inside a `for_each` member
    names the member.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        pipeline = step.get("pipeline")
        if not isinstance(pipeline, dict):
            continue
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict) or arguments.get(HOLD_AUDIO_INPUT) is None:
            continue
        source = references.author_index(source_indices, index)
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        path = render_path(("steps", source, "pipeline", "arguments", HOLD_AUDIO_INPUT))
        for problem in (_not_h3(pipeline), _not_audio(arguments[HOLD_AUDIO_INPUT])):
            if problem is not None:
                errors.append({"path": path, "message": f"{problem}{where}"})
    return errors


def _refine_problems(arguments):
    """Why this step's `refine_strength` cannot run, as a list - empty when it
    can or cannot be told before the run."""
    problems = []
    strength = arguments[REFINE_STRENGTH_INPUT]
    if isinstance(strength, str) and references.is_ref(references.UNRESOLVED, strength):
        pass
    elif isinstance(strength, bool) or not isinstance(strength, (int, float)):
        problems.append(
            f"refine_strength is a number in (0, 1), and {strength!r} is not a number"
        )
    elif not 0 < strength < 1:
        problems.append(
            f"refine_strength is a sigma in (0, 1) - about 0.2 refines an "
            f"upscaled take - and {strength} is outside it"
        )
    if arguments.get("latents") is None:
        problems.append(
            "refine_strength re-denoises the 'latents' it is passed - pass the "
            "upscaled latents, e.g. 'previous_result:up'"
        )
    if arguments.get(HOLD_AUDIO_INPUT) is None:
        problems.append(
            "refine_strength re-denoises the video only, so it needs 'hold_audio' "
            "to keep a soundtrack - e.g. the base pass's 'previous_result:base.audio'"
        )
    steps = arguments.get("num_inference_steps")
    if isinstance(steps, (int, float)) and not isinstance(steps, bool) and steps < 2:
        problems.append(
            f"refine_strength runs num_inference_steps - 1 denoise steps, so "
            f"num_inference_steps must be 2 or more, not {steps}"
        )
    return problems


def refine_strength_errors(workflow_definition, source_indices=None):
    """Every `refine_strength` argument refused before the run, as
    [{path, message}] - walked the way `hold_audio_errors` walks."""
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        pipeline = step.get("pipeline")
        if not isinstance(pipeline, dict):
            continue
        arguments = pipeline.get("arguments")
        if (
            not isinstance(arguments, dict)
            or arguments.get(REFINE_STRENGTH_INPUT) is None
        ):
            continue
        source = references.author_index(source_indices, index)
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        path = render_path(
            ("steps", source, "pipeline", "arguments", REFINE_STRENGTH_INPUT)
        )
        not_h3 = _not_h3(pipeline, REFINE_STRENGTH_INPUT)
        problems = ([not_h3] if not_h3 else []) + _refine_problems(arguments)
        for problem in problems:
            errors.append({"path": path, "message": f"{problem}{where}"})
    return errors


__all__ = ["hold_audio_errors", "refine_strength_errors"]
