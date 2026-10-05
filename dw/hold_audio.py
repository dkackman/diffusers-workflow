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
- a literal path, or an `asset:`/`output:` reference whose name carries an
  extension that is not audio, and any `{"media_type": ...}` reference - those
  load an image or a video.

A `previous_result:` names no file yet and is left to the run-time check, the
same as `dw/video_extensions.py` leaves it.
"""

from . import references
from .adapter_compatibility import FROM_PRETRAINED_KEY, H3_WORKFLOWS, WORKFLOW_KEY
from .argument_media import is_media_reference
from .for_each import MEMBER_SEPARATOR, render_path
from .pipeline_processors.h3_blocks import HOLD_AUDIO_INPUT
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
    if not isinstance(value, str):
        return None
    if value.startswith(("http://", "https://")):
        return None
    if references.is_ref(references.UNRESOLVED, value):
        return None
    if references.is_ref((references.CONSTANT, references.PROMPT), value):
        return None
    ext = value.rsplit(".", 1)
    if len(ext) != 2 or not ext[1] or "/" in ext[1]:
        return None
    ext = f".{ext[1].lower()}"
    if ext in ALLOWED_AUDIO_EXTENSIONS:
        return None
    return (
        f"hold_audio holds a soundtrack, and '{value}' is not an audio file "
        f"({', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))})"
    )


def _not_h3(pipeline):
    """Why this step's pipeline cannot hold audio, or None when it can or
    cannot be told before the load."""
    configuration = pipeline.get("configuration")
    component_type = (
        configuration.get("component_type") if isinstance(configuration, dict) else None
    )
    from_pretrained = pipeline.get(FROM_PRETRAINED_KEY)
    workflow = (
        from_pretrained.get(WORKFLOW_KEY) if isinstance(from_pretrained, dict) else None
    )
    holds = (
        f"hold_audio is a MiniMax-H3 argument, taken by its "
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


__all__ = ["hold_audio_errors"]
