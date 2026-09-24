"""A step's result 'fps': the frame rate a video is written at.

Widened the same way `subfolder` was (dw/subfolders.py): the schema lets the
raw document hold a `variable:` or `item:` reference so a workflow can keep
one frame-rate variable rather than a literal duplicated between the step
that generates and the step that writes (#363). This module owns the
resolved-value check: once substitution has run, `result.fps` has to be a
real, positive frame rate, and it must be a whole number - `encode_video`
(dw/result.py, the writer used whenever a step's result carries audio) hands
the value straight to PyAV as `rate=int(fps)`, so a fractional rate like
23.976 would be silently floored to 23 rather than written as asked. A
frame_rate variable declared as a float (24.0) still passes, since it carries
no fractional part; only a genuine fraction is refused.

Containment doesn't apply here - there's no path to join - so unlike
subfolder_errors this only ever checks value shape.
"""

from .for_each import MEMBER_SEPARATOR, render_path

FPS_KEY = "fps"

# Reference prefixes substitution resolves before this pass runs. One still
# spelled out here is one nothing resolved, and that is the undeclared-
# variable pass's complaint rather than a shape error
_UNRESOLVED_PREFIXES = ("variable:", "item:")


def fps_errors(workflow_definition, source_indices=None):
    """Every result 'fps' that cannot be written, as [{path, message}].

    The definition handed here has already been substituted and expanded,
    so every value in it is literal; a 'variable:' or 'item:' still spelled
    out is left alone. `source_indices`, when given, is the source step
    index of each step - a 'for_each' group turns one written step into
    several, and the path an error carries has to be one the author can
    find in the file they wrote; the member is named in the message.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        result = step.get("result")
        if not isinstance(result, dict) or FPS_KEY not in result:
            continue
        value = result[FPS_KEY]
        if isinstance(value, str) and value.startswith(_UNRESOLVED_PREFIXES):
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
        message = None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            message = f"Invalid fps: {value!r} - fps is a number of frames per second"
        elif value <= 0:
            message = f"Invalid fps: {value!r} - fps must be greater than zero"
        elif float(value) != int(value):
            message = (
                f"Invalid fps: {value!r} - fps must be a whole number; the video "
                f"writer truncates a fractional rate rather than honoring it"
            )
        if message is not None:
            errors.append(
                {
                    "path": render_path(("steps", source, "result", FPS_KEY)),
                    "message": f"{message}{where}",
                }
            )
    return errors
