"""A step's result 'subfolder': where under the run directory its files go.

A run writes everything into one directory, so a finished episode sits
beside the twenty scratch files that went into it, distinguished only by
the step name in each file name. 'subfolder' on a step's result block puts
that step's files into a subfolder of the run directory instead - by
convention 'final' or 'intermediate', though the engine treats no name
specially. Nothing else about placement changes: a step without one writes
where it always did.

This module owns the shape check and the static pass over an expanded
definition. Containment - that the joined path really is inside the run
directory - is the engine's, at the moment it joins (see
Workflow.step_output_dir).
"""

from .for_each import MEMBER_SEPARATOR, render_path
from .security import (
    InvalidInputError,
    validate_file_base_name,
    validate_subfolder,
)

SUBFOLDER_KEY = "subfolder"
FILE_BASE_NAME_KEY = "file_base_name"

# Reference prefixes substitution resolves before this pass runs. One still
# spelled out here is one nothing resolved, and that is the undeclared-
# variable pass's complaint rather than a shape error
_UNRESOLVED_PREFIXES = ("variable:", "item:")


def step_subfolder(step_definition):
    """The validated subfolder a step's result names, or '' when it names
    none.

    Raises:
        InvalidInputError: If the value is not a string or not a valid
            subfolder
    """
    result = step_definition.get("result")
    if not isinstance(result, dict) or SUBFOLDER_KEY not in result:
        return ""
    value = result[SUBFOLDER_KEY]
    if not isinstance(value, str):
        raise InvalidInputError(
            f"Invalid subfolder: {value!r} - a subfolder is a string like 'final'"
        )
    return validate_subfolder(value)


def subfolder_errors(workflow_definition, source_indices=None):
    """Every result 'subfolder' or 'file_base_name' that cannot be written,
    as [{path, message}].

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
        if not isinstance(result, dict):
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
        for key, check in (
            (SUBFOLDER_KEY, validate_subfolder),
            (FILE_BASE_NAME_KEY, validate_file_base_name),
        ):
            if key not in result:
                continue
            value = result[key]
            if isinstance(value, str) and value.startswith(_UNRESOLVED_PREFIXES):
                continue
            try:
                if not isinstance(value, str):
                    raise InvalidInputError(
                        f"Invalid {key}: {value!r} - expected a string"
                    )
                check(value)
            except InvalidInputError as e:
                errors.append(
                    {
                        "path": render_path(("steps", source, "result", key)),
                        "message": f"{e}{where}",
                    }
                )
    return errors
