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

A subfolder is `output:`-addressable only up to OUTPUT_REFERENCE_PATTERN's
ceiling of seven path segments in all: a nested identity or subfolder counts
one per `/`, plus the run id and the file name. SUBFOLDER_PATTERN has no depth bound of its own, so a deep
subfolder under a nested identity validates and runs but cannot be named by a
later `output:` reference.
"""

from . import references
from .for_each import render_path
from .security import (
    InvalidInputError,
    validate_file_base_name,
    validate_subfolder,
)

SUBFOLDER_KEY = "subfolder"
FILE_BASE_NAME_KEY = "file_base_name"


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
    errors = []
    for _, _, result, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices, "result"
    ):
        for key, check in (
            (SUBFOLDER_KEY, validate_subfolder),
            (FILE_BASE_NAME_KEY, validate_file_base_name),
        ):
            if key not in result:
                continue
            value = result[key]
            # A prefix substitution resolves before this pass: one still
            # spelled out is the undeclared-variable pass's complaint, not a
            # shape error
            if references.is_ref(references.SUBSTITUTED, value):
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
