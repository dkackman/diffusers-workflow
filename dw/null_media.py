"""Catch, at validation time, an object description whose media resolved null.

`realize_args` builds a pipeline argument that names a type and where its media
comes from - MiniMax-H3's references are the catalog's example. When that media
source resolves to null (typically a `variable:` left unset), the object is
`OMITTED`: dropped silently when it sits in a list, since that is how a
template makes a reference optional (`dw/arguments.py`'s list branch). But the
*same* dict sitting directly under a key, not in a list, has nothing to leave
it out of - `realize_args` raises there, at run time, after the checkpoint is
already loaded (#478).

This mirrors that raise, not the silent drop: only a bare object description
in "dict context" is an error here, exactly the shapes `realize_args`' own
dict branch would refuse. One found inside a list is left alone, since a
template author relies on that being silently dropped rather than reported.
"""

from .arguments import (
    FROM_ARGUMENTS_KEY,
    FROM_FILE_KEY,
    FROM_PREVIOUS_RESULT_KEY,
    _names_no_media,
)
from .for_each import MEMBER_SEPARATOR, render_path

_FROM_KEYS = (FROM_FILE_KEY, FROM_PREVIOUS_RESULT_KEY, FROM_ARGUMENTS_KEY)


def _walk(value, path, in_list, errors):
    if isinstance(value, dict):
        if not in_list and _names_no_media(value):
            from_key = next(k for k in _FROM_KEYS if value.get(k, False) is None)
            errors.append(
                {
                    "path": render_path(path + (from_key,)),
                    "message": (
                        f"'{path[-1]}' names an object to build but the media "
                        f"it would be built from is null. An optional one "
                        f"belongs in a list, where it can be left out; on its "
                        f"own there is nothing to leave it out of"
                    ),
                }
            )
            return
        for key, item in value.items():
            _walk(item, path + (key,), False, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk(item, path + (index,), True, errors)


def null_media_errors(workflow_definition, source_indices=None):
    """Every bare object description whose media is null, as [{path, message}].

    The definition handed here has already been substituted and expanded, so a
    `for_each` member's own arguments are checked as they will run;
    `source_indices` maps each expanded step back to the step the author
    wrote, and the member is named in the message - the same convention
    reference_limit_errors uses.
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
        step_errors = []
        for key, value in arguments.items():
            _walk(
                value,
                ("steps", source, "pipeline", "arguments", key),
                False,
                step_errors,
            )
        for error in step_errors:
            errors.append(
                {
                    "path": error["path"],
                    "message": f"{error['message']}{where}.",
                }
            )
    return errors
