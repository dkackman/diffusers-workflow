"""A `result` block on a step whose command returns a scalar, not an
artifact (#212).

`judge` returns a bare `float` - a natural fit for `result` to look like it
applies (the guide's Result Configuration section describes saving text,
and a score reads like one), but there is no file to write. `validate_workflow`
answered `valid: true` and the run generated its full fan-out before dying
deep inside `save_artifact` on `write() argument must be str, not float`,
naming neither the step nor the command. Checked here against the command's
own declared `returns` kind (`dw/tasks/task.py`'s `register_command`) rather
than a name match on `judge`, so a future scalar-returning task is covered
by declaring itself rather than by a second special case here.
"""

from .for_each import MEMBER_SEPARATOR, render_path
from .tasks.task import task_command_info

RESULT_KEY = "result"


def scalar_result_errors(workflow_definition, source_indices=None):
    """Every `result` block on a scalar-returning command, as [{path, message}].

    The definition handed here has already been substituted and expanded,
    so a step's `task.command` is literal. `source_indices`, when given, is
    the source step index of each step - a `for_each` group turns one
    written step into several, and the path an error carries has to be one
    the author can find in the file they wrote; the member is named in the
    message.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        result = step.get(RESULT_KEY)
        if not isinstance(result, dict):
            continue
        task = step.get("task")
        if not isinstance(task, dict):
            continue
        command = task.get("command")
        if not isinstance(command, str):
            continue
        try:
            info = task_command_info(command)
        except ValueError:
            continue
        if info.get("returns") != "scalar":
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
        errors.append(
            {
                "path": render_path(("steps", source, RESULT_KEY)),
                "message": (
                    f"{command} returns a number, not an artifact - "
                    f"'result' cannot be saved{where}"
                ),
            }
        )
    return errors


__all__ = ["scalar_result_errors"]
