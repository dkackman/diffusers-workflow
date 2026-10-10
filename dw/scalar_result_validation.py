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

A "json" command (the assessment probes, #387) answers a dict of
measurements, which `Result.save` writes whole only under
`application/json`; any other content type explodes the dict key by key
into files, or dies on a number. So a `result` on one is allowed and must
say `application/json`.

`transcribe_audio` (#498) is a third shape: its `returns` is declared
"artifact" because that is what it answers by default (plain text), but a
literal `timestamps` of "segment"/"word" switches its return value to a
{text, chunks} dict - chunks being a list, which explodes fine under
`application/json` but dies with a bare `write() argument must be str, not
list` under `text/plain`. Checked here as a special case keyed on the
command name and the literal argument value, because unlike `judge`'s or
the probes' `returns`, the shape is not a property of the command alone -
`timestamps` reached through a `variable:` is invisible here and is instead
named at save time (`Result.save_artifact`).
"""

from .for_each import render_path
from .references import iter_steps
from .tasks.task import task_command_info

RESULT_KEY = "result"
JSON_CONTENT_TYPE = "application/json"
_TIMESTAMPED_TRANSCRIPTION_COMMAND = "transcribe_audio"


def _literal_timestamps(arguments):
    """True when `arguments` sets a literal `timestamps` of "segment"/"word"
    - the one shape that switches `transcribe_audio`'s return value to a
    {text, chunks} dict. A `variable:`-supplied value is not literal here and
    is left to `Result.save_artifact` to name at run time.

    Imports `dw.tasks.audio_transcription` lazily - that module imports
    transformers at module scope (deliberately, per its own docstring, to
    keep the heavy import out of workflows that never transcribe), and this
    module is on the path of every `validate_workflow` call.
    """
    if not isinstance(arguments, dict):
        return False
    from .tasks.audio_transcription import TIMESTAMP_KINDS

    return arguments.get("timestamps") in TIMESTAMP_KINDS


def scalar_result_errors(workflow_definition, source_indices=None):
    """Every `result` block a command's `returns` kind cannot save, as
    [{path, message}].

    The definition handed here has already been substituted and expanded,
    so a step's `task.command` is literal. `source_indices`, when given, is
    the source step index of each step - a `for_each` group turns one
    written step into several, and the path an error carries has to be one
    the author can find in the file they wrote; the member is named in the
    message.
    """
    errors = []
    for _, step, result, source, where in iter_steps(
        workflow_definition.get("steps"), source_indices, RESULT_KEY
    ):
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
        returns = info.get("returns")
        timestamped = (
            command == _TIMESTAMPED_TRANSCRIPTION_COMMAND
            and _literal_timestamps(task.get("arguments"))
        )
        if returns == "json" or timestamped:
            if result.get("content_type") == JSON_CONTENT_TYPE:
                continue
            if timestamped:
                message = (
                    f"{command} with timestamps={task['arguments']['timestamps']!r} "
                    f"answers a {{text, chunks}} document, not plain text - "
                    f"'result' must set content_type '{JSON_CONTENT_TYPE}', not "
                    f"{result.get('content_type')!r}"
                )
            else:
                message = (
                    f"{command} answers a JSON document - 'result' must set "
                    f"content_type '{JSON_CONTENT_TYPE}', not "
                    f"{result.get('content_type')!r}"
                )
        elif returns == "scalar":
            message = (
                f"{command} returns a number, not an artifact - "
                f"'result' cannot be saved"
            )
        else:
            continue

        errors.append(
            {
                "path": render_path(("steps", source, RESULT_KEY)),
                "message": f"{message}{where}",
            }
        )
    return errors


__all__ = ["scalar_result_errors"]
