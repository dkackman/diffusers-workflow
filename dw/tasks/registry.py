"""The task command registry: `register_command` and the two tables it fills.

Split from `dw/tasks/task.py` so a task module can register its own handler
(`dw/tasks/beats.py`) without growing that file or importing it, which would
be a cycle - task.py imports this module and every self-registering task
module, and re-exports the tables under their old names.
"""

import logging
from typing import Callable, Dict

logger = logging.getLogger("dw")


# Command registry: maps command names to handler functions
_COMMAND_REGISTRY: Dict[str, Callable] = {}

# What each command's arguments actually are. The handlers forward
# **arguments into an implementation function, so that function's signature
# is the command's argument schema - registering its dotted path here (a
# string, to preserve the lazy-import discipline) lets the introspection
# layer read the same signature the runtime calls. 'provided' names the
# parameters the dispatch supplies itself, which are not workflow arguments.
_COMMAND_INFO: Dict[str, dict] = {}


def register_command(
    command_name: str,
    implementation=None,
    provided=(),
    consumes_device=False,
    returns="artifact",
    summary=None,
    parameter_descriptions=None,
    assessment=False,
):
    """
    Decorator to register a command handler function.

    Args:
        command_name: The command name to register
        implementation: Dotted path to the function whose signature defines
            the command's arguments (None for a command that consumes a
            free-form dict)
        provided: Parameter names the dispatch supplies itself
        consumes_device: True for a handler that calls task.device_for(arguments)
            itself, to pick the accelerator a model-backed task runs on. False
            (the default) is for a task that runs no model and forwards
            arguments straight to its implementation - device is introspected
            as a universal, always-safe-to-pass argument (dw/introspection.py),
            so a non-consuming handler must drop it itself rather than let it
            reach the implementation as an unexpected keyword argument (#185).
            Some commands (e.g. gather_inputs) receive a non-dict argument
            value, which never carries a device to drop
        returns: "artifact" (the default - an image/video/audio/frames object
            `Result.save` knows how to write) or "scalar" for a command whose
            return value is a bare number with no file to save (`judge`). A
            `result` block on a "scalar" command is refused in
            `validation_errors` (dw/scalar_result_validation.py, #212) rather
            than reaching `save_artifact` at run time, where a float has
            nothing left identifying which command produced it
            - or "json" for a command answering a JSON-safe dict (the
            assessment probes, `attribute_voices`): its `result` may only be
            `application/json`, since any other content type would explode
            the dict key by key into files
        summary: Overrides the command's `get_task` summary, which otherwise
            reads the implementation function's docstring. For a command
            whose handler dispatches its implementation per video frame
            (`_per_frame`), that docstring describes the single-frame
            function rather than the command a caller invokes - same reason
            `_VIDEO_PROCESSOR_INFO` overrides `get_first_frame`/
            `get_last_frame` (#366, #383)
        parameter_descriptions: Overrides one or more of the implementation's
            per-parameter `get_task` descriptions by name, for the same
            single-frame-vs-command reason as `summary`
        assessment: True for an assessment probe (`dw/tasks/assess.py`) - a
            command that measures a finished cut and says where to look,
            listed in `list_tasks`' `assessment`. Declared, not inferred from
            `returns="json"`: `attribute_voices` answers JSON too and is an
            analysis of a song, not a check of a cut (#485)

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        if consumes_device:
            handler = func
        else:

            def handler(task, arguments, previous_pipelines, _func=func):
                if isinstance(arguments, dict):
                    arguments.pop("device", None)
                return _func(task, arguments, previous_pipelines)

        _COMMAND_REGISTRY[command_name] = handler
        info = {
            "kind": "command",
            "implementation": implementation,
            "provided": tuple(provided),
            "returns": returns,
        }
        if assessment:
            info["assessment"] = True
        if summary:
            info["summary"] = summary
        if parameter_descriptions:
            info["parameter_descriptions"] = dict(parameter_descriptions)
        _COMMAND_INFO[command_name] = info
        logger.debug(f"Registered command handler: {command_name}")
        return func

    return decorator
