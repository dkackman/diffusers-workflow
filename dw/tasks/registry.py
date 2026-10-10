"""The task command registry: `register_command` and the tables it fills.

Split from `dw/tasks/task.py` so a task module can register its own handler
(`dw/tasks/beats.py`) without growing that file or importing it, which would
be a cycle - task.py imports this module and every self-registering task
module, and re-exports the tables under their old names.

A command's validate-time rules - its argument domains and choices, its
cross-argument check, the arguments that name a file to read - are declared
on its registration too (#692). `TASK_ARGUMENT_DOMAINS`,
`TASK_ARGUMENT_CHOICES` and `TASK_MEDIA_ARGUMENTS` are `RegistryTable` views
of them, so a command cannot be registered and miss a table: before, each was
a hand-kept dict, and a command left out of one lost that check silently.
"""

import importlib
import logging
from collections.abc import Mapping
from types import MappingProxyType
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

# command -> the validate-time rules its registration declared: any of
# 'domains', 'choices', 'static_check', 'media_arguments' and
# 'whole_numbers'. Kept apart from
# _COMMAND_INFO, whose entries describe the signature and stay plain data
_COMMAND_RULES: Dict[str, dict] = {}

# The module whose import registers every command - itself, and each
# self-registering task module through its own import lines
_REGISTERING_MODULE = "dw.tasks.task"


class RegistryTable(Mapping):
    """A read-only {command: value} view of one rule the registrations declare.

    Built on each read rather than once at import: a table read before a
    self-registering module was imported would otherwise come out without
    its commands. Reading imports `dw.tasks.task` first - the module every
    registration is reached through, and the path validation takes - so the
    view is never partial. A value that is a dict is handed out read-only.
    """

    def __init__(self, rule):
        self._rule = rule

    def _table(self):
        importlib.import_module(_REGISTERING_MODULE)
        return {
            command: (
                MappingProxyType(rules[self._rule])
                if isinstance(rules[self._rule], dict)
                else rules[self._rule]
            )
            for command, rules in _COMMAND_RULES.items()
            if self._rule in rules
        }

    def __getitem__(self, command):
        return self._table()[command]

    def __iter__(self):
        return iter(self._table())

    def __len__(self):
        return len(self._table())

    def __repr__(self):
        return f"RegistryTable({self._rule!r}, {self._table()!r})"


def register_command(
    command_name: str,
    implementation=None,
    provided=(),
    consumes_device=False,
    returns="artifact",
    summary=None,
    parameter_descriptions=None,
    assessment=False,
    domains=None,
    choices=None,
    static_check=None,
    media_arguments=(),
    whole_numbers=(),
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
            (`image_ops.per_frame`), that docstring describes the single-frame
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
        domains: {argument: domain} - the numeric domain of each argument
            whose range is not a judgement call (`dw/task_domains.py`'s
            POSITIVE, NON_NEGATIVE, ...), refused at validation and read by
            the command's own run-time `check_arguments`
        choices: {argument: (values...)} - the literal values an argument
            accepts, refused at validation and listed by `get_task`
        static_check: `arguments -> [(argument, message)]` - the command's
            cross-argument rules a literal workflow can break before it runs
            (`task_problems.cuts_errors`), called by `task_argument_errors`
            with the step's expanded arguments
        media_arguments: Names of arguments that name a file to read but not
            by the media-key convention (`dw/locations.py`) - confined like
            a media key at validation (#630)
        whole_numbers: Names of the arguments with a domain that the command
            reads as whole numbers (`task_domains.whole_number`): 3, 3.0,
            "3" and "3.0" are 3, and 3.5 is refused - at validation as at
            run time. Every other argument with a domain is a real number

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
        rules = {}
        if domains:
            rules["domains"] = dict(domains)
        if choices:
            rules["choices"] = {name: tuple(values) for name, values in choices.items()}
        if static_check is not None:
            rules["static_check"] = static_check
        if media_arguments:
            rules["media_arguments"] = tuple(media_arguments)
        if whole_numbers:
            rules["whole_numbers"] = tuple(whole_numbers)
        _COMMAND_RULES[command_name] = rules
        logger.debug(f"Registered command handler: {command_name}")
        return func

    return decorator
