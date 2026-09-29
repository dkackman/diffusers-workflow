"""Validation's plumbing: one finding type, one context per request, and one
exception policy for every check.

A check is a pure function of a `ValidationContext` returning today's legacy
shapes - `[{path, message, ...}]` for an error pass, `["path: message", ...]`
for a warning source - and `run_checks` turns what each returns into
`Finding`s. A check that raises becomes one finding of kind `internal` and
every other check still runs (B10): an error pass that loses a check can no
longer admit a job the check never looked at, and a warning source that
raises is said out loud rather than dropped. `to_errors` / `to_warnings` are
the one place a finding is serialized back to the response shapes callers
already read.

`ValidationContext` carries what every check reads - the expanded definition
and the workflow it came from, taken as a value: this module never imports
dw.workflow - and memoizes the media metadata probe for the one validation
it was built for (B9). It is never stored on the Workflow, so a replaced
asset is probed afresh by the next request.

Two checkers whose only importer was dw/workflow.py live here as well:
`fps_errors` (was dw/result_fps.py) and `null_media_errors` (was
dw/null_media.py).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable

from . import references
from .arguments import (
    FROM_ARGUMENTS_KEY,
    FROM_FILE_KEY,
    FROM_PREVIOUS_RESULT_KEY,
    _names_no_media,
)
from .for_each import MEMBER_SEPARATOR, render_path
from .media_info import probe_metadata
from .references import author_index

logger = logging.getLogger("dw")

ERROR = "error"
WARNING = "warning"
INTERNAL = "internal"

# "path: message" - how a warning source spells a finding today
_WARNING_SEPARATOR = ": "


@dataclass(frozen=True)
class Finding:
    """One thing a check found. `kind` is the check's registry name, or
    `internal` when the check itself failed; `extra` carries any key a
    legacy error dict held beyond path and message (`variable`, #364)."""

    severity: str
    kind: str
    path: str | None
    message: str
    extra: dict = field(default_factory=dict)


@dataclass
class ValidationContext:
    """What one validation request's checks read. Built once per request
    (`Workflow.validation_context`) and handed to the error pass and the
    warning pass alike; never stored on the Workflow."""

    workflow: object
    arguments: dict | None
    expanded: dict
    source_indices: list
    base_dir: str | None
    composing: tuple = ()
    _probes: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def probe(self, path):
        """`probe_metadata(path)`, memoized by real path for this context
        only - two checks reading one file decode its header once."""
        key = os.path.realpath(path)
        if key not in self._probes:
            self._probes[key] = probe_metadata(path)
        return self._probes[key]


@dataclass(frozen=True)
class Check:
    """A registry entry: `run(context)` returns legacy error dicts or
    warning strings."""

    name: str
    run: Callable[[ValidationContext], list]


def _finding(item, kind, severity):
    if isinstance(item, dict):
        extra = {k: v for k, v in item.items() if k not in ("path", "message")}
        return Finding(severity, kind, item.get("path"), item.get("message"), extra)
    text = str(item)
    path, separator, message = text.partition(_WARNING_SEPARATOR)
    if not separator:
        return Finding(severity, kind, None, text)
    return Finding(severity, kind, path, message)


def _internal(check, error, severity):
    failure = f"{type(error).__name__}: {error}"
    if severity == ERROR:
        message = (
            f"check '{check.name}' failed ({failure}) - "
            f"the server log has the traceback"
        )
    else:
        message = f"{INTERNAL}: {severity} check '{check.name}' failed ({failure})"
    return Finding(severity, INTERNAL, None, message)


def run_checks(context, checks, severity):
    """Every check's findings, in registry order. A check that raises is
    logged with its traceback and becomes one `internal` finding of the
    pass's own severity; the checks after it still run."""
    findings = []
    for check in checks:
        try:
            items = check.run(context)
        except Exception as e:
            logger.exception("validation %s check '%s' failed", severity, check.name)
            findings.append(_internal(check, e, severity))
            continue
        findings.extend(_finding(item, check.name, severity) for item in items or [])
    return findings


def to_errors(findings):
    """Today's error shape: `{path, message}` plus whatever extra keys the
    finding carried, in that order."""
    return [
        {"path": f.path, "message": f.message, **f.extra}
        for f in findings
        if f.severity == ERROR
    ]


def to_warnings(findings):
    """Today's warning shape: `"path: message"`, or the bare message when
    the finding has no path."""
    return [
        f.message if f.path is None else f"{f.path}{_WARNING_SEPARATOR}{f.message}"
        for f in findings
        if f.severity == WARNING
    ]


# --- A step's result 'fps' (was dw/result_fps.py) ---------------------------
#
# Widened the same way `subfolder` was (dw/subfolders.py): the schema lets the
# raw document hold a `variable:` or `item:` reference so a workflow can keep
# one frame-rate variable rather than a literal duplicated between the step
# that generates and the step that writes (#363). This owns the resolved-value
# check: once substitution has run, `result.fps` has to be a real, positive
# frame rate, and it must be a whole number - `encode_video` (dw/result.py,
# the writer used whenever a step's result carries audio) hands the value
# straight to PyAV as `rate=int(fps)`, so a fractional rate like 23.976 would
# be silently floored to 23 rather than written as asked. A frame_rate
# variable declared as a float (24.0) still passes, since it carries no
# fractional part; only a genuine fraction is refused.
#
# Containment doesn't apply here - there's no path to join - so unlike
# subfolder_errors this only ever checks value shape.

FPS_KEY = "fps"

# Reference prefixes substitution resolves before this pass runs. One still
# spelled out here is one nothing resolved, and that is the undeclared-
# variable pass's complaint rather than a shape error
_UNRESOLVED_PREFIXES = references.SUBSTITUTED


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

        source = references.author_index(source_indices, index)
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


# --- An object description whose media resolved null (was dw/null_media.py) --
#
# `realize_args` builds a pipeline argument that names a type and where its
# media comes from - MiniMax-H3's references are the catalog's example. When
# that media source resolves to null (typically a `variable:` left unset), the
# object is `OMITTED`: dropped silently when it sits in a list, since that is
# how a template makes a reference optional (`dw/arguments.py`'s list branch).
# But the *same* dict sitting directly under a key, not in a list, has nothing
# to leave it out of - `realize_args` raises there, at run time, after the
# checkpoint is already loaded (#478).
#
# This mirrors that raise, not the silent drop: only a bare object description
# in "dict context" is an error here, exactly the shapes `realize_args`' own
# dict branch would refuse. One found inside a list is left alone, since a
# template author relies on that being silently dropped rather than reported.

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
        source = author_index(source_indices, index)
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
