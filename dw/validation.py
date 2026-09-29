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
from .adapter_compatibility import adapter_errors
from .content_types import content_type_errors
from .dissolve_frame_errors import dissolve_frame_errors
from .for_each import MEMBER_SEPARATOR, render_path
from .introspection import (
    component_name_errors,
    component_type_errors,
    task_signature_errors,
)
from .kernel_availability import kernel_availability_errors
from .locations import location_errors
from .media_info import probe_metadata
from .previous_results import previous_result_reference_errors
from .reference_limits import reference_limit_errors
from .reference_names import reference_name_errors
from .references import author_index
from .scalar_result_validation import scalar_result_errors
from .select_validation import select_errors
from .subfolders import subfolder_errors
from .task_domains import task_argument_errors
from .tasks.voice_attribution import voices_errors
from .variable_constraints import constraint_errors
from .video_extensions import video_extension_errors
from .video_size_errors import video_size_errors
from .vram_estimate import vram_estimate_errors

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
    # What the device the request is validated for can hold - read by the
    # factory from dw.workflow's own names, where the vram tests patch them
    device_type: str | None = None
    capacity_gb: float | None = None
    _probes: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def definition(self):
        """The definition as written - before substitution and expansion."""
        return self.workflow.workflow_definition

    @property
    def supplied(self):
        """The names of the variables the caller supplied."""
        return set(self.arguments or {})

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


# --- The error registry ------------------------------------------------------
#
# Every check validation_errors runs once its gates (schema, 'constraint:'
# references, the expansion) have passed, in the order the response lists
# what they find - the editor shows the first error first, so a reorder is a
# surface change. Each entry is today's call, reading what it needs from the
# context.


def _task_errors(context):
    """A required task argument left unset validated as `valid: true` and
    then failed the job on Python's own signature error (dw/introspection.py,
    #141). When the step supplies it by `variable:name` and only the
    variable's value is null, the error carries a `variable` key (#364).

    With no arguments - save_workflow, or validate_workflow checking the
    document rather than a specific run - that case is dropped here, since
    the variable just hasn't been given a value yet; it is reported as a
    warning instead (null_variable_argument_warnings), since a run left
    as-is would still fail. Anything else task_signature_errors reports - a
    genuinely missing or unknown argument - stays a hard error regardless.
    """
    errors = task_signature_errors(
        context.expanded, context.source_indices, context.definition
    )
    if context.arguments is None:
        errors = [e for e in errors if "variable" not in e]
    return errors


ERROR_CHECKS = [
    Check(
        "previous_result_references",
        lambda c: previous_result_reference_errors(c.expanded, c.source_indices),
    ),
    Check("subfolders", lambda c: subfolder_errors(c.expanded, c.source_indices)),
    Check("fps", lambda c: fps_errors(c.expanded, c.source_indices)),
    # A reference name no workspace could ever resolve - the '@' a for_each
    # member's own file carries, rejected after the queue by a message that
    # named a valid form and not the objection (dw/reference_names.py, #162)
    Check(
        "reference_names",
        lambda c: reference_name_errors(c.expanded, c.source_indices),
    ),
    # A still image handed to a 'video' argument by path or asset:/output:
    # reference validated clean and then died inside fetch_video's extension
    # gate in the first seconds of the run (dw/video_extensions.py, #347)
    Check(
        "video_extensions",
        lambda c: video_extension_errors(c.expanded, c.source_indices),
    ),
    # A result content_type no writer will accept - a bare word like "video"
    # validated clean and then died inside the writer (dw/content_types.py,
    # #168)
    Check("content_types", lambda c: content_type_errors(c.expanded, c.source_indices)),
    # A 'result' block on a step whose command returns a scalar, not an
    # artifact (dw/scalar_result_validation.py, #212)
    Check(
        "scalar_results", lambda c: scalar_result_errors(c.expanded, c.source_indices)
    ),
    # A location policy refuses before a model load is spent on the run
    # rather than after it (dw/locations.py)
    Check(
        "locations",
        lambda c: location_errors(c.expanded, c.source_indices, c.base_dir),
    ),
    # A reference set the pipeline would refuse costs a checkpoint load to
    # find out about otherwise (dw/reference_limits.py, #136)
    Check(
        "reference_limits",
        lambda c: reference_limit_errors(c.expanded, c.source_indices),
    ),
    # A step a declared vram_estimate projects past the card 'cost' was
    # measured on - per step, after expansion, so a for_each member is
    # projected with its own frames and references (dw/vram_estimate.py,
    # #265, #479)
    Check(
        "vram_estimate",
        lambda c: vram_estimate_errors(
            c.expanded,
            c.arguments,
            supplied=c.supplied,
            device_type=c.device_type,
            capacity_gb=c.capacity_gb,
            source_indices=c.source_indices,
            written=c.definition,
        ),
    ),
    # A bare object description whose media resolved null - realize_args
    # refuses this at run time, a few seconds into the job (#478)
    Check("null_media", lambda c: null_media_errors(c.expanded, c.source_indices)),
    # An adapter trained for the other checkpoint partition, which the
    # pipeline loads without complaint and answers worse for
    # (dw/adapter_compatibility.py, #155)
    Check(
        "adapters",
        lambda c: adapter_errors(
            c.expanded,
            c.source_indices,
            written=c.definition,
            supplied=c.supplied,
        ),
    ),
    # A number outside a task argument's declared domain (dw/task_domains.py,
    # #139, #140)
    Check(
        "task_argument_domains",
        lambda c: task_argument_errors(c.expanded, c.source_indices),
    ),
    # An attribute_voices `voices` it would refuse (#494)
    Check("voices", lambda c: voices_errors(c.expanded, c.source_indices)),
    # A dissolve_videos overlap wider than a statically-resolvable input's
    # real frame count (dw/dissolve_frame_errors.py, #400)
    Check(
        "dissolve_frames",
        lambda c: dissolve_frame_errors(
            c.expanded, c.source_indices, c.base_dir, probe=c.probe
        ),
    ),
    # A dissolve_videos/concat_videos size mismatch (dw/video_size_errors.py,
    # #504)
    Check(
        "video_sizes",
        lambda c: video_size_errors(
            c.expanded, c.source_indices, c.base_dir, probe=c.probe
        ),
    ),
    # A select step whose rule is misspelled, or whose threshold/index does
    # not match its rule (dw/select_validation.py)
    Check("select", lambda c: select_errors(c.expanded, c.source_indices)),
    Check("task_signatures", _task_errors),
    # A component_type/scheduler_type/config_type that does not exist, or is
    # outside the trusted ecosystem (dw/introspection.py, #345)
    Check(
        "component_types",
        lambda c: component_type_errors(c.expanded, c.source_indices),
    ),
    # A `configuration.components` entry its component_type does not
    # register (dw/introspection.py, #442)
    Check(
        "component_names",
        lambda c: component_name_errors(c.expanded, c.source_indices),
    ),
    # A value outside a rule the workflow declares (dw/variable_constraints.py,
    # #96)
    Check(
        "constraints",
        lambda c: constraint_errors(c.definition, c.arguments, supplied=c.supplied),
    ),
    # An 'attn_processor_type' whose Hub kernel this machine has no build
    # variant for (dw/kernel_availability.py, #178)
    Check(
        "kernel_availability",
        lambda c: kernel_availability_errors(c.expanded, c.source_indices),
    ),
    # A sub-workflow step naming nothing reachable, a composition cycle, or a
    # child that does not itself validate
    Check(
        "sub_workflows",
        lambda c: c.workflow.sub_workflow_errors(
            c.expanded, c.source_indices, list(c.composing)
        ),
    ),
]
