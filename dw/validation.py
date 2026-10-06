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

Callers use the functions at the end of this module directly
(`workflow_errors`, `workflow_context`, `run_warning_check`,
`undeclared_variable_errors`, `sub_workflow_errors`,
`sub_workflow_argument_warnings`); only `Workflow.validation_errors` and
`Workflow.validate` remain as methods. The three step-value checkers the error
registry runs (`fps_errors`, `null_media_errors`, `select_errors`) are in
dw/step_value_checks.py.
"""

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Callable

from . import references
from . import get_device_type, device_capacity_gb
from .adapter_compatibility import adapter_errors, adapter_warnings
from .argument_warnings import workflow_argument_warnings
from .content_types import content_type_errors
from .dissolve_frame_errors import dissolve_frame_errors
from .for_each import ForEachError, entry_field_warnings
from .hold_audio import hold_audio_errors
from .introspection import task_signature_errors
from .kernel_availability import kernel_availability_errors
from .library import SubWorkflowNotFound
from .locations import location_errors
from .media import probe_metadata
from .previous_results import previous_result_reference_errors
from .reference_limits import reference_limit_errors
from .reference_names import reference_name_errors
from .scalar_result_validation import scalar_result_errors
from .schema import load_schema, validate_data_all
from .security import InvalidInputError, SecurityError
from .shot_span_preflight import shot_span_warnings
from .slice_preflight import slice_past_end_warnings
from .subfolders import subfolder_errors
from .step_value_checks import (
    chain_prompts_errors,
    fps_errors,
    null_media_errors,
    select_errors,
)
from .task_domains import task_argument_errors
from .tasks.voice_attribution import voices_errors
from .tasks.script_check import lines_errors as script_lines_errors
from .type_references import component_name_errors, component_type_errors
from .variable_constraints import (
    ConstraintReferenceError,
    constraint_errors,
    constraint_reference_errors,
    constraint_warnings,
)
from .variables import (
    ConstantError,
    VariableCycleError,
    VariableNotFoundError,
    argument_errors,
    set_variables,
    undeclared_variable_references,
)
from .video_extensions import video_extension_errors
from .video_size_errors import video_size_errors
from .vram_estimate import vram_estimate_errors
from .vram_inheritance import inherited_vram_warnings

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


class ValidationContext:
    """What one validation request's checks read. Built once per request
    (`workflow_context`) and handed to the error pass and the
    warning pass alike; never stored on the Workflow.

    The expansion is lazy: `expand`, when given, is called the first time a
    check reads `expanded` or `source_indices`, so a context can be built
    ahead of validation_errors' gates without raising what those gates
    answer as a finding. A failed expansion is not remembered - the next
    read raises it again, inside whichever check made it."""

    def __init__(
        self,
        workflow,
        arguments,
        expanded=None,
        source_indices=None,
        base_dir=None,
        composing=(),
        # What the device the request is validated for can hold - read by
        # workflow_context from this module's own names, where the vram
        # tests patch them
        device_type=None,
        capacity_gb=None,
        # The catalog's VRAM ceilings, for inherited_vram_warnings
        ceiling_index=None,
        expand=None,
    ):
        self.workflow = workflow
        self.arguments = arguments
        self.base_dir = base_dir
        self.composing = tuple(composing or ())
        self.device_type = device_type
        self.capacity_gb = capacity_gb
        self.ceiling_index = ceiling_index
        self._expand = expand
        self._expanded = expanded
        self._source_indices = source_indices
        self._probes = {}

    def _expansion(self):
        if self._expanded is None and self._expand is not None:
            self._expanded, self._source_indices = self._expand()
        return self._expanded, self._source_indices

    @property
    def expanded(self):
        """The definition as the run will see it (substituted, expanded)."""
        return self._expansion()[0]

    @property
    def source_indices(self):
        """The written step index of every expanded step."""
        return self._expansion()[1]

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
    """The finding a check that raised becomes. It names the exception's
    type only - its text may carry internals (a path, a value) and the log
    has it with the traceback."""
    failure = (
        f"check '{check.name}' failed ({type(error).__name__}) - "
        f"the server log has the detail"
    )
    if severity == ERROR:
        message = failure
    else:
        message = f"{INTERNAL}: {severity} {failure}"
    return Finding(severity, INTERNAL, None, message)


def run_checks(context, checks, severity, *, loud=True):
    """Every check's findings, in registry order. A check that raises is
    logged with its traceback and becomes one `internal` finding of the
    pass's own severity; the checks after it still run.

    `loud=False` logs that traceback at DEBUG rather than ERROR - for a
    request already refused, where a check tripping over the arguments it
    was refused for is expected. The internal finding is the same."""
    findings = []
    for check in checks:
        try:
            items = check.run(context)
        except Exception as e:
            logger.log(
                logging.ERROR if loud else logging.DEBUG,
                "validation %s check '%s' failed",
                severity,
                check.name,
                exc_info=True,
            )
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
    # hold_audio on a pipeline with no H3 hold blocks, or holding something
    # that is not audio, costs a checkpoint load otherwise (dw/hold_audio.py)
    Check("hold_audio", lambda c: hold_audio_errors(c.expanded, c.source_indices)),
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
    # A check_script `lines` it would refuse (#609)
    Check("script_lines", lambda c: script_lines_errors(c.expanded, c.source_indices)),
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
    # not match its rule (select_errors, above)
    Check("select", lambda c: select_errors(c.expanded, c.source_indices)),
    # A chain 'prompts' that resolved to a bare string, which the run would
    # index a character per segment (#653)
    Check(
        "chain_prompts",
        lambda c: chain_prompts_errors(c.expanded, c.source_indices),
    ),
    Check("task_signatures", _task_errors),
    # A component_type/scheduler_type/config_type that does not exist, or is
    # outside the trusted ecosystem (dw/type_references.py, #345)
    Check(
        "component_types",
        lambda c: component_type_errors(c.expanded, c.source_indices),
    ),
    # A `configuration.components` entry its component_type does not
    # register (dw/type_references.py, #442)
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
        lambda c: sub_workflow_errors(
            c.workflow, c.expanded, c.source_indices, list(c.composing)
        ),
    ),
]


# --- Whether a run gets the step cache (was in dw/plan.py) ----------------
#
# unseeded_cache_warnings is a warning source; it and its two helpers live
# here so the registry need not import dw.plan, which reaches dw.workflow.


def unseeded_cache_warnings(definition, arguments=None):
    """Say once, where a caller is already looking, that an unseeded workflow
    gets no step cache at all.

    `cached_steps: 0` is indistinguishable from 'probed, nothing hit' out
    there, and the difference is the one that matters: without a `seed` the
    cache is off, so nothing is ever reused however many times the same
    workflow runs (#107).

    Silent for a workflow with no `pipeline`/`pipeline_reference`/`workflow`
    step: a task-only utility has no generative randomness a `seed` would
    pin down in the first place, and each of its steps is a pure function of
    its inputs - a repeat run is already free without one (#247)
    """
    if _is_seeded(definition, arguments) or not _has_seedable_step(definition):
        return []
    return [
        "This workflow sets no 'seed', so the step cache is disabled and "
        "'cached_steps' is 0 without being probed - every step regenerates "
        "on every run. Set a top-level 'seed': 'variable:seed' with a "
        "declared default in 'variables' to make a repeat run reuse what it "
        "already produced"
    ]


def _has_seedable_step(definition):
    """Whether any step could consume a seed: a pipeline (inline or
    referenced) or a sub-workflow, which may hold one in turn. A workflow
    built entirely of `task` steps has nothing a seed would affect."""
    for step in definition.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if "pipeline" in step or "pipeline_reference" in step or "workflow" in step:
            return True
    return False


def _is_seeded(definition, arguments):
    """Whether a run of this workflow has a seed before it draws one - read
    from the definition as written and the caller's arguments, since
    realization pins a seed of its own into the copy."""
    seed = definition.get("seed")
    if references.is_ref(references.VARIABLE, seed):
        name = references.ref_name(references.VARIABLE, seed)
        if name in (arguments or {}):
            return arguments[name] is not None
        return (definition.get("variables") or {}).get(name) is not None
    return seed is not None


# --- The warning registry ----------------------------------------------------
#
# Every warning source admit() reports that does not need the plan, in the
# order admit() listed them - the response's order. Each returns today's
# "path: message" strings. A source that raises is one internal warning and
# never refuses (B10); `run_warning_check` runs the same entry, so a direct
# call and a request agree.
#
# Four read the definition as written, with the caller's arguments over its
# defaults; the rest read the request's one expansion. `arguments` may be
# None (the document is checked) where admit() used to pass {} - every
# source reads the two alike (`arguments or {}`), except
# null_variable_argument_warnings, which is exactly the one that draws the
# line.


def _null_variable_argument_warnings(context):
    """A required task argument fed by `variable:name` whose value is null.
    validation_errors drops that case when the document is checked with no
    arguments (#364); here it is said, since a run left as-is would fail.
    Empty once arguments are given - it is a hard error then."""
    if context.arguments is not None:
        return []
    return [
        f"{entry['path']}: {entry['message']}"
        for entry in task_signature_errors(
            context.expanded, context.source_indices, context.definition
        )
        if "variable" in entry
    ]


WARNING_CHECKS = [
    Check(
        "workflow_argument_warnings",
        lambda c: workflow_argument_warnings(c.definition, c.arguments),
    ),
    # A value a declared constraint will round up - the silent half of #96:
    # the run changed the caller's frame count and only the server's log
    # said so
    Check(
        "constraint_warnings", lambda c: constraint_warnings(c.definition, c.arguments)
    ),
    Check(
        "entry_field_warnings",
        lambda c: entry_field_warnings(c.definition, c.arguments),
    ),
    # Why `plan.cached_steps` is 0 for a workflow with no seed - the cache is
    # off, not empty
    Check(
        "unseeded_cache_warnings",
        lambda c: unseeded_cache_warnings(c.definition, c.arguments),
    ),
    # An adapter whose file name says nothing about which checkpoint
    # partition it was trained for: valid, since the name of a future
    # checkpoint cannot be predicted, but nothing at run time would say it
    # loaded onto the wrong one (#155)
    Check(
        "adapter_warnings",
        lambda c: adapter_warnings(
            c.expanded, c.source_indices, written=c.definition, supplied=c.supplied
        ),
    ),
    Check("null_variable_argument_warnings", _null_variable_argument_warnings),
    # An argument a sub-workflow step passes to a workflow that declares no
    # variable for it - dropped in silence at run time (#89)
    Check(
        "sub_workflow_warnings",
        lambda c: sub_workflow_argument_warnings(
            c.workflow, c.expanded, c.source_indices
        ),
    ),
    # A slice_audio source whose real duration is already knowable and whose
    # requested slice reaches past it - zero-padded rather than refused
    # (#402)
    Check(
        "slice_past_end_warnings",
        lambda c: slice_past_end_warnings(
            c.expanded, c.source_indices, c.base_dir, probe=c.probe
        ),
    ),
    # An assessment probe's shots argument reaching past a statically-
    # knowable video's real frame count - silently clipped (#425)
    Check(
        "shot_span_warnings",
        lambda c: shot_span_warnings(
            c.expanded, c.source_indices, c.base_dir, probe=c.probe
        ),
    ),
    # A step loading a pipeline the catalog declares a VRAM ceiling for, in a
    # workflow that declares none, projected past it - a warning, since this
    # workflow's offload/quantization may be leaner than the template's
    # (#502)
    Check(
        "inherited_vram_warnings",
        # No index, nothing to inherit - and no expansion read for it
        lambda c: (
            []
            if not c.ceiling_index
            else inherited_vram_warnings(
                c.expanded,
                c.ceiling_index,
                c.arguments,
                supplied=c.supplied,
                device_type=c.device_type,
                capacity_gb=c.capacity_gb,
                source_indices=c.source_indices,
                written=c.definition,
            )
        ),
    ),
]


def warning_check(name):
    """The warning registry's entry called `name`, looked up at call time so
    a direct call runs whatever the registry currently holds."""
    check = next((check for check in WARNING_CHECKS if check.name == name), None)
    if check is None:
        raise KeyError(name)
    return check


# --- Entry points over one Workflow -------------------------------------------
#
# `workflow` is a Workflow handed in as a value: this module never imports
# dw.workflow. A composed child is opened through `Workflow.open_sub_workflow`.


def workflow_context(workflow, arguments=None, composing=(), ceiling_index=None):
    """One validation request's ValidationContext: the expansion over
    `arguments` (lazy and memoized, so the error pass and the warning
    pass share it), where relative paths resolve from, the device the
    request is checked against and the catalog's VRAM ceilings
    (`ceiling_index`, for inherited_vram_warnings). Built per request
    and never stored on the Workflow, so its probe cache cannot serve a
    replaced file stale.

    Building one expands nothing: the expansion runs the first time a
    check reads it, so a context can be built ahead of
    workflow_errors' gates, which answer an expansion failure as a
    finding.
    """

    def expand():
        source_indices = []
        expanded = workflow.expanded_definition(arguments, source_indices)
        return expanded, source_indices

    return ValidationContext(
        workflow=workflow,
        arguments=arguments,
        base_dir=(
            os.path.dirname(os.path.abspath(workflow.file_spec))
            if workflow.file_spec
            else None
        ),
        composing=composing,
        device_type=get_device_type(),
        capacity_gb=device_capacity_gb(),
        ceiling_index=ceiling_index,
        expand=expand,
    )


def workflow_errors(workflow, arguments=None, composing=None, context=None):
    """Every schema violation in the definition, as [{path, message}];
    empty when it validates. `arguments` are the caller's, so a
    for_each over a list the caller supplies is checked as it will run.

    `composing` carries the chain of sub-workflows above this one, so a
    workflow that composes itself is an error rather than a recursion.

    `context`, when given, is the request's own ValidationContext - its
    arguments and composing chain are the ones checked - so one request
    expands and probes once. Otherwise one is built here. Passing a
    context together with different `arguments` or `composing` is a
    caller bug and raises ValueError.

    Past the gates (schema, 'constraint:' references, the expansion)
    every check in `ERROR_CHECKS` runs, in order; one that raises is an
    internal error rather than a lost verdict (B10).
    """
    if context is None:
        context = workflow_context(workflow, arguments, composing)
    elif (arguments is not None and arguments != context.arguments) or (
        composing is not None and tuple(composing) != context.composing
    ):
        raise ValueError(
            "validation_errors was given a context and different "
            "arguments or composing - pass one or the other"
        )
    errors = validate_data_all(workflow.workflow_definition, load_schema("workflow"))
    # Only once the shape is known good: the checks walk the steps
    # array and a definition that fails the schema may have no such
    # array to walk
    if errors:
        return errors
    # A 'constraint:' frame_snap naming nothing declared, every one of
    # them at the path it sits at, before expanding. One that only
    # becomes a 'constraint:' name once a variable substitutes is
    # raised by expansion as ConstraintReferenceError, answered below
    errors = constraint_reference_errors(workflow.workflow_definition)
    if errors:
        return errors
    try:
        # The context's expansion is lazy; run it here, where its
        # failures are findings rather than an internal error in
        # whichever check read it first
        context.expanded
    except ForEachError as e:
        return [{"path": e.path, "message": str(e)}]
    except ConstantError as e:
        return [{"path": e.path, "message": str(e)}]
    except ConstraintReferenceError as e:
        return [{"path": e.path, "message": e.message}]
    except VariableNotFoundError:
        # Every undeclared reference, not just the first one
        # substitution tripped over - and reported where each sits
        # rather than as a for_each whose list arrived unsubstituted
        return undeclared_variable_errors(workflow, context.arguments)
    except VariableCycleError as e:
        # A variable that references itself, directly or through
        # others - there is no single path inside the definition to
        # blame, so it is reported against 'variables' as a whole
        return [{"path": "variables", "message": str(e)}]
    return to_errors(run_checks(context, ERROR_CHECKS, ERROR))


def run_warning_check(workflow, name, arguments, **context_fields):
    """The registry's warning check `name` over a context of this
    call's own - how one check answers when called directly rather
    than through admit(). An expansion that fails
    raises inside the check, which makes it one internal warning."""
    context = workflow_context(workflow, arguments, **context_fields)
    check = warning_check(name)
    return to_warnings(run_checks(context, [check], WARNING))


def undeclared_variable_errors(workflow, arguments=None):
    """Every 'variable:' reference naming nothing the workflow declares.

    Fatal rather than a warning: once a workflow has a 'variables'
    block, replace_variables refuses an undeclared reference, so this is
    a run that cannot start. Good caller `arguments` are folded in first,
    and a reference inside one of them is reported under `arguments.`,
    where the caller wrote it.
    """
    definition = copy.deepcopy(workflow.workflow_definition)
    variables = definition.get("variables")
    supplied = set()
    if isinstance(variables, dict) and arguments:
        if not argument_errors(definition, arguments):
            set_variables(arguments, variables)
            supplied = set(arguments)
    declared = sorted(variables or {})

    def where(path):
        head, _, rest = path.partition(".")
        if head == "variables":
            name = rest.split(".", 1)[0].split("[", 1)[0]
            if name in supplied:
                return "arguments." + rest
        return path

    return [
        {
            "path": where(path),
            "message": (
                f"'{references.make_ref(references.VARIABLE, name)}' names no "
                f"declared variable; declared: {', '.join(declared) or '<none>'}"
            ),
        }
        for path, name in undeclared_variable_references(definition)
    ]


def sub_workflow_errors(workflow, expanded, source_indices=None, composing=None):
    """Every sub-workflow step whose `path` names nothing this server can
    reach, composes a workflow already on the chain, or resolves to a
    workflow that does not itself validate.

    `composing` is the resolved path of every workflow above this one,
    which is what makes a cycle an error here rather than a recursion
    the run discovers.
    """
    errors = []
    composing = list(composing or [])
    for index, step in enumerate(expanded.get("steps", []) or []):
        reference = step.get("workflow")
        if not isinstance(reference, dict) or not isinstance(
            reference.get("path"), str
        ):
            continue
        source = references.author_index(source_indices, index)
        where = f"steps[{source}].workflow.path"
        path = reference["path"]
        try:
            resolved, root = workflow.resolve_sub_workflow_path(path)
        except (SubWorkflowNotFound, SecurityError, InvalidInputError) as e:
            errors.append({"path": where, "message": str(e)})
            continue
        if resolved in composing:
            errors.append(
                {
                    "path": where,
                    "message": (
                        f"Sub-workflow '{path}' composes a workflow that "
                        "is already composing it - a cycle: "
                        + " -> ".join(composing + [resolved])
                    ),
                }
            )
            continue
        try:
            child, _ = workflow.open_sub_workflow(path, (resolved, root))
        except Exception as e:
            errors.append({"path": where, "message": f"Sub-workflow '{path}': {e}"})
            continue
        for error in child.validation_errors(composing=composing + [resolved]):
            errors.append(
                {
                    "path": where
                    if error["path"] is None
                    else f"{where} -> {error['path']}",
                    "message": f"Sub-workflow '{path}': {error['message']}",
                }
            )
    return errors


def sub_workflow_argument_warnings(workflow, expanded, source_indices=None):
    """sub_workflow_warnings over an expansion already made - what the
    registry check calls with the request's own."""
    warnings = []
    for index, step in enumerate(expanded.get("steps", []) or []):
        reference = step.get("workflow")
        if not isinstance(reference, dict):
            continue
        passed = reference.get("arguments")
        if not isinstance(passed, dict) or not isinstance(reference.get("path"), str):
            continue
        try:
            child, _ = workflow.open_sub_workflow(reference["path"])
        except Exception:
            # An unresolvable path is an error, reported by
            # sub_workflow_errors - not a second complaint here
            continue
        declared = child.workflow_definition.get("variables") or {}
        source = references.author_index(source_indices, index)
        for name in sorted(set(passed) - set(declared)):
            warnings.append(
                f"steps[{source}].workflow.arguments.{name}: "
                f"'{reference['path']}' declares no variable '{name}' - the "
                "value is dropped. Declared: "
                + (", ".join(sorted(declared)) or "<none>")
            )
    return warnings
