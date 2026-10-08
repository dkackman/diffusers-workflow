"""One admission service for every route that takes a job request.

`POST /api/validate`, `POST /api/jobs`, `POST /api/jobs/{id}/rerun` and
`POST /api/enhance` each admit a request through `admit()`: the workflow is
loaded once, checked once - the schema and everything validation_errors
derives from the expansion, the caller's arguments, the references they
make - and warned about once, with the workspace's asset library active for
all of it. JobManager.submit records and queues what was admitted - the
admitted Workflow's definition and file_spec travel with the job - and it
does not re-check. The worker builds that snapshot (workflow_from_snapshot)
and runs it without re-reading the file or validating it again.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from ..assets import (
    activate_asset_dir,
    deactivate_asset_dir,
    is_asset_reference,
    resolve_asset_reference,
)
from .. import get_device_type, references, validation
from ..plan import build_plan
from ..prompts import resolve_prompt_reference
from ..runs import (
    activate_output_root,
    deactivate_output_root,
    is_output_reference,
    resolve_output_reference,
)
from ..validation import WARNING, run_checks, to_warnings
from ..variables import argument_errors
from ..vram_estimate import required_vram_gb
from ..workflow import Workflow, workflow_from_definition, workflow_from_file
from .deps import ceiling_index, server_prompt_library
from .job_record import ACK_BOOLEAN, ACK_BOUND, ACK_NONE
from .outputs import resolution_library

logger = logging.getLogger("dw")


class ValidatorFailure(Exception):
    """The validator failing outright rather than answering: building the
    request's validation context, the gates validation_errors runs before
    its checks (schema, 'constraint:' references, the expansion) or the
    argument checks raised something they do not answer as a finding. A
    single check that raises is not this - it is an internal finding, and
    the verdict is invalid (B10). The validate route answers it as an
    invalid workflow whose detail is in the server log; a submit answers
    400 with the message, which names the failure's exception type only -
    its text can carry internals (a path, a value), so the log keeps it."""


def _validator_failure(error):
    """The ValidatorFailure for `error`, raised from inside its except
    block: logged with its traceback, answered by its type alone (as a
    check that raises is, B10)."""
    logger.exception("Validation failed outright")
    return ValidatorFailure(
        f"validation failed ({type(error).__name__}) - the server log has the detail"
    )


@dataclass
class Admission:
    workflow: Workflow  # the one instance this request builds
    arguments: dict  # as the caller sent them ({} when omitted)
    supplied: bool  # the caller sent `arguments` at all
    errors: list = field(default_factory=list)  # [{path, message}]
    warnings: list = field(default_factory=list)  # every /api/validate warning
    plan: Optional[dict] = None  # build_plan's answer, when plan_for was given
    # Whether `errors` are the definition's own (schema and expansion)
    # rather than the arguments' - validate names the checked arguments only
    # for the latter
    schema_errors: bool = False
    # What a card must hold to run it, (gb, hard) or None - the worker pool
    # dispatches on it (dw/vram_estimate.py required_vram_gb, #462)
    vram_need: Optional[tuple] = None

    @property
    def ok(self):
        return not self.errors

    def message(self):
        """The errors as one line, the form a 400 has always carried."""
        return "; ".join(
            problem["message"]
            if problem.get("path") is None
            else f"{problem['path']}: {problem['message']}"
            for problem in self.errors
        )


def admit(
    *,
    workflow_path,
    workflow,
    arguments,
    base_dir,
    workspace,
    ceiling_index,
    output_dir,
    workflow_dir,
    asset_library,
    prompt_library,
    supplied=True,
    plan_for=None,
):
    """Load the request's workflow once and check it once, with the
    workspace's asset library active for every check (the validate route
    used to leave it off for its warnings). `plan_for`, a callable taking
    the Workflow and the admission's `vram_need` (the need dispatch routes
    by), is called inside the same scope when the caller needs the
    plan (validate always; submit only for a bound acknowledgement) and only
    when the request is admissible.

    `asset_library` and `prompt_library` are the `LibraryPath`s the workspace's
    'asset:' and 'prompt:' references resolve over; `ceiling_index` is the
    catalog's VRAM ceilings for inherited_vram_warnings.

    One ValidationContext serves the whole request: the error pass and the
    warning pass (`validation.WARNING_CHECKS`) read one expansion and one
    media probe cache. A check that raises is an internal finding - an
    invalid verdict for an error check, one `internal:` warning for a
    warning source, which never refuses (B10).

    Raises what constructing the Workflow raises (ValueError, SecurityError,
    ...) and ValidatorFailure when building the context, validation_errors'
    gates or the argument checks themselves fail.
    """
    if (workflow_path is None) == (workflow is None):
        raise ValueError("Provide exactly one of workflow_path or workflow")
    arguments = arguments if arguments is not None else {}
    if workflow_path is not None:
        candidate = workflow_from_file(workflow_path, output_dir, workflow_dir)
    else:
        candidate = workflow_from_definition(
            copy.deepcopy(workflow), output_dir, base_dir, workflow_dir
        )
    # Omitted means "check the document", explicit means "check a run with
    # these arguments" (#364) - the difference validation_errors and two of
    # the warnings draw
    checked = arguments if supplied else None
    admission = Admission(candidate, arguments, supplied)

    # validation_errors() and the warnings resolve 'asset:' references
    # themselves (dissolve_frame_errors, video_size_errors,
    # window_count_errors, slice_past_end_warnings, shot_span_warnings) through dw.assets' default
    # discovery, which a real deployment's DW_ASSET_DIR pins to the default
    # workspace - so this request's own workspace is the active library for
    # every check, the same ContextVar the worker activates before it runs.
    # Its outputs are the active root for the same reason: an 'output:' source
    # a check probes otherwise resolves under the default workspace's outputs,
    # misses, and the check stays silent (#666)
    token = activate_asset_dir(workspace.assets) if workspace.assets else None
    output_token = (
        activate_output_root(workspace.outputs) if workspace.outputs else None
    )
    try:
        try:
            # Checked against the caller's arguments, not the document alone:
            # a content_type (or reference_name, ...) that only becomes
            # active once a 'variable:' resolves is caught here (#414, #415),
            # and the caller's list is the one a for_each expands over. The
            # context expands lazily, inside validation_errors' gates, so an
            # expansion failure is still answered there as a finding
            context = validation.workflow_context(
                candidate, checked, ceiling_index=ceiling_index
            )
            admission.errors = candidate.validation_errors(context=context)
        except Exception as e:
            raise _validator_failure(e) from e
        admission.vram_need = _vram_need(candidate, context, arguments)
        if admission.errors:
            # The warnings walk the steps array, which a definition failing
            # the schema may not have
            admission.schema_errors = True
            return admission

        definition = candidate.workflow_definition
        # The arguments checked the way the run would check them: an
        # undeclared name, a value that will not coerce, a reference that
        # names nothing in this workspace. A raise here is the checks
        # failing, not the Workflow failing to construct - validate answers
        # it as a validator failure, as it does one from validation_errors
        try:
            admission.errors = argument_errors(definition, arguments)
            admission.errors += argument_reference_errors(
                definition,
                arguments,
                outputs=workspace.outputs,
                asset_library=asset_library,
                prompt_library=prompt_library,
            )
        except Exception as e:
            raise _validator_failure(e) from e
        # A warning never refuses: a source that fails is logged and said
        # as one internal warning, and the others still report
        admission.warnings = to_warnings(
            run_checks(context, validation.WARNING_CHECKS, WARNING, loud=admission.ok)
        )
        if plan_for is not None and admission.ok:
            admission.plan = plan_for(candidate, admission.vram_need)
        return admission
    finally:
        if output_token is not None:
            deactivate_output_root(output_token)
        if token is not None:
            deactivate_asset_dir(token)


def _vram_need(candidate, context, arguments):
    """required_vram_gb over the definition the run will execute, or None
    where it cannot be read - a need that cannot be computed dispatches the
    job to any card rather than failing a request validation answered."""
    try:
        try:
            definition = context.expanded
        except Exception:
            definition = None
        if not isinstance(definition, dict):
            definition = candidate.workflow_definition
        return required_vram_gb(definition, arguments, get_device_type())
    except Exception as e:
        logger.debug(f"Could not compute the job's VRAM need: {e}")
        return None


def argument_reference_errors(
    definition, arguments, *, outputs, asset_library, prompt_library
):
    """The 'asset:', 'prompt:' and 'output:' references that name nothing
    this workspace can reach, in the values a run would actually use -
    the caller's `arguments`, every declared `variables` default the
    caller did not override, and the literals written into the steps.

    A stored default is exactly as much a promise as a caller's value:
    `validate_workflow(name="templates/ltx2/reference-sheet")` with no
    arguments at all used to answer valid because only `arguments` was
    checked, while the same call with the stored default handed back
    explicitly answered invalid - one run, two verdicts (#166). Reported
    at `variables.<name>` so the message still says whether the caller
    wrote the bad reference or merely didn't override one.

    Resolved through the engine's own resolvers over the roots this
    workspace searches (`asset_library`, `prompt_library`, and its `outputs`),
    so validation agrees with what the run would find - an asset that
    exists in another workspace is a miss here for the same reason it would
    be a miss there. Only the reference is resolved, never loaded: the point
    is to answer before any bytes move.
    """

    def _string_leaves(value, path):
        """Every string in `value`, paired with the path it sits at.

        `value` is walked the way a for_each entry is - a list or dict
        of arbitrary nesting - so a reference inside `shots[2].
        references[1].from_file` is found the same as one at the
        argument's own top level."""
        if isinstance(value, str):
            yield path, value
        elif isinstance(value, list):
            for i, item in enumerate(value):
                yield from _string_leaves(item, f"{path}[{i}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from _string_leaves(item, f"{path}.{key}")

    if arguments is not None and not isinstance(arguments, dict):
        return []
    supplied = arguments if isinstance(arguments, dict) else {}

    effective = []
    declared = definition.get("variables") if isinstance(definition, dict) else None
    if isinstance(declared, dict):
        for name, value in declared.items():
            if name not in supplied:
                effective.append((f"variables.{name}", value))
    for name, value in supplied.items():
        effective.append((f"arguments.{name}", value))
    # A reference written straight into a step is as much a promise as
    # one in a variable: `asset:cast/no-such-voice.wav` as a literal
    # attribute_voices voice validated clean and died on the step
    # (#494). Walked as written, so the path is the author's
    steps = definition.get("steps") if isinstance(definition, dict) else None
    if isinstance(steps, list):
        for index, step in enumerate(steps):
            effective.append((f"steps[{index}]", step))

    errors = []
    for base_path, value in effective:
        for path, leaf in _string_leaves(value, base_path):
            try:
                if is_asset_reference(leaf):
                    if not asset_library.roots():
                        # A server configured with no asset library has
                        # no root to fail against: the resolver would
                        # name no directory it searched
                        name = references.ref_name(references.ASSET, leaf).strip()
                        raise ValueError(
                            f"Unknown asset {name!r}: "
                            "this workspace has no asset library"
                        )
                    # One resolve over the whole path, the way a run does:
                    # a miss names every library it looked in, the
                    # workspace's own first
                    resolve_asset_reference(leaf, library=asset_library)
                elif references.is_ref(references.PROMPT, leaf):
                    resolve_prompt_reference(leaf, library=prompt_library)
                elif is_output_reference(leaf):
                    resolve_output_reference(leaf, root=outputs)
            except Exception as e:
                # Every resolver here raises with a message written for
                # the person who wrote the reference - a traversal
                # refusal from the security layer included
                errors.append({"path": path, "message": str(e)})
    return errors


class AcknowledgedCost(BaseModel):
    """A cost acknowledgement bound to the plan a validate call answered
    with (#85): the server refuses to queue a run whose plan no longer
    matches it. `minutes` is recorded, never compared."""

    fingerprint: str = Field(description="plan.fingerprint from POST /api/validate")
    minutes: Optional[float] = Field(
        default=None, description="plan.estimate.minutes, recorded on the job"
    )
    downloads: List[Optional[str]] = Field(
        default_factory=list,
        description="The repos in plan.downloads_required that were acknowledged",
    )

    @field_validator("downloads", mode="after")
    @classmethod
    def _drop_unnamed(cls, downloads):
        # A from_single_file URL sits in downloads_required with repo null;
        # an acknowledgement copied from the plan verbatim carries it, and
        # a URL has no repo to acknowledge
        return [repo for repo in downloads if repo]


ACKNOWLEDGED_COST_FIELD = Field(
    default=None,
    description="Cost acknowledgement: true (recorded), or an object "
    "{fingerprint, minutes, downloads} bound to the plan validate answered "
    "with - then the run is refused with 409 if its plan changed",
)


class JobRequest(BaseModel):
    workflow_path: Optional[str] = Field(
        default=None, description="Path to a workflow JSON file on the server"
    )
    workflow: Optional[Dict[str, Any]] = Field(
        default=None, description="Inline workflow definition"
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Workflow variable overrides"
    )
    base_dir: Optional[str] = Field(
        default=None,
        description="Directory relative paths in an inline workflow resolve against",
    )
    workspace: Optional[str] = Field(
        default=None,
        description="Which workspace to run or resolve in; the default when omitted",
    )
    acknowledged_cost: Optional[Union[bool, AcknowledgedCost]] = ACKNOWLEDGED_COST_FIELD


def admit_for(state, workspace, **request):
    """`admit()` with this server's view of `workspace` - the asset and
    prompt search paths and the catalog's VRAM ceilings, which live on the
    app's state rather than in the request."""
    return admit(
        workspace=workspace,
        ceiling_index=ceiling_index(state, workspace),
        asset_library=resolution_library(state, workspace),
        prompt_library=server_prompt_library(state),
        **request,
    )


def acknowledgement_form(value):
    """none | boolean | bound - classified once, here, so the check and
    the record agree (#85)."""
    if isinstance(value, AcknowledgedCost):
        return ACK_BOUND
    return ACK_BOOLEAN if value is True else ACK_NONE


def bound_plan_for(arguments, workspace):
    """The `plan_for` a bound acknowledgement is checked against - the
    run these arguments execute, planned without asking the hub for
    sizes. None when it cannot be built, which the check refuses."""

    def plan_for(candidate, vram_need=None):
        try:
            from .. import get_device, get_device_type

            return build_plan(
                candidate,
                arguments,
                device=get_device_type(get_device()),
                prompt_dir=workspace.prompts,
                lookup_sizes=False,
            )
        except Exception:
            logger.exception("Plan could not be built for a bound acknowledgement")
            return None

    return plan_for


def check_bound_acknowledgement(current, acknowledged, workspace):
    """Refuse with 409 when `current`, the plan of the run the request
    admitted, is not the one `acknowledged` was bound to: a different
    fingerprint, or a download the caller did not acknowledge. The body
    carries the current plan so the agent re-quotes from it without a
    second validate call. A plan that could not be built (None) is a
    refusal too - never a silent pass (#85).
    """
    record = acknowledged.model_dump()

    def refuse(message, reason, plan):
        detail = {
            "message": message,
            "reason": reason,
            "acknowledged": record,
            "plan": plan,
        }
        if plan is not None:
            # What to resend, ready-made: a client re-quotes from `plan` and
            # resubmits with this rather than rebuilding it field by field
            detail["acknowledge"] = {
                "fingerprint": plan["fingerprint"],
                "minutes": (plan.get("estimate") or {}).get("minutes"),
                "downloads": [
                    entry["repo"]
                    for entry in plan.get("downloads_required") or []
                    if entry.get("repo")
                ],
            }
        raise HTTPException(status_code=409, detail=detail)

    if current is None:
        refuse(
            "The run could not be planned, so a bound acknowledgement "
            "cannot be checked; acknowledge with true or validate again",
            "unplannable",
            None,
        )
    current["workspace"] = workspace.name
    current["output_dir"] = workspace.outputs
    if current["fingerprint"] != acknowledged.fingerprint:
        refuse(
            "The run's shape changed since it was acknowledged: the "
            "workflow or its arguments differ from what was validated.",
            "fingerprint",
            current,
        )
    missing = [
        entry["repo"]
        for entry in current["downloads_required"]
        if entry.get("repo") and entry["repo"] not in acknowledged.downloads
    ]
    if missing:
        refuse(
            "The run's shape changed since it was acknowledged: it now "
            f"has to download {', '.join(missing)} first",
            "downloads",
            current,
        )
