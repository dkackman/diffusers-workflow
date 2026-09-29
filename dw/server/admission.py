"""One admission service for every route that takes a job request.

`POST /api/validate`, `POST /api/jobs`, `POST /api/jobs/{id}/rerun` and
`POST /api/enhance` each admit a request through `admit()`: the workflow is
loaded once, checked once - the schema and everything validation_errors
derives from the expansion, the caller's arguments, the references they
make - and warned about once, with the workspace's asset library active for
all of it. JobManager.submit records and queues what was admitted; it does
not re-check. The worker still validates what it loads, which is a
different process with a different library activation.
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

from ..assets import (
    ASSET_PREFIX,
    activate_asset_dir,
    deactivate_asset_dir,
    is_asset_reference,
    resolve_asset_reference,
)
from .. import validation
from ..prompts import PROMPT_PREFIX, resolve_prompt_reference
from ..runs import is_output_reference, resolve_output_reference
from ..validation import WARNING, run_checks, to_warnings
from ..variables import argument_errors
from ..workflow import Workflow, workflow_from_definition, workflow_from_file


class ValidatorFailure(Exception):
    """The validator failing outright rather than answering: building the
    request's validation context, the gates validation_errors runs before
    its checks (schema, 'constraint:' references, the expansion) or the
    argument checks raised something they do not answer as a finding. A
    single check that raises is not this - it is an internal finding, and
    the verdict is invalid (B10). The validate route answers it as an
    invalid workflow whose detail is in the server log; a submit answers
    400 with the message, as it always has."""


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
    asset_roots,
    prompt_roots,
    supplied=True,
    plan_for=None,
):
    """Load the request's workflow once and check it once, with the
    workspace's asset library active for every check (the validate route
    used to leave it off for its warnings). `plan_for`, a callable taking
    the Workflow, is called inside the same scope when the caller needs the
    plan (validate always; submit only for a bound acknowledgement) and only
    when the request is admissible.

    `asset_roots` and `prompt_roots` are the search paths the workspace's
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
    # slice_past_end_warnings, shot_span_warnings) through dw.assets' default
    # discovery, which a real deployment's DW_ASSET_DIR pins to the default
    # workspace - so this request's own workspace is the active library for
    # every check, the same ContextVar the worker activates before it runs
    token = activate_asset_dir(workspace.assets) if workspace.assets else None
    try:
        try:
            # Checked against the caller's arguments, not the document alone:
            # a content_type (or reference_name, ...) that only becomes
            # active once a 'variable:' resolves is caught here (#414, #415),
            # and the caller's list is the one a for_each expands over. The
            # context expands lazily, inside validation_errors' gates, so an
            # expansion failure is still answered there as a finding
            context = candidate.validation_context(checked, ceiling_index=ceiling_index)
            admission.errors = candidate.validation_errors(context=context)
        except Exception as e:
            raise ValidatorFailure(str(e)) from e
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
                asset_roots=asset_roots,
                prompt_roots=prompt_roots,
            )
        except Exception as e:
            raise ValidatorFailure(str(e)) from e
        # A warning never refuses: a source that fails is logged and said
        # as one internal warning, and the others still report
        admission.warnings = to_warnings(
            run_checks(context, validation.WARNING_CHECKS, WARNING)
        )
        if plan_for is not None and admission.ok:
            admission.plan = plan_for(candidate)
        return admission
    finally:
        if token is not None:
            deactivate_asset_dir(token)


def argument_reference_errors(
    definition, arguments, *, outputs, asset_roots, prompt_roots
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
    workspace searches (`asset_roots`, `prompt_roots`, and its `outputs`),
    so validation agrees with what the run would find - an asset that
    exists in another workspace is a miss here for the same reason it would
    be a miss there. Only the reference is resolved, never loaded: the point
    is to answer before any bytes move.
    """

    def over_roots(roots, resolve):
        """Resolve against each root in turn, and on a total miss raise
        the *first* root's error rather than the last.

        The resolvers name the path they searched in their message, and
        the first root is the workspace's own library plus the read-only
        fallbacks the environment pins - which is the path a run would
        report. The last root's message would name an examples directory
        and leave out the workspace, reading as though the library the
        caller works in was never looked in."""
        first = None
        for root in roots:
            try:
                return resolve(root)
            except Exception as e:
                first = first or e
        # `roots` is never empty here: the asset branch answers an empty
        # search path itself, and the prompt path always holds the
        # server's own library. Re-raising None would be a TypeError
        raise first

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
                    if not asset_roots:
                        # A server configured with no asset library has
                        # no root to fail against: over_roots would
                        # re-raise its "first error", which is None,
                        # and the caller would read a TypeError about
                        # BaseException in place of a verdict
                        name = leaf.removeprefix(ASSET_PREFIX).strip()
                        raise ValueError(
                            f"Unknown asset {name!r}: "
                            "this workspace has no asset library"
                        )
                    over_roots(
                        asset_roots,
                        lambda root: resolve_asset_reference(leaf, asset_dir=root),
                    )
                elif leaf.startswith(PROMPT_PREFIX):
                    over_roots(
                        prompt_roots,
                        lambda root: resolve_prompt_reference(leaf, prompt_dir=root),
                    )
                elif is_output_reference(leaf):
                    resolve_output_reference(leaf, root=outputs)
            except Exception as e:
                # Every resolver here raises with a message written for
                # the person who wrote the reference - a traversal
                # refusal from the security layer included
                errors.append({"path": path, "message": str(e)})
    return errors
