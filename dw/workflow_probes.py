"""What a run of a Workflow would do, asked without running it.

The worker asks before it spends anything: `require_trusted_pipelines` (would
the run be refused), `pipeline_keys` (which resident pipelines the next
workflow loads anyway) and `cache_hits` (which steps the step cache would
serve). Each prepares the definition as the run does - `cache_hits` and
`pipeline_keys` through `dw.workflow_run`'s own `prepare_definition` and
`cache_lookup`, so the probe answers for the run that would happen - and
executes nothing: no run directory, no events, no pipeline.

This module does not import `dw.workflow`, and `dw.workflow_run` does not
import this one.
"""

import copy
import logging

from .arguments import realize_args
from .runs import activate_output_root, deactivate_output_root
from .security import UntrustedWorkflowError
from .step_cache import step_definition_keys, step_pipeline_keys
from .trust import require_trusted_block, workflows_are_trusted
from .type_references import require_trusted_type_references
from .variables import ConstantError
from .workflow_run import cache_lookup, prepare_definition, run_base_dir

logger = logging.getLogger("dw")


def cache_hits(workflow, arguments):
    """The steps the step cache would serve for a run with `arguments`,
    in step order - what the plan reports as cached_steps (#85).

    Prepares the definition exactly as run() does and asks the cache the
    question run() asks, step by step with the hits so far, and executes
    nothing: no run directory, no events, no pipeline. An unseeded
    workflow has no cache, so it answers [] without asking.
    """
    output_root_token = activate_output_root(workflow.output_dir)
    try:
        workflow_def = copy.deepcopy(workflow.workflow_definition)
        workflow_id = workflow_def["id"]
        base_dir = run_base_dir(workflow)
        workflow_def, default_seed, _ = prepare_definition(
            workflow, workflow_def, arguments or {}, base_dir
        )
        if default_seed is None or not workflow._cache_enabled_by_parent:
            return []
        steps = workflow_def.get("steps", [])
        realize_args(steps, base_dir)
        # The same table run() takes at the same point, so a borrowed
        # key here is the key the run stored its entry under
        definition_keys = step_definition_keys(steps)
        hits_this_run = set()
        hits = []
        for index, step_data in enumerate(steps):
            step_seed = step_data.get("seed", default_seed)
            cached_result, _, _, _ = cache_lookup(
                workflow,
                workflow_id,
                steps,
                index,
                step_data,
                step_seed,
                hits_this_run,
                True,
                definition_keys,
            )
            if cached_result is not None:
                hits_this_run.add(step_data["name"])
                hits.append(step_data["name"])
        return hits
    finally:
        deactivate_output_root(output_root_token)


def _prepared_steps(workflow, arguments):
    output_root_token = activate_output_root(workflow.output_dir)
    try:
        workflow_def = copy.deepcopy(workflow.workflow_definition)
        base_dir = run_base_dir(workflow)
        workflow_def, _, _ = prepare_definition(
            workflow, workflow_def, arguments or {}, base_dir
        )
        steps = workflow_def.get("steps", [])
        realize_args(steps, base_dir)
        return steps
    finally:
        deactivate_output_root(output_root_token)


def require_trusted_pipelines(workflow, arguments):
    """Refuse an untrusted workflow the run would refuse - a pipeline step
    that reaches for remote code or names a pre_load_modules entry outside
    the ecosystem, a dotted type or a 'constant:' outside it - before the
    worker releases the previous workflow's models: a refusal should leave
    the warm cache where it was (#1333, #1511). Raises what the run's own
    gates raise; a definition that cannot be expanded is left for the run to
    report.

    Reads keys only, over validate's expansion (Workflow.expanded_definition):
    no asset, clip or URL is loaded, since the run realizes them itself. A
    trusted server has nothing to refuse, so it does no work at all (#1510).
    """
    if workflows_are_trusted():
        return
    try:
        steps = workflow.expanded_definition(arguments or {}).get("steps", [])
    except ConstantError as e:
        # A 'constant:' variable default the gate refused
        if isinstance(e.__cause__, UntrustedWorkflowError):
            raise e.__cause__
        logger.debug(f"Could not expand the definition for the trust check: {e}")
        return
    except Exception as e:
        logger.debug(f"Could not expand the definition for the trust check: {e}")
        return
    require_trusted_steps(steps)


def require_trusted_steps(steps):
    """The gate over expanded steps the worker runs before it releases the
    warm models: each pipeline block as validate's pipeline_remote_code pass
    reads it (#1410), and each step's type and 'constant:' references as its
    component_types pass does."""
    for step in steps:
        if isinstance(step, dict):
            require_trusted_block(step.get("pipeline"), "pipeline")
    require_trusted_type_references(steps)


def pipeline_keys(workflow, arguments):
    """The set of pipeline cache keys a run of `workflow` with `arguments`
    loads its steps under - the table run() and cache_hits() take
    (step_pipeline_keys over the prepared definition). Prepares the
    definition exactly as a run does and executes nothing: the worker asks
    this before releasing the previous workflow's models, so a model the
    next workflow loads anyway stays warm. A sub-workflow step loads its
    own pipelines later and is not in the set; an unseeded workflow still
    answers, since warmth does not depend on the step cache. The probe
    realizes the steps because the run keys on the realized definition, at
    the price of loading step media once per switch (the price cache_hits
    already pays).
    """
    return set(step_pipeline_keys(_prepared_steps(workflow, arguments)).values())
