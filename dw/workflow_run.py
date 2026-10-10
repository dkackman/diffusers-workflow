"""One execution of a Workflow, in named phases.

`Workflow.run` is the orchestrator: it activates the run's context, opens a
`RunRecord`, and calls the phases here in order - `prepare_run` (the
definition as the run works from it, and the seed), `open_run` (the run
directory, the realized workflow and `run_start`), `begin_steps` (the step
loop's state and `workflow_start`), then `run_step` once per step. The step
cache probe (`cache_hits`) shares `prepare_definition` and `cache_lookup`
with the run, so the probe answers for the run that would happen.

Every phase takes the Workflow and writes what others read back onto it:
`manifest`, `_run_dir`, `_run_version`, `_elided_steps` and
`_cache_enabled_this_run`. A composed child is recognized as an instance of
the running workflow's own type.

This module does not import `dw.workflow`.
"""

import copy
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import __version__, device_capacity_gb, get_device, get_device_type
from . import references
from .adapter_compatibility import warn_adapters
from .arguments import realize_args
from .elision import elide_definition, warn_elided
from .events import emit_warning
from .media import probe_metadata
from .pipeline_ownership import allocated_mb, finish_release
from .previous_results import StepResults
from .realize import realize_workflow
from .runs import (
    FLAT_LAYOUT,
    REALIZED_FILE_NAME,
    activate_output_root,
    deactivate_output_root,
    manifest_relative_files,
    new_run_id,
    open_run as claim_run_dir,
    output_layout,
    workflow_identity,
    write_manifest,
    write_realized_workflow,
)
from .shots import (
    carries_shots,
    duplicate_shot_names,
    name_unsaved_shots,
    shot_references,
    step_shots,
)
from .step import Step
from .step_cache import (
    audio_replaced_downstream,
    borrowed_pipeline_keys,
    copy_containers,
    normalized_downstream,
    reference_resolves_to,
    referenced_result_names,
    step_cache,
    step_definition_keys,
    step_pipeline_keys,
)
from .subfolders import step_subfolder
from .variable_constraints import apply_constraints
from .vram_estimate import apply_vram_estimate

logger = logging.getLogger("dw")

# The widest integer JavaScript's double represents exactly - the ceiling on
# any seed the engine draws, since seeds travel as JSON through a browser
SEED_BITS = 53


def _now():
    return datetime.now(timezone.utc).isoformat()


def _empty_annotations():
    return {"prompts": [], "sub_workflows": {}}


@dataclass
class RunRecord:
    """What a run records about itself, read by every manifest write.

    Initialized before anything can fail, so a failure before the run
    directory exists still writes a well-formed manifest. `status` stays
    "failed" unless the run says otherwise on its way out - a run that
    leaves it alone died on an exception the manifest should say so about.
    `arguments` are the caller's, rebound to the composed child's own copy
    by `prepare_run`; `seed` is the one the run actually used, which a
    workflow naming none draws.
    """

    arguments: object = None
    status: str = "failed"
    run_id: object = None
    started_at: str = field(default_factory=_now)
    seed: object = None
    realized_name: object = None
    annotations: dict = field(default_factory=_empty_annotations)


@dataclass
class PreparedRun:
    """The definition as the run works from it (`prepare_run`)."""

    workflow_def: dict
    workflow_id: str
    base_dir: object
    default_seed: int
    recorded_variables: object
    cache_enabled: bool


@dataclass
class StepLoop:
    """The state the step loop carries from one step to the next."""

    workflow_id: str
    steps: list
    default_seed: int
    cache_enabled: bool
    run_context: object
    pipelines: dict
    # Results from each step, remembering the names of steps whose results
    # have since been released
    results: StepResults = field(default_factory=StepResults)
    # Shared resources between steps
    shared_components: dict = field(default_factory=dict)
    # Step name -> whether that step's result this run came from the cache,
    # so a step that reads another step's result can tell whether its own
    # inputs are still all cache-fresh
    hits: set = field(default_factory=set)
    # Final result is the workflow return value
    last_result: object = None
    # step_definition_keys of `steps`, taken before any step loaded - what
    # the step cache folds in for a borrowed pipeline (cache_lookup)
    definition_keys: dict = field(default_factory=dict)


def run_base_dir(workflow):
    """The directory file paths in the workflow resolve against - the
    workflow file's own."""
    return workflow.base_dir


def owns_run_dir(workflow):
    """Whether this run writes the run directory's records - false for a
    sub-workflow, which inherits the parent's, and in the flat layout."""
    return bool(workflow._run_dir and not workflow._run_dir_inherited)


def _relative_shots(entry, run_dir):
    """A manifest entry's `shots`, each `file` made relative as `files` is."""
    shots = entry.get("shots")
    if not shots or not any("file" in shot for shot in shots):
        return {}
    files = manifest_relative_files([shot["file"] for shot in shots], run_dir)
    return {"shots": [{**shot, "file": f} for shot, f in zip(shots, files)]}


def selected_field(step_data, selected):
    """The manifest/step_end 'selected' block for a step's Result.selected.

    Carries the winning position and score always; adds 'entry' - the
    for_each member name ('shot@b') - only when the step's 'candidates'
    argument was built from a gather: reference, recoverable at this point
    only because step_data still holds the post-for_each-expansion,
    pre-argument-resolution reference strings (dw/for_each.py's _gather()).
    """
    if selected is None:
        return None

    field = dict(selected)
    candidates = step_data.get("task", {}).get("arguments", {}).get("candidates")
    if isinstance(candidates, list):
        position = selected.get("position")
        if isinstance(position, int) and 0 <= position < len(candidates):
            candidate = candidates[position]
            entry = references.ref_name(references.PREVIOUS_RESULT, candidate)
            if entry is not None:
                field["entry"] = entry
    return field


def release_unreferenced_results(results, remaining_refs):
    """Drop results no remaining reference can resolve to.

    A reference resolves to a result whose name it equals or extends with a
    property ('step.mask'), so any result that is such a prefix stays. Saved
    artifacts are already on disk - holding every intermediate image and frame
    list in RAM until the workflow ends is what OOMs long chains.
    """
    for name in [
        n
        for n in results
        if not any(reference_resolves_to(ref, n) for ref in remaining_refs)
    ]:
        logger.debug(f"Releasing result: {name}")
        del results[name]


def write_run_manifest(workflow, record, status=None):
    """Leave a record of the run beside the files it wrote.

    A server run is in jobs.sqlite as well, but a CLI run has never been
    recorded anywhere, and a database on one machine cannot describe a
    directory copied to another. Paths are relative to the run directory
    so the directory keeps describing itself wherever it goes.

    `status` overrides the record's own - the writes made while the run is
    still going say "running" without touching the status the run ends
    with.
    """
    status = status or record.status
    annotations = record.annotations
    write_manifest(
        workflow._run_dir,
        {
            "run_id": record.run_id,
            # This run's ordinal among the workflow's runs - 'v4' in the
            # gallery. Recorded, never recomputed
            "version": workflow._run_version,
            "status": status,
            "started_at": record.started_at,
            # None on the manifest written as the run opens
            "finished_at": (None if status == "running" else _now()),
            "dw_version": __version__,
            "device": str(get_device()),
            "workflow": {
                "id": workflow.name,
                "file": workflow.file_spec,
                "identity": workflow_identity(workflow.file_spec, workflow.name),
                # The realized copy beside this manifest, or null when
                # writing it did not land - the manifest is the only
                # place that difference is visible
                "realized": record.realized_name,
                # Annotations the schema has nowhere to put: which
                # stored prompts were inlined, and what each local
                # sub-workflow file held when it ran
                "prompts": (annotations or {}).get("prompts", []),
                "sub_workflows": (annotations or {}).get("sub_workflows", {}),
            },
            "seed": record.seed,
            "arguments": record.arguments or {},
            # What did not run, and why - a run says what it did not do
            # as well as what it did (#122)
            "elided_steps": workflow._elided_steps,
            "steps": [
                {
                    **entry,
                    "files": manifest_relative_files(
                        entry.get("files"), workflow._run_dir
                    ),
                    **_relative_shots(entry, workflow._run_dir),
                }
                for entry in workflow.manifest
            ],
        },
    )


def prepare_definition(workflow, workflow_def, arguments, base_dir):
    """The definition as a run works from it: constants realized,
    arguments folded into the variables, list entries' own references
    resolved, variable values realized (assets loaded), every
    'variable:' substituted, every for_each expanded, and the seed read
    and coerced. Returns (workflow_def, default_seed,
    recorded_variables) - the seed is None when the workflow names none,
    and the caller decides what that means (run() draws one;
    cache_hits() reports no hits); recorded_variables are the folded
    values taken before anything loads, what the realized workflow
    records (None when none are declared).

    Shared by run() and cache_hits() so the probe prepares exactly what
    the run prepares - the step cache keys on the realized step, and a
    probe that prepared it differently would answer for a run that
    never happens. The fold and the expansion are the stages validation
    runs too (expanded_definition), so what validates is what runs.

    Records the steps elision dropped on `workflow._elided_steps` (#122) -
    the same list run() warns about and writes into the manifest.
    """
    workflow_id = workflow_def["id"]
    logger.debug(f"Setting variables for workflow: {workflow_id}")
    # A value outside a declared rule is refused here, and one the rule
    # rounds is rounded with a warning saying so (apply_constraints)
    variables = workflow._fold(
        workflow_def, arguments, fold_arguments=True, constrain=apply_constraints
    )
    recorded_variables = copy_containers(variables)
    if variables is not None:
        # The definition carries the folded values rather than the
        # loaded ones: realize_args below loads assets into `variables`,
        # and substitution copies the variables block along with the
        # steps
        workflow_def["variables"] = recorded_variables
        # realize the variables - explicit references only (asset:,
        # output:, constant:, prompt:, a {media_type, location} dict).
        # Key-name conventions (an 'image'/'video'/'_type' argument) are
        # left off here: a variable's own name is not the argument it
        # will end up filling, so a variable named 'image' fed to a step's
        # 'video' argument was pre-loaded as a PIL Image before that step
        # was ever substituted in (#365). The step-level realize_args
        # passes below apply the conventions under the real argument key
        realize_args(variables, base_dir, apply_key_conventions=False)
    ## then replace any variable references in the workflow definition
    # with the actual values, and expand
    workflow_def = workflow._expand(workflow_def, variables)

    # A step a declared vram_estimate projects past the card 'cost' was
    # measured on - the run-time backstop for a caller that skips
    # validate_workflow, so this raises the same refusal rather than
    # starting a job the decode step was always going to OOM on. After
    # expansion, so a for_each member is projected with its own frames
    # and references (dw/vram_estimate.py, #265, #479). The header-only
    # probe counts each guide clip's frames as validate's does (#694)
    apply_vram_estimate(
        workflow_def,
        variables,
        device_type=get_device_type(),
        capacity_gb=device_capacity_gb(),
        base_dir=base_dir,
        probe=probe_metadata,
    )

    # A step nothing after it reads, and which saves no file, does not
    # run - after expansion, so a for_each member is judged like any
    # other step, and before the seed and the run id, so everything
    # downstream counts the steps that will actually execute
    # (dw/elision.py, #122)
    # The definition as written is passed too: a step dropped because a
    # caller replaced the variable that read it was elided on purpose,
    # and says so, rather than being reported as a suspected typo (#157)
    workflow._elided_steps = elide_definition(
        workflow_def, workflow.workflow_definition
    )

    # Set up random seed for reproducibility. Resolved lazily - as a
    # dict.get default, torch.seed() would run on every call and reseed
    # the global RNG even when the workflow names an explicit seed
    default_seed = workflow_def.get("seed")
    # The schema lets 'seed' be a string so it can hold a 'variable:'
    # reference, which the substitution above has already resolved -
    # but a variable overridden from the command line arrives as a
    # string whenever the workflow declared no integer default to
    # coerce against, and manual_seed would fail deep inside the run
    if isinstance(default_seed, str):
        try:
            default_seed = int(default_seed)
        except ValueError:
            raise ValueError(
                f"Workflow {workflow_id} seed must be an integer, got {default_seed!r}"
            )
        workflow_def["seed"] = default_seed
    return workflow_def, default_seed, recorded_variables


def cache_lookup(
    workflow,
    workflow_id,
    steps,
    index,
    step_data,
    step_seed,
    hits_this_run,
    cache_enabled,
    definition_keys,
):
    """Whether the step cache serves step `index`, as (cached_result or
    None, the step_data snapshot the entry is keyed on or None, whether
    a later step still reads this one's result, the names later steps
    still reference). Shared by run() and cache_hits() - see
    prepare_definition for why. `definition_keys` is step_definition_keys
    of `steps`, taken before the first step ran - not the pipeline
    ownership table, whose keys leave out a source's LoRA scale, alpha and
    shift, each of which changes what a borrowing step produces.
    """
    # What later steps still read, which decides both whether this
    # step's result has to be kept alive after the step (release_unreferenced_results
    # at the bottom of the run loop) and whether a cached entry that
    # kept none can serve this run
    remaining_refs = referenced_result_names(steps[index + 1 :])
    result_needed = index == len(steps) - 1 or any(
        reference_resolves_to(ref, step_data["name"]) for ref in remaining_refs
    )
    # create_step_action (and the pipeline load it triggers) mutates
    # step_data in place - injecting a "generator" key - so the cache
    # must key off a snapshot taken before that happens, and that same
    # snapshot must be reused for the put() later. Caching off the live,
    # later-mutated step_data would make every step's dict keys diverge
    # from a freshly deep-copied future run's step_data, so get() would
    # never match again after the first run.
    # A sub-workflow step is never cacheable: its files roll up from the
    # child's own manifest, which a hit does not rebuild.
    is_cacheable = "workflow" not in step_data and cache_enabled
    # The last step of a composed child whose parent does the saving
    # (#92) - its files are the parent step's, written once, under the
    # parent's name and subfolder
    parent_saves_this = workflow._final_save_owned_by_parent and index == len(steps) - 1
    step_data_snapshot = None
    if is_cacheable:
        try:
            step_data_snapshot = copy.deepcopy(step_data)
            if parent_saves_this:
                # Keyed apart from the same step run standalone: this
                # entry's result was never saved here, so a standalone
                # hit on it would report no files
                step_data_snapshot["__saved_by_parent__"] = True
            # This step's own step_data never names a borrowed pipeline's
            # model - only the source step's does - so without this a
            # source model change would leave the borrowing step's
            # snapshot unchanged and serve a stale hit
            borrowed = borrowed_pipeline_keys(steps, index, definition_keys)
            if borrowed:
                step_data_snapshot["__borrowed_pipelines__"] = borrowed
        except Exception as ex:
            # A realized argument that cannot be deep-copied (an open
            # handle, a live model object) just means this step is not
            # cacheable - never a failed run
            logger.debug(
                f"Step '{step_data['name']}' arguments are not copyable "
                f"({ex}) - skipping the step cache for it"
            )
            is_cacheable = False
    if not is_cacheable:
        return None, None, result_needed, remaining_refs
    cached_result = step_cache.get(
        workflow_id,
        step_data_snapshot,
        step_seed,
        hits_this_run,
        # The root, not this run's directory: a hit reports the earlier
        # run's files and writes nothing new, so keying on a directory
        # that is new every run would mean the cache could never hit
        # again. What the root still guards is a run redirected
        # somewhere else, where the earlier files are not what the
        # caller asked for
        workflow.output_dir,
        needs_result=result_needed,
    )
    return cached_result, step_data_snapshot, result_needed, remaining_refs


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


def pipeline_keys(workflow, arguments):
    """The set of pipeline cache keys a run of `workflow` with `arguments`
    loads its steps under - the table run() and cache_hits() take
    (step_pipeline_keys over the prepared definition). Prepares the
    definition exactly as a run does and executes nothing: the worker asks
    this before releasing the previous workflow's models, so a model the
    next workflow loads anyway stays warm. A sub-workflow step loads its
    own pipelines later and is not in the set; an unseeded workflow still
    answers, since warmth does not depend on the step cache.
    """
    output_root_token = activate_output_root(workflow.output_dir)
    try:
        workflow_def = copy.deepcopy(workflow.workflow_definition)
        base_dir = run_base_dir(workflow)
        workflow_def, _, _ = prepare_definition(
            workflow, workflow_def, arguments or {}, base_dir
        )
        return set(step_pipeline_keys(workflow_def.get("steps", [])).values())
    finally:
        deactivate_output_root(output_root_token)


def prepare_run(workflow, record):
    """Phase B: the definition as this run works from it, and its seed.

    A composed child takes its one copy of what its parent handed it here,
    and the copy is what the record carries - the run id and every manifest
    write read `record.arguments`.
    """
    if workflow._composed:
        record.arguments = workflow._owned_arguments(record.arguments)
    # CRITICAL: Work on a copy to avoid mutating the original workflow definition
    # This allows the workflow to be run multiple times with different arguments
    workflow_def = copy.deepcopy(workflow.workflow_definition)

    workflow_id = workflow_def["id"]
    logger.debug(f"Processing workflow: {workflow_id}")

    # File paths in workflows are relative to the workflow file
    base_dir = run_base_dir(workflow)

    workflow_def, default_seed, recorded_variables = prepare_definition(
        workflow, workflow_def, record.arguments, base_dir
    )
    # An adapter whose name says nothing about what it was trained
    # for cannot be checked, and a run that started from the CLI or
    # from a rerun never passed the validate route (#155)
    warn_adapters(workflow_def)
    # Said out loud before anything loads: a step that vanishes
    # because a reference to it is misspelled would otherwise show
    # up only as a different picture (#122)
    warn_elided(workflow._elided_steps)
    # A workflow that names no seed gets a fresh one every run, so no
    # step's cache entry can ever match again - skip the cache
    # wholesale rather than deep-copying every step's realized images
    # and pinning every Result for a hit that cannot happen
    cache_enabled = workflow._cache_enabled_by_parent and default_seed is not None
    # create_step_action hands this down to a sub-workflow: it injects
    # the parent's seed into a child that names none, so a child of a
    # seedless parent would otherwise look seeded - and cacheable -
    # while its seed still changes every run
    workflow._cache_enabled_this_run = cache_enabled
    if default_seed is None:
        # OS entropy rather than torch or random, so a process that
        # seeded either for reproducibility is not disturbed. Bounded
        # to 53 bits rather than the 64 torch allows: the seed is
        # embedded in the image, the manifest and the realized
        # workflow as JSON, and a browser reads every integer as a
        # double - a seed that changed on the way through would be a
        # seed nobody can reproduce
        default_seed = secrets.randbits(SEED_BITS)
    workflow_def["seed"] = default_seed
    # The seed this run actually used, which is not the workflow's own
    # definition's: the run works on a deep copy, and a workflow naming no
    # seed draws a random one into that copy. Recording the original would
    # write null into the manifest of every seedless run and lose the only
    # record of what produced its files - the seed is what makes a run
    # repeatable
    record.seed = default_seed
    return PreparedRun(
        workflow_def,
        workflow_id,
        base_dir,
        default_seed,
        recorded_variables,
        cache_enabled,
    )


def _claim_run_dir(workflow, prepared, record):
    """The run directory and its version, claimed for a run that does not
    inherit one - none in the flat layout."""
    if output_layout() == FLAT_LAYOUT:
        workflow._run_dir = None
        return
    record.run_id = new_run_id(
        {"workflow": prepared.workflow_def, "arguments": record.arguments}
    )
    # Directory and version are claimed together, under one
    # lock, so two processes opening a run of this workflow
    # at once - a CLI run beside a server job - cannot take
    # the same directory or the same number
    workflow._run_dir, workflow._run_version = claim_run_dir(
        workflow.output_dir, workflow.file_spec, prepared.workflow_id, record.run_id
    )
    # The claimed directory's own name, which may carry a
    # '-N' counter when run_id was already taken - the
    # manifest and the run_start event must carry the name
    # that was actually claimed
    record.run_id = os.path.basename(workflow._run_dir)
    logger.debug(f"Run directory: {workflow._run_dir} (v{workflow._run_version})")


def open_run(workflow, prepared, record, run_context):
    """Phase C: one execution, one directory - opened after variable
    substitution and the seed have settled, so the run's identity covers
    what actually ran rather than what was written down. A sub-workflow
    inherits the parent's and never opens its own."""
    record.started_at = _now()
    record.run_id = None
    if not workflow._run_dir_inherited:
        _claim_run_dir(workflow, prepared, record)

    # The record of what actually ran, written before the first step
    # so a crash or a cancel still leaves it. A sub-workflow inherits
    # the parent's directory and writes none of its own, as with the
    # manifest, and the flat layout has no directory to write into
    if not owns_run_dir(workflow):
        return
    try:
        realized, record.annotations = realize_workflow(
            workflow.workflow_definition,
            prepared.recorded_variables,
            prepared.default_seed,
            base_dir=prepared.base_dir,
            output_root=workflow.output_dir,
            workflow_dir=workflow.workflow_dir,
        )
        if write_realized_workflow(workflow._run_dir, realized):
            record.realized_name = REALIZED_FILE_NAME
    except Exception as e:
        # Never fatal: the record is worth less than the run
        logger.warning(f"Could not realize workflow {prepared.workflow_id}: {e}")

    # A manifest now, rewritten in full when the run ends: the
    # version held only in memory until then was lost to a hard
    # kill, and a second process opening a run of this workflow
    # meanwhile could not see it and took the same number
    write_run_manifest(workflow, record, "running")

    # Which run this is, so a server job can find the directory
    # it wrote. Emitted even when the realized file did not land:
    # the manifest is still there, and so are the files
    run_context.emit(
        "run_start",
        run_id=record.run_id,
        version=workflow._run_version,
        identity=workflow_identity(workflow.file_spec, prepared.workflow_id),
        run_dir=os.path.relpath(workflow._run_dir, workflow.output_dir).replace(
            os.sep, "/"
        ),
    )


def begin_steps(workflow, prepared, previous_pipelines, run_context):
    """Phase D: the step loop's state, the steps' own arguments realized
    and `workflow_start` - or None for a workflow with no steps."""
    # Use provided pipelines cache or create new dict
    # This allows pipeline reuse across multiple workflow runs
    if previous_pipelines is None:
        pipelines = {}
        logger.debug("Starting with empty pipeline cache")
    else:
        pipelines = previous_pipelines
        logger.debug(f"Reusing pipeline cache with {len(pipelines)} pipelines")

    # realize any arguments for the steps, i.e. load images etc
    # that are referenced directly in the step
    steps = prepared.workflow_def.get("steps", [])

    if not steps:
        logger.warning(f"Workflow {prepared.workflow_id} has no steps defined")
        return None

    realize_args(steps, prepared.base_dir)

    # The key each pipeline step of THIS run loads under, computed
    # from the same realized dicts create_step_action hashes, so the
    # two agree. This is what "still shared" means there: a key
    # another running step maps to NOW - not the key it mapped to
    # last run (every step sharing a changed model variable has the
    # old key as its prior key and none has it as its current one),
    # and not a key some step of a past, unrelated workflow left in
    # the cross-job prior-keys map. Taken before any load edits a
    # definition, it is also the table every borrowed-key lookup reads
    workflow.pipeline_ownership.begin(steps)

    run_context.emit(
        "workflow_start",
        workflow=prepared.workflow_id,
        total_steps=len(steps),
        steps=[step_data["name"] for step_data in steps],
        seed=prepared.default_seed,
    )
    return StepLoop(
        prepared.workflow_id,
        steps,
        prepared.default_seed,
        prepared.cache_enabled,
        run_context,
        pipelines,
        definition_keys=step_definition_keys(steps),
    )


def wire_child(workflow, child, step_data, index, total_steps):
    """A composed child reports into this run's step counter, and leaves
    its last save to the parent step when that step saves."""
    # The child reports into this run's counter rather than
    # its own, and a grandchild reports into the same one
    child._parent_progress = workflow._parent_progress or {
        "step": step_data["name"],
        "index": index,
        "total_steps": total_steps,
    }
    # Only when the parent's own result would write
    # something: a result block that names no content_type,
    # or says save: false, saves nothing, and suppressing
    # the child's save for it would lose the artifact
    parent_result = step_data.get("result")
    child._final_save_owned_by_parent = bool(
        isinstance(parent_result, dict)
        and parent_result.get("content_type")
        and parent_result.get("save", True)
    )


def release_step_pipeline(workflow, loop, step_name):
    """Pop a release_pipeline step's pipeline and mark it released. Returns
    the popped pipeline, or None - the caller drops its own references to
    it and to the step's action before the release is finished."""
    released = loop.pipelines.pop(workflow.pipeline_ownership.key_for(step_name), None)
    workflow.pipeline_ownership.mark_released(step_name)
    return released


def save_step(workflow, loop, index, step_data, step, result, cache_entry):
    """Write a step's result and store it in the step cache. Returns the
    saved files - none for the last step of a child whose parent saves."""
    step_data_snapshot, step_seed, result_needed, parent_saves_this = cache_entry
    if parent_saves_this:
        # No file is written here - the parent owns that
        # (#92) - but this step's own declared fps and its
        # audio-to-frames fit must still land on the artifact
        # before it is handed up, or the parent (and any
        # previous_result: consumer) sees an unstamped one
        # and falls back to DEFAULT_VIDEO_FPS (#561)
        result.conform_artifacts()
        saved_files = []
    else:
        saved_files = result.save(
            workflow.step_output_dir(step_data),
            workflow.step_save_name(loop.workflow_id, step.name, index),
        )
    if step_data_snapshot is not None:
        step_cache.put(
            loop.workflow_id,
            step_data_snapshot,
            step_seed,
            result,
            workflow.output_dir,
            retain_result=result_needed,
        )
    return saved_files


def _warn_shot_collisions(step_data, step_name, shots):
    """Joined shots that share a name, said where the join made them."""
    # Only the step that joined made the collision; a step
    # that carries an input's shots over (pair_audio,
    # interpolate_frames) repeats what the caller can only
    # act on at the join (#568)
    command = step_data.get("task", {}).get("command")
    duplicates = None if carries_shots(command) else duplicate_shot_names(shots)
    if not duplicates:
        return
    # Frames and samples stay exact either way - only a
    # name-based lookup (a `shots=` argument, a finding)
    # can no longer tell the collided shots apart (#508)
    for file, names in duplicates.items():
        extra = {"file": file} if file is not None else {}
        emit_warning(
            "joined shots share a name ("
            + ", ".join(names)
            + ") and can no longer be told apart by "
            "name - start_frame still disambiguates.",
            kind="shot_name_collision",
            command=step_name,
            names=names,
            **extra,
        )


def record_step(workflow, loop, index, step_data, outcome, sub_manifest):
    """Phase E7: the step's manifest entry, the child's rolled-up entries,
    and `step_end`. `outcome` is (result, saved_files, reused,
    parent_saves_this)."""
    result, saved_files, reused, parent_saves_this = outcome
    step_name = step_data["name"]
    # 'reused' marks files an earlier run wrote and this one only
    # republished, so nothing downstream (job_for_file, the
    # gallery) credits this run with writing them
    subfolder = step_subfolder(step_data)
    selected = selected_field(step_data, result.selected)
    details = {}
    if reused:
        details["reused"] = True
    if selected is not None:
        details["selected"] = selected
    # Where each joined shot sits in the file, named by the
    # step's own input references (dw/shots.py)
    references = shot_references(step_data.get("task", {}).get("arguments", {}))
    name_unsaved_shots(result, references)
    shots = step_shots(getattr(result, "saved_shots", None), saved_files, references)
    if shots:
        details["shots"] = shots
        _warn_shot_collisions(step_data, step_name, shots)
    # No entry at all for a step the parent saves for: the
    # parent's own entry names the same files, under the step
    # name the caller wrote (#92)
    if not parent_saves_this:
        workflow.manifest.append(
            {"step": step_name, "files": saved_files, "subfolder": subfolder} | details
        )
    # roll the child's saves up so job history and the gallery see
    # every file. Each entry is tagged with the composing step
    # that produced it - a for_each member's files otherwise sit
    # under the child template's own (repeated) step name with
    # nothing tying an entry back to its member (#560). A deeper
    # rollup (a child composing a grandchild) already carries its
    # own tag, which stays: the nearest composing step is the one
    # that matters for grouping
    for sub_entry in sub_manifest:
        sub_entry.setdefault("parent_step", step_name)
    workflow.manifest.extend(sub_manifest)
    loop.run_context.emit(
        "step_end",
        workflow=loop.workflow_id,
        step=step_name,
        index=index,
        total_steps=len(loop.steps),
        **workflow._parent_progress_fields(),
        files=saved_files,
        subfolder=subfolder,
        **details,
    )
    logger.debug(f"Step {step_name} completed with result: {result}")


def _start_step(workflow, loop, index, step_data):
    """`step_start`, and the Step with the seed it runs under."""
    steps = loop.steps
    loop.run_context.check_cancelled()
    logger.debug(f"Running step {index + 1}/{len(steps)}: {step_data['name']}")
    loop.run_context.emit(
        "step_start",
        workflow=loop.workflow_id,
        step=step_data["name"],
        index=index,
        total_steps=len(steps),
        **workflow._parent_progress_fields(),
    )

    # Seeds resolve most-specific-first: pipeline > step > workflow
    step_seed = step_data.get("seed", loop.default_seed)

    step = Step(
        step_data,
        step_seed,
        workflow.workflow_definition,
        consumed_by_normalizer=normalized_downstream(
            steps[index + 1 :], step_data["name"]
        ),
        audio_replaced_downstream=audio_replaced_downstream(
            steps[index + 1 :], step_data["name"]
        ),
    )
    return step, step_seed


def run_step(workflow, loop, index, step_data, record):
    """Phase E: one step, start to `step_end`. Returns the names later
    steps still reference; the result is the loop's (`last_result`,
    `results`), so the caller holds no reference of its own to it.

    Holds the step's action in this frame and nowhere else, so the release
    a step asks for can drop it here before memory is reclaimed.
    """
    steps = loop.steps
    step, step_seed = _start_step(workflow, loop, index, step_data)
    cached_result, step_data_snapshot, result_needed, remaining_refs = cache_lookup(
        workflow,
        loop.workflow_id,
        steps,
        index,
        step_data,
        step_seed,
        loop.hits,
        loop.cache_enabled,
        loop.definition_keys,
    )
    # The last step of a composed child whose parent does the
    # saving (#92) - its files are written once, by the parent
    parent_saves_this = workflow._final_save_owned_by_parent and index == len(steps) - 1

    # A hit skips the step's work, never its bookkeeping:
    # create_step_action is the only place that touches the
    # step's pipeline (the worker evicts every pipeline a run did
    # not touch), republishes a resident pipeline's
    # shared_components for a later reusing step, and records the
    # step's pipeline key for release_pipeline and
    # pipeline_reference to address it by. Loading is not
    # bookkeeping: a hit whose pipeline is not resident defers
    # it, and it loads only when a step that actually runs
    # borrows it - whether a later step will be a hit too is not
    # known until that step's own lookup
    if cached_result is None:
        workflow._load_deferred_borrows(
            loop.workflow_id, steps, index, loop.shared_components, loop.pipelines
        )
    step_action = workflow.create_step_action(
        step_data,
        loop.shared_components,
        loop.pipelines,
        step_seed,
        get_device(),
        cache_hit=cached_result is not None,
    )
    is_child = isinstance(step_action, type(workflow))
    if is_child:
        wire_child(workflow, step_action, step_data, index, len(steps))
    reused = cached_result is not None
    if reused:
        logger.info(f"Step '{step.name}' unchanged - reusing cached result")
        result = cached_result
        saved_files = result.saved_files
        loop.hits.add(step.name)
    else:
        result = step.run(loop.results, loop.pipelines, step_action)

    # A sub-workflow's saves land in the child's manifest - read it
    # here, before the release below may drop the child
    sub_manifest = list(getattr(step_action, "manifest", [])) if is_child else []

    # A released pipeline frees its memory for later steps - the
    # alternative on a card that cannot hold two models is offloading
    # everything, which taxes every run to survive one transition.
    # Before the write, not after: the result is already in host
    # memory and saving never touches the pipeline, so a release
    # that waited for the write would hold ~10 GB on the device
    # through the longest phase of a video step. This frame's own
    # locals are the last references to this step's action, so
    # clearing that is part of the release - a popped pipeline this
    # frame still holds is not freed, and it would otherwise stay
    # resident through the next step's load, which is exactly when
    # both models would be in memory at once
    release = step_data.get("release_pipeline", False)
    released = release_step_pipeline(workflow, loop, step.name) if release else None
    # A hit that loaded nothing holds nothing: announcing a
    # release would report a drop that never happened. Not gated
    # on the pop alone - a sub-workflow step has no pipeline key,
    # and clearing step_action is what frees its child Workflow
    if release and (released is not None or step_action is not None):
        logger.info(f"Releasing pipeline for step: {step.name}")
        before = allocated_mb()
        released = None
        step_action = None
        finish_release(loop.workflow_id, step.name, index, before)

    if not reused:
        saved_files = save_step(
            workflow,
            loop,
            index,
            step_data,
            step,
            result,
            (step_data_snapshot, step_seed, result_needed, parent_saves_this),
        )

    loop.last_result = result
    loop.results[step.name] = result
    record_step(
        workflow,
        loop,
        index,
        step_data,
        (result, saved_files, reused, parent_saves_this),
        sub_manifest,
    )

    # Rewritten after every step, not only at the end (#480): a
    # for_each member that just landed is otherwise invisible to
    # anything reading manifest.json until the whole job finishes
    # or dies, leaving a killed worker's finished shots
    # unrecorded. Best effort, like the run-open and final
    # writes - a step that saved its files has succeeded whether
    # or not this lands
    if owns_run_dir(workflow):
        write_run_manifest(workflow, record, "running")
    return remaining_refs
