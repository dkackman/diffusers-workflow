# Core functionality for loading and executing workflows
import os
import json
import torch
import copy
import gc
import hashlib
import logging
from datetime import datetime, timezone
from .arguments import realize_args, realize_constants
from .events import (
    RunContext,
    emit_phase,
    WorkflowCancelled,
    get_context,
    current_context,
    activate_context,
    deactivate_context,
)
from .step import Step
from .step_cache import (
    step_cache,
    referenced_result_names,
    reference_resolves_to,
)
from .runs import (
    FLAT_LAYOUT,
    workflow_identity,
    manifest_relative_files,
    new_run_id,
    output_layout,
    run_directory,
    write_manifest,
)
from .schema import validate_data, load_schema
from .variables import replace_variables, set_variables
from .pipeline_processors.pipeline import Pipeline
from .tasks.model_cache import clear_model_cache
from .tasks.task import Task
from . import get_device, empty_device_cache
from .security import (
    validate_path,
    validate_workflow_path,
    validate_json_size,
    validate_output_path,
    SecurityError,
    PathTraversalError,
    InvalidInputError,
)

logger = logging.getLogger("dw")


def workflow_from_file(file_spec, output_dir, workflow_dir=None):
    """Loads a workflow from a JSON file with security validation.

    workflow_dir, when given, confines file_spec (and, via the returned
    Workflow, any sub-workflow steps it references) to that directory - the
    server passes its configured workflow_dir so a caller cannot escape it
    via an inline workflow's base_dir or a sub-workflow step's path. CLI/REPL
    callers leave it None: a locally-run workflow file is not a trust
    boundary.
    """
    logger.debug(f"Loading workflow from file: {file_spec}")

    try:
        # Validate file path and size
        validated_path = validate_workflow_path(file_spec, workflow_dir)
        validate_json_size(validated_path)
        validated_output = validate_output_path(output_dir, None)

        with open(validated_path, "r") as file:
            workflow_data = json.load(file)

        return Workflow(workflow_data, validated_output, validated_path, workflow_dir)

    except SecurityError as e:
        logger.error(f"Security validation failed for workflow {file_spec}: {e}")
        raise
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load workflow from {file_spec}: {e}")
        raise


def workflow_from_definition(
    workflow_definition, output_dir, base_dir=None, workflow_dir=None
):
    """A Workflow from an inline definition (no file on disk).

    The synthetic '__inline__.json' file_spec exists only to carry the
    directory that relative paths inside the definition resolve against.
    base_dir is caller-supplied (over HTTP, client-supplied) path-shaped
    input, so it goes through the security validator like every other path -
    confined to workflow_dir when the caller gives one, same as file_spec in
    workflow_from_file, so a client cannot point an inline workflow's assets
    (or a sub-workflow step it defines) anywhere on disk.
    """
    validated_output = validate_output_path(output_dir, None)
    if base_dir:
        validated_base = validate_path(base_dir, workflow_dir, allow_create=False)
        if not os.path.isdir(validated_base):
            raise InvalidInputError(f"base_dir is not a directory: {base_dir}")
    else:
        # A confined run without a base_dir rests at the boundary itself, so
        # the worker's re-validation of the stored base_dir agrees with this one
        validated_base = os.path.abspath(workflow_dir) if workflow_dir else os.getcwd()
    return Workflow(
        workflow_definition,
        validated_output,
        os.path.join(validated_base, "__inline__.json"),
        workflow_dir,
    )


def workflow_output_subfolder(file_spec):
    """The subfolder a workflow's outputs land in, mirroring its position
    under the nearest directory literally named 'workflows' in its path.

    'workflows/ltx/Foo.json' -> 'ltx'; 'workflows/Foo.json' (or a builtin,
    always dw/workflows/<name>.json) -> '' (flat, no spurious subfolder);
    a path with no 'workflows' segment at all (an inline definition's
    synthetic file_spec, say) -> '' as a fallback. The *last* 'workflows'
    segment wins, matching the packaged dw/workflows tree when a checkout
    also has a top-level workflows/ directory somewhere in its ancestry.
    """
    if not file_spec:
        return ""

    directory = os.path.dirname(os.path.abspath(file_spec))
    parts = os.path.normpath(directory).split(os.sep)
    try:
        index = len(parts) - 1 - parts[::-1].index("workflows")
    except ValueError:
        return ""

    return os.path.join(*parts[index + 1 :]) if index + 1 < len(parts) else ""


def pipeline_cache_key(pipeline_definition):
    """Stable identity for a loaded pipeline.

    Hashes everything that shapes loading - configuration, components,
    quantization, loras - and excludes what varies per call (arguments, seed,
    chain), so a cache hit means "this exact model stack is already loaded".
    Keying the cache by identity instead of step name means two workflows
    whose steps happen to share a name can no longer collide, and a rerun of
    an edited workflow keeps every pipeline whose definition did not change.

    Computed after variable substitution but the excluded keys keep realized
    per-run values (images, generators) out of the hash; realized types and
    dtypes stringify stably via default=str.
    """
    load_definition = {
        k: v
        for k, v in pipeline_definition.items()
        if k not in ("arguments", "seed", "chain")
    }
    serialized = json.dumps(load_definition, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


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


class Workflow:
    """
    Main class for managing and executing workflows defined in JSON format
    Handles variable substitution, step execution, and result management
    """

    # Whether the workflow that delegated to this one had the step cache
    # enabled. A top-level run has no parent and decides for itself; a
    # sub-workflow's parent overwrites this in create_step_action
    _cache_enabled_by_parent = True
    # What run() decided for the run in progress, read by create_step_action
    # to hand down to a sub-workflow
    _cache_enabled_this_run = True

    # The directory the run in progress writes into, set by run() and, for a
    # sub-workflow, handed down by the parent - one execution is one
    # directory, whichever workflow inside it did the writing. None in the
    # flat layout, and before a run starts
    _run_dir = None
    # Whether that directory came from a parent workflow. A sub-workflow is
    # part of the parent's execution: it writes into the same directory and
    # leaves no manifest of its own, since its steps are already rolled up
    # into the parent's
    _run_dir_inherited = False

    def __init__(self, workflow_definition, output_dir, file_spec, workflow_dir=None):
        self.workflow_definition = workflow_definition
        self.output_dir = output_dir
        self.file_spec = file_spec
        # Confines sub-workflow step resolution (below) when set - the
        # server passes its configured workflow_dir; CLI/REPL callers leave
        # it None since a locally-run workflow is not a trust boundary
        self.workflow_dir = workflow_dir

    @property
    def name(self):
        return self.workflow_definition.get("id", "unknown")

    @property
    def argument_template(self):
        return self.workflow_definition.get("argument_template", {})

    @property
    def variables(self):
        return self.workflow_definition.get("variables", {})

    def step_file_prefix(self, step_name):
        """Naming prefix for files a step writes on its own (chain segment
        spills), matching the workflow-id-step naming its results are saved
        under."""
        return f"{self.name}-{step_name}"

    @property
    def effective_output_dir(self):
        """Where this workflow's own results are written.

        In the default layout that is the run directory run() opened -
        '<output_dir>/<identity>/<run id>/' - shared by every step of the
        run, sub-workflows included, so one execution leaves one directory.

        In the flat layout it is output_dir plus a subfolder mirroring the
        workflow file's position under a 'workflows' directory, if it has
        one: a workflow at 'workflows/ltx/Foo.json' writes under
        '<output_dir>/ltx/'; one directly inside a 'workflows' folder (or a
        builtin, which always resolves to dw/workflows/<name>.json) writes
        flat at '<output_dir>/', same as one outside any 'workflows' tree
        entirely. self.output_dir itself always stays the plain root - this
        is derived fresh from it every time, so a flat-layout sub-workflow
        computes its own subfolder from its own file, not the parent's.
        """
        if self._run_dir:
            return self._run_dir
        subfolder = workflow_output_subfolder(self.file_spec)
        return (
            os.path.join(self.output_dir, subfolder) if subfolder else self.output_dir
        )

    def validate(self):
        """Validates workflow definition against JSON schema"""
        logger.debug(f"Validating workflow: {self.name}")
        status, message = validate_data(
            self.workflow_definition, load_schema("workflow")
        )
        if not status:
            # message already carries the 'Validation error at <path>:' prefix
            logger.error(message)
            raise Exception(message)
        logger.debug(f"Workflow {self.name} validated successfully")

    def run(
        self, arguments, previous_pipelines=None, context=None, prior_step_keys=None
    ):
        """
        Executes the workflow by:
        1. Processing variables
        2. Setting up random seed
        3. Running each step in sequence
        4. Managing results between steps

        An explicit RunContext receives progress events and can cancel the
        run; without one, the ambient context is reused (a sub-workflow
        reports into its parent's run) or a no-op context is created.
        Saved file paths accumulate in self.manifest, one entry per step.
        """
        run_context = context or current_context() or RunContext()
        context_token = activate_context(run_context)
        # Step name -> cache key for this run, so release_pipeline and
        # pipeline_reference still address pipelines by the step that made them
        self._pipeline_keys_by_step = {}
        # Last run's step->key map: a redefined step's old model is evicted
        # BEFORE its replacement loads, or the transition holds both at once
        self._prior_step_keys = prior_step_keys or {}
        self.manifest = []
        # Overwritten on the way out of the try below - a run that leaves
        # this alone died on an exception the manifest should say so about
        status = "failed"
        run_id = None
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            # CRITICAL: Work on a copy to avoid mutating the original workflow definition
            # This allows the workflow to be run multiple times with different arguments
            workflow_def = copy.deepcopy(self.workflow_definition)

            workflow_id = workflow_def["id"]
            logger.debug(f"Processing workflow: {workflow_id}")

            # File paths in workflows are relative to the workflow file
            base_dir = (
                os.path.dirname(os.path.abspath(self.file_spec))
                if self.file_spec
                else None
            )

            # Handle variable substitution if variables are defined
            variables = workflow_def.get("variables", None)
            if variables is not None:
                logger.debug(f"Setting variables for workflow: {workflow_id}")
                # a constant is the value a variable declares, so it resolves before
                # anything is converted to the type of that declaration
                realize_constants(variables)
                # first set variable values base don the arguments passed to the workflow
                # these may come form the command line or form a parent workflow
                set_variables(arguments, variables)
                # realize the variables, initialiting downloads of images etc
                realize_args(variables, base_dir)
                ## then replace any variable references in the workflow definition with the actual values
                # replace_variables returns a new structure rather than mutating in
                # place, so the result must be captured here
                workflow_def = replace_variables(workflow_def, variables)

            # Set up random seed for reproducibility. Resolved lazily - as a
            # dict.get default, torch.seed() would run on every call and reseed
            # the global RNG even when the workflow names an explicit seed
            default_seed = workflow_def.get("seed")
            # A workflow that names no seed gets a fresh one every run, so no
            # step's cache entry can ever match again - skip the cache
            # wholesale rather than deep-copying every step's realized images
            # and pinning every Result for a hit that cannot happen
            cache_enabled_this_run = (
                self._cache_enabled_by_parent and default_seed is not None
            )
            # create_step_action hands this down to a sub-workflow: it injects
            # the parent's seed into a child that names none, so a child of a
            # seedless parent would otherwise look seeded - and cacheable -
            # while its seed still changes every run
            self._cache_enabled_this_run = cache_enabled_this_run
            if default_seed is None:
                # A fresh generator draws a random seed without touching the
                # global RNG the process may have seeded for reproducibility
                default_seed = torch.Generator().seed()
            workflow_def["seed"] = default_seed

            # One execution, one directory - opened here, after variable
            # substitution and the seed have settled, so the run's identity
            # covers what actually ran rather than what was written down. A
            # sub-workflow inherits the parent's and never opens its own
            started_at = datetime.now(timezone.utc).isoformat()
            run_id = None
            if not self._run_dir_inherited:
                if output_layout() == FLAT_LAYOUT:
                    self._run_dir = None
                else:
                    run_id = new_run_id(
                        {"workflow": workflow_def, "arguments": arguments}
                    )
                    self._run_dir = run_directory(
                        self.output_dir, self.file_spec, workflow_id, run_id
                    )
                    logger.debug(f"Run directory: {self._run_dir}")

            # Initialize collections for sharing state between steps
            results = {}  # Stores results from each step
            shared_components = {}  # Shared resources between steps

            # Use provided pipelines cache or create new dict
            # This allows pipeline reuse across multiple workflow runs
            if previous_pipelines is None:
                pipelines = {}
                logger.debug("Starting with empty pipeline cache")
            else:
                pipelines = previous_pipelines
                logger.debug(f"Reusing pipeline cache with {len(pipelines)} pipelines")

            last_result = None  # Final result is the workflow return value

            # realize any arguments for the steps, i.e. load images etc
            # that are referenced directly in the step
            steps = workflow_def.get("steps", [])

            if not steps:
                logger.warning(f"Workflow {workflow_id} has no steps defined")
                return []

            realize_args(steps, base_dir)

            run_context.emit(
                "workflow_start",
                workflow=workflow_id,
                total_steps=len(steps),
                steps=[step_data["name"] for step_data in steps],
                seed=default_seed,
            )

            # Step name -> whether that step's result this run came from the
            # cache, so a step that reads another step's result can tell
            # whether its own inputs are still all cache-fresh
            hits_this_run = set()

            # Execute each step in sequence
            for i, step_data in enumerate(steps):
                run_context.check_cancelled()
                logger.debug(f"Running step {i+1}/{len(steps)}: {step_data['name']}")
                run_context.emit(
                    "step_start",
                    workflow=workflow_id,
                    step=step_data["name"],
                    index=i,
                    total_steps=len(steps),
                )

                # Seeds resolve most-specific-first: pipeline > step > workflow
                step_seed = step_data.get("seed", default_seed)

                step = Step(step_data, step_seed, self.workflow_definition)

                # What later steps still read, which decides both whether
                # this step's result has to be kept alive after the step
                # (below, and release_unreferenced_results at the bottom of
                # the loop) and whether a cached entry that kept none can
                # serve this run
                remaining_refs = referenced_result_names(steps[i + 1 :])
                result_needed = i == len(steps) - 1 or any(
                    reference_resolves_to(ref, step_data["name"])
                    for ref in remaining_refs
                )

                # create_step_action (and the pipeline load it triggers)
                # mutates step_data in place - injecting a "generator" key -
                # so the cache must key off a snapshot taken before that
                # happens, and that same snapshot must be reused for the
                # put() below. Caching off the live, later-mutated step_data
                # would make every step's dict keys diverge from a freshly
                # deep-copied future run's step_data, so get() would never
                # match again after the first run.
                # A sub-workflow step is never cacheable: its files roll up
                # from the child's own manifest, which a hit does not rebuild.
                is_cacheable = "workflow" not in step_data and cache_enabled_this_run
                step_data_snapshot = None
                if is_cacheable:
                    try:
                        step_data_snapshot = copy.deepcopy(step_data)
                    except Exception as ex:
                        # A realized argument that cannot be deep-copied (an
                        # open handle, a live model object) just means this
                        # step is not cacheable - never a failed run
                        logger.debug(
                            f"Step '{step.name}' arguments are not copyable "
                            f"({ex}) - skipping the step cache for it"
                        )
                        is_cacheable = False
                cached_result = (
                    step_cache.get(
                        workflow_id,
                        step_data_snapshot,
                        step_seed,
                        hits_this_run,
                        # The root, not this run's directory: a hit reports
                        # the earlier run's files and writes nothing new, so
                        # keying on a directory that is new every run would
                        # mean the cache could never hit again. What the root
                        # still guards is a run redirected somewhere else,
                        # where the earlier files are not what the caller asked for
                        self.output_dir,
                        needs_result=result_needed,
                    )
                    if is_cacheable
                    else None
                )

                # A hit skips the step's work, never its bookkeeping:
                # create_step_action is the only place that touches the
                # step's pipeline (the worker evicts every pipeline a run did
                # not touch), republishes a cached pipeline's
                # shared_components for a later reusing step, and records the
                # step's pipeline key for release_pipeline and
                # pipeline_reference to address it by
                step_action = self.create_step_action(
                    step_data,
                    shared_components,
                    pipelines,
                    step_seed,
                    get_device(),
                )
                reused = cached_result is not None
                if reused:
                    logger.info(f"Step '{step.name}' unchanged - reusing cached result")
                    result = cached_result
                    saved_files = result.saved_files
                    hits_this_run.add(step.name)
                else:
                    result = step.run(results, pipelines, step_action)
                    saved_files = result.save(
                        self.effective_output_dir, f"{workflow_id}-{step.name}.{i}"
                    )
                    if is_cacheable:
                        step_cache.put(
                            workflow_id,
                            step_data_snapshot,
                            step_seed,
                            result,
                            self.output_dir,
                            retain_result=result_needed,
                        )

                last_result = result
                results[step.name] = result
                # 'reused' marks files an earlier run wrote and this one only
                # republished, so nothing downstream (job_for_file, the
                # gallery) credits this run with writing them
                manifest_entry = {"step": step.name, "files": saved_files}
                if reused:
                    manifest_entry["reused"] = True
                self.manifest.append(manifest_entry)
                # A sub-workflow's saves land in the child's manifest - roll
                # them up so job history and the gallery see every file
                if isinstance(step_action, Workflow):
                    self.manifest.extend(getattr(step_action, "manifest", []))
                step_end_data = {"files": saved_files}
                if reused:
                    step_end_data["reused"] = True
                run_context.emit(
                    "step_end",
                    workflow=workflow_id,
                    step=step.name,
                    index=i,
                    total_steps=len(steps),
                    **step_end_data,
                )
                logger.debug(f"Step {step.name} completed with result: {result}")

                # Release results no later step references - saved to disk
                # already, and last_result keeps the workflow's return value
                release_unreferenced_results(results, remaining_refs)

                # A released pipeline frees its memory for later steps - the
                # alternative on a card that cannot hold two models is offloading
                # everything, which taxes every run to survive one transition
                if step_data.get("release_pipeline", False):
                    logger.info(f"Releasing pipeline for step: {step.name}")
                    pipelines.pop(self._pipeline_keys_by_step.get(step.name), None)

                # The loop's own locals are the last references to this step's
                # action and result - a released pipeline would otherwise stay
                # resident through the next step's load, which is exactly when
                # both models would be in memory at once
                step_action = None
                result = None

                # Task models are cached for the life of the process - the cache
                # exists so a step's cartesian product loads its model once, and
                # nothing else evicts it. A prompt-expanding language model
                # feeding a generation step would otherwise hold its weights on
                # the device for the whole run
                if step_data.get("release_models", False):
                    logger.info(f"Releasing task models for step: {step.name}")
                    clear_model_cache()

                # Cleanup between steps (but keep pipelines loaded). Returning
                # cached blocks to the device lets the next step's differently
                # shaped allocations use them
                gc.collect()
                empty_device_cache()

            logger.debug(f"Workflow {workflow_id} completed successfully")
            run_context.emit(
                "workflow_end", workflow=workflow_id, manifest=self.manifest
            )
            # Return only the last step's results for child workflows
            status = "completed"
            return last_result.result_list if last_result is not None else []

        except WorkflowCancelled:
            # The user asked for this - report it without an error traceback
            workflow_id = self.workflow_definition.get("id", "unknown")
            logger.info(f"Workflow {workflow_id} cancelled")
            status = "cancelled"
            raise
        except (SecurityError, PathTraversalError, InvalidInputError) as e:
            # Security validation failures - these should fail fast, without the
            # traceback noise of the general handler
            workflow_id = self.workflow_definition.get("id", "unknown")
            logger.error(f"Security error in workflow {workflow_id}: {e}")
            raise
        except Exception as e:
            # One log line with the full traceback - the step already logged its
            # own context, and every clause here did the same log-and-reraise
            workflow_id = self.workflow_definition.get("id", "unknown")
            logger.error(
                f"{type(e).__name__} in workflow {workflow_id}: {e}", exc_info=True
            )
            raise
        finally:
            # Recorded even for a run that failed part way: the files it did
            # write are on disk either way, and what produced them is exactly
            # what a failed run needs to explain itself
            if self._run_dir and not self._run_dir_inherited:
                self._write_run_manifest(run_id, status, started_at, arguments)
            deactivate_context(context_token)

    def _write_run_manifest(self, run_id, status, started_at, arguments):
        """Leave a record of the run beside the files it wrote.

        A server run is in jobs.sqlite as well, but a CLI run has never been
        recorded anywhere, and a database on one machine cannot describe a
        directory copied to another. Paths are relative to the run directory
        so the directory keeps describing itself wherever it goes.
        """
        from . import __version__

        write_manifest(
            self._run_dir,
            {
                "run_id": run_id,
                "status": status,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "dw_version": __version__,
                "device": str(get_device()),
                "workflow": {
                    "id": self.name,
                    "file": self.file_spec,
                    "identity": workflow_identity(self.file_spec, self.name),
                },
                "seed": self.workflow_definition.get("seed"),
                "arguments": arguments or {},
                "steps": [
                    {
                        **entry,
                        "files": manifest_relative_files(
                            entry.get("files"), self._run_dir
                        ),
                    }
                    for entry in self.manifest
                ],
            },
        )

    def _step_pipeline_key(self, step_name, cache_key):
        """Record which cache key a step's pipeline lives under this run."""
        if not hasattr(self, "_pipeline_keys_by_step"):
            self._pipeline_keys_by_step = {}
        self._pipeline_keys_by_step[step_name] = cache_key

    def create_step_action(
        self,
        step_definition,
        shared_components,
        previous_pipelines,
        default_seed,
        device,
    ):
        """
        Creates the appropriate action object based on step type:
        - Pipeline: Creates new pipeline or reuses cached one
        - Pipeline reference: References existing pipeline
        - Workflow: Loads and validates sub-workflow
        - Task: Creates task object
        """
        # Handle pipeline creation
        if "pipeline" in step_definition:
            step_name = step_definition["name"]

            # Pipelines are cached by what they load, not what step loads them
            cache_key = pipeline_cache_key(step_definition["pipeline"])
            self._step_pipeline_key(step_name, cache_key)
            get_context().touch_pipeline(cache_key)

            # Check if pipeline already loaded in cache (GPU persistence)
            if cache_key in previous_pipelines:
                logger.debug(f"Reusing cached pipeline for step: {step_name}")
                cached_pipeline = previous_pipelines[cache_key]
                # The shared_components dict is fresh every run and only load()
                # fills it - a cache hit must republish or a later step's
                # reused_components finds nothing (impossible under the old
                # whole-file cache, the normal case under identity keys)
                cached_pipeline.publish_shared_components(shared_components)
                # Create new Pipeline wrapper with updated step definition
                # but reuse the loaded model from cache
                new_pipeline_wrapper = Pipeline(
                    step_definition["pipeline"],
                    default_seed,
                    device,
                    cached_pipeline.pipeline,  # Reuse the actual loaded model
                    output_dir=self.effective_output_dir,
                    file_prefix=self.step_file_prefix(step_name),
                )
                # Set up generator with potentially new seed. no_generator is a
                # boolean - only an explicit true disables the generator - and the
                # generator lives on the pipeline's own device, which may override
                # the workflow default (the fresh-load path resolves it the same way)
                if not new_pipeline_wrapper.configuration.get("no_generator", False):
                    logger.debug(
                        "Setting up generator for cached pipeline with new arguments"
                    )
                    new_pipeline_wrapper.argument_template[
                        "generator"
                    ] = torch.Generator(new_pipeline_wrapper.device).manual_seed(
                        new_pipeline_wrapper.pipeline_definition.get(
                            "seed", default_seed
                        )
                    )

                # A cache hit and a cold load look identical from the outside -
                # same step, same dot - and they differ by minutes
                emit_phase("cached", detail=new_pipeline_wrapper.name)
                return new_pipeline_wrapper

            # Not in cache - a redefined step frees its previous model first,
            # so the swap never holds old and new stacks simultaneously
            prior_key = getattr(self, "_prior_step_keys", {}).get(step_name)
            if prior_key and prior_key != cache_key and prior_key in previous_pipelines:
                logger.info(
                    f"Step '{step_name}' was redefined - releasing its previous "
                    "pipeline before loading the new one"
                )
                previous_pipelines.pop(prior_key, None)
                gc.collect()
                empty_device_cache()

            logger.debug(f"Creating pipeline for step: {step_name}")
            pipeline = Pipeline(
                step_definition["pipeline"],
                default_seed,
                device,
                output_dir=self.effective_output_dir,
                file_prefix=self.step_file_prefix(step_name),
            )
            # Loading is the longest silence in a run: weights, quantization,
            # adapters and placement all happen inside this call
            emit_phase("loading", detail=pipeline.name)
            pipeline.load(shared_components)
            previous_pipelines[cache_key] = pipeline
            return pipeline

        # Handle pipeline reference
        if "pipeline_reference" in step_definition:
            logger.debug(
                f"Referencing existing pipeline for step: {step_definition['name']}"
            )
            pipeline_reference = step_definition["pipeline_reference"]
            reference_name = pipeline_reference["reference_name"]
            referenced_key = self._pipeline_keys_by_step.get(reference_name)
            if referenced_key is None or referenced_key not in previous_pipelines:
                raise ValueError(
                    f"pipeline_reference '{reference_name}' does not name an "
                    "earlier pipeline step in this run (or it was released)"
                )
            previous_pipeline = previous_pipelines[referenced_key]
            return Pipeline(
                pipeline_reference,
                default_seed,
                device,
                previous_pipeline.pipeline,
                output_dir=self.effective_output_dir,
                file_prefix=self.step_file_prefix(step_definition["name"]),
            )

        # Handle sub-workflow
        if "workflow" in step_definition:
            logger.debug(f"Loading sub-workflow for step: {step_definition['name']}")
            workflow_reference = step_definition["workflow"]
            path = workflow_reference["path"]

            try:
                # Sub-workflow steps are confined to the same directory this
                # workflow is (workflow_dir for a server-submitted run)
                confine_to = self.workflow_dir
                # Handle built-in workflows
                if path.startswith("builtin:"):
                    builtin_name = path.replace("builtin:", "")
                    # Validate builtin workflow name
                    if (
                        not builtin_name.endswith(".json")
                        or "/" in builtin_name
                        or "\\" in builtin_name
                    ):
                        raise InvalidInputError(
                            f"Invalid builtin workflow name: {builtin_name}"
                        )
                    # Builtins ship inside the package, outside any
                    # workflow_dir - confine them to their own directory
                    # instead (the name check above already forbids escaping it)
                    confine_to = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "workflows"
                    )
                    path = os.path.join(confine_to, builtin_name)
                # Handle relative paths
                elif not os.path.isabs(path):
                    base_dir = os.path.dirname(self.file_spec)
                    path = os.path.join(base_dir, path)

                # Validate the resolved path - confined when this workflow
                # itself is (an inline/server-submitted run), so a
                # sub-workflow step cannot escape that boundary
                validated_path = validate_workflow_path(path, confine_to)
                workflow = workflow_from_file(
                    validated_path, self.output_dir, confine_to
                )

            except SecurityError as e:
                logger.error(f"Security validation failed for sub-workflow {path}: {e}")
                raise

            # this is where the arguments in the paretn script are passed to the child workflow
            # they will already be populated with values from previous steps or parent variables
            workflow.workflow_definition["argument_template"] = workflow_reference.get(
                "arguments", {}
            )
            # A child left to itself draws its own random seed, which makes the
            # parent's seed stop short of the work it delegates. Inheriting it
            # keeps one seed reproducing the whole run; a child that names its
            # own still wins, the same way a step overrides its workflow
            workflow.workflow_definition.setdefault("seed", default_seed)
            # A seedless parent injects a seed that is fresh every run, so no
            # step of the child can ever hit - the child must not pay the
            # cache's deepcopy and Result pinning for it
            workflow._cache_enabled_by_parent = self._cache_enabled_this_run
            # One execution, one directory: the child writes into the
            # parent's run directory and leaves no manifest of its own - its
            # steps roll up into the parent's manifest already
            workflow._run_dir = self._run_dir
            workflow._run_dir_inherited = self._run_dir is not None
            workflow.validate()
            return workflow

        logger.debug(f"Creating task for step: {step_definition['name']}")
        # Handle task creation
        task_definition = step_definition["task"]
        task = Task(task_definition, device)
        return task
