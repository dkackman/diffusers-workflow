# Core functionality for loading and executing workflows
import os
import json
import copy
import logging
from .arguments import realize_constants, fetch_constant, is_constant_reference
from .events import (
    RunContext,
    WorkflowCancelled,
    get_context,
    current_context,
    activate_context,
    deactivate_context,
)
from .variable_constraints import resolve_constraint_references, snap_constraints
from .subfolders import step_subfolder
from . import validation
from . import workflow_run
from .step_cache import borrowed_pipeline_keys
from .runs import activate_output_root, deactivate_output_root
from .schema import format_validation_errors
from .for_each import expand_for_each
from .variables import (
    ConstantError,
    argument_errors,
    replace_variables,
    resolve_variable_values,
    set_variables,
)
from .pipeline_processors.pipeline import Pipeline
from .tasks.task import Task
from . import get_device
from .pipeline_ownership import (
    PipelineOwnership,
    allocated_mb,
    evict_superseded,
    finish_release,
    load_fresh,
    reclaim_after_step,
    wrap_resident,
)
from .security import (
    validate_path,
    validate_workflow_path,
    validate_json_size,
    validate_output_path,
    SecurityError,
    PathTraversalError,
    InvalidInputError,
    UntrustedWorkflowError,
)
from .library import (
    resolve_sub_workflow_reference,
    workflow_output_subfolder,
)

logger = logging.getLogger("dw")


def workflow_from_file(file_spec, output_dir, workflow_dir=None):
    """Loads a workflow from a JSON file with security validation.

    workflow_dir, when given, confines file_spec (and, via the returned
    Workflow, any sub-workflow steps it references) to that directory - the
    server passes its configured workflow_dir so a caller cannot escape it
    via an inline workflow's base_dir or a sub-workflow step's path. CLI
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
        # the worker's confinement of the file_spec agrees with this one
        validated_base = os.path.abspath(workflow_dir) if workflow_dir else os.getcwd()
    return Workflow(
        workflow_definition,
        validated_output,
        os.path.join(validated_base, "__inline__.json"),
        workflow_dir,
    )


def workflow_from_snapshot(definition, output_dir, file_spec, workflow_dir=None):
    """The Workflow admission built, rebuilt from what it recorded - the
    definition it checked and the file_spec it resolved - without opening
    file_spec. A job runs the definition that was admitted, however the file
    has changed (or vanished) since; file_spec still decides the run's
    identity, its output subfolder and where sub-workflow steps resolve.

    The caller must pass an already-validated file_spec: the path
    workflow_from_file stored, or an inline definition's synthetic
    '__inline__.json' - admission normalizes it. It is passed through as it
    is so every name derived from it matches admission's. workflow_dir,
    when given, confines it again here: the command crossed a process
    boundary, and a file_spec outside the root is refused as
    workflow_from_file refuses the path. With workflow_dir None file_spec
    is made absolute but not confined.
    """
    validated_output = validate_output_path(output_dir, None)
    if workflow_dir:
        validate_path(os.path.dirname(file_spec), workflow_dir)
    else:
        file_spec = os.path.abspath(file_spec)
    return Workflow(definition, validated_output, file_spec, workflow_dir)


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
    # That directory's ordinal among this workflow's runs - what the gallery
    # shows as 'v4'. None in the flat layout, for a sub-workflow (which is
    # part of the parent's run, not a run of its own), and before a run
    # starts
    _run_version = None
    # Where the parent step that delegated to this workflow sits in the
    # run the caller queued: {"step", "index", "total_steps"}. A child
    # counts its own steps from zero, so without this a composed run
    # reported "step 1 of 1" from inside the first of the parent's three
    # (#90) - and "is this nearly finished" is the whole question progress
    # answers. Handed straight down to a grandchild, so the numbers always
    # describe the run that was queued
    _parent_progress = None
    # Whether the parent step that composed this workflow declares a
    # `result` of its own. It does the saving then, and this run's last step
    # does not: the two used to write the same artifact twice, once under
    # the parent step's name and subfolder and once under the child's, with
    # the child's entry shadowing a manifest key the caller never wrote
    # (#92). A child whose parent declares nothing still saves, since
    # otherwise the output would exist nowhere
    _final_save_owned_by_parent = False
    # Whether this workflow was composed by a parent's `workflow` step, whose
    # arguments hand down the parent's own objects - a previous_result:
    # artifact, a realized image. The child takes one copy of them on entry
    # (run, below), so a child step that writes onto its artifact in place -
    # conform_artifact stamping a declared fps onto what `select` handed back
    # by identity - edits the child's copy and never the parent's result.
    # Past that boundary nothing copies a leaf: variables are resolved,
    # substituted and recorded sharing it
    _composed = False
    # The arguments the parent step that composed this workflow hands it -
    # the parent's realized objects, read by Step.run through
    # argument_template. Kept here rather than written into
    # workflow_definition, where every validate() and run() of the child
    # deep-copied them again; the child's one copy is _owned_arguments, on
    # entry. None for a workflow nothing composed
    _handed_arguments = None

    def __init__(self, workflow_definition, output_dir, file_spec, workflow_dir=None):
        self.workflow_definition = workflow_definition
        self.output_dir = output_dir
        self.file_spec = file_spec
        # Confines sub-workflow step resolution (below) when set - the
        # server passes its configured workflow_dir; CLI callers leave
        # it None since a locally-run workflow is not a trust boundary
        self.workflow_dir = workflow_dir
        # expanded_definition's memo, keyed by the caller's arguments
        self._expansions = {}
        # Which pipeline each step owns. run() replaces it per run; this one
        # serves a step action created outside run()
        self.pipeline_ownership = PipelineOwnership()

    @property
    def name(self):
        return self.workflow_definition.get("id", "unknown")

    @property
    def argument_template(self):
        if self._handed_arguments is not None:
            return self._handed_arguments
        return self.workflow_definition.get("argument_template", {})

    @property
    def variables(self):
        return self.workflow_definition.get("variables", {})

    def step_save_name(self, workflow_id, step_name, index):
        """The base name a step's files are written under.

        Inside a composed child the parent step's name leads, so two steps
        composing the same workflow do not both want one name and get told
        apart by a '-2' suffix that says nothing about which step made it
        (#92).
        """
        base = f"{workflow_id}-{step_name}.{index}"
        parent = self._parent_progress
        return f"{parent['step']}.{base}" if parent else base

    def _parent_progress_fields(self):
        """The queued run's own step counter, on an event a sub-workflow
        emits - empty for a top-level run, whose index is already that."""
        parent = self._parent_progress
        if not parent:
            return {}
        return {
            "parent_step": parent["step"],
            "parent_index": parent["index"],
            "parent_total_steps": parent["total_steps"],
        }

    def step_file_prefix(self, step_name):
        """Naming prefix for files a step writes on its own (chain segment
        spills), matching the workflow-id-step naming its results are saved
        under."""
        prefix = f"{self.name}-{step_name}"
        parent = self._parent_progress
        return f"{parent['step']}.{prefix}" if parent else prefix

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

    def step_output_dir(self, step_definition):
        """Where one step writes: the run directory, or the subfolder of it
        the step's result names.

        Computed here, once, rather than inside Result.save, because two
        things write on a step's behalf - Result.save for its results and
        the pipeline wrapper for a chain's save_segments spill - and both
        have to land in the same place. The shape was checked statically by
        validation_errors; it is checked again here for a definition that
        reached the engine without it, and containment (that the joined
        path is really inside the run directory) is checked on the join.
        """
        base = self.effective_output_dir
        subfolder = step_subfolder(step_definition)
        if not subfolder:
            return base
        target = validate_output_path(os.path.join(base, subfolder), base)
        os.makedirs(target, exist_ok=True)
        return target

    def _fold(self, definition, arguments, *, fold_arguments, constrain):
        """Stage one of preparing a definition, shared by validation, the
        run and the record: constants realized, the caller's `arguments`
        folded in when `fold_arguments`, list entries' own references
        resolved, and `constrain` applied - snap_constraints for validation,
        apply_constraints (which also warns and refuses) for the run.

        Mutates `definition`, whose 'variables' block ends up holding the
        folded values, and returns them - None when it declares none.

        Raises ConstantError for a 'constant:' variable default that fails
        to resolve, naming the variable.
        """
        variables = definition.get("variables")
        if not isinstance(variables, dict):
            return None
        # a constant is the value a variable declares, so it resolves before
        # anything is converted to the type of that declaration - and a list
        # defaulted to a 'constant:' name must expand in validation as it
        # does in the run: a name lookup, no download. Realizing a constant
        # imports the module it names, so validating one runs the same
        # trust gate (require_trusted_dotted_name) a run would - only the
        # diffusers ecosystem allowlist, unless the caller trusts the
        # workflow. Realized per top-level variable, not as one call over
        # the whole dict, so a failure names the variable.
        for name, value in variables.items():
            try:
                if is_constant_reference(value):
                    variables[name] = fetch_constant(value)
                else:
                    realize_constants(value)
            except (ValueError, InvalidInputError, UntrustedWorkflowError) as e:
                raise ConstantError(f"variables.{name}", str(e)) from e
        if fold_arguments:
            # set variable values from the arguments passed to the workflow;
            # these may come from the command line or from a parent workflow
            set_variables(arguments, variables)
        # an entry of a list-valued variable may name another variable;
        # resolve those before anything inside it is realized, so a
        # reference type in an entry is a type name - and before the
        # constraints pass, so an entry written as "variable:tail_len" is a
        # number by the time the rule looks
        variables = resolve_variable_values(variables)
        # A value outside a rule the workflow declares is refused (by the
        # run), and one the rule rounds is rounded - before anything loads,
        # and before substitution puts the value everywhere it is
        # referenced, so validation checks what the run will use
        # (dw/variable_constraints.py, #96)
        constrain(definition, variables)
        definition["variables"] = variables
        return variables

    @staticmethod
    def _expand(definition, variables, source_indices=None):
        """Stage two of preparing a definition: every 'variable:'
        substituted, every 'constraint:' frame_snap resolved and every
        for_each expanded. Returns the new definition."""
        if variables is not None:
            # replace_variables returns a new structure rather than mutating
            # in place, so the result must be captured here
            definition = replace_variables(definition, variables)
        # A chain step's `frame_snap` may name the declared constraint
        # rather than repeating its numbers, so a template states the rule
        # once (#96)
        resolve_constraint_references(definition)
        # One ordinary step per entry of every for_each list, before the
        # seed, the run id and the realized workflow are computed, so each
        # covers what actually runs. A ForEachError here fails the run
        # before anything loads
        return expand_for_each(definition, source_indices)

    def _folded_and_expanded(self, arguments):
        """(expanded definition, its source indices, folded variables) for
        `arguments`, computed once per workflow and arguments - validation
        and its six warning passes all ask for the same expansion. An
        exception is raised again on the next call rather than cached."""
        # None (check the document) and {} fold exactly the same - no
        # arguments to fold either way - so they share one entry
        key = json.dumps(arguments or {}, sort_keys=True, default=repr)
        cache = self._expansions
        if key not in cache:
            definition = copy.deepcopy(self.workflow_definition)
            variables = self._fold(
                definition,
                arguments,
                fold_arguments=bool(arguments)
                and not argument_errors(definition, arguments),
                constrain=snap_constraints,
            )
            source_indices = []
            expanded = self._expand(definition, variables, source_indices)
            cache[key] = (expanded, source_indices, variables)
        return cache[key]

    def expanded_definition(self, arguments=None, source_indices=None):
        """The definition as the run will see it: constants realized,
        variables substituted - the caller's `arguments` folded in when they
        are all good, else the declared defaults - 'constraint:' names
        resolved, and every for_each step expanded.

        Raises ForEachError for a for_each that cannot be expanded,
        ConstantError for a 'constant:' variable default that fails to
        resolve, and VariableNotFoundError for a 'variable:' that names
        nothing - which is exactly what the run itself would raise, since a
        definition that declares variables is always substituted before it
        runs.

        `source_indices`, when a list is passed, comes back holding the
        index in *this* definition's steps of every expanded step, so an
        error can be reported at a path in the file the author wrote.

        Memoized per arguments; each call returns its own copy.
        """
        expanded, indices, _ = self._folded_and_expanded(arguments)
        if source_indices is not None:
            source_indices.extend(indices)
        return copy.deepcopy(expanded)

    def folded_variables(self, arguments=None):
        """The variables as expanded_definition folded them - what the
        realized workflow records - or None when none are declared."""
        return copy.deepcopy(self._folded_and_expanded(arguments)[2])

    def resolve_sub_workflow_path(self, path):
        """Where one sub-workflow step's `path` resolves to, as
        (path, root) - the same resolution create_step_action does, asked
        ahead of the run so validation can answer for free what used to cost
        a queued job to find out (#89).

        Raises SubWorkflowNotFound, SecurityError or InvalidInputError,
        each carrying the message the run would have failed with.
        """
        return resolve_sub_workflow_reference(
            path, os.path.dirname(self.file_spec), self.workflow_dir
        )

    def open_sub_workflow(self, path, resolved=None):
        """The workflow one sub-workflow step's `path` names, opened as
        (child, resolved) - resolved is its path, which is what a
        composition chain records. How dw.validation builds a composed child
        without importing this module. `resolved` is what
        `resolve_sub_workflow_path` already answered for `path`, as
        (path, root); given, it is opened without resolving again. Raises
        what resolution or the load raises."""
        resolved, root = resolved or self.resolve_sub_workflow_path(path)
        return workflow_from_file(resolved, self.output_dir, root), resolved

    def validation_errors(self, arguments=None, composing=None, *, context=None):
        """Every schema violation in the definition, as [{path, message}];
        empty when it validates. `arguments` are the caller's, so a
        for_each over a list the caller supplies is checked as it will run.

        `composing` carries the chain of sub-workflows above this one, so a
        workflow that composes itself is an error rather than a recursion.

        `context`, when given, is the request's own ValidationContext - its
        arguments and composing chain are the ones checked - so one request
        expands and probes once. Passing a context together with different
        `arguments` or `composing` raises ValueError
        (dw.validation's `workflow_errors`).
        """
        return validation.workflow_errors(self, arguments, composing, context)

    def validate(self, arguments=None):
        """Validates workflow definition against JSON schema.

        Every violation is reported, one per line, so the CLI
        and an agent iterating on a draft fix them in one pass rather than
        one per round trip. ``arguments``, when given, are folded in before
        checking - a caller's override (e.g. a content_type-driving variable)
        must be judged as it will actually run, not against the document's
        unsubstituted defaults.
        """
        logger.debug(f"Validating workflow: {self.name}")
        errors = self.validation_errors(arguments=arguments)
        if errors:
            # message already carries the 'Validation error' prefix
            message = format_validation_errors(errors)
            logger.error(message)
            raise Exception(message)
        logger.debug(f"Workflow {self.name} validated successfully")

    def _owned_arguments(self, arguments):
        """The composed child's own copy of what its parent handed it
        (_composed) - only of the names its fold keeps, the ones its
        `variables` block declares. An undeclared name is left for
        set_variables to refuse by name, and a child declaring nothing
        ignores its arguments, so neither is worth a copy that could fail."""
        declared = self.workflow_definition.get("variables")
        if not isinstance(declared, dict) or not isinstance(arguments, dict):
            return arguments
        return {
            name: copy.deepcopy(value) if name in declared else value
            for name, value in arguments.items()
        }

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
        The phases are dw.workflow_run's.
        """
        run_context = context or current_context() or RunContext()
        context_token = activate_context(run_context)
        # Depth-counted: a sub-workflow shares its parent's RunContext, so the
        # phase-stall watchdog starts once on the outermost run() and stops
        # once that outermost call's finally below runs, not on every nested
        # sub-workflow call
        run_context.enter_run()
        # 'output:' references resolve against the directory this run was
        # told to write to - the root, not this run's own subdirectory, since
        # what they name is what an earlier run left there
        output_root_token = activate_output_root(self.output_dir)
        # This run's step->pipeline tables, fresh per run: a persistent
        # worker reuses this Workflow across jobs, and what one run recorded
        # or deferred says nothing about what the next has resident. Its key
        # table is set before the first step (begin_steps), so a reused
        # Workflow never serves load_key last run's table
        self.pipeline_ownership = PipelineOwnership(prior_step_keys)
        self.manifest = []
        # What elision dropped this run, filled by prepare_definition and
        # read by the warning pass and the manifest (#122)
        self._elided_steps = []
        # A reused Workflow (the persistent worker's) still holds the last
        # run's directory: a run that fails before open_run would otherwise
        # rewrite that run's manifest. A composed child's values were set by
        # its parent just before this call, so they stay
        if not self._run_dir_inherited:
            self._run_dir = None
            self._run_version = None
        record = workflow_run.RunRecord(arguments)
        try:
            prepared = workflow_run.prepare_run(self, record)
            workflow_run.open_run(self, prepared, record, run_context)
            loop = workflow_run.begin_steps(
                self, prepared, previous_pipelines, run_context
            )
            if loop is None:
                record.status = "completed"
                return []
            for index, step_data in enumerate(loop.steps):
                remaining_refs = workflow_run.run_step(
                    self, loop, index, step_data, record
                )
                # Release results no later step references - saved to disk
                # already, and last_result keeps the workflow's return value
                workflow_run.release_unreferenced_results(loop.results, remaining_refs)
                # Task models the step asked to release, then the cleanup
                # between steps (pipelines stay loaded)
                reclaim_after_step(step_data, step_data["name"])

            logger.debug(f"Workflow {loop.workflow_id} completed successfully")
            run_context.emit(
                "workflow_end", workflow=loop.workflow_id, manifest=self.manifest
            )
            # Return only the last step's results for child workflows
            record.status = "completed"
            last_result = loop.last_result
            return last_result.result_list if last_result is not None else []

        except WorkflowCancelled:
            # The user asked for this - report it without an error traceback
            workflow_id = self.workflow_definition.get("id", "unknown")
            logger.info(f"Workflow {workflow_id} cancelled")
            record.status = "cancelled"
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
            if workflow_run.owns_run_dir(self):
                workflow_run.write_run_manifest(self, record)
            deactivate_output_root(output_root_token)
            run_context.exit_run()
            deactivate_context(context_token)

    def _load_deferred_borrows(
        self, workflow_id, steps, index, shared_components, pipelines
    ):
        """Load, in step order, every deferred pipeline step `index` borrows.

        Through create_step_action's cold path, so a lazy load is a load like
        any other: the loading phase, the touch, the superseded-key release
        and the shared_components publish all happen as they would have at
        the deferred step. A deferred step's own deferred borrows load first,
        since its load resolves the components they share. A released entry
        is loaded for its components only, then dropped again - what a cold
        load followed by its release leaves behind.
        """
        step = steps[index]
        ownership = self.pipeline_ownership
        running_keys = ownership.running or {}
        own_key = running_keys.get(step["name"])
        if own_key is not None and own_key in pipelines:
            # Reused components are resolved only inside load(), and a
            # resident pipeline does not load
            return
        deferred = ownership.deferred
        borrowed = borrowed_pipeline_keys(steps, index, running_keys)
        reference = step.get("pipeline_reference")
        referenced_name = (
            reference.get("reference_name") if isinstance(reference, dict) else None
        )
        # Ascending, so a source's own earlier borrows never come later in
        # this list
        for source_index, source in enumerate(steps[:index]):
            name = source.get("name")
            if name not in borrowed or name not in deferred:
                continue
            if name == referenced_name and deferred[name].released:
                # A released pipeline cannot be referenced; loading it only
                # to drop it would delay the error the reference raises
                continue
            self._load_deferred_borrows(
                workflow_id, steps, source_index, shared_components, pipelines
            )
            entry = deferred.pop(name)
            self.create_step_action(
                entry.step_data,
                shared_components,
                pipelines,
                entry.seed,
                get_device(),
            )
            if entry.released:
                # The release the step asked for, happening now: freed and
                # announced as its own release would have been, so the
                # borrower does not load on top of it
                before = allocated_mb()
                pipelines.pop(ownership.key_for(name), None)
                finish_release(workflow_id, name, source_index, before)

    def create_step_action(
        self,
        step_definition,
        shared_components,
        previous_pipelines,
        default_seed,
        device,
        cache_hit=False,
    ):
        """
        Creates the appropriate action object based on step type:
        - Pipeline: Creates new pipeline or reuses cached one. With
          cache_hit (a step-cache hit), a pipeline that is not already
          resident is not loaded: its key is recorded and touched, the step
          is deferred for a later step that runs to load, and None is
          returned
        - Pipeline reference: References existing pipeline; on a cache hit
          whose referenced pipeline is not resident, None
        - Workflow: Loads and validates sub-workflow
        - Task: Creates task object
        """
        if "pipeline" in step_definition:
            return self._pipeline_action(
                step_definition,
                shared_components,
                previous_pipelines,
                default_seed,
                device,
                cache_hit,
            )
        if "pipeline_reference" in step_definition:
            return self._reference_action(
                step_definition, previous_pipelines, default_seed, device, cache_hit
            )
        if "workflow" in step_definition:
            return self._sub_workflow_action(step_definition, default_seed)

        logger.debug(f"Creating task for step: {step_definition['name']}")
        task_definition = step_definition["task"]
        return Task(task_definition, device, seed=default_seed)

    def _pipeline_action(
        self,
        step_definition,
        shared_components,
        previous_pipelines,
        default_seed,
        device,
        cache_hit,
    ):
        """A pipeline step's action: the resident pipeline rewrapped, a
        fresh load, or None for a hit whose pipeline is not resident."""
        step_name = step_definition["name"]

        # Pipelines are cached by what they load, not what step loads them
        ownership = self.pipeline_ownership
        cache_key = ownership.load_key(step_definition)
        ownership.record(step_name, cache_key)
        get_context().touch_pipeline(cache_key)

        if cache_hit and cache_key not in previous_pipelines:
            # A hit needs the key recorded (release_pipeline and
            # pipeline_reference address it by name), not the weights:
            # nothing calls the pipeline unless a step that runs borrows
            # it, and that step loads it then
            ownership.defer(step_name, step_definition, default_seed)
            return None

        # Check if pipeline already loaded in cache (GPU persistence)
        if cache_key in previous_pipelines:
            logger.debug(f"Reusing cached pipeline for step: {step_name}")
            # The shared_components dict is fresh every run and only load()
            # fills it - a cache hit must republish, before anything is
            # computed for the step's own output
            previous_pipelines[cache_key].publish_shared_components(shared_components)
            return wrap_resident(
                previous_pipelines[cache_key],
                step_definition,
                default_seed,
                device,
                self.step_output_dir(step_definition),
                self.step_file_prefix(step_name),
            )

        # Not in cache - a redefined step frees its previous model first,
        # so the swap never holds old and new stacks simultaneously
        prior_key = ownership.superseded_key(step_name, cache_key, previous_pipelines)
        if prior_key is not None:
            evict_superseded(previous_pipelines, step_name, prior_key)

        logger.debug(f"Creating pipeline for step: {step_name}")
        pipeline = load_fresh(
            step_definition,
            shared_components,
            default_seed,
            device,
            self.step_output_dir(step_definition),
            self.step_file_prefix(step_name),
        )
        previous_pipelines[cache_key] = pipeline
        return pipeline

    def _reference_action(
        self, step_definition, previous_pipelines, default_seed, device, cache_hit
    ):
        """A pipeline_reference step's action over an earlier step's
        pipeline; None on a hit whose referenced step deferred its load."""
        logger.debug(
            f"Referencing existing pipeline for step: {step_definition['name']}"
        )
        pipeline_reference = step_definition["pipeline_reference"]
        reference_name = pipeline_reference["reference_name"]
        referenced_key = self.pipeline_ownership.key_for(reference_name)
        if cache_hit and reference_name in self.pipeline_ownership.deferred:
            # The referenced step was a hit that deferred its load, and
            # this step's result is cached too: nothing will call it
            return None
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
            output_dir=self.step_output_dir(step_definition),
            file_prefix=self.step_file_prefix(step_definition["name"]),
        )

    def _sub_workflow_action(self, step_definition, default_seed):
        """The composed child Workflow a `workflow` step runs, resolved,
        confined and validated."""
        logger.debug(f"Loading sub-workflow for step: {step_definition['name']}")
        workflow_reference = step_definition["workflow"]
        path = workflow_reference["path"]

        try:
            # Sub-workflow steps are confined to the same directory this
            # workflow is (workflow_dir for a server-submitted run); the
            # resolver confines a builtin to the packaged root instead and
            # validates the path it hands back
            validated_path, confine_to = self.resolve_sub_workflow_path(path)
            workflow = workflow_from_file(validated_path, self.output_dir, confine_to)

        except SecurityError as e:
            logger.error(f"Security validation failed for sub-workflow {path}: {e}")
            raise

        # this is where the arguments in the parent script are passed to the child workflow
        # they will already be populated with values from previous steps or parent variables
        workflow._handed_arguments = workflow_reference.get("arguments", {})
        # A child left to itself draws its own random seed, which makes the
        # parent's seed stop short of the work it delegates. Inheriting it
        # keeps one seed reproducing the whole run; a child that names its
        # own still wins, the same way a step overrides its workflow
        workflow.workflow_definition.setdefault("seed", default_seed)
        # A seedless parent injects a seed that is fresh every run, so no
        # step of the child can ever hit - the child must not pay the
        # cache's deepcopy and Result pinning for it
        workflow._cache_enabled_by_parent = self._cache_enabled_this_run
        # The parent's objects arrive as this child's arguments; the child
        # copies them once on entry rather than editing the parent's
        workflow._composed = True
        # One execution, one directory: the child writes into the
        # parent's run directory and leaves no manifest of its own - its
        # steps roll up into the parent's manifest already
        workflow._run_dir = self._run_dir
        workflow._run_dir_inherited = self._run_dir is not None
        workflow.validate()
        return workflow
