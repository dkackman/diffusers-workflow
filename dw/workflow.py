# Core functionality for loading and executing workflows
import os
import json
import torch
import copy
import gc
import hashlib
import logging
import secrets
from datetime import datetime, timezone
from .arguments import (
    realize_args,
    realize_constants,
    fetch_constant,
    is_constant_reference,
)
from .events import (
    RunContext,
    emit_phase,
    WorkflowCancelled,
    get_context,
    current_context,
    activate_context,
    deactivate_context,
)
from .previous_results import (
    StepResults,
    previous_result_reference_errors,
)
from .locations import location_errors
from .subfolders import step_subfolder, subfolder_errors
from .step import Step
from .step_cache import (
    step_cache,
    referenced_result_names,
    reference_resolves_to,
)
from .runs import (
    FLAT_LAYOUT,
    REALIZED_FILE_NAME,
    activate_output_root,
    deactivate_output_root,
    workflow_identity,
    manifest_relative_files,
    new_run_id,
    output_layout,
    run_directory,
    write_manifest,
    write_realized_workflow,
)
from .realize import realize_workflow
from .schema import validate_data_all, format_validation_errors, load_schema
from .for_each import expand_for_each, ForEachError
from .variables import (
    argument_errors,
    replace_variables,
    resolve_variable_values,
    set_variables,
    undeclared_variable_references,
    VariableCycleError,
    VariableNotFoundError,
)
from .pipeline_processors.pipeline import Pipeline
from .tasks.model_cache import clear_model_cache
from .tasks.task import Task
from . import get_device, empty_device_cache, device_memory_stats
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
from .workflow_sources import (
    builtin_root,
    catalog_root,
    resolve_sub_workflow,
    SubWorkflowNotFound,
)

logger = logging.getLogger("dw")

# The widest integer JavaScript's double represents exactly - the ceiling on
# any seed the engine draws, since seeds travel as JSON through a browser
SEED_BITS = 53


class ConstantError(ValueError):
    """A 'constant:' variable default that failed to resolve during
    validation, with the 'variables.<name>' path at fault."""

    def __init__(self, path, message):
        super().__init__(message)
        self.path = path


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


def catalog_root_dir(file_spec):
    """The nearest ancestor directory literally named 'workflows' of
    file_spec, else file_spec's own directory.

    Used to confine a relative sub-workflow reference when a run carries no
    workflow_dir of its own (an unconfined CLI run) - the same "last
    'workflows' segment" rule workflow_output_subfolder uses for output
    naming, but returning the directory itself rather than what sits under
    it. It is `catalog_root` asked for a file rather than a directory, so
    the resolver (dw/workflow_sources.py) confines to exactly this root.
    """
    return catalog_root(os.path.dirname(os.path.abspath(file_spec)))


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


def _allocated_mb():
    """Device memory in use right now, for the pipeline_released event -
    None where the backend cannot say, so a reading is never confused with
    a genuine zero."""
    try:
        stats = device_memory_stats()
    except Exception:  # a progress figure is never worth failing a run over
        return None
    return stats["allocated_mb"] if stats["available"] else None


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

    def expanded_definition(self, arguments=None, source_indices=None):
        """The definition as the run will see it: constants realized,
        variables substituted - the caller's `arguments` folded in when they
        are all good, else the declared defaults - and every for_each step
        expanded.

        Raises ForEachError for a for_each that cannot be expanded,
        ConstantError for a 'constant:' variable default that fails to
        resolve, and VariableNotFoundError for a 'variable:' that names
        nothing - which is exactly what the run itself would raise, since a
        definition that declares variables is always substituted before it
        runs.

        `source_indices`, when a list is passed, comes back holding the
        index in *this* definition's steps of every expanded step, so an
        error can be reported at a path in the file the author wrote.
        """
        definition = copy.deepcopy(self.workflow_definition)
        variables = definition.get("variables")
        if isinstance(variables, dict):
            # the run realizes constants before folding arguments, and a
            # list defaulted to a 'constant:' name must expand here as it
            # does there - a name lookup, no download. Realizing a constant
            # imports the module it names, so validating one runs the same
            # trust gate (require_trusted_dotted_name) a run would - only
            # the diffusers ecosystem allowlist, unless the caller trusts
            # the workflow. Realized per top-level variable, not as one
            # call over the whole dict, so a failure names the variable.
            for name, value in variables.items():
                try:
                    if is_constant_reference(value):
                        variables[name] = fetch_constant(value)
                    else:
                        realize_constants(value)
                except (ValueError, InvalidInputError, UntrustedWorkflowError) as e:
                    raise ConstantError(f"variables.{name}", str(e)) from e
            if arguments and not argument_errors(definition, arguments):
                set_variables(arguments, variables)
            variables = resolve_variable_values(variables)
            definition = replace_variables(definition, variables)
        return expand_for_each(definition, source_indices)

    def resolve_sub_workflow_path(self, path):
        """Where one sub-workflow step's `path` resolves to, as
        (path, root) - the same resolution create_step_action does, asked
        ahead of the run so validation can answer for free what used to cost
        a queued job to find out (#89).

        Raises SubWorkflowNotFound, SecurityError or InvalidInputError,
        each carrying the message the run would have failed with.
        """
        confine_to = self.workflow_dir
        if path.startswith("builtin:"):
            builtin_name = path.replace("builtin:", "")
            if (
                not builtin_name.endswith(".json")
                or "/" in builtin_name
                or "\\" in builtin_name
            ):
                raise InvalidInputError(
                    f"Invalid builtin workflow name: {builtin_name}"
                )
            confine_to = builtin_root()
            resolved = os.path.join(confine_to, builtin_name)
            if not os.path.isfile(resolved):
                raise SubWorkflowNotFound(path, [resolved])
            return validate_workflow_path(resolved, confine_to), confine_to
        if confine_to is None and not os.path.isabs(path):
            confine_to = catalog_root_dir(self.file_spec)
        resolved, confine_to = resolve_sub_workflow(
            path, os.path.dirname(self.file_spec), confine_to
        )
        return validate_workflow_path(resolved, confine_to), confine_to

    def sub_workflow_errors(self, expanded, source_indices=None, composing=None):
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
            source = source_indices[index] if source_indices else index
            where = f"steps[{source}].workflow.path"
            path = reference["path"]
            try:
                resolved, root = self.resolve_sub_workflow_path(path)
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
                child = workflow_from_file(resolved, self.output_dir, root)
            except Exception as e:
                errors.append({"path": where, "message": f"Sub-workflow '{path}': {e}"})
                continue
            for error in child.validation_errors(composing=composing + [resolved]):
                errors.append(
                    {
                        "path": f"{where} -> {error['path']}",
                        "message": f"Sub-workflow '{path}': {error['message']}",
                    }
                )
        return errors

    def sub_workflow_warnings(self, expanded=None):
        """An argument a sub-workflow step passes down that the workflow it
        composes declares no variable for - dropped in silence at run time,
        and composition is exactly where a name drifts (#89)."""
        warnings = []
        try:
            expanded = expanded if expanded is not None else self.expanded_definition()
        except Exception:
            return warnings
        for index, step in enumerate(expanded.get("steps", []) or []):
            reference = step.get("workflow")
            if not isinstance(reference, dict):
                continue
            passed = reference.get("arguments")
            if not isinstance(passed, dict) or not isinstance(
                reference.get("path"), str
            ):
                continue
            try:
                resolved, root = self.resolve_sub_workflow_path(reference["path"])
                child = workflow_from_file(resolved, self.output_dir, root)
            except Exception:
                # An unresolvable path is an error, reported by
                # sub_workflow_errors - not a second complaint here
                continue
            declared = child.workflow_definition.get("variables") or {}
            for name in sorted(set(passed) - set(declared)):
                warnings.append(
                    {
                        "path": f"steps[{index}].workflow.arguments.{name}",
                        "message": (
                            f"'{reference['path']}' declares no variable "
                            f"'{name}' - the value is dropped. Declared: "
                            + (", ".join(sorted(declared)) or "<none>")
                        ),
                    }
                )
        return warnings

    def validation_errors(self, arguments=None, composing=None):
        """Every schema violation in the definition, as [{path, message}];
        empty when it validates. `arguments` are the caller's, so a
        for_each over a list the caller supplies is checked as it will run.

        `composing` carries the chain of sub-workflows above this one, so a
        workflow that composes itself is an error rather than a recursion.
        """
        errors = validate_data_all(self.workflow_definition, load_schema("workflow"))
        # Only once the shape is known good: the passes below walk the
        # steps array and a definition that fails the schema may have no
        # such array to walk
        if errors:
            return errors
        source_indices = []
        try:
            expanded = self.expanded_definition(arguments, source_indices)
        except ForEachError as e:
            return [{"path": e.path, "message": str(e)}]
        except ConstantError as e:
            return [{"path": e.path, "message": str(e)}]
        except VariableNotFoundError:
            # Every undeclared reference, not just the first one substitution
            # tripped over - and reported where each sits rather than as a
            # for_each whose list arrived unsubstituted, which is what a
            # half-substituted definition used to look like from here
            return self._undeclared_variable_errors(arguments)
        except VariableCycleError as e:
            # resolve_variable_values raises this for a variable that
            # references itself, directly or through others - there is
            # no single path inside the definition to blame, so it is
            # reported against 'variables' as a whole rather than escaping
            # as an unhandled exception
            return [{"path": "variables", "message": str(e)}]
        base_dir = (
            os.path.dirname(os.path.abspath(self.file_spec)) if self.file_spec else None
        )
        return (
            previous_result_reference_errors(expanded, source_indices)
            + subfolder_errors(expanded, source_indices)
            # A location policy refuses before a model load is spent on the
            # run rather than after it (dw/locations.py)
            + location_errors(expanded, source_indices, base_dir)
            + self.sub_workflow_errors(expanded, source_indices, composing)
        )

    def _undeclared_variable_errors(self, arguments=None):
        """Every 'variable:' reference naming nothing the workflow declares.

        Fatal rather than a warning: once a workflow has a 'variables'
        block, replace_variables refuses an undeclared reference, so this is
        a run that cannot start. Good caller `arguments` are folded in first,
        and a reference inside one of them is reported under `arguments.`,
        where the caller wrote it.
        """
        definition = copy.deepcopy(self.workflow_definition)
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
                    f"'variable:{name}' names no declared variable; "
                    f"declared: {', '.join(declared) or '<none>'}"
                ),
            }
            for path, name in undeclared_variable_references(definition)
        ]

    def validate(self):
        """Validates workflow definition against JSON schema.

        Every violation is reported, one per line, so the CLI, the REPL
        and an agent iterating on a draft fix them in one pass rather than
        one per round trip.
        """
        logger.debug(f"Validating workflow: {self.name}")
        errors = self.validation_errors()
        if errors:
            # message already carries the 'Validation error' prefix
            message = format_validation_errors(errors)
            logger.error(message)
            raise Exception(message)
        logger.debug(f"Workflow {self.name} validated successfully")

    def _prepare_definition(self, workflow_def, arguments, base_dir):
        """The definition as a run works from it: constants realized,
        arguments folded into the variables, list entries' own references
        resolved, variable values realized (assets loaded), every
        'variable:' substituted, every for_each expanded, and the seed read
        and coerced. Returns (workflow_def, default_seed) - the seed is
        None when the workflow names none, and the caller decides what
        that means (run() draws one; cache_hits() reports no hits).

        Shared by run() and cache_hits() so the probe prepares exactly what
        the run prepares - the step cache keys on the realized step, and a
        probe that prepared it differently would answer for a run that
        never happens.
        """
        workflow_id = workflow_def["id"]
        variables = workflow_def.get("variables", None)
        if variables is not None:
            logger.debug(f"Setting variables for workflow: {workflow_id}")
            # a constant is the value a variable declares, so it resolves before
            # anything is converted to the type of that declaration
            realize_constants(variables)
            # first set variable values base don the arguments passed to the workflow
            # these may come form the command line or form a parent workflow
            set_variables(arguments, variables)
            # an entry of a list-valued variable may name another
            # variable; resolve those before anything inside it is
            # realized, so a reference type in an entry is a type name
            variables = resolve_variable_values(variables)
            # realize the variables, initializing downloads of images etc
            realize_args(variables, base_dir)
            ## then replace any variable references in the workflow definition with the actual values
            # replace_variables returns a new structure rather than mutating in
            # place, so the result must be captured here
            workflow_def = replace_variables(workflow_def, variables)

        # One ordinary step per entry of every for_each list, before the
        # seed, the run id and the realized workflow are computed, so
        # each covers what actually runs. A ForEachError here fails the
        # run before anything loads
        workflow_def = expand_for_each(workflow_def)

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
                    f"Workflow {workflow_id} seed must be an integer, "
                    f"got {default_seed!r}"
                )
            workflow_def["seed"] = default_seed
        return workflow_def, default_seed

    def _cache_lookup(
        self,
        workflow_id,
        steps,
        index,
        step_data,
        step_seed,
        hits_this_run,
        cache_enabled,
    ):
        """Whether the step cache serves step `index`, as (cached_result or
        None, the step_data snapshot the entry is keyed on or None, whether
        a later step still reads this one's result, the names later steps
        still reference). Shared by run() and cache_hits() - see
        _prepare_definition for why.
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
        parent_saves_this = self._final_save_owned_by_parent and index == len(steps) - 1
        step_data_snapshot = None
        if is_cacheable:
            try:
                step_data_snapshot = copy.deepcopy(step_data)
                if parent_saves_this:
                    # Keyed apart from the same step run standalone: this
                    # entry's result was never saved here, so a standalone
                    # hit on it would report no files
                    step_data_snapshot["__saved_by_parent__"] = True
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
            self.output_dir,
            needs_result=result_needed,
        )
        return cached_result, step_data_snapshot, result_needed, remaining_refs

    def cache_hits(self, arguments):
        """The steps the step cache would serve for a run with `arguments`,
        in step order - what the plan reports as cached_steps (#85).

        Prepares the definition exactly as run() does and asks the cache the
        question run() asks, step by step with the hits so far, and executes
        nothing: no run directory, no events, no pipeline. An unseeded
        workflow has no cache, so it answers [] without asking.
        """
        output_root_token = activate_output_root(self.output_dir)
        try:
            workflow_def = copy.deepcopy(self.workflow_definition)
            workflow_id = workflow_def["id"]
            base_dir = (
                os.path.dirname(os.path.abspath(self.file_spec))
                if self.file_spec
                else None
            )
            workflow_def, default_seed = self._prepare_definition(
                workflow_def, arguments or {}, base_dir
            )
            if default_seed is None or not self._cache_enabled_by_parent:
                return []
            steps = workflow_def.get("steps", [])
            realize_args(steps, base_dir)
            hits_this_run = set()
            hits = []
            for index, step_data in enumerate(steps):
                step_seed = step_data.get("seed", default_seed)
                cached_result, _, _, _ = self._cache_lookup(
                    workflow_id,
                    steps,
                    index,
                    step_data,
                    step_seed,
                    hits_this_run,
                    True,
                )
                if cached_result is not None:
                    hits_this_run.add(step_data["name"])
                    hits.append(step_data["name"])
            return hits
        finally:
            deactivate_output_root(output_root_token)

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
        # 'output:' references resolve against the directory this run was
        # told to write to - the root, not this run's own subdirectory, since
        # what they name is what an earlier run left there
        output_root_token = activate_output_root(self.output_dir)
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
        # The seed this run actually used, which is not self.workflow_definition's:
        # run() works on a deep copy, and a workflow naming no seed draws a random
        # one into that copy. Recording the original would write null into the
        # manifest of every seedless run and lose the only record of what produced
        # its files - the seed is what makes a run repeatable
        resolved_seed = None
        # What the run recorded about itself, read by _write_run_manifest in
        # the finally below - initialized here so a failure before the run
        # directory exists still writes a well-formed manifest
        realized_name = None
        annotations = {"prompts": [], "sub_workflows": {}}
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

            workflow_def, default_seed = self._prepare_definition(
                workflow_def, arguments, base_dir
            )
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
                # OS entropy rather than torch or random, so a process that
                # seeded either for reproducibility is not disturbed. Bounded
                # to 53 bits rather than the 64 torch allows: the seed is
                # embedded in the image, the manifest and the realized
                # workflow as JSON, and a browser reads every integer as a
                # double - a seed that changed on the way through would be a
                # seed nobody can reproduce
                default_seed = secrets.randbits(SEED_BITS)
            workflow_def["seed"] = default_seed
            resolved_seed = default_seed

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

            # The record of what actually ran, written before the first step
            # so a crash or a cancel still leaves it. A sub-workflow inherits
            # the parent's directory and writes none of its own, as with the
            # manifest, and the flat layout has no directory to write into
            if self._run_dir and not self._run_dir_inherited:
                try:
                    realized, annotations = realize_workflow(
                        self.workflow_definition,
                        arguments,
                        default_seed,
                        base_dir=base_dir,
                        output_root=self.output_dir,
                        workflow_dir=self.workflow_dir,
                    )
                    if write_realized_workflow(self._run_dir, realized):
                        realized_name = REALIZED_FILE_NAME
                except Exception as e:
                    # Never fatal: the record is worth less than the run
                    logger.warning(f"Could not realize workflow {workflow_id}: {e}")

                # Which run this is, so a server job can find the directory
                # it wrote. Emitted even when the realized file did not land:
                # the manifest is still there, and so are the files
                run_context.emit(
                    "run_start",
                    run_id=run_id,
                    identity=workflow_identity(self.file_spec, workflow_id),
                    run_dir=os.path.relpath(self._run_dir, self.output_dir).replace(
                        os.sep, "/"
                    ),
                )

            # Initialize collections for sharing state between steps
            # Stores results from each step, and remembers the names of
            # steps whose results have since been released
            results = StepResults()
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
                status = "completed"
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
                logger.debug(f"Running step {i + 1}/{len(steps)}: {step_data['name']}")
                run_context.emit(
                    "step_start",
                    workflow=workflow_id,
                    step=step_data["name"],
                    index=i,
                    total_steps=len(steps),
                    **self._parent_progress_fields(),
                )

                # Seeds resolve most-specific-first: pipeline > step > workflow
                step_seed = step_data.get("seed", default_seed)

                step = Step(step_data, step_seed, self.workflow_definition)

                cached_result, step_data_snapshot, result_needed, remaining_refs = (
                    self._cache_lookup(
                        workflow_id,
                        steps,
                        i,
                        step_data,
                        step_seed,
                        hits_this_run,
                        cache_enabled_this_run,
                    )
                )
                is_cacheable = step_data_snapshot is not None
                # The last step of a composed child whose parent does the
                # saving (#92) - its files are written once, by the parent
                parent_saves_this = (
                    self._final_save_owned_by_parent and i == len(steps) - 1
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
                if isinstance(step_action, Workflow):
                    # The child reports into this run's counter rather than
                    # its own, and a grandchild reports into the same one
                    step_action._parent_progress = self._parent_progress or {
                        "step": step_data["name"],
                        "index": i,
                        "total_steps": len(steps),
                    }
                    # Only when the parent's own result would write
                    # something: a result block that names no content_type,
                    # or says save: false, saves nothing, and suppressing
                    # the child's save for it would lose the artifact
                    parent_result = step_data.get("result")
                    step_action._final_save_owned_by_parent = bool(
                        isinstance(parent_result, dict)
                        and parent_result.get("content_type")
                        and parent_result.get("save", True)
                    )
                reused = cached_result is not None
                if reused:
                    logger.info(f"Step '{step.name}' unchanged - reusing cached result")
                    result = cached_result
                    saved_files = result.saved_files
                    hits_this_run.add(step.name)
                else:
                    result = step.run(results, pipelines, step_action)

                # A sub-workflow's saves land in the child's manifest - read it
                # here, before the release below may drop the child
                sub_manifest = (
                    list(getattr(step_action, "manifest", []))
                    if isinstance(step_action, Workflow)
                    else []
                )

                # A released pipeline frees its memory for later steps - the
                # alternative on a card that cannot hold two models is offloading
                # everything, which taxes every run to survive one transition.
                # Before the write, not after: the result is already in host
                # memory and saving never touches the pipeline, so a release
                # that waited for the write would hold ~10 GB on the device
                # through the longest phase of a video step. The loop's own
                # locals are the last references to this step's action, so
                # clearing that is part of the release - a popped pipeline this
                # frame still holds is not freed, and it would otherwise stay
                # resident through the next step's load, which is exactly when
                # both models would be in memory at once
                if step_data.get("release_pipeline", False):
                    logger.info(f"Releasing pipeline for step: {step.name}")
                    before = _allocated_mb()
                    pipelines.pop(self._pipeline_keys_by_step.get(step.name), None)
                    step_action = None
                    gc.collect()
                    empty_device_cache()
                    # Say so on the event stream. The release is otherwise
                    # invisible to a consumer: it sits inside the sub-second
                    # window between a step's generation and its files
                    # appearing, which is too narrow to catch by polling
                    # get_memory, and it is exactly the ordering this event
                    # exists to make readable (it precedes the step's
                    # step_end, and on a released card the figures show the
                    # drop rather than implying it)
                    run_context.emit(
                        "pipeline_released",
                        workflow=workflow_id,
                        step=step.name,
                        index=i,
                        gpu_memory_allocated_mb=_allocated_mb(),
                        gpu_memory_allocated_before_mb=before,
                    )

                if not reused:
                    saved_files = (
                        []
                        if parent_saves_this
                        else result.save(
                            self.step_output_dir(step_data),
                            self.step_save_name(workflow_id, step.name, i),
                        )
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
                subfolder = step_subfolder(step_data)
                manifest_entry = {
                    "step": step.name,
                    "files": saved_files,
                    "subfolder": subfolder,
                }
                if reused:
                    manifest_entry["reused"] = True
                # No entry at all for a step the parent saves for: the
                # parent's own entry names the same files, under the step
                # name the caller wrote (#92)
                if not parent_saves_this:
                    self.manifest.append(manifest_entry)
                # roll the child's saves up so job history and the gallery see
                # every file
                self.manifest.extend(sub_manifest)
                step_end_data = {"files": saved_files, "subfolder": subfolder}
                if reused:
                    step_end_data["reused"] = True
                run_context.emit(
                    "step_end",
                    workflow=workflow_id,
                    step=step.name,
                    index=i,
                    total_steps=len(steps),
                    **self._parent_progress_fields(),
                    **step_end_data,
                )
                logger.debug(f"Step {step.name} completed with result: {result}")

                # Release results no later step references - saved to disk
                # already, and last_result keeps the workflow's return value
                release_unreferenced_results(results, remaining_refs)

                # The loop's own locals are the last references to this step's
                # action and result - anything they still hold would stay
                # resident through the next step's load
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
                self._write_run_manifest(
                    run_id,
                    status,
                    started_at,
                    arguments,
                    resolved_seed,
                    realized_name,
                    annotations,
                )
            deactivate_output_root(output_root_token)
            deactivate_context(context_token)

    def _write_run_manifest(
        self,
        run_id,
        status,
        started_at,
        arguments,
        seed,
        realized_name=None,
        annotations=None,
    ):
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
                    # The realized copy beside this manifest, or null when
                    # writing it did not land - the manifest is the only
                    # place that difference is visible
                    "realized": realized_name,
                    # Annotations the schema has nowhere to put: which
                    # stored prompts were inlined, and what each local
                    # sub-workflow file held when it ran
                    "prompts": (annotations or {}).get("prompts", []),
                    "sub_workflows": (annotations or {}).get("sub_workflows", {}),
                },
                "seed": seed,
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
                    output_dir=self.step_output_dir(step_definition),
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
                    new_pipeline_wrapper.argument_template["generator"] = (
                        torch.Generator(new_pipeline_wrapper.device).manual_seed(
                            new_pipeline_wrapper.pipeline_definition.get(
                                "seed", default_seed
                            )
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
                output_dir=self.step_output_dir(step_definition),
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
                output_dir=self.step_output_dir(step_definition),
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
                # Everything else - a relative path, or a catalog name as
                # list_workflows reports it - goes through the search path.
                # A template under templates/ names a model config as
                # '../models/x.json', so a path relative to the referencing
                # file still resolves first and the '..' is collapsed here,
                # which is what lets the validator judge where the path
                # actually lands rather than refusing the spelling;
                # containment is still checked on the resolved path below.
                # An unconfined run (no workflow_dir - a bare CLI
                # invocation) used to rely on the '..' regex alone to stop a
                # relative reference from leaving the file's own directory;
                # normalizing the path removes that guard, so confine it to
                # the catalog root instead - the referencing file's nearest
                # ancestor literally named 'workflows', which still lets it
                # climb to a sibling folder like models/ but not out of the
                # catalog
                else:
                    if confine_to is None and not os.path.isabs(path):
                        confine_to = catalog_root_dir(self.file_spec)
                    path, confine_to = resolve_sub_workflow(
                        path, os.path.dirname(self.file_spec), confine_to
                    )

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

            # this is where the arguments in the parent script are passed to the child workflow
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
