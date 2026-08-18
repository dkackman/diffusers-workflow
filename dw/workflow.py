# Core functionality for loading and executing workflows
import os
import json
import torch
import copy
import gc
import logging
from .arguments import realize_args, realize_constants
from .step import Step
from .schema import validate_data, load_schema
from .variables import replace_variables, set_variables
from .pipeline_processors.pipeline import Pipeline
from .tasks.model_cache import clear_model_cache
from .tasks.task import Task
from . import get_device, empty_device_cache
from .security import (
    validate_workflow_path,
    validate_json_size,
    validate_output_path,
    SecurityError,
    PathTraversalError,
    InvalidInputError,
)

logger = logging.getLogger("dw")


def workflow_from_file(file_spec, output_dir):
    """Loads a workflow from a JSON file with security validation"""
    logger.debug(f"Loading workflow from file: {file_spec}")

    try:
        # Validate file path and size
        validated_path = validate_workflow_path(file_spec)
        validate_json_size(validated_path)
        validated_output = validate_output_path(output_dir, None)

        with open(validated_path, "r") as file:
            workflow_data = json.load(file)

        return Workflow(workflow_data, validated_output, validated_path)

    except SecurityError as e:
        logger.error(f"Security validation failed for workflow {file_spec}: {e}")
        raise
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load workflow from {file_spec}: {e}")
        raise


def referenced_result_names(steps):
    """Every previous_result reference the given steps make, as full names.

    Scans nested dicts and lists, so references inside pipeline arguments,
    task arguments and sub-workflow argument maps are all found.
    """
    prefix = "previous_result:"
    names = set()

    def scan(value):
        if isinstance(value, str) and value.startswith(prefix):
            names.add(value[len(prefix) :])
        elif isinstance(value, dict):
            for item in value.values():
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)

    for step in steps:
        scan(step)
    return names


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
        if not any(ref == n or ref.startswith(n + ".") for ref in remaining_refs)
    ]:
        logger.debug(f"Releasing result: {name}")
        del results[name]


class Workflow:
    """
    Main class for managing and executing workflows defined in JSON format
    Handles variable substitution, step execution, and result management
    """

    def __init__(self, workflow_definition, output_dir, file_spec):
        self.workflow_definition = workflow_definition
        self.output_dir = output_dir
        self.file_spec = file_spec

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

    def validate(self):
        """Validates workflow definition against JSON schema"""
        logger.debug(f"Validating workflow: {self.name}")
        status, message = validate_data(
            self.workflow_definition, load_schema("workflow")
        )
        if not status:
            logger.error(f"Validation error: {message}")
            raise Exception(f"Validation error: {message}")
        logger.debug(f"Workflow {self.name} validated successfully")

    def run(self, arguments, previous_pipelines=None):
        """
        Executes the workflow by:
        1. Processing variables
        2. Setting up random seed
        3. Running each step in sequence
        4. Managing results between steps
        """
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
                replace_variables(workflow_def, variables)

            # Set up random seed for reproducibility. Resolved lazily - as a
            # dict.get default, torch.seed() would run on every call and reseed
            # the global RNG even when the workflow names an explicit seed
            default_seed = workflow_def.get("seed")
            if default_seed is None:
                # A fresh generator draws a random seed without touching the
                # global RNG the process may have seeded for reproducibility
                default_seed = torch.Generator().seed()
            workflow_def["seed"] = default_seed

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

            # Execute each step in sequence
            for i, step_data in enumerate(steps):
                logger.debug(f"Running step {i+1}/{len(steps)}: {step_data['name']}")

                # Seeds resolve most-specific-first: pipeline > step > workflow
                step_seed = step_data.get("seed", default_seed)

                step = Step(step_data, step_seed)
                result = step.run(
                    results,
                    pipelines,
                    self.create_step_action(
                        step_data,
                        shared_components,
                        pipelines,
                        step_seed,
                        get_device(),
                    ),
                )
                last_result = result
                results[step.name] = result
                result.save(self.output_dir, f"{workflow_id}-{step.name}.{i}")
                logger.debug(f"Step {step.name} completed with result: {result}")

                # Release results no later step references - saved to disk
                # already, and last_result keeps the workflow's return value
                release_unreferenced_results(
                    results, referenced_result_names(steps[i + 1 :])
                )

                # A released pipeline frees its memory for later steps - the
                # alternative on a card that cannot hold two models is offloading
                # everything, which taxes every run to survive one transition
                if step_data.get("release_pipeline", False):
                    logger.info(f"Releasing pipeline for step: {step.name}")
                    pipelines.pop(step.name, None)

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
            # Return only the last step's results for child workflows
            return last_result.result_list if last_result is not None else []

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

            # Check if pipeline already loaded in cache (GPU persistence)
            if step_name in previous_pipelines:
                logger.debug(f"Reusing cached pipeline for step: {step_name}")
                cached_pipeline = previous_pipelines[step_name]
                # Create new Pipeline wrapper with updated step definition
                # but reuse the loaded model from cache
                new_pipeline_wrapper = Pipeline(
                    step_definition["pipeline"],
                    default_seed,
                    device,
                    cached_pipeline.pipeline,  # Reuse the actual loaded model
                    output_dir=self.output_dir,
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

                return new_pipeline_wrapper

            # Not in cache - load fresh
            logger.debug(f"Creating pipeline for step: {step_name}")
            pipeline = Pipeline(
                step_definition["pipeline"],
                default_seed,
                device,
                output_dir=self.output_dir,
                file_prefix=self.step_file_prefix(step_name),
            )
            pipeline.load(shared_components)
            previous_pipelines[step_name] = pipeline
            return pipeline

        # Handle pipeline reference
        if "pipeline_reference" in step_definition:
            logger.debug(
                f"Referencing existing pipeline for step: {step_definition['name']}"
            )
            pipeline_reference = step_definition["pipeline_reference"]
            previous_pipeline = previous_pipelines[pipeline_reference["reference_name"]]
            return Pipeline(
                pipeline_reference,
                default_seed,
                device,
                previous_pipeline.pipeline,
                output_dir=self.output_dir,
                file_prefix=self.step_file_prefix(step_definition["name"]),
            )

        # Handle sub-workflow
        if "workflow" in step_definition:
            logger.debug(f"Loading sub-workflow for step: {step_definition['name']}")
            workflow_reference = step_definition["workflow"]
            path = workflow_reference["path"]

            try:
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
                    path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "workflows",
                        builtin_name,
                    )
                # Handle relative paths
                elif not os.path.isabs(path):
                    base_dir = os.path.dirname(self.file_spec)
                    path = os.path.join(base_dir, path)

                # Validate the resolved path
                validated_path = validate_workflow_path(path)
                workflow = workflow_from_file(validated_path, self.output_dir)

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
            workflow.validate()
            return workflow

        logger.debug(f"Creating task for step: {step_definition['name']}")
        # Handle task creation
        task_definition = step_definition["task"]
        task = Task(task_definition, device)
        return task
