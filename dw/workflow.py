# Core functionality for loading and executing workflows
import os
import json
import torch
import copy
import logging
from .arguments import realize_args
from .step import Step
from .schema import validate_data, load_schema
from .variables import replace_variables, set_variables
from .pipeline_processors.pipeline import Pipeline
from .tasks.task import Task
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

            # Handle variable substitution if variables are defined
            variables = workflow_def.get("variables", None)
            if variables is not None:
                logger.debug(f"Setting variables for workflow: {workflow_id}")
                # first set variable values base don the arguments passed to the workflow
                # these may come form the command line or form a parent workflow
                set_variables(arguments, variables)
                # realize the variables, initialiting downloads of images etc
                realize_args(variables)
                ## then replace any variable references in the workflow definition with the actual values
                replace_variables(workflow_def, variables)

            # Set up random seed for reproducibility
            default_seed = workflow_def.get("seed", torch.seed())
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

            realize_args(steps)

            # Execute each step in sequence
            for i, step_data in enumerate(steps):
                logger.debug(f"Running step {i+1}/{len(steps)}: {step_data['name']}")

                step = Step(step_data, default_seed)
                result = step.run(
                    results,
                    pipelines,
                    self.create_step_action(
                        step_data, shared_components, pipelines, default_seed, "cuda"
                    ),
                )
                last_result = result
                results[step.name] = result
                result.save(self.output_dir, f"{workflow_id}-{step.name}.{i}")
                logger.debug(f"Step {step.name} completed with result: {result}")

                # Cleanup between steps (but keep pipelines loaded)
                import gc

                gc.collect()

            logger.debug(f"Workflow {workflow_id} completed successfully")
            # Return only the last step's results for child workflows
            return last_result.result_list if last_result is not None else []

        except Exception as e:
            workflow_id = self.workflow_definition.get("id", "unknown")
            logger.error(f"Error running workflow {workflow_id}: {e}", exc_info=True)
            raise

    def create_step_action(
        self,
        step_definition,
        shared_components,
        previous_pipelines,
        default_seed,
        device_identifier,
    ):
        """
        Creates the appropriate action object based on step type:
        - Pipeline: Creates new pipeline
        - Pipeline reference: References existing pipeline
        - Workflow: Loads and validates sub-workflow
        - Task: Creates task object
        """
        # Handle pipeline creation
        if "pipeline" in step_definition:
            logger.debug(f"Creating pipeline for step: {step_definition['name']}")
            pipeline = Pipeline(
                step_definition["pipeline"],
                default_seed,
                device_identifier,
            )
            pipeline.load(shared_components)
            previous_pipelines[step_definition["name"]] = pipeline
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
                device_identifier,
                previous_pipeline.pipeline,
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
            workflow.validate()
            return workflow

        logger.debug(f"Creating task for step: {step_definition['name']}")
        # Handle task creation
        task_definition = step_definition["task"]
        task = Task(task_definition, device_identifier)
        return task
