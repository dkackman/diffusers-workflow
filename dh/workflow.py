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


logger = logging.getLogger("dh")


def workflow_from_file(file_spec, output_dir):
    """Loads a workflow from a JSON file"""
    logger.info(f"Loading workflow from file: {file_spec}")
    with open(file_spec, "r") as file:
        return Workflow(json.load(file), output_dir, file_spec)


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
        """Returns workflow ID or 'unknown' if not specified"""
        return self.workflow_definition.get("id", "unknown")

    @property
    def argument_template(self):
        """Returns the argument template for this workflow"""
        return self.workflow_definition.get("argument_template", {})

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
            workflow_id = self.workflow_definition["id"]
            logger.debug(f"Processing workflow: {workflow_id}")

            # Handle variable substitution if variables are defined
            variables = self.workflow_definition.get("variables", None)
            if variables is not None:
                logger.debug(f"Setting variables for workflow: {workflow_id}")
                set_variables(arguments, variables)
                replace_variables(self.workflow_definition, variables)

            # Set up random seed for reproducibility
            default_seed = self.workflow_definition.get("seed", torch.seed())
            self.workflow_definition["seed"] = default_seed

            # Prepare workflow by processing arguments
            workflow = prepare_workflow(self.workflow_definition)

            # Initialize collections for sharing state between steps
            results = {}  # Stores results from each step
            shared_components = {}  # Shared resources between steps
            pipelines = {}  # Active pipelines
            last_result = None  # Final result to return

            # Execute each step in sequence
            for i, step_data in enumerate(workflow["steps"]):
                logger.debug(
                    f"Running step {i+1}/{len(workflow['steps'])}: {step_data['name']}"
                )
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

            logger.debug(f"Workflow {workflow_id} completed successfully")
            # Return only the last step's results for child workflows
            return_value = last_result.result_list if last_result is not None else []
            return return_value

        except Exception as e:
            logger.error(
                f"Error running workflow {self.workflow_definition.get('id', 'unknown')}: {e}"
            )

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
            workflow_definition = step_definition["workflow"]
            path = workflow_definition["path"]
            # Handle built-in workflows
            if path.startswith("builtin:"):
                path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "workflows",
                    path.replace("builtin:", ""),
                )
            # Handle relative paths
            elif not os.path.isabs(path):
                path = os.path.join(os.path.dirname(self.file_spec), path)

            workflow = workflow_from_file(path, self.output_dir)
            workflow.workflow_definition["argument_template"] = workflow_definition.get(
                "arguments", {}
            )
            workflow.validate()
            return workflow

        logger.debug(f"Creating task for step: {step_definition['name']}")
        # Handle task creation
        task_definition = step_definition["task"]
        task = Task(task_definition, device_identifier)
        return task


def prepare_workflow(input_workflow):
    """
    Creates a copy of the workflow and processes its arguments
    Returns the prepared workflow definition
    """
    if input_workflow is None:
        return {}, 0

    workflow = copy.deepcopy(input_workflow)
    realize_args(workflow)
    return workflow
