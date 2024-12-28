import os
import json
import torch
import copy
from .arguments import realize_args
from .step import Step
from .schema import validate_data, load_schema
from .variables import replace_variables, set_variables
from .pipeline_processors.pipeline import Pipeline
from .tasks.task import Task


def workflow_from_file(file_spec, output_dir):
    with open(file_spec, "r") as file:
        return Workflow(json.load(file), output_dir)


class Workflow:
    def __init__(self, workflow_definition, output_dir):
        self.workflow_definition = workflow_definition
        self.output_dir = output_dir

    def validate(self):
        status, message = validate_data(
            self.workflow_definition, load_schema("workflow")
        )
        if not status:
            raise Exception(f"Validation error: {message}")

    def run(self, arguments={}):
        try:
            workflow_id = self.workflow_definition["id"]
            print(f"Processing {workflow_id}")

            variables = self.workflow_definition.get("variables", None)
            if variables is not None:
                set_variables(arguments, variables)
                replace_variables(self.workflow_definition, variables)

            default_seed = self.workflow_definition.get("seed", torch.seed())
            self.workflow_definition["seed"] = (
                default_seed  # save the seed for reproducibility
            )

            # prepare the workflow by realizing the arguments
            # ie fetching images, replacing type names with actual types etc
            workflow = prepare_workflow(self.workflow_definition)

            # collections that are passed between steps to share state
            results = {}
            shared_components = {}
            pipelines = {}
            for i, step_data in enumerate(workflow["steps"]):
                step = Step(step_data, default_seed)
                result = step.run(
                    results,
                    pipelines,
                    create_step_action(
                        step_data, shared_components, pipelines, default_seed, "cuda"
                    ),
                )
                results[step.name] = result
                result.save(self.output_dir, f"{workflow_id}-{step.name}.{i}")

            with open(
                os.path.join(self.output_dir, f"{workflow_id}.json"), "w"
            ) as file:
                json.dump(self.workflow_definition, file, indent=4)

            print("ok")

        except Exception as e:
            print(
                f"Error running workflow {self.workflow_definition.get('id', 'unknown')}"
            )
            print(e)


def prepare_workflow(input_workflow):
    if input_workflow is None:
        return {}, 0

    workflow = copy.deepcopy(input_workflow)

    realize_args(workflow)

    return workflow


def create_step_action(
    step_definition,
    shared_components,
    previous_pipelines,
    default_seed,
    device_identifier,
):
    # this comprises the complete lists of actions that can be taken
    if "pipeline" in step_definition:
        pipeline = Pipeline(
            step_definition["pipeline"],
            default_seed,
            device_identifier,
        )
        pipeline.load(shared_components)
        return pipeline

    if "pipeline_reference" in step_definition:
        pipeline_reference = step_definition["pipeline_reference"]
        previous_pipeline = previous_pipelines[pipeline_reference["reference_name"]]
        return Pipeline(
            pipeline_reference,
            default_seed,
            device_identifier,
            previous_pipeline.pipeline,
        )

    if "workflow" in step_definition:
        workflow_definition = step_definition["workflow"]
        workflow = workflow_from_file(workflow_definition)
        return workflow

    task_definition = step_definition["task"]
    task = Task(task_definition, device_identifier)
    return task
