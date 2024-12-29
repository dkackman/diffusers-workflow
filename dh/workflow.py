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
        return Workflow(json.load(file), output_dir, file_spec)


class Workflow:
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

    def validate(self):
        status, message = validate_data(
            self.workflow_definition, load_schema("workflow")
        )
        if not status:
            raise Exception(f"Validation error: {message}")

    def run(self, arguments, previous_pipelines=None):
        # previous_pipelines needed for generically invoking a step action
        # but it is ignored here, meaning pipelines are not sahred between parent and child workflows
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
            # these are not shared between workflows
            results = {}
            shared_components = {}
            pipelines = {}
            last_result = None  # will be the return value of this workflow
            for i, step_data in enumerate(workflow["steps"]):
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

            # a child workflow only returns the results of the last step
            return_value = last_result.result_list if last_result is not None else []
            return return_value

        except Exception as e:
            print(
                f"Error running workflow {self.workflow_definition.get('id', 'unknown')}"
            )
            print(e)

    def create_step_action(
        self,
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
            previous_pipelines[step_definition["name"]] = pipeline
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
            path = workflow_definition["path"]
            if path.startswith("builtin:"):
                path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "workflows",
                    path.replace("builtin:", ""),
                )
            elif not os.path.isabs(path):
                # if the path is relative it is relative to the current workflow
                path = os.path.join(os.path.dirname(self.file_spec), path)

            workflow = workflow_from_file(path, self.output_dir)
            workflow.workflow_definition["argument_template"] = workflow_definition.get(
                "arguments", {}
            )
            workflow.validate()
            return workflow

        task_definition = step_definition["task"]
        task = Task(task_definition, device_identifier)
        return task


def prepare_workflow(input_workflow):
    if input_workflow is None:
        return {}, 0

    workflow = copy.deepcopy(input_workflow)

    realize_args(workflow)

    return workflow
