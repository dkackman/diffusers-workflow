import os
import json
import torch
import copy
from .arguments import realize_args
from .step import Step
from .schema import validate_data, load_schema
from .variables import replace_variables, set_variables


class Job:
    def __init__(self, job_definition):
        self.job_definition = job_definition

    def validate(self):
        status, message = validate_data(self.job_definition, load_schema("job"))
        if not status:
            raise Exception(f"Validation error: {message}")

    def run(self, output_dir, variable_assignments={}):
        try:
            job_id = self.job_definition["id"]
            print(f"Processing {job_id}")

            variables = self.job_definition.get("variables", None)
            if variables is not None:
                set_variables(variable_assignments, variables)
                replace_variables(self.job_definition, variables)

            default_seed = self.job_definition.get("seed", torch.seed())
            self.job_definition["seed"] = (
                default_seed  # save the seed for reproducibility
            )

            # prepare the job by realizing the arguments
            # ie fetching images, replacing type names with actual types etc
            job = prepare_job(self.job_definition)

            # collections that are passed between steps to share state
            results = {}
            shared_components = {}
            for i, step_data in enumerate(job["steps"]):
                step = Step(step_data, default_seed)
                result = step.run(results, shared_components)
                results[step.name] = result
                result.save(output_dir, f"{job_id}-{step.name}.{i}")

            with open(os.path.join(output_dir, f"{job_id}.json"), "w") as file:
                json.dump(self.job_definition, file, indent=4)

            print("ok")

        except Exception as e:
            print(f"Error running job {self.job_definition.get('id', 'unknown')}")
            print(e)


def prepare_job(input_job):
    if input_job is None:
        return {}, 0

    job = copy.deepcopy(input_job)

    realize_args(job)

    return job
