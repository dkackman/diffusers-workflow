import os
import json
import torch
import copy
from .arguments import realize_args
from .step import Step


class Job:
    def __init__(self, job_definition):
        self.job_definition = job_definition

    def run(self, output_dir):
        try:
            job_id = self.job_definition["id"]
            print(f"Processing {job_id}")

            # prepare the job by realizing the arguments
            # ie fetching images, replacing type names with actual types etc
            job, default_seed = prepare_job(self.job_definition)

            # collections that are passed between steps to share state
            results = {}
            shared_components = {}
            for step_data in job["steps"]:
                step = Step(step_data, default_seed)
                result = step.run(results, shared_components)
                results[step.name] = result

            with open(os.path.join(output_dir, f"{job_id}.json"), 'w') as file:
                json.dump(self.job_definition, file, indent=4)

            for i, result in enumerate(results.items()):                
                name = result[0]
                value = result[1]
                value.save(output_dir, f"{job_id}-{name}-{i}")        

            print("ok")

        except Exception as e:
            print(f"Error running job {self.job_definition.get('id', 'unknown')}")
            print(e)


def prepare_job(input_job):
    if input_job is None:
        return {}, 0

    # modify the source dictionary with a default seed value
    # so that it will be captured when the arguments are saved
    input_job["seed"] = input_job.get("seed", torch.seed())    

    job = copy.deepcopy(input_job)

    realize_args(job)

    return job, input_job["seed"]
