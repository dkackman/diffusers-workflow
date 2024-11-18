import os
import json
import torch
import copy
from .pipeline_processors.pipeline import run_step
from .pipeline_processors.arguments import realize_args

class Job:
    def __init__(self, data):
        self.data = data

    def run(self, output_dir):
        try:
            do_work(self.data, output_dir)
            print("ok")

        except Exception as e:
            print(f"Error running job {self.data.get('id', 'unknown')}")
            print(e)


def do_work(input_job, output_dir):
    job_id = input_job["id"]
    print(f"Processing {job_id}")

    job, default_seed = prepare_job(input_job)

    # collections that are passed between steps to share state
    results = []
    intermediate_results = {}
    shared_components = {}
    for step in job["steps"]:
        name = step["name"]
        print(f"Running step {name}...")

        step["seed"] = step.get("seed", default_seed)
        result = run_step(step, "cuda", intermediate_results, shared_components)
        if result is not None:
            results.extend(result)  

    with open(os.path.join(output_dir, f"{job_id}.json"), 'w') as file:
        json.dump(input_job, file, indent=4)

    for i, result in enumerate(results):
        default_name = f"{job_id}-{i}{result.guess_extension()}"
        result.save(output_dir, default_name)        


def prepare_job(input_job):
    if input_job is None:
        return {}, 0

    # modify the source dictionary with a default seed value
    # so that it will be captured when the arguments are saved
    default_seed = input_job.get("seed", torch.seed())    
    input_job["seed"] = default_seed

    job = copy.deepcopy(input_job)

    realize_args(job)

    return job, default_seed
