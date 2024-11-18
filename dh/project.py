import json
import os
from jsonschema import validate, ValidationError
from .synchronous_worker import do_work, startup
from .pipeline_processors.arguments import prepare_args


def create_project(data):
    if isinstance(data, dict) and "jobs" in data:
        return Project(data)
    
    # the data object is an array of jobs, so we need to wrap it in a dictionary
    if isinstance(data, list):
        return Project(
            {
                "jobs": data
            }
        )
    
    # the data object is a single job, so we need to wrap it in a list
    return Project(
        {
            "jobs": [data]
        }
    )

class Project:
    def __init__(self, data):
        self.data = data

    def validate(self):
        status, message = validate_job(self.data, load_schema("project"))
        if not status:
            raise Exception(f"Validation error: {message}")
        
    def run(self, job_id = "*", output_dir = "./outputs"):
        startup()

        if job_id == "*":
            print("Running all jobs...")
            for job in self.data.get("jobs", []):
                run_job(job, output_dir)

        else:
            job = None
            for item in self.data.get("jobs", []):
                if item["id"] == job_id:
                    job = item
                    break

            if job is not None:
                run_job(job, output_dir)
            else:
                print("Job not found " + job_id)


def run_job(job, output_dir):
    job_id = job["id"]
    try:
        do_work(job, output_dir)
        print("ok")

    except Exception as e:
        print(f"Error running job {job_id}")
        print(e)


def validate_job(data, schema):
    try:
        validate(instance=data, schema=schema)
        return True, "Validation successful"
        
    except ValidationError as ve:
        return False, f"Validation error: {ve.message}"
    except json.JSONDecodeError as je:
        return False, f"JSON parsing error: {str(je)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"
    

def load_json_file(file_spec):
    with open(file_spec, "r") as file:
        return json.load(file)
    

def load_schema(schema_name):
    return load_json_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{schema_name}_schema.json"))