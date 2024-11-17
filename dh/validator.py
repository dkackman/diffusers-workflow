import json
import os
from jsonschema import validate, ValidationError


def validate_job(job, schema):
    job_id = job["id"]
    print(f"Validating job {job_id}...")
    try:
        # Validate the data against the schema
        validate(instance=job, schema=schema)
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
    

def load_schema():
    return load_json_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_schema.json"))