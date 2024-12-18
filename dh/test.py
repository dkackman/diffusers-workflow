import os
from .schema import load_json_file
from .job import Job
from . import startup

if __name__ == "__main__":
    data = load_json_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_job.json")
    )
    job = Job(data)

    try:
        job.validate()
    except Exception as e:
        print(f"Error validating job: {e}")
        exit(1)

    try:
        startup()
        job.run(".")
    except Exception as e:
        print(f"Error running job: {e}")
        exit(1)
