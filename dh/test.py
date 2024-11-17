import os
from .synchronous_worker import startup, do_work
from .validator import validate_job, load_json_file, load_schema


def run_job(job, schema, output_dir):
    job_id = job["id"]
    try:
        status, msg = validate_job(job, schema)
        if not status:
            print(f"Validation error for job {job_id}: {msg}")
        else:
            do_work(job, output_dir)
            print("ok")

    except Exception as e:
        print(f"Error running job {job_id}")
        print(e)


if __name__ == "__main__":
    schema = load_schema()
    job = load_json_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_job.json"))
    startup()

    run_job(job, schema, ".")
