import argparse
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
    parser = argparse.ArgumentParser(description="Run a job from a file.")
    parser.add_argument(
        "file_name", type=str, help="The filespec of a files with job definitions"
    )
    parser.add_argument("job_id", type=str, nargs="?", help="The ID of the job to run")
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="./outputs",
        help="The folder to write the output to",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")

    job_id = args.job_id
    data = load_json_file(args.file_name)
    schema = load_schema()

    startup()

    if isinstance(data, list):
        if job_id == "*":
            print("Running all jobs...")
            for job in data:
                run_job(job, schema, args.output_dir)

        else:
            job = None
            for item in data:
                if item.get("id") == job_id:
                    job = item
                    break

            if job is not None:
                run_job(job, schema, args.output_dir)
            else:
                print("Job not found " + job_id)

    else:
        run_job(data, schema, args.output_dir)
