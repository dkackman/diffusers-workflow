
import argparse
from .validator import validate_job, load_json_file, load_schema


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a job definition.")
    parser.add_argument(
        "file_name", type=str, help="The filespec of a files with job definitions"
    )
    parser.add_argument("job_id", type=str, nargs="?", help="The ID of the job to run")
    args = parser.parse_args()

    job_id = args.job_id
    data = load_json_file(args.file_name)
    schema = load_schema()

    if isinstance(data, list):
        if job_id == "*":
            print("Validating all jobs...")
            passed = 0
            failed = 0
            for job in data:
                status, msg = validate_job(job, schema)
                if status:
                    passed += 1
                else:
                    failed += 1
                print(f"Job {job['id']}: {msg}")

            print(f"Validation completed: {passed} passed, {failed} failed.")
            
        else:
            job = None


            for item in data:
                if item.get("id") == job_id:
                    job = item
                    break

            if job is not None:
                status, msg = validate_job(job, schema)
                print(f"Job {job['id']}: {msg}")
            else:
                print("Job not found " + job_id)

    else:
        status, msg = validate_job(data, schema)
        print(f"Job {args.file_name}: {msg}")        
