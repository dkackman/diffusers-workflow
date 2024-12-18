import argparse
import os
from .schema import load_json_file
from .job import Job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a job from a file.")
    parser.add_argument(
        "file_name", type=str, help="The filespec of the job to validate"
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_name):
        raise FileNotFoundError(f"File {args.file_name} does not exist")

    try:
        job = Job(load_json_file(args.file_name))
        job.validate()
    except Exception as e:
        print(f"Error validating job: {args.file_name}")
        exit(1)

    print(f"Job {args.file_name} validated successfully")
