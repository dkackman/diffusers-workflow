import argparse
import os
from . import startup
from .schema import load_json_file
from .job import Job

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a workflow from a file.")
    parser.add_argument(
        "file_name", type=str, help="The filespec to of the workflow to run"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./outputs",
        help="The folder to write the outputs to",
    )
    parser.add_argument(
        "variables",
        nargs="*",  # Accept 0 or more parameters
        help="Optional parameters in name=value format",
    )
    args = parser.parse_args()

    # Parse key-value pairs
    variables = {}
    for variable in args.variables:
        try:
            name, value = variable.split("=", 1)
            variables[name.strip()] = value.strip()
        except ValueError:
            print(f"Error: Variable '{variable}' is not in name=value format")
            exit(1)

    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")

    if not os.path.exists(args.file_name):
        raise FileNotFoundError(f"File {args.file_name} does not exist")

    job = Job(load_json_file(args.file_name))
    try:
        job.validate()
    except Exception as e:
        print(f"Error validating job: {e}")
        exit(1)

    try:
        startup()
        job.run(args.output_dir, variables)
    except Exception as e:
        print(f"Error running job: {e}")
        exit(1)
