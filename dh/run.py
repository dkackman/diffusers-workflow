import argparse
import os
from . import startup
from .workflow import workflow_from_file

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

    workflow = workflow_from_file(args.file_name, args.output_dir)
    try:
        workflow.validate()
    except Exception as e:
        print(f"Error validating workflow: {e}")
        exit(1)

    try:
        startup()
        workflow.run(variables)
    except Exception as e:
        print(f"Error running workflow: {e}")
        exit(1)
