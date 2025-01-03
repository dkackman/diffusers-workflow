import argparse
import os
from .workflow import workflow_from_file
from . import startup

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a workflow from a file.")
    parser.add_argument(
        "file_name", type=str, help="The filespec of the workflow to validate"
    )

    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_name):
        raise FileNotFoundError(f"File {args.file_name} does not exist")

    startup(args.log_level)

    try:
        workflow = workflow_from_file(args.file_name, ".")
        workflow.validate()
        print("Workflow validated successfully")
    except Exception as e:
        print(f"Error validating workflow: {args.file_name}")
        exit(1)
