import argparse
import os
from .workflow import workflow_from_file
from . import startup
from .security import validate_workflow_path, SecurityError

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

    try:
        validated_file_path = validate_workflow_path(args.file_name)
        if not os.path.exists(validated_file_path):
            raise FileNotFoundError(f"File {validated_file_path} does not exist")
    except SecurityError as e:
        print(f"Error: Security validation failed: {e}")
        exit(1)

    startup(args.log_level)

    try:
        workflow = workflow_from_file(validated_file_path, ".")
        workflow.validate()
        print("Workflow validated successfully")
    except Exception as e:
        print(f"Error validating workflow '{args.file_name}': {e}")
        exit(1)
