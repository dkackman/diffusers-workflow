import argparse
import os
from .workflow import workflow_from_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a workflow from a file.")
    parser.add_argument(
        "file_name", type=str, help="The filespec of the workflow to validate"
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_name):
        raise FileNotFoundError(f"File {args.file_name} does not exist")

    try:
        workflow = workflow_from_file(args.file_name, ".")
        workflow.validate()
    except Exception as e:
        print(f"Error validating workflow: {args.file_name}")
        exit(1)

    print(f"Workflow {args.file_name} validated successfully")
