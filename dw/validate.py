import argparse
import os
from .workflow import workflow_from_file
from . import startup
from .security import validate_workflow_path, set_trust_workflows, SecurityError


def main():
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
    parser.add_argument(
        "--trust-workflows",
        action="store_true",
        default=False,
        help="Trust this workflow file to execute arbitrary Python: allow "
        "pre_load_modules and any dotted *_type/*_dtype/dtype/config_type "
        "value, not just ones inside the diffusers/torch/transformers/"
        "quantization-backend ecosystem the tool already depends on. Off "
        "by default - see docs/SECURITY.md's Trust model. Only pass this "
        "for a workflow file whose source you trust.",
    )
    args = parser.parse_args()

    set_trust_workflows(args.trust_workflows)

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
    except Exception as e:
        print(f"Error validating workflow '{args.file_name}': {e}")
        exit(1)
        return

    try:
        # Workflow.validate() names the JSON path of a schema failure and
        # carries the 'Validation error' prefix exactly once
        workflow.validate()
        print("Workflow validated successfully")
    except Exception as e:
        print(
            str(e)
            if str(e).startswith("Validation error")
            else f"Error validating workflow '{args.file_name}': {e}"
        )
        exit(1)


if __name__ == "__main__":
    main()
