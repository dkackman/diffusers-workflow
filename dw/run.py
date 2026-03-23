import argparse
import os
from . import startup
from .workflow import workflow_from_file
from .security import (
    validate_workflow_path,
    validate_output_path,
    validate_variable_name,
    validate_string_input,
    SecurityError,
    MAX_VARIABLE_VALUE_LENGTH,
)

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
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    # Parse key-value pairs with validation
    variables = {}
    for variable in args.variables:
        try:
            name, value = variable.split("=", 1)
            # Validate variable name and value
            validated_name = validate_variable_name(name.strip())
            validated_value = validate_string_input(
                value.strip(), max_length=MAX_VARIABLE_VALUE_LENGTH, allow_empty=True
            )
            variables[validated_name] = validated_value
        except ValueError:
            print(f"Error: Variable '{variable}' is not in name=value format")
            exit(1)
        except SecurityError as e:
            print(f"Error: Invalid variable input: {e}")
            exit(1)

    # Validate and secure file paths
    try:
        validated_output_dir = validate_output_path(args.output_dir, None)
        if not os.path.exists(validated_output_dir):
            # Create output directory if it doesn't exist
            os.makedirs(validated_output_dir, exist_ok=True)
            print(f"Created output directory: {validated_output_dir}")

        validated_file_path = validate_workflow_path(args.file_name)
        if not os.path.exists(validated_file_path):
            raise FileNotFoundError(f"File {validated_file_path} does not exist")

    except SecurityError as e:
        print(f"Error: Security validation failed: {e}")
        exit(1)

    startup(args.log_level)

    workflow = workflow_from_file(validated_file_path, validated_output_dir)
    try:
        workflow.validate()
    except Exception as e:
        print(f"Error validating workflow '{args.file_name}': {e}")
        exit(1)

    try:
        workflow.run(variables)
    except Exception as e:
        print(f"Error running workflow '{args.file_name}': {e}")
        exit(1)
