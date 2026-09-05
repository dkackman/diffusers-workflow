import argparse
import os
from . import startup
from .workflow import workflow_from_file
from .workspace import resolve_workspace, set_workspace
from .security import (
    validate_workflow_path,
    validate_output_path,
    validate_variable_name,
    validate_string_input,
    set_trust_workflows,
    SecurityError,
    MAX_VARIABLE_VALUE_LENGTH,
)


def main():
    parser = argparse.ArgumentParser(description="Run a workflow from a file.")
    parser.add_argument(
        "file_name", type=str, help="The filespec to of the workflow to run"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="The folder to write the outputs to (default: the workspace's "
        "outputs/ - ./outputs when run from a workspace, as a checkout is)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Directory holding your workflows, prompts, assets and outputs "
        "(default: DW_WORKSPACE, else the 'workspace' setting, else the "
        "working directory when it looks like a workspace, else "
        "~/diffusers-workspace)",
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
    parser.add_argument(
        "--prompt-dir",
        type=str,
        default=None,
        help="Directory 'prompt:' references resolve against (default: "
        "DW_PROMPT_DIR, else ./prompts if it exists, else the nearest "
        "prompts/ above the workflow file)",
    )
    parser.add_argument(
        "--asset-dir",
        type=str,
        default=None,
        help="Directory 'asset:' references resolve against (default: "
        "DW_ASSET_DIR, else the workspace's assets/ when a workspace was "
        "named, else ./assets if it exists, else the nearest assets/ above "
        "the workflow file)",
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
    # parse_intermixed_args, not parse_args: argparse splits positionals into
    # groups around an option, and with a nargs='*' positional that means
    # 'dw-run wf.json -o outputs prompt=cat' - an option between the file
    # and the name=value pairs - fails as unrecognized arguments
    args = parser.parse_intermixed_args()

    # Pinned before anything resolves a directory from it, and exported so a
    # subprocess sees the same answer
    workspace = set_workspace(resolve_workspace(args.workspace))
    output_dir = args.output_dir or workspace.outputs

    if args.prompt_dir:
        os.environ["DW_PROMPT_DIR"] = os.path.abspath(args.prompt_dir)

    if args.asset_dir:
        os.environ["DW_ASSET_DIR"] = os.path.abspath(args.asset_dir)

    set_trust_workflows(args.trust_workflows)

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
        validated_output_dir = validate_output_path(output_dir, None)
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


if __name__ == "__main__":
    main()
