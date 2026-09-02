import argparse
import os
from jsonschema import validate as jsonschema_validate, ValidationError
from .workflow import workflow_from_file
from .schema import load_schema
from . import startup
from .security import validate_workflow_path, SecurityError


def _json_path(absolute_path):
    """
    Render a jsonschema ValidationError's absolute_path (a deque of dict keys and
    list indices) as a dotted/bracket path, e.g. steps[0].pipeline.arguments.prompt.
    Returns None if the path is empty (the error is at the document root).
    """
    if not absolute_path:
        return None

    parts = []
    for element in absolute_path:
        if isinstance(element, int):
            parts.append(f"[{element}]")
        elif not parts:
            parts.append(str(element))
        else:
            parts.append(f".{element}")
    return "".join(parts)


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
    except Exception as e:
        print(f"Error validating workflow '{args.file_name}': {e}")
        exit(1)
        return

    # Validate against the schema directly (rather than via Workflow.validate(),
    # which discards the jsonschema ValidationError down to a plain string) so the
    # JSON path of the failure - e.g. steps[0].pipeline.arguments.prompt - can be
    # reported, and the "Validation error:" prefix is added exactly once.
    try:
        jsonschema_validate(
            instance=workflow.workflow_definition, schema=load_schema("workflow")
        )
        print("Workflow validated successfully")
    except ValidationError as ve:
        path = _json_path(ve.absolute_path)
        location = f" at {path}" if path else ""
        print(f"Validation error{location}: {ve.message}")
        exit(1)
    except Exception as e:
        print(f"Error validating workflow '{args.file_name}': {e}")
        exit(1)


if __name__ == "__main__":
    main()
