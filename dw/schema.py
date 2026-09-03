import json
import os
from jsonschema import validate, ValidationError


def validate_data(data, schema):
    try:
        validate(instance=data, schema=schema)
        return True, "Validation successful"

    except ValidationError as ve:
        path = json_path(ve.absolute_path)
        location = f" at {path}" if path else ""
        return False, f"Validation error{location}: {ve.message}"
    except json.JSONDecodeError as je:
        return False, f"JSON parsing error: {str(je)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def json_path(absolute_path):
    """Render a jsonschema ValidationError's absolute_path (a deque of dict
    keys and list indices) as a dotted/bracket path, e.g.
    steps[0].pipeline.arguments.prompt. None if the error is at the root."""
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


def load_schema(schema_name):
    file_spec = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"{schema_name}_schema.json"
    )
    with open(file_spec, "r") as file:
        return json.load(file)
