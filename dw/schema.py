import json
import os
from jsonschema import validate, ValidationError
from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for


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


# anyOf branches produce dozens of near-identical entries; this is more than
# an agent fixes in one pass, and enough that nothing real is hidden
MAX_VALIDATION_ERRORS = 25


def validate_data_all(data, schema):
    """Every schema violation in `data`, as [{path, message}].

    Sorted by path, deduplicated on (path, message), capped at
    MAX_VALIDATION_ERRORS. Each top-level error is reduced with best_match
    - the same descent into anyOf branches jsonschema.validate performs to
    pick the one exception it raises - so a definition with a single
    violation is reported exactly as validate_data reports it.
    """
    validator = validator_for(schema)(schema)
    seen = {}
    for error in validator.iter_errors(data):
        chosen = best_match([error])
        key = (json_path(chosen.absolute_path), chosen.message)
        seen.setdefault(key, None)
    ordered = sorted(seen, key=lambda key: (key[0] or "", key[1]))
    return [
        {"path": path, "message": message}
        for path, message in ordered[:MAX_VALIDATION_ERRORS]
    ]


def format_validation_errors(errors):
    """The message a raised validation failure carries.

    One error keeps the line every caller already shows -
    'Validation error at <path>: <message>'. Several are listed one per
    line under a single heading, so the text 'Validation error' still
    appears once per failure (the CLI and the REPL count on that).
    """
    if len(errors) == 1:
        path, message = errors[0]["path"], errors[0]["message"]
        location = f" at {path}" if path else ""
        return f"Validation error{location}: {message}"
    count = (
        f"first {MAX_VALIDATION_ERRORS}"
        if len(errors) >= MAX_VALIDATION_ERRORS
        else str(len(errors))
    )
    lines = [f"Validation errors ({count}):"]
    for error in errors:
        lines.append(f"  at {error['path'] or 'root'}: {error['message']}")
    return "\n".join(lines)


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
