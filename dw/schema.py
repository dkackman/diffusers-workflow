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


# The parts of the workflow schema that can be asked for on their own, and
# what each holds. Whole, the schema is ~36 KB - ~8.6k tokens in one call,
# spent by an agent that needed the shape of a `result` block (#101). A
# section answers that question at a tenth the size; the no-argument call
# is unchanged, so nothing that reads the whole schema is affected.
SCHEMA_SECTIONS = {
    "variables": {
        "properties": [
            "id",
            "description",
            "summary",
            "shape",
            "traits",
            "configures",
            "cost",
            "variables",
            "seed",
        ],
        "defs": [],
    },
    "steps": {"properties": ["steps"], "defs": ["step", "workflow_reference"]},
    "pipelines": {
        "properties": [],
        "defs": ["pipeline", "pipeline_reference", "chain", "arguments", "image"],
    },
    "tasks": {"properties": [], "defs": ["task"]},
    "result": {"properties": [], "defs": ["result"]},
    "configuration": {
        "properties": [],
        "defs": [
            "pipeline_configuration",
            "pipeline_component",
            "shared_components",
            "reused_components",
            "quantization_config",
            "scheduler",
            "lora",
            "controlnet",
            "ip_adapter",
            "from_pretrained_arguments",
            "group_offload",
            "compile_config",
            "enable_layerwise_casting",
        ],
    },
}

DEFS_KEY = "$defs"
DEFS_REF_PREFIX = f"#/{DEFS_KEY}/"


class SchemaSectionError(LookupError):
    """A section name the schema has no part for. The message names the
    ones it does, so a route can hand it back as a 404 detail."""


def _referenced_defs(node, found):
    """Every '#/$defs/x' name reachable from a subtree, into `found`."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(DEFS_REF_PREFIX):
            found.add(ref[len(DEFS_REF_PREFIX) :])
        for value in node.values():
            _referenced_defs(value, found)
    elif isinstance(node, list):
        for value in node:
            _referenced_defs(value, found)
    return found


def schema_section(schema, section):
    """One part of the workflow schema, plus where the rest of it is.

    The fragment carries only the definitions this section owns; a `$ref`
    to a definition another section owns is left standing and named in
    `elsewhere` ({definition: section}), because expanding it transitively
    would pull most of the schema back in and there would be no section
    left to speak of. `sections` lists every section name.

    Raises:
        SchemaSectionError: If `section` is not one of SCHEMA_SECTIONS.
    """
    if section not in SCHEMA_SECTIONS:
        raise SchemaSectionError(
            f"No schema section '{section}'. The sections are: "
            f"{', '.join(sorted(SCHEMA_SECTIONS))}."
        )
    wanted = SCHEMA_SECTIONS[section]
    defs = schema.get(DEFS_KEY, {})
    fragment = {
        key: schema[key]
        for key in ("$schema", "$id", "type", "description")
        if key in schema
    }
    properties = {
        name: schema["properties"][name]
        for name in wanted["properties"]
        if name in schema.get("properties", {})
    }
    if properties:
        fragment["properties"] = properties
        required = [name for name in schema.get("required", []) if name in properties]
        if required:
            fragment["required"] = required
    held = {name: defs[name] for name in wanted["defs"] if name in defs}
    if held:
        fragment[DEFS_KEY] = held
    owner = {
        name: other for other, part in SCHEMA_SECTIONS.items() for name in part["defs"]
    }
    referenced = _referenced_defs(fragment, set())
    elsewhere = {
        name: owner[name] for name in sorted(referenced - set(held)) if name in owner
    }
    return {
        "section": section,
        "sections": sorted(SCHEMA_SECTIONS),
        "elsewhere": elsewhere,
        "schema": fragment,
    }
