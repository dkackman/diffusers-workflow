import json
import os
from jsonschema import validate, ValidationError


def validate_data(data, schema):
    try:
        validate(instance=data, schema=schema)
        return True, "Validation successful"

    except ValidationError as ve:
        return False, f"Validation error: {ve.message}"
    except json.JSONDecodeError as je:
        return False, f"JSON parsing error: {str(je)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def load_schema(schema_name):
    file_spec = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"{schema_name}_schema.json"
    )
    with open(file_spec, "r") as file:
        return json.load(file)
