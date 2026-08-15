import logging
import PIL
from .security import (
    validate_variable_name,
    validate_string_input,
    SecurityError,
    MAX_VARIABLE_VALUE_LENGTH,
)

logger = logging.getLogger("dw")


def replace_variables(data, variables):
    """
    Recursively replaces variable references in data structures with their actual values
    Args:
        data: The data structure (dict or list) containing variable references
        variables: Dictionary of variable names and their values
    """
    if variables is not None:
        logger.debug(f"Processing variables: {list(variables.keys())}")

        # Handle lists - replace any "variable:name" strings with their values
        if isinstance(data, list):
            logger.debug(f"Processing list of length {len(data)}")
            for i, item in enumerate(data):
                # Check for variable reference format "variable:name"
                if isinstance(item, str) and item.startswith("variable:"):
                    variable_name = item.removeprefix("variable:")
                    logger.debug(f"Replacing variable reference: {variable_name}")
                    if not variable_name in variables:
                        logger.error(f"Variable <{variable_name}> not found")
                        raise Exception(f"Variable <{variable_name}> not found")
                    data[i] = variables[variable_name]
                else:
                    # Recursively process nested structures
                    replace_variables(item, variables)

        # Handle dictionaries - replace values that are variable references
        elif isinstance(data, dict):
            logger.debug(f"Processing dictionary with keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, str) and v.startswith("variable:"):
                    variable_name = v.removeprefix("variable:")
                    logger.debug(f"Replacing variable reference: {variable_name}")
                    if not variable_name in variables:
                        logger.error(f"Variable <{variable_name}> not found")
                        raise Exception(f"Variable <{variable_name}> not found")
                    data[k] = variables[variable_name]
                else:
                    # Recursively process nested structures in dictionary values
                    replace_variables(v, variables)


def set_variables(values, variables):
    """
    Sets the values of variables from a dictionary of new values with validation
    Args:
        values: Dictionary of new values to set
        variables: Dictionary of existing variables with their default values/types
    """
    logger.debug(f"Setting variables: {list(values.keys())}")

    if not isinstance(values, dict) or not isinstance(variables, dict):
        logger.error("Both values and variables must be dictionaries")
        raise TypeError("Both values and variables must be dictionaries")

    for k, v in values.items():
        try:
            # Validate variable name
            validated_name = validate_variable_name(k)

            # The workflow must have already declared this variable (with a default
            # value/type) - reject unknown names instead of raising a bare KeyError
            if validated_name not in variables:
                declared = ", ".join(sorted(variables.keys()))
                logger.error(
                    f"Unknown variable '{validated_name}'; declared variables: {declared}"
                )
                raise ValueError(
                    f"Unknown variable '{validated_name}'; declared variables: {declared}"
                )

            # Validate string values
            if isinstance(v, str):
                validated_value = validate_string_input(
                    v, max_length=MAX_VARIABLE_VALUE_LENGTH, allow_empty=True
                )
            else:
                validated_value = v

            logger.debug(
                f"Setting variable {validated_name} to value: {validated_value}"
            )
            # Use the type of the existing variable to convert the new value
            variables[validated_name] = get_value(
                validated_value, type(variables[validated_name]), validated_name
            )

        except SecurityError as e:
            logger.error(f"Security validation failed for variable {k}: {e}")
            raise


def get_value(v, desired_type, name=None):
    """
    Converts a value to the desired type, with special handling for booleans
    Args:
        v: Value to convert
        desired_type: Target type for conversion
        name: Name of the variable being converted, used for error messages
    Returns:
        Converted value, or original value if conversion fails
    """
    logger.debug(f"Converting value {v} to type {desired_type}")

    if desired_type is None or desired_type is type(None):
        logger.warning("No type specified for conversion, returning original value")
        return v

    # Special handling for boolean string values - bool("0") and bool("no") are
    # both truthy in Python, which would silently invert the user's intent, so
    # only a known set of true/false spellings is accepted here
    if isinstance(v, str) and desired_type is bool:
        lowered = v.lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        var_label = name if name is not None else "<unknown>"
        message = f"Cannot interpret '{v}' as true/false for variable '{var_label}'"
        logger.error(message)
        raise ValueError(message)

    # Special handling for list string values - list("cat") would mangle the
    # string into ['c', 'a', 't'], so a comma-separated string is split instead
    if isinstance(v, str) and desired_type is list:
        return [item.strip() for item in v.split(",")]

    # special handling for images that have already been realized
    if isinstance(v, PIL.Image.Image):
        return v

    # Attempt type conversion, return original value if it fails
    try:
        converted = desired_type(v)
        logger.debug(f"Successfully converted to {desired_type.__name__}: {converted}")
        return converted
    except Exception as e:
        logger.warning(f"Failed to convert to {desired_type.__name__}: {e}")
        return v
