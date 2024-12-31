import logging
import PIL

logger = logging.getLogger("dh")


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
    Sets the values of variables from a dictionary of new values
    Args:
        values: Dictionary of new values to set
        variables: Dictionary of existing variables with their default values/types
    """
    logger.debug(f"Setting variables: {list(values.keys())}")

    if not isinstance(values, dict) or not isinstance(variables, dict):
        logger.error("Both values and variables must be dictionaries")
        raise TypeError("Both values and variables must be dictionaries")

    for k, v in values.items():
        logger.debug(f"Setting variable {k} to value: {v}")
        # Use the type of the existing variable to convert the new value
        variables[k] = get_value(v, type(variables[k]))


def get_value(v, desired_type):
    """
    Converts a value to the desired type, with special handling for booleans
    Args:
        v: Value to convert
        desired_type: Target type for conversion
    Returns:
        Converted value, or original value if conversion fails
    """
    logger.debug(f"Converting value {v} to type {desired_type}")

    if desired_type is None:
        logger.warning("No type specified for conversion, returning original value")
        return v

    # Special handling for boolean string values
    if isinstance(v, str):
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False

    # special handling for images that have already been realized
    elif isinstance(v, PIL.Image.Image):
        return v

    # Attempt type conversion, return original value if it fails
    try:
        converted = desired_type(v)
        logger.debug(f"Successfully converted to {desired_type.__name__}: {converted}")
        return converted
    except Exception as e:
        logger.warning(f"Failed to convert to {desired_type.__name__}: {e}")
        return v
