def replace_variables(data, variables):
    """
    Recursively replaces variable references in data structures with their actual values
    Args:
        data: The data structure (dict or list) containing variable references
        variables: Dictionary of variable names and their values
    """
    if variables is not None:
        # Handle lists - replace any "variable:name" strings with their values
        if isinstance(data, list):
            for i, item in enumerate(data):
                # Check for variable reference format "variable:name"
                if isinstance(item, str) and item.startswith("variable:"):
                    variable_name = item.removeprefix("variable:")
                    if not variable_name in variables:
                        raise Exception(f"Variable <{variable_name}> not found")
                    data[i] = variables[variable_name]
                else:
                    # Recursively process nested structures
                    replace_variables(item, variables)

        # Handle dictionaries - replace values that are variable references
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str) and v.startswith("variable:"):
                    variable_name = v.removeprefix("variable:")
                    if not variable_name in variables:
                        raise Exception(f"Variable <{variable_name}> not found")
                    data[k] = variables[variable_name]
                else:
                    # Recursively process nested structures
                    replace_variables(v, variables)


def set_variables(values, variables):
    """
    Updates variable values while preserving their original types
    Args:
        values: Dictionary of new values to set
        variables: Dictionary of existing variables with their default values/types
    """
    for k, v in values.items():
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
    # Special handling for boolean string values
    if isinstance(v, str):
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False

    # Attempt type conversion, return original value if it fails
    try:
        return desired_type(v)
    except Exception:
        return v
