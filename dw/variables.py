import copy
import logging
import PIL
from .security import (
    validate_variable_name,
    validate_string_input,
    SecurityError,
    MAX_VARIABLE_VALUE_LENGTH,
)

logger = logging.getLogger("dw")


class VariableNotFoundError(ValueError):
    """Raised when a workflow references a "variable:name" that isn't declared."""


def _resolve_variable_reference(value, variables):
    """
    If value is a "variable:name" reference, look it up and return (True, resolved).
    Otherwise return (False, None) so the caller knows to recurse instead.

    Raises:
        VariableNotFoundError: if the referenced name isn't in variables, naming the
            variables that are actually available.
    """
    if isinstance(value, str) and value.startswith("variable:"):
        variable_name = value.removeprefix("variable:")
        logger.debug(f"Replacing variable reference: {variable_name}")
        if variable_name not in variables:
            available = ", ".join(sorted(variables.keys())) or "<none>"
            message = f"Variable <{variable_name}> not found; available variables: {available}"
            raise VariableNotFoundError(message)
        return True, variables[variable_name]
    return False, None


def replace_variables(data, variables):
    """
    Recursively replaces variable references in data structures with their actual values.

    Does not mutate its input - a new structure is returned and `data` is left as it
    was passed in, so callers don't have to deep-copy defensively before calling.

    Args:
        data: The data structure (dict or list) containing variable references
        variables: Dictionary of variable names and their values
    Returns:
        A new structure with "variable:name" references replaced. Any part of `data`
        that isn't a dict/list/reference string is returned as-is.
    """
    if variables is None:
        return data

    logger.debug(f"Processing variables: {list(variables.keys())}")

    # Handle lists - replace any "variable:name" strings with their values
    if isinstance(data, list):
        logger.debug(f"Processing list of length {len(data)}")
        result = []
        for item in data:
            matched, resolved = _resolve_variable_reference(item, variables)
            if matched:
                result.append(resolved)
            else:
                # Recursively process nested structures
                result.append(replace_variables(item, variables))
        return result

    # Handle dictionaries - replace values that are variable references
    if isinstance(data, dict):
        logger.debug(f"Processing dictionary with keys: {list(data.keys())}")
        result = {}
        for k, v in data.items():
            matched, resolved = _resolve_variable_reference(v, variables)
            if matched:
                result[k] = resolved
            else:
                # Recursively process nested structures in dictionary values
                result[k] = replace_variables(v, variables)
        return result

    # Scalars (and anything else) pass through unchanged. copy.deepcopy guards
    # against a caller mutating a returned mutable leaf (e.g. a PIL.Image or a
    # list-typed variable's value) and having that reach back into `variables`.
    return copy.deepcopy(data)


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
        Converted value

    Raises:
        ValueError: if v cannot be converted to desired_type, naming the variable,
            its target type, and the offending value.
    """
    logger.debug(f"Converting value {v} to type {desired_type}")

    # A variable declared null is an optional one the workflow states no type
    # for - passing a value to it is the expected case, not a suspicious one
    if desired_type is None or desired_type is type(None):
        logger.debug("Variable has no declared type, using the value as given")
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

    # A string cannot be coerced into a dict or a None - dict('/a/b.png') is
    # nonsense, NoneType('x') a TypeError. Those defaults are how media
    # variables ({'location': ...}) and optional inputs (null) are declared,
    # and a string override is a path or a reference that realize_args
    # resolves later, so it passes through as written
    if isinstance(v, str) and desired_type in (dict, type(None)):
        return v

    # Attempt type conversion. A failure here is surfaced immediately with a clear,
    # named error instead of silently passing the unconverted value through - letting
    # it through would fail several layers later inside diffusers/torch with a
    # confusing traceback that doesn't mention the variable at fault.
    try:
        converted = desired_type(v)
        logger.debug(f"Successfully converted to {desired_type.__name__}: {converted}")
        return converted
    except Exception as e:
        var_label = name if name is not None else "<unknown>"
        message = (
            f"Cannot convert variable '{var_label}' value {v!r} to type "
            f"{desired_type.__name__}: {e}"
        )
        logger.error(message)
        raise ValueError(message) from e
