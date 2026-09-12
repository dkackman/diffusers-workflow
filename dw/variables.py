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


class VariableCycleError(ValueError):
    """Raised when a list- or dict-valued variable references itself, directly
    or through others, inside `resolve_variable_values`."""


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


def resolve_variable_values(variables):
    """A copy of `variables` in which every "variable:name" inside a list-
    or dict-valued variable is replaced by that variable's value.

    A list-driven step reads its entries from a variable, and an entry that
    says "from_file": "variable:character_a_voice" is how one variable sets
    a voice in every shot the character speaks in. replace_variables only
    walks the definition, so those references would reach the step as the
    literal strings; this resolves them once, before realize_args, so a
    reference type inside an entry is a type name by the time it is loaded.

    Only list and dict values are walked. A scalar value that begins with
    "variable:" is passed through as it always was.

    Raises:
        VariableNotFoundError: a reference names nothing declared
        VariableCycleError: a value references itself, directly or through others
    """
    resolved = {}

    def resolve(name, chain):
        if name in resolved:
            return resolved[name]
        if name in chain:
            loop = " -> ".join(chain[chain.index(name) :] + [name])
            raise VariableCycleError(
                f"Variable '{name}' references itself through: {loop}"
            )
        value = variables[name]
        if isinstance(value, (list, dict)):
            value = walk(value, chain + [name])
        else:
            value = copy.deepcopy(value)
        resolved[name] = value
        return value

    def walk(node, chain):
        matched, _ = _resolve_variable_reference(node, variables)
        if matched:
            return resolve(node.removeprefix("variable:"), chain)
        if isinstance(node, list):
            return [walk(item, chain) for item in node]
        if isinstance(node, dict):
            return {key: walk(item, chain) for key, item in node.items()}
        return copy.deepcopy(node)

    for name in variables:
        resolve(name, [])
    return resolved


def undeclared_variable_references(definition):
    """The "variable:name" references in a workflow definition that name no
    entry of its `variables` - the ones `replace_variables` will refuse at run
    time, found before anything loads.

    Walks everything but `variables` itself, plus the inside of every list-
    or dict-valued variable, the way `resolve_variable_values` and
    `replace_variables` together do. Returns a list of (path, name) pairs,
    path being where the reference sits (`steps[0].pipeline.arguments.prompt`)
    and name what it asked for - which is the whole remainder of the string,
    since a reference is the entire value and nothing is interpolated
    around it.
    """
    declared = definition.get("variables") or {}
    found = []

    def walk(node, path):
        if isinstance(node, str) and node.startswith("variable:"):
            name = node.removeprefix("variable:")
            if name not in declared:
                found.append((path, name))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    for key, value in definition.items():
        if key != "variables":
            walk(value, key)
    if isinstance(declared, dict):
        for name, value in declared.items():
            if isinstance(value, (list, dict)):
                walk(value, f"variables.{name}")
    return found


def _validated_strings(value):
    """A copy of a list- or dict-valued caller argument with every string
    leaf passed through `validate_string_input` - the same check a
    top-level string argument gets in `set_variables` below. A for_each
    entry's `prompt` or `from_file` is exactly as reachable to an attacker
    as a top-level variable, and `isinstance(v, str)` alone would skip it.

    Non-string leaves (numbers, bools, None, nested lists/dicts) pass
    through unchanged; only str is validated.
    """
    if isinstance(value, str):
        return validate_string_input(
            value, max_length=MAX_VARIABLE_VALUE_LENGTH, allow_empty=True
        )
    if isinstance(value, list):
        return [_validated_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _validated_strings(item) for key, item in value.items()}
    return value


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

            # Validate string values - including strings nested inside a
            # list- or dict-valued argument, which a for_each entry's
            # prompt or from_file always is
            if isinstance(v, str):
                validated_value = validate_string_input(
                    v, max_length=MAX_VARIABLE_VALUE_LENGTH, allow_empty=True
                )
            elif isinstance(v, (list, dict)):
                validated_value = _validated_strings(v)
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


def argument_errors(definition, arguments):
    """What is wrong with a caller's `arguments` for this workflow, before
    anything is queued.

    Exactly the check `set_variables` makes at the top of a run - an
    undeclared name, a value that will not coerce to the type the default
    declares - made against a copy, so nothing is mutated and the answer
    costs no GPU time. A workflow that declares no variables at all takes no
    arguments: today those are dropped in silence (`Workflow.run` only
    substitutes when a `variables` block exists), which is the one case the
    run itself does not report.

    Args:
        definition: A workflow definition
        arguments: The caller's argument dict; empty or None means no errors

    Returns:
        [{"path": "arguments.<name>", "message": str}, ...], one per bad
        argument, in the order they were given
    """
    if not arguments:
        return []
    if not isinstance(arguments, dict):
        return [{"path": "arguments", "message": "arguments must be an object"}]

    declared = definition.get("variables") if isinstance(definition, dict) else None
    if not isinstance(declared, dict) or not declared:
        return [
            {
                "path": f"arguments.{name}",
                "message": f"This workflow declares no variables, so '{name}' "
                "has nowhere to land - it would be ignored by the run",
            }
            for name in arguments
        ]

    errors = []
    for name, value in arguments.items():
        # One at a time against a fresh copy, so each bad argument is
        # reported with its own name rather than the first one stopping the
        # rest from being checked
        try:
            set_variables({name: value}, copy.deepcopy(declared))
        except (ValueError, TypeError, SecurityError) as e:
            errors.append({"path": f"arguments.{name}", "message": str(e)})
    return errors
