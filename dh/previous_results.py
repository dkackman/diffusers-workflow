import copy
from itertools import product


def get_iterations(argument_template, previous_results):
    """Generate argument combinations using previous task results.

    Args:
        argument_template: Base arguments dictionary with potential result references
        previous_results: Dictionary of results from previous tasks

    Returns:
        List of argument dictionaries with all possible combinations
    """
    if isinstance(argument_template, list):
        return argument_template

    # Find all references to previous results in arguments
    result_refs = find_previous_result_refs(argument_template)

    # If no references found, return a copy of the template template as-is
    if len(result_refs) == 0:
        return [copy.deepcopy(argument_template)]

    # For each reference, get all matching previous results
    ref_results = {}
    for ref_key, ref_value in result_refs.items():
        ref_results[ref_key] = list(get_previous_results(previous_results, ref_value))

    # Get keys for cartesian product
    keys = list(ref_results.keys())
    # Generate all possible combinations of values using itertools.product
    value_combinations = product(*[ref_results[k] for k in keys])

    # Create new argument dict for each combination
    iterations = []
    for values in value_combinations:
        arguments = copy.deepcopy(argument_template)
        # Replace reference placeholders with actual values
        for key, value in zip(keys, values):
            arguments[key] = value
        iterations.append(arguments)

    return iterations


def get_previous_results(previous_results, previous_result_name):
    if "." in previous_result_name:
        # this is named property of the previous result parse and get that property
        parts = previous_result_name.split(".")
        return previous_results[parts[0]].get_artifact_properties(parts[1])

    return previous_results[previous_result_name].get_artifacts()


def find_previous_result_refs(arguments):
    """Find all values in arguments dict that reference previous results.

    Args:
        arguments: Dictionary of arguments to check

    Returns:
        Dict containing only key/value pairs where value starts with "previous_result:"
    """
    prefix = "previous_result:"
    return {
        k: v[len(prefix) :]
        for k, v in arguments.items()
        if isinstance(v, str) and v.startswith(prefix)
    }
