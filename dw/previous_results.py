import copy
import logging
from itertools import product

logger = logging.getLogger("dw")


def get_iterations(argument_template, previous_results):
    """Generate argument combinations using previous task results.

    Takes a template of arguments and expands any references to previous results
    into all possible combinations of those results.

    Args:
        argument_template: Dict or list containing argument definitions
        previous_results: Dict of results from previously executed steps

    Returns:
        List of argument dictionaries, one for each possible combination
    """
    # Special case: if template is a list, use it directly without processing
    if isinstance(argument_template, list):
        logger.debug("Using list argument template directly")
        return argument_template

    # Find any references to previous results in the template
    # Returns dict of {arg_key: result_reference}
    result_refs = find_previous_result_refs(argument_template)

    # If no references found, return the template as-is
    if not result_refs:
        logger.debug("No result references found in template")
        return [copy.deepcopy(argument_template)]

    logger.debug(f"Found {len(result_refs)} result references: {result_refs}")

    # Create a dictionary mapping each reference key to its possible values
    # Example: {'image': [img1, img2], 'prompt': ['text1', 'text2']}
    ref_results = {
        ref_key: list(get_previous_results(previous_results, ref_value))
        for ref_key, ref_value in result_refs.items()
    }

    # Generate all possible combinations of argument values
    keys = list(ref_results.keys())
    iterations = []

    # Use itertools.product to create cartesian product of all possible values
    # Example: if ref_results has 2 images and 2 prompts, creates 4 combinations
    for values in product(*[ref_results[k] for k in keys]):
        # Create fresh copy of template for each combination
        arguments = copy.deepcopy(argument_template)

        # Replace each reference with its actual value
        for key, value in zip(keys, values):
            # Handle nested dictionary properties
            # If value is dict and contains the key we're looking for, use that property
            arguments[key] = (
                value[key] if isinstance(value, dict) and key in value else value
            )
        iterations.append(arguments)

    logger.debug(f"Generated {len(iterations)} argument combinations")
    return iterations


def get_previous_results(previous_results, previous_result_name):
    """Retrieve results or specific properties from previous tasks.

    Args:
        previous_results: Dict of results from previous steps
        previous_result_name: String identifying the result, optionally with property
                            Format: "step_name" or "step_name.property_name"

    Returns:
        List of results or specific properties from the referenced step
    """
    # Check if we're looking for a specific property
    if "." not in previous_result_name:
        logger.debug(f"Getting all artifacts from result {previous_result_name}")
        return previous_results[previous_result_name].get_artifacts()

    # Split into result name and property name
    result_name, property_name = previous_result_name.split(".")
    logger.debug(f"Getting property {property_name} from result {result_name}")
    return previous_results[result_name].get_artifact_properties(property_name)


def find_previous_result_refs(arguments):
    """Find all values in arguments dict that reference previous results.

    Looks for string values starting with "previous_result:" and creates a mapping
    of argument keys to their referenced result names.

    Args:
        arguments: Dictionary of argument definitions

    Returns:
        Dict mapping argument keys to their referenced result names
        Example: {'image': 'step1', 'prompt': 'step2.text'}
    """
    prefix = "previous_result:"
    return {
        k: v[len(prefix) :]
        for k, v in arguments.items()
        if isinstance(v, str) and v.startswith(prefix)
    }
