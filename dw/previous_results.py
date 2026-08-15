import logging
from itertools import product

logger = logging.getLogger("dw")

# Maximum number of iterations to prevent resource exhaustion
MAX_ITERATIONS = 10000


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
        # Shallow copy: realize_args may have already loaded large media
        # (PIL images, full video frame lists) into the template, so a deep
        # copy would multiply memory use. Contract: iteration dicts may only
        # be mutated at the top level (key pop/assign); nested values are
        # shared across iterations and must never be mutated in place.
        return [dict(argument_template)]

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
        # Create fresh shallow copy of template for each combination.
        # Nested values (e.g. loaded PIL images, video frame lists) are
        # shared across iterations, not deep-copied, to avoid multiplying
        # media memory usage by the iteration count. Contract: iteration
        # dicts may only be mutated at the top level (key pop/assign);
        # nested values must never be mutated in place.
        arguments = dict(argument_template)

        # Replace each reference with its actual value
        for key, value in zip(keys, values):
            # Handle nested dictionary properties
            # If value is dict and contains the key we're looking for, use that property
            arguments[key] = (
                value[key] if isinstance(value, dict) and key in value else value
            )
        iterations.append(arguments)

    # Safety check to prevent cartesian product explosion
    if len(iterations) > MAX_ITERATIONS:
        raise ValueError(
            f"Too many iterations generated: {len(iterations)} exceeds maximum of {MAX_ITERATIONS}. "
            f"This usually indicates too many previous_result references creating a cartesian product. "
            f"Consider reducing the number of multi-value results or splitting into multiple steps."
        )

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
    # Step names are unrestricted strings and may themselves contain dots
    # (e.g. "v1.0"), so resolve against the known step names rather than
    # blindly splitting on the first/only ".".

    # Exact match: the whole reference is a known step name, no property.
    if previous_result_name in previous_results:
        logger.debug(f"Getting all artifacts from result {previous_result_name}")
        return previous_results[previous_result_name].get_artifacts()

    if "." not in previous_result_name:
        raise KeyError(
            f"Previous result '{previous_result_name}' not found. Available results: {list(previous_results.keys())}"
        )

    # Find the longest known step name that is a prefix of the reference
    # followed by ".", and treat the remainder as the property name.
    result_name = max(
        (
            name
            for name in previous_results
            if previous_result_name.startswith(name + ".")
        ),
        key=len,
        default=None,
    )

    if result_name is None:
        raise KeyError(
            f"Previous result '{previous_result_name}' not found. Available results: {list(previous_results.keys())}"
        )

    property_name = previous_result_name[len(result_name) + 1 :]
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
