import logging
from itertools import product

from .arguments import (
    FROM_PREVIOUS_RESULT_KEY,
    PREVIOUS_RESULT_PREFIX,
    build_objects,
)

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

    # Create a dictionary mapping each reference path to its possible values
    # Example: {('image',): [img1, img2], ('prompt',): ['text1', 'text2']}
    ref_results = {
        ref_path: list(get_previous_results(previous_results, ref_value))
        for ref_path, ref_value in result_refs.items()
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
        for path, value in zip(keys, values):
            # Handle nested dictionary properties
            # If value is dict and contains the key we're looking for, use that property
            key = path[-1]
            arguments = substitute_at_path(
                arguments,
                path,
                value[key] if isinstance(value, dict) and key in value else value,
            )

        # Now that the media exists, build the objects that were waiting for it -
        # a reference constructed from a step's output rather than from a file
        iterations.append(build_objects(arguments))

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
    """Find all values in an argument structure that reference previous results.

    A reference is written either as a value with the "previous_result:" prefix, or
    as the step name a 'from_previous_result' object description is built from. Both
    are found at any depth: an argument that takes a constructed object holds it
    inside a list - MiniMax-H3's 'references' - so the reference is nested rather
    than sitting at the top of the arguments.

    Args:
        arguments: Dictionary of argument definitions

    Returns:
        Dict mapping the path of each reference to the result name it names. A path
        is the tuple of keys and list indices that reaches the value, so a top-level
        {'image': 'previous_result:step1'} comes back as {('image',): 'step1'}
    """
    found = {}
    _collect_refs(arguments, (), found)
    return found


def _collect_refs(value, path, found):
    """Walk an argument structure, collecting every reference by its path."""
    if isinstance(value, dict):
        for key, item in value.items():
            # The object description names its step bare, the way it would name a
            # file - the prefix would only repeat what the key already says
            if key == FROM_PREVIOUS_RESULT_KEY and isinstance(item, str):
                found[path + (key,)] = item
            else:
                _collect_refs(item, path + (key,), found)

    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_refs(item, path + (index,), found)

    elif isinstance(value, str) and value.startswith(PREVIOUS_RESULT_PREFIX):
        found[path] = value[len(PREVIOUS_RESULT_PREFIX) :]


def substitute_at_path(container, path, value):
    """A copy of container with value placed at path.

    Only the containers along the path are copied. Everything beside them stays
    shared, which is the same contract the top-level copy keeps: iterations share
    their nested values, so a substitution deep in one must not be visible in the
    others.

    Args:
        container: The dict or list to substitute into
        path: Tuple of keys and indices reaching the value to replace
        value: What to put there

    Returns:
        The copied container
    """
    key = path[0]
    replacement = (
        value if len(path) == 1 else substitute_at_path(container[key], path[1:], value)
    )

    if isinstance(container, list):
        copied = list(container)
        copied[key] = replacement
        return copied

    copied = dict(container)
    copied[key] = replacement
    return copied
