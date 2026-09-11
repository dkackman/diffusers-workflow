import logging
from itertools import product

from .arguments import (
    FROM_PREVIOUS_RESULT_KEY,
    PREVIOUS_RESULT_PREFIX,
    build_objects,
)
from .for_each import MEMBER_SEPARATOR, render_path
from .step_cache import reference_resolves_to

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

    # Guard the cartesian product BEFORE building it - the product's size is
    # known from the reference counts, and checking after the loop would only
    # report the resource exhaustion it exists to prevent
    combinations = 1
    for key in keys:
        combinations *= len(ref_results[key])
    if combinations > MAX_ITERATIONS:
        raise ValueError(
            f"Too many iterations generated: {combinations} exceeds maximum of {MAX_ITERATIONS}. "
            f"This usually indicates too many previous_result references creating a cartesian product. "
            f"Consider reducing the number of multi-value results or splitting into multiple steps."
        )

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
        raise _not_found(previous_results, previous_result_name)

    # Find the longest known step name the reference resolves to, and treat
    # the remainder as the property name. The exact-match case returned
    # above, so every name reaching here is a strict '<name>.' prefix.
    result_name = max(
        (
            name
            for name in previous_results
            if reference_resolves_to(previous_result_name, name)
        ),
        key=len,
        default=None,
    )

    if result_name is None:
        raise _not_found(previous_results, previous_result_name)

    property_name = previous_result_name[len(result_name) + 1 :]
    logger.debug(f"Getting property {property_name} from result {result_name}")
    return previous_results[result_name].get_artifact_properties(property_name)


def resolve_chain_prompts(step_action, previous_results):
    """Resolve a pipeline chain's per-segment prompts against previous results.

    A chain's "prompts" list is not part of the step's argument template, so the
    cartesian pass that expands "previous_result:" everywhere else never reaches
    it. That matters for a chain whose opening segment is written by a different
    step from the ones that continue it - a continuation prompt declares a video
    reference the first segment does not have.

    Each entry resolves independently and yields one prompt, so this never
    multiplies iterations the way an argument reference does; a reference that
    produced several artifacts uses the first.

    The resolved list is left on the step action for run_chain to pick up, and
    nothing happens at all for a pipeline without chain prompts.
    """
    definition = getattr(step_action, "pipeline_definition", None)
    if not isinstance(definition, dict):
        return

    chain = definition.get("chain", None) or {}
    prompts = chain.get("prompts", None)
    if not prompts:
        return

    resolved = []
    for entry in prompts:
        if isinstance(entry, str) and entry.startswith("previous_result:"):
            artifacts = get_previous_results(
                previous_results, entry.removeprefix("previous_result:")
            )
            if not artifacts:
                raise ValueError(f"Chain prompt reference '{entry}' produced no result")
            if len(artifacts) > 1:
                logger.warning(
                    f"Chain prompt reference '{entry}' produced {len(artifacts)} "
                    f"results - using the first"
                )
            entry = artifacts[0]
        resolved.append(entry)

    step_action.chain_prompts = resolved


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


class StepResults(dict):
    """The run's results, which also remembers every step that produced one.

    `release_unreferenced_results` deletes a result the moment no remaining
    step references it, so by the time a misspelled reference fails, the
    steps that ran are gone from the dict and the error printed
    'Available results: []' on a run where several steps had completed
    (T015). Keeping the names - not the results, which is the whole point of
    releasing them - costs nothing and is the one thing that diagnostic
    needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completed_steps = list(self.keys())

    def __setitem__(self, key, value):
        if key not in self.completed_steps:
            self.completed_steps.append(key)
        super().__setitem__(key, value)


def _not_found(previous_results, previous_result_name):
    """The error a reference that names nothing raises.

    Names what is there as well as what was asked for: the gap between them
    is the fix, and on a long run it is the only thing standing between a
    typo and another 40 minutes of GPU.
    """
    message = (
        f"Previous result '{previous_result_name}' not found. "
        f"Available results: {list(previous_results.keys())}"
    )
    released = [
        name
        for name in getattr(previous_results, "completed_steps", ())
        if name not in previous_results
    ]
    if released:
        message += (
            f". Earlier steps that ran: {released} - their results were "
            f"released because no remaining step references them"
        )
    return KeyError(message)


def previous_result_reference_errors(workflow_definition, source_indices=None):
    """Every 'previous_result:' reference that names no earlier step.

    References resolve lazily, one step at a time, so a reference naming a
    step that does not exist is only discovered when execution reaches it -
    after everything before it has run. On a workflow whose steps are
    generation steps that is 40 minutes of GPU spent to learn about a typo
    a read of the file would have caught (T005). The names are all in the
    definition, so this is answerable before anything runs.

    The definition handed here has already been substituted and expanded,
    so every reference in it is literal; a 'variable:' still spelled out is
    one nothing resolved and is left alone.

    `source_indices`, when given, is the source step index of each step -
    expansion of a 'for_each' group turns one written step into several, and
    the path an error carries has to be one the author can find in the file
    they wrote.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    seen = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        name = step.get("name")
        found = {}
        _collect_reference_paths(step, (), found)
        for path, reference in sorted(found.items(), key=lambda item: str(item[0])):
            if any(reference_resolves_to(reference, name) for name in seen):
                continue
            source = (
                source_indices[index]
                if source_indices is not None and index < len(source_indices)
                else index
            )
            location = render_path(("steps", source) + path)
            # Which expansion it was: the source path alone points at the one
            # step the author wrote, and every member reports the same path
            where = (
                f" in member '{name}'"
                if isinstance(name, str) and MEMBER_SEPARATOR in name
                else ""
            )
            errors.append(
                {
                    "path": location,
                    "message": (
                        f"previous_result '{reference}'{where} names no earlier "
                        f"step. Steps available here: {seen}"
                    ),
                }
            )
        if isinstance(name, str):
            seen.append(name)
    return errors


def _collect_reference_paths(value, path, found):
    """Every literal previous-result reference under `value`, by JSON path.

    Both spellings: the 'previous_result:' prefix on a string, and the
    'from_previous_result' key of a constructed object, which names a step
    without the prefix.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if key == FROM_PREVIOUS_RESULT_KEY and isinstance(item, str):
                if not item.startswith("variable:"):
                    found[path + (key,)] = item
                continue
            _collect_reference_paths(item, path + (key,), found)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_reference_paths(item, path + (index,), found)
    elif isinstance(value, str) and value.startswith(PREVIOUS_RESULT_PREFIX):
        found[path] = value[len(PREVIOUS_RESULT_PREFIX) :]
