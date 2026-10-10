"""Checks on the class names a workflow definition writes.

`component_type_errors` refuses a `*_type` or `constant:` value the run itself
could not load, and `component_name_errors` a component name a step's
`component_type` does not register - both before the job is queued, so a
misspelling does not die after a checkpoint has loaded (#345, #409, #442).
The class catalog they check against is `dw.introspection`'s.
"""

import difflib
import re

from . import references
from .argument_media import is_media_reference
from .arguments import fetch_constant, is_constant_reference
from .for_each import render_path
from .introspection import (
    CLASS_NAME_PATTERN,
    list_classes,
    list_pipelines,
    unknown_pipeline_components,
)
from .security import InvalidInputError, UntrustedWorkflowError
from .type_helpers import NON_TYPE_KEYS, load_type_from_name

_TYPE_REFERENCE_KEYS = ("component_type", "scheduler_type", "config_type")

# A class-name-shaped string, bare or dotted - excludes a {}-escaped literal
# and a variable:/constant:/asset:/... reference, which use ':' or braces
# and are checked elsewhere
_DOTTED_NAME_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


def _type_reference_candidates(key):
    """Names to suggest a close match from, keyed by which field was wrong."""
    if key == "component_type":
        return list_pipelines() + list_classes("models")
    if key == "scheduler_type":
        return list_classes("schedulers")
    return list_classes("quantization")


def _type_reference_error(key, value, path):
    """One component_type/scheduler_type/config_type value, checked against
    the resolver the run itself uses for a '*_type' value
    (type_helpers.load_type_from_name) - not load_allowed_class's narrower
    ALLOWED_MODULES, which would refuse names the catalog already relies on
    (e.g. 'transformers.AutoProcessor', 'dw.community_pipelines...') that
    TRUSTED_TOP_LEVEL_PACKAGES lets the run itself load. Using the real
    resolver is what makes #345's own invariant hold: this can never refuse
    a name that would in fact have run.

    Returns an error dict ({path, message}), or None if `value` would resolve.
    """
    if not isinstance(value, str) or not _DOTTED_NAME_PATTERN.match(value):
        return None

    try:
        load_type_from_name(value, key)
    except UntrustedWorkflowError as e:
        return {"path": path, "message": str(e)}
    except (ImportError, AttributeError, ValueError):
        class_name = value.rsplit(".", 1)[-1]
        suggestions = difflib.get_close_matches(
            class_name, _type_reference_candidates(key), n=3, cutoff=0.6
        )
        message = f"{key} {value!r} does not exist"
        if suggestions:
            message += f" (closest matches: {', '.join(suggestions)})"
        return {"path": path, "message": message}
    return None


def _is_type_key(key):
    """Whether realize_args loads this key's value as a type - the same test
    it applies at run time, so validation refuses only what the run would."""
    return (
        isinstance(key, str)
        and key not in NON_TYPE_KEYS
        and (key.endswith("_type") or key.endswith("_dtype") or key == "dtype")
    )


def _loose_type_reference_error(key, value, path):
    """Any other '*_type' / '*_dtype' / 'dtype' value the run loads as a type
    (a pipeline's torch_dtype, say), put to the same untrusted gate. Only the
    gate's refusal is reported: a name that merely fails to resolve is left
    to the run, since realize_args reads some of these keys as something
    other than a type."""
    if not isinstance(value, str) or not _DOTTED_NAME_PATTERN.match(value):
        return None

    try:
        load_type_from_name(value, key)
    except UntrustedWorkflowError as e:
        return {"path": path, "message": str(e)}
    except (ImportError, AttributeError, ValueError):
        pass
    return None


def _constant_reference_error(value, path):
    """One literal 'constant:' value, resolved the way the run resolves it
    (arguments.fetch_constant) - so the untrusted walk rules, a callable and
    a name that does not exist are each refused here rather than after the
    job is queued. A 'variables' default is resolved earlier, by
    expanded_definition, and reported at 'variables.<name>'."""
    if not is_constant_reference(value):
        return None
    try:
        fetch_constant(value)
    except (ValueError, InvalidInputError, UntrustedWorkflowError) as e:
        return {"path": path, "message": str(e)}
    return None


def _walk_type_references(node, path, errors):
    # A {media_type, location} dict is loaded as media, and its media_type
    # names a kind rather than a type - realize_args never reads it as one
    if isinstance(node, dict) and not is_media_reference(node):
        for k, v in node.items():
            if k in _TYPE_REFERENCE_KEYS:
                error = _type_reference_error(k, v, path + (k,))
            elif _is_type_key(k):
                error = _loose_type_reference_error(k, v, path + (k,))
            else:
                error = _constant_reference_error(v, path + (k,))
            if error is not None:
                errors.append(error)
            else:
                _walk_type_references(v, path + (k,), errors)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            error = _constant_reference_error(item, path + (i,))
            if error is not None:
                errors.append(error)
            else:
                _walk_type_references(item, path + (i,), errors)


def component_type_errors(workflow_definition, source_indices=None):
    """Every component_type/scheduler_type/config_type in a step's pipeline
    naming a class the run itself could not load, as [{path, message}] - a
    misspelled class used to validate clean and only die ~3s into the run,
    after the worker had already loaded a checkpoint the plan's
    downloads_required quoted for a pipeline that could never exist (#345).
    Every other key the run loads as a type ('torch_dtype', any '*_type')
    and every literal 'constant:' value in the step are checked the same way,
    so the untrusted gate refuses them here rather than after the queue
    (#409).

    A class outside the trusted ecosystem entirely (UntrustedWorkflowError,
    see _type_reference_error) is reported with a distinct message from one
    that is merely spelled wrong - "not allowed" is not "does not exist".

    The definition handed here has already been substituted and expanded,
    matching task_signature_errors; source_indices maps each expanded step
    back to the step the author wrote.
    """
    errors = []
    for _, step, _, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices
    ):
        found = []
        # The whole step, since realize_args loads a type or a constant
        # wherever one sits in it - a task's arguments as much as a pipeline
        _walk_type_references(step, (), found)
        for error in found:
            full_message = f"{error['message']}{where}"
            if not full_message.endswith("."):
                full_message += "."
            errors.append(
                {
                    "path": render_path(("steps", source) + error["path"]),
                    "message": full_message,
                }
            )
    return errors


def component_name_errors(workflow_definition, source_indices=None):
    """Every name under a step's `pipeline.configuration.components` that
    the named component_type's constructor does not register, as
    [{path, message}] - a component name it does not have was previously
    caught only ~3s into the run's `loading` phase, after a checkpoint (and
    for an IC-LoRA step, the LoRA weights) the plan had already quoted for
    downloading (#442). Checked the same way `workflow_argument_warnings`
    checks a `__call__` argument: against the class's own constructor
    signature, so the rule can never refuse a name that would in fact have
    worked, and only for a bare, loadable component_type - escaped and
    dotted ones are left alone.

    `reused_components` names are excluded: those are configured by the step
    that shared them, not loaded here, so a name only valid because it was
    reused is not a mistake.

    The definition handed here has already been substituted and expanded,
    matching component_type_errors; source_indices maps each expanded step
    back to the step the author wrote.
    """
    errors = []
    for _, step, pipeline, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        configuration = pipeline.get("configuration")
        if not isinstance(configuration, dict):
            continue
        component_type = configuration.get("component_type")
        if not isinstance(component_type, str) or not CLASS_NAME_PATTERN.match(
            component_type
        ):
            continue
        components = configuration.get("components")
        if not isinstance(components, dict):
            continue
        reused = set(configuration.get("reused_components") or [])
        component_names = [name for name in components if name not in reused]
        unknown = unknown_pipeline_components(component_type, component_names)
        if not unknown:
            continue
        for component_name in unknown:
            errors.append(
                {
                    "path": render_path(
                        (
                            "steps",
                            source,
                            "pipeline",
                            "configuration",
                            "components",
                            component_name,
                        )
                    ),
                    "message": (
                        f"Step '{step.get('name')}': {component_type} has no "
                        f"component '{component_name}'{where}."
                    ),
                }
            )
    return errors
