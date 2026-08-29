"""Discover what diffusers exposes and what its pipelines accept.

This is the metadata layer a form-generating UI builds on: pipeline names
come from the installed diffusers (so a new release's pipelines appear with
no code change here), and a pipeline's argument schema comes from its
__call__ signature and docstring. Nothing here executes a pipeline.

Only bare class names resolved against the diffusers namespace are accepted
from callers - never arbitrary dotted import paths, which would let an HTTP
client import any module on the system.
"""

import re
import inspect
import logging

logger = logging.getLogger("dw")

_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Matches a diffusers-style docstring parameter header:
#     prompt (`str` or `List[str]`, *optional*):
_DOC_PARAM_PATTERN = re.compile(r"^(\w+) \((.+?)\):\s*(.*)$")


def list_pipelines():
    """Names of every pipeline class the installed diffusers exports.

    Reads the export list without importing each pipeline's module -
    diffusers is lazy and enumerating hundreds of classes must stay cheap.
    """
    import diffusers

    return sorted(
        name
        for name in dir(diffusers)
        if name.endswith("Pipeline") and not name.startswith("_")
    )


def load_pipeline_class(name):
    """Resolve a bare pipeline class name against diffusers.

    Raises:
        ValueError: for a malformed name or one diffusers does not export
    """
    if not _NAME_PATTERN.match(name or ""):
        raise ValueError(f"Not a valid class name: {name!r}")
    import diffusers

    try:
        cls = getattr(diffusers, name)
    except AttributeError:
        raise ValueError(f"diffusers exports no class named {name!r}")
    if not isinstance(cls, type):
        raise ValueError(f"{name!r} is not a class")
    return cls


def _json_safe_default(value):
    if value is inspect.Parameter.empty:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _parse_docstring_args(docstring):
    """Parameter descriptions from a diffusers-style 'Args:' docstring block.

    Best effort by design: a pipeline with an unusual docstring simply
    yields fewer descriptions, never an error.
    """
    descriptions = {}
    if not docstring:
        return descriptions

    lines = docstring.splitlines()
    try:
        start = next(
            i
            for i, line in enumerate(lines)
            if line.strip() in ("Args:", "Parameters:")
        )
    except StopIteration:
        return descriptions

    current = None
    parts = []
    base_indent = None
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent
        if indent < base_indent:
            break  # left the Args block (Returns:, Examples:, ...)

        header = _DOC_PARAM_PATTERN.match(stripped) if indent == base_indent else None
        if header:
            if current:
                descriptions[current]["description"] = " ".join(parts).strip()
            current = header.group(1)
            descriptions[current] = {"doc_type": header.group(2)}
            parts = [header.group(3)] if header.group(3) else []
        elif current:
            parts.append(stripped)
    if current:
        descriptions[current]["description"] = " ".join(parts).strip()
    return descriptions


def describe_pipeline(name):
    """The argument schema of a pipeline's __call__, for form generation.

    Returns a dict with the class name, its one-line summary, and an ordered
    parameter list: name, kind, required, default, annotation, and the
    docstring's type note and description when present.
    """
    cls = load_pipeline_class(name)
    call = cls.__call__
    signature = inspect.signature(call)
    documented = _parse_docstring_args(inspect.getdoc(call))

    parameters = []
    accepts_kwargs = False
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
            continue
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        entry = {
            "name": parameter.name,
            "required": parameter.default is inspect.Parameter.empty,
            "default": _json_safe_default(parameter.default),
            "annotation": (
                None
                if parameter.annotation is inspect.Parameter.empty
                else str(parameter.annotation)
            ),
        }
        entry.update(documented.get(parameter.name, {}))
        parameters.append(entry)

    # The class's own docstring only - getdoc walks the MRO and would call
    # every pipeline "Base class for all pipelines."
    class_doc = inspect.cleandoc(cls.__dict__.get("__doc__") or "")
    summary = class_doc.split("\n\n")[0].replace("\n", " ").strip()

    return {
        "name": cls.__name__,
        "summary": summary,
        "accepts_kwargs": accepts_kwargs,
        "parameters": parameters,
    }


def unknown_call_arguments(name, argument_names):
    """The given argument names a pipeline's __call__ will reject.

    Empty when the signature takes **kwargs (no name can be proven wrong)
    or when the class cannot be resolved or inspected - this feeds warnings,
    and a warning must never be wrong.
    """
    try:
        cls = load_pipeline_class(name)
        signature = inspect.signature(cls.__call__)
    except (ValueError, TypeError):
        return []
    parameters = signature.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
        return []
    known = {p.name for p in parameters}
    return sorted(set(argument_names) - known)


def list_tasks():
    """Every task command a workflow's task step can name."""
    from .tasks.task import _COMMAND_REGISTRY, _VIDEO_PROCESSOR_COMMANDS
    from .tasks.image_utils import available_processors

    return {
        "commands": sorted(_COMMAND_REGISTRY.keys()),
        "image_processors": sorted(available_processors()),
        "video_processors": list(_VIDEO_PROCESSOR_COMMANDS),
    }


def workflow_argument_warnings(workflow_definition):
    """Best-effort pre-load check of a workflow's pipeline arguments.

    For each pipeline step whose component_type is a bare diffusers class
    name, reports argument names that class's __call__ does not accept - the
    typo that today surfaces as a TypeError after the model has loaded.
    Escaped ({...}) and dotted component types are left alone.
    """
    warnings = []
    for step in workflow_definition.get("steps", []):
        pipeline = step.get("pipeline")
        if not pipeline:
            continue
        component_type = pipeline.get("configuration", {}).get("component_type")
        if not isinstance(component_type, str) or not _NAME_PATTERN.match(
            component_type
        ):
            continue
        argument_names = list(pipeline.get("arguments", {}))
        unknown = unknown_call_arguments(component_type, argument_names)
        for argument_name in unknown:
            warnings.append(
                f"Step '{step.get('name')}': {component_type} does not accept "
                f"argument '{argument_name}'"
            )
    return warnings
