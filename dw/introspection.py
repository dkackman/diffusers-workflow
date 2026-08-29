"""Discover what diffusers exposes and what its pipelines accept.

This is the metadata layer a form-generating UI builds on: pipeline names
come from the installed diffusers (so a new release's pipelines appear with
no code change here), and a pipeline's argument schema comes from its
__call__ signature and docstring. Nothing here executes a pipeline.

Only bare class names resolved against the diffusers namespace - plus an
explicit allowlist of companion packages (sdnq) - are accepted from callers;
never arbitrary dotted import paths, which would let an HTTP client import
any module on the system.
"""

import re
import inspect
import logging

logger = logging.getLogger("dw")

_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Matches a diffusers-style docstring parameter header:
#     prompt (`str` or `List[str]`, *optional*):
_DOC_PARAM_PATTERN = re.compile(r"^(\w+) \((.+?)\):\s*(.*)$")


# Companion packages whose classes workflows commonly name. Extending this
# is a deliberate act; nothing else outside diffusers ever resolves.
ALLOWED_MODULES = ("sdnq",)

# from_pretrained is **kwargs-based on ModelMixin, so these generic loading
# knobs are curated rather than discovered - merged with whatever a
# signature does name. No typo warnings are possible behind **kwargs.
COMPONENT_LOADING_KNOBS = [
    {
        "name": "torch_dtype",
        "required": False,
        "default": None,
        "annotation": "torch.dtype",
        "description": "Weight dtype to load as, e.g. torch.bfloat16.",
    },
    {
        "name": "variant",
        "required": False,
        "default": None,
        "annotation": "str",
        "description": "Checkpoint variant to load, e.g. 'fp16'.",
    },
    {
        "name": "subfolder",
        "required": False,
        "default": None,
        "annotation": "str",
        "description": "Subfolder of the repository the weights live in.",
    },
    {
        "name": "revision",
        "required": False,
        "default": None,
        "annotation": "str",
        "description": "Git revision (branch, tag or commit) to load from.",
    },
]


def _filtered_exports(predicate):
    """diffusers export names passing predicate - names only, no imports."""
    import diffusers

    return sorted(
        name for name in dir(diffusers) if not name.startswith("_") and predicate(name)
    )


def list_pipelines():
    """Names of every pipeline class the installed diffusers exports.

    Reads the export list without importing each pipeline's module -
    diffusers is lazy and enumerating hundreds of classes must stay cheap.
    """
    return _filtered_exports(lambda name: name.endswith("Pipeline"))


def list_classes(kind):
    """Class names of one kind, for UI pickers.

    Enumerates the way list_pipelines does - suffix filters over the export
    list, nothing imported. Autoencoders are models that don't carry the
    Model suffix, so the model filter names them explicitly.
    """
    if kind == "pipelines":
        return list_pipelines()
    if kind == "models":
        return _filtered_exports(
            lambda name: name.endswith("Model") or "Autoencoder" in name
        )
    if kind == "schedulers":
        return _filtered_exports(lambda name: name.endswith("Scheduler"))
    if kind == "quantization":
        names = _filtered_exports(lambda name: name.endswith("Config"))
        import importlib.util

        if importlib.util.find_spec("sdnq") is not None:
            names.append("sdnq.SDNQConfig")
        return names
    raise ValueError(f"Unknown class kind: {kind!r}")


def load_allowed_class(name):
    """Resolve a class name: bare against diffusers, or module.Class where
    the module is on the explicit allowlist.

    Raises:
        ValueError: for a malformed name, a module outside the allowlist,
            or a name the module does not export
    """
    module_name, _, class_name = (name or "").rpartition(".")
    if module_name and module_name not in ALLOWED_MODULES:
        raise ValueError(f"Module {module_name!r} is not on the allowlist")
    if not _NAME_PATTERN.match(class_name):
        raise ValueError(f"Not a valid class name: {name!r}")

    import importlib

    try:
        module = importlib.import_module(module_name or "diffusers")
    except ImportError as e:
        raise ValueError(f"Could not import {module_name}: {e}")
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ValueError(
            f"{module_name or 'diffusers'} exports no class named {class_name!r}"
        )
    if not isinstance(cls, type):
        raise ValueError(f"{name!r} is not a class")
    return cls


# The original, pipeline-flavored name - existing callers keep working
load_pipeline_class = load_allowed_class


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


def describe_class(name, target="call"):
    """The argument schema of a class, for form generation.

    target picks what gets inspected: 'call' reads __call__ (pipelines),
    'init' reads __init__ (quantization configs, schedulers, models), and
    'load' reads from_pretrained merged with the curated loading knobs -
    from_pretrained hides everything behind **kwargs, so the knobs are the
    honest answer there. Output shape is identical across targets, so one
    arguments editor consumes all three. Scheduler classes additionally
    report their compatibles list.
    """
    cls = load_allowed_class(name)
    if target == "call":
        target_callable = cls.__call__
    elif target == "init":
        target_callable = cls.__init__
    elif target == "load":
        target_callable = getattr(cls, "from_pretrained", cls.__init__)
    else:
        raise ValueError(f"Unknown inspection target: {target!r}")

    signature = inspect.signature(target_callable)
    documented = _parse_docstring_args(inspect.getdoc(target_callable))

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
    if target == "load":
        named = {parameter["name"] for parameter in parameters}
        parameters = [
            knob for knob in COMPONENT_LOADING_KNOBS if knob["name"] not in named
        ] + parameters
        # the first positional of from_pretrained is the model path, which
        # the editor's own model field carries
        parameters = [
            p for p in parameters if p["name"] != "pretrained_model_name_or_path"
        ]

    # The class's own docstring only - getdoc walks the MRO and would call
    # every pipeline "Base class for all pipelines."
    class_doc = inspect.cleandoc(cls.__dict__.get("__doc__") or "")
    summary = class_doc.split("\n\n")[0].replace("\n", " ").strip()

    description = {
        "name": name,
        "summary": summary,
        "accepts_kwargs": accepts_kwargs,
        "parameters": parameters,
    }

    compatibles = getattr(cls, "_compatibles", None)
    if compatibles:
        description["compatibles"] = sorted(
            c if isinstance(c, str) else getattr(c, "__name__", str(c))
            for c in compatibles
        )
    return description


def describe_pipeline(name):
    """A pipeline's __call__ argument schema - describe_class's original."""
    return describe_class(name, target="call")


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
