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
import math
import logging
from . import references

logger = logging.getLogger("dw")

CLASS_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Matches a docstring parameter header, with or without a declared type:
#     prompt (`str` or `List[str]`, *optional*):
#     prompt: The user message
# The leading stars let a '**kwargs:' block be recognized as a header of its
# own, so what it documents can be read as parameters too.
_DOC_PARAM_PATTERN = re.compile(r"^(\*{0,2}[A-Za-z_]\w*)(?: \((.+?)\))?:\s*(.*)$")


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
    if not CLASS_NAME_PATTERN.match(class_name):
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
    # JSON has no inf or nan: named as a string, the way any other value
    # JSON cannot carry is, rather than nulled into "no default"
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _args_block_start(lines):
    """Index of the 'Args:' / 'Parameters:' line, or None."""
    for i, line in enumerate(lines):
        if line.strip() in ("Args:", "Parameters:"):
            return i
    return None


def _open_entry(target, header):
    """Start the entry `header` (a _DOC_PARAM_PATTERN match) names in
    `target`; returns (name, the description parts so far)."""
    name = header.group(1)
    target[name] = {"doc_type": header.group(2)}
    return name, [header.group(3)] if header.group(3) else []


def _parse_docstring_args(docstring):
    """Parameter descriptions from a docstring's 'Args:' block.

    Returns (documented, documented_kwargs): the entries at the block's own
    indent, and those nested one level under a '**kwargs:' entry. The nested
    ones are the only declaration a task that funnels its real arguments
    through **kwargs ever makes, so discovery needs them as much as the
    signature.

    Best effort by design: a docstring with an unusual shape simply yields
    fewer descriptions, never an error.
    """
    documented = {}
    documented_kwargs = {}
    lines = docstring.splitlines() if docstring else []
    start = _args_block_start(lines)
    if start is None:
        return documented, documented_kwargs

    # The entry being described, and where its description is accumulating:
    # a top-level entry, or one nested inside the **kwargs block
    current = None
    target = documented
    parts = []
    base_indent = None
    kwargs_indent = None
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent
        if indent < base_indent:
            break  # left the Args block (Returns:, Examples:, ...)

        nested = kwargs_indent is not None and indent == kwargs_indent
        header = (
            _DOC_PARAM_PATTERN.match(stripped)
            if indent == base_indent or nested
            else None
        )
        if header:
            if current:
                target[current]["description"] = " ".join(parts).strip()
            name = header.group(1)
            if name.startswith("*"):
                # The kwargs entry itself is not a parameter; what it
                # indents is. Its own indent is unknown until the first
                # nested line arrives
                current = None
                target = documented_kwargs
                kwargs_indent = None
                parts = []
                continue
            if indent == base_indent:
                target = documented
                kwargs_indent = None
            elif kwargs_indent is None:
                kwargs_indent = indent
            current, parts = _open_entry(target, header)
        elif current:
            parts.append(stripped)
        elif target is documented_kwargs and kwargs_indent is None:
            # First line under '**kwargs:' - it sets the nested indent, and
            # is a header if it reads like one
            kwargs_indent = indent
            header = _DOC_PARAM_PATTERN.match(stripped)
            if header and not header.group(1).startswith("*"):
                current, parts = _open_entry(target, header)
    if current:
        target[current]["description"] = " ".join(parts).strip()
    return documented, documented_kwargs


def _callable_parameters(target_callable):
    """A callable's parameters as form-ready entries, merged with its
    docstring's Args descriptions. Returns (parameters, accepts_kwargs)."""
    signature = inspect.signature(target_callable)
    documented, documented_kwargs = _parse_docstring_args(
        inspect.getdoc(target_callable)
    )

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
        documented_kwargs.pop(parameter.name, None)
        parameters.append(entry)

    if accepts_kwargs:
        # Whatever the **kwargs block names is a real argument the callable
        # takes; the signature just never says so. There is no default to
        # read, and anything funnelled through kwargs is optional
        for name, entry in documented_kwargs.items():
            parameters.append(
                {
                    "name": name,
                    "required": False,
                    "default": None,
                    "annotation": None,
                    **entry,
                }
            )
    return parameters, accepts_kwargs


def _modular_block_parameters(cls):
    """A modular pipeline's call arguments, read from its default block graph.

    A modular pipeline's __call__ is `(state, output, **kwargs)`: what it
    takes is whatever its blocks declare as inputs, so those are the honest
    answer. The graph is built without weights (`init_pipeline()` with no
    repository), and carries dw's own blocks - the H3 audio hold
    (dw/pipeline_processors/h3_blocks.py) - as a loaded one does. Empty for
    a class with no default blocks, the bare `ModularPipeline` among them.
    """
    blocks_name = getattr(cls, "default_blocks_name", None)
    if not isinstance(blocks_name, str):
        return []
    import importlib

    from .pipeline_processors.h3_blocks import insert_audio_hold

    blocks_class = getattr(importlib.import_module(cls.__module__), blocks_name, None)
    if blocks_class is None:
        import diffusers

        blocks_class = getattr(diffusers, blocks_name, None)
    if not isinstance(blocks_class, type):
        return []
    try:
        pipeline = blocks_class().init_pipeline()
        insert_audio_hold(pipeline)
        block_inputs = pipeline._blocks.inputs
    except Exception as error:
        # The signature still answers without them
        logger.debug(f"Could not read {blocks_name}'s inputs: {error}")
        return []
    parameters = []
    for block_input in block_inputs:
        if not block_input.name:
            continue
        parameters.append(
            {
                "name": block_input.name,
                "required": bool(block_input.required),
                "default": _json_safe_default(block_input.default),
                "annotation": (
                    None
                    if block_input.type_hint is None
                    else str(block_input.type_hint)
                ),
                "description": block_input.description or "",
            }
        )
    return parameters


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

    parameters, accepts_kwargs = _callable_parameters(target_callable)

    if target == "call":
        named = {parameter["name"] for parameter in parameters}
        parameters += [
            parameter
            for parameter in _modular_block_parameters(cls)
            if parameter["name"] not in named
        ]

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


def unknown_pipeline_components(name, component_names):
    """The given component names a pipeline's constructor does not register.

    A dotted name ('text_encoder.model') is checked by its first segment -
    the component itself is what the constructor registers; what a dotted
    path reaches inside it is not this check's business.

    Empty when the constructor takes **kwargs (no name can be proven wrong)
    or when the class cannot be resolved or inspected - this feeds warnings,
    and a warning must never be wrong.
    """
    try:
        cls = load_pipeline_class(name)
        signature = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return []
    parameters = [p for p in signature.parameters.values() if p.name != "self"]
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
        return []
    known = {p.name for p in parameters}
    return sorted({name for name in component_names if name.split(".")[0] not in known})


def list_tasks():
    """Every task command a workflow's task step can name.

    `assessment` names the probes among the commands (#387) - the ones that
    measure a finished cut and say where to look rather than make one - so a
    caller looking for a way to check a cut finds them without reading every
    command's schema. They stay in `commands` too, since a step still names
    one as its `command`. Membership is the `assessment` flag the command
    registered with, not its JSON return: `attribute_voices` answers JSON
    and is not a check of a cut (#485).
    """
    from .tasks.task import (
        _COMMAND_INFO,
        _COMMAND_REGISTRY,
        _VIDEO_PROCESSOR_COMMANDS,
    )
    from .tasks.image_utils import available_processors

    return {
        "commands": sorted(_COMMAND_REGISTRY.keys()),
        "image_processors": sorted(available_processors()),
        "video_processors": list(_VIDEO_PROCESSOR_COMMANDS),
        "assessment": sorted(
            name for name, info in _COMMAND_INFO.items() if info.get("assessment")
        ),
    }


def _first_paragraph(docstring):
    cleaned = inspect.cleandoc(docstring or "")
    return cleaned.split("\n\n")[0].replace("\n", " ").strip()


def describe_task(command):
    """A task command's argument schema, in describe_class's shape, so the
    editor's one arguments form consumes both.

    The schema is the registered implementation function's real signature -
    the same function the dispatch forwards **arguments into - so it cannot
    drift from the runtime. Parameters the dispatch supplies itself are
    removed; 'device' is appended because every task accepts it (the
    dispatch consumes it before the implementation is called). Raises
    ValueError for a name that is not a task command.
    """
    import importlib

    from .tasks.task import task_command_info

    info = task_command_info(command)

    device_parameter = {
        "name": "device",
        "required": False,
        "default": None,
        "annotation": None,
        "description": "Device override for this task (e.g. cpu, cuda:1) - "
        "keeps a helper model off the accelerator a pipeline is using",
    }

    if info["kind"] == "image_processor":
        from .tasks.image_utils import image_processor_target

        target = image_processor_target(command)
        image_parameter = {
            "name": "image",
            "required": True,
            "default": None,
            "annotation": None,
            "description": "The image to process",
        }
        if target is None:
            return {
                "name": command,
                "summary": f"'{command}' image processor (ControlNet preprocessor)",
                "accepts_kwargs": True,
                "parameters": [image_parameter, device_parameter],
            }

        # target is a plain (image, **kwargs) function - introspect it directly
        # rather than reporting the generic (image, device) shape every other
        # image processor shares (#350). Its first positional parameter is
        # the image (named "image" or "img" across these functions), dropped
        # in favor of the uniform image_parameter above.
        parameters, accepts_kwargs = _callable_parameters(target)
        parameters = [image_parameter] + parameters[1:]
        if not any(p["name"] == "device" for p in parameters):
            parameters.append(device_parameter)
        summary = _first_paragraph(inspect.getdoc(target))
        return {
            "name": command,
            "summary": summary,
            "accepts_kwargs": accepts_kwargs,
            "parameters": parameters,
        }

    if info["implementation"] is None:
        # Free-form by design (gather_inputs): any keys, passed through
        from .tasks.task import _COMMAND_REGISTRY

        handler = _COMMAND_REGISTRY.get(command)
        return {
            "name": command,
            "summary": _first_paragraph(inspect.getdoc(handler)),
            "accepts_kwargs": True,
            "parameters": [],
        }

    module_name, _, function_name = info["implementation"].rpartition(".")
    implementation = getattr(importlib.import_module(module_name), function_name)
    parameters, accepts_kwargs = _callable_parameters(implementation)
    parameters = [p for p in parameters if p["name"] not in info["provided"]]
    if not any(p["name"] == "device" for p in parameters):
        parameters.append(device_parameter)

    # A signature carries no range, so a declared domain is reported beside
    # the parameter it constrains - an agent reading get_task saw
    # 'annotation: null' and no domain at all, and wrote the negative frame
    # count validation now refuses (dw/task_domains.py, #139, #140)
    from .task_domains import TASK_ARGUMENT_CHOICES, TASK_ARGUMENT_DOMAINS

    domains = TASK_ARGUMENT_DOMAINS.get(command, {})
    # A string argument's accepted values, the same way (#602)
    choices = TASK_ARGUMENT_CHOICES.get(command, {})
    for parameter in parameters:
        domain = domains.get(parameter["name"])
        if domain is not None:
            parameter["domain"] = domain
        accepted = choices.get(parameter["name"])
        if accepted is not None:
            parameter["choices"] = list(accepted)

    parameter_descriptions = info.get("parameter_descriptions") or {}
    for parameter in parameters:
        description = parameter_descriptions.get(parameter["name"])
        if description:
            parameter["description"] = description

    summary = info.get("summary")
    if not summary:
        summary = _first_paragraph(inspect.getdoc(implementation))
    if not summary:
        from .tasks.task import _COMMAND_REGISTRY

        summary = _first_paragraph(inspect.getdoc(_COMMAND_REGISTRY.get(command)))

    return {
        "name": command,
        "summary": summary,
        "accepts_kwargs": accepts_kwargs,
        "parameters": parameters,
    }


def unknown_task_arguments(command, argument_names):
    """The given argument names a task command will not accept.

    Same never-wrong contract as unknown_call_arguments: empty when the
    implementation takes **kwargs, when the command consumes a free-form
    dict, or when the command cannot be described at all. 'device' is
    always accepted - the dispatch consumes it before the implementation
    runs.
    """
    try:
        description = describe_task(command)
    except Exception:
        return []
    if description["accepts_kwargs"]:
        return []
    known = {p["name"] for p in description["parameters"]} | {"device"}
    return sorted(set(argument_names) - known)


def unknown_task_argument_message(command, name):
    """The wording for one argument a task command does not take."""
    return (
        f"task '{command}' does not accept argument '{name}' - its "
        f"implementation's signature is the whole of what it takes, so the "
        f"argument would reach Python as an unexpected keyword"
    )


def missing_task_arguments(command, argument_names):
    """The arguments a task command requires that the given names do not supply.

    A task step's `arguments` dict is the whole of what reaches the
    implementation - nothing is injected around it, so a required parameter
    absent from the dict is a run that cannot start. Unlike
    unknown_task_arguments this does not stop at `accepts_kwargs`: **kwargs
    says more names are allowed, never that a required one may be left out
    (an image processor takes any keys and still needs its `image`).

    Same never-wrong contract otherwise: empty for a command that cannot be
    described at all, and 'device' is never required.
    """
    try:
        description = describe_task(command)
    except Exception:
        return []
    supplied = set(argument_names)
    return sorted(
        p["name"]
        for p in description["parameters"]
        if p.get("required") and p["name"] != "device" and p["name"] not in supplied
    )


def missing_task_argument_message(command, missing):
    """The one wording both the static pass and the run-time guard use for a
    task step that leaves a required argument unset."""
    named = ", ".join(f"'{name}'" for name in missing)
    return (
        f"task '{command}' requires {named}, which the step does not supply. "
        f"A task's 'arguments' are the whole of what reaches the command, so "
        f"a required argument left out is a run that cannot start"
    )


def null_variable_task_argument_message(command, missing_arg, variable_name):
    """The wording for a required argument the step *does* supply, by
    `variable:<variable_name>`, but the variable's value is null (#364).

    `missing_task_argument_message` says "the step does not supply" it,
    which is false here - the step names the variable, the variable just
    hasn't been given a real value yet. That is a caller's job to do at
    run time, not a defect in the document.
    """
    return (
        f"'{missing_arg}' is fed by variable '{variable_name}', which is "
        f"null - task '{command}' requires a real value for it. Pass "
        f"arguments={{'{variable_name}': ...}} when running or validating, "
        f"or give '{variable_name}' a non-null default"
    )


def _null_fed_variable(written_steps, source_index, key, declared_variables):
    """The variable name, if the argument at `key` was written as
    `variable:<name>` naming a declared variable - the shape that makes a
    "missing" required argument actually a null-variable one (#364). None
    otherwise, including when `written_steps` can't be indexed (a for_each
    template step, whose members are checked by `item:`/`gather:` instead).
    """
    if not isinstance(source_index, int) or source_index >= len(written_steps):
        return None
    step = written_steps[source_index]
    if not isinstance(step, dict):
        return None
    task = step.get("task")
    if not isinstance(task, dict):
        return None
    arguments = task.get("arguments")
    if not isinstance(arguments, dict):
        return None
    value = arguments.get(key)
    name = references.ref_name(references.VARIABLE, value)
    if name is None:
        return None
    return name if name in declared_variables else None


def task_signature_errors(
    workflow_definition, source_indices=None, written_definition=None
):
    """Every task step whose arguments its command's signature refuses, as
    [{path, message}] - a required argument left unset, and an argument the
    command does not take - plus a step naming a command that is not
    registered at all.

    The one class of mistake a free pre-flight is most obviously for, and the
    one it used to let through: `validate_workflow` answered `valid: true`
    and the job then failed with Python's own
    "resample_audio() missing 1 required positional argument: 'audio'"
    (#141). An unknown argument was a warning beside it, so a step with every
    argument it was given rejected and every argument it needs missing still
    validated - both are a guaranteed TypeError at the same call, so both are
    errors now.

    A misspelled or removed `task.command` (e.g. the shipped
    `templates/image-processors`'s `face_detector`, #285) used to validate
    clean too: `missing_task_arguments`/`unknown_task_arguments` both catch
    `describe_task`'s ValueError for an unregistered name and answer "no
    complaint" rather than "this command does not exist", so the step ran 9
    steps into a 26-step template before failing on the engine's own
    "Unknown task command" error. Checked here, at `task.command` itself,
    before the per-argument checks (which stay silent for a command they
    cannot describe).

    The definition handed here has already been substituted and expanded, so
    a for_each member is checked as it will run; `source_indices` maps each
    expanded step back to the step the author wrote.

    `written_definition`, when given, is that step *as the author wrote it* -
    before substitution - plus the declared `variables` block. A required
    argument reported missing whose written form is `variable:<name>` naming
    a declared variable is not a step that "does not supply" it (#364): the
    step does name it, the variable's value just resolved to null (the only
    way substitution drops a `variable:` reference, per #209). That error
    carries a `variable` key naming it, so a caller checking a document with
    no arguments of its own can treat it as caller input rather than a
    defect in the document.
    """
    from .for_each import MEMBER_SEPARATOR, render_path
    from .tasks.task import task_command_info

    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    written_steps = (
        (written_definition or {}).get("steps") or []
        if isinstance(written_definition, dict)
        else []
    )
    declared_variables = (
        (written_definition or {}).get("variables") or {}
        if isinstance(written_definition, dict)
        else {}
    )

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        task = step.get("task")
        if not isinstance(task, dict):
            continue
        command = task.get("command")
        # 'inputs' is a list template rather than a named-argument dict -
        # the command consumes it whole, so there is no name to miss
        arguments = task.get("arguments")
        if not isinstance(command, str):
            continue
        source = references.author_index(source_indices, index)
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )

        try:
            task_command_info(command)
        except ValueError:
            errors.append(
                {
                    "path": render_path(("steps", source, "task", "command")),
                    "message": (
                        f"'{command}' is not a registered task command{where}. "
                        f"This step would fail at run time with the engine's "
                        f'own "Unknown task command" error'
                    ),
                }
            )
            continue

        if not isinstance(arguments, dict):
            continue
        missing = missing_task_arguments(command, arguments.keys())
        unknown = unknown_task_arguments(command, arguments.keys())
        if not missing and not unknown:
            continue

        def report(key, message, variable=None):
            entry = {
                "path": render_path(("steps", source, "task", "arguments", key)),
                "message": f"{message}{where}.",
            }
            if variable is not None:
                entry["variable"] = variable
            errors.append(entry)

        if missing:
            key = missing[0]
            variable = _null_fed_variable(
                written_steps, source, key, declared_variables
            )
            if variable is not None:
                report(
                    key,
                    null_variable_task_argument_message(command, key, variable),
                    variable=variable,
                )
            else:
                report(key, missing_task_argument_message(command, missing))
        for key in unknown:
            report(key, unknown_task_argument_message(command, key))
    return errors
