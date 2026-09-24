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
import difflib
from .variables import undeclared_variable_references

logger = logging.getLogger("dw")

_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

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
    if not docstring:
        return documented, documented_kwargs

    lines = docstring.splitlines()
    try:
        start = next(
            i
            for i, line in enumerate(lines)
            if line.strip() in ("Args:", "Parameters:")
        )
    except StopIteration:
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
            current = name
            target[current] = {"doc_type": header.group(2)}
            parts = [header.group(3)] if header.group(3) else []
        elif current:
            parts.append(stripped)
        elif target is documented_kwargs and kwargs_indent is None:
            # First line under '**kwargs:' - it sets the nested indent, and
            # is a header if it reads like one
            kwargs_indent = indent
            header = _DOC_PARAM_PATTERN.match(stripped)
            if header and not header.group(1).startswith("*"):
                current = header.group(1)
                target[current] = {"doc_type": header.group(2)}
                parts = [header.group(3)] if header.group(3) else []
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
    """Every task command a workflow's task step can name.

    `assessment` names the probes among the commands (#387) - the ones that
    answer a JSON document of measurements about a finished file rather than
    make one - so a caller looking for a way to check a cut finds them
    without reading every command's schema. They stay in `commands` too,
    since a step still names one as its `command`.
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
            name
            for name, info in _COMMAND_INFO.items()
            if info.get("returns") == "json"
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
    from .task_domains import TASK_ARGUMENT_DOMAINS

    domains = TASK_ARGUMENT_DOMAINS.get(command, {})
    for parameter in parameters:
        domain = domains.get(parameter["name"])
        if domain is not None:
            parameter["domain"] = domain

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
    if not isinstance(value, str) or not value.startswith("variable:"):
        return None
    name = value[len("variable:") :]
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
        source = (
            source_indices[index]
            if source_indices is not None and index < len(source_indices)
            else index
        )
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

    from .type_helpers import load_type_from_name
    from .security import UntrustedWorkflowError

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


def _walk_type_references(node, path, errors):
    if isinstance(node, dict):
        for key in _TYPE_REFERENCE_KEYS:
            if key in node:
                error = _type_reference_error(key, node[key], path + (key,))
                if error is not None:
                    errors.append(error)
        for k, v in node.items():
            _walk_type_references(v, path + (k,), errors)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _walk_type_references(item, path + (i,), errors)


def component_type_errors(workflow_definition, source_indices=None):
    """Every component_type/scheduler_type/config_type in a step's pipeline
    naming a class the run itself could not load, as [{path, message}] - a
    misspelled class used to validate clean and only die ~3s into the run,
    after the worker had already loaded a checkpoint the plan's
    downloads_required quoted for a pipeline that could never exist (#345).

    A class outside the trusted ecosystem entirely (UntrustedWorkflowError,
    see _type_reference_error) is reported with a distinct message from one
    that is merely spelled wrong - "not allowed" is not "does not exist".

    The definition handed here has already been substituted and expanded,
    matching task_signature_errors; source_indices maps each expanded step
    back to the step the author wrote.
    """
    from .for_each import MEMBER_SEPARATOR, render_path

    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        pipeline = step.get("pipeline")
        if not isinstance(pipeline, dict):
            continue
        source = (
            source_indices[index]
            if source_indices is not None and index < len(source_indices)
            else index
        )
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        found = []
        _walk_type_references(pipeline, ("pipeline",), found)
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


def _resolved_value(arguments, key, values):
    """`arguments[key]` as a number, resolving a `variable:name` reference
    against `values` (declared defaults merged with the caller's own
    arguments, the way `constraint_warnings` resolves a constrained
    variable). `None` when the key is absent, not a `variable:` reference or
    a literal, or the reference does not resolve to a number - callers tell
    that apart from an actual 0 by checking `key in arguments` themselves
    where it matters."""
    if key not in arguments:
        return None
    value = arguments[key]
    if isinstance(value, str) and value.startswith("variable:"):
        value = values.get(value[len("variable:") :])
    return value if isinstance(value, (int, float)) else None


def _inert_crossfade_warnings(step, command, arguments):
    """concat_videos draws its crossfade from the trimmed-off material, so
    with nothing trimmed a `crossfade_ms` the author wrote does nothing. A
    referenced trim is unknown until the run and is left alone."""
    if command != "concat_videos":
        return []
    crossfade = arguments.get("crossfade_ms")
    trim = arguments.get("trim_frames", 0)
    if not isinstance(crossfade, (int, float)) or crossfade <= 0 or trim != 0:
        return []
    return [
        f"Step '{step.get('name')}': 'crossfade_ms' has no effect when "
        f"'trim_frames' is 0 - the crossfade is drawn from the trimmed "
        f"material. At a hard cut, 'audio_bleed_ms' or 'seam_fade_ms' is "
        f"what shapes the seam"
    ]


def _inert_seam_fade_warnings(step, command, arguments, values):
    """concat_videos takes the bleed path, not the fade path, at a hard cut
    with nothing trimmed while audio_bleed_ms is non-zero - so a seam_fade_ms
    the author wrote alongside it does nothing (#288). Both variables are
    ordinary `variable:` references in the templates that pair them, so
    `seam_fade_ms` and `audio_bleed_ms` are resolved against `values`
    (declared defaults merged with the caller's own arguments) rather than
    left alone the way an unresolved `trim_frames` is - it is exactly the
    templated case, with `audio_bleed_ms` left at its non-zero default and
    only `seam_fade_ms` passed as an argument, that this warning exists for.
    `trim_frames` stays a literal-only check, as in `_inert_crossfade_warnings`."""
    if command != "concat_videos":
        return []
    if "seam_fade_ms" not in arguments or "audio_bleed_ms" not in arguments:
        return []
    seam_fade = _resolved_value(arguments, "seam_fade_ms", values)
    bleed = _resolved_value(arguments, "audio_bleed_ms", values)
    trim = arguments.get("trim_frames", 0)
    if seam_fade is None or seam_fade <= 0 or bleed is None or bleed <= 0 or trim != 0:
        return []
    return [
        f"Step '{step.get('name')}': 'seam_fade_ms' has no effect while "
        f"'audio_bleed_ms' is {bleed} - a hard cut takes the bleed path "
        f"instead of the fade path. Pass 'audio_bleed_ms': 0 for "
        f"'seam_fade_ms' to apply."
    ]


def _inert_bleed_gain_warnings(step, command, arguments, values):
    """concat_videos applies audio_bleed_gain_db to the bled tail
    audio_bleed_ms carries across the seam - with no bleed there is nothing
    for the gain to shape, so an audio_bleed_gain_db the author wrote does
    nothing while audio_bleed_ms is 0, whether that 0 is an explicit
    argument or the task's own default left untouched (#290, the same no-op
    class #288 closed for seam_fade_ms). `audio_bleed_ms` is read with the
    task's default of 0 rather than requiring the key, since "forgot the
    bleed" is exactly the case this warning is for; resolved against
    `values` for the same reason _inert_seam_fade_warnings is - a templated
    case pairs both as `variable:` references."""
    if command != "concat_videos":
        return []
    if "audio_bleed_gain_db" not in arguments:
        return []
    gain = _resolved_value(arguments, "audio_bleed_gain_db", values)
    bleed = (
        _resolved_value(arguments, "audio_bleed_ms", values)
        if "audio_bleed_ms" in arguments
        else 0
    )
    if gain is None or gain == 0 or bleed is None or bleed != 0:
        return []
    return [
        f"Step '{step.get('name')}': 'audio_bleed_gain_db' has no effect "
        f"when 'audio_bleed_ms' is 0 - pass a non-zero 'audio_bleed_ms' for "
        f"the gain to apply."
    ]


def _inert_match_levels_dbfs_warnings(step, command, arguments):
    """concat_videos and dissolve_videos only call match_levels() - the
    function that reads match_levels_dbfs as its target - when match_levels
    itself is truthy (`if match_levels:`), so a caller who passes only the
    target dBFS and leaves match_levels unset (off by default) has stated an
    intent the engine silently drops: the shots join unmatched with no trace,
    warning or otherwise (#291, the same "modifier without its enabler" class
    #288 and #290 closed for seam_fade_ms and audio_bleed_gain_db). Literal
    check only, like _inert_crossfade_warnings' trim_frames - match_levels is
    "rms"/"peak"/falsy, not a number a variable: reference would need
    resolving to compare against a domain."""
    if command not in ("concat_videos", "dissolve_videos"):
        return []
    if "match_levels_dbfs" not in arguments or arguments.get("match_levels"):
        return []
    dbfs = arguments.get("match_levels_dbfs")
    if not isinstance(dbfs, (int, float)):
        return []
    return [
        f"Step '{step.get('name')}': 'match_levels_dbfs' has no effect when "
        f'\'match_levels\' is unset - pass "rms" or "peak" for the target '
        f"to apply."
    ]


def workflow_argument_warnings(workflow_definition, arguments=None):
    """Best-effort pre-load check of a workflow's arguments.

    For each pipeline step whose component_type is a bare diffusers class
    name, reports argument names that class's __call__ does not accept - the
    typo that today surfaces as a TypeError after the model has loaded.
    Escaped ({...}) and dotted component types are left alone. Task steps
    get the same check against their registered implementation's signature.

    `arguments`, when given, is a caller's own values for this run -
    checks that need a task argument's actual value (an inert `crossfade_ms`
    or `seam_fade_ms`) resolve a `variable:name` reference against the
    caller's arguments merged over the workflow's declared defaults, the
    same values `constraint_warnings` checks a constraint against.
    """
    warnings = []
    values = {**(workflow_definition.get("variables") or {}), **(arguments or {})}
    declared = sorted(workflow_definition.get("variables") or {})
    for path, name in undeclared_variable_references(workflow_definition):
        hint = (
            " - a reference is the whole value, nothing is interpolated around it"
            if any(c in name for c in " ,")
            else ""
        )
        warnings.append(
            f"{path}: 'variable:{name}' names no declared variable{hint}; "
            f"declared: {', '.join(declared) or '<none>'}"
        )
    for step in workflow_definition.get("steps", []):
        task = step.get("task")
        if task and isinstance(task.get("arguments"), dict):
            command = task.get("command")
            # An unknown or missing task argument is an error rather than a
            # warning now (task_signature_errors, #141) - reported once, by
            # the pass whose verdict it changes
            warnings.extend(_inert_crossfade_warnings(step, command, task["arguments"]))
            warnings.extend(
                _inert_seam_fade_warnings(step, command, task["arguments"], values)
            )
            warnings.extend(
                _inert_bleed_gain_warnings(step, command, task["arguments"], values)
            )
            warnings.extend(
                _inert_match_levels_dbfs_warnings(step, command, task["arguments"])
            )
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
