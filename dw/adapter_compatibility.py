"""The LoRA a step loads has to be trained for the path that step runs.

MiniMax-H3 ships one repository holding two checkpoint partitions -
`transformer/` for the `t2va` and `fl2va` workflows, `transformer_ref/` for
`ref2va` - and a separate turbo adapter trained against each. Handing the
FL2VA adapter to a `ref2va` step is the one H3 misconfiguration that never
shows up as a failure: a `ref2va` step holds `transformer_ref` alone, so
diffusers loads whatever it is given straight onto it, the run succeeds, and
the only symptom is identity retention that quietly is not as good as it
should be. Four templates carried exactly that mistake and nothing ever
complained (#149, #155).

Everything a pipeline itself would refuse is refused by the pipeline. This
is the other kind: a request the pipeline accepts and answers wrongly, which
costs a whole render to discover and cannot be seen in the output. So it is
checked where it is free - `validation_errors`, so `POST /api/validate` and
the pre-queue check both refuse it, at the JSON path the value sits at.

What is *not* written here: the adapter file names. `lora_weight_name` is a
free string and a future reference-trained checkpoint cannot be predicted, so
an unrecognised name is a warning naming the rule rather than a refusal - the
default inverts from "anything passes" to "the one documented mistake is
caught", and someone testing `my-new-ref-lora.safetensors` still gets
through. What *is* written here is a vendor naming convention (`ref2v` /
`fl2v` in the file name) that no diffusers symbol declares; the workflow
names and the partition each one denoises against are diffusers' own, and
`tests/test_h3_adapters.py` pins them to it.
"""

import logging

from .for_each import MEMBER_SEPARATOR, render_path

logger = logging.getLogger("dw")

# The `workflow=` values MiniMaxH3Blocks._workflow_map declares, split by the
# transformer partition each one denoises against - a `ref2va` step loads
# `transformer_ref` and nothing else, every other H3 workflow loads
# `transformer`. A workflow name not listed here is not checked
REFERENCE_WORKFLOWS = frozenset({"ref2va"})
KEYFRAME_WORKFLOWS = frozenset({"t2va", "fl2va"})
H3_WORKFLOWS = REFERENCE_WORKFLOWS | KEYFRAME_WORKFLOWS

# The token MiniMax puts in an adapter's file name to say which partition it
# was trained against. `ref2v` first: it is the longer match and a name
# carrying both is ambiguous rather than a keyframe adapter
REFERENCE_TOKEN = "ref2v"
KEYFRAME_TOKEN = "fl2v"

LORAS_KEY = "loras"
WEIGHT_NAME_KEY = "weight_name"
WORKFLOW_KEY = "workflow"
FROM_PRETRAINED_KEY = "from_pretrained_arguments"
VARIABLE_PREFIX = "variable:"
# Values another pass resolves; one still spelled out here is not this
# pass's complaint
_UNRESOLVED_PREFIXES = ("variable:", "item:", "previous_result:", "gather:")


def _trained_for(weight_name):
    """Which partition an adapter's file name says it was trained against -
    'reference', 'keyframe', or None when the name says neither."""
    lowered = weight_name.lower()
    if REFERENCE_TOKEN in lowered:
        return "reference"
    if KEYFRAME_TOKEN in lowered:
        return "keyframe"
    return None


def _problem(workflow, weight_name):
    """(severity, message) for one adapter on one step, or None when the
    pairing is right."""
    trained = _trained_for(weight_name)
    wants = "reference" if workflow in REFERENCE_WORKFLOWS else "keyframe"
    if trained == wants:
        return None
    if trained is None:
        return (
            "warning",
            f"'{weight_name}' names neither '{REFERENCE_TOKEN}' nor "
            f"'{KEYFRAME_TOKEN}', so it cannot be checked against this "
            f"step's '{workflow}' workflow. A '{workflow}' step denoises "
            f"against the "
            f"'{'transformer_ref' if wants == 'reference' else 'transformer'}' "
            f"partition, and an adapter trained against the other one loads "
            f"onto it without error and only degrades the result",
        )
    return (
        "error",
        f"'{weight_name}' is trained against the "
        f"'{'transformer_ref' if trained == 'reference' else 'transformer'}' "
        f"partition and this step runs the '{workflow}' workflow, which "
        f"denoises against "
        f"'{'transformer_ref' if wants == 'reference' else 'transformer'}'. "
        f"MiniMax-H3 loads it anyway and the run succeeds - the only symptom "
        f"is a worse result - so it is refused here. Use an adapter trained "
        f"for this workflow (its name carries "
        f"'{REFERENCE_TOKEN if wants == 'reference' else KEYFRAME_TOKEN}')",
    )


def _lora_problems(steps, source_indices, written=None, supplied=()):
    """Every adapter/workflow mismatch in an expanded step list, as
    (severity, path, message).

    `written` is the definition as the author wrote it and `supplied` the
    argument names the caller passed: when the value arrived through a
    variable the caller set, the path reported is `arguments.<name>`, where
    they wrote it, rather than the step it landed in.
    """
    written_steps = (written or {}).get("steps")
    found = []
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
        from_pretrained = pipeline.get(FROM_PRETRAINED_KEY)
        workflow = (
            from_pretrained.get(WORKFLOW_KEY)
            if isinstance(from_pretrained, dict)
            else None
        )
        if workflow not in H3_WORKFLOWS:
            continue
        loras = pipeline.get(LORAS_KEY)
        if not isinstance(loras, list):
            continue
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        for position, lora in enumerate(loras):
            if not isinstance(lora, dict):
                continue
            weight_name = lora.get(WEIGHT_NAME_KEY)
            if not isinstance(weight_name, str) or weight_name.startswith(
                _UNRESOLVED_PREFIXES
            ):
                continue
            problem = _problem(workflow, weight_name)
            if problem is None:
                continue
            severity, message = problem
            found.append(
                (
                    severity,
                    _path_for(written_steps, source, position, supplied)
                    or render_path(
                        (
                            "steps",
                            source,
                            "pipeline",
                            LORAS_KEY,
                            position,
                            WEIGHT_NAME_KEY,
                        )
                    ),
                    message + where,
                )
            )
    return found


def _path_for(written_steps, source, position, supplied):
    """`arguments.<name>` when the adapter came from a variable the caller
    set, else None - the value is reported where it was written."""
    if not isinstance(written_steps, list) or source >= len(written_steps):
        return None
    step = written_steps[source]
    pipeline = step.get("pipeline") if isinstance(step, dict) else None
    loras = pipeline.get(LORAS_KEY) if isinstance(pipeline, dict) else None
    if not isinstance(loras, list) or position >= len(loras):
        return None
    entry = loras[position]
    reference = entry.get(WEIGHT_NAME_KEY) if isinstance(entry, dict) else None
    if not isinstance(reference, str) or not reference.startswith(VARIABLE_PREFIX):
        return None
    variable = reference.removeprefix(VARIABLE_PREFIX)
    return f"arguments.{variable}" if variable in (supplied or ()) else None


def adapter_errors(workflow_definition, source_indices=None, written=None, supplied=()):
    """Every adapter loaded onto the wrong checkpoint partition, as
    [{path, message}]. The definition is the expanded, substituted one, so a
    `for_each` member's own adapter is checked as it will run."""
    return [
        {"path": path, "message": message}
        for severity, path, message in _lora_problems(
            workflow_definition.get("steps") or [], source_indices, written, supplied
        )
        if severity == "error"
    ]


def adapter_warnings(
    workflow_definition, source_indices=None, written=None, supplied=()
):
    """Every adapter whose name says nothing about what it was trained for,
    as messages - valid, and said out loud because nothing at run time will."""
    return [
        f"{path}: {message}"
        for severity, path, message in _lora_problems(
            workflow_definition.get("steps") or [], source_indices, written, supplied
        )
        if severity == "warning"
    ]


def warn_adapters(workflow_definition):
    """Say the unrecognised-adapter warning where whoever asked for the run
    can read it - a run started from the CLI or a rerun never passed through
    the validate route."""
    from .events import emit_warning

    for message in adapter_warnings(workflow_definition):
        emit_warning(message, kind="adapter_unrecognized")
