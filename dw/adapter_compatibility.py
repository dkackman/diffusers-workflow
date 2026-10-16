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

The LoRA catalog comes first, the file name second (#870). The server hands
in `catalog_index` over the one catalog every workspace shares
(`dw/server/deps.py` `lora_index`). An entry whose `workflow` list sits in
one partition decides the adapter's partition, and its `status` decides how
much that counts:

- a `proven` entry is used: a step on the other partition is an error
  naming the entry, and a step on its own one is clean;
- a `trial` or `rejected` entry only warns when the step is on the other
  partition, and clears the "names neither" warning when it is not - a
  file-name error still stands, since only `proven` refuses;
- a `proven` entry and a file-name token that disagree are a warning naming
  both, never an error;
- a file the catalog does not cover, or an entry whose `workflow` is null or
  spans both partitions, falls to the file-name rule unchanged.

The CLI, the worker's sub-workflow re-validation and `warn_adapters` pass no
index and keep the file-name rule alone, so the server is stricter than the
worker, never looser.
"""

import logging

from . import references
from .references import render_path

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
MODEL_NAME_KEY = "model_name"
WEIGHT_NAME_KEY = "weight_name"
WORKFLOW_KEY = "workflow"
FROM_PRETRAINED_KEY = "from_pretrained_arguments"


def _trained_for(weight_name):
    """Which partition an adapter's file name says it was trained against -
    'reference', 'keyframe', or None when the name says neither."""
    lowered = weight_name.lower()
    if REFERENCE_TOKEN in lowered:
        return "reference"
    if KEYFRAME_TOKEN in lowered:
        return "keyframe"
    return None


PARTITIONS = {"reference": "transformer_ref", "keyframe": "transformer"}
# Which of two entries for one file the index keeps: the stronger status
STATUS_STRENGTH = {"proven": 0, "trial": 1, "rejected": 2}


def catalog_index(entries):
    """`{(model_name, weight_name): (entry name, workflows, status)}` over
    the catalog entries (`{name: entry}`) that declare a `workflow` and a
    file. The revision is not part of the key (#870 Q2)."""
    index = {}
    for name, entry in sorted(entries.items()):
        workflows = entry.get(WORKFLOW_KEY)
        weight_name = entry.get(WEIGHT_NAME_KEY)
        if not workflows or not isinstance(weight_name, str):
            continue
        if isinstance(workflows, str):
            workflows = [workflows]
        key = (entry.get(MODEL_NAME_KEY), weight_name)
        status = entry.get("status")
        held = index.get(key)
        weakest = len(STATUS_STRENGTH)
        if held is not None and STATUS_STRENGTH.get(
            held[2], weakest
        ) <= STATUS_STRENGTH.get(status, weakest):
            continue
        index[key] = (name, tuple(workflows), status)
    return index


def _catalog_partition(workflows):
    """The partition a catalog entry's `workflow` list sits in -
    'reference', 'keyframe' - or None when it spans both."""
    listed = set(workflows)
    if listed and listed <= REFERENCE_WORKFLOWS:
        return "reference"
    if listed and listed <= KEYFRAME_WORKFLOWS:
        return "keyframe"
    return None


def _token(partition):
    return REFERENCE_TOKEN if partition == "reference" else KEYFRAME_TOKEN


def _filename_problem(workflow, weight_name, trained, wants):
    """(severity, message) the file-name rule gives, or None."""
    if trained == wants:
        return None
    if trained is None:
        return (
            "warning",
            f"'{weight_name}' names neither '{REFERENCE_TOKEN}' nor "
            f"'{KEYFRAME_TOKEN}', so it cannot be checked against this "
            f"step's '{workflow}' workflow. A '{workflow}' step denoises "
            f"against the '{PARTITIONS[wants]}' partition, and an adapter "
            f"trained against the other one loads onto it without error and "
            f"only degrades the result",
        )
    return (
        "error",
        f"'{weight_name}' is trained against the '{PARTITIONS[trained]}' "
        f"partition and this step runs the '{workflow}' workflow, which "
        f"denoises against '{PARTITIONS[wants]}'. MiniMax-H3 loads it anyway "
        f"and the run succeeds - the only symptom is a worse result - so it "
        f"is refused here. Use an adapter trained for this workflow (its name "
        f"carries '{_token(wants)}')",
    )


def _problems(workflow, weight_name, model_name=None, catalog=None):
    """Every (severity, message) for one adapter on one step; empty when
    the pairing is right. The catalog first, the file name second (the
    module docstring)."""
    trained = _trained_for(weight_name)
    wants = "reference" if workflow in REFERENCE_WORKFLOWS else "keyframe"
    found = (catalog or {}).get((model_name, weight_name))
    decided = _catalog_partition(found[1]) if found else None
    if decided is None:
        problem = _filename_problem(workflow, weight_name, trained, wants)
        return [problem] if problem else []
    entry_name, workflows, status = found
    lists_it = (
        f"'{weight_name}': catalog entry '{entry_name}' ({status}) lists it "
        f"for '{', '.join(workflows)}', which denoises against "
        f"'{PARTITIONS[decided]}'"
    )
    if status == "proven":
        # The catalog and the file name disagree: say both, refuse neither
        # (#870 Q1)
        if trained is not None and trained != decided:
            return [
                (
                    "warning",
                    f"{lists_it}, but its name carries '{_token(trained)}', "
                    f"which says '{PARTITIONS[trained]}'. This step runs "
                    f"'{workflow}', which denoises against "
                    f"'{PARTITIONS[wants]}'. Check which is right; if it is "
                    f"the catalog, correct the entry with save_lora",
                )
            ]
        if decided == wants:
            return []
        return [
            (
                "error",
                f"{lists_it}, and this step runs the '{workflow}' workflow, "
                f"which denoises against '{PARTITIONS[wants]}'. MiniMax-H3 "
                f"loads it anyway and the run succeeds - the only symptom is "
                f"a worse result - so it is refused here. Use an adapter the "
                f"catalog lists for '{workflow}' (list_loras), or correct the "
                f"entry with save_lora if it is wrong",
            )
        ]
    # A trial or rejected entry only warns. It never lifts a file-name
    # error: only a proven entry may overrule one
    problems = []
    if trained is not None and trained != wants:
        problems.append(_filename_problem(workflow, weight_name, trained, wants))
    if decided != wants:
        problems.append(
            (
                "warning",
                f"{lists_it}, and this step runs the '{workflow}' workflow, "
                f"which denoises against '{PARTITIONS[wants]}'. An adapter "
                f"trained against the other partition loads without error "
                f"and only degrades the result. The entry is '{status}', so "
                f"this is not refused; promoting it to proven would make it "
                f"refuse",
            )
        )
    return problems


def _lora_problems(steps, source_indices, written=None, supplied=(), catalog=None):
    """Every adapter/workflow mismatch in an expanded step list, as
    (severity, path, message). `catalog` is `catalog_index`'s answer, built
    once by the caller and looked up here per adapter; None is the file-name
    rule alone.

    `written` is the definition as the author wrote it and `supplied` the
    argument names the caller passed: when the value arrived through a
    variable the caller set, the path reported is `arguments.<name>`, where
    they wrote it, rather than the step it landed in.
    """
    written_steps = (written or {}).get("steps")
    found = []
    for _, _, pipeline, source, where in references.iter_steps(
        steps, source_indices, "pipeline"
    ):
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
        for position, lora in enumerate(loras):
            if not isinstance(lora, dict):
                continue
            weight_name = lora.get(WEIGHT_NAME_KEY)
            # Values another pass resolves; one still spelled out here is not
            # this pass's complaint
            if not isinstance(weight_name, str) or references.is_ref(
                references.UNRESOLVED, weight_name
            ):
                continue
            path = _path_for(written_steps, source, position, supplied) or render_path(
                ("steps", source, "pipeline", LORAS_KEY, position, WEIGHT_NAME_KEY)
            )
            for severity, message in _problems(
                workflow, weight_name, lora.get(MODEL_NAME_KEY), catalog
            ):
                found.append((severity, path, message + where))
    return found


def _path_for(written_steps, source, position, supplied, key=WEIGHT_NAME_KEY):
    """`arguments.<name>` when the entry's `key` came from a variable the
    caller set, else None - the value is reported where it was written."""
    if not isinstance(written_steps, list) or source >= len(written_steps):
        return None
    step = written_steps[source]
    pipeline = step.get("pipeline") if isinstance(step, dict) else None
    loras = pipeline.get(LORAS_KEY) if isinstance(pipeline, dict) else None
    if not isinstance(loras, list) or position >= len(loras):
        return None
    entry = loras[position]
    reference = entry.get(key) if isinstance(entry, dict) else None
    variable = references.ref_name(references.VARIABLE, reference)
    if variable is None:
        return None
    return f"arguments.{variable}" if variable in (supplied or ()) else None


def adapter_errors(
    workflow_definition, source_indices=None, written=None, supplied=(), catalog=None
):
    """Every adapter loaded onto the wrong checkpoint partition, as
    [{path, message}]. The definition is the expanded, substituted one, so a
    `for_each` member's own adapter is checked as it will run. `catalog` is
    `catalog_index`'s answer, when the caller has the catalog."""
    return [
        {"path": path, "message": message}
        for severity, path, message in _lora_problems(
            workflow_definition.get("steps") or [],
            source_indices,
            written,
            supplied,
            catalog,
        )
        if severity == "error"
    ]


DISABLED_MESSAGE = (
    "this lora's model_name is null, so it is not loaded and the step runs "
    "without it. If it was a step-distillation (turbo) lora, the step's "
    "num_inference_steps and sigma shift were set for it - raise them to "
    "the base model's schedule too, or the step runs undertrained"
)


def _disabled_loras(steps, source_indices, written=None, supplied=()):
    """Every `loras` entry switched off with a null model_name, as
    (path, message). Any pipeline, not only H3: the off switch is the
    engine's, and so is saying that it was used."""
    written_steps = (written or {}).get("steps")
    found = []
    for _, _, pipeline, source, where in references.iter_steps(
        steps, source_indices, "pipeline"
    ):
        loras = pipeline.get(LORAS_KEY)
        if not isinstance(loras, list):
            continue
        for position, lora in enumerate(loras):
            if not isinstance(lora, dict) or lora.get(MODEL_NAME_KEY) is not None:
                continue
            path = _path_for(
                written_steps, source, position, supplied, key=MODEL_NAME_KEY
            ) or render_path(
                ("steps", source, "pipeline", LORAS_KEY, position, MODEL_NAME_KEY)
            )
            found.append((path, DISABLED_MESSAGE + where))
    return found


def adapter_warnings(
    workflow_definition, source_indices=None, written=None, supplied=(), catalog=None
):
    """Every adapter whose name says nothing about what it was trained for,
    and every one switched off, as messages - valid, and said out loud
    because nothing at run time will."""
    steps = workflow_definition.get("steps") or []
    return [
        f"{path}: {message}"
        for severity, path, message in _lora_problems(
            steps, source_indices, written, supplied, catalog
        )
        if severity == "warning"
    ] + [
        f"{path}: {message}"
        for path, message in _disabled_loras(steps, source_indices, written, supplied)
    ]


def warn_adapters(workflow_definition):
    """Say the adapter warnings where whoever asked for the run can read it
    - a run started from the CLI or a rerun never passed through the
    validate route."""
    from . import events

    steps = workflow_definition.get("steps") or []
    for severity, path, message in _lora_problems(steps, None):
        if severity == "warning":
            events.emit_warning(f"{path}: {message}", kind="adapter_unrecognized")
    for path, message in _disabled_loras(steps, None):
        events.emit_warning(f"{path}: {message}", kind="lora_disabled")
