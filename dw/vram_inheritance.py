"""A VRAM ceiling a hand-built workflow inherits from the catalog.

`vram_estimate` is declared by a template, so a workflow an agent writes by
hand - the same pipeline, the same checkpoint, its own shots - declares none
and was never checked. Both OOMs in #479's report were inline Ref2VA
workflows that `validate_workflow` passed.

The ceiling belongs to the pipeline, not to the file that declared it, so a
step is matched to the catalog by *pipeline identity*: the step's
`component_type`, its `from_pretrained_arguments.model_name` and its
`from_pretrained_arguments.workflow` (the fields `adapter_compatibility`
reads - H3's `ref2va` and `t2va` load different partitions and declare
different numbers). The index is derived from the catalog's own
declarations; this module holds no numbers and names no model.

A template contributes only when every pipeline step it holds shares one
identity: a workflow-level `vram_estimate` on a template that also loads an
image model and a music model says nothing about which of the three it was
measured for, so attributing it to all of them would be a guess. It also has
to declare `cost` entries, since those carry the card the ceiling is checked
against.

An inherited ceiling warns and never refuses (#479, Q3): the hand-built
workflow may offload or quantize differently from the template, and the
escape hatch for a leaner config is declaring its own `vram_estimate`, which
always wins. There is no run-time backstop - the worker has no catalog, and
the pre-queue check covers every server submission.
"""

from .vram_estimate import KEY as ESTIMATE_KEY
from .vram_estimate import pipeline_identity, vram_estimate_errors

KIND = "vram_projection_inherited"


def template_identity(definition):
    """The one pipeline identity every pipeline step of `definition` shares,
    or None when it holds none or more than one."""
    variables = definition.get("variables")
    variables = variables if isinstance(variables, dict) else {}
    steps = definition.get("steps")
    identities = {
        pipeline_identity(step, variables)
        for step in (steps if isinstance(steps, list) else [])
        if isinstance(step, dict) and isinstance(step.get("pipeline"), dict)
    }
    if len(identities) != 1 or None in identities:
        return None
    return identities.pop()


def declarations(catalog):
    """Every (identity, name, definition) a catalog of (name, definition)
    pairs offers the index: a declared `vram_estimate` on a single-identity
    template. Cost is not required here - the agreement test reads the
    numbers of a template that declares no card too."""
    for name, definition in catalog:
        if not isinstance(definition, dict):
            continue
        if not isinstance(definition.get(ESTIMATE_KEY), dict):
            continue
        identity = template_identity(definition)
        if identity is not None:
            yield identity, name, definition


def _preference(entry):
    # Fewest steps first - the plainest statement of the pipeline's ceiling -
    # then by name, so the source a warning names never depends on listing
    # order
    _, name, definition = entry
    return (len(definition.get("steps") or []), name)


def build_index(catalog):
    """identity -> {vram_estimate, cost, template} from a catalog of
    (name, definition) pairs. Only a template that declares `cost` entries
    to check against contributes; where several declare the same identity,
    the one with the fewest steps is the named source (the agreement test
    pins that their numbers are the same)."""
    found = {}
    for entry in sorted(declarations(catalog), key=_preference):
        identity, name, definition = entry
        cost = definition.get("cost")
        if identity in found or not isinstance(cost, list) or not cost:
            continue
        found[identity] = {
            "vram_estimate": definition[ESTIMATE_KEY],
            "cost": cost,
            "template": name,
        }
    return found


def inherited_vram_warnings(
    definition,
    index,
    arguments=None,
    supplied=(),
    device_type=None,
    capacity_gb=None,
    source_indices=None,
    written=None,
    capacity_label=None,
):
    """Every catalog ceiling `definition` (expanded, as the run executes it)
    projects past, as warning strings - one per matched identity, naming the
    largest step. Nothing when the definition declares its own
    `vram_estimate`, which always wins."""
    if not index or not isinstance(definition, dict):
        return []
    if isinstance(definition.get(ESTIMATE_KEY), dict):
        return []
    steps = definition.get("steps")
    if not isinstance(steps, list):
        return []
    variables = definition.get("variables")
    variables = variables if isinstance(variables, dict) else {}
    identities = [pipeline_identity(step, variables) for step in steps]
    warnings = []
    for identity in dict.fromkeys(i for i in identities if i in index):
        entry = index[identity]
        # Only this identity's steps, at their own indices so a member's
        # path still maps back through source_indices
        matched = {
            **definition,
            "steps": [
                step if step_identity == identity else {}
                for step, step_identity in zip(steps, identities)
            ],
            # The template's reason is its own calibration story; the
            # warning names the template, which is where to read it
            ESTIMATE_KEY: {
                key: value
                for key, value in entry["vram_estimate"].items()
                if key != "reason"
            },
            "cost": entry["cost"],
        }
        errors = vram_estimate_errors(
            matched,
            arguments,
            supplied=supplied,
            device_type=device_type,
            capacity_gb=capacity_gb,
            capacity_label=capacity_label,
            source_indices=source_indices,
            written=written,
        )
        if not errors:
            continue
        error = errors[0]
        warnings.append(
            f"{error['path']}: {KIND}: {error['message'].rstrip('.')}. The "
            f"ceiling is inherited from '{entry['template']}', which loads "
            f"the same pipeline; this workflow declares no vram_estimate and "
            f"its offload and quantization config may differ from the "
            f"template's, so this warns rather than refuses - declare a "
            f"vram_estimate to be judged by your own numbers"
        )
    return warnings
