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

import json

from .for_each import ForEachError, expand_for_each
from .vram_estimate import KEY as ESTIMATE_KEY
from .vram_estimate import reference_count
from .variables import (
    VariableNotFoundError,
    replace_variables,
    same_number,
    undeclared_variable_references,
)
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


def _found(node, key, variables):
    """Every value `key` carries anywhere under `node`, as sorted JSON text,
    after `node`'s `variable:` references are substituted the way a run
    substitutes them (`replace_variables`: at any depth, and a key whose
    reference resolves to null dropped). A `loras` list carries its repo as
    a reference inside each entry (#866)."""
    return _carried(_substituted(node, variables), key)


def _substituted(node, variables):
    """`replace_variables(node, variables)`, except that a reference to a
    variable `variables` doesn't declare stays as written instead of failing
    the whole node, so the rest of it is still compared."""
    try:
        return replace_variables(node, variables)
    except VariableNotFoundError:
        pass
    if isinstance(node, dict):
        result = {}
        for name, value in node.items():
            try:
                result.update(replace_variables({name: value}, variables))
            except VariableNotFoundError:
                result[name] = _substituted(value, variables)
        return result
    if isinstance(node, list):
        result = []
        for item in node:
            try:
                result.extend(replace_variables([item], variables))
            except VariableNotFoundError:
                result.append(_substituted(item, variables))
        return result
    return node


def _carried(node, key):
    values = set()
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                # By value, not spelling: a client's JSON round trip turns
                # the template's `lora_scale` 1.0 into 1 (#908)
                values.add(json.dumps(same_number(value), sort_keys=True))
            values |= _carried(value, key)
    elif isinstance(node, list):
        for item in node:
            values |= _carried(item, key)
    return values


# What changes a pipeline's cost without changing its identity (#797):
# (label, key) pairs read wherever a step sets them. The last three are the
# keys a pipeline step sets its attention backend, step cache and LoRAs
# under (#866)
_COST_SETTINGS = (
    ("offload", "offload"),
    ("quantization", "quantization_config"),
    ("frame count", "num_frames"),
    ("attention backend", "attention_backend"),
    ("cache", "cache"),
    ("LoRA", "loras"),
)


def inherited_differences(definition, template, arguments=None):
    """What `definition` sets differently from `template` among the things
    that move a pipeline's cost without changing which pipeline it is -
    offload, quantization, frame count, attention backend, cache, LoRA - as
    short phrases (#797, #866). A figure inherited from `template` is an
    approximation to the extent these differ. `arguments` (the caller's)
    override `definition`'s variables."""
    differs = [
        label
        for label, key in _COST_SETTINGS
        if sets_differently(definition, template, key, arguments)
    ]
    # A reference is encoded and attended on every pass, so a count other
    # than the template's moves the cost (#915); it is named, not scaled
    if reference_counts(definition, arguments) != reference_counts(template):
        differs.append("reference count")
    return differs


def reference_counts(definition, arguments=None):
    """The set of reference counts the pipeline steps of `definition` pass,
    `variable:` references resolved and `for_each` expanded, so each member
    step counts its own entry's `references` (#915). A step's count is
    `vram_estimate.reference_count`'s, the one the VRAM projection uses."""
    variables = dict(definition.get("variables") or {})
    variables.update(arguments or {})
    expanded = _expanded_steps(definition, variables)
    if expanded is None:
        return set()
    return {
        reference_count(step["pipeline"].get("arguments"))
        for step in expanded
        if isinstance(step, dict) and isinstance(step.get("pipeline"), dict)
    }


def _expanded_steps(definition, variables):
    """`definition`'s steps with `variables` substituted and `for_each`
    expanded as the run does, so `item:` references carry each entry's own
    value; None when the expansion fails."""
    steps = _substituted(definition.get("steps"), variables)
    try:
        return expand_for_each({"steps": steps}).get("steps") or []
    except ForEachError:
        return None


def sets_differently(definition, template, key, arguments=None):
    """Whether `definition`'s steps set `key` to other values than
    `template`'s, `variable:` references resolved (`arguments` override
    `definition`'s variables)."""
    return _found_expanded(
        definition, key, _compared_variables(definition, arguments)
    ) != _found_expanded(template, key, _compared_variables(template))


def _compared_variables(definition, arguments=None):
    """`definition`'s variables, `arguments` over them, with every name its
    references use but it doesn't declare set to null. A copy that leaves out
    a variable whose default is null (`lora_alpha`) then drops that key as
    the template it was copied from does, instead of comparing the reference
    text against nothing (#866)."""
    variables = dict(definition.get("variables") or {})
    variables.update(arguments or {})
    for _, name in undeclared_variable_references(definition):
        variables.setdefault(name, None)
    return variables


def _found_expanded(definition, key, variables):
    # A for_each step's `item:` references resolve per entry, so the value an
    # entry sets is compared, not the reference text (#916)
    expanded = _expanded_steps(definition, variables)
    if expanded is None:
        return _found(definition.get("steps"), key, variables)
    return _carried(expanded, key)


def pipeline_values(definition, key, arguments=None):
    """The number each pipeline step sets `key` to (a `variable:` reference
    resolved; `arguments` override the variables; the largest where a step
    sets it twice), None for a step that sets none or a non-numeric one -
    for scaling an inherited price along it (#851, #866)."""
    variables = dict(definition.get("variables") or {})
    variables.update(arguments or {})
    found = []
    for step in definition.get("steps") or []:
        if not isinstance(step, dict) or not isinstance(step.get("pipeline"), dict):
            continue
        numbers = [
            value
            for value in map(json.loads, _found(step, key, variables))
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        found.append(max(numbers) if numbers else None)
    return found


def pipeline_frames(definition, arguments=None):
    """The frame count each pipeline step sets (`num_frames`), as
    `pipeline_values` reads it - for scaling an inherited price by frame
    count (#851)."""
    return pipeline_values(definition, "num_frames", arguments)


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
    base_dir=None,
    probe=None,
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
            base_dir=base_dir,
            probe=probe,
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
