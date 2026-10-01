"""A declared VRAM ceiling over a workflow's own cost_drivers.

A video template's `cost_drivers` name `num_frames`, `width` and `height`
because they move wall clock - but nothing checked whether a combination of
them also moves past the card the template's `cost` entry was measured on.
`validate_workflow` answered `valid: true` for a frame count the VAE's own
decode step could not fit beside whatever the pipeline keeps resident, and
the run found out 90+ seconds into denoising rather than before it started
(#265).

`vram_estimate` is declared the same way `cost` is - a maintainer's own
reading of what a stage needs, calibrated from a real run's reported
requirement at a known combination, never derived from a live probe. The
formula is deliberately the simplest one that fits a video VAE's own
scaling: memory beyond a fixed base grows with the voxel count a decode
tensor holds, which is why the variables it scales with are named rather
than assumed to be `num_frames` alone - an image template's width and
height matter here just as much as a video template's frame count does.

The projection is per step, over the definition the run executes -
substituted and expanded - so a `for_each` member is projected with its own
`num_frames` and its own reference list rather than with the workflow's
top-level variables (#479). A reference-conditioned pipeline holds every
reference's encoding resident beside the video latents, so an optional
`gb_per_reference` adds a fixed amount per reference the step will actually
pass; one whose media resolved null is dropped before the pipeline sees it
(#478) and costs nothing.

A workflow-level estimate was measured against one pipeline, not against
every step that happens to load one (#516) - a music-video template's image
step at a caller-chosen resolution has no business being judged by the H3
video estimate beside it just because both name `width`/`height`. Which
identity the estimate describes is inferred rather than declared twice: the
one every step that names *all* of the estimate's `voxel_variables` in its
own pipeline arguments shares (`_estimate_identity`, using
`pipeline_identity`). When that comes out ambiguous - no
step owns every voxel variable directly, more than one identity does, or the
steps carry no pipeline identity metadata at all, as every workflow before
#516 and most of this module's own tests do - projection falls back to every
pipeline step, unchanged from before.

The entries checked are the serving device's own (see `_entries_for`);
`bytes_per_voxel` was calibrated on CUDA, so a check against a Mac's capacity
is an estimate of an estimate.
"""

import numbers

from . import references as ref_prefixes
from .references import FROM_FILE_KEY, FROM_PREVIOUS_RESULT_KEY
from .for_each import FOR_EACH_KEY, MEMBER_SEPARATOR, render_path

KEY = "vram_estimate"
REFERENCES_KEY = "references"
_SOURCE_KEYS = (FROM_FILE_KEY, FROM_PREVIOUS_RESULT_KEY)


def _resolved(value, variables):
    """A `variable:` reference resolved against a template's own defaults -
    the index reads templates as written, not substituted."""
    if isinstance(value, str) and value.startswith(ref_prefixes.VARIABLE):
        return variables.get(value[len(ref_prefixes.VARIABLE) :])
    return value


def pipeline_identity(step, variables=None):
    """(component_type, model_name, workflow) for a step that loads a
    pipeline, or None for a step that does not."""
    pipeline = step.get("pipeline") if isinstance(step, dict) else None
    if not isinstance(pipeline, dict):
        return None
    variables = variables or {}
    configuration = pipeline.get("configuration")
    from_pretrained = pipeline.get("from_pretrained_arguments")
    configuration = configuration if isinstance(configuration, dict) else {}
    from_pretrained = from_pretrained if isinstance(from_pretrained, dict) else {}
    identity = tuple(
        _resolved(value, variables)
        for value in (
            configuration.get("component_type"),
            from_pretrained.get("model_name"),
            from_pretrained.get("workflow"),
        )
    )
    if not all(isinstance(part, (str, type(None))) for part in identity):
        return None
    if identity[1] is None:
        # No checkpoint named - nothing to match a catalog entry on
        return None
    return identity


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def declared_estimate(definition):
    estimate = definition.get(KEY)
    return estimate if isinstance(estimate, dict) else None


def reference_count(arguments):
    """How many of a step's `references` the pipeline will receive.

    An entry whose `from_file`/`from_previous_result` is null is dropped from
    the list before the pipeline sees it (#478, `realize_object`), so it
    costs nothing and is not counted; neither is a null entry. Every kind
    counts once - nothing measured says an audio reference costs less than
    an image one (#479).
    """
    references = arguments.get(REFERENCES_KEY) if isinstance(arguments, dict) else None
    if not isinstance(references, list):
        return 0
    count = 0
    for entry in references:
        if not isinstance(entry, dict):
            continue
        if any(entry.get(key) is not None for key in _SOURCE_KEYS):
            count += 1
    return count


def required_gb(estimate, values, references=0):
    """The projected requirement for `values` with `references` counted, or
    None when a voxel variable's value is not a number this pass can compute
    with - an undeclared reference or a list is somebody else's error."""
    product = 1.0
    for name in estimate.get("voxel_variables", []):
        number = _as_number(values.get(name))
        if number is None:
            return None
        product *= number
    base = estimate.get("base_gb", 0)
    per_voxel = estimate.get("bytes_per_voxel", 0)
    per_reference = estimate.get("gb_per_reference", 0)
    return base + per_voxel * product / (1024**3) + per_reference * references


def _entries_for(cost, device_type, capacity_gb):
    """The cost entries a projection is checked against on this device.

    Entries measured on this backend first. A backend no entry describes - a
    Mac, against a catalog measured on CUDA cards - is checked against its own
    capacity when it can report one, and against every entry when it cannot,
    so an unreadable ceiling never turns the guard off."""
    entries = [entry for entry in cost if isinstance(entry, dict)]
    if device_type is None:
        return entries
    matching = [
        entry
        for entry in entries
        if str(entry.get("device", "")).split(":")[0] == device_type
    ]
    if matching:
        return matching
    if capacity_gb is not None:
        return [
            {
                "name": f"this {device_type} device (recommended maximum)",
                "vram_gb": round(capacity_gb, 1),
            }
        ]
    return entries


def _estimate_identity(steps, variables, names):
    """The pipeline identity a workflow-level estimate describes, or None
    when that is ambiguous.

    The identity every step that names *all* of `names` in its own pipeline
    arguments shares - a step only partly naming the voxel variables (an
    image step with `width`/`height` but no `num_frames`) is not a candidate,
    so it cannot pull an unrelated estimate onto itself. None when no step
    qualifies, when the qualifying steps disagree, or when the identity found
    carries no real pipeline metadata (`pipeline_identity` returns None) -
    each of those means "cannot tell", not "no pipeline", and the caller
    falls back to projecting every step as it did before #516.
    """
    if not names:
        return None
    identities = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        pipeline = step.get("pipeline")
        if not isinstance(pipeline, dict):
            continue
        step_arguments = pipeline.get("arguments")
        step_arguments = step_arguments if isinstance(step_arguments, dict) else {}
        if not all(name in step_arguments for name in names):
            continue
        identities.add(pipeline_identity(step, variables))
    if len(identities) == 1:
        found = next(iter(identities))
        if found is not None:
            return found
    return None


def _projections(definition, estimate, arguments):
    """(step index, step, values, references, projected GB) for every step
    the estimate can project.

    Only a step that loads a pipeline is projected, since the ceiling is the
    pipeline's. Each voxel variable is read from that step's own arguments -
    substituted, so a for_each member carries its entry's `num_frames` -
    and from the workflow's variables (the caller's arguments over them) when
    the step does not name it. A definition with no steps at all is projected
    from its variables alone, as a single step. When the estimate's own
    pipeline identity can be determined (`_estimate_identity`), a step whose
    identity does not match it is not projected at all (#516) - the estimate
    was never measured for that pipeline.
    """
    variables = {**(definition.get("variables") or {}), **(arguments or {})}
    steps = definition.get("steps")
    if not isinstance(steps, list):
        projected = required_gb(estimate, variables)
        if projected is not None:
            yield None, None, variables, 0, projected
        return
    names = estimate.get("voxel_variables", [])
    identity = _estimate_identity(steps, variables, names)
    for index, step in enumerate(steps):
        pipeline = step.get("pipeline") if isinstance(step, dict) else None
        if not isinstance(pipeline, dict):
            continue
        if identity is not None and pipeline_identity(step, variables) != identity:
            continue
        step_arguments = pipeline.get("arguments")
        if not isinstance(step_arguments, dict):
            step_arguments = {}
        values = {
            name: step_arguments[name]
            if name in step_arguments
            else variables.get(name)
            for name in names
        }
        references = reference_count(step_arguments)
        projected = required_gb(estimate, values, references)
        if projected is not None:
            yield index, step, values, references, projected


def _variable_names(value):
    """Every name a `variable:` reference inside `value` spells."""
    if isinstance(value, str):
        if value.startswith(ref_prefixes.VARIABLE):
            yield value[len(ref_prefixes.VARIABLE) :]
    elif isinstance(value, dict):
        for item in value.values():
            yield from _variable_names(item)
    elif isinstance(value, list):
        for item in value:
            yield from _variable_names(item)


def _entry_position(source_indices, index):
    """Which member of its group expanded step `index` is - its entry's
    index in the for_each list, since members are emitted in list order."""
    source = source_indices[index]
    return sum(1 for s in source_indices[:index] if s == source)


def _where(estimate, index, step, supplied, source_indices, written):
    """The path an error sits at: the for_each entry a member came from, when
    one did, else `arguments` when the caller supplied anything the
    projection read and `variables` when every value was a default."""
    supplied = set(supplied or ())
    written_steps = written.get("steps") if isinstance(written, dict) else None
    source = (
        source_indices[index]
        if source_indices is not None
        and index is not None
        and index < len(source_indices)
        else index
    )
    written_step = (
        written_steps[source]
        if isinstance(written_steps, list)
        and isinstance(source, int)
        and source < len(written_steps)
        else None
    )
    if isinstance(written_step, dict) and FOR_EACH_KEY in written_step:
        position = (
            _entry_position(source_indices, index) if source_indices is not None else 0
        )
        entries = written_step[FOR_EACH_KEY]
        if isinstance(entries, str) and entries.startswith(ref_prefixes.VARIABLE):
            name = entries[len(ref_prefixes.VARIABLE) :]
            root = "arguments" if name in supplied else "variables"
            return f"{root}.{name}[{position}]"
        return render_path(("steps", source, FOR_EACH_KEY, position))
    read = set(estimate.get("voxel_variables", []))
    if isinstance(written_step, dict):
        pipeline = written_step.get("pipeline")
        if isinstance(pipeline, dict):
            read.update(_variable_names(pipeline.get("arguments")))
    return "arguments" if read & supplied else "variables"


def _formula(estimate):
    voxel_variables = estimate.get("voxel_variables", [])
    formula = (
        f"Declared ceiling: {estimate.get('base_gb', 0)} GB base plus "
        f"{estimate.get('bytes_per_voxel', 0):.4g} bytes per unit of "
        f"{' * '.join(voxel_variables)}"
    )
    if "gb_per_reference" in estimate:
        formula += f" plus {estimate['gb_per_reference']} GB per reference"
    return formula


def vram_estimate_errors(
    definition,
    arguments=None,
    supplied=(),
    device_type=None,
    capacity_gb=None,
    source_indices=None,
    written=None,
):
    """The step a declared vram_estimate projects past a `cost` entry, as
    [{path, message}] - refused, not warned, since the failure this guards
    is an OOM partway through a run that already spent minutes loading.

    `definition` is the one the run will execute: substituted and expanded,
    so every for_each member is projected with its own frames and references
    (#479). `source_indices` and `written` (the definition as written) map
    a member back to the list entry it came from. One step is reported - the
    largest - rather than one error per shot: cutting that one is the fix
    the caller has to make first, and the rest follow from the same formula.
    """
    estimate = declared_estimate(definition)
    if estimate is None:
        return []
    cost = definition.get("cost")
    cost = cost if isinstance(cost, list) else []
    entries = _entries_for(cost, device_type, capacity_gb)
    capacities = [
        entry.get("vram_gb") for entry in entries if entry.get("vram_gb") is not None
    ]
    if not capacities:
        return []
    over = [
        projection
        for projection in _projections(definition, estimate, arguments)
        if projection[4] > min(capacities)
    ]
    if not over:
        return []
    index, step, values, references, projected = max(over, key=lambda p: p[4])

    voxel_variables = estimate.get("voxel_variables", [])
    reason = estimate.get("reason")
    because = f" - {reason}" if reason else ""
    name = step.get("name") if isinstance(step, dict) else None
    member = (
        f"Member '{name}': "
        if isinstance(name, str) and MEMBER_SEPARATOR in name
        else ""
    )
    with_references = (
        f" with {references} reference{'s' if references != 1 else ''}"
        if "gb_per_reference" in estimate or references
        else ""
    )
    where = _where(estimate, index, step, supplied, source_indices, written)
    errors = []
    for entry in entries:
        capacity = entry.get("vram_gb")
        if capacity is None or projected <= capacity:
            continue
        device_label = entry.get("name") or entry.get("device", "this device")
        errors.append(
            {
                "path": where,
                "message": (
                    f"{member}{'*'.join(voxel_variables)} = "
                    f"{'*'.join(str(values.get(v)) for v in voxel_variables)}"
                    f"{with_references} projects to {projected:.2f} GB VRAM, "
                    f"above the {capacity} GB declared for {device_label}"
                    f"{because}. {_formula(estimate)}"
                ),
            }
        )
    return errors


def apply_vram_estimate(definition, variables, device_type=None, capacity_gb=None):
    """The run-time half of the check above - the backstop for a caller
    that skips validate_workflow (or an inline/composed workflow static
    validation never saw). `definition` is the expanded one the run
    executes. Raises ValueError for the same projection vram_estimate_errors
    refuses, so run() cannot start a job the pipeline was always going to
    OOM on (#265, #479)."""
    errors = vram_estimate_errors(
        definition,
        variables if isinstance(variables, dict) else None,
        device_type=device_type,
        capacity_gb=capacity_gb,
    )
    if errors:
        raise ValueError(errors[0]["message"])
