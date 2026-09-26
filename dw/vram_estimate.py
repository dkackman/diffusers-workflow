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

The entries checked are the serving device's own (see `_entries_for`);
`bytes_per_voxel` was calibrated on CUDA, so a check against a Mac's capacity
is an estimate of an estimate.
"""

import numbers

KEY = "vram_estimate"


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


def required_gb(estimate, values):
    """The projected requirement for `values`, or None when a voxel
    variable's value is not a number this pass can compute with - an
    undeclared reference or a for_each list is somebody else's error."""
    product = 1.0
    for name in estimate.get("voxel_variables", []):
        number = _as_number(values.get(name))
        if number is None:
            return None
        product *= number
    base = estimate.get("base_gb", 0)
    per_voxel = estimate.get("bytes_per_voxel", 0)
    return base + per_voxel * product / (1024**3)


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


def vram_estimate_errors(
    definition, arguments=None, supplied=(), device_type=None, capacity_gb=None
):
    """Every `cost` entry a declared vram_estimate projects to exceed, as
    [{path, message}] - refused, not warned, since the failure this guards
    is an OOM partway through a run that already spent minutes loading."""
    estimate = declared_estimate(definition)
    if estimate is None:
        return []
    voxel_variables = estimate.get("voxel_variables", [])
    values = {**(definition.get("variables") or {}), **(arguments or {})}
    projected = required_gb(estimate, values)
    if projected is None:
        return []
    cost = definition.get("cost")
    if not isinstance(cost, list):
        return []
    reason = estimate.get("reason")
    because = f" - {reason}" if reason else ""
    where = (
        "arguments"
        if any(v in (supplied or ()) for v in voxel_variables)
        else "variables"
    )
    errors = []
    for entry in _entries_for(cost, device_type, capacity_gb):
        capacity = entry.get("vram_gb")
        if capacity is None or projected <= capacity:
            continue
        device_label = entry.get("name") or entry.get("device", "this device")
        errors.append(
            {
                "path": where,
                "message": (
                    f"{'*'.join(voxel_variables)} = "
                    f"{'*'.join(str(values.get(v)) for v in voxel_variables)} "
                    f"projects to {projected:.2f} GB VRAM, above the "
                    f"{capacity} GB declared for {device_label}{because}. "
                    f"Declared ceiling: {estimate.get('base_gb', 0)} GB base plus "
                    f"{estimate.get('bytes_per_voxel', 0):.4g} bytes per unit of "
                    f"{' * '.join(voxel_variables)}"
                ),
            }
        )
    return errors


def apply_vram_estimate(definition, variables, device_type=None, capacity_gb=None):
    """The run-time half of the check above - the backstop for a caller
    that skips validate_workflow (or an inline/composed workflow static
    validation never saw). Raises ValueError for the same projection
    vram_estimate_errors refuses, so run() cannot start a job the decode
    step was always going to OOM on (#265)."""
    if not isinstance(variables, dict):
        return
    errors = vram_estimate_errors(
        definition, variables, device_type=device_type, capacity_gb=capacity_gb
    )
    if errors:
        raise ValueError(errors[0]["message"])
