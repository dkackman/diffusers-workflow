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


def vram_estimate_errors(definition, arguments=None, supplied=()):
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
    where = "arguments" if any(v in (supplied or ()) for v in voxel_variables) else "variables"
    errors = []
    for entry in cost:
        if not isinstance(entry, dict):
            continue
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
