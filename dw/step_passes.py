"""How many transformer passes a pipeline call runs (#1316).

Most samplers take `num_inference_steps` N. A flow-matching sampler such as
MiniMax-H3's counts N sigma grid points and so runs N - 1 passes; that is the
rule the step-count consumers have read until now. A distilled sampler can
instead fix its schedule in the pipeline's config as a `denoising_step_list`
of sigma points and take no `num_inference_steps` at all: it runs one pass
between each pair of points, `len - 1`. Read without this module, every
consumer would fall back to its own default (50 steps for TST, the
`num_inference_steps` a cache declares, a bar of unknown length) and mis-key.

The rule is generic: no model is named here. The step list is looked for in
the call's arguments first, then in the pipeline's config - a loaded
pipeline's `config` (or an attribute of that name, as a modular pipeline
exposes its config values), or, at validate time, a workflow step's
`pipeline` block, whose `arguments` and `from_pretrained_arguments` are the
only places a workflow can state one.

Consumers: TST's step ramp (`pipeline_processors/h3_tst.py`), MagCache
calibration (`pipeline_processors/mag_calibration.py`), per-step progress
(`pipeline_processors/pipeline.py`), cost scaling (`cost_scaling.py`) and the
short-schedule cache warning (`acceleration_warnings.py`). A step-list
pipeline skips MagCache calibration and TST's step-keyed schedule, with
`STEP_LIST_UNSUPPORTED` said instead.
"""

import math

STEP_LIST_KEY = "denoising_step_list"
STEPS_KEY = "num_inference_steps"

# The tail of the warning a step-keyed feature gives on a step-list pipeline
STEP_LIST_UNSUPPORTED = "not supported on a fixed step-list sampler"


def _as_step_list(value):
    """`value` as a list of finite numbers of at least two points, or None."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    for point in value:
        if isinstance(point, bool) or not isinstance(point, (int, float)):
            return None
        if not math.isfinite(point):
            return None
    return list(value)


def _config_value(pipeline):
    """The `denoising_step_list` a pipeline's config carries, as written."""
    if pipeline is None:
        return None
    if isinstance(pipeline, dict):
        # A workflow step's `pipeline` block, at validate time
        for key in ("arguments", "from_pretrained_arguments"):
            block = pipeline.get(key)
            if isinstance(block, dict) and STEP_LIST_KEY in block:
                return block[STEP_LIST_KEY]
        return None
    config = getattr(pipeline, "config", None)
    if hasattr(config, "get"):
        try:
            value = config.get(STEP_LIST_KEY)
        except Exception:
            value = None
        if value is not None:
            return value
    try:
        return getattr(pipeline, STEP_LIST_KEY, None)
    except Exception:
        return None


def step_list(pipeline, arguments=None):
    """The fixed step list a call of `pipeline` with `arguments` runs, or None
    when it runs a `num_inference_steps` schedule.

    Args:
        pipeline: A loaded pipeline, a workflow step's `pipeline` block, or
            None (only the arguments are read).
        arguments: The call's arguments, or None.
    """
    if isinstance(arguments, dict) and STEP_LIST_KEY in arguments:
        points = _as_step_list(arguments[STEP_LIST_KEY])
        if points is not None:
            return points
    return _as_step_list(_config_value(pipeline))


def step_list_passes(pipeline, arguments=None):
    """The passes a step-list pipeline runs, `len - 1`, or None when the call
    has no step list (the caller keeps its own `num_inference_steps` count)."""
    points = step_list(pipeline, arguments)
    return None if points is None else len(points) - 1


def pipeline_step_passes(pipeline, arguments=None, default=None):
    """The transformer passes a call of `pipeline` with `arguments` runs.

    A step list's `len - 1`; otherwise `num_inference_steps` N read as N - 1
    passes (N sigma points, the flow-matching rule). An N that is missing or
    not a whole number of at least 2 falls back to `default` steps, and to
    None when there is no default.
    """
    passes = step_list_passes(pipeline, arguments)
    if passes is not None:
        return passes
    steps = arguments.get(STEPS_KEY) if isinstance(arguments, dict) else None
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        steps = default
    return None if steps is None else steps - 1
