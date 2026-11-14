"""`turbo_lead_in`: run a MiniMax-H3 call's first evaluations without its
turbo LoRA (#1326).

A turbo (step-distilled) LoRA fixes the composition in its first, high-noise
evaluations; running those few at the base model's weights and the rest at
the adapter's configured scale lets the base model lay out the scene and the
distilled adapter finish it in few steps. The idea is ComfyUI-Continuity's
(docs/ATTRIBUTIONS.md); the code here is dw's own.

`turbo_lead_in` N, an H3 call argument, counts transformer evaluations: the
first N of each pipeline call run the adapter named `turbo` at scale 0, then
its configured scale. Every chain segment and re-roll is its own pipeline
call, so the count starts again at each.

H3's denoise loop is a diffusers modular block with no step callback, so the
switch is a forward pre-hook on the H3 transformer(s), as TST's
(`h3_tst.py`): it counts evaluations by distinct timestep - H3 is guidance
distilled, one forward per step, and the two DiT partitions share one count -
and sets the adapter's scale on the evaluation that starts past N. Only
`turbo` is touched; every other adapter keeps its scale. The scale is set
per layer through peft's `set_scale` (what `adapters.set_adapter_scales`
does, never `set_adapters`, which fails on a warm offloaded pipeline), and
each layer's exact `turbo` scaling is put back in a `finally`, on success,
error and cancel alike.

Under torch.compile peft's scaling is a Python float the compiled forward
guards on, so the scale change can cost a recompile (docs/ACCELERATION.md).

Refused, at validate and at the run: a value that is not a whole number of
at least 0; N at or past the call's evaluations; a step with no adapter
named `turbo` (the error lists the names it has); and a `first_block` cache,
whose residual threshold reads the very change the lead-in makes.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

from .. import references
from ..events import emit_log
from ..for_each import render_path
from ..sparse_attention import forward_arg
from ..step_passes import pipeline_step_passes
from .h3_rules import not_h3

logger = logging.getLogger("dw")

LEAD_IN_INPUT = "turbo_lead_in"
TURBO_ADAPTER = "turbo"
# The cache a lead-in refuses: its first-block residual threshold sees the
# adapter switch as a large change, and cached steps would skip the switch
FIRST_BLOCK_CACHE = "first_block"
# H3's `num_inference_steps` default (diffusers' input template), used when a
# call names none - the same figure as h3_tst.DEFAULT_STEPS
DEFAULT_STEPS = 50


def _unresolved(value):
    return isinstance(value, str) and references.is_ref(references.UNRESOLVED, value)


def lead_in_value_problem(value):
    """Why `value` cannot be a `turbo_lead_in`, or None."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return (
            f"{LEAD_IN_INPUT} is a whole number of transformer evaluations, "
            f"0 or more, got {value!r}"
        )
    return None


def evaluations(pipeline, arguments):
    """The transformer evaluations a call runs: a step list's `len - 1`, or
    `num_inference_steps` N read as N - 1, H3's default when it names none.
    `pipeline` is a loaded pipeline or, at validate, the step's block."""
    return pipeline_step_passes(pipeline, arguments, default=DEFAULT_STEPS)


def count_problem(value, total):
    """Why a lead-in of `value` cannot run in a call of `total` evaluations."""
    if total is not None and value >= total:
        return (
            f"{LEAD_IN_INPUT} {value} leaves no evaluation for the turbo adapter: "
            f"this call runs {total} transformer evaluation(s) "
            f"(num_inference_steps N runs N - 1), so it must be less than {total}"
        )
    return None


def no_turbo_problem(names):
    """The refusal for a step with no adapter named `turbo`; `names` are the
    adapter names it does have."""
    have = ", ".join(repr(name) for name in names) if names else "none"
    return (
        f"{LEAD_IN_INPUT} switches the LoRA adapter named '{TURBO_ADAPTER}', and "
        f"this step has no adapter by that name (its adapters: {have}) - "
        f"set that LoRA's adapter_name to '{TURBO_ADAPTER}'"
    )


def first_block_problem():
    return (
        f"{LEAD_IN_INPUT} cannot run with a '{FIRST_BLOCK_CACHE}' cache: its "
        f"residual threshold reads the adapter switch as a change and cached "
        f"steps skip it - use cache null, or another cache type"
    )


def _cache_type(configuration):
    cache = configuration.get("cache") if isinstance(configuration, dict) else None
    return cache.get("type") if isinstance(cache, dict) else None


def _adapter_names(loras):
    """The adapter names a step's `loras` load, as `adapters.adapter_settings`
    names them, or None when one is still an unresolved reference."""
    if not isinstance(loras, list):
        return []
    names = []
    for index, lora in enumerate(loras):
        if not isinstance(lora, dict) or lora.get("model_name") is None:
            continue
        name = lora.get("adapter_name")
        if _unresolved(name) or _unresolved(lora.get("model_name")):
            return None
        names.append(name or str(index))
    return names


def lead_in_errors(workflow_definition, source_indices=None):
    """Every `turbo_lead_in` refused before the run, as [{path, message}]:
    a step off MiniMax-H3, a value that is no whole number of at least 0, N
    at or past the call's evaluations, no adapter named `turbo`, or a
    `first_block` cache. An unresolved reference is left to the run."""
    errors = []
    for _, _, pipeline, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict) or LEAD_IN_INPUT not in arguments:
            continue
        value = arguments.get(LEAD_IN_INPUT)
        if value is None or _unresolved(value):
            continue
        path = render_path(("steps", source, "pipeline", "arguments", LEAD_IN_INPUT))
        problem = not_h3(pipeline, LEAD_IN_INPUT) or lead_in_value_problem(value)
        if problem is None and value > 0:
            steps = arguments.get("num_inference_steps")
            if not _unresolved(steps):
                problem = count_problem(value, evaluations(pipeline, arguments))
            if problem is None:
                names = _adapter_names(pipeline.get("loras"))
                if names is not None and TURBO_ADAPTER not in names:
                    problem = no_turbo_problem(names)
            if (
                problem is None
                and _cache_type(pipeline.get("configuration")) == FIRST_BLOCK_CACHE
            ):
                problem = first_block_problem()
        if problem:
            errors.append({"path": path, "message": f"{problem}{where}"})
    return errors


def lead_in_count(value, pipeline, arguments, configuration):
    """The call's lead-in N, or None when it runs without one (absent, null
    or 0). The run's half of `lead_in_errors`.

    Raises:
        ValueError: If the value, the evaluation count or the cache refuse it
    """
    if value is None:
        return None
    problem = lead_in_value_problem(value)
    if problem:
        raise ValueError(problem)
    if value == 0:
        return None
    problem = count_problem(value, evaluations(pipeline, arguments))
    if problem is None and _cache_type(configuration) == FIRST_BLOCK_CACHE:
        problem = first_block_problem()
    if problem:
        raise ValueError(problem)
    return value


# The run


@dataclass
class _LeadInState:
    """One call's count, shared by every H3 transformer it hooks."""

    lead_in: int
    restore: object
    evaluation: int = 0
    last_timestep: float | None = None
    restored: bool = False
    handles: list = field(default_factory=list)

    def observe(self, timestep):
        value = _timestep_value(timestep)
        if value is None or value != self.last_timestep:
            self.evaluation += 1
            self.last_timestep = value
        if not self.restored and self.evaluation > self.lead_in:
            self.restore()
            self.restored = True


def _timestep_value(timestep):
    """One forward's target timestep as a float: the largest entry, as clean
    conditioning rows sit at 0 and the target at the step's noise level."""
    if timestep is None:
        return None
    if hasattr(timestep, "max") and hasattr(timestep, "numel"):
        if timestep.numel() == 0:
            return None
        return float(timestep.max())
    try:
        return float(timestep)
    except (TypeError, ValueError):
        return None


def _make_pre_hook(state):
    def _pre_hook(component, args, kwargs):
        state.observe(forward_arg(args, kwargs, "timestep"))
        return None

    return _pre_hook


def _turbo_layers(pipeline):
    """[(layer, its turbo scaling)] over every peft layer carrying `turbo`,
    and the sorted adapter names the pipeline's layers carry."""
    from .adapters import _lora_layers

    carried, names = [], set()
    for layer in _lora_layers(pipeline):
        scaling = getattr(layer, "scaling", None)
        if not isinstance(scaling, dict):
            continue
        names.update(scaling)
        if TURBO_ADAPTER in scaling:
            carried.append((layer, scaling[TURBO_ADAPTER]))
    return carried, sorted(names)


@contextmanager
def lead_in_call(pipeline, lead_in, total, label):
    """Run the call's first `lead_in` evaluations with adapter `turbo` at
    scale 0 and the rest at its configured scale; put each layer's exact
    `turbo` scaling back and remove the hooks when the call ends, also when
    it raises or is cancelled. `total` is the call's evaluations, for the
    event. A `lead_in` of None or 0 is a no-op.

    Raises:
        ValueError: If the pipeline has no H3 transformer or no layer
            carries an adapter named `turbo`
    """
    if not lead_in:
        yield None
        return
    from .h3_tst import _h3_transformers

    transformers = _h3_transformers(pipeline)
    if not transformers:
        raise ValueError(
            f"{label}: {LEAD_IN_INPUT} found no MiniMax-H3 transformer on this pipeline"
        )
    carried, names = _turbo_layers(pipeline)
    if not carried:
        raise ValueError(f"{label}: {no_turbo_problem(names)}")

    def restore():
        for layer, scaling in carried:
            layer.scaling[TURBO_ADAPTER] = scaling

    state = _LeadInState(lead_in=lead_in, restore=restore)
    message = (
        f"turbo lead-in: evaluations 1..{lead_in} with adapter "
        f"'{TURBO_ADAPTER}' at 0 of {total}"
    )
    logger.info(f"{label}: {message}")
    emit_log(f"{label}: {message}", lead_in=lead_in, evaluations=total)
    try:
        for layer, _ in carried:
            layer.set_scale(TURBO_ADAPTER, 0.0)
        for _, component in transformers:
            state.handles.append(
                component.register_forward_pre_hook(
                    _make_pre_hook(state), with_kwargs=True
                )
            )
        yield state
    finally:
        for handle in state.handles:
            handle.remove()
        restore()
