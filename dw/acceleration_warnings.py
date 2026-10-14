"""Acceleration settings that validate passes and that cannot pay off (#868).

Each of these cost a render to find out in Don's `h3-quality` matrix:

- `cache_short_schedule`: a threshold cache (`first_block`, `mag`) on a
  schedule under `CACHE_MIN_USEFUL_STEPS`. Consecutive steps of a short
  schedule differ too much for the residual threshold, so the cache skips
  nothing and still perturbs the result - on MiniMax-H3 a `first_block` cache
  at 0.05 skipped no step at 12 steps and took 22% off at 30. `docs/RECIPES_24GB.md` states the
  rule and `tests/test_catalog_speed_defaults.py` holds the catalog to it;
  this is where a hand-written workflow meets it. The interval caches
  (`taylorseer`, `faster`) skip on a fixed interval and `text_kv` skips no
  step at all, so none of them is held to it.
- `taylorseer_state_unestimated`: TaylorSeer keeps per-block state on the
  device that no VRAM estimate charges - P5 ran out of memory two minutes
  into a run validate had passed. Always said when TaylorSeer is set. Sizing
  that state would need the architecture data the engine never holds, and
  charging it in `vram_estimate` would turn a warning into a refusal and move
  jobs between cards, so the warning only states the estimate it sits on top
  of.
- `compile_fullgraph_with_cache`: `fullgraph` forbids the graph break a
  cache's skip decision needs (`docs/ACCELERATION.md`).
- `attention_backend_unavailable`: a CUDA-only attention backend on a device
  that isn't CUDA passes validate and fails at load.

Every one is a warning and never a refusal. A value that cannot be resolved
before the run (a null variable, an unknown device) leaves its warning
silent rather than guessed at.
"""

from . import get_device_type
from .references import iter_steps, render_path
from .vram_estimate import required_vram_gb

# The fewest denoise steps a threshold cache is expected to skip any of. A
# general fact about residual-threshold caching, not a per-model rule: H3
# skipped nothing at 12 steps and 22% at 30 (RECIPES_24GB.md); nothing
# between has been measured yet (#875 measures 20)
CACHE_MIN_USEFUL_STEPS = 20

# The cache types that decide each step's skip on a residual threshold
THRESHOLD_CACHES = ("first_block", "mag")

# The attention backends diffusers registers that need CUDA: the flash,
# flash-3, sage and xformers families, Hub kernels included. Matched by name,
# never by a `_hub` suffix; no `native`-family backend is here. A name
# missing from this set only means no warning - today's behavior
CUDA_ONLY_ATTENTION_BACKENDS = frozenset(
    {
        "flash",
        "flash_hub",
        "flash_varlen",
        "flash_varlen_hub",
        "flash_4_hub",
        "_flash_3",
        "_flash_varlen_3",
        "_flash_3_hub",
        "_flash_3_varlen_hub",
        "aiter_fa2_hub",
        "sage",
        "sage_hub",
        "sage_blackwell_hub",
        "sage_varlen",
        "_sage_qk_int8_pv_fp8_cuda",
        "_sage_qk_int8_pv_fp8_cuda_sm90",
        "_sage_qk_int8_pv_fp16_cuda",
        "_sage_qk_int8_pv_fp16_triton",
        "xformers",
    }
)

SHORT_SCHEDULE = "cache_short_schedule"
TAYLORSEER_STATE = "taylorseer_state_unestimated"
FULLGRAPH_WITH_CACHE = "compile_fullgraph_with_cache"
BACKEND_UNAVAILABLE = "attention_backend_unavailable"


def _steps(pipeline):
    """The resolved `num_inference_steps` the pipeline is called with, or
    None when it is not a number known before the run."""
    arguments = pipeline.get("arguments")
    value = (arguments if isinstance(arguments, dict) else {}).get(
        "num_inference_steps"
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _backend(device_type):
    """The backend `device_type` belongs to ('cuda' for 'cuda:1'), or None
    when the device is unknown."""
    if not isinstance(device_type, str) or not device_type:
        return None
    try:
        return get_device_type(device_type)
    except (RuntimeError, ValueError):
        return None


def _components(configuration):
    components = configuration.get("components")
    if not isinstance(components, dict):
        return []
    return [
        (name, component)
        for name, component in components.items()
        if isinstance(component, dict)
    ]


def _estimate_clause(context):
    """'; this job is estimated at X of Y GB' when the workflow declares a
    VRAM estimate and the device's capacity is known, else ''."""
    if context.capacity_gb is None:
        return ""
    required = required_vram_gb(
        context.expanded,
        context.arguments,
        device_type=context.device_type,
        base_dir=context.base_dir,
        probe=context.probe,
    )
    if required is None or not required[1]:
        # Only a declared vram_estimate is an estimate; a measured `cost`
        # entry records a card, not what this job needs
        return ""
    return (
        f"; this job is estimated at {required[0]:.1f} of "
        f"{context.capacity_gb:.1f} GB, and that state comes on top"
    )


def _cache_warnings(cache, pipeline, path, where, context):
    cache_type = cache.get("type")
    warnings = []
    steps = _steps(pipeline)
    if (
        cache_type in THRESHOLD_CACHES
        and steps is not None
        and steps < CACHE_MIN_USEFUL_STEPS
    ):
        warnings.append(
            f"{path}: {SHORT_SCHEDULE}: a '{cache_type}' cache on {steps:g} "
            f"steps will likely skip nothing (consecutive steps differ too much "
            f"for its threshold) and still perturbs the result{where}; drop the "
            f"cache or run >= {CACHE_MIN_USEFUL_STEPS} steps"
        )
    if cache_type == "taylorseer":
        max_order = cache.get("max_order")
        if isinstance(max_order, bool) or not isinstance(max_order, int):
            max_order = 1
        warnings.append(
            f"{path}: {TAYLORSEER_STATE}: TaylorSeer keeps per-block state on "
            f"the GPU (more with a higher max_order; this one is {max_order}) "
            f"that the VRAM estimate doesn't include{_estimate_clause(context)}"
            f"{where}. A run that fits without the cache can run out of memory "
            f"with it"
        )
    return warnings


def acceleration_warnings(context):
    """Every acceleration setting in the expanded definition that cannot pay
    off, as `"<path>: <code>: <message>"` strings, each said once."""
    warnings = []
    backend = _backend(context.device_type)
    for _, _, pipeline, source, where in iter_steps(
        context.expanded.get("steps"), context.source_indices, "pipeline"
    ):
        configuration = pipeline.get("configuration")
        if not isinstance(configuration, dict):
            continue
        base = ("steps", source, "pipeline", "configuration")
        cache = configuration.get("cache")
        if isinstance(cache, dict):
            warnings += _cache_warnings(
                cache, pipeline, render_path(base + ("cache",)), where, context
            )

        backends = [(base + ("attention_backend",), configuration)]
        for name, component in _components(configuration):
            component_path = base + ("components", name)
            compile_config = component.get("compile")
            if (
                isinstance(cache, dict)
                and isinstance(compile_config, dict)
                and compile_config.get("fullgraph") is True
            ):
                warnings.append(
                    f"{render_path(component_path + ('compile', 'fullgraph'))}: "
                    f"{FULLGRAPH_WITH_CACHE}: fullgraph forbids the graph break "
                    f"the '{cache.get('type')}' cache's skip decision needs"
                    f"{where}; use partial graph mode (fullgraph false) with a "
                    f"cache"
                )
            backends.append((component_path + ("attention_backend",), component))

        if backend is None or backend == "cuda":
            continue
        for path, block in backends:
            name = block.get("attention_backend")
            if name not in CUDA_ONLY_ATTENTION_BACKENDS:
                continue
            warnings.append(
                f"{render_path(path)}: {BACKEND_UNAVAILABLE}: attention backend "
                f"'{name}' needs CUDA; this server is '{context.device_type}', "
                f"so the job will fail at load{where}; leave it unset or use a "
                f"'native'-family backend"
            )
    # A for_each step's members share their written path: say each code once
    # there, naming the first member it fired for
    said = {}
    for warning in warnings:
        path, _, rest = warning.partition(": ")
        said.setdefault((path, rest.partition(": ")[0]), warning)
    return list(said.values())
