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
- `cache_unsupported`: a cache on a pipeline whose class declares no
  transformer (a UNet pipeline such as SD 1.5), or whose transformer class has
  no `enable_cache()` (Z-Image's, in the diffusers this server runs). Read from
  the pipeline class's constructor alone, nothing is loaded; the run only says
  "not applied" and carries on uncached.
- `compile_fullgraph_with_cache`: `fullgraph` forbids the graph break a
  cache's skip decision needs (`docs/ACCELERATION.md`).
- `attention_backend_unavailable`: a CUDA-only attention backend on a device
  that isn't CUDA passes validate and fails at load.
- `sparse_attention_fallback`: the `sparse` attention backend on a device that
  isn't CUDA. It never fails a run: it falls back to dense attention, so this
  says the speedup is not coming, not that the job will break.

Every one is a warning and never a refusal. A value that cannot be resolved
before the run (a null variable, an unknown device) leaves its warning
silent rather than guessed at.
"""

import inspect
import typing

from . import get_device_type
from .introspection import load_pipeline_class
from .references import iter_steps, render_path
from . import sparse_attention
from .sparse_attention import SPARSE_BACKEND
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
SPARSE_FALLBACK = "sparse_attention_fallback"
CACHE_UNSUPPORTED = "cache_unsupported"

# The component names a cache hook attaches to, as in
# dw.pipeline_processors.components.get_cache_transformer
_CACHE_COMPONENTS = ("transformer", "transformer_ref")


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


def _cache_unsupported(component_type):
    """Why a cache cannot attach to the pipeline class `component_type`
    names, or None when it can or that cannot be told from the class alone
    (an unresolvable class, a **kwargs constructor, a component whose type is
    not a single class)."""
    if not isinstance(component_type, str):
        return None
    try:
        cls = load_pipeline_class(component_type)
        parameters = inspect.signature(cls.__init__).parameters
    except (ValueError, TypeError):
        return None
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    names = [name for name in _CACHE_COMPONENTS if name in parameters]
    if not names:
        return (
            f"{cls.__name__} has no transformer component to attach a cache to "
            f"(a UNet pipeline is not supported)"
        )
    try:
        hints = typing.get_type_hints(cls.__init__)
    except Exception:
        return None
    classes = [hints.get(name) for name in names]
    if all(isinstance(c, type) and not hasattr(c, "enable_cache") for c in classes):
        return (
            f"{cls.__name__}'s transformer ({classes[0].__name__}) does not "
            f"support diffusers' enable_cache()"
        )
    return None


def _cache_warnings(cache, pipeline, path, where, context, component_type=None):
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
    reason = _cache_unsupported(component_type)
    if reason:
        warnings.append(
            f"{path}: {CACHE_UNSUPPORTED}: the '{cache_type}' cache will not be "
            f"applied: {reason}{where}; the step runs uncached, so drop the cache"
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
                cache,
                pipeline,
                render_path(base + ("cache",)),
                where,
                context,
                configuration.get("component_type"),
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
            if name == SPARSE_BACKEND:
                warnings.append(
                    f"{render_path(path)}: {SPARSE_FALLBACK}: sparse attention "
                    f"needs CUDA; this server is '{context.device_type}', so "
                    f"the run uses dense attention{where}"
                )
                continue
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


def available_attention_backends(device_type: str) -> list[str]:
    """The attention backends a server on `device_type` can run, in menu order.

    `native` first, then the other `native`-family names; on CUDA, `sparse`
    (only where the compiled kernels loaded) and the CUDA-only names. A name
    diffusers' `AttentionBackendName` lacks is dropped, so the menu never
    offers one the pipeline would refuse.
    """
    from diffusers.models.attention_dispatch import AttentionBackendName

    known = {backend.value for backend in AttentionBackendName}
    cuda = device_type == "cuda"
    names = ["native", "_native_flash", "_native_efficient", "_native_math"]
    if cuda:
        names.append("_native_cudnn")
    names.append("flex")
    if cuda:
        if sparse_attention._kernels_importable():
            names.append(SPARSE_BACKEND)
        names.extend(sorted(CUDA_ONLY_ATTENTION_BACKENDS))
    return [name for name in names if name == SPARSE_BACKEND or name in known]
