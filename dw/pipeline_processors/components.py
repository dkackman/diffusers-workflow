import torch
import contextlib
import logging
from .config_objects import (
    get_group_offload_configuration,
    get_load_components_arguments,
)
from ..cache_blocks import register_cache_blocks
from ..type_helpers import has_method
from ..trust import (
    require_trusted_from_pretrained_arguments,
)
from .. import get_device_type, resolve_device
from ..events import WorkflowCancelled, emit_phase
from .. import download_watch
from huggingface_hub.errors import HfHubHTTPError
from .h3_guides import insert_guides
from .h3_hold import insert_audio_hold
from .placement import (
    apply_on_demand_placement,
    create_components_manager,
    loading_device,
    place_component,
)

logger = logging.getLogger("dw")


# Names the cache state a run accumulates. Any stable string works - it only has to
# match itself across the steps of one call
_CACHE_CONTEXT_NAME = "dw"


def configure_components(pipeline, configuration, default_device, reused_components=()):
    """Place the components a pipeline loaded for itself.

    A modular pipeline pulls its own component weights, so they are only reachable once
    the pipeline is loaded - too late for the offloading load_component sets up. Group
    offloading a component here streams it between system memory and the accelerator a
    piece at a time, which is what fits a pipeline whose components are each larger than
    the device.

    A component this step reused is not one it loaded: it already carries the placement
    the step that shared it gave it, and offloading hooks do not survive being applied
    twice. Those are skipped, so a workflow can reuse a component into a step whose
    configuration was written for loading it.

    Args:
        pipeline: The loaded pipeline
        configuration: Pipeline configuration dictionary
        default_device: Device the pipeline runs on
        reused_components: Names of the components an earlier step shared into this one
    """
    for component_name, component_configuration in configuration.get(
        "components", {}
    ).items():
        # A dotted path reaches inside a component, and it is the component itself
        # that was shared - 'text_encoder.model' belongs to a reused 'text_encoder'
        if component_name.split(".")[0] in reused_components:
            logger.info(
                f"Component '{component_name}' was shared by an earlier step - "
                "keeping the placement that step gave it"
            )
            continue

        component = get_component(pipeline, component_name)
        if component is None:
            # Registered but unloaded (e.g. a components map reused across workflow
            # selections, or a component diffusers warned-and-skipped past at load) -
            # skip just this entry rather than aborting the whole run
            logger.warning(
                f"Component '{component_name}' is not loaded (workflow selection "
                "may not use it) - skipping its configuration"
            )
            continue

        # Prune before any hooks are installed - an offload hook pins and tracks
        # exactly the modules that exist when it is applied, so pruning afterwards
        # would leave it streaming weights that can never run
        truncate_module_lists(component, component_name, component_configuration)
        replace_modules_with_identity(
            component, component_name, component_configuration
        )

        group_offload_configuration = get_group_offload_configuration(
            component_configuration, default_device
        )
        if group_offload_configuration is not None:
            # apply_group_offloading rather than the component's own
            # enable_group_offload - a component may be a transformers model, or a
            # module inside one, and only diffusers models have the method
            from diffusers.hooks import apply_group_offloading

            logger.info(f"Group offloading {component_name}")
            apply_group_offloading(component, **group_offload_configuration)

        # Tiled decoding, for a component that decodes but is not the one called
        # 'vae' - LTX-2.5's diffusion decoder, which decodes the whole video volume
        # in one allocation unless it is told to tile
        enable_tiling(component, component_name, component_configuration)

        # The attention processor this component runs, for a component that is
        # neither the unet nor the transformer (both covered by their own
        # pipeline-level blocks) - LTX-2.5's diffusion decoder, whose default
        # processor is a portable fallback rather than the path it was trained to run
        set_attn_processor(component, component_name, component_configuration)

        device = resolve_device(component_configuration.get("device", None))
        residency = component_configuration.get("residency", "resident")
        if residency == "on_demand":
            apply_on_demand_placement(
                component,
                component_name,
                device if device is not None else default_device,
                group_offload_configuration is not None,
            )
        elif device is not None:
            logger.info(f"Moving {component_name} to device: {device}")
            component.to(device)

        # A compiled component should pin its attention backend - the per-call
        # attention_backend context manager would switch implementations under a
        # compiled graph and force a recompile on every run
        component_attention_backend = component_configuration.get(
            "attention_backend", None
        )
        if component_attention_backend is not None:
            logger.info(
                f"Setting {component_name} attention backend: {component_attention_backend}"
            )
            component.set_attention_backend(component_attention_backend)

        # Compile last - the graph must capture final dtypes, adapters,
        # quantization, and offload hooks
        compile_configuration = component_configuration.get("compile", None)
        if compile_configuration is not None:
            apply_compile(
                component,
                component_name,
                compile_configuration,
                device if device is not None else default_device,
            )


def enable_tiling(component, component_name, component_configuration):
    """Turn on tiled decoding for a component whose configuration asks for it.

    The pipeline-level `vae` block covers the component actually named 'vae'. This
    covers any other component that decodes - LTX-2.5's `diffusion_decoder`, which
    otherwise decodes the whole video volume in a single allocation and asks for
    tens of GiB at 2x resolutions. `true` takes the model's own default tile size;
    a dict passes the tile and stride sizes through, which is what a card smaller
    than those defaults needs.

    Args:
        component: The loaded component
        component_name: Its name, for logging and errors
        component_configuration: That component's configuration block

    Raises:
        ValueError: If the component has no enable_tiling() to call
    """
    tiling = component_configuration.get("enable_tiling", False)
    if not tiling:
        return

    if not has_method(component, "enable_tiling"):
        raise ValueError(
            f"'{component_name}' does not support tiling - "
            f"{type(component).__name__} has no enable_tiling()"
        )

    arguments = tiling if isinstance(tiling, dict) else {}
    logger.info(
        f"Enabling tiling on {component_name}"
        + (f" with {', '.join(arguments)}" if arguments else "")
    )
    component.enable_tiling(**arguments)


def set_attn_processor(component, component_name, component_configuration):
    """Swap a component's attention processor for the one its configuration names.

    The pipeline-level `unet` and `transformer` blocks cover those two components.
    This covers any other one that carries attention - LTX-2.5's
    `diffusion_decoder`, whose default `LTX2VideoVaeNeighborhoodAttnProcessor` is a
    portable FlexAttention fallback rather than the NATTEN path the decoder was
    built around, and which diffusers' own docstring calls larger than device memory
    at production grids.

    The value is a type, resolved by the `_type` suffix convention, and is
    constructed with no arguments - the same shape the `unet`/`transformer` blocks
    have used since they were written.

    Args:
        component: The loaded component
        component_name: Its name, for logging and errors
        component_configuration: That component's configuration block

    Raises:
        ValueError: If the component has no set_attn_processor() to call
    """
    attn_processor_type = component_configuration.get("attn_processor_type", None)
    if attn_processor_type is None:
        return

    if not has_method(component, "set_attn_processor"):
        raise ValueError(
            f"'{component_name}' does not take an attention processor - "
            f"{type(component).__name__} has no set_attn_processor()"
        )

    logger.info(
        f"Setting {component_name} attention processor: {attn_processor_type.__name__}"
    )
    component.set_attn_processor(attn_processor_type())


def _resolve_submodule(component, component_name, path):
    """Follow a dotted path from a component to a module inside it.

    Args:
        component: The component the path starts from
        component_name: Its name, for errors
        path: Dotted attribute path relative to the component, e.g.
            'language_model.layers'

    Returns:
        (parent, attribute_name, module) - the module and where it hangs

    Raises:
        ValueError: If any step of the path is not an attribute
    """
    parent = None
    module = component
    for attribute_name in path.split("."):
        parent = module
        module = getattr(parent, attribute_name, _MISSING)
        if module is _MISSING:
            raise ValueError(
                f"'{component_name}' has no module at '{path}' - "
                f"{type(parent).__name__} has no attribute '{attribute_name}'"
            )
    return parent, path.rsplit(".", 1)[-1], module


def truncate_module_lists(component, component_name, component_configuration):
    """Drop the tail of a ModuleList a run never reads.

    An encoder used for its hidden states can run layers whose output nothing
    consumes: MiniMax-H3 conditions on hidden_states[50] of its 64-layer
    Qwen3-VL, so layers 51-63 compute - and, offloaded, stream from system
    memory - for nothing, on every encode. Keeping 51 layers leaves
    hidden_states[50] bit-identical (index 50 of the returned tuple is the
    input to layer 50, recorded before it runs; keeping only 50 would make it
    the final-norm output instead, which is a different tensor).

    The configuration maps a dotted path inside the component to the number of
    entries to keep:

        "truncate_layers": { "language_model.layers": 51 }

    Truncation is in place, so the component's registration on its pipeline and
    its config are untouched - a block that validates against
    config.num_hidden_layers still sees the checkpoint's own count.

    Args:
        component: The loaded component
        component_name: Its name, for logging and errors
        component_configuration: That component's configuration block

    Raises:
        ValueError: If a path does not lead to a ModuleList, or keep is not
            a positive count
    """
    truncations = component_configuration.get("truncate_layers", None)
    if not truncations:
        return

    for path, keep in truncations.items():
        _, _, module_list = _resolve_submodule(component, component_name, path)
        if not isinstance(module_list, torch.nn.ModuleList):
            raise ValueError(
                f"'{component_name}' cannot truncate '{path}' - it is a "
                f"{type(module_list).__name__}, not a ModuleList"
            )
        keep = int(keep)
        if keep < 1:
            raise ValueError(
                f"'{component_name}' truncate_layers keeps {keep} of '{path}' - "
                "at least one layer has to remain"
            )
        if keep >= len(module_list):
            logger.warning(
                f"'{component_name}' truncate_layers keeps {keep} of '{path}', "
                f"which already has {len(module_list)} - nothing to drop"
            )
            continue
        logger.info(
            f"Truncating {component_name} '{path}' from {len(module_list)} "
            f"layers to {keep}"
        )
        del module_list[keep:]


def replace_modules_with_identity(component, component_name, component_configuration):
    """Swap out modules a run never calls, freeing what they hold.

    For weights that exist on the checkpoint but are outside the path the
    pipeline actually runs - a language-model head on a model used as an
    encoder, say. The module is replaced with an Identity so the model keeps
    its shape for anything that looks the attribute up, while its parameters
    are dropped rather than held (and, offloaded, pinned) for a call that
    never comes.

        "remove_modules": [ "lm_head" ]

    Args:
        component: The loaded component
        component_name: Its name, for logging and errors
        component_configuration: That component's configuration block

    Raises:
        ValueError: If a named module does not exist on the component
    """
    for path in component_configuration.get("remove_modules", []):
        parent, attribute_name, _ = _resolve_submodule(component, component_name, path)
        logger.info(f"Replacing {component_name} '{path}' with Identity")
        setattr(parent, attribute_name, torch.nn.Identity())


def apply_compile(component, component_name, compile_configuration, device):
    """Compile a component with torch.compile.

    Compilation happens in place (nn.Module.compile) so the module stays registered
    on its pipeline. With 'repeated_blocks' true, only the model's repeated block
    classes are compiled (diffusers' regional compilation) - near the same speedup
    as full compilation with a fraction of the cold-start cost.

    Args:
        component: The component to compile
        component_name: Name of the component, for logging
        compile_configuration: Dict of options - 'repeated_blocks' selects regional
            compilation, everything else ('mode', 'fullgraph', 'dynamic', ...) is
            passed to torch.compile
        device: Device the component runs on
    """
    # Inductor support on MPS is too immature to be worth the compile time
    if get_device_type(device) == "mps":
        logger.warning(
            f"torch.compile is not supported on MPS, skipping {component_name}"
        )
        return

    options = dict(compile_configuration)
    repeated_blocks = options.pop("repeated_blocks", False)

    if repeated_blocks:
        if not has_method(component, "compile_repeated_blocks"):
            raise ValueError(
                f"repeated_blocks compilation requires a diffusers model with "
                f"repeated block support, {type(component).__name__} does not have it"
            )
        logger.info(f"Compiling repeated blocks of {component_name}")
        component.compile_repeated_blocks(**options)
    else:
        logger.info(f"Compiling {component_name}")
        component.compile(**options)


_MISSING = object()


def get_component(pipeline, component_name):
    """Look a component up on a pipeline, by name or by a dotted path into it.

    A dotted path reaches a module inside a component, which is how a component that
    holds the model rather than being one - a transformers model wrapping its own - is
    offloaded.

    A modular pipeline registers a component it has not loaded (a workflow selection
    that does not use it, or one diffusers warned-and-skipped past) as a None-valued
    attribute rather than omitting it entirely - that is a real attribute, not a typo,
    so it is returned as None rather than raising. A missing attribute is still a hard
    error: it means the name itself is wrong. Callers decide what "unloaded" should
    mean for them (skip with a warning, skip silently, ...); this just tells them apart.

    Args:
        pipeline: The loaded pipeline
        component_name: Name of the component, e.g. 'vae' or 'text_encoder.model'

    Returns:
        The named component, or None if it (or a step along a dotted path) is
        registered but not loaded

    Raises:
        ValueError: If the pipeline has no attribute by that name (or dotted path)
    """
    component = pipeline
    for attribute_name in component_name.split("."):
        component = getattr(component, attribute_name, _MISSING)
        if component is _MISSING:
            raise ValueError(
                f"{type(pipeline).__name__} has no component '{component_name}'"
            )
        if component is None:
            return None

    return component


def load_and_configure_scheduler(
    scheduler_definition, pipeline, component_name="scheduler"
):
    """Load and configure a pipeline's scheduler if specified.

    A definition does either or both of two things, in that order: replace the
    scheduler with one built from another type's config, and set the sigma
    shift on whatever scheduler the pipeline then holds.

    The component is named rather than assumed because a pipeline can carry
    more than one. MiniMax-H3 steps video and audio latents down two schedules
    inside a single transformer call - 'scheduler' and 'audio_scheduler', whose
    shifts (12.0 and 3.0 in the released checkpoint) are set independently, and
    the video one is what a few-step schedule has to lower: at the checkpoint's
    12.0 a five-point sigma grid spends every step above 0.8 and then drops to
    zero in one, which denoises to noise.

    Args:
        scheduler_definition: The step's scheduler block, or None
        pipeline: The loaded pipeline
        component_name: Which scheduler the definition configures
    """
    if scheduler_definition is None:
        return

    scheduler_configuration = scheduler_definition.get("configuration", None) or {}
    scheduler_type = scheduler_configuration.get("scheduler_type", None)
    if scheduler_type is not None:
        from_config_args = scheduler_definition.get("from_config_args", {})
        logger.info(f"Loading {component_name}: {scheduler_type}")
        setattr(
            pipeline,
            component_name,
            scheduler_type.from_config(
                get_component(pipeline, component_name).config, **from_config_args
            ),
        )

    shift = scheduler_definition.get("shift", None)
    if shift is None:
        return

    scheduler = get_component(pipeline, component_name)
    if scheduler is None:
        raise ValueError(
            f"Cannot set a shift on '{component_name}' - the pipeline registers "
            "it but has not loaded it"
        )
    if not has_method(scheduler, "set_shift"):
        raise ValueError(
            f"{type(scheduler).__name__} does not take a sigma shift - "
            f"'{component_name}' has no set_shift()"
        )

    # Instance state the scheduler keeps until its next set_timesteps, which is
    # the run itself - so this survives loading and every later run of the step
    logger.info(f"Setting {component_name} shift: {shift}")
    scheduler.set_shift(float(shift))


def get_block_configs(configuration, component):
    """The block configs a workflow sets on a modular pipeline, checked against it.

    A modular pipeline's blocks declare configs of their own - values they read while
    they run rather than components or call arguments. MiniMax-H3 declares three, and
    they are how the canvas the request generates on and the resolution its references
    are encoded at are set:

        "configs": { "canvas_short_edge": 768, "reference_image_short_edge": 1024 }

    This is deliberately not a knob per config per model. Every modular pipeline
    declares its own set, `update_components()` sets any of them, and what a workflow
    may say here is whatever the pipeline it named declares. The names are checked
    because update_components ignores the ones it does not know with a warning, and a
    silently dropped config reads as a setting that did nothing.

    Args:
        configuration: Pipeline configuration dictionary
        component: The loaded pipeline the configs are for

    Returns:
        Dict of config name to value, empty when the workflow sets none

    Raises:
        ValueError: If the pipeline takes no configs, or does not declare one by name
    """
    configs = configuration.get("configs", None)
    if not configs:
        return {}

    if not has_method(component, "update_components"):
        raise ValueError(
            f"'configs' is only supported on modular pipelines, "
            f"{type(component).__name__} does not have update_components"
        )

    # The specs a pipeline builds from its blocks. Guarded rather than indexed - a
    # pipeline that stops keeping them under this name should lose the check, not
    # the feature
    declared = getattr(component, "_config_specs", None)
    if declared is not None:
        unknown = [name for name in configs if name not in declared]
        if unknown:
            raise ValueError(
                f"{type(component).__name__} declares no config named "
                f"{', '.join(sorted(unknown))} - the ones it declares are "
                f"{', '.join(sorted(declared)) or 'none'}"
            )

    logger.info(f"Setting block configs: {', '.join(configs)}")
    return dict(configs)


def load_component(
    component_name,
    configuration,
    from_pretrained_arguments,
    device,
    reused_components=None,
    defer_placement=False,
):
    """Load and configure a pipeline or component.

    Args:
        component_name: What is being loaded, for the log
        configuration: The component's configuration block
        from_pretrained_arguments: Arguments for the constructor
        device: Device the component is loaded for
        reused_components: Components an earlier step shared into this one, by name
        defer_placement: Load the component without placing it - the caller calls
            place_component once it has finished altering the weights
    """
    component_type = configuration["component_type"]
    component = None

    # Refused before anything reaches the Hub: an untrusted workflow must not
    # be able to have diffusers fetch and run code on its behalf
    require_trusted_from_pretrained_arguments(from_pretrained_arguments, component_name)

    # A standard pipeline takes a component as a constructor argument. A modular one
    # cannot: it is built from the component specs in its own index and given the
    # objects afterwards, which is also what keeps load_components() from pulling a
    # second copy of the weights - it skips the components already registered
    reused_components = reused_components or {}
    takes_components_after_load = has_method(component_type, "update_components")
    if reused_components and not takes_components_after_load:
        from_pretrained_arguments.update(reused_components)

    # A modular pipeline can hand its components to a ComponentsManager, which then
    # owns their device placement
    components_manager = create_components_manager(configuration, device)
    if components_manager is not None:
        from_pretrained_arguments["components_manager"] = components_manager

    _warn_float16_on_mps(component_name, from_pretrained_arguments, device)

    model_name = None
    try:
        with loading_device(configuration):
            # Load from model name
            if "model_name" in from_pretrained_arguments:
                model_name = from_pretrained_arguments.pop("model_name")
                logger.info(f"Loading {component_name} from model: {model_name}")
                emit_phase("loading", detail=f"{component_name}: {model_name}")
                with download_watch.watch(
                    model_name, cache_dir=from_pretrained_arguments.get("cache_dir")
                ):
                    component = component_type.from_pretrained(
                        model_name, **from_pretrained_arguments
                    )

            # Load from single file
            elif "from_single_file" in from_pretrained_arguments:
                from_single_file = from_pretrained_arguments.pop("from_single_file")
                logger.info(
                    f"Loading {component_name} from single file: {from_single_file}"
                )
                emit_phase("loading", detail=f"{component_name}: {from_single_file}")
                component = component_type.from_single_file(
                    from_single_file, **from_pretrained_arguments
                )

            # Create new component
            else:
                logger.info(f"Creating new {component_name}")
                component = component_type(**from_pretrained_arguments)

            # Register the shared components before anything is pulled, so the
            # weights an earlier step already loaded and quantized are the ones
            # this step runs on rather than a second copy of them. The block
            # configs go in the same call - update_components takes both
            update_arguments = get_block_configs(configuration, component)
            if reused_components and takes_components_after_load:
                logger.info(
                    f"Reusing {', '.join(reused_components)} from an earlier step"
                )
                update_arguments.update(reused_components)
            if update_arguments:
                component.update_components(**update_arguments)

            # Modular pipelines load only their config in from_pretrained - the component
            # weights are pulled separately by load_components()
            load_components_arguments = get_load_components_arguments(configuration)
            if load_components_arguments is not None:
                if not has_method(component, "load_components"):
                    raise ValueError(
                        f"load_components is only supported on modular pipelines, "
                        f"{component_type.__name__} does not have it"
                    )
                logger.info(f"Loading components for {component_name}")
                component.load_components(**load_components_arguments)

            # MiniMax-H3 takes `hold_audio` on every core-denoise workflow; a
            # no-op on any other pipeline (dw/pipeline_processors/h3_hold.py)
            insert_audio_hold(component)
            # ...and `guides` on t2va and fl2va
            insert_guides(component)

        if defer_placement:
            # The caller places this itself, once it has finished loading the
            # things that alter the weights - see place_component
            logger.debug(f"Deferring placement of {component_name}")
            return component

        return place_component(
            component, component_name, configuration, device, components_manager
        )

    except WorkflowCancelled:
        # A cancel that aborted a download (dw/download_watch.py) - not a
        # load failure, so no error log
        raise
    except Exception as e:
        # Every error but a Hub auth refusal (a real outage, a bad repo id) is
        # logged once with its traceback and re-raised unchanged
        auth_error = _hub_auth_error(e, model_name, component_name)
        if auth_error is not None:
            raise auth_error from e
        logger.error(f"{type(e).__name__} loading {component_name}: {e}", exc_info=True)
        raise


def _warn_float16_on_mps(component_name, from_pretrained_arguments, device):
    """MPS (Apple Silicon) has numerical instability with float16 matmul operations,
    producing NaN values that result in black images. The dtype is left as asked for -
    silently loading a model in a dtype the workflow did not request would be worse -
    so this only warns."""
    if (
        get_device_type(device) == "mps"
        and from_pretrained_arguments.get("torch_dtype") == torch.float16
    ):
        logger.warning(
            f"{component_name} loads in float16 on MPS, which can produce NaN "
            "values (black images) on Apple Silicon - bfloat16 is the usual fix"
        )


def _hub_auth_error(error, model_name, component_name):
    """The RuntimeError to raise for a Hub auth refusal, or None for any other
    load failure. Called from load_component's except block, so the log line
    carries the traceback.

    401/403 from the Hub means the account behind whatever token (or lack of
    one) HfApi is using cannot read this repo - almost always a gated model
    the user has not requested access to, or has not logged in for. A
    whole-pipeline load raises the HfHubHTTPError itself; a per-component load
    gets it wrapped in an EnvironmentError by diffusers' _get_model_file /
    transformers' cached_file, so the cause chain is searched."""
    if _hub_auth_status(error) is None:
        return None
    repo = model_name or component_name
    logger.error(
        f"Hugging Face authentication required loading {component_name} "
        f"({repo}): {error}",
        exc_info=True,
    )
    return RuntimeError(
        f"Model '{repo}' requires Hugging Face authentication - run "
        f"'huggingface-cli login', or request access at "
        f"https://huggingface.co/{repo}"
    )


def _hub_auth_status(error):
    """The 401/403 status an exception (or anything in its cause/context
    chain) carries from the Hub, else None."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, HfHubHTTPError):
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status in (401, 403):
                return status
        error = error.__cause__ or error.__context__
    return None


def apply_sdnq_optimizations(pipeline, component_names):
    """Apply SDNQ quantized matmul optimization to pipeline components.

    Uses sdnq's apply_sdnq_options_to_model to enable INT8 matmul
    on supported hardware (CUDA, XPU).

    Args:
        pipeline: The loaded diffusers pipeline
        component_names: List of component names to optimize (e.g., ["transformer", "text_encoder"])
    """
    try:
        from sdnq.loader import apply_sdnq_options_to_model
        from sdnq.common import use_torch_compile as triton_is_available
    except ImportError:
        logger.warning("sdnq not installed, skipping SDNQ optimizations")
        return

    if not triton_is_available:
        logger.info("Triton not available, skipping SDNQ quantized matmul optimization")
        return

    if not (
        torch.cuda.is_available() or hasattr(torch, "xpu") and torch.xpu.is_available()
    ):
        logger.info(
            "SDNQ quantized matmul requires CUDA or XPU, skipping on this device"
        )
        return

    for name in component_names:
        # A missing name (typo) and a registered-but-unloaded one both mean "nothing
        # to optimize here" for this call - same warn-and-skip either way
        try:
            component = get_component(pipeline, name)
        except ValueError:
            component = None

        if component is not None:
            logger.info(f"Applying SDNQ quantized matmul to {name}")
            setattr(
                pipeline,
                name,
                apply_sdnq_options_to_model(component, use_quantized_matmul=True),
            )
        else:
            logger.warning(
                f"Component '{name}' not found on pipeline, skipping SDNQ optimization"
            )


def get_cache_transformer(pipeline):
    """Find the denoiser a cache hook attaches to.

    Most pipelines register theirs as 'transformer', but a modular pipeline names
    it after the workflow it serves - MiniMax-H3's ref2va denoises through
    'transformer_ref'. Looking only for 'transformer' silently skips caching on
    those, so try the alternates diffusers' modular pipelines actually use.

    Args:
        pipeline: The loaded diffusers pipeline

    Returns:
        The transformer component, or None when the pipeline has none
    """
    for name in ("transformer", "transformer_ref"):
        transformer = getattr(pipeline, name, None)
        if transformer is not None:
            return transformer
    return None


@contextlib.contextmanager
def stateful_cache_context(pipeline):
    """Provide the context a stateful cache hook reads its state through.

    first_block, mag and layer_skip keep per-context state, and their hooks go
    through diffusers' StateManager, which raises "No context is set" unless a
    context is active. A DiffusionPipeline sets one around each denoising step and
    clears the state afterwards in maybe_free_model_hooks; ModularPipeline is not a
    DiffusionPipeline and does neither, so caching a modular pipeline dies on the
    first step - and would otherwise carry the previous run's residuals into the
    next run of a pipeline this process keeps loaded.

    One context spans the whole call rather than each step. The state is keyed by
    context name, so re-entering per step only re-reads the same entry. Pipelines
    that run separate conditional and unconditional passes name a context per pass
    to keep their caches apart, which a shared context would defeat - but a modular
    pipeline that needed that would be setting its own contexts already, and this
    is a no-op for pipelines whose cache is not enabled.
    """
    transformer = get_cache_transformer(pipeline)
    if transformer is None or not getattr(transformer, "is_cache_enabled", False):
        yield
        return

    logger.debug(f"Entering cache context for {transformer.__class__.__name__}")
    try:
        with transformer.cache_context(_CACHE_CONTEXT_NAME):
            yield
    finally:
        # Private, but it is what diffusers' own pipelines call and there is no
        # public equivalent. Also clears the context an errored call left set
        transformer._reset_stateful_cache()


def enable_cache_on_transformer(pipeline, cache_config):
    """Enable cache configuration on the pipeline's transformer.

    Args:
        pipeline: The loaded diffusers pipeline
        cache_config: Cache configuration object from get_cache_configuration()
    """
    transformer = get_cache_transformer(pipeline)
    if transformer is None:
        logger.warning("Pipeline has no transformer, skipping cache configuration")
        return

    if not hasattr(transformer, "enable_cache"):
        logger.warning(
            f"{transformer.__class__.__name__} does not support enable_cache(), skipping"
        )
        return

    # FasterCache decides skipping from the pipeline's current timestep. The
    # callback is a callable, which workflow JSON cannot express, and diffusers
    # calls it unconditionally on every denoiser forward - left None, the first
    # inference step dies. Wire it to the pipeline here, where both exist
    if (
        cache_config.__class__.__name__ == "FasterCacheConfig"
        and getattr(cache_config, "current_timestep_callback", None) is None
    ):
        logger.debug("Wiring FasterCache current_timestep_callback to the pipeline")
        cache_config.current_timestep_callback = lambda: pipeline._current_timestep

    # first_block, mag and layer_skip resolve the transformer's block class
    # through diffusers' registry and raise when it is absent - fill in the
    # blocks diffusers has not registered before handing the config over
    register_cache_blocks()

    logger.info(
        f"Enabling {cache_config.__class__.__name__} on {transformer.__class__.__name__}"
    )
    transformer.enable_cache(cache_config)
