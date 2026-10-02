import torch
import contextlib
import functools
import logging
from .config_objects import (
    get_group_offload_configuration,
)
from .. import empty_device_cache, get_device_type, resolve_device

logger = logging.getLogger("dw")


# The calls that mean "this component is working now". A component is moved to
# the accelerator around whichever of these it actually defines
_ON_DEMAND_ENTRY_POINTS = ("forward", "encode", "decode")


def apply_on_demand_placement(
    component, component_name, device, group_offloaded, offload_device="cpu"
):
    """Keep a component in system memory and move it to the device only while it runs.

    Sits between the two placements dw already has. A 'device' component is resident
    for the whole run, which wastes the accelerator on something used twice; group
    offloading streams per submodule forward, which restreams the whole model once
    per call of every leaf - ruinous for a VAE, whose tiled decode calls its blocks
    once per tile. This moves the model as a whole around each entry point, so a
    tiling loop sits inside a single pair of transfers.

    That trade only pays for components called a handful of times per run. A
    denoising transformer is called once per step, so per-call transfers would cost
    far more than they save - group offloading is the tool for those.

    Args:
        component: The component to place
        component_name: Name of the component, for logging
        device: Device to run the component on
        group_offloaded: Whether group offloading was applied to this component
        offload_device: Where the component rests between calls

    Raises:
        ValueError: If the component is also group offloaded
    """
    if group_offloaded:
        raise ValueError(
            f"Component '{component_name}' sets both 'group_offload' and "
            "'residency: on_demand'. A group offloaded module holds one group at a "
            "time and ignores the whole-model moves on-demand placement makes, so "
            "the two cannot both own its placement - pick one"
        )

    if get_device_type(device) == "cpu":
        # Nothing to move it off of, so the wrappers would be pure overhead
        logger.debug(
            f"Ignoring 'residency: on_demand' for {component_name} - {device} is the "
            "device it would rest on anyway"
        )
        return

    component.to(offload_device)

    # One depth counter for the whole component, not one per entry point: decode()
    # calls forward() internally, and an inner return must not offload the model
    # out from under the call that is still running
    state = {"depth": 0}

    def wrap(entry_point):
        original = getattr(component, entry_point, None)
        if not callable(original):
            return False

        @functools.wraps(original)
        def on_demand(*args, **kwargs):
            if state["depth"] == 0:
                component.to(device)
            state["depth"] += 1
            try:
                return original(*args, **kwargs)
            finally:
                state["depth"] -= 1
                if state["depth"] == 0:
                    component.to(offload_device)
                    # Hand the freed space back to the driver rather than leaving it
                    # reserved - the headroom is the entire point of doing this
                    empty_device_cache()

        # functools.wraps carries __wrapped__, so inspect.signature() still reports
        # the real parameters. Callers introspect them: MiniMax H3's denoiser picks
        # which arguments to pass by reading signature(transformer.forward)
        setattr(component, entry_point, on_demand)
        return True

    wrapped = [name for name in _ON_DEMAND_ENTRY_POINTS if wrap(name)]
    if not wrapped:
        raise ValueError(
            f"Component '{component_name}' sets 'residency: on_demand' but defines "
            f"none of {', '.join(_ON_DEMAND_ENTRY_POINTS)}, so there is no call to "
            "move it around"
        )
    logger.info(
        f"Placing {component_name} on demand: resting on {offload_device}, "
        f"running on {device} around {', '.join(wrapped)}"
    )


def apply_mps_rope_precision(pipeline, device):
    """Run a RoPE that asks for float64 in float32 on MPS, which has no float64.

    diffusers makes this choice itself for Wan, Lumina2, SkyReels-V2, ChronoEdit
    and Sana-Video. LTX-2's transformer and text connectors read a
    `double_precision` flag at forward time instead, and left on (the default)
    it failed LTX-2.5's first step on a Mac with "Cannot convert a MPS Tensor to
    float64". Keyed on the flag rather than a model name, so any module that
    exposes one gets the same treatment."""
    if get_device_type(device) != "mps":
        return
    components = getattr(pipeline, "components", None)
    if not isinstance(components, dict):
        return
    for component_name, component in components.items():
        if not isinstance(component, torch.nn.Module):
            continue
        switched = 0
        for module in component.modules():
            if getattr(module, "double_precision", None) is True:
                module.double_precision = False
                switched += 1
        if switched:
            logger.warning(
                f"Computing {switched} float64 RoPE module(s) in {component_name} in "
                "float32 - MPS has no float64 (diffusers does the same for Wan)"
            )


def attention_slicing_requested(configuration, device):
    """Whether a pipeline's attention runs sliced: automatic on MPS unless
    'disable_attention_slicing' is set, opt-in ('enable_attention_slicing')
    everywhere else.

    Which is faster on MPS depends on the model. Measured on an M5 Pro (torch
    2.14), MPS SDPA is slow at head dims 40, 48 and 160 and fast at 32 and
    64-128 - slicing made SD 1.5 (40/80/160) ~20% faster end to end and
    SDXL-shaped attention (64) 2.4x slower. It stays automatic because the
    catalog's quick-start is SD 1.5; SDXL on a Mac can opt out."""
    if configuration.get("enable_attention_slicing", False):
        return True
    return get_device_type(device) == "mps" and not configuration.get(
        "disable_attention_slicing", False
    )


def auto_cpu_offload_enabled(configuration):
    """Whether the configuration asks its components manager to offload to the CPU."""
    return configuration.get("components_manager", {}).get(
        "enable_auto_cpu_offload", False
    )


def auto_cpu_offload_active(configuration, device):
    """Whether the components manager actually owns device placement.

    Mirrors the MPS skip in create_components_manager() - on MPS the manager
    never installs its offload hooks, so callers must not assume it owns
    device placement there.
    """
    return auto_cpu_offload_enabled(configuration) and get_device_type(device) != "mps"


def create_components_manager(configuration, device):
    """Create the components manager for a modular pipeline, when one is configured.

    A ComponentsManager tracks the components of a modular pipeline and can keep only
    the ones currently running on the device, moving the rest to system memory.

    Args:
        configuration: Pipeline configuration dictionary
        device: Device the pipeline runs on

    Returns:
        A configured ComponentsManager, or None when the pipeline does not use one
    """
    manager_configuration = configuration.get("components_manager", None)
    if manager_configuration is None:
        return None

    # Imported here because importing modular diffusers warns that it is experimental
    from diffusers import ComponentsManager

    logger.info("Creating components manager")
    components_manager = ComponentsManager()

    if auto_cpu_offload_enabled(configuration):
        # ComponentsManager.enable_auto_cpu_offload() calls device.mem_get_info(),
        # which torch does not implement for MPS. Unified memory also makes the
        # feature far less useful there than on CUDA, so skip it rather than fail.
        if get_device_type(device) == "mps":
            logger.warning(
                "components_manager auto CPU offload is not supported on MPS, skipping"
            )
        else:
            offload_arguments = {}
            memory_reserve_margin = manager_configuration.get(
                "memory_reserve_margin", None
            )
            if memory_reserve_margin is not None:
                offload_arguments["memory_reserve_margin"] = memory_reserve_margin

            # Enabled before the components load so each one is hooked as it is added
            logger.info(f"Enabling components manager auto CPU offload on {device}")
            components_manager.enable_auto_cpu_offload(
                device=device, **offload_arguments
            )

    return components_manager


def has_component_group_offload(configuration):
    """Whether a per-component entry keeps its component off the device.

    The 'components' block is applied by configure_components() after the pipeline is
    loaded, but load_component() has to decide where to materialize weights and whether
    to move the pipeline to the device before that block is ever read. A workflow whose
    only offload configuration lives under components.* still needs both of those
    earlier decisions to treat it as offloading.

    Group offloading and on-demand residency both qualify: each leaves its component in
    system memory between uses, so materializing the pipeline on the device first would
    load in full exactly what these were configured to avoid holding.

    Args:
        configuration: Configuration of the component being loaded

    Returns:
        True when any per-component entry keeps its component off the device
    """
    components = configuration.get("components") or {}
    return any(
        isinstance(settings, dict)
        and (
            settings.get("group_offload") is not None
            or settings.get("residency") == "on_demand"
        )
        for settings in components.values()
    )


def loading_device(configuration):
    """The device a component's weights are materialized on while it loads.

    Offloading brings each part of a model onto the device only while it runs, so the
    weights have to land in system memory first. A default torch device pointing at the
    GPU would build every module directly in VRAM instead, running a large pipeline out
    of memory before its offload hooks are ever installed.

    Args:
        configuration: Configuration of the component being loaded

    Returns:
        A context manager active for the duration of the load
    """
    offloads = (
        configuration.get("offload", None) is not None
        or configuration.get("group_offload", None) is not None
        or has_component_group_offload(configuration)
    )

    if offloads:
        logger.debug("Loading into system memory - the component will be offloaded")
        return torch.device("cpu")

    return contextlib.nullcontext()


def place_component(
    component, component_name, configuration, device, components_manager=None
):
    """Give a loaded component its offloading hooks and its device.

    Split out of load_component because placement has to come last. Every hook
    here - group offloading, layerwise casting, model or sequential CPU offload -
    pins the modules and weights that exist when it is installed, so anything that
    adds or replaces weights afterwards (a LoRA, an IP-Adapter) is left outside the
    hook's bookkeeping: sequential offload streams the weights it recorded onto the
    accelerator and the adapter's own tensors are never among them, which runs the
    step on uninitialized weights and produces NaN.

    `device` is translated by `resolve_device` first, before anything reads the
    backend, so the MPS accommodations below (the sequential-to-model downgrade
    here, and attention slicing and the compile skip elsewhere) fire for a
    device translated to MPS as they do for one written as `mps`.
    `exclude_from_cpu_offload` is sequential-only: it does not survive the
    downgrade to model offload, which warns that it was dropped.

    Args:
        component: The loaded pipeline or component
        component_name: What is being placed, for the log
        configuration: The component's configuration block
        device: Device the component runs on
        components_manager: The modular pipeline's components manager, if it has one

    Returns:
        The placed component
    """
    # Translate before anything reads the backend - the offload downgrades below
    # have to see the device the component will actually run on
    device = resolve_device(device)

    # Handle group_offload configuration
    group_offload_configuration = get_group_offload_configuration(configuration, device)
    if group_offload_configuration is not None:
        component.enable_group_offload(**group_offload_configuration)

    # Handle enable_layerwise_casting configuration
    enable_layerwise_casting_configuration = configuration.get(
        "enable_layerwise_casting", None
    )
    if enable_layerwise_casting_configuration is not None:
        component.enable_layerwise_casting(**enable_layerwise_casting_configuration)

    # Configure component device settings
    preserve_device_placement = configuration.get("preserve_device_placement", False)
    offload = configuration.get("offload", None)

    # Offloading streams a model between system memory and an accelerator - there is
    # nothing to stream to when the run is on the CPU
    if offload is not None and get_device_type(device) == "cpu":
        logger.warning(f"Ignoring '{offload}' offload - {device} is not an accelerator")
        offload = None

    # Sequential offload streams each submodule onto the accelerator as it runs,
    # a trade only a separate memory pool rewards. MPS shares one pool with the
    # CPU, so the streaming hands back no residency and costs a copy per
    # submodule per step. Model offload keeps the coarse win - idle components
    # off the Metal allocator - without paying that
    if offload == "sequential" and get_device_type(device) == "mps":
        excluded = configuration.get("exclude_from_cpu_offload", [])
        ignored = (
            f"; 'exclude_from_cpu_offload' ({', '.join(excluded)}) is sequential-only "
            "and does not carry over"
            if excluded
            else ""
        )
        logger.warning(
            f"Using model offload in place of sequential on {device} - sequential "
            f"streams weights per submodule, which buys back no memory on unified "
            f"memory{ignored}"
        )
        offload = "model"

    if offload == "model":
        logger.debug(f"Enabling model CPU offload onto {device}")
        component.enable_model_cpu_offload(device=device)
    elif offload == "sequential":
        logger.debug(f"Enabling sequential CPU offload onto {device}")
        for excluded_name in configuration.get("exclude_from_cpu_offload", []):
            logger.debug(f"Excluding {excluded_name} from CPU offload")
            component._exclude_from_cpu_offload.append(excluded_name)
        component.enable_sequential_cpu_offload(device=device)
    elif components_manager is not None and auto_cpu_offload_active(
        configuration, device
    ):
        # Moving everything to the device here would defeat the offloading - the
        # manager's hooks bring each component on device as the pipeline needs it
        logger.debug("Device placement is owned by the components manager")
    elif has_component_group_offload(configuration):
        # configure_components() installs group-offload hooks per-component after
        # this returns - moving the whole pipeline to the device now would load it
        # in full before those hooks exist, defeating the offloading
        logger.info(
            f"components configure group offloading - not moving pipeline to {device}"
        )
    elif hasattr(component, "to") and not preserve_device_placement:
        logger.debug(f"Moving {component_name} to device: {device}")
        component = component.to(device)

    return component
