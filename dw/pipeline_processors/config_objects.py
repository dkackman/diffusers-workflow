import logging
import torch

from .. import get_device_type, resolve_device

logger = logging.getLogger("dw")


def get_quantization_configuration(configuration):
    """
    Get the appropriate quantization configuration based on the input configuration.

    Args:
        configuration: Dictionary containing quantization settings

    Returns:
        Quantization configuration object or None if no valid configuration found
    """
    logger.debug(f"Processing quantization configuration: {configuration}")

    quantization_config = configuration.get("quantization_config", None)
    if quantization_config is None:
        logger.debug("No quantization configuration found")
        return None

    return create_quantization_config(quantization_config)


# Quantization config arguments that name a device. A workflow authored on the
# CUDA box writes "cuda" here, and nothing downstream translates it - SDNQ moves
# every weight to it inside from_pretrained, which fails outright on a Mac
DEVICE_ARGUMENTS = ("quantization_device", "return_device")

# SDNQ's quantized matmul has no Triton on MPS and falls back to torch._int_mm,
# measured at ~500x the bf16 matmul it replaces (1096 ms vs 2.1 ms, 512x3072 by
# 3072x3072 on an M5 Pro). The dequantize path is the fast one there
QUANTIZED_MATMUL_ARGUMENTS = ("use_quantized_matmul", "use_quantized_matmul_conv")


def portable_quantization_arguments(arguments):
    """Adapt a quantization config's arguments to this machine without touching
    the definition they came from - a cached pipeline builds from it again."""
    adapted = dict(arguments)
    for key in DEVICE_ARGUMENTS:
        if isinstance(adapted.get(key), str):
            adapted[key] = resolve_device(adapted[key])
    if get_device_type() == "mps":
        for key in QUANTIZED_MATMUL_ARGUMENTS:
            if adapted.get(key):
                logger.warning(
                    f"Turning off '{key}' on MPS - without Triton it runs through "
                    "torch._int_mm, which is far slower there than dequantizing"
                )
                adapted[key] = False
    return adapted


def create_quantization_config(quantization_config):
    """
    Create a quantization configuration object from its definition.

    Args:
        quantization_config: Dictionary holding the config type and its arguments

    Returns:
        Quantization configuration object
    """
    logger.info("Loading quantization configuration...")
    logger.debug(f"Quantization parameters: {quantization_config}")
    try:
        quantization_config_type = quantization_config["configuration"]["config_type"]
        # Some quantization configs (e.g. TorchAoConfig) require argument values
        # to be instances rather than classes. realize_args converts *_type keys to
        # classes; instantiate them here with no args so callers can write e.g.
        # "quant_type": "torchao.quantization.Int8WeightOnlyConfig" in JSON.
        args = {
            k: v() if isinstance(v, type) else v
            for k, v in portable_quantization_arguments(
                quantization_config["arguments"]
            ).items()
        }
        return quantization_config_type(**args)
    except Exception as e:
        logger.error(f"Failed to create quantization_config: {str(e)}", exc_info=True)
        raise


def get_load_components_arguments(configuration):
    """
    Get the arguments for a modular pipeline's load_components(), with any quantization
    configurations built from their definitions.

    load_components() hands its arguments to each component's from_pretrained, looking a
    dict value up under the component's own name - so quantization is declared per
    component, and a component the map does not name is loaded unquantized.

    Args:
        configuration: Pipeline configuration dictionary

    Returns:
        Dictionary of arguments for load_components(), or None when the pipeline does not
        load its components separately
    """
    load_components_arguments = configuration.get("load_components", None)
    if load_components_arguments is None:
        return None

    load_components_arguments = dict(load_components_arguments)
    quantization_configs = load_components_arguments.get("quantization_config", None)
    if quantization_configs is not None:
        logger.debug(
            f"Building quantization configurations for: {list(quantization_configs.keys())}"
        )
        load_components_arguments["quantization_config"] = {
            component_name: create_quantization_config(definition)
            for component_name, definition in quantization_configs.items()
        }

    return load_components_arguments


def get_group_offload_configuration(configuration, default_device):
    """
    Get the appropriate group offload configuration based on the input configuration.

    Args:
        configuration: Dictionary containing group offload settings

    Returns:
        Group offload configuration object or None if no valid configuration found
        https://huggingface.co/docs/diffusers/optimization/memory#group-offloading
    """
    logger.debug(f"Processing group offload configuration: {configuration}")

    group_offload_config = configuration.get("group_offload", None)
    if group_offload_config is not None:
        logger.info("Loading group offload configuration...")
        logger.debug(f"Group offload parameters: {group_offload_config}")
        # replace device references with device objects
        group_offload_config["onload_device"] = torch.device(
            resolve_device(group_offload_config.get("onload_device", default_device))
        )
        group_offload_config["offload_device"] = torch.device(
            resolve_device(group_offload_config.get("offload_device", "cpu"))
        )

        # CUDA streams overlap the next group's transfer with this one's compute.
        # diffusers refuses them without CUDA or XPU, and refuses record_stream
        # without use_stream, so both go together - a catalog template written
        # for the CUDA box would otherwise fail to load on a Mac
        if group_offload_config["onload_device"].type not in ("cuda", "xpu"):
            dropped = [
                key
                for key in ("use_stream", "record_stream")
                if group_offload_config.pop(key, False)
            ]
            if dropped:
                logger.warning(
                    f"Ignoring group offload {', '.join(dropped)} - streams need "
                    f"CUDA or XPU, and this onloads to "
                    f"{group_offload_config['onload_device']}"
                )

        return group_offload_config

    logger.debug("No group offload configuration found")
    return None


def _resolve_mag_ratios(mag_ratios):
    """Resolve a mag_ratios declaration into what MagCacheConfig accepts.

    The ratios are checkpoint-dependent, so a workflow either spells them out as
    a per-step array or names one of the presets diffusers ships in
    diffusers.hooks.mag_cache - "flux" resolving to FLUX_MAG_RATIOS. The lookup
    is dynamic, so a preset added by a later diffusers release works here with no
    change, the same way quantization config_type does.

    Args:
        mag_ratios: A preset name, or a list of per-step ratios

    Returns:
        The ratios to hand to MagCacheConfig
    """
    if not isinstance(mag_ratios, str):
        return mag_ratios

    from diffusers.hooks import mag_cache

    ratios = getattr(mag_cache, f"{mag_ratios.upper()}_MAG_RATIOS", None)
    if ratios is None:
        available = sorted(
            name.removesuffix("_MAG_RATIOS").lower()
            for name in dir(mag_cache)
            if name.endswith("_MAG_RATIOS")
        )
        raise ValueError(
            f"Unknown mag_ratios preset: {mag_ratios}. "
            f"Available presets: {available}. "
            f"A list of per-step ratios can be given instead."
        )
    return ratios


# Arguments each cache type forwards to its diffusers config, when present.
# Anything omitted keeps the diffusers default.
_MAG_CACHE_KEYS = (
    "threshold",
    "num_inference_steps",
    "max_skip_steps",
    "retention_ratio",
    "calibrate",
)
_TAYLORSEER_CACHE_KEYS = ("cache_interval", "max_order")


def get_cache_configuration(configuration):
    """
    Get the appropriate diffusers cache configuration based on the input configuration.

    Args:
        configuration: Dictionary containing cache settings

    Returns:
        Cache configuration object or None if no valid configuration found
    """
    logger.debug(f"Processing cache configuration: {configuration}")

    cache_config = configuration.get("cache", None)
    if cache_config is not None:
        # Imported here rather than at module scope - the cache configs live in
        # diffusers.hooks, which drags in peft (~2s) every startup otherwise
        from diffusers import (
            FirstBlockCacheConfig,
            FasterCacheConfig,
            MagCacheConfig,
            TaylorSeerCacheConfig,
            TextKVCacheConfig,
        )

        logger.info("Loading cache configuration...")
        logger.debug(f"Cache parameters: {cache_config}")
        try:
            cache_type = cache_config["type"]

            if cache_type == "first_block":
                config = FirstBlockCacheConfig(
                    threshold=cache_config.get("threshold", 0.05),
                )
            elif cache_type == "faster":
                config = FasterCacheConfig()
            elif cache_type == "text_kv":
                config = TextKVCacheConfig()
            elif cache_type == "mag":
                kwargs = {
                    key: cache_config[key]
                    for key in _MAG_CACHE_KEYS
                    if key in cache_config
                }
                # Checkpoint-dependent, and required unless calibrating - without
                # forwarding it MagCacheConfig rejects every configuration
                if "mag_ratios" in cache_config:
                    kwargs["mag_ratios"] = _resolve_mag_ratios(
                        cache_config["mag_ratios"]
                    )
                config = MagCacheConfig(**kwargs)
            elif cache_type == "taylorseer":
                kwargs = {
                    key: cache_config[key]
                    for key in _TAYLORSEER_CACHE_KEYS
                    if key in cache_config
                }
                config = TaylorSeerCacheConfig(**kwargs)
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")

            return config
        except Exception as e:
            logger.error(
                f"Failed to create cache configuration: {str(e)}", exc_info=True
            )
            raise

    logger.debug("No cache configuration found")
    return None
