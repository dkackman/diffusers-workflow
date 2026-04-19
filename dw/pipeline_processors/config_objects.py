import logging
import torch
from diffusers import (
    FirstBlockCacheConfig,
    FasterCacheConfig,
    MagCacheConfig,
    TaylorSeerCacheConfig,
    TextKVCacheConfig,
)

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
    if quantization_config is not None:
        logger.info("Loading quantization configuration...")
        logger.debug(f"Quantization parameters: {quantization_config}")
        try:
            quantization_config_type = quantization_config["configuration"][
                "config_type"
            ]
            # Some quantization configs (e.g. TorchAoConfig) require argument values
            # to be instances rather than classes. realize_args converts *_type keys to
            # classes; instantiate them here with no args so callers can write e.g.
            # "quant_type": "torchao.quantization.Int8WeightOnlyConfig" in JSON.
            args = {
                k: v() if isinstance(v, type) else v
                for k, v in quantization_config["arguments"].items()
            }
            return quantization_config_type(**args)
        except Exception as e:
            logger.error(
                f"Failed to create quantization_config: {str(e)}", exc_info=True
            )
            raise

    logger.debug("No quantization configuration found")
    return None


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
            group_offload_config.get("onload_device", default_device)
        )
        group_offload_config["offload_device"] = torch.device(
            group_offload_config.get("offload_device", "cpu")
        )

        return group_offload_config

    logger.debug("No group offload configuration found")
    return None


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
                kwargs = {}
                if "threshold" in cache_config:
                    kwargs["threshold"] = cache_config["threshold"]
                if "num_inference_steps" in cache_config:
                    kwargs["num_inference_steps"] = cache_config["num_inference_steps"]
                if "max_skip_steps" in cache_config:
                    kwargs["max_skip_steps"] = cache_config["max_skip_steps"]
                if "retention_ratio" in cache_config:
                    kwargs["retention_ratio"] = cache_config["retention_ratio"]
                config = MagCacheConfig(**kwargs)
            elif cache_type == "taylorseer":
                kwargs = {}
                if "cache_interval" in cache_config:
                    kwargs["cache_interval"] = cache_config["cache_interval"]
                if "max_order" in cache_config:
                    kwargs["max_order"] = cache_config["max_order"]
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