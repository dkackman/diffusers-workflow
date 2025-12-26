import logging
import torch

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
            return quantization_config_type(**quantization_config["arguments"])
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
