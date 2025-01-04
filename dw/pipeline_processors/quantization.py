import logging

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
