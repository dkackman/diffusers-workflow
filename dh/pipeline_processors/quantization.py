import logging
from diffusers import BitsAndBytesConfig, GGUFQuantizationConfig, TorchAoConfig

logger = logging.getLogger("dh")


def get_quantization_configuration(configuration):
    """
    Get the appropriate quantization configuration based on the input configuration.

    Args:
        configuration: Dictionary containing quantization settings

    Returns:
        Quantization configuration object or None if no valid configuration found
    """
    logger.debug(f"Processing quantization configuration: {configuration}")

    # Check for bits and bytes configuration
    bits_and_bytes_configuration = configuration.get(
        "bits_and_bytes_configuration", None
    )
    if bits_and_bytes_configuration is not None:
        logger.info("Loading bits and bytes configuration...")
        logger.debug(f"Bits and bytes parameters: {bits_and_bytes_configuration}")
        try:
            return BitsAndBytesConfig(**bits_and_bytes_configuration)
        except Exception as e:
            logger.error(
                f"Failed to create BitsAndBytesConfig: {str(e)}", exc_info=True
            )
            raise

    # Check for GGUF configuration
    gguf_configuration = configuration.get("gguf_configuration", None)
    if gguf_configuration is not None:
        logger.info("Loading gguf configuration...")
        logger.debug(f"GGUF parameters: {gguf_configuration}")
        try:
            return GGUFQuantizationConfig(**gguf_configuration)
        except Exception as e:
            logger.error(
                f"Failed to create GGUFQuantizationConfig: {str(e)}", exc_info=True
            )
            raise

    # Check for TorchAO configuration
    torchao_configuration = configuration.get("torchao_configuration", None)
    if torchao_configuration is not None:
        logger.info("Loading torchao configuration...")
        logger.debug(f"TorchAO parameters: {torchao_configuration}")
        try:
            quantization = torchao_configuration.pop("quantization")
            return TorchAoConfig(quantization, **torchao_configuration)
        except Exception as e:
            logger.error(f"Failed to create TorchAoConfig: {str(e)}", exc_info=True)
            raise

    logger.debug("No quantization configuration found")
    return None
