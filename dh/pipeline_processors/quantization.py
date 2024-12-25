from diffusers import BitsAndBytesConfig, GGUFQuantizationConfig, TorchAoConfig


def get_quantization_configuration(configuration):
    bits_and_bytes_configuration = configuration.get(
        "bits_and_bytes_configuration", None
    )
    if bits_and_bytes_configuration is not None:
        print("Loading bits and bytes configuration...")
        return BitsAndBytesConfig(**bits_and_bytes_configuration)

    gguf_configuration = configuration.get("gguf_configuration", None)
    if gguf_configuration is not None:
        print("Loading gguf configuration...")
        return GGUFQuantizationConfig(**gguf_configuration)

    torchao_configuration = configuration.get("torchao_configuration", None)
    if torchao_configuration is not None:
        print("Loading torchao configuration...")
        quantization = torchao_configuration.pop("quantization")
        return TorchAoConfig(quantization, **torchao_configuration)

    return None
