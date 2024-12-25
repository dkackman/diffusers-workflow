from torchao.quantization import autoquant, quantize_ as torchao_quantize
from optimum.quanto import freeze, quantize as quanto_quantize
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


def quantize(component, quantization_definition, device_identifier):
    if quantization_definition is not None:
        quantization_library = quantization_definition["quantization_library"]
        print(f"Quantizing {type(component).__name__} using {quantization_library}...")
        weights_type = quantization_definition["weights_type"]

        try:
            if quantization_library == "torchao":
                torchao_quantize(component, weights_type(), device=device_identifier)
                return component

            if quantization_library == "torchao.autoquant":
                return autoquant(component, error_on_unseen=False)

            if quantization_library == "optimum.quanto":
                activations_type = quantization_definition["activations_type"]
                quanto_quantize(
                    component, weights=weights_type, activations=activations_type
                )
                freeze(component)
                return component

            else:
                raise ValueError(
                    f"Quantization library {quantization_library} not supported"
                )

        except Exception as e:
            print(
                f"Error while quantizing {type(component).__name__} using {quantization_library}"
            )
            print(e)
            raise e

    return component
