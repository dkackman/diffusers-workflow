from torchao.quantization import quantize_ as torchao_quantize
from optimum.quanto import freeze, quantize as quanto_quantize


def quantize(component, quantization_definition):
    if quantization_definition is not None:
        quantization_library = quantization_definition["quantization_library"]
        print(f"Quantizing {type(component).__name__} using {quantization_library}...")
        weights_type = quantization_definition["weights_type"]

        try:
            if quantization_library == "torchao":
                torchao_quantize(component, weights_type())

            elif quantization_library == "optimum.quanto":
                activations_type = quantization_definition["activations_type"]
                quanto_quantize(component, weights=weights_type, activations=activations_type)
                freeze(component)

            else:
                raise ValueError(f"Quantization library {quantization_library} not supported")
            
        except Exception as e:
            print(f"Error while quantizing {type(component).__name__} using {quantization_library}")
            print(e)
            raise e