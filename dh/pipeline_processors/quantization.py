from torchao.quantization import quantize_ as torachao_quantize
from optimum.quanto import freeze, quantize as quanto_quantize

def quantize(component, quantization_definition):
    if quantization_definition is not None:
        quantization_library = quantization_definition["quantization_library"]
        weights_type = quantization_definition["weights_type"]

        if quantization_library == "torachao":
            torachao_quantize(component, weights_type())

        elif quantization_library == "optimum.quanto":
            quanto_quantize(component, weights=weights_type)
            freeze(component)

        else:
            raise ValueError(f"Quantization library {quantization_library} not supported")