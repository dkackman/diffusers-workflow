from .processor_dispatch import process_image
from ..result import Result

class ImageProcessor:
    def __init__(self, processor_definition):
        self.processor_definition = processor_definition

    def run(self, device_identifier):
        processor_output = process_image(self.processor_definition["image"], self.processor_definition["processor_name"], device_identifier, self.processor_definition.get("arguments", {}))
        result = Result(processor_output, self.processor_definition.get("result", {}))
        return result