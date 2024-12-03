import copy
from .qr_code import get_qrcode_image
from .image_processor_dispatch import process_image
from .gather import gather
from ..previous_results import get_previous_results, find_previous_result_refs

class Task:
    def __init__(self, task_definition):
        self.task_definition = task_definition

    def get_iterations(self, previous_results):
        argument_template = self.task_definition.get("arguments", {})
        result_refs = find_previous_result_refs(argument_template)
        if len(result_refs) == 0:
            return [argument_template]
            
        if len(result_refs) > 1:
            raise ValueError(f"Task {self.command} can have only one previous_result reference")

        result_ref_key, result_ref_value = next(iter(result_refs.items()))    
        iterations = []
        for previous_result in get_previous_results(previous_results, result_ref_value):
            arguments = copy.deepcopy(argument_template)
            arguments[result_ref_key] = previous_result
            iterations.append(arguments)
            
        return iterations
    
    def run(self, device_identifier, arguments):                                            
        if self.command == "qr_code":
            return get_qrcode_image(**arguments)
        
        if self.command == "gather":
            return gather(**arguments)

        if "image" in arguments:
            return process_image(arguments.pop("image"), self.command, device_identifier, arguments)
                    
        raise ValueError(f"Unknown task {self.command}")
    
    @property
    def command(self):
        return self.task_definition.get("command", "unknown")
    