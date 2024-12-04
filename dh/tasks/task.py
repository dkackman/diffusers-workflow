from .qr_code import get_qrcode_image
from .image_processor_dispatch import process_image
from .gather import gather


class Task:
    def __init__(self, task_definition):
        self.task_definition = task_definition

    @property
    def argument_template(self):
        return self.task_definition["arguments"]
    
    @property
    def command(self):
        return self.task_definition.get("command", "unknown")
        
    def run(self, device_identifier, arguments):                                            
        if self.command == "qr_code":
            return get_qrcode_image(**arguments)
        
        if self.command == "gather":
            return gather(**arguments)

        if "image" in arguments:
            return process_image(arguments.pop("image"), self.command, device_identifier, arguments)
                    
        raise ValueError(f"Unknown task {self.command}")
    