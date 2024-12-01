from .qr_code import get_qrcode_image
from .image_processor_dispatch import process_image

class Task:
    def __init__(self, task_definition):
        self.task_definition = task_definition

    def run(self, device_identifier):
        command = self.task_definition["command"]
        kwargs = self.task_definition.get("arguments", {})
                                             
        if command == "qr_code":
            return get_qrcode_image(**kwargs)

        if "image" in kwargs:
            return process_image(kwargs.pop("image"), command, device_identifier, kwargs)
                    
        raise ValueError(f"Unknown task {command}")

    
    @property
    def command(self):
        return self.task_definition.get("command", "unknown")