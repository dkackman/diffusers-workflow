from .qr_code import get_qrcode_image
from .image_processor_dispatch import process_image


class Task:
    def __init__(self, task_definition):
        self.task_definition = task_definition

    def run(self, device_identifier):
        kwargs = self.task_definition.get("arguments", {})
                                             
        if self.command == "qr_code":
            return get_qrcode_image(**kwargs)

        if "image" in kwargs:
            return process_image(kwargs.pop("image"), self.command, device_identifier, kwargs)
                    
        raise ValueError(f"Unknown task {self.command}")

    
    @property
    def command(self):
        return self.task_definition.get("command", "unknown")