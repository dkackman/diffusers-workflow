from .qr_code import get_qrcode_image
from .image_processor_dispatch import process_image
from .gather import gather_images, gather_inputs, gather_videos
from .format_messages import format_chat_message


class Task:
    def __init__(self, task_definition):
        self.task_definition = task_definition

    @property
    def argument_template(self):
        # a task will either be an input array or a dictionary of arguments
        if "inputs" in self.task_definition:
            return self.task_definition["inputs"]

        return self.task_definition["arguments"]

    @property
    def command(self):
        return self.task_definition.get("command", "unknown")

    def run(self, device_identifier, arguments):
        if self.command == "qr_code":
            return get_qrcode_image(**arguments)

        if self.command == "gather_images":
            return gather_images(**arguments)

        if self.command == "gather_videos":
            return gather_videos(**arguments)

        if self.command == "gather_inputs":
            return gather_inputs(arguments)

        if self.command == "format_chat_message":
            return format_chat_message(**arguments)

        if "image" in arguments:
            return process_image(
                arguments.pop("image"), self.command, device_identifier, arguments
            )

        raise ValueError(f"Unknown task {self.command}")
