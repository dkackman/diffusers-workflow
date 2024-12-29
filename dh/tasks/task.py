from .qr_code import get_qrcode_image
from .image_processor_dispatch import process_image
from .gather import gather_images, gather_inputs, gather_videos
from .format_messages import (
    format_chat_message,
    batch_decode_post_process,
    get_dict_value,
)


class Task:
    def __init__(self, task_definition, device_identifier):
        self.task_definition = task_definition
        self.device_identifier = device_identifier

    @property
    def name(self):
        return self.command

    @property
    def argument_template(self):
        # a task will either be an input array or a dictionary of arguments
        if "inputs" in self.task_definition:
            return self.task_definition["inputs"]

        return self.task_definition["arguments"]

    @property
    def command(self):
        return self.task_definition.get("command", "unknown")

    def run(self, arguments, previous_pipelines={}):
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

        if self.command == "get_dict_value":
            return get_dict_value(**arguments)

        if self.command == "batch_decode_post_process":
            pipeline_reference = self.task_definition["pipeline_reference"]
            processor = previous_pipelines[pipeline_reference].pipeline
            return batch_decode_post_process(processor, **arguments)

        if "image" in arguments:
            return process_image(
                arguments.pop("image"), self.command, self.device_identifier, arguments
            )

        raise ValueError(f"Unknown task {self.command}")
