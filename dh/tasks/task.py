import logging
from .qr_code import get_qrcode_image
from .image_processor_dispatch import process_image
from .gather import gather_images, gather_inputs, gather_videos
from .format_messages import (
    format_chat_message,
    batch_decode_post_process,
    get_dict_value,
)

logger = logging.getLogger("dh")


class Task:
    """
    Represents a task that can be executed as part of a workflow.
    Tasks are atomic operations like image processing, data gathering, or message formatting.
    """

    def __init__(self, task_definition, device_identifier):
        """
        Initialize task with its configuration and device settings.

        Args:
            task_definition: Dictionary containing task configuration and parameters
            device_identifier: Device to run task on (e.g., 'cuda')
        """
        self.task_definition = task_definition
        self.device_identifier = device_identifier
        logger.debug(f"Initialized task: {self.name} for device: {device_identifier}")

    @property
    def name(self):
        """Get task name from command property"""
        return self.command

    @property
    def argument_template(self):
        """
        Get argument template for this task.

        Returns:
            Dictionary of arguments from inputs or arguments section
        """
        # A task will either be an input array or a dictionary of arguments
        if "inputs" in self.task_definition:
            logger.debug("Using inputs as argument template")
            return self.task_definition["inputs"]

        logger.debug("Using arguments as argument template")
        return self.task_definition["arguments"]

    @property
    def command(self):
        """Get command name or 'unknown' if not specified"""
        return self.task_definition.get("command", "unknown")

    def run(self, arguments, previous_pipelines={}):
        """
        Execute the task with given arguments.

        Args:
            arguments: Dictionary of arguments for task execution
            previous_pipelines: Dictionary of previously created pipelines

        Returns:
            Task output based on command type

        Raises:
            ValueError: If command is unknown
        """
        logger.debug(f"Running task: {self.command}")
        logger.debug(f"Task arguments: {arguments}")

        try:
            # QR Code generation
            if self.command == "qr_code":
                logger.debug("Generating QR code")
                return get_qrcode_image(**arguments)

            # Image gathering
            if self.command == "gather_images":
                logger.debug("Gathering images")
                return gather_images(**arguments)

            # Video gathering
            if self.command == "gather_videos":
                logger.debug("Gathering videos")
                return gather_videos(**arguments)

            # Input gathering
            if self.command == "gather_inputs":
                logger.debug("Gathering inputs")
                return gather_inputs(arguments)

            # Chat message formatting
            if self.command == "format_chat_message":
                logger.debug("Formatting chat message")
                return format_chat_message(**arguments)

            # Dictionary value retrieval
            if self.command == "get_dict_value":
                logger.debug("Getting dictionary value")
                return get_dict_value(**arguments)

            # Batch decode post-processing
            if self.command == "batch_decode_post_process":
                logger.debug("Performing batch decode post-processing")
                pipeline_reference = self.task_definition["pipeline_reference"]
                processor = previous_pipelines[pipeline_reference].pipeline
                return batch_decode_post_process(processor, **arguments)

            # Image processing
            if "image" in arguments:
                logger.debug("Processing image")
                return process_image(
                    arguments.pop("image"),
                    self.command,
                    self.device_identifier,
                    arguments,
                )

            # Unknown command
            error_msg = f"Unknown task {self.command}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            logger.error(
                f"Error executing task {self.command}: {str(e)}", exc_info=True
            )
            raise
