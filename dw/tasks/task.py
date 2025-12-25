import logging
from typing import Callable, Dict
from .qr_code import get_qrcode_image
from .image_utils import process_image
from .video_utils import process_video
from .gather import gather_images, gather_inputs, gather_videos
from .format_messages import (
    format_chat_message,
    batch_decode_post_process,
    get_dict_value,
)

logger = logging.getLogger("dw")


# Command registry: maps command names to handler functions
_COMMAND_REGISTRY: Dict[str, Callable] = {}


def register_command(command_name: str):
    """
    Decorator to register a command handler function.

    Args:
        command_name: The command name to register

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        _COMMAND_REGISTRY[command_name] = func
        logger.debug(f"Registered command handler: {command_name}")
        return func

    return decorator


# Command handler functions
@register_command("qr_code")
def _handle_qr_code(task, arguments, previous_pipelines):
    """Generate QR code image"""
    logger.debug("Generating QR code")
    return get_qrcode_image(**arguments)


@register_command("gather_images")
def _handle_gather_images(task, arguments, previous_pipelines):
    """Gather multiple images"""
    logger.debug("Gathering images")
    return gather_images(**arguments)


@register_command("gather_videos")
def _handle_gather_videos(task, arguments, previous_pipelines):
    """Gather multiple videos"""
    logger.debug("Gathering videos")
    return gather_videos(**arguments)


@register_command("gather_inputs")
def _handle_gather_inputs(task, arguments, previous_pipelines):
    """Gather inputs from various sources"""
    logger.debug("Gathering inputs")
    return gather_inputs(arguments)


@register_command("format_chat_message")
def _handle_format_chat_message(task, arguments, previous_pipelines):
    """Format chat message for LLM input"""
    logger.debug("Formatting chat message")
    return format_chat_message(**arguments)


@register_command("get_dict_value")
def _handle_get_dict_value(task, arguments, previous_pipelines):
    """Extract value from dictionary"""
    logger.debug("Getting dictionary value")
    return get_dict_value(**arguments)


@register_command("batch_decode_post_process")
def _handle_batch_decode(task, arguments, previous_pipelines):
    """Batch decode post-processing with pipeline reference"""
    logger.debug("Performing batch decode post-processing")
    pipeline_reference = task.task_definition["pipeline_reference"]
    if pipeline_reference not in previous_pipelines:
        raise KeyError(
            f"Pipeline reference '{pipeline_reference}' not found in previous pipelines. "
            f"Available pipelines: {list(previous_pipelines.keys())}"
        )
    processor = previous_pipelines[pipeline_reference].pipeline
    return batch_decode_post_process(processor, **arguments)


def _handle_image_processing(task, arguments, previous_pipelines):
    """Handle image processing commands"""
    logger.debug("Processing image")
    return process_image(
        arguments.pop("image"),
        task.command,
        task.device,
        arguments,
    )


def _handle_video_processing(task, arguments, previous_pipelines):
    """Handle video processing commands"""
    logger.debug("Processing video")
    return process_video(
        arguments.pop("video"),
        task.command,
        task.device,
        arguments,
    )


class Task:
    """
    Represents a task that can be executed as part of a workflow.
    Tasks are atomic operations like image processing, data gathering, or message formatting.
    """

    def __init__(self, task_definition, device):
        """
        Initialize task with its configuration and device settings.

        Args:
            task_definition: Dictionary containing task configuration and parameters
            device: Device to run task on (e.g., 'cuda', 'mps', 'cpu')
        """
        self.task_definition = task_definition
        self.device = device
        logger.debug(f"Initialized task: {self.name} for device: {device}")

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
        Execute the task with given arguments using the command registry.

        Args:
            arguments: Dictionary of arguments for task execution
            previous_pipelines: Dictionary of previously created pipelines

        Returns:
            Task output based on command type

        Raises:
            ValueError: If command is unknown
            KeyError: If required arguments or pipeline references are missing
        """
        logger.debug(f"Running task: {self.command}")
        logger.debug(f"Task arguments: {arguments}")

        try:
            # Look up command in registry
            if self.command in _COMMAND_REGISTRY:
                handler = _COMMAND_REGISTRY[self.command]
                return handler(self, arguments, previous_pipelines)

            # Fallback: check for image/video processing (generic handlers)
            if "image" in arguments:
                return _handle_image_processing(self, arguments, previous_pipelines)

            if "video" in arguments:
                return _handle_video_processing(self, arguments, previous_pipelines)

            # Unknown command
            error_msg = (
                f"Unknown task command: '{self.command}'. "
                f"Registered commands: {sorted(_COMMAND_REGISTRY.keys())}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        except KeyError as e:
            # Missing required arguments or pipeline references
            logger.error(
                f"Missing required data for task {self.command}: {e}", exc_info=True
            )
            raise
        except (ValueError, TypeError) as e:
            # Invalid arguments or type mismatches
            logger.error(
                f"Invalid arguments for task {self.command}: {e}", exc_info=True
            )
            raise
        except (OSError, IOError) as e:
            # File operations, resource loading errors
            logger.error(f"I/O error in task {self.command}: {e}", exc_info=True)
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(
                f"Unexpected error ({type(e).__name__}) executing task {self.command}: {e}",
                exc_info=True,
            )
            raise
