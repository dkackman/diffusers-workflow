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

# The model-backed handlers (upscale, restore_faces, segment, interpolate_frames,
# image_to_text, text_generation, diffusion_upscale) are imported inside their
# handlers - at module scope their transformers/model imports add seconds to
# every startup for workflows that never run those tasks

logger = logging.getLogger("dw")


# Command registry: maps command names to handler functions
_COMMAND_REGISTRY: Dict[str, Callable] = {}

# What each command's arguments actually are. The handlers forward
# **arguments into an implementation function, so that function's signature
# is the command's argument schema - registering its dotted path here (a
# string, to preserve the lazy-import discipline) lets the introspection
# layer read the same signature the runtime calls. 'provided' names the
# parameters the dispatch supplies itself, which are not workflow arguments.
_COMMAND_INFO: Dict[str, dict] = {}


def register_command(command_name: str, implementation=None, provided=()):
    """
    Decorator to register a command handler function.

    Args:
        command_name: The command name to register
        implementation: Dotted path to the function whose signature defines
            the command's arguments (None for a command that consumes a
            free-form dict)
        provided: Parameter names the dispatch supplies itself

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        _COMMAND_REGISTRY[command_name] = func
        _COMMAND_INFO[command_name] = {
            "kind": "command",
            "implementation": implementation,
            "provided": tuple(provided),
        }
        logger.debug(f"Registered command handler: {command_name}")
        return func

    return decorator


def task_command_info(command_name):
    """Where a task command's argument schema lives: a dict with 'kind'
    ('command', 'image_processor' or 'video_processor'), 'implementation'
    (dotted path or None for free-form), and 'provided'. Raises ValueError
    for a name that is not a task command at all."""
    info = _COMMAND_INFO.get(command_name)
    if info is not None:
        return info
    if command_name in _VIDEO_PROCESSOR_INFO:
        return _VIDEO_PROCESSOR_INFO[command_name]
    from .image_utils import available_processors

    if command_name in available_processors():
        return {"kind": "image_processor", "implementation": None, "provided": ()}
    raise ValueError(f"Unknown task command: '{command_name}'")


# Command handler functions
@register_command("qr_code", implementation="dw.tasks.qr_code.get_qrcode_image")
def _handle_qr_code(task, arguments, previous_pipelines):
    """Generate QR code image"""
    logger.debug("Generating QR code")
    return get_qrcode_image(**arguments)


@register_command("gather_images", implementation="dw.tasks.gather.gather_images")
def _handle_gather_images(task, arguments, previous_pipelines):
    """Gather multiple images"""
    logger.debug("Gathering images")
    return gather_images(**arguments)


@register_command("gather_videos", implementation="dw.tasks.gather.gather_videos")
def _handle_gather_videos(task, arguments, previous_pipelines):
    """Gather multiple videos"""
    logger.debug("Gathering videos")
    return gather_videos(**arguments)


# gather_inputs passes its whole dict through unchanged - free-form by design
@register_command("gather_inputs")
def _handle_gather_inputs(task, arguments, previous_pipelines):
    """Gather inputs from various sources"""
    logger.debug("Gathering inputs")
    return gather_inputs(arguments)


@register_command(
    "concat_videos", implementation="dw.tasks.concat_videos.concat_videos"
)
def _handle_concat_videos(task, arguments, previous_pipelines):
    """Concatenate videos - and the audio generated with them - into one"""
    logger.debug("Concatenating videos")
    from .concat_videos import concat_videos

    return concat_videos(**arguments)


@register_command("slice_audio", implementation="dw.tasks.audio_utils.slice_audio")
def _handle_slice_audio(task, arguments, previous_pipelines):
    """Cut a time- or frame-aligned slice out of an audio track"""
    logger.debug("Slicing audio")
    from .audio_utils import slice_audio

    return slice_audio(**arguments)


@register_command("video_frames", implementation="dw.tasks.video_utils.frames_as_array")
def _handle_video_frames(task, arguments, previous_pipelines):
    """The frames of a generated video, as one array a later step can condition on"""
    logger.debug("Extracting video frames")
    from .video_utils import frames_as_array

    return frames_as_array(**arguments)


@register_command("pair_audio", implementation="dw.tasks.pair_audio.pair_audio")
def _handle_pair_audio(task, arguments, previous_pipelines):
    """Pair a video's frames with an audio track generated beside them"""
    logger.debug("Pairing audio with video")
    from .pair_audio import pair_audio

    return pair_audio(**arguments)


@register_command(
    "crossfade_audio", implementation="dw.tasks.audio_utils.crossfade_audio"
)
def _handle_crossfade_audio(task, arguments, previous_pipelines):
    """Join audio tracks with an equal-power crossfade"""
    logger.debug("Crossfading audio")
    from .audio_utils import crossfade_audio

    return crossfade_audio(**arguments)


@register_command(
    "format_chat_message", implementation="dw.tasks.format_messages.format_chat_message"
)
def _handle_format_chat_message(task, arguments, previous_pipelines):
    """Format chat message for LLM input"""
    logger.debug("Formatting chat message")
    return format_chat_message(**arguments)


@register_command(
    "get_dict_value", implementation="dw.tasks.format_messages.get_dict_value"
)
def _handle_get_dict_value(task, arguments, previous_pipelines):
    """Extract value from dictionary"""
    logger.debug("Getting dictionary value")
    return get_dict_value(**arguments)


@register_command("upscale", implementation="dw.tasks.upscale.upscale_image")
def _handle_upscale(task, arguments, previous_pipelines):
    """Upscale an image using a spandrel-compatible super-resolution model"""
    logger.debug("Upscaling image")
    image = arguments.pop("image")
    model_name = arguments.pop("model_name")
    from .upscale import upscale_image

    return upscale_image(
        image, model_name, device=task.device_for(arguments), **arguments
    )


@register_command(
    "diffusion_upscale", implementation="dw.tasks.diffusion_upscale.diffusion_upscale"
)
def _handle_diffusion_upscale(task, arguments, previous_pipelines):
    """Upscale an image using a diffusion-based upscale pipeline"""
    logger.debug("Diffusion upscaling image")
    image = arguments.pop("image")
    from .diffusion_upscale import diffusion_upscale

    return diffusion_upscale(image, device=task.device_for(arguments), **arguments)


@register_command(
    "restore_faces", implementation="dw.tasks.restore_faces.restore_faces"
)
def _handle_restore_faces(task, arguments, previous_pipelines):
    """Restore faces in an image using a spandrel-compatible face restoration model"""
    logger.debug("Restoring faces")
    image = arguments.pop("image")
    model_name = arguments.pop("model_name")
    from .restore_faces import restore_faces

    return restore_faces(
        image, model_name, device=task.device_for(arguments), **arguments
    )


@register_command("segment", implementation="dw.tasks.segment.segment_image")
def _handle_segment(task, arguments, previous_pipelines):
    """Segment objects in an image using text prompt"""
    logger.debug("Segmenting image")
    image = arguments.pop("image")
    prompt = arguments.pop("prompt")
    from .segment import segment_image

    return segment_image(image, prompt, device=task.device_for(arguments), **arguments)


@register_command(
    "interpolate_frames",
    implementation="dw.tasks.interpolate_frames.interpolate_frames",
)
def _handle_interpolate_frames(task, arguments, previous_pipelines):
    """Interpolate video frames to increase frame rate"""
    logger.debug("Interpolating frames")
    video = arguments.pop("video")
    from .interpolate_frames import interpolate_frames

    return interpolate_frames(video, device=task.device_for(arguments), **arguments)


@register_command(
    "image_to_text", implementation="dw.tasks.image_to_text.image_to_text"
)
def _handle_image_to_text(task, arguments, previous_pipelines):
    """Generate text caption from an image"""
    logger.debug("Captioning image")
    image = arguments.pop("image")
    from .image_to_text import image_to_text

    return image_to_text(image, device=task.device_for(arguments), **arguments)


@register_command(
    "text_generation", implementation="dw.tasks.text_generation.generate_text"
)
def _handle_text_generation(task, arguments, previous_pipelines):
    """Generate text from a prompt using a local LLM"""
    logger.debug("Generating text")
    prompt = arguments.pop("prompt")
    from .text_generation import generate_text

    return generate_text(prompt, device=task.device_for(arguments), **arguments)


@register_command(
    "extract_sections", implementation="dw.tasks.text_sections.extract_sections"
)
def _handle_extract_sections(task, arguments, previous_pipelines):
    """Reduce generated text to a known set of labelled sections"""
    logger.debug("Extracting sections")
    from .text_sections import extract_sections

    return extract_sections(**arguments)


@register_command(
    "batch_decode_post_process",
    implementation="dw.tasks.format_messages.batch_decode_post_process",
    provided=("processor",),
)
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
    device = task.device_for(arguments)
    return process_image(
        arguments.pop("image"),
        task.command,
        device,
        arguments,
    )


def _handle_video_processing(task, arguments, previous_pipelines):
    """Handle video processing commands"""
    logger.debug("Processing video")
    device = task.device_for(arguments)
    return process_video(
        arguments.pop("video"),
        task.command,
        device,
        arguments,
    )


# Command names process_video (video_utils.py) accepts, with the function
# whose signature carries their arguments. video_utils dispatches via a plain
# if-chain, so keep this in sync with the branches in process_video().
# get_first/last_frame pin frame_index themselves, so it is 'provided'.
_VIDEO_PROCESSOR_INFO = {
    "get_frame": {
        "kind": "video_processor",
        "implementation": "dw.tasks.video_utils.get_frame",
        "provided": (),
    },
    "get_first_frame": {
        "kind": "video_processor",
        "implementation": "dw.tasks.video_utils.get_frame",
        "provided": ("frame_index",),
    },
    "get_last_frame": {
        "kind": "video_processor",
        "implementation": "dw.tasks.video_utils.get_frame",
        "provided": ("frame_index",),
    },
}
_VIDEO_PROCESSOR_COMMANDS = sorted(_VIDEO_PROCESSOR_INFO)


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

    def device_for(self, arguments):
        """Get the device this task runs on, consuming any override in its arguments.

        A task can pin itself to a device - a captioning model on the CPU while the GPU
        holds a pipeline, for instance. The argument is removed either way so it does
        not reach the command as a duplicate.

        Args:
            arguments: Arguments for this run of the task

        Returns:
            Device identifier the task should run on
        """
        return arguments.pop("device", self.device)

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
            # Cooperative cancellation reaches task steps too - without this
            # a cancel during a long task waits for the whole task to finish
            from ..events import get_context

            get_context().check_cancelled()

            # Look up command in registry
            if self.command in _COMMAND_REGISTRY:
                handler = _COMMAND_REGISTRY[self.command]
                return handler(self, arguments, previous_pipelines)

            # Not a registered command - check whether it names an image or
            # video processor instead. Imported lazily here to preserve
            # image_utils' lazy-import discipline for callers that never
            # touch image processing.
            from .image_utils import available_processors

            if self.command in available_processors():
                return _handle_image_processing(self, arguments, previous_pipelines)

            if self.command in _VIDEO_PROCESSOR_COMMANDS:
                return _handle_video_processing(self, arguments, previous_pipelines)

            # Unknown command - not in the registry, and not a known image or
            # video processor name either
            error_msg = (
                f"Unknown task command: '{self.command}'. "
                f"Registered commands: {sorted(_COMMAND_REGISTRY.keys())}. "
                f"Image processors: {available_processors()}. "
                f"Video processors: {_VIDEO_PROCESSOR_COMMANDS}"
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
