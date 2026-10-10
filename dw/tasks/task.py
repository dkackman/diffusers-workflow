import logging

from .. import resolve_device
from .image_ops import per_frame
from .qr_code import get_qrcode_image
from .image_utils import process_image
from .video_utils import process_video
from .gather import gather_images, gather_inputs, gather_videos
from .format_messages import (
    format_chat_message,
    batch_decode_post_process,
    get_dict_value,
)

# The registry lives in its own module so a task module can register its own
# handler (`beats`) without a cycle through this one; re-exported here under
# the names the rest of dw and the tests read
from .registry import _COMMAND_INFO, _COMMAND_REGISTRY, register_command  # noqa: F401
from ..task_domains import FINITE, coerce_arguments
from . import beats  # noqa: F401 - registers analyze_beats
from . import cuts  # noqa: F401 - registers plan_cuts
from . import trim  # noqa: F401 - registers trim_video

# The handlers by family, split out of this module when it outgrew the size
# ceiling (#790). Each registers its commands on import, so these lines are
# what puts them in the registry; the general-purpose commands - gather, text,
# messages, select, qr_code - and the image/video processor dispatch stay here
from . import audio_handlers  # noqa: F401 - audio and cut-assessment commands
from . import finish_handlers  # noqa: F401 - grade, sharpen, film_grain, apply_lut
from . import model_handlers  # noqa: F401 - the model-backed commands
from . import video_handlers  # noqa: F401 - video and frame commands

logger = logging.getLogger("dw")


def task_command_info(command_name):
    """Where a task command's argument schema lives: a dict with 'kind'
    ('command', 'image_processor' or 'video_processor'), 'implementation'
    (dotted path or None for free-form), 'provided', and 'returns'
    ('artifact', the default, or 'scalar' for a bare-number command like
    `judge` - missing entirely for an image/video processor, which is
    always artifact-shaped). Raises ValueError for a name that is not a
    task command at all."""
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
@register_command(
    "qr_code",
    implementation="dw.tasks.qr_code.get_qrcode_image",
    domains={"height": FINITE, "width": FINITE},
    whole_numbers=("height", "width"),
)
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


@register_command("compose_text", implementation="dw.tasks.compose_text.compose_text")
def _handle_compose_text(task, arguments, previous_pipelines):
    """Join parts written once into one block of text"""
    logger.debug("Composing text")
    from .compose_text import compose_text

    return compose_text(**arguments)


@register_command("select", implementation="dw.tasks.select.select")
def _handle_select(task, arguments, previous_pipelines):
    """Reduce a list of candidates to one by a deterministic rule"""
    logger.debug("Selecting")
    from .select import select

    return select(**arguments)


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
    return per_frame(
        arguments.pop("image"),
        lambda frame: process_image(frame, task.command, device, arguments),
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
# get_first/last_frame pin frame_index themselves, so it is 'provided'; they
# share get_frame's implementation and so would share its generic docstring
# summary too (#366) - 'summary' overrides that per command.
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
        "summary": "The first frame of a video, as a PIL image.",
    },
    "get_last_frame": {
        "kind": "video_processor",
        "implementation": "dw.tasks.video_utils.get_frame",
        "provided": ("frame_index",),
        "summary": "The last frame of a video, as a PIL image.",
    },
}
_VIDEO_PROCESSOR_COMMANDS = sorted(_VIDEO_PROCESSOR_INFO)


class Task:
    """
    Represents a task that can be executed as part of a workflow.
    Tasks are atomic operations like image processing, data gathering, or message formatting.
    """

    def __init__(self, task_definition, device, seed=None):
        """
        Initialize task with its configuration and device settings.

        Args:
            task_definition: Dictionary containing task configuration and parameters
            device: Device to run task on (e.g., 'cuda', 'mps', 'cpu')
            seed: The workflow/step-resolved seed, when one was set - None for
                an unseeded run. Only a handler that calls seed_for(arguments)
                consumes it; most tasks run no generator and ignore it
        """
        self.task_definition = task_definition
        self.device = device
        self.seed = seed
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
        return resolve_device(arguments.pop("device", self.device))

    def seed_for(self, arguments):
        """Get the seed this task run should use, consuming any override in
        its arguments.

        A task step reproducible the way a pipeline step is: the
        workflow/step-resolved seed by default, an explicit `seed` in the
        step's own arguments taking precedence - and the argument is removed
        either way so it does not reach the command as a duplicate. None
        means no seed was ever set anywhere, so the task should run exactly
        as it always did - unseeded and non-reproducible.

        Args:
            arguments: Arguments for this run of the task
        """
        return arguments.pop("seed", self.seed)

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
            from ..events import emit_phase, get_context

            get_context().check_cancelled()
            # A task reports nothing of its own - a captioning model loading
            # and decoding is otherwise indistinguishable from a hang
            emit_phase("task", detail=self.command)

            # Look up command in registry
            if self.command in _COMMAND_REGISTRY:
                handler = _COMMAND_REGISTRY[self.command]
                # Every numeric argument reaches the handler as a number, read
                # by the rule the static pass applies (#774)
                arguments = coerce_arguments(self.command, arguments)
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
