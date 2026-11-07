"""A `dissolve_videos`/`concat_videos` input whose frame size disagrees with
its siblings, refused before the run when the sizes are already knowable.

Both tasks join clips frame-by-frame (`check_same_frame_size`,
`dw/tasks/video_utils.py`) and raise once every input has been decoded: "video
N is WxH, video M is WxH". Unlike a sample-rate mismatch (#108/#287), which is
auto-resampled with a warning, neither join task resizes a mismatched video
for you - Don declined a resize/fit argument on the join tasks themselves
(#504, #512) - so the fix is upstream of the join: `video_frames` to get the
odd video's frames, `resize_rescale` to the target size (`resize_center_crop`
squares the frame first and then stretches it, distorting a non-square
target), then `pair_audio(fit="video")` to put its soundtrack back before
passing it to `dissolve_videos`/`concat_videos` (#551). The served error
message names the route too, since a consumer runs into this at validate,
not by reading this module. The only thing to move earlier is the refusal
itself.

Moved here, into `validation_errors`, for exactly the cases a size is
knowable without running anything: an `asset:`/`output:` reference, a
literal file path inside the directories the run may read
(`dw/probe_paths.py`), or a `{"location": ...}` dict wrapping either (the
same idiom `load_audio_video` accepts, #510) - unwrapped before
`resolve_probe_path` sees it, since that resolver only takes a string. A
`previous_result:` (or any reference `expand_for_each` left unresolved) or a
`variable:`/`item:`/`gather:` reference names no size yet and is left to the
existing run-time check - silence there is correct, not a gap, since the
size is not known until the step that produces it runs.
"""

from .for_each import render_path
from .media import probe_metadata
from .probe_paths import resolve_probe_path
from .references import iter_steps
from .task_domains import as_number, frame_size_error
from .task_problems import stills_memory_problem, stills_to_video_problems

_CHECKED_COMMANDS = ("dissolve_videos", "concat_videos")


def _frame_size(path, probe):
    """The (width, height) `dissolve_videos`/`concat_videos` would see for
    this file, or None when it cannot be probed or carries no video stream."""
    info = probe(path)
    if info is None or info.get("kind") != "video":
        return None
    width, height = info.get("width"), info.get("height")
    return (width, height) if width and height else None


def video_size_errors(
    workflow_definition, source_indices=None, base_dir=None, *, probe=probe_metadata
):
    """Every `dissolve_videos`/`concat_videos` step whose statically-resolvable
    inputs already disagree in frame size, as [{path, message}].

    Walks the substituted, expanded definition, the same convention
    `dissolve_frame_errors` and `task_argument_errors` follow: `source_indices`
    maps an expanded step back to the one the author wrote, and a path inside
    a `for_each` member names the member.

    `probe` defaults to the metadata-only `probe_metadata` (B9); see
    `dissolve_frame_errors` for why and for the memoizing-wrapper contract.
    """
    errors = []
    for _, _, task, source, where in iter_steps(
        workflow_definition.get("steps"), source_indices, "task"
    ):
        if task.get("command") not in _CHECKED_COMMANDS:
            continue
        command = task["command"]
        arguments = task.get("arguments")
        if not isinstance(arguments, dict):
            continue
        videos = arguments.get("videos")
        if not isinstance(videos, list) or len(videos) < 2:
            continue

        sizes = {}
        for video_index, video in enumerate(videos):
            path = resolve_probe_path(video, base_dir, "a video argument")
            if path is None:
                continue
            size = _frame_size(path, probe)
            if size is None:
                continue
            sizes[video_index] = size
        sentence = frame_size_error(command, sizes)
        if sentence is None:
            continue

        fit_width, fit_height = next(iter(sizes.values()))
        errors.append(
            {
                "path": render_path(("steps", source, "task", "arguments", "videos")),
                "message": f"{sentence}{where} - fit the odd one with "
                f"video_frames → resize_rescale(width={fit_width}, "
                f'height={fit_height}) → pair_audio(fit="video")',
            }
        )
    return errors


def _image_size(path):
    """A still's (width, height) from its header alone, or None."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def _stills_seconds(durations, count):
    """The seconds each of `count` stills is held, or None when a duration is
    not a literal number yet."""
    items = list(durations) if isinstance(durations, (list, tuple)) else None
    numbers = [as_number(item) for item in (items or [durations] * count)]
    return None if any(n is None for n in numbers) else numbers


def stills_memory_errors(workflow_definition, source_indices=None, base_dir=None):
    """Every `stills_to_video` step whose frames would pass the in-memory
    budget, as [{path, message}] - the refusal the run makes before it
    allocates (`stills_memory_problem`), moved to validate (#1199).

    Knowable when `durations` and `fps` are literal numbers after
    substitution and every still is a file (`asset:`, `output:`, a path, or a
    `{"location": ...}` dict) whose header gives its size; the stills are
    one size or the run refuses them, so the first one's is the clip's. A
    still from an earlier step, a reference left unresolved, or a duration
    rule `stills_to_video_problems` already refuses is left alone.
    """
    errors = []
    for _, _, task, source, where in iter_steps(
        workflow_definition.get("steps"), source_indices, "task"
    ):
        if task.get("command") != "stills_to_video":
            continue
        arguments = task.get("arguments")
        if not isinstance(arguments, dict):
            continue
        images = arguments.get("images")
        images = [images] if isinstance(images, (str, dict)) else images
        if not isinstance(images, list) or not images:
            continue
        durations, fps = arguments.get("durations"), arguments.get("fps", 24)
        rate = as_number(fps)
        if rate is None or stills_to_video_problems(durations, fps, len(images)):
            continue
        seconds = _stills_seconds(durations, len(images))
        paths = [resolve_probe_path(i, base_dir, "an image argument") for i in images]
        if seconds is None or None in paths:
            continue
        size = _image_size(paths[0])
        if size is None:
            continue
        message = stills_memory_problem(seconds, rate, *size)
        if message is not None:
            errors.append(
                {
                    "path": render_path(
                        ("steps", source, "task", "arguments", "images")
                    ),
                    "message": f"{message}{where}",
                }
            )
    return errors


__all__ = ["stills_memory_errors", "video_size_errors"]
