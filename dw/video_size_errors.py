"""A `dissolve_videos`/`concat_videos` input whose frame size disagrees with
its siblings, refused before the run when the sizes are already knowable.

Both tasks join clips frame-by-frame (`check_same_frame_size`,
`dw/tasks/video_utils.py`) and raise once every input has been decoded: "video
N is WxH, video M is WxH". Unlike a sample-rate mismatch (#108/#287), which is
auto-resampled with a warning, there is no reconciliation for a size mismatch
- Don declined a resize/fit argument on the join tasks (#504, #512) - so the
only thing to move earlier is the refusal itself.

Moved here, into `validation_errors`, for exactly the cases a size is
knowable without running anything: an `asset:`/`output:` reference, or a
literal file path inside the directories the run may read
(`dw/probe_paths.py`). A `previous_result:` (or any reference `expand_for_each`
left unresolved), a `variable:`/`item:`/`gather:` reference, or a
`{"location": ...}` dict, names no size yet and is left to the existing
run-time check - silence there is correct, not a gap, since the size is not
known until the step that produces it runs.
"""

from .for_each import MEMBER_SEPARATOR, render_path
from .media_info import probe_media
from .probe_paths import resolve_probe_path

_CHECKED_COMMANDS = ("dissolve_videos", "concat_videos")


def _frame_size(path):
    """The (width, height) `dissolve_videos`/`concat_videos` would see for
    this file, or None when it cannot be probed or carries no video stream."""
    info = probe_media(path)
    if info is None or info.get("kind") != "video":
        return None
    width, height = info.get("width"), info.get("height")
    return (width, height) if width and height else None


def video_size_errors(workflow_definition, source_indices=None, base_dir=None):
    """Every `dissolve_videos`/`concat_videos` step whose statically-resolvable
    inputs already disagree in frame size, as [{path, message}].

    Walks the substituted, expanded definition, the same convention
    `dissolve_frame_errors` and `task_argument_errors` follow: `source_indices`
    maps an expanded step back to the one the author wrote, and a path inside
    a `for_each` member names the member.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        task = step.get("task")
        if not isinstance(task, dict) or task.get("command") not in _CHECKED_COMMANDS:
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
            size = _frame_size(path)
            if size is None:
                continue
            sizes[video_index] = size
        if len(set(sizes.values())) < 2:
            continue

        first_index = next(iter(sizes))
        first_size = sizes[first_index]
        problems = [f"video {first_index} is {first_size[0]}x{first_size[1]}"]
        for video_index, size in sizes.items():
            if video_index == first_index or size == first_size:
                continue
            problems.append(f"video {video_index} is {size[0]}x{size[1]}")

        source = (
            source_indices[index]
            if source_indices is not None and index < len(source_indices)
            else index
        )
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        errors.append(
            {
                "path": render_path(("steps", source, "task", "arguments", "videos")),
                "message": f"{command} needs every video at one size: "
                f"{', '.join(problems)}{where}",
            }
        )
    return errors


__all__ = ["video_size_errors"]
