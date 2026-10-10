"""The extension of a `video` argument, checked before the run when it can be.

`fetch_video` (`dw/argument_media.py`) loads every argument named `video` or
`*_video`, and every `{"media_type": "video", "location": ...}` reference
whatever it is named, as a video file - and refuses one whose extension is
not `ALLOWED_VIDEO_EXTENSIONS` there, at run time. A still image handed to
one of those arguments (`"video": "asset:sheet.png"`) validated clean and
then failed the job in seconds with "Video file extension not allowed:
.png", after `get_task("loop_frames")` had described the same argument as
taking "a still image, or frames in any shape a result carries" (#347).

Moved here, into `validation_errors`, for exactly the cases whose extension
is knowable without running anything: a literal file path, or an
`asset:`/`output:` reference whose name carries one - `resolve_path_references`
turns either into a real path before `fetch_video` ever sees it, and the
extension survives that unchanged. A `previous_result:` (or any reference
`expand_for_each` left unresolved) names no file yet and is left to the
existing run-time check, and a URL is left alone too - `fetch_video` never
gates a URL's extension, so refusing one here would refuse something the run
itself accepts.

`video_into_image_errors` is the neighbouring check (#813, after #509): a
pipeline's or composed workflow's `image` / `last_image` argument fed
`previous_result:<step>` where that earlier step declares a `video/*` result.
It resolves to a video at run time and the pipeline refuses it - after the
earlier members of a `for_each` have spent their GPU time. Only that literal,
fully static shape is checked; tasks take a video per frame and are left alone.
"""

from . import references
from .argument_media import is_media_reference
from .for_each import render_path
from .security import ALLOWED_IMAGE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS


def _is_video_key(key):
    return isinstance(key, str) and (key == "video" or key.endswith("_video"))


def _extension_problem(value):
    """Why this string's extension is not one `fetch_video` will accept, or
    None - including None for anything whose extension is not yet knowable."""
    if not isinstance(value, str):
        return None
    if value.startswith(("http://", "https://")):
        return None
    # Left to the run-time check: not yet resolved to anything an extension
    # can be read off, at the point validation walks the expanded definition
    if references.is_ref(references.UNRESOLVED, value):
        return None
    if references.is_ref((references.CONSTANT, references.PROMPT), value):
        return None
    ext = value.rsplit(".", 1)
    if len(ext) != 2 or not ext[1] or "/" in ext[1]:
        return None
    ext = f".{ext[1].lower()}"
    if ext in ALLOWED_VIDEO_EXTENSIONS:
        return None
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return (
            f"'{value}' is a still image, and a video argument loads video "
            f"files - pass it as "
            f'{{"media_type": "image", "location": "{value}"}} to load it as '
            f"a still, or reference a prior image step with 'previous_result:'"
        )
    return f"Video file extension not allowed: {ext}"


def _video_key_values(value, path):
    """Strings a video-named key hands to `fetch_video` - itself, or each
    element of a list of them. A dict here is a `media_type` reference and
    is walked by `_video_values` instead, same as `fetch_video` handles it."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _video_key_values(item, path + (index,))
    elif isinstance(value, dict):
        yield from _video_values(value, path)


def _video_values(value, path):
    """Every string this walk can attribute to a location `fetch_video`
    will load, paired with the path it sits at."""
    if isinstance(value, dict):
        if is_media_reference(value) and value.get("media_type") == "video":
            location = value.get("location")
            if isinstance(location, str):
                yield path + ("location",), location
            return
        for key, item in value.items():
            if _is_video_key(key):
                yield from _video_key_values(item, path + (key,))
            else:
                yield from _video_values(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _video_values(item, path + (index,))


def video_extension_errors(workflow_definition, source_indices=None):
    """Every video argument whose extension `fetch_video` will refuse, for
    every case that extension is knowable before the run, as
    [{path, message}].

    Walks the substituted, expanded definition, the same convention
    `reference_name_errors` and `task_argument_errors` follow: `source_indices`
    maps an expanded step back to the one the author wrote, and a path inside
    a `for_each` member names the member.
    """
    errors = []
    for _, step, _, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices
    ):
        for path, value in _video_values(step, ()):
            problem = _extension_problem(value)
            if problem is None:
                continue
            errors.append(
                {
                    "path": render_path(("steps", source) + path),
                    "message": f"{problem}{where}",
                }
            )
    return errors


IMAGE_KEYS = ("image", "last_image")


def video_into_image_errors(workflow_definition, source_indices=None):
    """Every pipeline or workflow-step `image`/`last_image` argument fed
    `previous_result:<step>` of an earlier step whose `result.content_type` is
    `video/*`, as [{path, message}]. Same walk and path convention as
    `video_extension_errors`."""
    errors = []
    video_steps = set()
    for _, step, _, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices
    ):
        for block_key in ("pipeline", "workflow"):
            block = step.get(block_key)
            arguments = block.get("arguments") if isinstance(block, dict) else None
            if not isinstance(arguments, dict):
                continue
            for key in IMAGE_KEYS:
                value = arguments.get(key)
                if not references.is_ref(references.PREVIOUS_RESULT, value):
                    continue
                name = references.ref_name(references.PREVIOUS_RESULT, value)
                if name not in video_steps:
                    continue
                errors.append(
                    {
                        "path": render_path(
                            ("steps", source, block_key, "arguments", key)
                        ),
                        "message": (
                            f"'{key}' is fed '{value}'{where}, but step '{name}' "
                            f"declares a video result and an image argument "
                            f"cannot load a video - take a still from it with "
                            f"get_last_frame (save: false) and reference that"
                        ),
                    }
                )
        result = step.get("result")
        content_type = result.get("content_type") if isinstance(result, dict) else None
        if (
            isinstance(content_type, str)
            and content_type.startswith("video/")
            and isinstance(step.get("name"), str)
        ):
            video_steps.add(step["name"])
    return errors


__all__ = ["video_extension_errors", "video_into_image_errors"]
