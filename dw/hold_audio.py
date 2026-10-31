"""A `hold_audio` argument, checked before the run when it can be.

`hold_audio` is the MiniMax-H3 soundtrack a step generates its video to
(dw/pipeline_processors/h3_hold.py). Only an H3 core-denoise pipeline has the
blocks that take it, and only audio can be held - so both are refused here,
where the answer is knowable without loading anything, rather than after a
checkpoint load:

- on a step whose pipeline is not a modular pipeline, or whose
  `from_pretrained` `workflow` is not one of H3's three (`t2va`, `fl2va`,
  `ref2va`). A modular pipeline loaded with no `workflow` is left to the
  run-time check, which asks the loaded block graph itself;
- a literal path, or an `asset:`/`output:` reference, whose name carries
  neither an audio nor a video extension (a bare word like `hello` among them),
  any value that is not a string or a mapping (a number), and any
  `{"media_type": ...}` reference - those load an image or video frames. A
  video file is held by its own soundtrack (a regional edit's `source_video`
  is the usual one), so one that resolves now and has no audio stream is
  refused too.

A `previous_result:` names no file yet and is left to the run-time check, the
same as `dw/video_extensions.py` leaves it.

`refine_strength`, the refine that re-denoises an upscaled take from a low sigma,
rides on the same blocks and is checked beside it: refused on a step that is not
H3, when it is not a number in (0, 1) - (0, 1] with a `source_video` - when the
step passes no `latents` to refine
or no `hold_audio` to keep its soundtrack, and when `num_inference_steps` is
below 2 - the refine schedule is that many points, so one point is no step.
An explicit `refine_sigmas` schedule is checked with it: a strictly decreasing
list from `refine_strength` to 0 with one point per `num_inference_steps`.
Those per-argument rules are `h3_rules.refine_problems`, which the run-time
check calls too; only the H3 test is here.

A regional edit - `source_video` with a `mask_video` or `mask_image` - rides on
them as well (dw/pipeline_processors/h3_mask.py): its rules are
`h3_rules.mask_problems`, called here and at the call alike. Where the files
resolve now, they are read here too, the way the guide-frame check reads its
videos (`dw/guides.py`): the source's and the mask's size and frame count
against the canvas (`h3_rules.media_problems`, which `Pipeline._with_mask` asks
again at the call), and a mask with no white pixel in it.
"""

from . import references
from .argument_media import is_media_reference
from .for_each import render_path
from .pipeline_processors.h3_rules import (
    HOLD_AUDIO_INPUT,
    MASK_IMAGE_INPUT,
    MASK_INPUTS,
    REFINE_SIGMAS_INPUT,
    REFINE_STRENGTH_INPUT,
    REFINE_TILES_INPUT,
    REGIONAL_EDIT_INPUTS,
    SOURCE_VIDEO_INPUT,
    edits_region,
    is_h3_step,
    mask_problems,
    media_problems,
    not_h3,
    refine_problems,
)
from .probe_paths import resolve_probe_path
from .security import ALLOWED_AUDIO_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS

# A video is held by its own soundtrack - a regional edit's source_video
HOLDABLE_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

UPSCALE_COMMAND = "upscale_h3_latents"


def _not_audio(value):
    """Why this `hold_audio` value is not audio, or None - including None for
    anything not yet knowable."""
    if is_media_reference(value):
        # fetch_media loads it as an image or a video before the call
        return (
            f"hold_audio holds a soundtrack, and this is a "
            f"'{value.get('media_type')}' reference - name the audio file "
            f"directly, or with 'asset:', 'output:' or 'previous_result:'"
        )
    if isinstance(value, dict):
        return None
    if not isinstance(value, str):
        # A number or a list is no file and no reference - the call refuses it
        return (
            f"hold_audio holds a soundtrack, and {value!r} is not one - name an "
            f"audio file directly, or with 'asset:', 'output:' or "
            f"'previous_result:'"
        )
    if value.startswith(("http://", "https://")):
        return None
    if references.is_ref(references.UNRESOLVED, value):
        return None
    if references.is_ref((references.CONSTANT, references.PROMPT), value):
        return None
    # A name with no extension - a bare word like 'hello' - is refused at the
    # call too, which checks the extension before it opens anything
    _, dot, ext = value.rpartition(".")
    if dot and ext and "/" not in ext and f".{ext.lower()}" in HOLDABLE_EXTENSIONS:
        return None
    return (
        f"hold_audio holds a soundtrack, and '{value}' is not an audio file "
        f"({', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}) or a video whose "
        f"track it takes ({', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))})"
    )


def _silent_video(value, base_dir, probe):
    """Why this `hold_audio` video holds nothing - it resolves now and has no
    audio stream - or None, including when it cannot be told yet."""
    if probe is None or not isinstance(value, str):
        return None
    _, dot, ext = value.rpartition(".")
    if f".{ext.lower()}" not in ALLOWED_VIDEO_EXTENSIONS:
        return None
    path = resolve_probe_path(value, base_dir, HOLD_AUDIO_INPUT)
    info = probe(path) if path is not None else None
    if isinstance(info, dict) and not info.get("sample_rate"):
        return (
            f"hold_audio takes the soundtrack of '{value}', and it has no audio "
            f"stream - pass a video with a soundtrack, or the audio file itself"
        )
    return None


def hold_audio_errors(
    workflow_definition, source_indices=None, base_dir=None, probe=None
):
    """Every `hold_audio` argument refused before the run, as [{path, message}].

    Walks the substituted, expanded definition, the convention
    `video_extension_errors` follows: `source_indices` maps an expanded step
    back to the one the author wrote, and a path inside a `for_each` member
    names the member. `probe` (`ValidationContext.probe`) reads a video that
    resolves against `base_dir` for its audio stream; without it a video is
    left to the run.
    """
    errors = []
    for _, _, pipeline, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict) or arguments.get(HOLD_AUDIO_INPUT) is None:
            continue
        path = render_path(("steps", source, "pipeline", "arguments", HOLD_AUDIO_INPUT))
        value = arguments[HOLD_AUDIO_INPUT]
        problem = _not_audio(value) or _silent_video(value, base_dir, probe)
        for problem in (not_h3(pipeline), problem):
            if problem is not None:
                errors.append({"path": path, "message": f"{problem}{where}"})
    return errors


def _upscaled_canvas(steps, latents):
    """The (width, height) an `upscale_h3_latents` step the `latents` argument
    names will produce, or None when it names something else or the task's size
    is not a literal pair."""
    name = references.ref_name(references.PREVIOUS_RESULT, latents)
    if name is None:
        return None
    name = name.partition(".")[0]
    for _, step, task, _, _ in references.iter_steps(steps, None, "task"):
        if step.get("name") != name:
            continue
        arguments = task.get("arguments")
        if task.get("command") != UPSCALE_COMMAND or not isinstance(arguments, dict):
            return None
        width, height = arguments.get("width"), arguments.get("height")
        if all(isinstance(v, int) and not isinstance(v, bool) for v in (width, height)):
            return width, height
        return None
    return None


_REFINE_INPUTS = (REFINE_STRENGTH_INPUT, REFINE_SIGMAS_INPUT, REFINE_TILES_INPUT)


def refine_strength_errors(workflow_definition, source_indices=None):
    """Every `refine_strength`, `refine_sigmas` or `refine_tiles` argument
    refused before the run, as [{path, message}] - walked the way
    `hold_audio_errors` walks. A problem about `refine_sigmas` or
    `refine_tiles` is at its path; the rest are at `refine_strength`'s."""
    errors = []
    steps = workflow_definition.get("steps")
    for _, _, pipeline, source, where in references.iter_steps(
        steps, source_indices, "pipeline"
    ):
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict) or all(
            arguments.get(name) is None for name in _REFINE_INPUTS
        ):
            continue
        paths = {
            name: render_path(("steps", source, "pipeline", "arguments", name))
            for name in _REFINE_INPUTS
        }
        problems = []
        for name in _REFINE_INPUTS:
            if arguments.get(name) is not None:
                problems.append((name, not_h3(pipeline, name)))
        canvas = _upscaled_canvas(steps, arguments.get("latents"))
        for problem in refine_problems(arguments, canvas):
            name = next(
                (n for n in _REFINE_INPUTS[1:] if problem.startswith(n)),
                REFINE_STRENGTH_INPUT,
            )
            problems.append((name, problem))
        for name, problem in problems:
            if problem is not None:
                errors.append({"path": paths[name], "message": f"{problem}{where}"})
    return errors


def _static(value):
    """`value` when it is a literal whole number, else None."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _mask_image_shape(path):
    """(1, height, width) and whether any pixel is white, for the mask image
    at `path` (already confined by `resolve_probe_path`), or None when it is
    no image."""
    from PIL import Image, UnidentifiedImageError

    from .pipeline_processors.h3_mask import binarized

    try:
        with Image.open(path) as image:
            mask = binarized(image)
    except (OSError, UnidentifiedImageError, ValueError):
        return None
    return mask.shape, bool(mask.any())


def _mask_video_white(path):
    """Whether any frame of the mask video at `path` has a white pixel,
    decoded until the first one does; None when it cannot be decoded."""
    import av

    from .media import iter_rgb_frames
    from .pipeline_processors.h3_mask import binarized

    try:
        for frame in iter_rgb_frames(path):
            if binarized(frame[None]).any():
                return True
    except (av.error.FFmpegError, OSError, ValueError, IndexError):
        return None
    return False


def _video_shape(name, value, base_dir, probe):
    """(problem, (frames, height, width), path) for a regional edit's video
    argument: why it is no video, its shape when the probe reads one, and the
    confined path it resolved to; Nones for what cannot be told yet."""
    path = resolve_probe_path(value, base_dir, name)
    info = probe(path) if path is not None else None
    if not isinstance(info, dict):
        return None, None, None
    if info.get("kind") != "video":
        return (
            f"{name} must be a video, and this is {info.get('kind') or 'unreadable'}",
            None,
            path,
        )
    frames, height, width = (info.get(k) for k in ("frame_count", "height", "width"))
    if not all(isinstance(v, int) and v > 0 for v in (frames, height, width)):
        return None, None, path
    return None, (frames, height, width), path


def _media_problems(arguments, base_dir, probe):
    """[(argument, problem)] for the regional edit's files that resolve now:
    the source's and the mask's size and frame count against the canvas - a
    canvas key the step leaves out is the source's, as at the call - and a
    mask with no white pixel."""
    if probe is None:
        return []
    problems = []
    width, height, num_frames = (
        arguments.get(k) for k in ("width", "height", "num_frames")
    )
    problem, shape, _ = _video_shape(
        SOURCE_VIDEO_INPUT, arguments.get(SOURCE_VIDEO_INPUT), base_dir, probe
    )
    if problem:
        problems.append((SOURCE_VIDEO_INPUT, problem))
    if shape is not None:
        width = shape[2] if width is None else width
        height = shape[1] if height is None else height
        num_frames = shape[0] if num_frames is None else num_frames
    canvas = (_static(width), _static(height), _static(num_frames))
    if shape is not None and None not in canvas[:2]:
        problems += [
            (SOURCE_VIDEO_INPUT, p)
            for p in media_problems(SOURCE_VIDEO_INPUT, shape, *canvas)
        ]
    name = next((n for n in MASK_INPUTS if arguments.get(n) is not None), None)
    if name is None:
        return problems
    white = None
    if name == MASK_IMAGE_INPUT:
        path = resolve_probe_path(arguments[name], base_dir, name)
        read = _mask_image_shape(path) if path is not None else None
        mask_shape, white = read if read is not None else (None, None)
    else:
        problem, mask_shape, path = _video_shape(name, arguments[name], base_dir, probe)
        if problem:
            problems.append((name, problem))
        elif path is not None:
            white = _mask_video_white(path)
    if white is False:
        problems.append(
            (
                name,
                f"{name} is all black, so nothing would regenerate - white marks "
                f"the region to regenerate",
            )
        )
    if mask_shape is not None and None not in canvas[:2]:
        problems += [(name, p) for p in media_problems(name, mask_shape, *canvas)]
    return problems


def mask_errors(workflow_definition, source_indices=None, base_dir=None, probe=None):
    """Every regional-edit argument (`source_video`, `mask_video`,
    `mask_image`, `mask_grow`) refused before the run, as [{path, message}].
    An argument rule is at the path of the first of them the step passes; a
    problem with a file - its size, frame count, or a mask with no white - is
    at that file's argument. A step not known to be H3 is a regional edit
    only with `source_video`: an inpaint pipeline's `mask_image` is its own.
    `probe` (`ValidationContext.probe`) reads the files that resolve against
    `base_dir`; without it only the argument rules are checked."""
    errors = []
    steps = workflow_definition.get("steps")
    for _, _, pipeline, source, where in references.iter_steps(
        steps, source_indices, "pipeline"
    ):
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict) or not edits_region(
            arguments, is_h3_step(pipeline)
        ):
            continue
        given = [n for n in REGIONAL_EDIT_INPUTS if arguments.get(n) is not None]
        problems = [(given[0], not_h3(pipeline, given[0]))]
        rules = mask_problems(arguments, chained=pipeline.get("chain") is not None)
        problems += [(given[0], problem) for problem in rules]
        if not rules:
            problems += _media_problems(arguments, base_dir, probe)
        for name, problem in problems:
            if problem is not None:
                path = render_path(("steps", source, "pipeline", "arguments", name))
                errors.append({"path": path, "message": f"{problem}{where}"})
    return errors


__all__ = ["hold_audio_errors", "mask_errors", "refine_strength_errors"]
