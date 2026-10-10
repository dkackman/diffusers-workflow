"""The finishing task commands' handlers: grade, sharpen, film grain, LUT.

Split from `dw/tasks/task.py` (#790), which had grown past the module-size
ceiling with every handler in one file. The four share one shape - a `media`
argument that is an image or a whole video, read by `_load_media` and run
frame by frame (#603) - and one rule: each checks its arguments again at run
time, since a value from a variable or an earlier step never met the static
pass. `dw/tasks/task.py` imports this module, which is what registers its
commands (`docs/TASKS.md` *Adding a task*).
"""

import logging

from ..events import emit_log
from .image_ops import per_frame
from .registry import register_command
from ..task_domains import (
    AT_LEAST_ONE,
    CHANNEL_LEVEL,
    CLOSED_UNIT,
    FINITE,
    NON_NEGATIVE,
    POSITIVE,
    SEED,
    UNIT,
    lut_errors,
)

logger = logging.getLogger("dw")


def _load_media(media):
    """An image-or-video argument as something per_frame takes.

    A string is a file path (an asset:/output: reference already resolved):
    a video extension is read with its audio, anything else as an image. A
    value that is not a string - a PIL Image, an AudioVideo, a frame list -
    is already loaded and comes back as itself. Shared by the finishing
    commands that take `media` (#603). An image file with transparency loads
    as RGBA, so the command puts its alpha back after the op (#775).
    """
    if not isinstance(media, str):
        return media
    import os

    from ..security import ALLOWED_VIDEO_EXTENSIONS
    from .video_utils import load_audio_video

    if os.path.splitext(media)[1].lower() in ALLOWED_VIDEO_EXTENSIONS:
        return load_audio_video(media)
    from ..argument_media import fetch_image

    return fetch_image(media, keep_alpha=True)


@register_command(
    "grade",
    implementation="dw.tasks.grade.grade_image",
    summary=(
        "Adjust exposure, contrast, tonal range (highlights, shadows, whites, "
        "blacks), clarity, white balance, saturation, fade and vignette of "
        "an image or a video."
    ),
    parameter_descriptions={
        "media": (
            "Image or video to grade. An image is a PIL Image; a video is a "
            "file path or an asset:/output: reference, read with its audio "
            "and graded frame by frame, keeping its frame rate and audio "
            "unchanged."
        ),
    },
    domains={
        "exposure": FINITE,
        "contrast": NON_NEGATIVE,
        "saturation": NON_NEGATIVE,
        "temperature": CLOSED_UNIT,
        "tint": CLOSED_UNIT,
        "highlights": CLOSED_UNIT,
        "shadows": CLOSED_UNIT,
        "whites": CLOSED_UNIT,
        "blacks": CLOSED_UNIT,
        "clarity": CLOSED_UNIT,
        "vignette": CLOSED_UNIT,
        "fade": UNIT,
    },
    media_arguments=("media",),
)
def _handle_grade(task, arguments, previous_pipelines):
    """Adjust the exposure, tone, white balance and colour of an image or a video"""
    logger.debug("Grading media")
    media = _load_media(arguments.pop("media"))
    import inspect

    from ..task_domains import check_arguments
    from .grade import grade_image

    # A value from a variable or an earlier step never met the static pass
    check_arguments("grade", **arguments)
    # Read off the signature, so a parameter added there cannot drop out of
    # the log (#603)
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(grade_image).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    applied = {
        name: arguments.get(name, default)
        for name, default in defaults.items()
        if arguments.get(name, default) != default
    }
    emit_log(
        f"grade: applied {applied}"
        if applied
        else "grade: no adjustment (all identity)",
        command="grade",
        **applied,
    )
    return per_frame(media, lambda frame: grade_image(frame, **arguments))


_MEDIA_DESCRIPTION = (
    "Image or video. An image is a PIL Image; a video is a file path or an "
    "asset:/output: reference, read with its audio and processed frame by "
    "frame, keeping its frame count, frame rate and audio unchanged."
)


@register_command(
    "sharpen",
    implementation="dw.tasks.finish.sharpen_image",
    summary="Sharpen an image or a video with an unsharp mask.",
    parameter_descriptions={"media": _MEDIA_DESCRIPTION},
    domains={"amount": NON_NEGATIVE, "radius": POSITIVE, "threshold": CHANNEL_LEVEL},
    media_arguments=("media",),
)
def _handle_sharpen(task, arguments, previous_pipelines):
    """Sharpen an image or a video with an unsharp mask"""
    logger.debug("Sharpening media")
    media = _load_media(arguments.pop("media"))
    from ..task_domains import check_arguments
    from .finish import sharpen_image

    # A value from a variable or an earlier step never met the static pass
    check_arguments("sharpen", **arguments)
    return per_frame(media, lambda frame: sharpen_image(frame, **arguments))


@register_command(
    "film_grain",
    implementation="dw.tasks.finish.film_grain",
    summary=(
        "Add seeded film grain to an image or a video, strongest in the "
        "midtones; every video frame gets different grain."
    ),
    parameter_descriptions={
        "media": _MEDIA_DESCRIPTION,
        "seed": (
            "The grain's seed. Defaults to the workflow's or step's seed, "
            "which a run always has (a random one when the workflow names "
            "none, recorded in the manifest), so a rerun reproduces the grain."
        ),
    },
    domains={"amount": UNIT, "size": AT_LEAST_ONE, "chroma": UNIT, "seed": SEED},
    media_arguments=("media",),
)
def _handle_film_grain(task, arguments, previous_pipelines):
    """Add film grain to an image or a video, reproducibly from a seed"""
    logger.debug("Adding film grain")
    media = _load_media(arguments.pop("media"))
    seed = task.seed_for(arguments)
    from ..task_domains import check_arguments
    from .finish import film_grain

    check_arguments("film_grain", seed=seed, **arguments)
    emit_log(f"film_grain: seed {seed}", command="film_grain", seed=seed)
    return film_grain(media, seed=seed, **arguments)


@register_command(
    "apply_lut",
    implementation="dw.tasks.lut.apply_lut",
    summary=(
        "Colour an image or a video through a 3D lookup table - from a .cube "
        "file or built from a palette - blended with the original by strength."
    ),
    parameter_descriptions={"media": _MEDIA_DESCRIPTION},
    domains={"strength": UNIT},
    static_check=lut_errors,
    media_arguments=(
        "media",
        "lut",
    ),
)
def _handle_apply_lut(task, arguments, previous_pipelines):
    """Apply a .cube or palette 3D lookup table to an image or a video"""
    logger.debug("Applying a LUT")
    media = _load_media(arguments.pop("media"))
    from ..task_domains import check_arguments
    from .lut import apply_lut, lookup_for

    lut = arguments.pop("lut", None)
    palette = arguments.pop("palette", None)
    # A value from a variable or an earlier step never met the static pass
    check_arguments("apply_lut", **arguments)
    # Read or built once, not once per video frame
    lookup = lookup_for(lut, palette)
    return per_frame(media, lambda frame: apply_lut(frame, lookup, **arguments))
