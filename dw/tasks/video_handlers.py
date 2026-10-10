"""The video and frame task commands' handlers.

Split from `dw/tasks/task.py` (#790), which had grown past the module-size
ceiling with every handler in one file. These are the commands that cut,
join, fit and re-frame video: joining shots (with the audio generated beside
them), frame runs and grids, long-video windows, fitting a clip to a model's
working size and back (#603), stabilizing, and the face-track crop and paste.

Each handler imports its implementation on use, as task.py's did - the video
modules pull in av and numpy, a cost a workflow that never edits video
should not pay at startup. `dw/tasks/task.py` imports this module, which is
what registers its commands (`docs/TASKS.md` *Adding a task*).
"""

import logging

from .registry import register_command
from ..task_domains import (
    FINITE,
    FIT_MODES,
    JOIN_WINDOWS_CURVES,
    NON_NEGATIVE,
    NON_POSITIVE,
    POSITIVE,
    fit_to_model_errors,
    join_windows_errors,
    window_video_errors,
)
from ..task_problems import (
    INGREDIENTS_FITS,
    INGREDIENTS_LAYOUTS,
    face_track_errors,
    ingredients_grid_errors,
    paste_face_track_errors,
)

logger = logging.getLogger("dw")


@register_command(
    "concat_videos",
    implementation="dw.tasks.concat_videos.concat_videos",
    domains={
        "trim_frames": NON_NEGATIVE,
        "crossfade_ms": NON_NEGATIVE,
        "audio_bleed_ms": NON_NEGATIVE,
        "audio_bleed_gain_db": FINITE,
        "match_levels_dbfs": FINITE,
        "seam_fade_ms": NON_NEGATIVE,
        "fps": POSITIVE,
        "sample_rate": POSITIVE,
    },
)
def _handle_concat_videos(task, arguments, previous_pipelines):
    """Concatenate videos - and the audio generated with them - into one"""
    logger.debug("Concatenating videos")
    from .concat_videos import concat_videos

    return concat_videos(**arguments)


@register_command(
    "dissolve_videos",
    implementation="dw.tasks.dissolve_videos.dissolve_videos",
    domains={
        "dissolve_frames": NON_NEGATIVE,
        "fade_in_frames": NON_NEGATIVE,
        "fade_out_frames": NON_NEGATIVE,
        "match_levels_dbfs": FINITE,
        "fps": POSITIVE,
    },
)
def _handle_dissolve_videos(task, arguments, previous_pipelines):
    """Join videos with cross-dissolves, fading the whole from and to a colour"""
    logger.debug("Dissolving videos")
    from .dissolve_videos import dissolve_videos

    return dissolve_videos(**arguments)


@register_command(
    "join_into_song",
    implementation="dw.tasks.join_into_song.join_into_song",
    domains={
        "cue_seconds": NON_NEGATIVE,
        "duck_delay_ms": NON_NEGATIVE,
        "duck_db": NON_POSITIVE,
        "duck_ramp_ms": NON_NEGATIVE,
        "dialogue_target_lufs": FINITE,
        "fps": POSITIVE,
    },
)
def _handle_join_into_song(task, arguments, previous_pipelines):
    """Join dialogue shots and song shots into one video over the unbroken song"""
    logger.debug("Joining dialogue into a song")
    from .join_into_song import join_into_song

    return join_into_song(**arguments)


@register_command("video_frames", implementation="dw.tasks.video_utils.frames_as_array")
def _handle_video_frames(task, arguments, previous_pipelines):
    """The frames of a generated video, as one array a later step can condition on"""
    logger.debug("Extracting video frames")
    from .video_utils import frames_as_array

    return frames_as_array(**arguments)


@register_command(
    "loop_frames",
    implementation="dw.tasks.video_utils.loop_frames",
    domains={"num_frames": POSITIVE},
    whole_numbers=("num_frames",),
)
def _handle_loop_frames(task, arguments, previous_pipelines):
    """Repeat a still or a short clip into a run of a given length"""
    logger.debug("Looping frames")
    from .video_utils import loop_frames

    return loop_frames(**arguments)


@register_command(
    "window_video",
    implementation="dw.tasks.windows.window_video",
    domains={
        "index": NON_NEGATIVE,
        "num_frames": POSITIVE,
        "overlap": NON_NEGATIVE,
        "fps": POSITIVE,
    },
    whole_numbers=("index", "num_frames", "overlap"),
    static_check=window_video_errors,
)
def _handle_window_video(task, arguments, previous_pipelines):
    """Cut one overlapping, fixed-length window out of a long video"""
    logger.debug("Cutting a video window")
    from .windows import window_video

    return window_video(**arguments)


@register_command(
    "fit_to_model",
    implementation="dw.tasks.fit.fit_to_model",
    summary=(
        "Fit a video into a model's working size and frame count - letterbox, "
        "stretch or crop - with a record restore_to_source reads to undo it."
    ),
    parameter_descriptions={
        "video": (
            "The source - an earlier step's video, or an asset:/output: "
            "reference. Its fps is kept; its soundtrack is not."
        ),
        "width": "The model's working width in pixels.",
        "height": "The model's working height in pixels.",
        "num_frames": (
            "The model's frame count. A longer source is cut to its first "
            "num_frames frames; a shorter one holds its last frame."
        ),
        "mode": (
            "letterbox (scale to fit, centred on black), stretch (resize to "
            "fill exactly) or crop (scale to fill, centre-crop)."
        ),
        "downscale": (
            "Fit into width/downscale x height/downscale instead, default 1 - "
            "2 when width/height are a 2x model's output size. Both must be "
            "divisible by it; the fit record's model size is the divided one."
        ),
    },
    domains={
        "width": POSITIVE,
        "height": POSITIVE,
        "num_frames": POSITIVE,
        "downscale": POSITIVE,
    },
    whole_numbers=("width", "height", "num_frames", "downscale"),
    choices={"mode": FIT_MODES},
    static_check=fit_to_model_errors,
)
def _handle_fit_to_model(task, arguments, previous_pipelines):
    """Fit a video into a model's working size and frame count"""
    logger.debug("Fitting a video to the model")
    from .fit import fit_to_model

    return fit_to_model(**arguments)


@register_command(
    "restore_to_source",
    implementation="dw.tasks.fit.restore_to_source",
    summary=(
        "Put a model's output for a fit_to_model video back at the source's "
        "size (times the model's scale, inferred) and frame count."
    ),
    parameter_descriptions={
        "video": (
            "The model's output for the fitted video, at the fitted size or a "
            "uniform multiple of it - 2x restores to twice the source."
        ),
        "fit": (
            "The record fit_to_model returned - previous_result:<step>.fit, or "
            "its saved .json. A crop fit restores only the part it kept, at "
            "the source's pixel density."
        ),
    },
    media_arguments=("fit",),
)
def _handle_restore_to_source(task, arguments, previous_pipelines):
    """Put a fitted video back at its source's size and length"""
    logger.debug("Restoring a video to its source")
    from .fit import restore_to_source

    return restore_to_source(**arguments)


@register_command(
    "join_windows",
    implementation="dw.tasks.windows.join_windows",
    domains={"num_frames": POSITIVE, "overlap": NON_NEGATIVE, "fps": POSITIVE},
    whole_numbers=("num_frames", "overlap"),
    choices={"curve": JOIN_WINDOWS_CURVES},
    static_check=join_windows_errors,
    media_arguments=("source",),
)
def _handle_join_windows(task, arguments, previous_pipelines):
    """Blend processed overlapping windows back into one video the source's length"""
    logger.debug("Joining video windows")
    from .windows import join_windows

    return join_windows(**arguments)


@register_command(
    "frame_grid",
    implementation="dw.tasks.video_utils.frame_grid",
    domains={"count": POSITIVE, "columns": POSITIVE, "tile_width": POSITIVE},
    whole_numbers=("count", "columns", "tile_width"),
)
def _handle_frame_grid(task, arguments, previous_pipelines):
    """Tile evenly sampled frames of a video into one contact-sheet image"""
    logger.debug("Building frame grid")
    from .video_utils import frame_grid

    return frame_grid(**arguments)


@register_command(
    "ingredients_grid",
    implementation="dw.tasks.image_utils.ingredients_grid",
    domains={
        "width": POSITIVE,
        "height": POSITIVE,
        "gap": NON_NEGATIVE,
        "max_images": POSITIVE,
    },
    whole_numbers=("width", "height", "gap", "max_images"),
    choices={"layout": INGREDIENTS_LAYOUTS, "fit": INGREDIENTS_FITS},
    static_check=ingredients_grid_errors,
)
def _handle_ingredients_grid(task, arguments, previous_pipelines):
    """Lay individual images out as one reference sheet"""
    logger.debug("Building ingredients grid")
    from .image_utils import ingredients_grid

    return ingredients_grid(**arguments)


@register_command(
    "stabilize_video",
    implementation="dw.tasks.stabilize.stabilize_video",
    domains={"smooth": FINITE},
    whole_numbers=("smooth",),
    media_arguments=("clip",),
)
def _handle_stabilize_video(task, arguments, previous_pipelines):
    """Remove a generated clip's accumulated framing drift"""
    logger.debug("Stabilizing video")
    from .stabilize import stabilize_video

    return stabilize_video(**arguments)


@register_command(
    "crop_face_track",
    implementation="dw.tasks.face_track.crop_face_track",
    consumes_device=True,
    summary=(
        "Follow one face through a clip and crop a steady square around it, "
        "with a per-frame record of the box and a face-detail strength."
    ),
    parameter_descriptions={
        "clip": (
            "Video to track - a file path, an asset:/output: reference, or an "
            "earlier step's video. A clip that records its shots resets the "
            "track at each boundary."
        ),
        "crop_size": "Side of every crop in pixels; a multiple of `multiple`.",
        "padding": (
            "Space added around the face on each side, as a fraction of its "
            "size (0 to 3)."
        ),
        "gate_full": (
            "Face width over frame width at or below which a frame's strength is 1."
        ),
        "gate_zero": (
            "Face width over frame width at or above which a frame's strength "
            "is 0; must exceed gate_full."
        ),
        "min_confidence": "Detector score below which a detection is ignored.",
        "modulus": (
            "The crop count is padded to modulus * n + remainder frames - the "
            "frame grid of the model the crops feed; 8 with remainder 1 is "
            "LTX's 8n+1."
        ),
        "remainder": "The padded count's remainder, from 0 to modulus - 1.",
        "multiple": (
            "crop_size must be a multiple of this - the model's frame-size "
            "step (32 for LTX)."
        ),
        "detector_repo": "Hugging Face repo holding the YuNet face detector.",
        "detector_file": "The .onnx file in detector_repo.",
    },
    domains={
        "crop_size": POSITIVE,
        "padding": NON_NEGATIVE,
        "gate_full": POSITIVE,
        "gate_zero": POSITIVE,
        "min_confidence": POSITIVE,
        "modulus": POSITIVE,
        "remainder": NON_NEGATIVE,
        "multiple": POSITIVE,
    },
    whole_numbers=("modulus", "remainder", "multiple"),
    static_check=face_track_errors,
    media_arguments=("clip",),
)
def _handle_crop_face_track(task, arguments, previous_pipelines):
    """Crop a steady square around the one face a clip follows"""
    logger.debug("Tracking a face")
    from .face_track import crop_face_track

    return crop_face_track(device=task.device_for(arguments), **arguments)


@register_command(
    "paste_face_track",
    implementation="dw.tasks.face_track.paste_face_track",
    summary=(
        "Blend repaired face crops back into the clip crop_face_track tracked, "
        "feathered and scaled by each frame's strength, keeping its audio."
    ),
    parameter_descriptions={
        "clip": (
            "The source video crop_face_track tracked - a file path, an "
            "asset:/output: reference, or an earlier step's video. Its audio, "
            "frame rate and shots are kept."
        ),
        "repaired": (
            "The crops after a face-detail pass, still padded to the count "
            "crop_face_track produced."
        ),
        "track": (
            "The track record crop_face_track returned - "
            "previous_result:<step>.track, or its saved .json."
        ),
        "feather": "Fraction of the paste's radius that fades out (0 to 1).",
        "color_match": (
            "Match each crop's mean colour to the source inside the mask "
            "before blending."
        ),
    },
    domains={"feather": NON_NEGATIVE},
    static_check=paste_face_track_errors,
    media_arguments=("clip", "repaired", "track"),
)
def _handle_paste_face_track(task, arguments, previous_pipelines):
    """Blend repaired face crops back into the clip they were cut from"""
    logger.debug("Pasting a face track")
    from .face_track import paste_face_track

    return paste_face_track(**arguments)
