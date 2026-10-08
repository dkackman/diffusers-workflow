"""Fit a source video to a model's working size, and restore it after (#602).

A video-to-video template's `width`, `height` and `num_frames` used to have to
match the source clip, and the two 2x LTX templates disagreed about a source
that did not: `upscale-clip` centre-cropped it (diffusers' IC reference
preprocess) and `refine-clip` stretched it (the latent upsampler). The pair
here makes those variables the model's working size instead.

`fit_to_model` resizes every frame into `width`x`height` - `stretch`, `crop`
(scale to fill, centre-crop) or `letterbox` (scale to fit, centre on black) -
and sets the frame count to exactly `num_frames`, cutting a long source and
holding the last frame of a short one. It returns the frames and a `fit`
record of what it did. `restore_to_source` reads that record to put the
model's output back at the source's own size and length: the scale is
whatever the model did to the fitted frame (2x for the upscalers), inferred
from the output's size, so no `scale` argument is needed.

Neither task touches a soundtrack. The fitted video has none (its length is
the model's), and the restored one has none either - a template pairs the
source's track with `pair_audio` after the restore.

The sizes stay template variables rather than being picked here, so
validation's 32n and 8n+1 rules and the cost quote still see them.
"""

import json
import logging

import numpy
import torch
import torch.nn.functional as F

from ..media_types import AudioVideo, FittedVideo, JsonRecord
from ..task_domains import (
    check_arguments,
    fit_downscale_problem,
    fit_mode_problem,
    whole_number,
)
from .video_utils import _frames_of, frames_as_array, load_audio_video

logger = logging.getLogger("dw")

_BOX_KEYS = ("x", "y", "w", "h")
_RECORD_INTS = (
    "source_width",
    "source_height",
    "source_frames",
    "model_width",
    "model_height",
    "model_frames",
)


def _float_frames(video, command):
    """The video's frames as one float32 (frames, height, width, 3) array in
    [0, 1], without a round trip through uint8 for frames already float."""
    try:
        frames = _frames_of(video)
    except TypeError:
        raise ValueError(
            f"{command} needs 'video' as a video, not {type(video).__name__}"
        )
    if len(frames) == 0:
        raise ValueError(f"{command} needs 'video' with at least one frame")
    if (
        isinstance(frames, numpy.ndarray)
        and frames.ndim == 4
        and frames.shape[-1] == 3
        and frames.dtype != numpy.uint8
    ):
        return numpy.clip(frames.astype(numpy.float32), 0.0, 1.0)
    return frames_as_array(frames).astype(numpy.float32) / 255.0


def _resize(frames, width, height):
    """Bicubic, antialiased resize of a (frames, h, w, 3) float array."""
    if frames.shape[1:3] == (height, width):
        return frames
    tensor = torch.from_numpy(numpy.ascontiguousarray(frames)).permute(0, 3, 1, 2)
    resized = F.interpolate(
        tensor,
        size=(height, width),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    )
    return resized.clamp(0.0, 1.0).permute(0, 2, 3, 1).contiguous().numpy()


def _centred(outer, inner):
    return (outer - inner) // 2


def fit_to_model(video, width, height, num_frames, mode="letterbox", downscale=1):
    """Task command: fit a video into a model's working size and frame count.

    Every frame is resized into `width`x`height` (bicubic, antialiased) and the
    frame count set to exactly `num_frames`: a longer source is cut to its
    first `num_frames` frames, a shorter one holds its last frame. Hand the
    `fit` record to restore_to_source with the model's output to get the
    source's own size and length back.

    `downscale` fits into `width`/downscale x `height`/downscale instead, so
    a template can keep `width`/`height` as a 2x model's output size and
    still fit the source to the model's input; the record's model size is
    the divided one.

    Args:
        video: The source - an earlier step's video, or an `asset:`/`output:`
            reference
        width: The model's working width in pixels
        height: The model's working height in pixels
        num_frames: The model's frame count, e.g. an 8n+1 LTX length
        mode: "letterbox" (scale to fit, centred on black), "stretch" (resize
            to fill exactly) or "crop" (scale to fill, centre-crop)
        downscale: Divides `width` and `height` before fitting - 2 for a 2x
            model whose output size they are. Both must be divisible by it

    Returns:
        {"video": the fitted frames as a FittedVideo - one float32 array in
        [0, 1] that a pipeline's `video` or a reference condition's `frames`
        takes as it is - at the source's fps and with no soundtrack, "fit": the record restore_to_source reads - mode,
        source_width/height/frames, model_width/height/frames, content_box
        (where the source sits in the model frame) and source_box (the part
        of the source kept; all of it unless mode is "crop"), each box
        {x, y, w, h}}
    """
    command = "fit_to_model"
    width = whole_number(width, "width", command, required=True)
    height = whole_number(height, "height", command, required=True)
    num_frames = whole_number(num_frames, "num_frames", command, required=True)
    downscale = whole_number(downscale, "downscale", command, required=True)
    check_arguments(
        command,
        width=width,
        height=height,
        num_frames=num_frames,
        downscale=downscale,
    )
    for problem in (
        fit_mode_problem(mode),
        fit_downscale_problem(width, height, downscale),
    ):
        if problem is not None:
            raise ValueError(problem)
    width, height = width // downscale, height // downscale

    if isinstance(video, str):
        video = load_audio_video(video)
    fps = getattr(video, "fps", None)
    frames = _float_frames(video, command)
    total, source_height, source_width = frames.shape[:3]

    # Frames first, so only the kept ones are resized
    picks = numpy.minimum(numpy.arange(num_frames), total - 1)
    frames = frames[picks]

    source_box = {"x": 0, "y": 0, "w": source_width, "h": source_height}
    content_box = {"x": 0, "y": 0, "w": width, "h": height}
    if mode == "stretch":
        fitted = _resize(frames, width, height)
    elif mode == "crop":
        scale = max(width / source_width, height / source_height)
        kept_w = min(source_width, max(1, round(width / scale)))
        kept_h = min(source_height, max(1, round(height / scale)))
        x, y = _centred(source_width, kept_w), _centred(source_height, kept_h)
        source_box = {"x": x, "y": y, "w": kept_w, "h": kept_h}
        fitted = _resize(frames[:, y : y + kept_h, x : x + kept_w], width, height)
    else:
        scale = min(width / source_width, height / source_height)
        inner_w = min(width, max(1, round(source_width * scale)))
        inner_h = min(height, max(1, round(source_height * scale)))
        x, y = _centred(width, inner_w), _centred(height, inner_h)
        content_box = {"x": x, "y": y, "w": inner_w, "h": inner_h}
        fitted = numpy.zeros((num_frames, height, width, 3), dtype=numpy.float32)
        fitted[:, y : y + inner_h, x : x + inner_w] = _resize(frames, inner_w, inner_h)

    record = JsonRecord(
        {
            "mode": mode,
            "source_width": source_width,
            "source_height": source_height,
            "source_frames": total,
            "model_width": width,
            "model_height": height,
            "model_frames": num_frames,
            "content_box": content_box,
            "source_box": source_box,
        }
    )
    logger.info(
        f"fit_to_model: {source_width}x{source_height}x{total} -> "
        f"{width}x{height}x{num_frames} ({mode}, content {content_box}, "
        f"{max(0, num_frames - total)} held frames)"
    )
    return {"video": FittedVideo(fitted, fps=fps), "fit": record}


def _read_fit(fit):
    """The fit record, from the dict a step handed on or its saved .json,
    checked field by field."""
    if isinstance(fit, str):
        from ..locations import validate_media_path
        from ..security import (
            ALLOWED_JSON_EXTENSIONS,
            validate_file_extension,
            validate_json_size,
        )

        path = validate_media_path(fit, None, "a fit argument")
        validate_file_extension(path, ALLOWED_JSON_EXTENSIONS)
        validate_json_size(path)
        with open(path, encoding="utf-8") as handle:
            fit = json.load(handle)

    def refuse(detail):
        return ValueError(
            f"restore_to_source needs 'fit' as the record fit_to_model returned "
            f"(previous_result:<step>.fit, or its saved .json): {detail}"
        )

    if not isinstance(fit, dict):
        raise refuse(f"got a {type(fit).__name__}")
    problem = fit_mode_problem(fit.get("mode"))
    if problem is not None:
        raise refuse(f"its 'mode' is {fit.get('mode')!r}")
    for key in _RECORD_INTS:
        value = fit.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise refuse(f"its '{key}' is {value!r}, not a whole number above zero")
    box_name = "source_box" if fit["mode"] == "crop" else "content_box"
    box = fit.get(box_name)
    if not isinstance(box, dict) or any(
        isinstance(box.get(k), bool) or not isinstance(box.get(k), int)
        for k in _BOX_KEYS
    ):
        raise refuse(f"its '{box_name}' is {box!r}, not a box of whole x, y, w, h")
    if box["x"] < 0 or box["y"] < 0 or box["w"] < 1 or box["h"] < 1:
        raise refuse(f"its '{box_name}' {box} is not a box inside the frame")
    bound_w, bound_h = (
        (fit["source_width"], fit["source_height"])
        if box_name == "source_box"
        else (fit["model_width"], fit["model_height"])
    )
    if box["x"] + box["w"] > bound_w or box["y"] + box["h"] > bound_h:
        raise refuse(
            f"its '{box_name}' {box} runs outside the {bound_w}x{bound_h} frame"
        )
    return fit


def restore_to_source(video, fit):
    """Task command: put a model's output back at its source's size and length.

    The scale is the model output's size over the fitted size, and must be
    the same on both axes - a 2x upscaler's output restores to twice the
    source. Letterbox crops off the bars and resizes the content to the
    source's size times that scale; stretch resizes straight to it. Crop
    cannot bring back the edges it cut, so it returns the kept part of the
    source (`source_box`) at the source's pixel density times the scale -
    exact in density, smaller than the source on the cropped axis. Frames
    are trimmed to the source's count, removing fit_to_model's held frames;
    an output shorter than that is returned as it is.

    Args:
        video: The model's output for the fitted video - an earlier step's
            video, or an `asset:`/`output:` reference
        fit: The record fit_to_model returned - previous_result:<step>.fit,
            or its saved .json

    Returns:
        The restored video, float [0, 1], at the input video's fps and with
        no soundtrack - pair the source's with pair_audio
    """
    command = "restore_to_source"
    fit = _read_fit(fit)
    if isinstance(video, str):
        video = load_audio_video(video)
    fps = getattr(video, "fps", None)
    frames = _float_frames(video, command)
    count, video_height, video_width = frames.shape[:3]

    model_width, model_height = fit["model_width"], fit["model_height"]
    # Compared as cross products so a non-integer scale is still exact
    if video_width * model_height != video_height * model_width:
        raise ValueError(
            f"restore_to_source needs 'video' at a uniform multiple of the "
            f"fitted size: the video is {video_width}x{video_height} but the "
            f"fit record's model size is {model_width}x{model_height} "
            f"({video_width / model_width:g}x across, "
            f"{video_height / model_height:g}x down)"
        )
    scale = video_width / model_width

    frames = frames[: min(fit["source_frames"], count)]

    def scaled(value):
        return max(1, round(value * scale))

    mode = fit["mode"]
    if mode == "crop":
        box = fit["source_box"]
        out_w, out_h = scaled(box["w"]), scaled(box["h"])
        restored = _resize(frames, out_w, out_h)
    else:
        out_w, out_h = scaled(fit["source_width"]), scaled(fit["source_height"])
        if mode == "letterbox":
            box = fit["content_box"]
            x0, y0 = round(box["x"] * scale), round(box["y"] * scale)
            x1 = min(video_width, round((box["x"] + box["w"]) * scale))
            y1 = min(video_height, round((box["y"] + box["h"]) * scale))
            frames = frames[:, y0:y1, x0:x1]
        restored = _resize(frames, out_w, out_h)

    logger.info(
        f"restore_to_source: {video_width}x{video_height}x{count} -> "
        f"{out_w}x{out_h}x{len(restored)} ({mode}, {scale:g}x)"
    )
    return AudioVideo(restored.astype(numpy.float32), None, None, fps=fps)
