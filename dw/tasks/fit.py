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

`mode: "crop"` also reframes a clip to another aspect (16:9 to 9:16); see
"Reframe to another aspect" in docs/TASKS.md. A keeper H3 take whose motion
must stay can instead keep its whole picture and have the border generated:
`templates/minimax/reframe`.

Letterbox mode also returns a `mask` (#1066): a canvas-size "L" image, white
on the bars and black on the picture, drawn from the same `content_box`, so a
masked edit can fill the bars. `align` snaps that box to a pixel grid - a
model's mask-cell size, passed by the template, never known here - by scaling
the source to fill the snapped box and centre-cropping the few pixels that no
longer fit; the record's `source_box` says what was kept. `paste_into_box`
puts the source back inside the box afterwards. `fill` says what the bars
start as - black, the picture's edges extended and blurred more with distance
(`edge`), or the picture scaled to fill the canvas and heavily blurred
(`blur`) - so a masked edit refining them starts from something like the
scene rather than from black. Only the pixels outside the box change.

Neither task touches a soundtrack. The fitted video has none (its length is
the model's), and the restored one has none either - a template pairs the
source's track with `pair_audio` after the restore.

The sizes stay template variables rather than being picked here, so
validation's 32n and 8n+1 rules and the cost quote still see them.
"""

import logging

import numpy
import scipy.ndimage
import torch
import torch.nn.functional as F

from ..media_types import AudioVideo, FittedVideo, JsonRecord
from ..task_domains import (
    SOURCE_FPS,
    check_arguments,
    fit_align_problem,
    fit_downscale_problem,
    fit_fill_problem,
    fit_mode_problem,
    fit_record_problem,
    real_number,
    source_fps_problem,
    whole_number,
)
from .video_utils import frames_of, frames_as_array, load_audio_video

logger = logging.getLogger("dw")


def float_frames(video, command):
    """The video's frames as one float32 (frames, height, width, 3) array in
    [0, 1], without a round trip through uint8 for frames already float."""
    try:
        frames = frames_of(video)
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


def resize_frames(frames, width, height):
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


def fill_crop(source_width, source_height, width, height):
    """The centred part of a source_width x source_height frame that, scaled to
    fill width x height, covers it exactly - {x, y, w, h} in source pixels.
    Shared by crop mode, an aligned letterbox and paste_into_box, so the three
    cut the same pixels."""
    scale = max(width / source_width, height / source_height)
    kept_w = min(source_width, max(1, round(width / scale)))
    kept_h = min(source_height, max(1, round(height / scale)))
    x, y = _centred(source_width, kept_w), _centred(source_height, kept_h)
    return {"x": x, "y": y, "w": kept_w, "h": kept_h}


def _snap(length, align, limit):
    """`length` rounded to the nearest multiple of `align`, kept within one
    `align` and `limit` (itself a multiple)."""
    return min(limit, max(align, round(length / align) * align))


def _letterbox_mask(width, height, box):
    """The canvas-size "L" mask of a letterbox: 255 outside `box`, 0 inside."""
    from PIL import Image

    mask = numpy.full((height, width), 255, dtype=numpy.uint8)
    mask[box["y"] : box["y"] + box["h"], box["x"] : box["x"] + box["w"]] = 0
    return Image.fromarray(mask, mode="L")


# The edge fill's blur radius per pixel of distance from the box: the first
# row or column outside the box is the picture's own edge, and the bar
# softens as it runs away from it
EDGE_BLUR_PER_PIXEL = 0.5
# The blur fill works at 1/BLUR_FILL_FACTOR of the canvas, which is its blur
BLUR_FILL_FACTOR = 16


def _box_blur_rows(rows, radius):
    """`rows` (frames, n, 3) box-blurred along n with `radius`, edges held."""
    if radius < 1:
        return rows
    return scipy.ndimage.uniform_filter1d(
        rows, 2 * radius + 1, axis=1, mode="nearest"
    ).astype(numpy.float32)


def _paint_edge(fitted, box):
    """Paint the bars of `fitted` with the box's outermost rows and columns
    extended outward, each line box-blurred along itself with a radius that
    grows with its distance from the box. Pixels inside `box` are untouched."""
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    height, width = fitted.shape[1:3]
    # Left and right first, on the box's rows, so the top and bottom bars
    # extend full canvas-width rows that already carry the corners
    for column, edge, step in ((x - 1, x, -1), (x + w, x + w - 1, 1)):
        source = fitted[:, y : y + h, edge]
        distance = 1
        while 0 <= column < width:
            radius = int((distance - 1) * EDGE_BLUR_PER_PIXEL)
            fitted[:, y : y + h, column] = _box_blur_rows(source, radius)
            column += step
            distance += 1
    for row, edge, step in ((y - 1, y, -1), (y + h, y + h - 1, 1)):
        source = fitted[:, edge].copy()
        distance = 1
        while 0 <= row < height:
            radius = int((distance - 1) * EDGE_BLUR_PER_PIXEL)
            fitted[:, row] = _box_blur_rows(source, radius)
            row += step
            distance += 1


def _paint_blur(fitted, frames, box):
    """Paint `fitted` with `frames` scaled to fill the canvas, centre-cropped
    and heavily blurred, keeping the picture inside `box` as it is."""
    count, height, width = fitted.shape[:3]
    small_w = max(1, width // BLUR_FILL_FACTOR)
    small_h = max(1, height // BLUR_FILL_FACTOR)
    source_h, source_w = frames.shape[1:3]
    small = resize_frames(
        cut_box(frames, fill_crop(source_w, source_h, small_w, small_h)),
        small_w,
        small_h,
    )
    # A box blur along both axes of the small copy, then a smooth upscale
    small = scipy.ndimage.uniform_filter(
        small, size=(1, 5, 5, 1), mode="nearest"
    ).astype(numpy.float32)
    tensor = torch.from_numpy(numpy.ascontiguousarray(small)).permute(0, 3, 1, 2)
    backdrop = F.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )
    rows = slice(box["y"], box["y"] + box["h"])
    columns = slice(box["x"], box["x"] + box["w"])
    inside = fitted[:, rows, columns].copy()
    fitted[:] = backdrop.clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
    fitted[:, rows, columns] = inside


def fit_to_model(
    video,
    width,
    height,
    num_frames,
    mode="letterbox",
    downscale=1,
    align=1,
    fill="black",
):
    """Task command: fit a video into a model's working size and frame count.

    Every frame is resized into `width`x`height` (bicubic, antialiased) and the
    frame count set to exactly `num_frames`: a longer source is cut to its
    first `num_frames` frames, a shorter one holds its last frame. Hand the
    `fit` record to restore_to_source with the model's output to get the
    source's own size and length back. To reframe a clip to another aspect
    (16:9 to 9:16), use `mode: "crop"`; see "Reframe to another aspect" in
    docs/TASKS.md. For a keeper H3 take, `templates/minimax/reframe` keeps the
    whole picture letterboxed and generates the bars instead.

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
            to fill exactly) or "crop" (scale to fill, centre-crop; also the
            way to reframe a clip to another aspect - see TASKS.md "Reframe
            to another aspect"; a keeper H3 take can instead keep its whole
            picture with the border generated: templates/minimax/reframe)
        downscale: Divides `width` and `height` before fitting - 2 for a 2x
            model whose output size they are. Both must be divisible by it
        align: Letterbox only - snap the content box's position and size to
            multiples of this many pixels (a model's mask-cell size), filling
            the snapped box and centre-cropping the few source pixels past it.
            1 (the default) does not snap. The fitted size must be divisible
            by it
        fill: Letterbox only - what the bars start as: "black" (the
            default), "edge" (the picture's outermost rows and columns
            extended across the bars, blurred more the further they run from
            the box) or "blur" (the picture scaled to fill the canvas,
            centre-cropped and heavily blurred). The box, `mask` and `align`
            are the same for every fill; only pixels outside the box change

    Returns:
        {"video": the fitted frames as a FittedVideo - one float32 array in
        [0, 1] that a pipeline's `video` or a reference condition's `frames`
        takes as it is - at the source's fps and with no soundtrack, "fit": the record restore_to_source reads - mode,
        source_width/height/frames, model_width/height/frames, content_box
        (where the source sits in the model frame) and source_box (the part
        of the source kept; all of it unless mode is "crop"), each box
        {x, y, w, h}, and source_fps (the source's frame rate, null when it
        carries none), and in letterbox mode "mask": a width x height PIL "L"
        image, white (255) outside content_box and black (0) inside it}
    """
    command = "fit_to_model"
    width = whole_number(width, "width", command, required=True)
    height = whole_number(height, "height", command, required=True)
    num_frames = whole_number(num_frames, "num_frames", command, required=True)
    downscale = whole_number(downscale, "downscale", command, required=True)
    align = whole_number(align, "align", command, required=True)
    check_arguments(
        command,
        width=width,
        height=height,
        num_frames=num_frames,
        downscale=downscale,
        align=align,
    )
    for problem in (
        fit_mode_problem(mode),
        fit_downscale_problem(width, height, downscale),
    ):
        if problem is not None:
            raise ValueError(problem)
    width, height = width // downscale, height // downscale
    for problem in (
        fit_align_problem(mode, width, height, align),
        fit_fill_problem(mode, fill),
    ):
        if problem is not None:
            raise ValueError(problem)

    if isinstance(video, str):
        video = load_audio_video(video)
    fps = getattr(video, "fps", None)
    frames = float_frames(video, command)
    total, source_height, source_width = frames.shape[:3]

    # Frames first, so only the kept ones are resized
    picks = numpy.minimum(numpy.arange(num_frames), total - 1)
    frames = frames[picks]

    source_box = {"x": 0, "y": 0, "w": source_width, "h": source_height}
    content_box = {"x": 0, "y": 0, "w": width, "h": height}
    if mode == "stretch":
        fitted = resize_frames(frames, width, height)
    elif mode == "crop":
        source_box = fill_crop(source_width, source_height, width, height)
        fitted = resize_frames(cut_box(frames, source_box), width, height)
    else:
        scale = min(width / source_width, height / source_height)
        inner_w = min(width, max(1, round(source_width * scale)))
        inner_h = min(height, max(1, round(source_height * scale)))
        if align > 1:
            inner_w = _snap(inner_w, align, width)
            inner_h = _snap(inner_h, align, height)
            source_box = fill_crop(source_width, source_height, inner_w, inner_h)
            x = _centred(width, inner_w) // align * align
            y = _centred(height, inner_h) // align * align
        else:
            x, y = _centred(width, inner_w), _centred(height, inner_h)
        content_box = {"x": x, "y": y, "w": inner_w, "h": inner_h}
        fitted = numpy.zeros((num_frames, height, width, 3), dtype=numpy.float32)
        fitted[:, y : y + inner_h, x : x + inner_w] = resize_frames(
            cut_box(frames, source_box), inner_w, inner_h
        )
        if fill == "edge":
            _paint_edge(fitted, content_box)
        elif fill == "blur":
            _paint_blur(fitted, cut_box(frames, source_box), content_box)

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
            # restore_to_source's fps: "source" reads it back (#1079)
            "source_fps": float(fps) if fps else None,
        }
    )
    logger.info(
        f"fit_to_model: {source_width}x{source_height}x{total} -> "
        f"{width}x{height}x{num_frames} ({mode}, fill {fill}, content {content_box}, "
        f"source {source_box}, {max(0, num_frames - total)} held frames)"
    )
    result = {"video": FittedVideo(fitted, fps=fps), "fit": record}
    if mode == "letterbox":
        result["mask"] = _letterbox_mask(width, height, content_box)
    return result


def cut_box(frames, box):
    """`frames` cut to `box`, a view when the box is the whole frame."""
    return frames[:, box["y"] : box["y"] + box["h"], box["x"] : box["x"] + box["w"]]


def read_fit_record(fit, command="restore_to_source", argument="fit"):
    """The fit record, from the dict a step handed on or its saved .json,
    checked by `task_domains.fit_record_problem` - the one rule every reader
    of the record shares (#1066)."""
    if isinstance(fit, str):
        from ..locations import load_json_record

        fit = load_json_record(fit, None, f"a {argument} argument")
    problem = fit_record_problem(fit)
    if problem is not None:
        raise ValueError(
            f"{command} needs '{argument}' as the record fit_to_model returned "
            f"(previous_result:<step>.fit, or its saved .json): {problem}"
        )
    return fit


def restore_to_source(video, fit, fps=None):
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

    `fps` sets the rate the restored video carries. Omitted, it is the input
    video's. "source" is the rate the fit record says the source carried
    (`source_fps`), for a fixed-edit route whose output has to line up frame
    for frame with its source (#1079); a source that carried none, or a
    record saved without it, falls back to the input video's. A number is
    that rate. No frames are added or dropped for it.

    Args:
        video: The model's output for the fitted video - an earlier step's
            video, or an `asset:`/`output:` reference
        fit: The record fit_to_model returned - previous_result:<step>.fit,
            or its saved .json
        fps: The restored video's frame rate - omitted for the input video's,
            "source" for the source's, or a number above zero

    Returns:
        The restored video, float [0, 1], at that fps and with no
        soundtrack - pair the source's with pair_audio
    """
    command = "restore_to_source"
    problem = source_fps_problem(fps)
    if problem is not None:
        raise ValueError(problem)
    fit = read_fit_record(fit)
    if isinstance(video, str):
        video = load_audio_video(video)
    carried = getattr(video, "fps", None)
    if fps is None:
        fps = carried
    elif fps == SOURCE_FPS:
        fps = fit.get("source_fps") or carried
    else:
        fps = real_number(fps, "fps", command)
    frames = float_frames(video, command)
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
        restored = resize_frames(frames, out_w, out_h)
    else:
        out_w, out_h = scaled(fit["source_width"]), scaled(fit["source_height"])
        kept = fit.get("source_box")
        if mode == "letterbox" and kept is not None:
            # An aligned letterbox kept a few px less than the whole source
            out_w, out_h = scaled(kept["w"]), scaled(kept["h"])
        if mode == "letterbox":
            box = fit["content_box"]
            x0, y0 = round(box["x"] * scale), round(box["y"] * scale)
            x1 = min(video_width, round((box["x"] + box["w"]) * scale))
            y1 = min(video_height, round((box["y"] + box["h"]) * scale))
            frames = frames[:, y0:y1, x0:x1]
        restored = resize_frames(frames, out_w, out_h)

    logger.info(
        f"restore_to_source: {video_width}x{video_height}x{count} -> "
        f"{out_w}x{out_h}x{len(restored)} ({mode}, {scale:g}x, {fps} fps)"
    )
    return AudioVideo(restored.astype(numpy.float32), None, None, fps=fps)
