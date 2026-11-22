"""Paste a source clip back into a box of a generated one (#1066).

The other half of a letterbox outpaint: `fit_to_model` letterboxes the source
and hands its `mask` to a masked edit, which generates the bars - and puts the
picture inside the box through the model's VAE and back. `paste_into_box`
returns the original pixels there: every frame of `source` is scaled into the
box, blended over `generated` with a linear ramp `feather` px wide inside the
box's edges, and `generated` is kept as it is outside it. A box edge flush with
the frame's edge has nothing generated beyond it, so it gets no ramp.

The result is `generated`'s frames, one for one, so its soundtrack, fps and
shots come through. The rules a literal workflow can break are
`task_problems.paste_into_box_problems`, shared with the validate check.
"""

import logging

import numpy

from ..events import emit_log
from ..media_types import AudioVideo
from ..shots import carried_shots
from ..task_domains import whole_number
from ..task_problems import paste_into_box_problems, read_paste_box
from .fit import cut_box, float_frames, resize_frames, fill_crop
from .video_utils import frames_of, load_audio_video

logger = logging.getLogger("dw")

COMMAND = "paste_into_box"
# Frames converted to float and blended at a time, so a long clip never holds
# a float copy of the whole source beside the output
_CHUNK = 16


def _ramp(length, feather, open_start, open_end):
    """Per-pixel source weight along one axis of the box: 1 beyond `feather`
    px of an edge, (d + 1) / (feather + 1) at distance d inside it. An edge
    flush with the frame (`open_*` False) has no ramp."""
    weight = numpy.ones(length, dtype=numpy.float32)
    if feather <= 0:
        return weight
    index = numpy.arange(length, dtype=numpy.float32)
    ramp = numpy.minimum(1.0, (index + 1) / (feather + 1))
    if open_start:
        weight = numpy.minimum(weight, ramp)
    if open_end:
        weight = numpy.minimum(weight, ramp[::-1])
    return weight


def box_weights(box, frame_width, frame_height, feather):
    """The (h, w, 1) float32 weight of the source over the box."""
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    across = _ramp(w, feather, x > 0, x + w < frame_width)
    down = _ramp(h, feather, y > 0, y + h < frame_height)
    return numpy.minimum(down[:, None], across[None, :])[..., None]


def _read_box(content_box):
    """(box, source_box or None, fit record or None) from a box, a fit
    record, or a fit record's saved .json - by `task_problems.read_paste_box`,
    the rule the validate check uses too."""
    if isinstance(content_box, str):
        from ..locations import load_json_record

        content_box = load_json_record(content_box, None, "a content_box argument")
    parts, problem = read_paste_box(content_box)
    if problem is not None:
        raise ValueError(problem)
    return parts


def paste_into_box(generated, source, content_box, feather=16):
    """Task command: put a source clip back inside a box of a generated clip.

    Every frame of `source` is scaled into `content_box` and blended over the
    matching frame of `generated`: fully the source from `feather` px inside
    the box's edges, a linear ramp across those `feather` px, and `generated`
    untouched outside the box. A box edge on the frame's own edge gets no
    ramp. Given the fit record, the source is cut to its `source_box` first,
    so an aligned letterbox lines up pixel for pixel; given a bare box, the
    source is scaled to fill it and centre-cropped.

    Args:
        generated: The generated clip - an earlier step's video, or an
            `asset:`/`output:` reference. Its soundtrack, fps and shots are
            the result's
        source: The clip to paste back, with one frame per generated frame
        content_box: Where the source goes - a box {x, y, w, h} in
            `generated`'s pixels, or the record fit_to_model returned
            (previous_result:<step>.fit, or its saved .json), whose
            `content_box` and `source_box` it reads
        feather: Width in pixels of the blend inside the box's edges, from 0
            (a hard edge) to below half the box's shorter side

    Returns:
        An AudioVideo of float [0, 1] frames at `generated`'s size, with its
        audio, sample rate, fps and shots
    """
    box, kept, record = _read_box(content_box)
    problems = paste_into_box_problems(feather, box)
    if problems:
        raise ValueError("; ".join(message for _, message in problems))
    feather = whole_number(feather, "feather", COMMAND, required=True)

    if isinstance(generated, str):
        generated = load_audio_video(generated)
    if isinstance(source, str):
        source = load_audio_video(source)
    generated_frames, source_frames = frames_of(generated), frames_of(source)
    count = len(generated_frames)
    if count == 0:
        raise ValueError(f"{COMMAND} needs 'generated' with at least one frame")
    if len(source_frames) != count:
        raise ValueError(
            f"{COMMAND} needs 'source' with one frame per frame of "
            f"'generated': 'generated' has {count} frames, 'source' "
            f"{len(source_frames)}"
        )
    first = float_frames(generated_frames[:1], COMMAND)
    frame_height, frame_width = first.shape[1:3]
    if box["x"] + box["w"] > frame_width or box["y"] + box["h"] > frame_height:
        raise ValueError(
            f"{COMMAND} needs 'content_box' inside 'generated': the box "
            f"{box} runs outside its {frame_width}x{frame_height} frame"
        )
    source_first = float_frames(source_frames[:1], COMMAND)
    source_height, source_width = source_first.shape[1:3]
    if record is not None and (
        record.get("source_width"),
        record.get("source_height"),
    ) != (source_width, source_height):
        raise ValueError(
            f"{COMMAND} needs 'source' to be the clip the fit record was made "
            f"from: the record's source is {record.get('source_width')}x"
            f"{record.get('source_height')}, 'source' is "
            f"{source_width}x{source_height}"
        )
    if kept is None:
        kept = fill_crop(source_width, source_height, box["w"], box["h"])

    weight = box_weights(box, frame_width, frame_height, feather)
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    out = numpy.empty((count, frame_height, frame_width, 3), dtype=numpy.float32)
    for start in range(0, count, _CHUNK):
        stop = min(count, start + _CHUNK)
        chunk = float_frames(generated_frames[start:stop], COMMAND)
        if chunk.shape[1:3] != (frame_height, frame_width):
            raise ValueError(
                f"{COMMAND} needs every frame of 'generated' at one size, "
                f"got {chunk.shape[2]}x{chunk.shape[1]} after "
                f"{frame_width}x{frame_height}"
            )
        scaled = resize_frames(
            cut_box(float_frames(source_frames[start:stop], COMMAND), kept), w, h
        )
        region = chunk[:, y : y + h, x : x + w]
        # Where the weight is 1 the source is taken as it is, not through the
        # blend's arithmetic, so it is the source exactly
        chunk[:, y : y + h, x : x + w] = numpy.where(
            weight >= 1, scaled, region + (scaled - region) * weight
        )
        out[start:stop] = chunk

    emit_log(
        f"{COMMAND}: pasted {source_width}x{source_height} (kept {kept}) into "
        f"{box} of {frame_width}x{frame_height} over {count} frames, feather "
        f"{feather}"
    )
    return AudioVideo(
        out,
        getattr(generated, "audio", None),
        getattr(generated, "sample_rate", None),
        fps=getattr(generated, "fps", None),
        shots=carried_shots(generated),
    )
