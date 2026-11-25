"""
Shared pure numpy/PIL helpers that grade, finish, lut and the assessment
probes use: Rec. 709 luma, splitting an image's alpha channel off and putting
it back, running an image command over a video frame by frame, and reducing
frames to the grey thumbnails and look stats `dw/tasks/assess.py` measures.
"""

import numpy as np
from PIL import Image

from .. import dsp
from ..media_types import AudioVideo

# Luma weights (Rec. 709), float64 - `luma` casts them to its caller's dtype
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def luma(rgb):
    """Rec. 709 luma of an array whose last axis is RGB.

    The weights are cast to the array's own dtype, so a float32 caller gets
    float32 arithmetic and result - exactly what a float32 weights copy gave
    it - rather than drifting through float64.
    """
    return np.tensordot(
        rgb, LUMA_WEIGHTS.astype(rgb.dtype, copy=False), axes=([-1], [0])
    )


def split_alpha(media):
    """(RGB float32 array in 0..255, the alpha channel or None)."""
    alpha = media.getchannel("A") if media.mode in ("RGBA", "LA") else None
    return np.asarray(media.convert("RGB"), dtype=np.float32), alpha


def join_alpha(array, alpha):
    """A PIL Image from a 0..255 float RGB array, with alpha put back."""
    array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    if alpha is not None:
        image = image.convert("RGBA")
        image.putalpha(alpha)
    return image


def per_frame(image, process):
    """Run an image command over a video, frame by frame.

    A video bound to an image argument - an AudioVideo from a generation,
    concat or dissolve step, or a frame array from video_frames - is processed
    one frame at a time and comes back as one video artifact, its soundtrack
    carried through untouched. A single image is processed as itself.
    """
    from ..shots import carried_shots
    from .video_utils import frames_as_pil_list, is_video

    if not is_video(image):
        return process(image)
    frames = [process(frame) for frame in frames_as_pil_list(image)]
    audio = getattr(image, "audio", None)
    sample_rate = getattr(image, "sample_rate", None)
    # One frame out per frame in, so the shot boundaries carry through too
    return AudioVideo(
        frames,
        audio,
        sample_rate,
        fps=getattr(image, "fps", None),
        shots=carried_shots(image),
    )


def look_stats(rgb):
    """A frame's look as four float32 scalars on the 0-255 scale: lightness
    (mean Rec. 709 luma), contrast (luma standard deviation), saturation (mean
    per-pixel max(RGB) - min(RGB)) and detail (mean |luma - its 3x3 box
    blur|, edges extended) - LOOK_MEASURES' order. The one reduction
    both the file reader and the in-memory path use."""
    import scipy.ndimage

    rgb = np.asarray(rgb, dtype=np.float32)
    plane = luma(rgb)
    blurred = scipy.ndimage.uniform_filter(plane, size=3, mode="nearest")
    return np.array(
        [
            plane.mean(),
            plane.std(),
            (rgb.max(axis=-1) - rgb.min(axis=-1)).mean(),
            np.abs(plane - blurred).mean(),
        ],
        dtype=np.float32,
    )


def as_image(frame):
    """One in-memory frame (PIL image, or an HxWxC array or tensor, uint8 or
    float in [0, 1]) as a PIL image."""
    if not isinstance(frame, Image.Image):
        array = frame
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        array = np.asarray(array)
        if array.dtype.kind == "f":
            array = np.clip(array * 255.0, 0, 255)
        array = array.astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        frame = Image.fromarray(array)
    return frame


def thumbs_and_looks(frames, thumb_size, look_size):
    """(grey uint8 thumbnails at `thumb_size`, `look_stats` of a `look_size`
    colour copy) of in-memory frames, one frame at a time: a list, an
    (N, H, W, C) array, or a SegmentedFrames replaying (N, H, W, 3) chunks.
    Both None for no frames. Sizes are (width, height)."""
    from ..pipeline_processors.chain_spill import SegmentedFrames

    if isinstance(frames, SegmentedFrames):
        frames = (frame for chunk in frames for frame in chunk)
    thumbs, looks = [], []
    for frame in frames:
        image = as_image(frame)
        thumbs.append(
            np.asarray(
                image.convert("L").resize(thumb_size, Image.BILINEAR), dtype=np.uint8
            )
        )
        rgb = image.convert("RGB").resize(look_size, Image.BILINEAR)
        looks.append(look_stats(np.asarray(rgb)))
    if not thumbs:
        return None, None
    return np.stack(thumbs), np.stack(looks)


LOOK_MEASURES = ("lightness", "contrast", "saturation", "detail")
# Frames either side of a seam its look is averaged over, and the floors
# under a look step's local rate and a detail ratio's denominator - all on
# the 0-255 scale, like frame_delta
LOOK_WINDOW = 4
LOOK_RATE_FLOOR = 1.0
DETAIL_FLOOR = 0.5
TYPICAL_DELTA_PERCENTILE = 90
_LOOK_STEPS = ("saturation", "lightness", "contrast")
_NO_LOOK = {
    "saturation_step": None,
    "lightness_step": None,
    "contrast_step": None,
    "look_ratio": None,
    "look_ratio_measure": None,
    "detail_ratio": None,
}


def _look_rate(values):
    """The TYPICAL_DELTA_PERCENTILE of the change between one LOOK_WINDOW-frame
    mean and the next inside `values` - the seam step's own statistic, taken
    where no seam is - or None for a span too short to hold two windows."""
    if values.shape[0] < 2 * LOOK_WINDOW:
        return None
    means = np.convolve(values, np.ones(LOOK_WINDOW) / LOOK_WINDOW, "valid")
    return float(
        np.percentile(
            np.abs(means[LOOK_WINDOW:] - means[:-LOOK_WINDOW]),
            TYPICAL_DELTA_PERCENTILE,
        )
    )


def _shot_end(shot, frames):
    count = int(shot.get("num_frames") or 0)
    start = int(shot.get("start_frame", 0))
    return min(frames, start + count) if count else frames


def measure_seam_look(media, previous, shot, fade):
    """Look measurements at the seam `shot` opens (#1031): the mean look of
    the LOOK_WINDOW frames before it (before a dissolve's fade) against the
    LOOK_WINDOW after, clipped to the shots. Each step is signed, after minus
    before; `look_ratio` is the largest step against that measure's own
    local rate inside either shot, floored, so a step inside a shot that is
    itself changing that way (a sunrise) scores low. Measures, not rules."""
    looks = media.looks
    if looks is None:
        return dict(_NO_LOOK)
    frames = looks.shape[0]
    start = min(frames, int(shot.get("start_frame", 0)))
    previous_start = min(
        start,
        int(previous.get("start_frame", 0)) + int(previous.get("overlap_frames") or 0),
    )
    after_start = min(frames, start + fade)
    after_end = _shot_end(shot, frames)
    before = looks[max(previous_start, start - LOOK_WINDOW) : start]
    after = looks[after_start : min(after_end, after_start + LOOK_WINDOW)]
    if before.shape[0] == 0 or after.shape[0] == 0:
        return dict(_NO_LOOK)
    before_mean = before.mean(axis=0, dtype=np.float64)
    after_mean = after.mean(axis=0, dtype=np.float64)
    record = {}
    ratio, ratio_measure = None, None
    for measure in _LOOK_STEPS:
        column = LOOK_MEASURES.index(measure)
        step = float(after_mean[column] - before_mean[column])
        record[f"{measure}_step"] = dsp.rounded(step)
        rates = [
            _look_rate(looks[previous_start:start, column]),
            _look_rate(looks[after_start:after_end, column]),
        ]
        rate = max([LOOK_RATE_FLOOR] + [value for value in rates if value is not None])
        if ratio is None or abs(step) / rate > ratio:
            ratio, ratio_measure = abs(step) / rate, measure
    detail = LOOK_MEASURES.index("detail")
    record["look_ratio"] = dsp.rounded(ratio)
    record["look_ratio_measure"] = ratio_measure
    record["detail_ratio"] = (
        dsp.rounded(float(after_mean[detail] / before_mean[detail]))
        if before_mean[detail] >= DETAIL_FLOOR
        else None
    )
    return record


def _window_look(looks, start, end):
    """The mean of each LOOK_MEASURES column over looks[start:end]."""
    means = looks[start:end].mean(axis=0, dtype=np.float64)
    return {
        measure: dsp.rounded(means[column], 3)
        for column, measure in enumerate(LOOK_MEASURES)
    }


def measure_segment_looks(media, records):
    """Each shot's look at the start and the end of its own new frames (#1324):
    the frames from after its `overlap_frames` (the dissolve fade
    `measure_seam_look` steps over) to the next shot's start, or for the last shot
    to the end of the take. `first` is the mean over the first second of
    those frames, `last` over the last second, each clipped to the segment;
    a segment shorter than two seconds splits its frames in half (a single
    frame is both). With `next.first - this.last` at each seam, the seam
    differences and the in-segment `last - first` changes sum exactly to the
    take's last-minus-first. A measure only: no rule reads it. Empty for
    media with no picture or fewer than two shots."""
    looks = media.looks
    if looks is None or not records or len(records) < 2:
        return []
    frames = looks.shape[0]
    window = max(1, int(round(media.fps))) if media.fps else LOOK_WINDOW
    segments = []
    for index, shot in enumerate(records):
        start = int(shot.get("start_frame", 0))
        begin = min(frames, start + int(shot.get("overlap_frames") or 0))
        end = _shot_end(shot, frames)
        if index + 1 < len(records):
            end = min(end, int(records[index + 1].get("start_frame", 0)))
        end = max(begin, min(frames, end))
        count = end - begin
        entry = {"shot": shot.get("name"), "between_frames": None}
        if count == 0:
            entry.update(first=None, last=None)
        else:
            entry["between_frames"] = [begin, end - 1]
            if count >= 2 * window:
                head, tail = window, window
            else:
                head = max(1, count // 2)
                tail = max(1, count - count // 2)
            entry["first"] = _window_look(looks, begin, begin + head)
            entry["last"] = _window_look(looks, end - tail, end)
        segments.append(entry)
    return segments
