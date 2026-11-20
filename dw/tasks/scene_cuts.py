"""Find the cuts in a clip, and lay each one out for an H3 render (#1194).

`content_cuts` is the detector: a frame is a cut when its colour histogram
drops away from the previous frame's, or when the difference from the
previous frame spikes (a cut between two framings of the same picture keeps
its colours, so the histogram alone misses it). It was `crop_face_track`'s
own (`dw/tasks/face_track.py`), which resets its track at each one, and moved
here so `detect_cuts` reads the same cuts the face track does.

`detect_cuts` answers the cut list as `dw/shots.py` shot records (`cut-1`,
`cut-2`, ...), each with the `render_frames` an H3 refine of it renders: the
smallest count on H3's 17n+5 grid inside the pipeline's duration window at or
above the cut's own length, so a cut is padded *up* to it (`trim_video`
`pad_to`) and trimmed back after, keeping its sync with the source audio. The
window and grid are not written here: they are read off the diffusers class
(`dw/pipeline_limits.py`) and `h3_rules.RENDER_GRID`. A cut longer than
`max_frames` (at most the window's top) is split into near-equal pieces, each
marked `split`; a cut shorter than the grid's first step past its remainder
(22 frames) is marked `too_short`, since it would pad to the window's floor
from almost nothing. Neither is dropped: the cut list is a literal the agent
edits, and `replace_cuts` (`dw/tasks/trim.py`) puts back whichever of them
were re-rendered, the rest of the take kept.

The detector is tuned for hard cuts: a dissolve or a slow fade is not found.
"""

import logging
import math

import numpy as np

from ..events import emit_warning
from ..media_types import JsonRecord
from ..shots import shot_record
from ..task_domains import POSITIVE, check_arguments, frames_to_samples, whole_number
from .registry import register_command
from .video_utils import VideoFileReference, frames_as_pil_list, load_audio_video

logger = logging.getLogger("dw")

COMMAND = "detect_cuts"

HISTOGRAM_CUT = 0.5  # HSV histogram correlation below which frames are a cut
# A cut that keeps the colours (a zoom between two framings of one picture):
# the mean difference of small thumbnails, as a fraction of full scale, must
# reach CUT_MIN_CHANGE and CUT_SPIKE times the median of the frames around it
THUMBNAIL_SIZE = (32, 18)
CUT_MIN_CHANGE = 0.06
CUT_SPIKE = 4.0
CUT_WINDOW = 8  # frames each side whose differences form that median

CUT_PREFIX = "cut"


def _hsv_histogram(rgb):
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).flatten()


def _thumbnail(rgb):
    import cv2

    return cv2.resize(rgb, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA).astype(
        np.float32
    )


def content_cuts(frames_rgb):
    """Cuts nobody recorded: [(frame index, detail)] where the picture jumps.

    A frame is a cut when its HSV histogram correlates with the previous
    frame's below HISTOGRAM_CUT, or when its thumbnail difference from the
    previous frame is a spike: at least CUT_MIN_CHANGE, and CUT_SPIKE times
    the median difference of the CUT_WINDOW frames either side.
    """
    import cv2

    correlations, changes = [None], [0.0]
    previous = None
    for rgb in frames_rgb:
        current = (_hsv_histogram(rgb), _thumbnail(rgb))
        if previous is not None:
            correlations.append(
                float(cv2.compareHist(previous[0], current[0], cv2.HISTCMP_CORREL))
            )
            changes.append(float(np.mean(np.abs(current[1] - previous[1]))) / 255)
        previous = current

    cuts = []
    for index in range(1, len(changes)):
        change = changes[index]
        around = (
            changes[max(1, index - CUT_WINDOW) : index]
            + changes[index + 1 : index + 1 + CUT_WINDOW]
        )
        baseline = float(np.median(around)) if around else 0.0
        spike = change >= CUT_MIN_CHANGE and change >= CUT_SPIKE * baseline
        if correlations[index] < HISTOGRAM_CUT or spike:
            cuts.append(
                (
                    index,
                    {
                        "histogram_correlation": round(correlations[index], 4),
                        "frame_change": round(change, 4),
                    },
                )
            )
    return cuts


# ------------------------------------------------------------ render layout


def render_window():
    """H3's render grid and frame window as {modulus, remainder, min_frames,
    max_frames}, read off the diffusers class - (17, 5, 124, 345) today - or
    None when this diffusers does not declare them."""
    from ..pipeline_limits import _limits, accepted_frames
    from ..pipeline_processors.h3_rules import RENDER_GRID

    limits = _limits()
    if limits is None:
        return None
    low, high = accepted_frames(limits)
    return {
        "modulus": RENDER_GRID["modulus"],
        "remainder": RENDER_GRID["remainder"],
        "min_frames": low,
        "max_frames": high,
    }


def too_short_below(window):
    """The cut length below which a cut is flagged too_short: the grid's
    first step past its remainder (22 on 17n+5)."""
    return window["modulus"] + window["remainder"]


def render_frames(num_frames, window):
    """The smallest on-grid count inside the window at or above num_frames."""
    from ..variable_constraints import aligned

    grid = {"modulus": window["modulus"], "remainder": window["remainder"]}
    return max(window["min_frames"], aligned(num_frames, grid))


def max_frames_problem(max_frames, window):
    """Why a literal max_frames cannot be used, or None."""
    if window is None or not isinstance(max_frames, int):
        return None
    if isinstance(max_frames, bool):
        return None
    if max_frames > window["max_frames"]:
        return (
            f"{COMMAND}: max_frames must be at most {window['max_frames']}, the "
            f"most frames MiniMax-H3 renders, got {max_frames}"
        )
    return None


def detect_cuts_errors(arguments):
    """[(argument, message)] for a literal max_frames above H3's ceiling."""
    value = arguments.get("max_frames")
    if isinstance(value, str):
        try:
            value = int(float(value))
        except ValueError:
            return []
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    problem = max_frames_problem(value, render_window())
    return [("max_frames", problem)] if problem else []


def split_lengths(num_frames, max_frames):
    """A cut's length as near-equal pieces, none longer than max_frames."""
    pieces = max(1, math.ceil(num_frames / max_frames))
    base, extra = divmod(num_frames, pieces)
    return [base + 1 if index < extra else base for index in range(pieces)]


def cut_records(starts, total, max_frames, window, fps=None, sample_rate=None):
    """The shot records for cuts starting at `starts` ([(frame, detail)],
    the first at 0) in a clip of `total` frames."""
    short = too_short_below(window)
    records = []
    bounds = [frame for frame, _ in starts[1:]] + [total]
    for (start, detail), end in zip(starts, bounds):
        lengths = split_lengths(end - start, max_frames)
        split = len(lengths) > 1
        offset = start
        for length in lengths:
            start_sample = num_samples = None
            if fps and sample_rate:
                start_sample = frames_to_samples(offset, fps, sample_rate)
                num_samples = (
                    frames_to_samples(offset + length, fps, sample_rate) - start_sample
                )
            extra = {
                "render_frames": render_frames(length, window),
                "detail": dict(detail),
            }
            if split:
                extra["split"] = True
            if length < short:
                extra["too_short"] = True
            records.append(
                shot_record(
                    f"{CUT_PREFIX}-{len(records) + 1}",
                    offset,
                    length,
                    start_sample,
                    num_samples,
                    **extra,
                )
            )
            offset += length
    return records


def detect_cuts(video, max_frames=345):
    """Task command: find a clip's cuts, each laid out for an H3 render.

    Args:
        video: The clip - a path, an asset:/output: reference, or an earlier
            step's video
        max_frames: The longest cut kept whole, at most 345 (MiniMax-H3's
            ceiling); a longer one is split into near-equal pieces, each
            marked "split"

    Returns:
        JSON: fps, num_frames, the render grid, and `cuts` - shot records
        named cut-1, cut-2, ... with start_frame, num_frames, start_sample
        and num_samples (when the clip has audio), render_frames (the
        smallest 17n+5 count inside 124-345 at or above num_frames), the
        detector's detail, and "split" / "too_short" (under 22 frames) flags
    """
    if video is None:
        raise ValueError(f"{COMMAND} needs 'video' - the clip to find cuts in")
    max_frames = whole_number(max_frames, "max_frames", COMMAND, required=True)
    check_arguments(COMMAND, max_frames=max_frames)
    window = render_window()
    if window is None:
        raise ValueError(
            f"{COMMAND}: this diffusers does not declare MiniMax-H3's frame "
            "window, so no render length can be laid out"
        )
    problem = max_frames_problem(max_frames, window)
    if problem:
        raise ValueError(problem)

    if isinstance(video, VideoFileReference):
        video = load_audio_video(video.path)
    elif isinstance(video, str):
        video = load_audio_video(video)
    frames = frames_as_pil_list(video)
    total = len(frames)
    if total == 0:
        raise ValueError(f"{COMMAND}: the video has no frames")
    fps = getattr(video, "fps", None)
    sample_rate = getattr(video, "sample_rate", None)
    if getattr(video, "audio", None) is None:
        sample_rate = None

    found = content_cuts(np.asarray(frame.convert("RGB")) for frame in frames)
    starts = [(0, {"reason": "clip start"})] + [
        (frame, {"reason": "content", **detail}) for frame, detail in found
    ]
    cuts = cut_records(starts, total, max_frames, window, fps, sample_rate)
    logger.info(f"{COMMAND}: {len(found) + 1} shot(s), {len(cuts)} cut(s)")

    split = [c["name"] for c in cuts if c.get("split")]
    short = [c["name"] for c in cuts if c.get("too_short")]
    if split:
        emit_warning(
            f"{COMMAND}: split near-equally from a shot longer than max_frames "
            f"({max_frames}): {', '.join(split)} - identity may drift between "
            "the pieces"
        )
    if short:
        emit_warning(
            f"{COMMAND}: under {too_short_below(window)} frames, so padded to "
            f"{window['min_frames']} from almost nothing: {', '.join(short)} - "
            "merge each into a neighbour, or leave it out of the cuts you "
            "re-render (replace_cuts keeps the take's frames for a cut not "
            "listed)"
        )
    return JsonRecord(
        {
            "fps": None if fps is None else float(fps),
            "num_frames": total,
            "max_frames": max_frames,
            "render_grid": window,
            "cuts": cuts,
        }
    )


@register_command(
    COMMAND,
    implementation="dw.tasks.scene_cuts.detect_cuts",
    returns="json",
    domains={"max_frames": POSITIVE},
    whole_numbers=("max_frames",),
    static_check=detect_cuts_errors,
)
def _handle_detect_cuts(task, arguments, previous_pipelines):
    """Find a clip's cuts, each with an on-grid H3 render length"""
    logger.debug("Detecting cuts")
    return detect_cuts(**arguments)
