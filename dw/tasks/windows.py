"""Overlapping windows over a long video (#601).

A video-to-video model takes at most one bucket of frames - 121 for LTX - so a
longer source could not be restored or refined at all. `window_video` cuts one
model-sized window out of it, picked by `index`; a workflow drives it with
`for_each` over a list of `{name, index}` entries, since a task cannot return a
list of windows (every later `previous_result:` would fan out over it) and
nothing can iterate a list made at run time.

Window `i` covers source frames `[i * stride - overlap, i * stride + stride)`,
with `stride = num_frames - overlap`. The first window therefore starts
`overlap` frames before the source does, and the last may run past its end;
those synthetic frames repeat the source's first and last frame, so every
window is exactly `num_frames` long and every real frame after the first
window's prefix is covered by exactly one window's own stride.

The audio follows #401's cumulative rule: a real frame `f` owns samples
`frames_to_samples(f)` up to `frames_to_samples(f + 1)`, measured on the
source's own frame boundaries, so the real spans of adjacent windows' strides
tile the source track with no sample lost or repeated. The synthetic frames
are measured on the same extended timeline and carry silence.
"""

import logging
import numbers

import numpy

from ..media_types import AudioVideo
from ..dsp import as_channels_samples, slice_samples
from ..task_domains import check_arguments, frames_to_samples, window_overlap_problem
from .audio_utils import coerce_number
from .video_utils import VideoFileReference, frames_as_array, load_audio_video

logger = logging.getLogger("dw")


def window_span(index, num_frames, overlap):
    """The source frames window `index` covers, as `(start, end)` - end
    exclusive, start negative for the first window's prefix."""
    stride = num_frames - overlap
    start = index * stride - overlap
    return start, start + num_frames


def window_video(video, index, num_frames, overlap, fps=None):
    """Task command: one overlapping, fixed-length window of a long video.

    Window `index` covers source frames `[index * stride - overlap,
    index * stride + stride)`, where `stride = num_frames - overlap`. Frames
    before the source's start repeat its first frame (the first window's
    `overlap`-frame prefix); frames past its end repeat its last frame (the
    final window's pad). Drive it with `for_each` over a list of
    `{name, index}` entries, one per window - the source needs
    `ceil(source_frames / stride)` of them.

    Args:
        video: The long source - an `asset:`/`output:` reference or a path,
            read with its audio, or an earlier step's video
        index: Which window, from 0. A window that would start at or past
            the source's last frame (`index * stride >= source_frames`) is
            refused, naming the last valid index
        num_frames: Frames per window - the model's bucket, e.g. an 8n+1
            LTX length
        overlap: Frames each window shares with the one before it; below
            `num_frames`
        fps: Frame rate of the source. Defaults to the rate its file was
            read at; needed only to cut its audio

    Returns:
        One AudioVideo of exactly `num_frames` float32 frames in [0, 1] -
        the form the LTX reference condition reads, as with `loop_frames`.
        Its audio, when the source has a track, is the source's samples for
        the window's real frames, on the source's own frame boundaries so
        adjacent windows tile the track exactly, with silence of matching
        length under the repeated frames. A source with no track gives a
        window with none
    """
    index = coerce_number(index, int, "index", "window_video")
    num_frames = coerce_number(num_frames, int, "num_frames", "window_video")
    overlap = coerce_number(overlap, int, "overlap", "window_video")
    fps = coerce_number(fps, float, "fps", "window_video")
    for name, value in (
        ("index", index),
        ("num_frames", num_frames),
        ("overlap", overlap),
    ):
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            raise ValueError(
                f"window_video needs '{name}' as a whole number, got {value!r}"
            )
    index, num_frames, overlap = int(index), int(num_frames), int(overlap)
    # The same refusals validate gives a literal for free, for a value that
    # arrived from a variable or an earlier step
    check_arguments(
        "window_video", index=index, num_frames=num_frames, overlap=overlap, fps=fps
    )
    problem = window_overlap_problem(num_frames, overlap)
    if problem is not None:
        raise ValueError(problem)

    if isinstance(video, VideoFileReference):
        video = load_audio_video(video.path)

    source = frames_as_array(video)
    total = len(source)
    if total == 0:
        raise ValueError("window_video was given a video with no frames")

    stride = num_frames - overlap
    if index * stride >= total:
        last = (total - 1) // stride
        raise ValueError(
            f"window_video: window {index} would start at source frame "
            f"{index * stride}, but the source has {total} frames - with "
            f"num_frames {num_frames} and overlap {overlap} the last window "
            f"is index {last} ({last + 1} windows)"
        )

    start, end = window_span(index, num_frames, overlap)
    # Clamping each frame index into the source repeats its first frame
    # before it and its last frame after it
    picks = numpy.clip(numpy.arange(start, end), 0, total - 1)
    frames = (source[picks].astype(numpy.float32) / 255.0).clip(0.0, 1.0)

    fps = fps or getattr(video, "fps", None)
    audio, sample_rate = _window_audio(video, start, end, total, fps)
    logger.info(
        f"Window {index}: source frames {start}..{end - 1} of {total} "
        f"({max(0, -start)} repeated before, {max(0, end - total)} after)"
    )
    return AudioVideo(frames, audio, sample_rate, fps=fps)


def _window_audio(video, start, end, total, fps):
    """The window's track and rate, or (None, None) for a silent source.

    Every boundary is `frames_to_samples` of a source frame index, so the
    real span is the same samples whichever window cuts it, and the silence
    either side is the length those synthetic frames would have had.
    """
    waveform = getattr(video, "audio", None)
    sample_rate = getattr(video, "sample_rate", None)
    if waveform is None:
        return None, None
    if not sample_rate:
        raise ValueError(
            "window_video: the source has audio with no sample rate, so its "
            "track cannot be cut to the window"
        )
    if not fps:
        raise ValueError(
            "window_video needs 'fps' to cut the source's audio to the "
            "window - the source does not carry a frame rate"
        )
    waveform = as_channels_samples(waveform)

    def f2s(frame):
        return frames_to_samples(frame, fps, sample_rate)

    real_start, real_end = max(start, 0), min(end, total)
    head = f2s(real_start) - f2s(start)
    tail = f2s(end) - f2s(real_end)
    # slice_samples pads a track a sample or two short of its frames' length,
    # so a window's length depends only on its frames
    body = slice_samples(waveform, f2s(real_start), f2s(real_end) - f2s(real_start))
    channels = waveform.shape[0]
    audio = numpy.concatenate(
        [
            numpy.zeros((channels, head), dtype=numpy.float32),
            body,
            numpy.zeros((channels, tail), dtype=numpy.float32),
        ],
        axis=1,
    )
    return audio, sample_rate
