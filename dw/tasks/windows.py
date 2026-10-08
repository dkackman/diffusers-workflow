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

`join_windows` is the other half: it takes the processed windows back, in
order, and blends each one's first `overlap` frames over the previous one's
last `overlap` real frames, so every source frame comes out exactly once and
the result is the source's length. It re-reads the source rather than the
windows for the plan (a pipeline's output carries no record of its input)
and for the soundtrack, which it puts back whole.

Memory (#695): a file source is never decoded whole. `window_video` reads
its window's frames with one keyframe seek (`read_frame_range`) and the
soundtrack on its own (`decode_soundtrack`); `join_windows` reads only the
source's header and soundtrack, and builds its output as uint8, a window at
a time, with only a seam's `overlap` frames in float32. A source given as
an earlier step's video, or fetched from a URL, is already in memory and is
cut the way it always was.
"""

import logging
import numbers

import numpy

from ..media import (
    NoSoundtrack,
    ShortFrameRange,
    count_video_frames,
    decode_soundtrack,
    read_frame_range,
    video_shape,
)
from ..media_types import AudioVideo, fit_codec_padding
from ..dsp import as_channels_samples, slice_samples

from ..shots import remeasured_shots, shot_record
from ..task_domains import (
    JOIN_WINDOWS_CURVES,
    check_arguments,
    frames_to_samples,
    window_count,
    window_count_problem,
    window_overlap_problem,
)
from .audio_utils import coerce_number
from .joins import video_names
from .video_utils import (
    VideoFileReference,
    frames_as_array,
    is_video_location,
    load_audio_video,
    local_video_path,
)

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

    start, end = window_span(index, num_frames, overlap)
    if isinstance(video, VideoFileReference):
        frames, total, waveform, sample_rate, file_fps = _file_window(
            video.path, index, num_frames, overlap
        )
        fps = fps or file_fps
    else:
        source = frames_as_array(video)
        total = len(source)
        _check_window_index(index, num_frames, overlap, total)
        # Clamping each frame index into the source repeats its first frame
        # before it and its last frame after it
        picks = numpy.clip(numpy.arange(start, end), 0, total - 1)
        frames = (source[picks].astype(numpy.float32) / 255.0).clip(0.0, 1.0)
        fps = fps or getattr(video, "fps", None)
        waveform = getattr(video, "audio", None)
        sample_rate = getattr(video, "sample_rate", None)

    audio, sample_rate = _window_audio(waveform, sample_rate, start, end, total, fps)
    logger.info(
        f"Window {index}: source frames {start}..{end - 1} of {total} "
        f"({max(0, -start)} repeated before, {max(0, end - total)} after)"
    )
    return AudioVideo(frames, audio, sample_rate, fps=fps)


def _check_window_index(index, num_frames, overlap, total):
    """Refuse a source with no frames, and a window that would start at or
    past its last frame, naming the last valid index."""
    if total == 0:
        raise ValueError("window_video was given a video with no frames")
    stride = num_frames - overlap
    if index * stride >= total:
        last = window_count(total, num_frames, overlap) - 1
        raise ValueError(
            f"window_video: window {index} would start at source frame "
            f"{index * stride}, but the source has {total} frames - with "
            f"num_frames {num_frames} and overlap {overlap} the last window "
            f"is index {last} ({last + 1} windows)"
        )


def _file_window(path, index, num_frames, overlap):
    """Window `index` of the file at `path`, read a range at a time (#695):
    `(frames, total, waveform, sample_rate, fps)`, the frames float32 in
    [0, 1] and the track the file's whole soundtrack (None when it has none).

    `total` is the header's frame count. A range read that runs out before
    the frames that count promises is retried once on a counted total, and
    refused, naming both counts, when the file still ends short.
    """
    shape = video_shape(path)
    total = shape["frame_count"]
    try:
        frames = _range_window(path, index, num_frames, overlap, total)
    except ShortFrameRange:
        counted = count_video_frames(path)
        logger.debug(f"{path}: header says {total} frames, decoding counted {counted}")
        try:
            frames = _range_window(path, index, num_frames, overlap, counted)
        except ShortFrameRange as short:
            raise ValueError(
                f"window_video: {path}'s header says it has {total} frames and "
                f"decoding counted {counted}, but reading window {index} ended "
                f"after {short.decoded}"
            ) from None
        total = counted
    waveform, sample_rate = _file_soundtrack(path, total, shape["fps"])
    return frames, total, waveform, sample_rate, shape["fps"]


def _range_window(path, index, num_frames, overlap, total):
    """The window's frames read from only the source frames it covers, the
    clamped edges repeated as the in-memory path repeats them."""
    _check_window_index(index, num_frames, overlap, total)
    start, end = window_span(index, num_frames, overlap)
    first, stop = max(start, 0), min(end, total)
    block = read_frame_range(path, first, stop)
    picks = numpy.clip(numpy.arange(start, end), 0, total - 1) - first
    frames = block[picks].astype(numpy.float32)
    del block
    frames /= 255.0
    return numpy.clip(frames, 0.0, 1.0, out=frames)


def _file_soundtrack(path, total, file_fps):
    """The file's soundtrack, decoded without its picture and fitted to
    `total` frames as a full decode fits it, with its rate - or (None, None)
    for a file with no track."""
    try:
        waveform, sample_rate = decode_soundtrack(path)
    except NoSoundtrack:
        return None, None
    if waveform.shape[-1] == 0:
        return None, None
    if file_fps:
        waveform = fit_codec_padding(waveform, total, file_fps, sample_rate)
    return waveform, sample_rate


def _window_audio(waveform, sample_rate, start, end, total, fps):
    """The window's track and rate, or (None, None) for a silent source.

    Every boundary is `frames_to_samples` of a source frame index, so the
    real span is the same samples whichever window cuts it, and the silence
    either side is the length those synthetic frames would have had.
    """
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


def _weights(overlap, curve):
    """Blend weights for the incoming window over an `overlap`-frame seam:
    `w(t)` at `t = (k + 1) / (overlap + 1)`, the open ramp dissolve_videos'
    `_ramp` uses, so neither end of the seam is a bare copy of one side."""
    t = (numpy.arange(overlap, dtype=numpy.float32) + 1) / (overlap + 1)
    if curve == "cosine":
        return (1 - numpy.cos(numpy.pi * t)) / 2
    if curve == "smoothstep":
        return 3 * t**2 - 2 * t**3
    return t


def join_windows(videos, source, num_frames, overlap, curve="cosine", fps=None):
    """Task command: blend processed windows back into one video the source's length.

    The other half of `window_video`. Give it the windows in order -
    `gather:<step>` over the step that processed them - and the same source
    they were cut from, with the same `num_frames` and `overlap`. Window 0
    gives its real frames (after its repeated-first-frame prefix); each later
    window's first `overlap` frames are blended over the previous window's
    last `overlap` frames, with the incoming weight `w(t)` at
    `t = (k + 1) / (overlap + 1)`; the final window's pad frames are dropped.
    Every source frame comes out exactly once.

    Args:
        videos: The processed windows, in window order. There must be exactly
            `ceil(source_frames / (num_frames - overlap))` of them, each
            `num_frames` frames long and all one frame size - which may
            differ from the source's (an upscaling model doubles it)
        source: The video the windows were cut from - an `asset:`/`output:`
            reference or a path, or an earlier step's video. Its frame count
            sets the plan, and its soundtrack is the output's
        num_frames: Frames per window, as given to window_video
        overlap: Frames each window shares with the one before, as given to
            window_video
        curve: The blend's shape across a seam - "cosine"
            `(1 - cos(pi t)) / 2`, "smoothstep" `3t^2 - 2t^3`, or "linear" `t`
        fps: Frame rate of the source. Defaults to the rate its file was read
            at; needed only to put its audio back

    Returns:
        One AudioVideo of exactly the source's frame count, at the windows'
        frame size, carrying the source's own audio over all of its frames
        (or none, when it has none) - the windows' audio is discarded. Its
        shot records hold one shot per window, each covering the frames that
        window owns, with `overlap_frames` on every seam and samples measured
        on the source's frame boundaries (#401), so assess_output reads the
        seams as dissolves
    """
    num_frames = coerce_number(num_frames, int, "num_frames", "join_windows")
    overlap = coerce_number(overlap, int, "overlap", "join_windows")
    fps = coerce_number(fps, float, "fps", "join_windows")
    for name, value in (("num_frames", num_frames), ("overlap", overlap)):
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            raise ValueError(
                f"join_windows needs '{name}' as a whole number, got {value!r}"
            )
    num_frames, overlap = int(num_frames), int(overlap)
    check_arguments("join_windows", num_frames=num_frames, overlap=overlap, fps=fps)
    if curve not in JOIN_WINDOWS_CURVES:
        raise ValueError(
            f"join_windows needs 'curve' as one of {list(JOIN_WINDOWS_CURVES)}, "
            f"got {curve!r}"
        )
    problem = window_overlap_problem(num_frames, overlap, "join_windows")
    if problem is not None:
        raise ValueError(problem)
    if not isinstance(videos, list) or not videos:
        raise ValueError("join_windows needs a non-empty list of windows in 'videos'")

    path = None
    if isinstance(source, VideoFileReference):
        path = source.path
    elif is_video_location(source):
        path = local_video_path(source)
        if path is None:
            source = load_audio_video(source)
    if path is not None:
        # The header and the soundtrack only: the picture is the windows'
        shape = video_shape(path)
        total, source_fps = shape["frame_count"], shape["fps"]
        waveform, sample_rate = _file_soundtrack(path, total, source_fps)
    else:
        total = len(frames_as_array(source))
        source_fps = getattr(source, "fps", None)
        waveform = getattr(source, "audio", None)
        sample_rate = getattr(source, "sample_rate", None)
    if total == 0:
        raise ValueError("join_windows was given a source with no frames")

    problem = window_count_problem(len(videos), total, num_frames, overlap)
    if problem is not None:
        raise ValueError(problem)

    names = video_names(videos)
    # A position in the list is what the caller fixes; a file name helps
    labels = [
        f"window {i}" + (f" ('{name}')" if isinstance(v, str) else "")
        for i, (name, v) in enumerate(zip(names, videos))
    ]
    windows = [
        frames_as_array(load_audio_video(v) if is_video_location(v) else v)
        for v in videos
    ]
    wrong = [
        f"{label} has {len(w)}"
        for label, w in zip(labels, windows)
        if len(w) != num_frames
    ]
    if wrong:
        raise ValueError(
            f"join_windows needs every window {num_frames} frames long "
            f"(num_frames): {'; '.join(wrong)}"
        )
    sizes = [w.shape[1:3] for w in windows]
    odd = [
        f"{label} is {size[1]}x{size[0]}"
        for label, size in zip(labels, sizes)
        if size != sizes[0]
    ]
    if odd:
        raise ValueError(
            f"join_windows needs every window at one size, the first's "
            f"{sizes[0][1]}x{sizes[0][0]}: {'; '.join(odd)}"
        )

    frames, starts, blended = _blend_windows(windows, total, num_frames, overlap, curve)

    fps = fps or source_fps
    audio, sample_rate = _source_audio(waveform, sample_rate, total, fps)
    shots = remeasured_shots(
        _window_shots(names, starts, blended, total),
        fps,
        sample_rate,
        None if audio is None else audio.shape[-1],
    )
    logger.info(
        f"Joined {len(windows)} windows of {num_frames} frames ({overlap}-frame "
        f"{curve} seams) into {total} frames"
    )
    return AudioVideo(frames, audio, sample_rate, fps=fps, shots=shots)


def _blend_windows(windows, total, num_frames, overlap, curve):
    """The windows joined into `total` uint8 frames, and each window's own
    first frame and blended-head length.

    The output is uint8 throughout (#695): a window's own frames are copied
    as they are, and only a seam is blended in float32. `carry` holds the
    last `overlap` frames written, unrounded, which are exactly the frames
    the next window's head blends over - so a frame that two seams blend
    (an overlap wider than the stride) is rounded once, at the end, as a
    float32 join of the whole video rounds it.
    """
    stride = num_frames - overlap
    weights = _weights(overlap, curve).reshape(-1, 1, 1, 1)
    joined = numpy.empty((total,) + windows[0].shape[1:], dtype=numpy.uint8)
    # Window 0's real frames, after its prefix
    first_end = min(stride, total)
    joined[:first_end] = windows[0][overlap : overlap + first_end]
    carry = joined[max(0, first_end - overlap) : first_end].astype(numpy.float32)
    # Where each window's own frames start: window 0 at the source's start,
    # each later one where its blended head does
    starts, blended = [0], [0]
    for index in range(1, len(windows)):
        start, _end = window_span(index, num_frames, overlap)
        window = windows[index]
        # An overlap wider than the stride reaches back past the source's
        # start; those head frames are synthetic and blend into nothing
        skip = max(0, -start)
        head = slice(start + skip, start + overlap)
        seam = (
            carry * (1 - weights[skip:])
            + window[skip:overlap].astype(numpy.float32) * weights[skip:]
        )
        joined[head] = seam.round().clip(0, 255).astype(numpy.uint8)
        tail_end = min(start + num_frames, total)
        own = window[overlap : tail_end - start]
        joined[start + overlap : tail_end] = own
        keep = min(overlap, tail_end)
        if len(own) >= keep:
            carry = own[len(own) - keep :].astype(numpy.float32)
        else:
            carry = numpy.concatenate(
                [seam[len(seam) - (keep - len(own)) :], own.astype(numpy.float32)]
            )
        starts.append(start + skip)
        blended.append(overlap - skip)
    return joined, starts, blended


def _source_audio(waveform, sample_rate, total, fps):
    """The source's track over exactly its `total` frames, and its rate, or
    (None, None) for a silent source."""
    if waveform is None:
        return None, None
    if not sample_rate:
        raise ValueError(
            "join_windows: the source has audio with no sample rate, so its "
            "track cannot be put back"
        )
    if not fps:
        raise ValueError(
            "join_windows needs 'fps' to put the source's audio back - the "
            "source does not carry a frame rate"
        )
    waveform = as_channels_samples(waveform)
    return slice_samples(
        waveform, 0, frames_to_samples(total, fps, sample_rate)
    ), sample_rate


def _window_shots(names, starts, blended, total):
    """One shot per window, over the frames it owns: from the start of its
    blended head (`overlap_frames`, the seam's dissolve) to the next window's.
    The frame side only: the track is the source's, put back whole rather
    than built window by window, so `remeasured_shots` lays these over it."""
    shots = []
    for index, (name, start, head) in enumerate(zip(names, starts, blended)):
        end = starts[index + 1] if index + 1 < len(starts) else total
        extra = {"overlap_frames": head} if index > 0 and head else {}
        shots.append(shot_record(name, start, end - start, **extra))
    return shots
