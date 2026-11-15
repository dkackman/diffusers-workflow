"""Keep a span of a video's frames, and its audio over the same span (#627).

`trim_video` keeps frames `[start_frame, start_frame + num_frames)`. The clip's
soundtrack, if it has one, is cut to the same span at the track's own rate:
each end is `frames_to_samples` of a frame index, rounded on its own (the
cumulative rule `slice_audio` and `window_video` follow), so consecutive trims
tile the track with no sample lost or repeated.

`pad_to` lengthens the kept span to a render length (#1194): the last kept
frame is held up to `pad_to` frames and the track runs on in silence to the
same end, so a cut `detect_cuts` laid out can be handed to a model whose frame
grid it falls short of, and trimmed back to its own length after.
"""

import logging
import numbers

import numpy as np

from ..dsp import as_channels_samples
from ..media_types import AudioVideo
from ..shots import shot_record, trimmed_shots
from ..task_domains import (
    NON_NEGATIVE,
    POSITIVE,
    check_arguments,
    frames_to_samples,
    real_number,
    whole_number,
)
from .registry import register_command
from .joins import video_names
from .video_utils import VideoFileReference, frames_as_pil_list, load_audio_video

logger = logging.getLogger("dw")

COMMAND = "trim_video"


def trim_video(video, start_frame, num_frames, fps=None, pad_to=None):
    """Task command: keep one span of a video's frames and its audio.

    Args:
        video: The clip - a frame list or array, an AudioVideo (from a
            previous_result reference), or the path or URL of a video file,
            read with its audio
        start_frame: The first frame kept, from 0
        num_frames: How many frames are kept. The span must lie inside the
            clip: asking past its end is refused rather than shortened
        fps: Frame rate of the clip. Defaults to the rate the clip carries;
            needed only to cut its audio, which is refused without one
        pad_to: A frame count at least num_frames: the last kept frame is
            held until the clip is this long, and its track padded with
            silence to the same end. Omitted, the span is kept as it is

    Returns:
        The kept frames as one AudioVideo - the clip's track cut to the same
        span, at its own rate, fps and sample rate kept; a clip that came in
        as bare frames comes out silent, at 'fps'. The shots the clip carried are clipped to
        the span (sample side cleared); a clip with none carries one shot
        spanning the kept frames, named after the file it was read from.
        With pad_to, the clip is pad_to frames long and its last shot runs
        over the held frames
    """
    start_frame = whole_number(start_frame, "start_frame", COMMAND)
    num_frames = whole_number(num_frames, "num_frames", COMMAND)
    pad_to = whole_number(pad_to, "pad_to", COMMAND)
    fps = real_number(fps, "fps", COMMAND)
    whole = [("start_frame", start_frame), ("num_frames", num_frames)]
    if pad_to is not None:
        whole.append(("pad_to", pad_to))
    for name, value in whole:
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            raise ValueError(
                f"{COMMAND} needs '{name}' as a whole number, got {value!r}"
            )
    start_frame, num_frames = int(start_frame), int(num_frames)
    pad_to = None if pad_to is None else int(pad_to)
    check_arguments(
        COMMAND, start_frame=start_frame, num_frames=num_frames, fps=fps, pad_to=pad_to
    )
    problem = pad_to_problem(num_frames, pad_to)
    if problem:
        raise ValueError(problem)

    name = video_names([video])[0]
    if isinstance(video, VideoFileReference):
        name = video_names([video.path])[0]
        video = load_audio_video(video.path)
    elif isinstance(video, str):
        video = load_audio_video(video)

    frames = frames_as_pil_list(video)
    total = len(frames)
    end = start_frame + num_frames
    if end > total:
        raise ValueError(
            f"{COMMAND}: start_frame {start_frame} + num_frames {num_frames} "
            f"reaches frame {end}, but the clip has only {total} frames"
        )
    kept = frames[start_frame:end]
    logger.info(f"Trimmed to frames {start_frame}..{end - 1} of {total}")
    held = (pad_to or num_frames) - num_frames
    if held:
        kept = list(kept) + [kept[-1]] * held
        logger.info(f"Held the last frame {held} more, to {pad_to} frames")

    # Bare frames come out as one silent clip, not a bare list: a task's list
    # result is one result per entry, so the step saved each frame as its own
    # video (#1199)
    clip = video if isinstance(video, AudioVideo) else None
    audio = clip.audio if clip else None
    sample_rate = clip.sample_rate if clip else None
    fps = fps or (clip.fps if clip else None)
    if audio is not None:
        audio = _span_of_audio(audio, sample_rate, fps, start_frame, end, held)
    shots = trimmed_shots(
        clip.shots if clip else None, start_frame, keep_frames=num_frames
    )
    if shots and held:
        # The held frames belong to the shot they repeat
        shots[-1] = {**shots[-1], "num_frames": shots[-1]["num_frames"] + held}
    if not shots:
        # A clip with no shots record is one shot: the trim is still a clip
        # a consumer reads spans from (#627). The track was cut at its own
        # rate, so its samples are measured here, not derived
        shots = [
            shot_record(
                name,
                0,
                num_frames + held,
                None if audio is None else 0,
                None if audio is None else audio.shape[1],
            )
        ]
    return AudioVideo(kept, audio, sample_rate, fps=fps, shots=shots)


def pad_to_problem(num_frames, pad_to):
    """Why a pad_to cannot lengthen a num_frames span, or None."""
    if pad_to is None or num_frames is None or pad_to >= num_frames:
        return None
    return (
        f"{COMMAND}: pad_to {pad_to} is shorter than num_frames {num_frames} - "
        "pad_to only lengthens the span, so it must be at least num_frames"
    )


def _literal_whole(value):
    return isinstance(value, int) and not isinstance(value, bool)


def trim_video_errors(arguments):
    """[(argument, message)] for a literal pad_to below a literal num_frames."""
    num_frames, pad_to = arguments.get("num_frames"), arguments.get("pad_to")
    if not (_literal_whole(num_frames) and _literal_whole(pad_to)):
        return []
    if pad_to < 1 or num_frames < 1:
        return []  # the domain's to refuse, once
    problem = pad_to_problem(num_frames, pad_to)
    return [("pad_to", problem)] if problem else []


def _span_of_audio(audio, sample_rate, fps, start_frame, end, held=0):
    """The track over frames [start_frame, end), cut at the track's own rate,
    and run on in silence over `held` more frames."""
    if not sample_rate:
        raise ValueError(
            f"{COMMAND}: the clip has audio with no sample rate, so its track "
            "cannot be cut to the span"
        )
    if not fps:
        raise ValueError(
            f"{COMMAND} needs 'fps' to cut the clip's audio to the span - "
            "the clip does not carry a frame rate"
        )
    waveform = as_channels_samples(audio)
    first = frames_to_samples(start_frame, fps, sample_rate)
    last = frames_to_samples(end, fps, sample_rate)
    span = waveform[:, first:last]
    if not held:
        return span
    padded = frames_to_samples(end + held, fps, sample_rate) - first
    return np.pad(span, ((0, 0), (0, max(0, padded - span.shape[1]))))


@register_command(
    COMMAND,
    implementation="dw.tasks.trim.trim_video",
    domains={
        "start_frame": NON_NEGATIVE,
        "num_frames": POSITIVE,
        "fps": POSITIVE,
        "pad_to": POSITIVE,
    },
    whole_numbers=("start_frame", "num_frames", "pad_to"),
    static_check=trim_video_errors,
)
def _handle_trim_video(task, arguments, previous_pipelines):
    """Keep a span of a video's frames, and its audio over the same span"""
    logger.debug("Trimming a video")
    return trim_video(**arguments)
