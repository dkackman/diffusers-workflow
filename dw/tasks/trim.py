"""Keep a span of a video's frames, and its audio over the same span (#627).

`trim_video` keeps frames `[start_frame, start_frame + num_frames)`. The clip's
soundtrack, if it has one, is cut to the same span at the track's own rate:
each end is `frames_to_samples` of a frame index, rounded on its own (the
cumulative rule `slice_audio` and `window_video` follow), so consecutive trims
tile the track with no sample lost or repeated.
"""

import logging
import numbers

from ..dsp import as_channels_samples
from ..media_types import AudioVideo
from ..shots import shot_record, trimmed_shots
from ..task_domains import check_arguments, frames_to_samples
from .registry import register_command
from .audio_utils import coerce_number
from .joins import video_names
from .video_utils import VideoFileReference, frames_as_pil_list, load_audio_video

logger = logging.getLogger("dw")

COMMAND = "trim_video"


def trim_video(video, start_frame, num_frames, fps=None):
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

    Returns:
        The kept frames - an AudioVideo when the clip came in as one (its
        track cut to the same span, at its own rate, fps and sample rate
        kept), else a frame list. The shots the clip carried are clipped to
        the span (sample side cleared); a clip with none carries one shot
        spanning the kept frames, named after the file it was read from
    """
    start_frame = coerce_number(start_frame, int, "start_frame", COMMAND)
    num_frames = coerce_number(num_frames, int, "num_frames", COMMAND)
    fps = coerce_number(fps, float, "fps", COMMAND)
    for name, value in (("start_frame", start_frame), ("num_frames", num_frames)):
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            raise ValueError(
                f"{COMMAND} needs '{name}' as a whole number, got {value!r}"
            )
    start_frame, num_frames = int(start_frame), int(num_frames)
    check_arguments(COMMAND, start_frame=start_frame, num_frames=num_frames, fps=fps)

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

    if not isinstance(video, AudioVideo):
        return kept

    audio, sample_rate = video.audio, video.sample_rate
    fps = fps or video.fps
    if audio is not None:
        audio = _span_of_audio(audio, sample_rate, fps, start_frame, end)
    shots = trimmed_shots(video.shots, start_frame, keep_frames=num_frames)
    if not shots:
        # A clip with no shots record is one shot: the trim is still a clip
        # a consumer reads spans from (#627). The track was cut at its own
        # rate, so its samples are measured here, not derived
        shots = [
            shot_record(
                name,
                0,
                num_frames,
                None if audio is None else 0,
                None if audio is None else audio.shape[1],
            )
        ]
    return AudioVideo(kept, audio, sample_rate, fps=fps, shots=shots)


def _span_of_audio(audio, sample_rate, fps, start_frame, end):
    """The track over frames [start_frame, end), cut at the track's own rate."""
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
    return waveform[:, first:last]


@register_command(COMMAND, implementation="dw.tasks.trim.trim_video")
def _handle_trim_video(task, arguments, previous_pipelines):
    """Keep a span of a video's frames, and its audio over the same span"""
    logger.debug("Trimming a video")
    return trim_video(**arguments)
