"""Concatenate videos - and the audio generated with them - into one video.

The standalone counterpart of what a chained pipeline step does internally:
frames are joined end to end with an optional head trim on every video after
the first, and audio tracks are joined at each seam with an equal-power
crossfade drawn from the trimmed-off material, so video and audio stay in
sync.
"""

import logging

from ..result import AudioVideo
from .audio_utils import (
    as_channels_samples,
    equal_power_crossfade_join,
    frames_to_samples,
)
from .video_utils import frames_as_pil_list

logger = logging.getLogger("dw")


def concat_videos(videos, trim_frames=0, crossfade_ms=75, fps=None):
    """Concatenate a list of videos into a single AudioVideo.

    Args:
        videos: The videos to join, in order - frame lists, frame arrays, or
            AudioVideos (from gather_videos or previous_result references)
        trim_frames: Frames dropped from the head of every video after the
            first - the trim used when each video was generated from the
            previous one's last frame
        crossfade_ms: Equal-power crossfade at each audio seam, clamped to
            the trimmed material
        fps: Frame rate of the videos - required to join audio when trimming

    Returns:
        One AudioVideo; its audio is None when no input video carries any
    """
    if not isinstance(videos, list) or not videos:
        raise ValueError("concat_videos needs a non-empty list of videos")

    frames = []
    audio = None
    sample_rate = None

    for index, video in enumerate(videos):
        head_trim = trim_frames if index > 0 else 0
        frames.extend(frames_as_pil_list(video)[head_trim:])

        if not isinstance(video, AudioVideo) or video.audio is None:
            continue

        waveform = as_channels_samples(video.audio)
        if audio is None:
            audio, sample_rate = waveform, video.sample_rate
            continue

        if video.sample_rate != sample_rate:
            raise ValueError(
                f"Videos carry audio at different sample rates: "
                f"{sample_rate} then {video.sample_rate}"
            )
        if head_trim > 0 and fps is None:
            raise ValueError(
                "concat_videos needs 'fps' to trim audio in step with the frames"
            )

        trim_samples = (
            frames_to_samples(head_trim, fps, sample_rate) if head_trim else 0
        )
        audio = equal_power_crossfade_join(
            audio,
            waveform[:, :trim_samples],
            waveform[:, trim_samples:],
            sample_rate,
            crossfade_ms,
        )

    logger.debug(f"Concatenated {len(videos)} videos into {len(frames)} frames")
    return AudioVideo(frames, audio, sample_rate)
