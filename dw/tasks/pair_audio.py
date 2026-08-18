"""Pair a video with an audio track so the two are saved as one file.

A pipeline that generates its own soundtrack returns the pair together, and the
result muxes them into a single mp4. Anything that works on the frames alone -
a latent upsampler, an interpolator, an upscaler - returns frames without it, so
the soundtrack has to be carried across the step that dropped it. That is what
this does: it puts the two back together for the step that saves them.
"""

import logging

from ..result import AudioVideo
from .audio_utils import as_channels_samples

logger = logging.getLogger("dw")


def pair_audio(video, audio, sample_rate=None):
    """Pair a video's frames with an audio track.

    Args:
        video: The frames - a frame list, a frame array or tensor, or an
            AudioVideo whose own soundtrack is replaced by this one
        audio: The soundtrack - a waveform, or an AudioVideo (or any object
            carrying '.audio') to take it from, which brings its sample rate
            along with it
        sample_rate: Sample rate of the waveform. Required unless `audio`
            carries one; given here it wins, for a track whose rate was
            reported wrong

    Returns:
        One AudioVideo holding the frames and the track

    Raises:
        ValueError: If no waveform was given, or if no sample rate can be
            established for the one that was
    """
    waveform = getattr(audio, "audio", audio)
    if waveform is None:
        raise ValueError(
            "pair_audio needs an audio track - the video it was given carries none"
        )

    rate = (
        sample_rate if sample_rate is not None else getattr(audio, "sample_rate", None)
    )
    if rate is None:
        raise ValueError(
            "pair_audio needs 'sample_rate' - the audio it was given does not "
            "carry one of its own"
        )

    # Frames are left in whatever shape they arrived in - the result saves a frame
    # list, an array and a tensor alike, and converting a long video here would
    # cost a copy of the whole thing for nothing
    frames = video.frames if isinstance(video, AudioVideo) else video
    logger.debug(f"Pairing frames with audio at {rate} Hz")
    return AudioVideo(frames, as_channels_samples(waveform), rate)
