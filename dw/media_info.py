"""What the server knows about a generated media file and would otherwise
not say. An agent cannot listen: duration against a ceiling, peak against
a normalization target and the level at a seam are the only checks it can
make on an audio deliverable, and every one of them was being made by
fetching the file and running ffprobe by hand.
"""

import logging
import math

import av
import numpy

logger = logging.getLogger("dw")

# The floor a level is reported at rather than -inf, which JSON cannot carry
SILENCE_DBFS = -120.0


def _dbfs(value):
    if value <= 0:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, 20.0 * math.log10(float(value)))


def _audio_levels(container, stream):
    """Peak and rms of the whole decoded track, in dBFS."""
    peak = 0.0
    total = 0.0
    count = 0
    for frame in container.decode(stream):
        samples = frame.to_ndarray()
        if samples.dtype.kind in "iu":
            samples = samples.astype(numpy.float32) / numpy.iinfo(samples.dtype).max
        samples = samples.astype(numpy.float32)
        peak = max(peak, float(numpy.abs(samples).max(initial=0.0)))
        total += float(numpy.square(samples).sum())
        count += samples.size
    rms = math.sqrt(total / count) if count else 0.0
    return _dbfs(peak), _dbfs(rms)


def probe_media(path):
    """Duration, format and level of an audio or video file, or None.

    Video answers fps, frame_count, width and height, plus the soundtrack's
    sample_rate, channels, peak_dbfs and mean_dbfs when it carries one;
    audio answers the soundtrack fields. Levels come from decoding the
    whole track, which is cheap next to generating it.
    """
    try:
        container = av.open(path)
    except Exception as e:
        logger.debug(f"Not probeable as media: {path}: {e}")
        return None
    with container:
        video = container.streams.video[0] if container.streams.video else None
        audio = container.streams.audio[0] if container.streams.audio else None
        if video is None and audio is None:
            return None
        info = {}
        if video is not None:
            info["kind"] = "video"
            info["fps"] = float(video.average_rate) if video.average_rate else None
            if video.frames:
                info["frame_count"] = int(video.frames)
            else:
                info["frame_count"] = sum(1 for _ in container.decode(video))
            info["width"] = int(video.width)
            info["height"] = int(video.height)
        else:
            info["kind"] = "audio"
        if container.duration is not None:
            info["duration_seconds"] = container.duration / av.time_base
        if audio is not None:
            info["sample_rate"] = int(audio.rate)
            info["channels"] = int(audio.channels)
            info["peak_dbfs"], info["mean_dbfs"] = _audio_levels(container, audio)
        return info
