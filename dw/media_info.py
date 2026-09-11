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


def probe_media(path):
    """Duration, format and level of an audio or video file, or None.

    Video answers fps, frame_count, width and height, plus the soundtrack's
    sample_rate, channels, peak_dbfs and mean_dbfs when it carries one;
    audio answers the soundtrack fields. Levels come from decoding the
    whole track, which is cheap next to generating it.

    When a frame count still needs counting and/or a soundtrack still needs
    its levels measured, both are gathered from a single decode pass over
    whichever streams are involved - `container.decode()` demuxes to EOF, so
    two separate passes (count video, then decode audio) would leave the
    second one nothing to read.
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
            info["width"] = int(video.width)
            info["height"] = int(video.height)
        else:
            info["kind"] = "audio"
        if container.duration is not None:
            info["duration_seconds"] = float(container.duration / av.time_base)
        if audio is not None:
            info["sample_rate"] = int(audio.rate)
            info["channels"] = int(audio.channels)

        # Some muxers don't write a frame count up front (0 means "count
        # them"); a soundtrack always needs decoding to measure its level.
        # Do both together, since decoding is a one-way trip through the file.
        need_frame_count = video is not None and not video.frames
        if video is not None and not need_frame_count:
            info["frame_count"] = int(video.frames)

        if need_frame_count or audio is not None:
            frame_count = 0
            peak = 0.0
            total = 0.0
            count = 0
            streams = [
                s
                for s in ((video if need_frame_count else None), audio)
                if s is not None
            ]
            try:
                for frame in container.decode(*streams):
                    if isinstance(frame, av.VideoFrame):
                        frame_count += 1
                    elif isinstance(frame, av.AudioFrame):
                        samples = frame.to_ndarray()
                        if samples.dtype.kind == "u":
                            iinfo = numpy.iinfo(samples.dtype)
                            half = (iinfo.max + 1) / 2
                            samples = (samples.astype(numpy.float32) - half) / half
                        elif samples.dtype.kind == "i":
                            samples = (
                                samples.astype(numpy.float32)
                                / numpy.iinfo(samples.dtype).max
                            )
                        samples = samples.astype(numpy.float32)
                        peak = max(peak, float(numpy.abs(samples).max(initial=0.0)))
                        total += float(numpy.square(samples).sum())
                        count += samples.size
            except Exception as e:
                # A track that opens fine can still fail mid-decode (damage
                # past the header); the fields already gathered - duration,
                # format - are still true, so report those rather than
                # failing the whole probe. Matches read_embedded_metadata's
                # precedent of degrading rather than raising.
                logger.debug(f"Decode failed partway through {path}: {e}")
                return info
            if need_frame_count:
                info["frame_count"] = frame_count
            if audio is not None:
                rms = math.sqrt(total / count) if count else 0.0
                info["peak_dbfs"] = _dbfs(peak)
                info["mean_dbfs"] = _dbfs(rms)
        return info
