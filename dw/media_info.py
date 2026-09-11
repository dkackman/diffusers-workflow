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


def probe_media(path, envelope=False):
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

    With `envelope=True` the same decode also reports the level second by
    second, as `envelope: {"interval_seconds": 1.0, "rms_dbfs": [...],
    "peak_dbfs": [...]}` - which is what tells an agent *where* in a track
    something is, rather than only how loud the whole thing was: whether a
    shot is still voiced at its last frame, where a score's quiet passage
    sits, how deep the hole at a seam goes. Off by default, because a
    ten-minute track is 600 numbers nobody asked for and the default
    metadata call has to stay small.

    Args:
        path: The file to probe
        envelope: Also report the per-second level of the soundtrack. The
            list covers what decodes, which for a lossy codec can run a
            fraction of a second past the reported duration - its own
            priming and padding
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
            # One bin per second of the soundtrack, filled as frames decode:
            # [sum of squares, sample count, peak] - the same numbers the
            # whole-track level is made of, kept per second instead of once
            bins = [] if envelope and audio is not None else None
            elapsed = 0  # samples of the soundtrack seen so far
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
                        if bins is not None:
                            elapsed = _fill_envelope(
                                bins, samples, elapsed, audio.rate, int(audio.channels)
                            )
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
                if bins is not None:
                    info["envelope"] = _as_envelope(bins)
        return info


def _as_frame_samples(samples, channels):
    """One decoded audio frame as a (samples, channels) array.

    A planar format decodes to (channels, samples); a packed one decodes to
    (1, samples * channels) interleaved. Both have to become a run of
    samples before they can be cut on a second boundary, or a stereo packed
    frame would be counted as twice as much time as it holds.
    """
    if samples.ndim == 1:
        return samples[:, numpy.newaxis]
    if samples.shape[0] == channels and channels > 1:
        return samples.T
    if samples.shape[0] == 1 and channels > 1:
        return samples.reshape(-1, channels)
    return samples.T if samples.shape[0] < samples.shape[1] else samples


def _fill_envelope(bins, samples, elapsed, rate, channels):
    """Add a decoded audio frame's samples to the per-second bins.

    A bin covers one second of the track regardless of how the decoder
    happened to chop it, so a frame straddling a second boundary is split
    across the two bins rather than counted in whichever one it started in.
    `elapsed` is how many samples of the track came before this frame; the
    new total is returned.
    """
    frame = _as_frame_samples(samples, channels)
    length = frame.shape[0]
    start = 0
    while start < length:
        second = (elapsed + start) // rate
        while len(bins) <= second:
            bins.append([0.0, 0, 0.0])
        # How much of this frame still belongs to the second it is in
        room = int((second + 1) * rate - (elapsed + start))
        stop = min(length, start + max(room, 1))
        piece = frame[start:stop]
        entry = bins[second]
        entry[0] += float(numpy.square(piece).sum())
        entry[1] += int(piece.size)
        entry[2] = max(entry[2], float(numpy.abs(piece).max(initial=0.0)))
        start = stop
    return elapsed + length


def _as_envelope(bins):
    """The per-second bins as the levels an agent reads."""
    return {
        "interval_seconds": 1.0,
        "rms_dbfs": [
            _dbfs(math.sqrt(total / count) if count else 0.0)
            for total, count, _peak in bins
        ],
        "peak_dbfs": [_dbfs(peak) for _total, _count, peak in bins],
    }
