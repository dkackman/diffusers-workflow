"""A soundtrack, or a named slice of one, as WAV bytes - the form
get_output_audio hands an agent for a muxed video (whose container it
cannot play) or for a track too long to send whole (#193).

PyAV only, like media_info: this runs in the server process, where a
request must not pull in torch or materialise frames.
"""

import io
import logging
import math
import wave

import av
import numpy
from av.audio.resampler import AudioResampler

logger = logging.getLogger("dw")

# The most a soundtrack may be as base64 before the gallery route refuses to
# extract it whole - the twin of dw_mcp/media.py's MAX_RETURNED_BYTES (the
# MCP package's cap on any inline payload). Two constants because the two
# packages do not import each other; a whole track over this is cut off at
# the header, before a frame is decoded, rather than decoded, shipped and
# then refused by the client.
MAX_INLINE_AUDIO_BYTES = 4 * 1024 * 1024


class NoSoundtrack(ValueError):
    """The file has no audio stream to extract."""


def _container_duration(container):
    """The container's own duration (seconds) from its header alone - no
    stream is decoded to produce this number."""
    return (
        float(container.duration / av.time_base)
        if container.duration is not None
        else None
    )


def media_duration(path):
    """The container's duration (seconds), read from its header alone - no
    stream is decoded. `extract_audio` needs the same figure as `total`;
    this is that computation, exposed for a caller that wants only the
    length, since decoding to get one number defeats the "nothing to
    extract" case (serving an audio file whole - #193)."""
    with av.open(path) as container:
        return _container_duration(container)


def audio_shape(path):
    """The soundtrack's duration (seconds), sample rate and channel count
    from the container's headers alone - what projecting the size of a
    whole-track WAV needs, and nothing decoded to get it. `None` when the
    file has no audio stream."""
    with av.open(path) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        return {
            "duration_seconds": _container_duration(container),
            "sample_rate": int(stream.rate),
            "channels": int(stream.channels),
        }


def projected_wav_base64_size(shape):
    """How many bytes the whole track would be as base64 16-bit PCM WAV -
    `extract_audio`'s output for the same file, sized from `audio_shape`
    without producing it. `None` when the container states no duration."""
    if shape is None or shape["duration_seconds"] is None:
        return None
    pcm = int(shape["duration_seconds"] * shape["sample_rate"] * shape["channels"] * 2)
    return 4 * math.ceil(pcm / 3)


def extract_audio(path, start=None, duration=None):
    """The soundtrack of `path` as 16-bit PCM WAV bytes, plus what was cut.

    With `start` and `duration` (seconds) only that slice is decoded and
    returned; the info dict says so (`excerpt: True`) and carries the whole
    track's length as `of_seconds`, so a slice always names itself. A
    slice that runs past the end is clipped to it; a `start` past the end
    is refused, since an empty answer would read as a silent track.

    Returns:
        (wav_bytes, info) with info = {sample_rate, channels,
        duration_seconds, of_seconds, start, excerpt}
    """
    excerpt = start is not None or duration is not None
    start = float(start or 0.0)
    if excerpt and (duration is None or float(duration) <= 0):
        raise ValueError("An excerpt needs a duration above zero")
    if start < 0:
        raise ValueError("An excerpt cannot start before zero")

    with av.open(path) as container:
        if not container.streams.audio:
            raise NoSoundtrack(f"{path} has no soundtrack")
        stream = container.streams.audio[0]
        total = _container_duration(container)
        if total is not None and start >= total:
            raise ValueError(
                f"start {start:.2f}s is past the end of a {total:.2f}s track"
            )
        stop = start + float(duration) if excerpt else None
        if stop is not None and total is not None:
            stop = min(stop, total)

        rate = int(stream.rate)
        channels = int(stream.channels)
        layout = "stereo" if channels == 2 else ("mono" if channels == 1 else stream.layout.name)
        resampler = AudioResampler(format="s16", layout=layout, rate=rate)

        # pts is a timestamp on the container's clock, not an offset from
        # this stream's first sample: a stream with an edit list or a
        # non-zero start (which real muxers write) carries a `start_time`,
        # and both the seek target and each frame's clock have to be
        # zeroed against it before they mean seconds into the track -
        # the same anchor `_read_frames` in media_frames uses. Unanchored,
        # `start=2.5` on a track shifted by one second came back from
        # about 1.5 s: the wrong audio, silently.
        anchor = stream.start_time if stream.start_time is not None else 0
        if start > 0:
            # Seek to the keyframe at or before `start`; the frames decoded
            # before `start` are then dropped sample-accurately below
            container.seek(
                int(start / stream.time_base) + anchor, stream=stream, backward=True
            )

        pieces = []
        seen = 0  # samples decoded so far - the clock for a frame with no pts
        started = False
        done = False  # Break outer loop when stop time is reached
        sought = start > 0
        restarted = False
        for frame in container.decode(stream):
            if done:
                break
            if frame.pts is None and sought and not restarted:
                # No pts means the seek's landing cannot be known, so the
                # sample count is the only clock there is - and it has to
                # count from the top. Start over once and read to `start`.
                container.seek(0, stream=stream, backward=True)
                resampler = AudioResampler(format="s16", layout=layout, rate=rate)
                restarted = True
                seen = 0
                continue
            frame_start = (
                float((frame.pts - anchor) * stream.time_base)
                if frame.pts is not None
                else seen / rate
            )
            for chunk in resampler.resample(frame):
                samples = chunk.to_ndarray()  # (1, samples * channels) packed s16
                samples = samples.reshape(-1, channels)
                chunk_start = frame_start
                chunk_end = chunk_start + samples.shape[0] / rate
                seen += samples.shape[0]
                frame_start = chunk_end  # a frame yielding two chunks: the second follows the first
                if chunk_end <= start:
                    continue
                if not started and chunk_start < start:
                    samples = samples[int((start - chunk_start) * rate) :]
                    chunk_start = start
                started = True
                if stop is not None and chunk_end > stop:
                    samples = samples[: max(0, int((stop - chunk_start) * rate))]
                pieces.append(samples)
                if stop is not None and chunk_end >= stop:
                    done = True
                    break
        # Flush the resampler whatever ended the loop: it may still hold
        # samples that belong before `stop`. Anything past `stop` is cut.
        held = sum(p.shape[0] for p in pieces)
        for chunk in resampler.resample(None):
            samples = chunk.to_ndarray().reshape(-1, channels)
            if stop is not None:
                room = max(0, int(round((stop - start) * rate)) - held)
                samples = samples[:room]
            if samples.shape[0]:
                pieces.append(samples)
                held += samples.shape[0]

    pcm = numpy.concatenate(pieces) if pieces else numpy.zeros((0, channels), "<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "w") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.astype("<i2").tobytes())

    returned = pcm.shape[0] / rate
    return buffer.getvalue(), {
        "sample_rate": rate,
        "channels": channels,
        "duration_seconds": returned,
        "of_seconds": total if total is not None else returned,
        "start": start,
        "excerpt": excerpt,
    }
