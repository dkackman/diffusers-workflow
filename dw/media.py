"""The one module that opens a media container.

Every `av.open` in the engine is here (`tests/test_media_layering.py`
reads the source tree to keep it so), which is what makes "one call is one
open" something a test can count and a patch can intercept:

- headers: `media_duration`, `container_fps`, `audio_shape`, `video_shape`,
  `probe_metadata`;
- measurement: `probe_media` - duration, format and level of a file, which
  is what an agent that cannot listen checks an audio deliverable with;
- a soundtrack as WAV bytes, or a named slice of one, the form
  get_output_audio hands an agent for a muxed video (whose container it
  cannot play) or for a track too long to send whole (#193);
- decodes that answer arrays and rates, never a step type: `decode_soundtrack`,
  `decode_audio_video`, `decode_rgb_frames`, `read_frames`,
  `read_thumbnails_and_track`.

PyAV only: this runs in the server process, where a request must not pull in
torch or materialise frames.
"""

import io
import logging
import math
import wave

import av
import numpy
from av.audio.resampler import AudioResampler
from PIL import Image

from .dsp import SILENCE_DBFS, dbfs, integrated_lufs, layout_name, true_peak_dbfs

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


def container_fps(path):
    """The picture stream's frame rate from the container's headers alone, or
    None for a file with no picture stream (or none it declares) - what turns
    a shot's frames into seconds without decoding a frame."""
    with av.open(path) as container:
        if not container.streams.video:
            return None
        rate = container.streams.video[0].average_rate
        return float(rate) if rate else None


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
        layout = layout_name(channels, stream.layout)
        resampler = AudioResampler(format="s16", layout=layout, rate=rate)

        # pts is a timestamp on the container's clock, not an offset from
        # this stream's first sample: a stream with an edit list or a
        # non-zero start (which real muxers write) carries a `start_time`,
        # and both the seek target and each frame's clock have to be
        # zeroed against it before they mean seconds into the track -
        # the same anchor `read_frames` uses. Unanchored,
        # `start=2.5` on a track shifted by one second came back from
        # about 1.5 s: the wrong audio, silently.
        anchor = stream.start_time if stream.start_time is not None else 0
        if start > 0:
            # Seek to the keyframe at or before `start`, less one packet of
            # pre-roll: the first packet a decoder sees after a seek is its
            # warm-up, and for AAC and mp3 the samples it yields come out
            # attenuated or silent - so an excerpt at 2.5 s opened with a
            # gap that is not in the file. Landing a packet early hands that
            # warm-up to samples the pts-based trim below drops anyway.
            frame_size = int(stream.codec_context.frame_size or 0)
            preroll = frame_size / rate if frame_size else 0.1
            container.seek(
                int(max(0.0, start - preroll) / stream.time_base) + anchor,
                stream=stream,
                backward=True,
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
                    # round, not floor: float pts arithmetic lands a hair
                    # under the exact sample and floor then keeps one too many
                    samples = samples[int(round((start - chunk_start) * rate)) :]
                    chunk_start = start
                started = True
                if stop is not None and chunk_end > stop:
                    samples = samples[: max(0, int(round((stop - chunk_start) * rate)))]
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


def decode_soundtrack(path):
    """The whole soundtrack of `path` as a float32 (channels, samples) array
    and its sample rate, decoded from the audio stream alone.

    `load_audio` on a video decodes every picture frame to get at the track
    (`load_audio_video`), which on a minutes-long cut is most of the work and
    all of the memory. A caller that only measures the sound - find_loop_bed
    (#218) - reads it here instead. Float rather than extract_audio's s16:
    room tone sits at -70 to -85 dBFS, close enough to 16-bit's floor that
    requantizing would add to what is being measured.
    """
    with av.open(path) as container:
        if not container.streams.audio:
            raise NoSoundtrack(f"{path} has no soundtrack")
        stream = container.streams.audio[0]
        rate = int(stream.rate)
        channels = int(stream.channels)
        layout = layout_name(channels, stream.layout)
        resampler = AudioResampler(format="flt", layout=layout, rate=rate)
        pieces = []
        for frame in container.decode(stream):
            for chunk in resampler.resample(frame):
                pieces.append(chunk.to_ndarray().reshape(-1, channels))
        for chunk in resampler.resample(None):
            pieces.append(chunk.to_ndarray().reshape(-1, channels))

    samples = (
        numpy.concatenate(pieces)
        if pieces
        else numpy.zeros((0, channels), numpy.float32)
    )
    return numpy.ascontiguousarray(to_float32(samples).T), rate


def to_float32(samples):
    """A decoded audio frame (or its ndarray) as float32 in [-1, 1].

    An unsigned format is centred on its midpoint, a signed integer one is
    scaled by its own maximum, and a float one is only cast - the three
    conversions every decode here (the probe, the assessment reader) used to
    spell for itself.
    """
    if isinstance(samples, av.AudioFrame):
        samples = samples.to_ndarray()
    if samples.dtype.kind == "u":
        iinfo = numpy.iinfo(samples.dtype)
        half = (iinfo.max + 1) / 2
        samples = (samples.astype(numpy.float32) - half) / half
    elif samples.dtype.kind == "i":
        samples = samples.astype(numpy.float32) / numpy.iinfo(samples.dtype).max
    return samples.astype(numpy.float32)


def as_frame_samples(samples, channels):
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


def stream_seconds(stream):
    """A stream's own reported duration in seconds, or None."""
    if stream is None or stream.duration is None or stream.time_base is None:
        return None
    return float(stream.duration * stream.time_base)


def video_shape(path):
    """Frame count, fps and size, from the container's own headers where
    they are written and by counting otherwise."""
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"{path} has no video stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        count = int(stream.frames) if stream.frames else None
        if count is None:
            count = sum(1 for _ in container.decode(stream))
        return {
            "frame_count": count,
            "fps": fps,
            "width": int(stream.width),
            "height": int(stream.height),
        }


def read_frames(path, indexes, fit=None):
    """The frames at these indexes, as {index: PIL image}, in one forward
    pass that seeks to the keyframe before each wanted frame rather than
    decoding from the top. Decodes are dropped as soon as they are past.

    `fit(image, index)`, when given, is applied to each frame as it is
    decoded, so a caller tiling many frames never holds one at source size.

    This assumes a constant frame rate, which every file this engine writes
    has (`encode_video` / `export_to_video` write a fixed `fps`): a
    `backward=True` seek lands on the keyframe at or before the target
    timestamp, and `frame.pts * stream.time_base` converts that keyframe's
    own presentation time back to an exact frame index (`round(seconds *
    fps)`) - verified against PyAV 18.1's actual seek landings (a
    single-keyframe short clip, where every seek lands on frame 0; a `g=10`
    multi-keyframe clip, where a seek to frame 95 lands exactly on frame 90;
    and a clip whose packets carry a 5-frame pts offset - an edit list or a
    non-zero start, which real muxers write - where the raw pts arithmetic
    landed 5 frames off until it was anchored on `stream.start_time`).
    `start_pts` is that anchor: pts is a timestamp against the *container's*
    clock, not a frame count from this stream's first frame, so it has to be
    zeroed against wherever this stream actually starts before it means a
    frame index. `position` is then a plain frame counter from that
    landing, decoding forward to the target and dropping what is skipped
    past.
    """
    wanted = sorted(set(int(i) for i in indexes))
    found = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        start_pts = stream.start_time if stream.start_time is not None else 0
        position = 0  # index of the next frame decode() will yield
        for target in wanted:
            if target < position or target - position > 2 * (int(fps) if fps else 24):
                # seek back or a long way forward: land on the keyframe at
                # or before the target, then read up to it. The seek target
                # is a container timestamp too, so it needs the same anchor.
                seconds = target / fps if fps else 0.0
                container.seek(
                    int(seconds / stream.time_base) + start_pts,
                    stream=stream,
                    backward=True,
                )
                position = None
            recovered = False
            for frame in container.decode(stream):
                if position is None:
                    # first frame after a seek says where we landed
                    position = (
                        int(
                            round(
                                float((frame.pts - start_pts) * stream.time_base) * fps
                            )
                        )
                        if fps and frame.pts is not None
                        else 0
                    )
                    if position > target and not recovered:
                        # Landed past the target: the keyframe estimate was
                        # wrong for this file (off-rate or VFR). Reading on
                        # would scan to EOF and blame the caller; read from
                        # the top once instead, which is always correct.
                        container.seek(start_pts, stream=stream, backward=True)
                        position = None
                        recovered = True
                        continue
                if position == target:
                    image = frame.to_image()
                    found[target] = fit(image, target) if fit is not None else image
                    position += 1
                    break
                position += 1
        missing = [i for i in wanted if i not in found]
        if missing:
            raise ValueError(f"could not decode frame(s) {missing} of {path}")
    return found


def decode_audio_video(handle):
    """A path or file object's picture and soundtrack, decoded in one pass.

    Returns (frames, audio, sample_rate, frame_rate): the frames as RGB PIL
    images, the track as float32 (channels, samples) - planar float, the
    layout `AudioVideo` carries - or None when there is none, the audio
    stream's rate (None with no audio stream) and the picture's declared
    rate (None when it states none). The codec's padding is the caller's to
    fit away; building the `AudioVideo` is too.
    """
    frames = []
    chunks = []
    sample_rate = None

    with av.open(handle) as container:
        video_stream = container.streams.video[0]
        frame_rate = (
            float(video_stream.average_rate) if video_stream.average_rate else None
        )
        streams = [video_stream]
        if container.streams.audio:
            audio_stream = container.streams.audio[0]
            streams.append(audio_stream)
            sample_rate = audio_stream.rate
            resampler = AudioResampler(format="fltp")

        for frame in container.decode(*streams):
            if isinstance(frame, av.VideoFrame):
                frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
            else:
                chunks.extend(f.to_ndarray() for f in resampler.resample(frame))

        if sample_rate is not None:
            chunks.extend(f.to_ndarray() for f in resampler.resample(None))

    audio = to_float32(numpy.concatenate(chunks, axis=1)) if chunks else None
    return frames, audio, sample_rate, frame_rate


def decode_rgb_frames(path):
    """Every picture frame of `path` as an (height, width, 3) uint8 array,
    in order."""
    with av.open(path) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def read_thumbnails_and_track(path, thumb_width, thumb_height):
    """Stream a file once into grey thumbnails and its soundtrack.

    Each decoded picture frame is reduced to a `thumb_height` x `thumb_width`
    grey array as it arrives and then dropped, so memory holds thumbnails,
    not pictures. Returns (thumbs, chunks, fps, video_seconds, audio_seconds,
    sample_rate): the thumbnails as a list, the track as a list of float32
    (samples, channels) chunks, and the streams' own rates and durations
    (None where a stream is absent or states none).
    """
    thumbs = []
    chunks = []
    with av.open(path) as container:
        video = container.streams.video[0] if container.streams.video else None
        audio = container.streams.audio[0] if container.streams.audio else None
        if video is None and audio is None:
            raise ValueError(f"{path} has neither a video nor an audio stream")
        fps = float(video.average_rate) if video and video.average_rate else None
        video_seconds = stream_seconds(video)
        audio_seconds = stream_seconds(audio)
        channels = int(audio.channels) if audio is not None else 0
        streams = [s for s in (video, audio) if s is not None]
        for frame in container.decode(*streams):
            if isinstance(frame, av.VideoFrame):
                thumbs.append(
                    frame.reformat(
                        width=thumb_width, height=thumb_height, format="gray"
                    ).to_ndarray()[:thumb_height, :thumb_width]
                )
            elif isinstance(frame, av.AudioFrame):
                chunks.append(as_frame_samples(to_float32(frame), channels))
        sample_rate = int(audio.rate) if audio is not None else None
    return thumbs, chunks, fps, video_seconds, audio_seconds, sample_rate


def _stream_info(container, audio_stream_seconds=False):
    """The header-level description of an opened container: (info, video,
    audio), or (None, None, None) when it holds neither stream.

    `kind`, `fps`, `width`, `height`, `duration_seconds`, `sample_rate` and
    `channels` - what `probe_metadata` answers and `probe_media` starts from.
    `audio_stream_seconds` adds the audio *stream's* own reported duration,
    not the container's: assess's `read_thumbnails_and_track` trims the
    decoded track to this figure, and a lossy mux can report the two slightly
    differently (#426), so a caller that needs to agree with what a probe will
    actually measure reads this rather than duration_seconds.
    """
    video = container.streams.video[0] if container.streams.video else None
    audio = container.streams.audio[0] if container.streams.audio else None
    if video is None and audio is None:
        return None, None, None
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
        if (
            audio_stream_seconds
            and audio.duration is not None
            and audio.time_base is not None
        ):
            info["audio_stream_seconds"] = float(audio.duration * audio.time_base)
    return info, video, audio


class _AudioAccumulator:
    """A soundtrack's levels, gathered frame by frame as the file decodes.

    `add` takes each decoded frame's ndarray; `finish` answers peak, mean,
    integrated loudness and true peak, plus the per-second envelope when one
    was asked for. Samples past the file's reported duration are a lossy
    codec's own priming/padding, not real track content, and are dropped
    rather than measured or allowed to fill (or half-fill) a bin of their own
    (#277); when the duration itself is unknown there is nothing to clip
    against, so the trailing-fragment merge is the fallback.
    """

    def __init__(self, rate, channels, duration_seconds=None, envelope=False):
        self.rate = rate
        self.channels = int(channels)
        self.peak = 0.0
        self.total = 0.0
        self.count = 0
        # One bin per second of the soundtrack: [sum of squares, sample
        # count, peak] - the same numbers the whole-track level is made of,
        # kept per second instead of once
        self.bins = [] if envelope else None
        self.elapsed = 0  # samples of the soundtrack seen so far
        self.max_samples = (
            int(round(duration_seconds * rate))
            if duration_seconds is not None
            else None
        )
        # The whole (trimmed) soundtrack: integrated loudness and true peak
        # are measured over the full track, not per frame
        self.chunks = []

    def add(self, decoded):
        samples = to_float32(decoded)
        self.peak = max(self.peak, float(numpy.abs(samples).max(initial=0.0)))
        self.total += float(numpy.square(samples).sum())
        self.count += samples.size
        frame = as_frame_samples(samples, self.channels)
        full_length = frame.shape[0]
        length = (
            full_length
            if self.max_samples is None
            else max(0, min(full_length, self.max_samples - self.elapsed))
        )
        if self.bins is not None:
            _fill_envelope(self.bins, frame[:length], self.elapsed, self.rate)
        if length > 0:
            self.chunks.append(frame[:length])
        # the frame's *full* length, so a later frame's position is measured
        # against the real track rather than the clipped one
        self.elapsed += full_length

    def finish(self):
        if self.bins is not None and self.max_samples is None:
            _merge_trailing_fragment(self.bins, self.rate)
        rms = math.sqrt(self.total / self.count) if self.count else 0.0
        full = numpy.concatenate(self.chunks, axis=0) if self.chunks else None
        levels = {
            "peak_dbfs": dbfs(self.peak, floor=SILENCE_DBFS),
            "mean_dbfs": dbfs(rms, floor=SILENCE_DBFS),
            "integrated_lufs": integrated_lufs(full, self.rate),
            "true_peak_dbfs": true_peak_dbfs(full),
        }
        if self.bins is not None:
            levels["envelope"] = _as_envelope(self.bins)
        return levels


def probe_media(path, envelope=False):
    """Duration, format and level of an audio or video file, or None.

    Video answers fps, frame_count, width and height, plus the soundtrack's
    sample_rate, channels, peak_dbfs, mean_dbfs, integrated_lufs and
    true_peak_dbfs when it carries one; audio answers the soundtrack fields.
    Levels come from decoding the whole track, which is cheap next to
    generating it.

    peak_dbfs and mean_dbfs are a single sample's level; integrated_lufs is
    the BS.1770 loudness of the whole track (#361) - a sparse voice-over and
    a dense score can share a peak and still sit tens of dB apart in how
    loud they sound. integrated_lufs is None for a track shorter than the
    400 ms gating block or one that is silent throughout - "unmeasurable",
    not zero. true_peak_dbfs is the inter-sample (oversampled) peak BS.1770
    also defines, which can read higher than peak_dbfs when an encoder's
    reconstruction filter rings a decoded peak up past what any single
    sample showed.

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
        envelope: Also report the per-second level of the soundtrack. A lossy
            codec's decode can run a fraction of a second past the file's
            reported duration - its own priming and padding - so samples past
            that duration are dropped rather than filling a bin of their own
            (#277). A track whose *real* length isn't a whole number of
            seconds still gets a genuine final bin shorter than the rest -
            `len(rms_dbfs)` is `ceil(duration_seconds)`, not `floor` (#278)
    """
    try:
        container = av.open(path)
    except Exception as e:
        logger.debug(f"Not probeable as media: {path}: {e}")
        return None
    with container:
        info, video, audio = _stream_info(container, audio_stream_seconds=True)
        if info is None:
            return None

        # Some muxers don't write a frame count up front (0 means "count
        # them"); a soundtrack always needs decoding to measure its level.
        # Do both together, since decoding is a one-way trip through the file.
        need_frame_count = video is not None and not video.frames
        if video is not None and not need_frame_count:
            info["frame_count"] = int(video.frames)
        if not (need_frame_count or audio is not None):
            return info

        accumulator = (
            _AudioAccumulator(
                audio.rate, audio.channels, info.get("duration_seconds"), envelope
            )
            if audio is not None
            else None
        )
        frame_count = 0
        streams = [
            s for s in ((video if need_frame_count else None), audio) if s is not None
        ]
        try:
            for frame in container.decode(*streams):
                if isinstance(frame, av.VideoFrame):
                    frame_count += 1
                elif isinstance(frame, av.AudioFrame):
                    accumulator.add(frame.to_ndarray())
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
        if accumulator is not None:
            info.update(accumulator.finish())
        return info


# Codecs where demuxing (no decode) counts frames exactly - one packet per
# encoded frame, no B-frame reordering or multi-packet-per-frame games -
# because these are what dw's own pipelines actually write. Anything else
# falls back to a real decode (`probe_media`) rather than risk trusting a
# count that might not hold for it.
_DEMUX_COUNTABLE_CODECS = frozenset(
    {"h264", "hevc", "mpeg4", "vp8", "vp9", "av1", "mjpeg", "prores"}
)


def _video_codec_name(video):
    """The video stream's codec name, or None - its own function so a test
    can substitute an unlisted codec without needing a fixture genuinely
    encoded with one."""
    codec_context = getattr(video, "codec_context", None)
    return getattr(codec_context, "name", None) if codec_context is not None else None


def _demuxed_frame_count(path, container, video):
    """The video stream's frame count when its header didn't carry one, or
    None when it cannot be determined at all.

    Demuxing packets without decoding them is exact only for a codec in
    `_DEMUX_COUNTABLE_CODECS`; for anything else - or if the demux itself
    raises partway through - this falls back to `probe_media`'s own decode,
    which is always correct, at the cost of the very decode `probe_metadata`
    otherwise avoids. `frame_count` must never come back silently missing
    when `probe_media` could still supply it (review round 1, B9).
    """
    codec_name = _video_codec_name(video)
    if codec_name in _DEMUX_COUNTABLE_CODECS:
        try:
            return sum(1 for packet in container.demux(video) if packet.size)
        except Exception as e:
            logger.debug(
                f"Demux failed partway through {path}: {e} - "
                "falling back to a decode for its frame count"
            )
    else:
        logger.debug(
            f"{path}: codec {codec_name!r} is not in the demux-safe "
            "allowlist - falling back to a decode for its frame count"
        )
    decoded = probe_media(path)
    return decoded.get("frame_count") if decoded else None


def probe_metadata(path):
    """Header-level view of a media file, or None - what validation needs
    from `probe_media` without paying for what it does not: a full decode.

    Reports the same `kind`, `width`, `height`, `fps`, `frame_count`,
    `duration_seconds`, `sample_rate` and `channels` `probe_media` does, for
    a video or an audio file, and nothing else - no loudness analysis, and
    the soundtrack of a video is never decoded here either, since validation
    only asks about frame counts and shapes.

    A muxer that writes a video stream's frame count into its header (mp4
    does) answers straight from that header - av.open() alone, no
    demuxing, no decoding. One that doesn't (0 means "count them"; mkv is
    the case dw hits) is counted by demuxing that stream's packets without
    decoding them, exact for a codec in `_DEMUX_COUNTABLE_CODECS`: one
    packet per encoded frame, plus a final empty flush packet once the
    stream is exhausted (`packet.size == 0`) that is not a frame and is not
    counted. A codec outside that allowlist, or a demux that raises partway
    through, falls back to `probe_media`'s real decode instead of risking a
    wrong count (`_demuxed_frame_count`).
    """
    try:
        container = av.open(path)
    except Exception as e:
        logger.debug(f"Not probeable as media: {path}: {e}")
        return None
    with container:
        info, video, _audio = _stream_info(container)
        if info is None:
            return None
        if video is not None:
            if video.frames:
                info["frame_count"] = int(video.frames)
            else:
                frame_count = _demuxed_frame_count(path, container, video)
                if frame_count is not None:
                    info["frame_count"] = frame_count
        return info


def _fill_envelope(bins, frame, elapsed, rate):
    """Add a decoded audio frame's samples, as (samples, channels), to the
    per-second bins.

    A bin covers one second of the track regardless of how the decoder
    happened to chop it, so a frame straddling a second boundary is split
    across the two bins rather than counted in whichever one it started in.
    `elapsed` is how many samples of the track came before this frame; the
    caller has already cut away anything past the file's reported duration (a
    lossy codec's own priming/padding, #277), so every sample here is binned.
    """
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


def _merge_trailing_fragment(bins, rate):
    """Fold a short trailing bin into the one before it, in place.

    Fallback for when the file's duration is unknown, so `_fill_envelope`
    had nothing to clip decoding against: a lossy codec's decode can still
    run a fraction of a second past the real track (its own priming and
    padding), which leaves the last bin holding only a handful of samples -
    not a real last second. Read at face value that fragment looks like a
    hole (near -inf, since so little energy lands in so few samples), when
    the actual last second is whatever the bin before it says. Merging need
    only ever touch the last bin: every earlier one was closed out by a full
    second's worth of samples arriving after it. When the duration *is*
    known, `_fill_envelope`'s `max_samples` drops the same padding before it
    ever reaches a bin, which also lets a genuinely partial final second
    (a real duration that isn't a whole number of seconds) stand on its own
    instead of being folded away (#278).
    """
    if len(bins) < 2:
        return
    total, count, peak = bins[-1]
    if count >= rate:
        return
    prev_total, prev_count, prev_peak = bins[-2]
    bins[-2] = [prev_total + total, prev_count + count, max(prev_peak, peak)]
    bins.pop()


def _as_envelope(bins):
    """The per-second bins as the levels an agent reads."""
    return {
        "interval_seconds": 1.0,
        "rms_dbfs": [
            dbfs(math.sqrt(total / count) if count else 0.0, floor=SILENCE_DBFS)
            for total, count, _peak in bins
        ],
        "peak_dbfs": [dbfs(peak, floor=SILENCE_DBFS) for _total, _count, peak in bins],
    }
