"""Reading a video's frames with PyAV: whole, sampled at a rate, a frame
range, thumbnails with the soundtrack, and the decoded audio+video pair - plus
the small readers both halves of the media module share (`_threaded`,
`video_shape`, `stream_seconds`, `to_float32`, `as_frame_samples`).

Split from dw/media.py, which re-exports every name here: callers keep
importing from `dw.media`, and `dw.media` imports from here, never the other
way (the `import_cycles` ratchet in scripts/arch_metrics.py). Probing,
soundtrack decoding, envelopes and faststart stay in dw/media.py.
"""

import logging

import av
import numpy
from av.audio.resampler import AudioResampler
from av.video.reformatter import VideoReformatter
from PIL import Image

logger = logging.getLogger("dw")


def _threaded(stream):
    """A video stream set to decode on every core. libav defaults a
    software h264 decode to one thread; AUTO lets it pick frame and
    slice threading for the codec. Header-only reads do not call this."""
    stream.thread_type = "AUTO"
    return stream


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
            count = sum(1 for _ in container.decode(_threaded(stream)))
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
        # Not _threaded: frame threads hold frames in flight, and a backward
        # seek after a decoded frame then yields stale ones (tests/test_media_decode_threads.py)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        start_pts = stream.start_time if stream.start_time is not None else 0
        position = 0  # index of the next frame decode() will yield
        # One decode generator for every target: a fresh `decode()` per
        # target restarts from a demuxer already at EOF when the previous
        # target sat on the last packets, and raises EOFError (#654)
        decoded = container.decode(stream)
        for target in wanted:
            if _seek_needed(target, position, fps):
                # seek back or a long way forward: land on the keyframe at
                # or before the target, then read up to it
                _seek_before(container, stream, target, fps, start_pts)
                position = None
            recovered = False
            for frame in decoded:
                if position is None:
                    # first frame after a seek says where we landed
                    position = _landed_at(frame, stream, fps, start_pts)
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


def sample_frames(path, fps, duration, size, crop=False):
    """A clip sampled at `fps` for `duration` seconds, as a numpy array.

    Frame k (of int(fps * duration), as upstream MMAudio truncates) is source
    frame round(k / fps * source_fps); indexes past the clip's last frame are
    dropped, so a short clip returns fewer frames, and an up-rate repeats
    some. Each frame is fitted as it decodes, so none is held at source size.

    `crop=False` is MMAudio's CLIP input (8 fps): bicubic to size x size,
    uint8 (N, size, size, 3). `crop=True` is its Synchformer input (25 fps,
    224): bicubic to a short side of `size`, centre crop (torchvision's
    CenterCrop rounding), float32 (N, size, size, 3) in [-1, 1] (mean 0.5,
    std 0.5). After mmaudio/eval_utils.py `load_video` at commit 974010a0.
    """
    if fps <= 0 or duration <= 0 or size <= 0:
        raise ValueError("fps, duration and size must be positive")
    shape = video_shape(path)  # raises for a file with no video stream
    if not shape["fps"]:
        raise ValueError(f"{path} declares no frame rate")
    indexes = [round(k / fps * shape["fps"]) for k in range(int(fps * duration))]
    indexes = [i for i in indexes if i < shape["frame_count"]]

    def fit(image, _index):
        if crop:
            scale = size / min(image.size)
            width, height = (max(size, round(side * scale)) for side in image.size)
            image = image.resize((width, height), Image.BICUBIC)
            left, top = (int(round((side - size) / 2.0)) for side in (width, height))
            return image.crop((left, top, left + size, top + size))
        return image.resize((size, size), Image.BICUBIC)

    found = read_frames(path, indexes, fit=fit) if indexes else {}
    frames = (
        numpy.stack([numpy.asarray(found[i]) for i in indexes])
        if indexes
        else (numpy.zeros((0, size, size, 3), numpy.uint8))
    )
    return frames.astype(numpy.float32) / 127.5 - 1 if crop else frames


def _seek_needed(target, position, fps):
    """Whether reaching frame `target` from `position` takes a seek: going
    back, or a long way forward (more than two seconds of decoding)."""
    return target < position or target - position > 2 * (int(fps) if fps else 24)


def _seek_before(container, stream, target, fps, start_pts):
    """Land on the keyframe at or before frame `target`. The seek target is
    a container timestamp too, so it needs the same `start_pts` anchor."""
    seconds = target / fps if fps else 0.0
    container.seek(
        int(seconds / stream.time_base) + start_pts,
        stream=stream,
        backward=True,
    )


def _landed_at(frame, stream, fps, start_pts):
    """The frame index of the first frame decoded after a seek."""
    if not fps or frame.pts is None:
        return 0
    return int(round(float((frame.pts - start_pts) * stream.time_base) * fps))


class ShortFrameRange(ValueError):
    """`read_frame_range` reached the end of the file before `stop`;
    `decoded` is the frame index the decode had reached."""

    def __init__(self, path, start, stop, decoded):
        super().__init__(
            f"could not decode frames {start}..{stop - 1} of {path}: the file "
            f"ended after {decoded} frames"
        )
        self.decoded = decoded


def read_frame_range(path, start, stop):
    """The frames `[start, stop)` of `path` as one (n, height, width, 3)
    uint8 array, from one keyframe seek and one forward decode.

    `read_frames`' seek, for a run of frames rather than a set: the same
    `start_time` anchor, the same read-from-the-top fallback when the seek
    lands past `start`, and one decode generator (#654). Each frame is
    converted to `rgb24` exactly as `decode_audio_video` converts it, so the
    run is byte for byte what a full decode holds at those indexes, without
    the rest of the file ever being resident (#695).

    Raises `ShortFrameRange` when the file ends before `stop`.
    """
    start, stop = int(start), int(stop)
    if start < 0 or stop <= start:
        raise ValueError(
            f"read_frame_range needs 0 <= start < stop, got {start} and {stop}"
        )
    out = None
    with av.open(path) as container:
        # Not _threaded: frame threads hold frames in flight, and a backward
        # seek after a decoded frame then yields stale ones (tests/test_media_decode_threads.py)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        start_pts = stream.start_time if stream.start_time is not None else 0
        position = 0
        decoded = container.decode(stream)
        if _seek_needed(start, position, fps):
            _seek_before(container, stream, start, fps, start_pts)
            position = None
        recovered = False
        for frame in decoded:
            if position is None:
                position = _landed_at(frame, stream, fps, start_pts)
                if position > start and not recovered:
                    # Landed past the start: the keyframe estimate was wrong
                    # for this file; read from the top once instead
                    container.seek(start_pts, stream=stream, backward=True)
                    position = None
                    recovered = True
                    continue
            if position >= start:
                picture = frame.to_ndarray(format="rgb24")
                if out is None:
                    out = numpy.empty((stop - start,) + picture.shape, numpy.uint8)
                out[position - start] = picture
            position += 1
            if position >= stop:
                return out
    raise ShortFrameRange(path, start, stop, position or 0)


class ForwardFrameReader:
    """`read_frame_range` for a run of reads that moves forward through
    `path` - a clip's frames at a time - from one open container and one
    forward decode, so no frame is decoded twice and no read seeks.

    `reader(start, stop)` returns frames `[start, stop)` as one
    (n, height, width, 3) uint8 array, byte for byte what `read_frame_range`
    returns: the same unthreaded decode and `rgb24` conversion, counted from
    the top. A range inside the last one returned is sliced from it (a
    clip's last frame repeated past the take's end); a range behind it falls
    back to `read_frame_range`. Raises `ShortFrameRange` when the file ends
    before `stop`. `close()` releases the container and forgets the position
    and the held range, so a later read starts the decode over from the
    top rather than labelling frame 0 as the old position."""

    def __init__(self, path):
        self.path = path
        self._container = None
        self._decoded = None
        self._position = 0
        self._held = None  # (start, frames) of the last range returned

    def __call__(self, start, stop):
        start, stop = int(start), int(stop)
        if start < 0 or stop <= start:
            raise ValueError(
                f"read_frame_range needs 0 <= start < stop, got {start} and {stop}"
            )
        if self._held is not None:
            held_start, held = self._held
            if held_start <= start and stop <= held_start + len(held):
                return held[start - held_start : stop - held_start]
        if start < self._position:
            return read_frame_range(self.path, start, stop)
        if self._container is None:
            self._container = av.open(self.path)
            # Not _threaded, as read_frame_range: the same frames, in order
            self._decoded = self._container.decode(self._container.streams.video[0])
        out = None
        while self._position < stop:
            frame = next(self._decoded, None)
            if frame is None:
                raise ShortFrameRange(self.path, start, stop, self._position)
            if self._position >= start:
                picture = frame.to_ndarray(format="rgb24")
                if out is None:
                    out = numpy.empty((stop - start,) + picture.shape, numpy.uint8)
                out[self._position - start] = picture
            self._position += 1
        self._held = (start, out)
        return out

    def close(self):
        if self._container is not None:
            self._container.close()
            self._container = self._decoded = None
        # A read after close reopens at frame 0, so it must count from there
        self._position = 0
        self._held = None


def count_video_frames(path):
    """The picture frame count of `path`, by decoding every frame rather
    than reading the header. Nothing is held but the frame in hand."""
    with av.open(path) as container:
        return sum(1 for _ in container.decode(_threaded(container.streams.video[0])))


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
        video_stream = _threaded(container.streams.video[0])
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
        return [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(_threaded(container.streams.video[0]))
        ]


def iter_rgb_frames(path):
    """The picture frames of `path` as (height, width, 3) uint8 arrays, one at
    a time, so a caller can stop at the frame it is looking for."""
    with av.open(path) as container:
        for frame in container.decode(_threaded(container.streams.video[0])):
            yield frame.to_ndarray(format="rgb24")


def read_thumbnails_and_track(
    path, thumb_width, thumb_height, look=None, look_width=128, look_height=72
):
    """Stream a file once into grey thumbnails and its soundtrack.

    Each decoded picture frame is reduced to a `thumb_height` x `thumb_width`
    grey array as it arrives and then dropped, so memory holds thumbnails,
    not pictures. With `look`, each frame is also reformatted to a
    `look_height` x `look_width` rgb24 array and handed to it at once, and
    only what it returns is kept (#1031). Returns (thumbs, chunks, fps,
    video_seconds, audio_seconds, sample_rate, looks): the thumbnails as a
    list, the track as a list of float32 (samples, channels) chunks, the
    streams' own rates and durations (None where a stream is absent or
    states none), and `look`'s results as a list (empty without one).
    """
    thumbs = []
    looks = []
    chunks = []
    # Its own scaler: frame.reformat shares one, and alternating two output
    # sizes through it rebuilds the context every frame - 10x the cost
    look_scaler = VideoReformatter()
    with av.open(path) as container:
        video = (
            _threaded(container.streams.video[0]) if container.streams.video else None
        )
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
                if look is not None:
                    looks.append(
                        look(
                            look_scaler.reformat(
                                frame,
                                width=look_width,
                                height=look_height,
                                format="rgb24",
                            ).to_ndarray()[:look_height, :look_width]
                        )
                    )
            elif isinstance(frame, av.AudioFrame):
                chunks.append(as_frame_samples(to_float32(frame), channels))
        sample_rate = int(audio.rate) if audio is not None else None
    return thumbs, chunks, fps, video_seconds, audio_seconds, sample_rate, looks
