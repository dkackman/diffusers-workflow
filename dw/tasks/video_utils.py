"""Frame access for videos in any of the shapes results carry them.

A video artifact can be a list of PIL images, a numpy array of frames
(frames, height, width, channels), a torch tensor (frames first, channels
first or last), or an AudioVideo pairing frames with their generated
soundtrack. extract_frame gives tasks and the segment-chaining loop one way
to pull a single frame out of any of them, always as a PIL image.
"""

import logging
import math
import re

import numpy
import torch
from PIL import Image

from ..result import AudioVideo

logger = logging.getLogger("dw")

# A location that names a scheme is a URL, whatever the scheme
_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def process_video(video, processor, device, kwargs):
    processor = processor.lower()

    if processor == "get_frame":
        return get_frame(video, kwargs.get("frame_index", 0))

    if processor == "get_last_frame":
        return get_frame(video, -1)

    if processor == "get_first_frame":
        return get_frame(video, 0)

    raise Exception(f"Unknown video processor type: {processor}")


class VideoFileReference:
    """A 'video' argument realized to a file on disk rather than an in-memory
    clip - built by dw/arguments.py's _realize_lazy_frame_arguments so
    get_frame can seek to the one frame it needs instead of decoding the
    whole file (#367), and so an assessment probe streams the file, soundtrack
    and all (#387). Not a public shape; nothing else constructs or consumes
    one."""

    __slots__ = ("path",)

    def __init__(self, path):
        self.path = path


def get_frame(video, frame_index=0):
    """Pull one frame out of a video as a PIL image.

    Args:
        video: List of PIL images, numpy array or torch tensor of frames, an
            AudioVideo, a one-video batch wrapping any of those, or a
            VideoFileReference naming a file this call reads by seeking
            rather than decoding in full
        frame_index: Frame to extract, 0-based; negative indexes count from
            the end (-1 is the last frame). Past either end of the clip
            raises an error naming the clip's frame count

    Returns:
        The frame as a PIL image
    """
    if isinstance(video, VideoFileReference):
        from ..media_frames import frames_at

        return frames_at(video.path, [f"frame:{frame_index}"])[0]["image"]
    return extract_frame(video, frame_index)


def extract_frame(video, index):
    """Pull one frame out of a video, whatever the video's in-memory shape.

    Args:
        video: List of PIL images, numpy array or torch tensor of frames,
            an AudioVideo, or a one-video batch wrapping any of those
        index: Frame to extract; negative indexes count from the end

    Returns:
        The frame as a PIL image. Frames that already are PIL images are
        returned as-is, not copied.
    """
    return _to_pil(_frames_of(video)[index])


def frame_count(video):
    """Number of frames in a video of any supported shape."""
    return len(_frames_of(video))


def check_same_frame_size(clips, task_name):
    """Refuse to join clips whose frames disagree in size.

    Args:
        clips: The clips about to be joined, each a PIL frame list or a
            (frames, height, width, channels) array
        task_name: Named in the error

    Joining is frame-by-frame concatenation, which either fails deep in numpy
    or, for a PIL list, produces a film that changes size mid-cut. Naming the
    two sizes points at the shot that was rendered differently rather than at
    the join.
    """
    sizes = []
    for clip in clips:
        if isinstance(clip, numpy.ndarray):
            sizes.append((int(clip.shape[2]), int(clip.shape[1])))
        else:
            sizes.append(tuple(clip[0].size) if len(clip) else None)
    first = next((size for size in sizes if size is not None), None)
    for index, size in enumerate(sizes):
        if size is not None and size != first:
            raise ValueError(
                f"{task_name} needs every video at one size: video 0 is "
                f"{first[0]}x{first[1]}, video {index} is {size[0]}x{size[1]}"
            )


def frames_as_pil_list(video):
    """The video's frames as a list of PIL images.

    Frames that already are PIL images are carried over by identity; array and
    tensor frames are converted the way extract_frame converts them.
    """
    return [_to_pil(frame) for frame in _frames_of(video)]


def frames_as_array(video):
    """The video's frames as one (frames, height, width, channels) uint8 array.

    The shape an argument that takes frames rather than a video wants - LTX-2's
    keyframe conditions and IC-LoRA references, which the workflow hands what an
    earlier step generated. One array is also one artifact, where a list of frames
    would become one artifact per frame and multiply the step that consumed it.

    Frames that already are a channels-last RGB array are converted in a single
    operation; anything else goes through the same per-frame conversion
    extract_frame uses.
    """
    frames = _frames_of(video)

    if isinstance(frames, numpy.ndarray) and frames.ndim == 4 and frames.shape[-1] == 3:
        if frames.dtype == numpy.uint8:
            return frames
        # Float frames are [0, 1] - diffusers' np output convention
        return (numpy.clip(frames, 0.0, 1.0) * 255).round().astype(numpy.uint8)

    return numpy.stack(
        [numpy.asarray(_to_pil(frame).convert("RGB")) for frame in frames]
    )


def loop_frames(video, num_frames):
    """Task command: a run of exactly `num_frames` frames, made by repeating
    what it is given.

    The video analogue of `loop_audio`, and it exists for the same reason: a
    conditioning input has a length its model was trained to read, and the
    material to hand is usually shorter. LTX-2.5's Ingredients IC-LoRA is
    the live case - the reference is a single still sheet, and the model
    wants it as a static video of at least 121 frames at the output's own
    length, because a shorter reference misses the 121-frame read bucket it
    was trained on.

    A still is repeated; a run of frames laps round from its first frame.
    No crossfade, unlike the audio version: these frames are read as
    reference latents rather than watched, so a visible cut at the lap is
    not a defect and blending two frames of a reference sheet would be.

    Args:
        video: Frames in any shape a result carries, or a still image - but
            the `video` argument loads *video files* by convention (#347), so
            a still on disk has to be passed as
            `{"media_type": "image", "location": "asset:x.png"}` rather than
            a bare path or `asset:`/`output:` reference; a still made earlier
            in the same workflow is `previous_result:<image step>`
        num_frames: How many frames to hand back, one or more

    Returns:
        A (num_frames, height, width, channels) uint8 array - one artifact,
        the shape an argument that takes frames wants
    """
    if isinstance(num_frames, str):
        try:
            num_frames = int(num_frames)
        except ValueError:
            raise ValueError(
                f"loop_frames needs 'num_frames' as a whole number, got {num_frames!r}"
            )
    if not isinstance(num_frames, int) or isinstance(num_frames, bool):
        raise ValueError(
            f"loop_frames needs 'num_frames' as a whole number, got {num_frames!r}"
        )
    if num_frames < 1:
        raise ValueError(
            f"loop_frames needs 'num_frames' of at least 1, got {num_frames}"
        )

    # A lone still is the Ingredients case, and `_frames_of` does not take
    # one - a reference sheet is an image, not a one-frame video
    frames = frames_as_array([video] if _is_frame(video) else video)
    if len(frames) == 0:
        raise ValueError("loop_frames was given no frames to repeat")
    laps = -(-num_frames // len(frames))  # ceiling, so the last lap is trimmed
    return numpy.concatenate([frames] * laps, axis=0)[:num_frames]


def frame_grid(video, count=12, columns=None, tile_width=320, label=True):
    """Task command: tile evenly sampled frames of a video into one contact
    sheet - a preview of a clip's shape without authoring a frames-extraction
    workflow (#245).

    Args:
        video: Frames in any shape a result carries
        count: How many frames to sample, spaced evenly across the clip's
            full duration (including its first and last frame). Clamped to
            the clip's own frame count when the clip is shorter
        columns: Tiles per row. Defaults to a grid biased wide - clips are
            usually landscape - with the last row left-justified when
            `count` is not a perfect multiple of it
        tile_width: Width in pixels of each tile; height follows the source
            frame's aspect ratio
        label: Burn the sampled timestamp (or frame index, when the video
            carries no frame rate) into each tile's corner

    Returns:
        One PIL image, the tiled contact sheet
    """
    count = _positive_int(count, "frame_grid", "count")
    if columns is not None:
        columns = _positive_int(columns, "frame_grid", "columns")
    tile_width = _positive_int(tile_width, "frame_grid", "tile_width")
    if not isinstance(label, bool):
        raise ValueError(f"frame_grid needs 'label' as true or false, got {label!r}")

    total = frame_count(video)
    if total == 0:
        raise ValueError("frame_grid was given a video with no frames")
    count = min(count, total)
    fps = getattr(video, "fps", None)

    indices = _evenly_spaced_indices(total, count)
    tiles = [
        _grid_tile(extract_frame(video, index), index, fps, tile_width, label)
        for index in indices
    ]

    if columns is None:
        columns = _default_columns(len(tiles))
    return _compose_grid(tiles, columns)


def _positive_int(value, command, name):
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            raise ValueError(
                f"{command} needs '{name}' as a whole number, got {value!r}"
            )
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{command} needs '{name}' as a whole number, got {value!r}")
    if value < 1:
        raise ValueError(f"{command} needs '{name}' of at least 1, got {value}")
    return value


def _evenly_spaced_indices(total, count):
    """`count` frame indices spaced evenly across [0, total - 1], inclusive
    of both ends. Rounding can coincide two spacings on one index in a short
    clip; those collapse rather than repeating the same frame as a tile."""
    if count == 1:
        return [0]
    raw = numpy.linspace(0, total - 1, num=count)
    seen = []
    for value in raw.round().astype(int).tolist():
        if not seen or seen[-1] != value:
            seen.append(value)
    return seen


def _default_columns(count):
    """A grid biased wide: rows no more than columns, columns >= sqrt(count)."""
    rows = math.isqrt(count) or 1
    return math.ceil(count / rows)


def _grid_tile(frame, index, fps, tile_width, label):
    tile_height = max(1, round(frame.height * tile_width / frame.width))
    tile = frame.resize((tile_width, tile_height), Image.LANCZOS).convert("RGB")
    if not label:
        return tile

    from PIL import ImageDraw, ImageFont

    text = _format_timestamp(index, fps) if fps else f"#{index}"
    draw = ImageDraw.Draw(tile)
    font_size = max(10, tile_width // 16)
    try:
        font = ImageFont.truetype("Arial", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default(size=font_size)
    draw.text(
        (4, 4), text, font=font, fill="white", stroke_width=2, stroke_fill="black"
    )
    return tile


def _format_timestamp(index, fps):
    seconds = index / fps
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes):02d}:{remainder:04.1f}"


def _compose_grid(tiles, columns):
    tile_width, tile_height = tiles[0].size
    rows = math.ceil(len(tiles) / columns)
    grid = Image.new("RGB", (columns * tile_width, rows * tile_height), (0, 0, 0))
    for position, tile in enumerate(tiles):
        row, col = divmod(position, columns)
        grid.paste(tile, (col * tile_width, row * tile_height))
    return grid


def is_video(value):
    """Whether a value is a run of frames rather than one image.

    An AudioVideo, a 4-dim frame array or tensor, or a list of frames. A
    single PIL image, a 3-dim array (one frame) and anything else is not.
    """
    if isinstance(value, AudioVideo):
        return True
    if isinstance(value, list):
        return len(value) > 0 and all(_is_frame(item) for item in value)
    if isinstance(value, numpy.ndarray) or torch.is_tensor(value):
        return value.ndim == 4 or (value.ndim == 5 and value.shape[0] == 1)
    return False


def _frames_of(video):
    """Unwrap containers until an indexable run of frames remains."""
    if isinstance(video, AudioVideo):
        return _frames_of(video.frames)

    if isinstance(video, list):
        # A one-video batch - [[frame, ...]] or [ndarray] - unwraps to the video;
        # a single-frame video - [frame] - is already the frames
        if len(video) == 1 and not _is_frame(video[0]):
            return _frames_of(video[0])
        return video

    if isinstance(video, numpy.ndarray):
        if video.ndim == 3:  # a lone frame
            return video[numpy.newaxis, ...]
        if video.ndim == 5 and video.shape[0] == 1:  # a one-video batch
            return video[0]
        return video

    if torch.is_tensor(video):
        tensor = video.detach().cpu()
        if tensor.ndim == 5 and tensor.shape[0] == 1:  # a one-video batch
            tensor = tensor[0]
        if tensor.ndim == 3:  # a lone frame
            tensor = tensor.unsqueeze(0)
        return tensor

    raise TypeError(f"Cannot extract frames from a {type(video).__name__}")


def _is_frame(item):
    """A single image: PIL, or a 3-dim array/tensor (height, width, channels)."""
    if isinstance(item, Image.Image):
        return True
    if isinstance(item, numpy.ndarray) or torch.is_tensor(item):
        return item.ndim == 3
    return False


def _to_pil(frame):
    """Convert one frame to a PIL image; PIL frames pass through untouched."""
    if isinstance(frame, Image.Image):
        return frame

    if torch.is_tensor(frame):
        frame = frame.detach().cpu().float().numpy()

    if isinstance(frame, numpy.ndarray):
        if frame.ndim != 3:
            raise ValueError(f"A frame must have 3 dimensions, got {frame.ndim}")

        # Channels-first (C, H, W) -> channels-last, the layout PIL expects
        if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = numpy.moveaxis(frame, 0, -1)

        if frame.dtype != numpy.uint8:
            # Float frames are [0, 1] - diffusers' np output convention
            frame = (numpy.clip(frame, 0.0, 1.0) * 255).round().astype(numpy.uint8)

        if frame.shape[-1] == 1:  # grayscale
            frame = frame[..., 0]

        return Image.fromarray(frame)

    raise TypeError(f"Cannot convert a {type(frame).__name__} to an image")


class FrameList(list):
    """The frames of a video file, carrying the rate the file plays at.

    `load_video` answers a plain list of images, which is what every
    pipeline argument and every task wants - and which says nothing about
    how fast those frames are meant to run. A step handed a 24 fps file
    then wrote it back at `result.fps`'s default of 8, three times long,
    with its soundtrack finishing a third of the way in and nothing said
    about it (#104, the file-loading half of #84). A list subclass keeps
    every consumer working unchanged while `getattr(video, "fps", None)` -
    the question AudioVideo, concat_videos and interpolate_frames already
    ask - gets a real answer.
    """

    def __init__(self, frames, fps=None):
        super().__init__(frames)
        self.fps = fps


def file_fps(path):
    """The rate a video file declares, or None - a container that will not
    open, carries no video stream or states no rate is a rate we do not
    know, never an error: the caller is loading frames it has already read.
    """
    try:
        import av

        with av.open(path) as container:
            stream = container.streams.video[0] if container.streams.video else None
            return (
                float(stream.average_rate) if stream and stream.average_rate else None
            )
    except Exception as e:
        logger.debug(f"No frame rate for {path}: {e}")
        return None


def load_audio_video(location, base_dir=None):
    """Load a video file - frames and the audio muxed with them - as an AudioVideo.

    `load_video` reads frames only, so a file written by an earlier run comes
    back silent. Reading both streams here is what lets a step join videos that
    are already on disk - the shots of an earlier run picked back up by name -
    without dropping the audio those runs generated alongside them.

    Args:
        location: Local path, or an http(s) URL, of a video file
        base_dir: Directory a relative path is resolved against

    Returns:
        An AudioVideo holding the frames as PIL images and, when the file
        carries an audio stream, its waveform as a (channels, samples) float32
        array with the stream's sample rate
    """
    from ..security import ALLOWED_VIDEO_EXTENSIONS, validate_file_extension
    from ..locations import validate_media_path, validate_media_url

    if _URL_SCHEME.match(location):
        import io
        import requests

        # Any other scheme - ftp:, file:, data: - is refused here rather than
        # falling through to be read as a relative path that happens to
        # contain a colon. An http(s) one still has to name a host outside
        # this deployment (dw/locations.py)
        validated_url = validate_media_url(location, "a video argument")
        logger.debug(f"Downloading video from {validated_url}")
        response = requests.get(validated_url, timeout=300)
        response.raise_for_status()
        handle = io.BytesIO(response.content)
    else:
        validated_path = validate_media_path(location, base_dir, "a video argument")
        validate_file_extension(validated_path, ALLOWED_VIDEO_EXTENSIONS)
        logger.debug(f"Reading video from {validated_path}")
        handle = validated_path

    return _decode_audio_video(handle)


def _decode_audio_video(handle):
    """Decode a path or file object's video and audio streams in one pass."""
    import av
    from av.audio.resampler import AudioResampler

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
            # Planar float is the layout AudioVideo carries: (channels, samples)
            resampler = AudioResampler(format="fltp")

        for frame in container.decode(*streams):
            if isinstance(frame, av.VideoFrame):
                frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
            else:
                chunks.extend(f.to_ndarray() for f in resampler.resample(frame))

        if sample_rate is not None:
            chunks.extend(f.to_ndarray() for f in resampler.resample(None))

    audio = numpy.concatenate(chunks, axis=1).astype(numpy.float32) if chunks else None
    if audio is not None and frame_rate:
        audio = _fit_audio_to_frames(audio, len(frames), frame_rate, sample_rate)
    logger.debug(
        f"Decoded {len(frames)} frames and "
        f"{audio.shape[1] if audio is not None else 0} audio samples"
    )
    # The file's own rate travels with it: a step that joins videos read
    # from disk knows what to write them back at without being told (#84).
    # A file carries no shots - the manifest that recorded them is the run's,
    # not the file's
    return AudioVideo(
        frames, audio, sample_rate if audio is not None else None, fps=frame_rate
    )


# How far a decoded track may be off the frames' own duration and still be
# treated as codec padding rather than a track of its own length. AAC codes
# 1024 samples at a time, so a file's audio runs up to one such block long -
# a hundredth of a second, which accumulates into visible lip-sync drift once
# a dozen shots are joined end to end
AUDIO_FIT_TOLERANCE_SECONDS = 0.25


def _fit_audio_to_frames(audio, frame_count, frame_rate, sample_rate):
    """Trim or pad a decoded track to exactly the frames' own duration.

    Only when the difference is codec padding. A track that genuinely runs to
    a different length than the picture - a song laid over a short clip - is
    left alone.

    `audio` may be a numpy array (the decode path) or a torch tensor still on
    its generating device (an in-memory pipeline output, #197) - the pad and
    trim below keep whichever type and device it arrived with rather than
    forcing a host round trip the caller may not want yet.
    """
    axis = _sample_axis(audio)
    if axis is None:
        return audio

    expected = round(frame_count / frame_rate * sample_rate)
    difference = audio.shape[axis] - expected
    if difference == 0 or abs(difference) > AUDIO_FIT_TOLERANCE_SECONDS * sample_rate:
        return audio

    logger.debug(
        f"Fitting decoded audio to {frame_count} frames ({difference:+} samples)"
    )
    if difference > 0:
        trim = [slice(None)] * audio.ndim
        trim[axis] = slice(None, expected)
        return audio[tuple(trim)]
    if isinstance(audio, torch.Tensor):
        # torch.nn.functional.pad takes its pairs from the last axis backwards
        padding = [0, 0] * audio.ndim
        padding[2 * (audio.ndim - 1 - axis) + 1] = -difference
        return torch.nn.functional.pad(audio, padding)
    widths = [(0, 0)] * audio.ndim
    widths[axis] = (0, -difference)
    return numpy.pad(audio, widths)


def _sample_axis(audio):
    """The axis a waveform's samples run along, or None if it has no such axis.

    Not a fixed index: a generated track arrives in any of the layouts
    _as_stereo reads - (channels, samples), (samples, channels), or a bare
    (samples,) - and a mono one written (samples,) or (samples, 1) used to
    reach shape[1] here and either raise IndexError or fit the wrong axis
    into a silent no-op. Channels are few and samples are many, so the
    longer axis is the sample axis.
    """
    if audio.ndim == 1:
        return 0
    if audio.ndim != 2:
        return None
    return 0 if audio.shape[0] > audio.shape[1] else 1
