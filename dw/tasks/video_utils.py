"""Frame access for videos in any of the shapes results carry them.

A video artifact can be a list of PIL images, a numpy array of frames
(frames, height, width, channels), a torch tensor (frames first, channels
first or last), or an AudioVideo pairing frames with their generated
soundtrack. extract_frame gives tasks and the segment-chaining loop one way
to pull a single frame out of any of them, always as a PIL image.
"""

import logging
import re

import numpy
import torch
from PIL import Image

from ..media import decode_audio_video
from ..media_frames import (
    compose_grid,
    default_columns,
    evenly_spaced_indices,
    frames_at,
    grid_tile,
)
from ..media_types import AudioVideo, FittedVideo, fit_codec_padding
from ..task_domains import frame_size_error

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
    and all (#387), and so window_video reads its source a range of frames
    at a time, with the soundtrack decoded on its own (#601, #695). Not a
    public shape: only _realize_lazy_frame_arguments builds one, for a
    task's 'video' argument, and the tasks that take a file-based 'video' -
    get_frame and its kin, the probes, trim_video, find_loop_bed and
    window_video - read it."""

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
    or, for a PIL list, produces a film that changes size mid-cut. Naming
    every mismatched size, against the first known one by its real index,
    points at each shot that was rendered differently rather than at the
    join - the same sentence validate gives for sizes it can already probe
    (dw/video_size_errors.py).
    """
    sizes = {}
    for index, clip in enumerate(clips):
        if isinstance(clip, numpy.ndarray):
            sizes[index] = (int(clip.shape[2]), int(clip.shape[1]))
        elif len(clip):
            sizes[index] = tuple(clip[0].size)
    error = frame_size_error(task_name, sizes)
    if error is not None:
        raise ValueError(error)


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
    # The source's own rate rides on the array: a bare array carries none, and
    # a later pair_audio or the writer would fall back to 8 fps (#673). The
    # source's shots do not survive - an array has nowhere to hold them
    fps = getattr(video, "fps", None)

    if isinstance(frames, numpy.ndarray) and frames.ndim == 4 and frames.shape[-1] == 3:
        if frames.dtype == numpy.uint8:
            return FittedVideo(frames, fps=fps, dtype=numpy.uint8) if fps else frames
        # Float frames are [0, 1] - diffusers' np output convention
        scaled = (numpy.clip(frames, 0.0, 1.0) * 255).round().astype(numpy.uint8)
        return FittedVideo(scaled, fps=fps, dtype=numpy.uint8) if fps else scaled

    stacked = numpy.stack(
        [numpy.asarray(_to_pil(frame).convert("RGB")) for frame in frames]
    )
    return FittedVideo(stacked, fps=fps, dtype=numpy.uint8) if fps else stacked


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
        A (num_frames, height, width, channels) float32 array scaled to
        [0, 1] - diffusers' own np frame convention, and what
        `LTX2ReferenceCondition.frames` and its kin need: a raw ndarray
        reaches `VaeImageProcessor.preprocess` untouched, with no /255
        rescaling applied along the way, so a uint8 [0, 255] array read as
        already-scaled data is 255x too bright (#444). Not for a
        keyframe (`LTX2VideoCondition`): its ndarray path expects uint8
        [0, 255] and refuses a float frame when `crf` is set -
        `frames_as_array` is the shape for that
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
    looped = numpy.concatenate([frames] * laps, axis=0)[:num_frames]
    return (looped.astype(numpy.float32) / 255.0).clip(0.0, 1.0)


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

    indices = evenly_spaced_indices(total, count)
    tiles = [
        grid_tile(extract_frame(video, index), index, fps, tile_width, label)
        for index in indices
    ]

    if columns is None:
        columns = default_columns(len(tiles))
    return compose_grid(tiles, columns)


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

    # A bare still - e.g. a {"media_type": "image", ...} reference fetch_video
    # now loads as a plain PIL image (#443) - is a one-frame video, the same
    # accommodation loop_frames already made for itself with _is_frame
    if isinstance(video, Image.Image):
        return [video]

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
    """The frames of a video file, carrying the rate the file plays at and
    the shot boundaries its run recorded, if any.

    `load_video` answers a plain list of images, which is what every
    pipeline argument and every task wants - and which says nothing about
    how fast those frames are meant to run. A step handed a 24 fps file
    then wrote it back at `result.fps`'s default of 8, three times long,
    with its soundtrack finishing a third of the way in and nothing said
    about it (#104, the file-loading half of #84). A list subclass keeps
    every consumer working unchanged while `getattr(video, "fps", None)` -
    the question AudioVideo, concat_videos and interpolate_frames already
    ask - gets a real answer.

    `shots` is the same idea for the boundaries `dw.runs.shots_beside`
    finds beside the file: a video loaded from an `asset:`/`output:` path
    carried no way to answer `getattr(video, "shots", None)`, so
    `pair_audio` had nothing to remeasure even though the file's own
    manifest (or its kept-asset sidecar) already held them (#398).
    """

    def __init__(self, frames, fps=None, shots=None):
        super().__init__(frames)
        self.fps = fps
        self.shots = shots


def load_audio_video(location, base_dir=None):
    """Load a video file - frames and the audio muxed with them - as an AudioVideo.

    `load_video` reads frames only, so a file written by an earlier run comes
    back silent. Reading both streams here is what lets a step join videos that
    are already on disk - the shots of an earlier run picked back up by name -
    without dropping the audio those runs generated alongside them.

    Args:
        location: Local path, or an http(s) URL, of a video file, or a
            {"location": ...} dict wrapping either - the same idiom
            `get_last_frame(video=...)` and a pipeline's `image` argument
            already accept (#510)
        base_dir: Directory a relative path is resolved against

    Returns:
        An AudioVideo holding the frames as PIL images and, when the file
        carries an audio stream, its waveform as a (channels, samples) float32
        array with the stream's sample rate. A local file also carries the
        shots its own run manifest recorded for it (`shots_beside`), so a
        join of a file that is itself an earlier join's output can see the
        seams inside it (#399); a URL carries none.
    """
    from ..outbound import safe_get

    if isinstance(location, dict):
        location = location["location"]

    if _URL_SCHEME.match(location):
        import io

        # Any other scheme - ftp:, file:, data: - is refused here rather than
        # falling through to be read as a relative path that happens to
        # contain a colon. An http(s) one still has to name a host outside
        # this deployment (dw/locations.py), and so does every redirect
        logger.debug(f"Downloading video from {location}")
        response = safe_get(location, "a video argument", timeout=300)
        handle = io.BytesIO(response.content)
        return _decode_audio_video(handle)

    validated_path = local_video_path(location, base_dir)
    logger.debug(f"Reading video from {validated_path}")
    video = _decode_audio_video(validated_path)
    from ..runs import shots_beside

    video.shots = shots_beside(validated_path)
    return video


def local_video_path(location, base_dir=None):
    """The validated local path a video location names, or None for a URL -
    the check `load_audio_video` makes before it reads a file, for a caller
    that reads the file some other way (join_windows' source, #695)."""
    from ..security import ALLOWED_VIDEO_EXTENSIONS, validate_file_extension
    from ..locations import validate_media_path

    if isinstance(location, dict):
        location = location["location"]
    if _URL_SCHEME.match(location):
        return None
    validated_path = validate_media_path(location, base_dir, "a video argument")
    validate_file_extension(validated_path, ALLOWED_VIDEO_EXTENSIONS)
    return validated_path


def is_video_location(value):
    """Whether value is something load_audio_video can load: a path/URL
    string, or a {"location": ...} dict wrapping one - the form validation
    lets through unresolved inside a list argument (#510), since arguments.py's
    key conventions only fire for a scalar 'video' argument, never a list entry."""
    return isinstance(value, str) or (isinstance(value, dict) and "location" in value)


def _decode_audio_video(handle):
    """Decode a path or file object's video and audio streams in one pass."""
    frames, audio, sample_rate, frame_rate = decode_audio_video(handle)
    if audio is not None and frame_rate:
        audio = fit_codec_padding(audio, len(frames), frame_rate, sample_rate)
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
