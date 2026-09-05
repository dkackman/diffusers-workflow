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


def get_frame(video, frame_index=0):
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
    from ..security import (
        ALLOWED_VIDEO_EXTENSIONS,
        validate_file_extension,
        validate_path,
        validate_url,
    )

    if _URL_SCHEME.match(location):
        import io
        import requests

        # Any other scheme - ftp:, file:, data: - is refused here rather than
        # falling through to be read as a relative path that happens to
        # contain a colon
        validated_url = validate_url(location)
        logger.debug(f"Downloading video from {validated_url}")
        response = requests.get(validated_url, timeout=300)
        response.raise_for_status()
        handle = io.BytesIO(response.content)
    else:
        validated_path = validate_path(location, base_dir=base_dir, allow_create=False)
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
    return AudioVideo(frames, audio, sample_rate if audio is not None else None)


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
    """
    expected = round(frame_count / frame_rate * sample_rate)
    difference = audio.shape[1] - expected
    if difference == 0 or abs(difference) > AUDIO_FIT_TOLERANCE_SECONDS * sample_rate:
        return audio

    logger.debug(
        f"Fitting decoded audio to {frame_count} frames ({difference:+} samples)"
    )
    if difference > 0:
        return audio[:, :expected]
    return numpy.pad(audio, ((0, 0), (0, -difference)))
