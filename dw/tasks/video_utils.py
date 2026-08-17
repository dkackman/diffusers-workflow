"""Frame access for videos in any of the shapes results carry them.

A video artifact can be a list of PIL images, a numpy array of frames
(frames, height, width, channels), a torch tensor (frames first, channels
first or last), or an AudioVideo pairing frames with their generated
soundtrack. extract_frame gives tasks and the segment-chaining loop one way
to pull a single frame out of any of them, always as a PIL image.
"""

import numpy
import torch
from PIL import Image

from ..result import AudioVideo


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
