"""
Shared pure numpy/PIL helpers that grade, finish, lut and the assessment
probes use: Rec. 709 luma, splitting an image's alpha channel off and putting
it back, running an image command over a video frame by frame, and reducing
frames to the grey thumbnails and look stats `dw/tasks/assess.py` measures.
"""

import numpy as np
from PIL import Image

from ..media_types import AudioVideo

# Luma weights (Rec. 709), float64 - `luma` casts them to its caller's dtype
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def luma(rgb):
    """Rec. 709 luma of an array whose last axis is RGB.

    The weights are cast to the array's own dtype, so a float32 caller gets
    float32 arithmetic and result - exactly what a float32 weights copy gave
    it - rather than drifting through float64.
    """
    return np.tensordot(
        rgb, LUMA_WEIGHTS.astype(rgb.dtype, copy=False), axes=([-1], [0])
    )


def split_alpha(media):
    """(RGB float32 array in 0..255, the alpha channel or None)."""
    alpha = media.getchannel("A") if media.mode in ("RGBA", "LA") else None
    return np.asarray(media.convert("RGB"), dtype=np.float32), alpha


def join_alpha(array, alpha):
    """A PIL Image from a 0..255 float RGB array, with alpha put back."""
    array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    if alpha is not None:
        image = image.convert("RGBA")
        image.putalpha(alpha)
    return image


def per_frame(image, process):
    """Run an image command over a video, frame by frame.

    A video bound to an image argument - an AudioVideo from a generation,
    concat or dissolve step, or a frame array from video_frames - is processed
    one frame at a time and comes back as one video artifact, its soundtrack
    carried through untouched. A single image is processed as itself.
    """
    from ..shots import carried_shots
    from .video_utils import frames_as_pil_list, is_video

    if not is_video(image):
        return process(image)
    frames = [process(frame) for frame in frames_as_pil_list(image)]
    audio = getattr(image, "audio", None)
    sample_rate = getattr(image, "sample_rate", None)
    # One frame out per frame in, so the shot boundaries carry through too
    return AudioVideo(
        frames,
        audio,
        sample_rate,
        fps=getattr(image, "fps", None),
        shots=carried_shots(image),
    )


def look_stats(rgb):
    """A frame's look as four float32 scalars on the 0-255 scale: lightness
    (mean Rec. 709 luma), contrast (luma standard deviation), saturation (mean
    per-pixel max(RGB) - min(RGB)) and detail (mean |luma - its 3x3 box
    blur|, edges extended) - assess.LOOK_MEASURES' order. The one reduction
    both the file reader and the in-memory path use."""
    import scipy.ndimage

    rgb = np.asarray(rgb, dtype=np.float32)
    plane = luma(rgb)
    blurred = scipy.ndimage.uniform_filter(plane, size=3, mode="nearest")
    return np.array(
        [
            plane.mean(),
            plane.std(),
            (rgb.max(axis=-1) - rgb.min(axis=-1)).mean(),
            np.abs(plane - blurred).mean(),
        ],
        dtype=np.float32,
    )


def as_image(frame):
    """One in-memory frame (PIL image, or an HxWxC array or tensor, uint8 or
    float in [0, 1]) as a PIL image."""
    if not isinstance(frame, Image.Image):
        array = frame
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        array = np.asarray(array)
        if array.dtype.kind == "f":
            array = np.clip(array * 255.0, 0, 255)
        array = array.astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        frame = Image.fromarray(array)
    return frame


def thumbs_and_looks(frames, thumb_size, look_size):
    """(grey uint8 thumbnails at `thumb_size`, `look_stats` of a `look_size`
    colour copy) of in-memory frames, one frame at a time: a list, an
    (N, H, W, C) array, or a SegmentedFrames replaying (N, H, W, 3) chunks.
    Both None for no frames. Sizes are (width, height)."""
    from ..pipeline_processors.chain_spill import SegmentedFrames

    if isinstance(frames, SegmentedFrames):
        frames = (frame for chunk in frames for frame in chunk)
    thumbs, looks = [], []
    for frame in frames:
        image = as_image(frame)
        thumbs.append(
            np.asarray(
                image.convert("L").resize(thumb_size, Image.BILINEAR), dtype=np.uint8
            )
        )
        rgb = image.convert("RGB").resize(look_size, Image.BILINEAR)
        looks.append(look_stats(np.asarray(rgb)))
    if not thumbs:
        return None, None
    return np.stack(thumbs), np.stack(looks)
