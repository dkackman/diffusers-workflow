"""
Shared pure numpy/PIL helpers that grade, finish and lut use: Rec. 709 luma,
splitting an image's alpha channel off and putting it back, and running an
image command over a video frame by frame.
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
