"""
CPU colour grading: exposure, contrast, saturation, temperature/tint.

Pure numpy/PIL - no model, no GPU. A video is graded per frame by the
command handler (task.py's _per_frame), so this module only ever sees a
single PIL Image.
"""

import numpy as np
from PIL import Image

# Luma weights (Rec. 709), used as the saturation pivot
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Fraction of the full [0, 1] channel range shifted at |temperature| == 1.0 or
# |tint| == 1.0. Temperature moves the red and blue channels apart (warmer is
# more red, less blue); tint moves the green channel against them (more
# magenta is less green). Both are linear in their argument and independent
# of each other - this is the "documented scale" v1 promises rather than a
# physical colour-temperature-in-Kelvin model, which is more expensive to
# compute and no more useful to a generation pipeline's output.
_WHITE_BALANCE_STRENGTH = 0.15


def grade_image(
    image,
    exposure=0.0,
    contrast=1.0,
    saturation=1.0,
    temperature=0.0,
    tint=0.0,
):
    """Adjust exposure, contrast, saturation and white balance of an image.

    Every parameter is optional; omitting one leaves that adjustment at its
    identity value, so calling with no arguments returns the input pixels
    unchanged (within rounding). Adjustments apply in this order: exposure,
    then contrast, then temperature/tint, then saturation.

    Args:
        image: PIL Image to grade.
        exposure: Stops to brighten (positive) or darken (negative) by,
            applied as a multiply of 2**exposure. 0.0 (default) is identity.
        contrast: Multiplier applied around the mid grey point (0.5).
            1.0 (default) is identity; above 1 increases contrast, below 1
            (down to 0) flattens it.
        saturation: Multiplier applied around each pixel's own luma
            (Rec. 709 weights). 1.0 (default) is identity; 0.0 is greyscale.
        temperature: Warm/cool white-balance shift from -1.0 (coolest, shifts
            toward blue) to 1.0 (warmest, shifts toward red), linear, moving
            the red and blue channels apart by up to 15% of the channel
            range at |1.0|. 0.0 (default) is identity.
        tint: Green/magenta white-balance shift from -1.0 (green) to 1.0
            (magenta), linear, moving the green channel by up to 15% of the
            channel range at |1.0| in the opposite direction to the shift's
            sign. 0.0 (default) is identity.

    Returns:
        PIL Image, same size and mode as the input (graded). An alpha
        channel, if the input has one, passes through untouched.
    """
    alpha = None
    if image.mode in ("RGBA", "LA"):
        alpha = image.getchannel("A")

    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0

    if exposure != 0.0:
        array = array * (2.0**exposure)

    if contrast != 1.0:
        array = (array - 0.5) * contrast + 0.5

    if temperature != 0.0:
        shift = temperature * _WHITE_BALANCE_STRENGTH
        array[..., 0] += shift
        array[..., 2] -= shift

    if tint != 0.0:
        shift = tint * _WHITE_BALANCE_STRENGTH
        array[..., 1] -= shift

    if saturation != 1.0:
        luma = np.tensordot(array, _LUMA_WEIGHTS, axes=([-1], [0]))
        array = luma[..., None] + (array - luma[..., None]) * saturation

    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    graded = Image.fromarray(array, mode="RGB")

    if alpha is not None:
        graded = graded.convert("RGBA")
        graded.putalpha(alpha)

    return graded
