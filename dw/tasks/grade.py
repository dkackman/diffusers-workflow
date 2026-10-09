"""
CPU colour grading: exposure, contrast, tonal range, clarity, white balance,
saturation, fade and vignette.

Pure numpy/PIL - no model, no GPU. A video is graded per frame by the
command handler (image_ops.per_frame), so this module only ever sees a
single PIL Image.
"""

from functools import lru_cache

import numpy as np
from PIL import Image

from . import image_ops

# Fraction of the full [0, 1] channel range shifted at |temperature| == 1.0 or
# |tint| == 1.0. Temperature moves the red and blue channels apart (warmer is
# more red, less blue); tint moves the green channel against them (more
# magenta is less green). Both are linear in their argument and independent
# of each other - this is the "documented scale" v1 promises rather than a
# physical colour-temperature-in-Kelvin model, which is more expensive to
# compute and no more useful to a generation pipeline's output.
_WHITE_BALANCE_STRENGTH = 0.15

# The tonal controls' strength at |value| == 1.0, as a fraction of the channel
# range. Each is small enough that the tone curve it bends stays monotonic
# across its whole documented range, so no control can invert a gradient
_ENDPOINT_STRENGTH = 0.25  # whites, blacks
_RANGE_STRENGTH = 0.25  # highlights, shadows
_CLARITY_STRENGTH = 1.0
_FADE_FLOOR = 0.25  # the black floor at fade == 1.0
_VIGNETTE_STRENGTH = 0.5  # the corners' change at |vignette| == 1.0

# Clarity's blur radius as a fraction of the frame's shorter side: medium
# detail, wider than texture and narrower than the subject
_CLARITY_RADIUS = 0.02


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gaussian_blur(plane, sigma):
    """A Gaussian blur of a 2-D float plane, edges extended, truncated at 3
    sigma. PIL only blurs integer modes, which would quantize the plane."""
    import scipy.ndimage

    radius = max(1, int(round(3.0 * sigma)))
    return scipy.ndimage.gaussian_filter(plane, sigma, mode="nearest", radius=radius)


@lru_cache(maxsize=4)
def _vignette_falloff(width, height):
    """0 across the centre, rising smoothly to 1 at the corners. Cached by
    size, so a video builds it once rather than once per frame."""
    ys = (np.arange(height, dtype=np.float32) + 0.5) / height * 2.0 - 1.0
    xs = (np.arange(width, dtype=np.float32) + 0.5) / width * 2.0 - 1.0
    radius = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2) / np.sqrt(2.0)
    falloff = _smoothstep(0.35, 1.0, radius).astype(np.float32)
    falloff.setflags(write=False)
    return falloff


def grade_image(
    media,
    exposure=0.0,
    contrast=1.0,
    saturation=1.0,
    temperature=0.0,
    tint=0.0,
    highlights=0.0,
    shadows=0.0,
    whites=0.0,
    blacks=0.0,
    clarity=0.0,
    vignette=0.0,
    fade=0.0,
):
    """Adjust the exposure, tone, white balance and colour of a single frame.

    Every parameter is optional; omitting one leaves that adjustment at its
    identity value, so calling with no arguments returns the input pixels
    unchanged. Adjustments apply in this order: exposure, contrast,
    whites/blacks, highlights/shadows, clarity, temperature/tint, saturation,
    fade, vignette.

    Args:
        media: PIL Image to grade. A video is dispatched to this one frame at
            a time by the command handler (image_ops.per_frame), so this
            function itself only ever sees a single frame.
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
        highlights: -1.0 to 1.0. Lifts (positive) or pulls down (negative)
            the tones above mid grey, weighted by a smooth luma mask that is
            zero at and below mid grey and full at white, by up to 25% of
            the channel range. 0.0 (default) is identity.
        shadows: -1.0 to 1.0. The same for the tones below mid grey: a mask
            full at black and zero at and above mid grey. 0.0 (default) is
            identity.
        whites: -1.0 to 1.0. Moves the white point: brightens (positive) or
            darkens (negative) the top of the curve by up to 25% at white,
            fading to nothing at black. 0.0 (default) is identity.
        blacks: -1.0 to 1.0. Moves the black point: lifts (positive) or
            crushes (negative) the bottom of the curve by up to 25% at
            black, fading to nothing at white. 0.0 (default) is identity.
        clarity: -1.0 to 1.0. Adds (positive) or removes (negative) local
            contrast in medium detail: the difference from a Gaussian blur
            of the luma (radius 2% of the shorter side), masked to the
            midtones. 0.0 (default) is identity.
        vignette: -1.0 to 1.0. Darkens (positive) or lightens (negative)
            toward the corners with a smooth radial falloff, the centre
            untouched and the corners changed by up to 50%. 0.0 (default)
            is identity.
        fade: 0.0 to 1.0. Lifts the black floor to up to 25% of the range
            and flattens the shadows toward it, keeping white at white, for
            a matte look. 0.0 (default) is identity.

    Returns:
        PIL Image, same size and mode as the input (graded). An alpha
        channel, if the input has one, passes through untouched.
    """
    alpha = None
    if media.mode in ("RGBA", "LA"):
        alpha = media.getchannel("A")

    rgb = media.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0

    if exposure != 0.0:
        array = array * (2.0**exposure)

    if contrast != 1.0:
        array = (array - 0.5) * contrast + 0.5

    if whites != 0.0 or blacks != 0.0:
        # Per channel: each end of the curve moves, the other end held
        level = np.clip(array, 0.0, 1.0)
        array = (
            array
            + whites * _ENDPOINT_STRENGTH * level**2
            + blacks * _ENDPOINT_STRENGTH * (1.0 - level) ** 2
        )

    if highlights != 0.0 or shadows != 0.0:
        # One shift per pixel, from its luma, so hue is kept
        luma = np.clip(image_ops.luma(array), 0.0, 1.0)
        shift = highlights * _RANGE_STRENGTH * _smoothstep(0.5, 1.0, luma)
        shift += shadows * _RANGE_STRENGTH * (1.0 - _smoothstep(0.0, 0.5, luma))
        array = array + shift[..., None]

    if clarity != 0.0:
        luma = image_ops.luma(array)
        sigma = max(1.0, _CLARITY_RADIUS * min(luma.shape))
        detail = luma - _gaussian_blur(luma, sigma)
        midtones = np.clip(1.0 - (2.0 * np.clip(luma, 0.0, 1.0) - 1.0) ** 2, 0, 1)
        array = array + (clarity * _CLARITY_STRENGTH * detail * midtones)[..., None]

    if temperature != 0.0:
        shift = temperature * _WHITE_BALANCE_STRENGTH
        array[..., 0] += shift
        array[..., 2] -= shift

    if tint != 0.0:
        shift = tint * _WHITE_BALANCE_STRENGTH
        array[..., 1] -= shift

    if saturation != 1.0:
        luma = image_ops.luma(array)
        array = luma[..., None] + (array - luma[..., None]) * saturation

    if fade != 0.0:
        floor = fade * _FADE_FLOOR
        array = floor + np.clip(array, 0.0, 1.0) * (1.0 - floor)

    if vignette != 0.0:
        height, width = array.shape[:2]
        falloff = _vignette_falloff(width, height)
        array = array * (1.0 - vignette * _VIGNETTE_STRENGTH * falloff)[..., None]

    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    graded = Image.fromarray(array, mode="RGB")

    if alpha is not None:
        graded = graded.convert("RGBA")
        graded.putalpha(alpha)

    return graded
