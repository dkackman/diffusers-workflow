"""
CPU finishing passes beside grade: an unsharp-mask sharpen and film grain.

Pure numpy/PIL - no model, no GPU. A video goes through the command
handler's _per_frame (task.py), so sharpen_image only ever sees a single PIL
Image. film_grain runs the frames itself, because one generator has to be
consumed across all of them: every frame gets different grain, and the whole
run reproduces from the seed (#603).
"""

import numpy as np
from PIL import Image

# Luma weights (Rec. 709), for grain's midtone weighting
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Grain's standard deviation at amount == 1.0, as a fraction of the channel
# range: heavy, but still grain rather than noise that hides the picture
_GRAIN_STRENGTH = 0.15
# Grain's weight at pure black and pure white, rising to 1.0 at mid grey -
# film grain is most visible in the midtones, and grain at the ends would
# mostly clip
_GRAIN_END_WEIGHT = 0.25


def _split_alpha(media):
    """(RGB float32 array in 0..255, the alpha channel or None)."""
    alpha = media.getchannel("A") if media.mode in ("RGBA", "LA") else None
    return np.asarray(media.convert("RGB"), dtype=np.float32), alpha


def _join_alpha(array, alpha):
    """A PIL Image from a 0..255 float RGB array, with alpha put back."""
    array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    if alpha is not None:
        image = image.convert("RGBA")
        image.putalpha(alpha)
    return image


def sharpen_image(media, amount=1.0, radius=2.0, threshold=0):
    """Sharpen an image with an unsharp mask (Pillow's ImageFilter.UnsharpMask).

    The image is blurred with a Gaussian of `radius` pixels, and the
    difference between the image and its blur - the detail - is added back,
    scaled by `amount`. Detail smaller than `threshold` is left alone, so
    flat areas and fine noise are not sharpened.

    Args:
        media: PIL Image to sharpen
        amount: How much of the detail to add back; 0 is identity, 1 doubles
            the edge contrast at the blur's scale. 0 or above, applied in
            whole percent (Pillow's `percent`), so it is rounded to 0.01
        radius: The blur's radius in pixels: the scale of the detail that is
            sharpened. Above zero
        threshold: The smallest difference, in 0..255 channel levels, between
            a pixel and its blur that is sharpened. 0 sharpens everything. A
            whole number of levels; a fractional value is rounded

    Returns:
        PIL Image, the same size, RGB (RGBA when the input had alpha, which
        passes through untouched)
    """
    from PIL import ImageFilter

    alpha = media.getchannel("A") if media.mode in ("RGBA", "LA") else None
    rgb = media.convert("RGB")
    if amount != 0:
        rgb = rgb.filter(
            ImageFilter.UnsharpMask(
                radius=float(radius),
                percent=round(float(amount) * 100),
                threshold=round(float(threshold)),
            )
        )
    if alpha is not None:
        rgb = rgb.convert("RGBA")
        rgb.putalpha(alpha)
    return rgb


def _noise_field(rng, height, width, size, channels):
    """Unit-variance Gaussian noise of shape (height, width, channels),
    generated at 1/size resolution and upsampled bilinearly, so one grain
    is about `size` pixels across."""
    if size <= 1:
        return rng.standard_normal((height, width, channels), dtype=np.float32)
    small_h = max(1, int(np.ceil(height / size)))
    small_w = max(1, int(np.ceil(width / size)))
    small = rng.standard_normal((small_h, small_w, channels), dtype=np.float32)
    planes = [
        np.asarray(
            Image.fromarray(small[:, :, c], mode="F").resize(
                (width, height), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        for c in range(channels)
    ]
    field = np.stack(planes, axis=-1)
    # Interpolation averages neighbours, which shrinks the spread: put it
    # back so `amount` means the same strength at every size
    std = float(field.std())
    return field / std if std > 0 else field


def grain_frame(media, rng, amount=0.1, size=1.0, chroma=0.0):
    """Add one frame's film grain, drawn from `rng`.

    Grain is Gaussian noise, weighted strongest in the midtones. At
    chroma 0 the same noise is added to all three channels, so it changes
    brightness only and never hue; chroma 1 draws each channel's noise
    independently. Values between mix the two at the same overall strength.
    """
    rgb, alpha = _split_alpha(media)
    if amount == 0:
        return _join_alpha(rgb, alpha)
    height, width = rgb.shape[:2]
    chroma = float(chroma)
    noise = _noise_field(rng, height, width, float(size), 1)
    if chroma > 0:
        independent = _noise_field(rng, height, width, float(size), 3)
        # Two independent unit-variance fields mixed linearly lose spread
        # toward the middle; dividing by the mix's own deviation keeps the
        # strength the same at every chroma
        noise = ((1.0 - chroma) * noise + chroma * independent) / np.sqrt(
            (1.0 - chroma) ** 2 + chroma**2
        )
    luma = np.tensordot(rgb / 255.0, _LUMA_WEIGHTS, axes=([-1], [0]))
    weight = _GRAIN_END_WEIGHT + (1.0 - _GRAIN_END_WEIGHT) * 4.0 * luma * (1.0 - luma)
    grain = noise * (float(amount) * _GRAIN_STRENGTH * 255.0) * weight[:, :, None]
    return _join_alpha(rgb + grain, alpha)


def film_grain(media, amount=0.1, size=1.0, chroma=0.0, seed=None):
    """Add film grain to an image or a video.

    One random generator is made from `seed` per call and consumed frame by
    frame, so every frame of a video gets different grain and the same seed
    reproduces the whole result exactly.

    Args:
        media: Image or video (a PIL Image, an AudioVideo or a frame list)
        amount: Grain strength, 0.0 (identity) to 1.0
        size: Grain size in pixels, 1 or above; the noise is generated at
            1/size resolution and upsampled
        chroma: 0.0 adds the same grain to every channel (brightness only),
            1.0 independent grain per channel (colour noise)
        seed: The generator's seed. A workflow run always supplies one (its
            own, recorded in the manifest), so a rerun reproduces the grain

    Returns:
        A PIL Image for an image, an AudioVideo with the same frame count,
        frame rate and audio for a video
    """
    from .task import _per_frame

    rng = np.random.default_rng(None if seed is None else int(seed))
    return _per_frame(
        media,
        lambda frame: grain_frame(frame, rng, amount=amount, size=size, chroma=chroma),
    )
